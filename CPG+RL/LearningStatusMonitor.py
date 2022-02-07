import os,sys, time
import socket, pickle
import numpy as np
import traceback
import threading
from PyQt5 import QtWidgets, QtCore, QtGui
import robot_foot_model as RFModel
import FootPlot as FootPlot
import UI.underTrainingUI as utUI
import UI.needsaveUI as nsUI
import UI.terminatelearningUI as tlUI
import UI.dynamicdataUI as dyUI
import UI.changerewardUI as rwUI
import TSNEVisualization as TSNE
import LearningConmunication as LCM
import LearningUtility as LUtil
import DXL_motor_control as DMC


# Code for Status Monitor App

ANIMATION_ON, PYQTGRAPH_ON = True, True
DYNAMIXEL_MOTOR_ON = True

class App(object):

    def __init__(self):
        global ANIMATION_ON, PYQTGRAPH_ON, DYNAMIXEL_MOTOR_ON
        # initial GUI
        if socket.gethostname() == 'ERIC-PERSONAL':
            os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
            app = QtWidgets.QApplication(sys.argv)
            app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        else:
            app = QtWidgets.QApplication(sys.argv)

        self.GUI = MonitorUI()
        self.needSaveUI = NeedSaveUI()
        self.ternimateUI = TernimateUI()
        self.dynamicLogUI = DynamicLogUI()
        self.changeReward = RewardUI()

        #GUI attribute
        self._run_flag = False
        self._manual_flag = False
        self._need_log = False
        self._thread_runing = False
        self.ploting_threads = list()
        self.data = LCM.TrainigData()
        self.m_HOST = "localhost"
        self.c_HOST = "localhost"
        self.PORT = 58000
        self.RL_FOOT = RFModel.Foot()
        self.REAL_FOOT = RFModel.Foot14_14_7_Real()
        self.history_loss = list()
        self.history_torque_difference = list()
        self.learninigServer = "localhost"
        self.recv_count = 0
        self.status_info = LCM.StatusInfo()
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.setInterval(10)
        self.sufficient_style = """
        QProgressBar::chunk {
        background-color: #4FED4F;
        }
        """
        self.GUI.tabWidget.setCurrentIndex(0)
        #GUI connection
        self.GUI.StartStopBTN.clicked.connect(self.StartStopClicked)
        self.GUI.terminateBtn.clicked.connect(self.terminateLearning)
        self.GUI.QuitBTN.clicked.connect(self.ExitApp)
        self.plotTimer.timeout.connect(self.processSocket)
        self.GUI.Server2.toggled.connect(lambda: self.SwitchServer('Server2'))
        self.GUI.Eric_Personal.toggled.connect(lambda: self.SwitchServer('ERIC_PERSONAL'))
        self.GUI.Nano.toggled.connect(lambda: self.SwitchServer('Nano'))
        self.GUI.ubuntu1060.toggled.connect(lambda: self.SwitchServer('Ubuntu1060'))
        self.GUI.twcc1.toggled.connect(lambda: self.SwitchServer("twcc1"))
        self.GUI.twcc2.toggled.connect(lambda: self.SwitchServer("twcc2"))
        self.GUI.twcc3.toggled.connect(lambda: self.SwitchServer("twcc3"))
        self.GUI.LocalhostRB.toggled.connect(lambda: self. SwitchServer('localhost'))

        # self.GUI.Server2.toggled.connect(self.serverChanged)
        # self.GUI.Eric_Personal.toggled.connect(self.serverChanged)
        # self.GUI.Nano.toggled.connect(self.serverChanged)
        # self.GUI.ubuntu1060.toggled.connect(self.serverChanged)
        # self.GUI.LocalhostRB.toggled.connect(self.serverChanged)
        # self.GUI.Manual.toggled.connect(self.serverChanged)
        
        self.needSaveUI.SaveBtn.clicked.connect(lambda: self.needSave("Save"))
        self.needSaveUI.CencelBtn.clicked.connect(lambda: self.needSave("Cencel"))
        self.needSaveUI.DiscardBtn.clicked.connect(lambda: self.needSave("Discard"))
        self.GUI.manualStart.toggled.connect(lambda: self.manualOnOff(True))
        self.GUI.manualStop.toggled.connect(lambda: self.manualOnOff(False))
        self.GUI.xCmdSli.valueChanged.connect(self.manualTargetSend)
        self.GUI.yCmdSli.valueChanged.connect(self.manualTargetSend)
        self.GUI.DynamicLogBtn.clicked.connect(self.OnOffDynamicLog)
        self.GUI.change_reward.clicked.connect(self.onOffChangeRW)
        self.changeReward.apply_btn.clicked.connect(self.sentRWFactor)
        self.GUI.real_time_rb.clicked.connect(self.updateRealTime)
        self.GUI.off_learning_rb.clicked.connect(self.updateInLearning)
        self.GUI.port_input.textChanged.connect(self.SwitchServer)
        self.GUI.tabWidget.currentChanged.connect(self.mainWidgetChange)



        self.ActionInfo = FootPlot.ActionInfo(
            GUI              = self.GUI,
            status_info      = self.status_info,
            data             = self.data, 
            learningServer   = self.learninigServer, 
            PORT             = self.PORT,
            _run_flag        = self._run_flag,
            PYQTGRAPH_ON     = PYQTGRAPH_ON,
            )

        # self.dynamixel = DMC.DXL_Conmunication('COM8')



        #Matplot Lib Setting

        self.PYQTGRAPH_ON = PYQTGRAPH_ON
        self.DYNAMICLOG_VERSION = 1
        self._dynamic_run = False
        self._dynamic_plot_exec_times = 0
        self.foot_plot_widget = FootPlot.FootWidget(self.GUI.MPLib, PYQTGRAPH_ON=self.PYQTGRAPH_ON)
        self._dynamic_ax, self.plot_widgets = self.foot_plot_widget.returnPlotWidget()
        self.foot_diagram = FootPlot.Foot(
            self.GUI.MPLib,
            dynamic_ax=self._dynamic_ax,
            plot_widgets=self.plot_widgets,
            need_trace=True,
            need_ground=True,
            need_obstacle=True,
            need_target=True,
            PYQTGRAPH_ON=self.PYQTGRAPH_ON,
            color_id=0
        )
        self.foot_diagram_real = FootPlot.Foot(
            self.GUI.MPLib,
            dynamic_ax=self._dynamic_ax,
            plot_widgets=self.plot_widgets,
            need_trace=False,
            need_ground=False,
            need_obstacle=False,
            PYQTGRAPH_ON=self.PYQTGRAPH_ON,
            color_id=1
        )


        self.foot_plotting_timer = QtCore.QTimer()
        self.foot_plot_widget_timer = QtCore.QTimer()
        self.foot_plotting_timer.timeout.connect(self.foot_diagram.plotText)
        self.foot_plot_widget_timer.timeout.connect(self.foot_plot_widget.updatePlotWidget)

        if not self.PYQTGRAPH_ON:
            self.foot_diagram.UpdatePlot(self.RL_FOOT)
        self.tsne_diagram = FootPlot.TSNE(self.GUI.tsne_layout)
        self.loss_diagram = FootPlot.Loss(self.GUI.loss_layout, PYQTGRAPH_ON=self.PYQTGRAPH_ON)
        #GUI Start
        self.GUI.monitor_window.show()
        sys.exit(app.exec_())

        # server.shutdown()


    def changeTitle(self):
        # self.GUI.lineEdit.setText("Hi")
        pass

    def mainWidgetChange(self):
        if self.GUI.tabWidget.currentIndex() == 2:
            pass


    def processSocket(self):
        '''Get data from learning system, and show/plot result on UI'''
        try:
            data_rv, _ = self.status_info.NET_sock.recvfrom(32768)
            self.recv_count += 1
        except socket.error as _:
            # traceback.print_exc()
            return

        if data_rv is not None:
            self.data = pickle.loads(data_rv)
            if self.recv_count == 1:
                if self.RL_FOOT.__class__.__name__ != self.data.foot_name:
                    self.RL_FOOT = LUtil.selectFoot(self.data)
                    self.REAL_FOOT = LUtil.selectFoot(self.data)
            if self.RL_FOOT.log_len != self.data.gait_len * self.data.each_target_times:
                self.RL_FOOT.resetLoglength(self.data.gait_len * self.data.each_target_times)


            self.RL_FOOT.excuByInfo(self.data.motor_info, self.data.loading)
            tsne_data = self.processTSNE(self.data)
            loss_data = self.processLoss(self.data.loss)
            self.processTorque(np.array([
                self.data.motor_info[0].torque,
                self.data.motor_info[1].torque,
                self.data.motor_info[2].torque
            ]))

            self.updateUI(self.data.learning_duration)

            if self.threadIsNotRuning():
                
                #Multithread ploting, avoid plotting delay receiv
                self.ploting_threads = list()


                ### ActionInfo Tab data Update, put here so that data still update even without plot.
                if hasattr(self.data, "learning_info"):
                    MOTOR_ACTION_SPACE = np.arange(self.data.learning_info[1])
                    self.each_motor_action = LUtil.processAction(self.data.tsne_info[0][1], MOTOR_ACTION_SPACE)
                else:
                    self.each_motor_action = [0, 0, 0]
                    self.data.learning_info[1] = 21
                self.ActionInfo.updateStatus(
                    self.GUI, 
                    self.status_info, 
                    self.data, 
                    learningServer=self.learninigServer, 
                    PORT=self.PORT,
                    _run_flag=self._run_flag,
                    plotTimer=self.plotTimer,
                    each_motor_action = self.each_motor_action,
                    action_resolution = self.data.learning_info[1],
                    data_receive_count=self.recv_count,
                )

                self.foot_plot_widget.updateData(self.recv_count, self.data)

                self.ploting_threads.append(FootPlot.FootDiagramThread(
                    self.foot_diagram, self.RL_FOOT, self.data,
                    v_cmd=self.each_motor_action,
                    PYQTGRAPH_ON=self.PYQTGRAPH_ON,
                ))
                if hasattr(self.data, "real_motor_info"):
                    if self.data.real_motor_info[0] is not None:
                        self.REAL_FOOT.excuByInfo(self.data.real_motor_info)
                        self.ploting_threads.append(FootPlot.FootDiagramThread(
                            self.foot_diagram_real, self.REAL_FOOT, self.data,
                            v_cmd=self.each_motor_action,
                            PYQTGRAPH_ON=self.PYQTGRAPH_ON,
                        ))

                self._dynamic_plot_exec_times += 1
                if hasattr(self, "log_diagram") and self.data.target[1][2]: #Excute when new cycle
                    self._dynamic_plot_exec_times = 0
                if self._dynamic_run:
                    self.ploting_threads.append(FootPlot.DynamicLogThread(
                        self.log_diagram, self.RL_FOOT, self.data.gait_len,
                        dynamic_plot_exec_times=self._dynamic_plot_exec_times,
                        PYQTGRAPH_ON=self.PYQTGRAPH_ON
                    ))



                if self.GUI.tabWidget.currentIndex() == 1:
                    self.ploting_threads.append(FootPlot.TSNEPlotingThread(
                        self.tsne_diagram, tsne_data
                    ))
                    self.ploting_threads.append(FootPlot.LossPlotingThread(
                        self.loss_diagram, loss_data
                    ))
                # Enable ActionInfo plot thread and plot it every 5 cycles
                elif self.GUI.tabWidget.currentIndex() == 2 and self.recv_count%5 == 0:
                    self.ploting_threads.append(FootPlot.ActionInfoThread(
                        self.ActionInfo
                    ))


                self.ploting_threads.append(threading.Thread(target=lambda:time.sleep(0.01), daemon=True))

                for thread in self.ploting_threads:
                    thread.start()
                
            else:
                print(f"Plotting passed {self.recv_count}")

    def threadIsNotRuning(self):
        if self.ploting_threads == []:
            return True
        else:
            running = False
            for thread in self.ploting_threads:
                # print(thread.is_alive())
                running = running or thread.is_alive()
                if running:
                    return False


            return not(running)


    def updateUI(self, info):

        target = self.data.target
        X_cmd_changed = self.GUI.X_cmd.text() is not str(round(target[1][0],2))
        Y_cmd_changed = self.GUI.X_cmd.text() is not str(round(target[1][1],2))
        if X_cmd_changed or Y_cmd_changed:
            if self._manual_flag:
                self.GUI.X_cmd.setText(str(round(target[1][0],2)))
                self.GUI.Y_cmd.setText(str(round(target[1][1],2)))
            else:
                self.GUI.X_cmd.setText(str(round(target[1][0],2)))
                self.GUI.Y_cmd.setText(str(round(target[1][1],2)))
                
                self.GUI.xCmdSli.setValue(int(target[1][0]*10))
                self.GUI.yCmdSli.setValue(int(target[1][1]*10))
                for slider in [self.GUI.xCmdSli, self.GUI.yCmdSli]:
                    slider.repaint()

        self.GUI.PosX.display(round(self.RL_FOOT.EndPoint[0],2))
        self.GUI.PosY.display(round(self.RL_FOOT.EndPoint[1],2))

        if self.GUI.tabWidget.currentIndex() == 1:
            self.GUI.learning_duration.setText(info)
            self.GUI.learning_duration.repaint()

            if hasattr(self,'tsne_generator'):
                if self.tsne_generator.sufficient_flag:
                    self.GUI.tsne_memory_progress.setStyleSheet(self.sufficient_style)
                    self.GUI.tsne_memory_progress.setFormat('sufficient')
                    self.GUI.tsne_memory_progress.setValue(len(self.tsne_generator.memory))
                else:
                    self.GUI.tsne_memory_progress.setValue(len(self.tsne_generator.memory))
                    
        if self.changeReward.reward_window.isVisible():
            self.changeReward.total_reward.setText(
                "{0:.6f}".format(self.data.tsne_info[0][3])
            )
            self.changeReward.explo_rateLb.setText(
                "{0:.2f} %".format(self.data.explorer_rate*100)
            )
            self.changeReward.updateGUI_Value(self.data.each_reward)

    # def updataTSNE(self, data):
    #     if self.GUI.tabWidget.currentIndex == 1:
    #         self.tsne_diagram.UpdatePlot(data)

    def StartStopClicked(self):
        # self.m_HOST = self.GUI.s_url.currentText()
        self._run_flag = not(self._run_flag)
        if self._run_flag == True:
            self.SwitchServer()
            self.status_info.status_cmd.check_learning = True
            self.status_info.StatusSend(self.learninigServer, self.PORT)
            self.plotTimer.start()
            self.foot_diagram.clearTrace()
            self.GUI.StartStopBTN.setText("Stop")
            self.GUI.StartStopBTN.repaint()
            if self.PYQTGRAPH_ON:
                self.foot_plotting_timer.start(100)
                self.foot_plot_widget_timer.start(100)

        elif self._run_flag == False:
            self.status_info.status_cmd.check_learning = False
            self.recv_count = 0
            self.status_info.StatusSend(self.learninigServer, self.PORT)
            self.GUI.manualStop.toggle()
            self.plotTimer.stop()
            self.status_info.resetSocket()
            if hasattr(self, 'tsne_generator'):
                delattr(self, 'tsne_generator')
            self.GUI.StartStopBTN.setText("Start")
            self.GUI.StartStopBTN.repaint()
            if self.PYQTGRAPH_ON:
                self.foot_plotting_timer.stop()
                self.foot_plot_widget_timer.stop()
    
    def terminateLearning(self):
        self.needSaveUI.NeedSaveWindow.show()

    def needSave(self, select_type):
        reply = 0
        if select_type == "Save":
            if self.needSaveUI.lineEdit.text() != "":
                file_name = self.needSaveUI.lineEdit.text()
                self.status_info.status_cmd.NeedSave = [True, file_name]
                reply = self.ternimateUI.TernimateDialog.exec_()
                self.needSaveUI.NeedSaveWindow.close()
            else:
                self.needSaveUI.statusbar.showMessage("Please enter valid filename",2000)
        if select_type == "Discard":
            reply = self.ternimateUI.TernimateDialog.exec_()
            self.needSaveUI.NeedSaveWindow.close()
        if select_type == "Cencel":
            self.needSaveUI.NeedSaveWindow.close()
        
        if reply == 1:
            self.status_info.status_cmd.Terminate = True
            self.StartStopClicked()
        else:
            self.status_info.StatusSend(self.learninigServer, self.PORT)
        self.status_info.status_cmd.Terminate = False

    def serverChanged(self):
        if self._run_flag: self.StartStopClicked()

    def SwitchServer(self, server="localhost"):
        if self.GUI.port_input.text().isnumeric():
            self.PORT = int(self.GUI.port_input.text())
        if self.GUI.Server2.isChecked():
            self.learninigServer = 'Server2'
        if self.GUI.Eric_Personal.isChecked():
            self.learninigServer = 'ERIC_PERSONAL'
        if self.GUI.Nano.isChecked():
            self.learninigServer = 'Nano'
        if self.GUI.ubuntu1060.isChecked():
            self.learninigServer = 'Ubuntu1060'
            self.PORT = 58205
            self.GUI.port_input.setText("58205")

        if server == "twcc1":
            self.learninigServer = 'twcc1'
            self.PORT = 52000
            self.GUI.port_input.setText("52000")
        if server == "twcc2":
            self.learninigServer = "twcc2"
            self.PORT = 52000
            self.GUI.port_input.setText("52000")
        if server == "twcc3":
            self.learninigServer = 'twcc3'
            self.PORT = 52000
            self.GUI.port_input.setText("52000")
        # if server == 'localhost':
        if self.GUI.LocalhostRB.isChecked():
            self.learninigServer = 'localhost'
        if self.GUI.Manual.isChecked():
            self.learninigServer = self.GUI.address_input.text()


    def manualOnOff(self, flag):
            if flag == True:
                if self._run_flag:
                    self._manual_flag = True
                    self.status_info.status_cmd.manual_target[0] = True
                    self.status_info.status_cmd.manual_target[1] =[
                        self.GUI.xCmdSli.value()/10,
                        self.GUI.yCmdSli.value()/10
                    ]
                    self.status_info.StatusSend(self.learninigServer, self.PORT)
                    
                else:
                    self.GUI.statusbar.showMessage("Can't Start manual",2000)
                    self.GUI.manualStop.toggle()
            elif flag == False:
                self._manual_flag = False
                self.status_info.status_cmd.manual_target[0] = False
                self.status_info.StatusSend(self.learninigServer, self.PORT)
            else:
                pass

    def manualTargetSend(self):
        if self._manual_flag and self._run_flag:
            self.status_info.status_cmd.manual_target[1] = [
                self.GUI.xCmdSli.value()/10,
                self.GUI.yCmdSli.value()/10
            ]
            self.status_info.StatusSend(self.learninigServer, self.PORT)
        else:
            pass

    def OnOffDynamicLog(self):
        if self._run_flag:
            self._need_log = True
            if not(hasattr(self, "log_diagram")):
                self.log_diagram = FootPlot.Dynamic(
                    self.data.dt, self.dynamicLogUI,
                    self.PYQTGRAPH_ON, self.RL_FOOT,
                    self.data.gait_len
                    )
                self._dynamic_run = True
            self.dynamicLogUI.window.show()
        else:
            pass

    def onOffChangeRW(self):
        if self.data.reward_factor is not None:
            self.changeReward.reward_window.show()
            self.changeReward.showTable(self.data.reward_factor)

    def sentRWFactor(self):
        new_factor = self.changeReward.update_factor()
        self.status_info.status_cmd.reward_factor = new_factor
        print(self.status_info.status_cmd.reward_factor)
        self.status_info.StatusSend(self.learninigServer, self.PORT)
        self.status_info.status_cmd.reward_factor = None
        # self.changeReward.reward_window.close()

    def processTSNE(self, data):
        if not hasattr(self, 'tsne_generator'):
            data_len = data.gait_len if data.each_target_times == 0 else data.each_target_times*data.gait_len
            self.tsne_generator = TSNE.TSNE_GrapthGenerator(
                *data.learning_info,
                data_len
                )
            self.GUI.tsne_memory_progress.setMaximum(self.tsne_generator.memory_len)
            self.tsne_generator.push(data.tsne_info)
            tsne_data = self.tsne_generator.getTsne(self.GUI.perplexity_slider.value())
            return tsne_data
            # self.tsne_diagram.UpdatePlot(tsne_data)
        else:
            self.tsne_generator.push(data.tsne_info)
            tsne_data = self.tsne_generator.getTsne(self.GUI.perplexity_slider.value())
            return tsne_data
            # self.tsne_diagram.UpdatePlot(tsne_data)

    def processLoss(self, loss):
        self.history_loss.append(loss)
        sample_data = list()
        sample = round(len(self.history_loss)/100)+1
        for index, loss_value in enumerate(self.history_loss):
            if index % sample == 0:
                sample_data.append(loss_value)
        return (list(range(len(sample_data))), sample_data)

    def processTorque(self, torque):
        torque = np.absolute(torque)
        torque_difference = torque.max() - torque.min()
        self.history_torque_difference.append(torque_difference)


    def updateRealTime(self):
        if self.GUI.real_time_rb.isChecked():
            self.status_info.status_cmd.real_time = True
        else:
            self.status_info.status_cmd.real_time = False
        self.status_info.StatusSend(self.learninigServer, self.PORT)

    def updateInLearning(self):
        if self.GUI.off_learning_rb.isChecked():
            self.status_info.status_cmd.under_learning = False
        else:
            self.status_info.status_cmd.under_learning = True
        self.status_info.StatusSend(self.learninigServer, self.PORT)

    def ExitApp(self):
        self.status_info.status_cmd.check_learning = False
        self.status_info.StatusSend(self.learninigServer, self.PORT)
        print("Program Closed")
        sys.exit()




class MonitorUI(utUI.Ui_MainWindow):
    def __init__(self):
        self.monitor_window = QtWidgets.QMainWindow()
        self.setupUi(self.monitor_window)

class NeedSaveUI(nsUI.Ui_MainWindow):
    def __init__(self):
        self.NeedSaveWindow = QtWidgets.QMainWindow()
        self.setupUi(self.NeedSaveWindow)

class TernimateUI(tlUI.Ui_Dialog):
    def __init__(self):
        self.TernimateDialog = QtWidgets.QDialog()
        self.setupUi(self.TernimateDialog)

class DynamicLogUI(dyUI.Ui_MainWindow):
    def __init__(self):
        self.window = QtWidgets.QMainWindow()
        self.setupUi(self.window)

class RewardUI(rwUI.Ui_MainWindow):
    
    def __init__(self):
        self.reward_window = QtWidgets.QMainWindow()
        self.setupUi(self.reward_window)
        self.cencel_btn.clicked.connect(self.reward_window.close)

    def showTable(self,factor):
        self.factor = factor
        self.rw_factor_view.EditTrigger(2)
        self.rw_factor_view.setRowCount(3)
        self.rw_factor_view.setRowCount(len(self.factor))
        self.value_items = dict()
        for index, key in zip(range(len(factor)) , self.factor.keys()):
            self.rw_factor_view.setItem(index, 0, QtWidgets.QTableWidgetItem(key))
            self.rw_factor_view.setItem(index, 1, QtWidgets.QTableWidgetItem(str(self.factor[key])))
            item = QtWidgets.QTableWidgetItem()
            self.rw_factor_view.setItem(index, 2, item)
            self.value_items[key] = item
        self.rw_factor_view.setHorizontalHeaderLabels(["Reward by","Facotr","value"])
        self.rw_factor_view.resizeColumnsToContents()
        self.rw_factor_view.resizeRowsToContents()
        self.reward_window.show()

    def updateGUI_Value(self, each_reward):
        for key, value in each_reward.items():
            self.value_items[key].setText("{0:+.4f}".format(value))
        self.rw_factor_view.resizeColumnsToContents()

    def update_factor(self):
        new_factor = dict()
        for index, key in zip(range(len(self.factor)) , self.factor.keys()):
            try:
                value = float(self.rw_factor_view.item(index,1).text())
            except ValueError:
                value = self.factor[key]
            new_factor[key] = value
        return new_factor


def TestGround():
    pass


def main():
    App()

if __name__ == "__main__":
    main()
    # TestGround()
