import numpy as np
import time
import threading
import multiprocessing

from numpy.core.records import array



import robot_leg_model
import log
try:
    import DXL_motor_control as DXL
    from DXL_motor_control import DXL_Conmunication
    HAS_DXL_SDK = True
except ModuleNotFoundError:
    HAS_DXL_SDK = False


class QuadrupedRobot:
    '''
    This class manage the operation of individual foot and shoulder motor
    of quadruped robot
    '''
    def __init__(self, log_level="info", log_file_level="debug"):
        '''
            Initial parameters
        '''
        self.log_level = log_level
        self.log_file_level = log_file_level
        self.log = log.LogHandler(self.__class__.__name__, __name__, self.log_level, log_file_level)
        self.log_len = 500
        self.bool_dxl_connected = False
        self.bool_real_motor_mode = False
        self.bool_need_write = False
        self.str_motor_operating_mode = "p"

        self.leg_lf = robot_leg_model.LegLF()
        self.leg_rf = robot_leg_model.LegRF()
        self.leg_lh = robot_leg_model.LegLH()
        self.leg_rh = robot_leg_model.LegRH()
        self.shoulder = robot_leg_model.Shoulder()

        self.leg_system = [self.leg_lf, self.leg_rf, self.leg_lh, self.leg_rh]
        self.lower_system = [self.leg_lf, self.leg_rf, self.leg_lh, self.leg_rh, self.shoulder]

        self.event_read = threading.Event()
        self.event_write = threading.Event()
        self.event_terminate_thread = threading.Event()

    def activateAllRealMotor(self, device_name='/dev/ttyUSB0') -> None:
        '''
            Activate all dxl motor
            No return
        '''
        if not self.bool_dxl_connected:
            self.activateDXLConnection(device_name)
        
        if self.bool_dxl_connected:
            for system in self.lower_system:
                if not system.real_motor_mode:
                    system.activateRealMotor(device_name, dxl_communicator=self.dxl_communicator, need_reboot=False)
                    self.bool_real_motor_mode = True
            self.rebootAllMotor()
            self.dxl_communicator.activateIndirectMode()
            self.event_read.set()
            self.event_terminate_thread.clear()
            self.event_write.clear()
            self.thread_update_motor_data = threading.Thread(
                target=self._threadUpdateAllMotorData,
                daemon=True
            )
            self.thread_update_motor_data.start()
            time.sleep(1)
                



    def activateDXLConnection(self, device_name='/dev/ttyUSB0'):
        '''
            Establish connection to DXL controller
        '''
        if HAS_DXL_SDK:
            if self.bool_dxl_connected:
                pass
            else:
                if not hasattr(self, "dxl_communicator"):
                    self.dxl_communicator = DXL.DXL_Conmunication(device_name, log_level=self.log_level, log_file_level=self.log_file_level)
                self.dxl_communicator.activateDXLConnection()
                self.bool_dxl_connected = True
        else:
            self.log.error("\nPlease Install Dynamixel SDK first to activate DXL motor")
            self.log.error("Follow the instruction : https://github.com/ROBOTIS-GIT/DynamixelSDK\n")

    def armAllMotor(self):
        '''
            Power all motors
            process : check connection >>> check activate >>> true: arm motor
        '''
        if self.bool_dxl_connected:
            self.eventWaitUntilWrite()
            for system in self.lower_system:
                if system.real_motor_mode:
                    system.armAllMotor()
            self.eventStartRead()

    def cutOffLog(self, num):
        for system in self.lower_system:
            system.cutOffLog(num)


    def disarmAllMotor(self):
        '''
            Cut the power for motor
            process : check connection >>> check if arm >>>true: disarm
        '''
        if self.bool_dxl_connected:
            self.eventWaitUntilWrite()
            for system in self.lower_system:
                if system.joint_motor_armed:
                    system.disarmAllMotor()
            self.eventStartRead()

    def deactivateDXLConnection(self):
        '''
            Cut connection of DXL controller
            process : check connection >>> true: disarm all motor, close port, change real_motor_mode tage
        '''
        if self.bool_dxl_connected:
            self.event_terminate_thread.set()
            if self.thread_update_motor_data.is_alive():
                time.sleep(0.05)
            self.disarmAllMotor()
            for system in self.lower_system:
                system.real_motor_mode = False
            time.sleep(0.2)
            self.dxl_communicator.closeHandler()
            time.sleep(0.1)
            self.bool_dxl_connected = False

    def endPointCalculation(self):
        '''
            Calculate all end point on all foot and shoulder
        '''
        for system in self.lower_system:
            system.endPointCalculation()


    def executeAllIK(self, cmd_x, cmd_y, dt=0.05, bool_dynamic=False):
        '''
            Control 4 feet with IK
        '''
        for leg in self.leg_system:
            leg.InverseKinematic(cmd_x, cmd_y, dt, dynamic=bool_dynamic, send_command=False)
        if self.bool_dxl_connected:
            # self.eventWaitUntilWrite()
            # self.dxl_communicator.sentAllCmd()
            self.bool_need_write = True
            # self.eventStartRead()
            time.sleep(dt)

    def executeIK(self,
        lf_cmd=None, rf_cmd=None, lh_cmd=None, rh_cmd=None,
        lf_pos=None, rf_pos=None, lh_pos=None, rh_pos=None,
        dt=0.05, **kwargs):
        '''
            For safty operation, please use keyword "lf_cmd", "rf_cmd", "lh_cmd", "rh_cmd", "shoulder_cmd"\n
            as keyword to use this method. Each keyword value should be a list of [X_cmd, Y_cmd]\n
            For should_cmd should be a "lf_pos", "rf_pos", "lh_pos", "rh_pos" \n
            If not intend to move some shoulder joint, then just leave it as None. For example [None, 250, None, 290]\n
            then only fr, rr shoulder will move.
        '''
        has_s_lf, has_s_rf, has_s_lh, has_s_rh = False, False, False, False
        if lf_cmd is not None:
            self.leg_lf.InverseKinematic(lf_cmd[0], lf_cmd[1], send_command=False)
        if rf_cmd is not None:
            self.leg_rf.InverseKinematic(rf_cmd[0], rf_cmd[1], send_command=False)
        if lh_cmd is not None:
            self.leg_lh.InverseKinematic(lh_cmd[0], lh_cmd[1], send_command=False)
        if rh_cmd is not None:
            self.leg_rh.InverseKinematic(rh_cmd[0], rh_cmd[1], send_command=False)
        if lf_pos is not None:
            has_s_lf = True
        if rf_pos is not None:
            has_s_rf = True
        if lh_pos is not None:
            has_s_lh = True
        if rh_pos is not None:
            has_s_rh = True

        self.log.debug(f"leg command {[lf_cmd, rf_cmd, lh_cmd, rh_cmd]}", self.executeIK)
        if any([has_s_lf, has_s_rf, has_s_lh, has_s_rh]):
            self.log.debug(f"shoulder command {[lf_pos, rf_pos, lh_pos, rh_pos]}", self.executeIK)
            self.shoulder.excu_position_cmd(
                    [lf_pos, rf_pos, lh_pos, rh_pos], send_command=False
                )
        if self.bool_dxl_connected:
            # self.eventWaitUntilWrite()
            # self.dxl_communicator.sentAllCmd()
            self.bool_need_write = True
            # self.eventStartRead()
            time.sleep(dt)
            # self.updateAllMotorData()
            # self.eventStartRead()

    def executePosCmd(self, joint_position=[0, 0], shoulder_position=[0, 0, 0, 0], dt=0.01):
        for leg in self.leg_system:
            leg.excu_position_cmd(p_cmd=joint_position, dt=dt, send_command=False)
        self.shoulder.excu_position_cmd(shoulder_position, send_command=False)
        if self.bool_dxl_connected:
            # self.eventWaitUntilWrite()
            # self.dxl_communicator.sentAllCmd()
            self.bool_need_write = True
            # self.eventStartRead()
            time.sleep(dt)

    def getFootEndPointPath(self, x_cmd, y_cmd, end_point, resolution) -> array:
        '''
            Return path from current EndPoint to target cmd, with resolution (2D [[x...], [y...]])
        '''
        path = np.stack([
            np.linspace(end_point[0], x_cmd, num=resolution),
            np.linspace(end_point[1], y_cmd, num=resolution)
        ])
        return path
    
    def getShoulderPath(self, cmd, current_pos, resolution) -> array:
        '''
            Return path from current pos to target pos with resolution (1D array)
        '''
        path = np.linspace(current_pos, cmd, num=resolution)
        return path

    def rebootAllMotor(self):
        '''
            Reboot all DXL motor
        '''
        if self.bool_dxl_connected:
            self.eventWaitUntilWrite()
            self.dxl_communicator.rebootAllMotor()
            self.eventStartRead()

    def reinitialAllMotor(self):
        if self.bool_dxl_connected:
            self.eventWaitUntilWrite()
        for system in self.lower_system:
            system.reinitial()
        self.eventStartRead()

    def resetLog(self):
        for system in self.lower_system:
            system.resetLog()

    def resetLoglength(self, log_len):
        '''
            Reset data log length in all motors
        '''
        for system in self.lower_system:
            system.resetLoglength(log_len)

    def resetToStandup(self):
        self.slowlySetAllMotorPositionIK(0, -35, 270)
    
    def resetToSitDown(self):
        self.slowlySetAllMotorPositionIK(8, -16)

    def setLogLevel(self, log_level="info", log_file_level="debug"):
        '''
            Set current log level to target level
        '''
        self.log.setLogLevel(log_level, log_file_level)
        self.log_level = log_level
        self.log_file_level = log_file_level
        if hasattr(self, "dxl_communicator"):
            self.dxl_communicator.setLogLevel(log_level, log_file_level)

    def slowlyPosition(self,
        lf_cmd=None, rf_cmd=None, lh_cmd=None, rh_cmd=None,
        lf_pos=None, rf_pos=None, lh_pos=None, rh_pos=None,
        resolution=50, dt=0.05, **kwargs):
        '''
            For safty operation, please use keyword "lf_cmd", "rf_cmd", "lh_cmd", "rh_cmd", "shoulder_cmd"\n
            as keyword to use this method. Each keyword value should be a list of [X_cmd, Y_cmd]\n
            For should_cmd should be a "lf_pos", "rf_pos", "lh_pos", "rh_pos" \n
            If not intend to move some shoulder joint, then just leave it as None. For example [None, 250, None, 290]\n
            then only fr, rr shoulder will move.
        '''
        has_lf, has_rf, has_lh, has_rh = False, False, False, False
        has_s_lf, has_s_rf, has_s_lh, has_s_rh = False, False, False, False
        path_lf, path_rf, path_lh, path_rh = None, None, None, None
        path_shoulder_lf, path_shoulder_rf, path_shoulder_lh, path_shoulder_rh = None, None, None, None
        if dt < 0.015: dt = 0.015
        if lf_cmd is not None:
            has_lf = True
            path_lf = self.getFootEndPointPath(*lf_cmd, self.leg_lf.EndPoint, resolution)
        if rf_cmd is not None:
            has_rf = True
            path_rf = self.getFootEndPointPath(*rf_cmd, self.leg_rf.EndPoint, resolution)
        if lh_cmd is not None:
            has_lh = True
            path_lh = self.getFootEndPointPath(*lh_cmd, self.leg_lh.EndPoint, resolution)
        if rh_cmd is not None:
            has_rh = True
            path_rh = self.getFootEndPointPath(*rh_cmd, self.leg_rh.EndPoint, resolution)
        if lf_pos is not None:
            has_s_lf = True
            path_shoulder_lf = self.getShoulderPath(lf_pos, self.shoulder.shoulder_front_left.GetPosition("deg"), resolution)
        else:
            path_shoulder_lf = self.getShoulderPath(
                self.shoulder.shoulder_front_left.GetPosition("deg"),
                self.shoulder.shoulder_front_left.GetPosition("deg"), resolution)
        if rf_pos is not None:
            has_s_rf = True
            path_shoulder_rf = self.getShoulderPath(rf_pos, self.shoulder.shoulder_front_right.GetPosition("deg"), resolution)
        else:
            path_shoulder_rf = self.getShoulderPath(
                self.shoulder.shoulder_front_right.GetPosition("deg"),
                self.shoulder.shoulder_front_right.GetPosition("deg"), resolution)
        if lh_pos is not None:
            has_s_lh = True
            path_shoulder_lh = self.getShoulderPath(lh_pos, self.shoulder.shoulder_rear_left.GetPosition("deg"), resolution)
        else:
            path_shoulder_lh = self.getShoulderPath(
                self.shoulder.shoulder_rear_left.GetPosition("deg"),
                self.shoulder.shoulder_rear_left.GetPosition("deg"), resolution)
        if rh_pos is not None:
            has_s_rh = True
            path_shoulder_rh = self.getShoulderPath(rh_pos, self.shoulder.shoulder_rear_right.GetPosition("deg"), resolution)
        else:
            path_shoulder_rh = self.getShoulderPath(
                self.shoulder.shoulder_rear_right.GetPosition("deg"),
                self.shoulder.shoulder_rear_right.GetPosition("deg"), resolution)

        self.log.debug(f"foot command {[lf_cmd, rf_cmd, lh_cmd, rh_cmd]}", self.slowlyPosition)
        if any([has_s_lf, has_s_rf, has_s_lh, has_s_rh]):
            self.log.debug(f"shoulder command {[lf_pos, rf_pos, lh_pos, rh_pos]}", self.slowlyPosition)
        # self.eventWaitUntilWrite()
        for i in range(resolution):
            if has_lf:
                if not self.leg_lf.checkWorkingSpace(path_lf[0][i], path_lf[1][i]):
                    self.leg_lf.InverseKinematic(path_lf[0][i], path_lf[1][i], send_command=False)
            if has_rf:
                if not self.leg_rf.checkWorkingSpace(path_rf[0][i], path_rf[1][i]):
                    self.leg_rf.InverseKinematic(path_rf[0][i], path_rf[1][i], send_command=False)
            if has_lh:
                if not self.leg_lh.checkWorkingSpace(path_lh[0][i], path_lh[1][i]):
                    self.leg_lh.InverseKinematic(path_lh[0][i], path_lh[1][i], send_command=False)
            if has_rh:
                if not self.leg_rh.checkWorkingSpace(path_rh[0][i], path_rh[1][i]):
                    self.leg_rh.InverseKinematic(path_rh[0][i], path_rh[1][i], send_command=False)
            if any([has_s_lf, has_s_rf, has_s_lh, has_s_rh]):
                self.shoulder.excu_position_cmd(
                    [path_shoulder_lf[i], path_shoulder_rf[i], path_shoulder_lh[i], path_shoulder_rh[i]], send_command=False
                )
            if self.bool_dxl_connected:
                # self.eventWaitUntilWrite()
                # self.dxl_communicator.sentAllCmd()
                self.bool_need_write = True
                # self.eventStartRead()
                time.sleep(dt)
                # self.updateAllMotorData()
        # self.eventStartRead()
            




    def slowlySetAllMotorPositionIK(self, x_cmd=0, y_cmd=-32, shoulder_cmd=270, resolution=100, dt=0.05):
        '''
            Reset all motor position to its initial pos\n
            argv : x_cmd , y_cmd : X,Y position of the foot end point\n
                shoulder cmd (deg) global angle of shoulder (270 is downward)
        '''
        # if self.bool_dxl_connected:
        #     if self.str_motor_operating_mode == 'v':
        #         self.disarmAllMotor()
        #         self.switchMotorMode("position")
        #     self.armAllMotor()
        # self.updateAllMotorData()
        self.endPointCalculation()
        path_lf = self.getFootEndPointPath(x_cmd, y_cmd, self.leg_lf.EndPoint, resolution)
        path_rf = self.getFootEndPointPath(x_cmd, y_cmd, self.leg_rf.EndPoint, resolution)
        path_lh = self.getFootEndPointPath(x_cmd, y_cmd, self.leg_lh.EndPoint, resolution)
        path_rh = self.getFootEndPointPath(x_cmd, y_cmd, self.leg_rh.EndPoint, resolution)
        path_shoulder_lf = self.getShoulderPath(shoulder_cmd, self.shoulder.shoulder_front_left.GetPosition("deg"), resolution)
        path_shoulder_rf = self.getShoulderPath(shoulder_cmd, self.shoulder.shoulder_front_right.GetPosition("deg"), resolution)
        path_shoulder_lh = self.getShoulderPath(shoulder_cmd, self.shoulder.shoulder_rear_left.GetPosition("deg"), resolution)
        path_shoulder_rh = self.getShoulderPath(shoulder_cmd, self.shoulder.shoulder_rear_right.GetPosition("deg"), resolution)

        # self.eventWaitUntilWrite()
        for i in range(resolution):
            if not self.leg_lf.checkWorkingSpace(path_lf[0][i], path_lf[1][i]):
                self.leg_lf.InverseKinematic(path_lf[0][i], path_lf[1][i], send_command=False)
            if not self.leg_rf.checkWorkingSpace(path_rf[0][i], path_rf[1][i]):
                self.leg_rf.InverseKinematic(path_rf[0][i], path_rf[1][i], send_command=False)
            if not self.leg_lh.checkWorkingSpace(path_lh[0][i], path_lh[1][i]):
                self.leg_lh.InverseKinematic(path_lh[0][i], path_lh[1][i], send_command=False)
            if not self.leg_rh.checkWorkingSpace(path_rh[0][i], path_rh[1][i]):
                self.leg_rh.InverseKinematic(path_rh[0][i], path_rh[1][i], send_command=False)
            # if not self.shoulder.checkWorkingSpace([path_shoulder_lf[i], path_shoulder_rf[i], path_shoulder_lh[i], path_shoulder_rh[i]]):
            self.shoulder.excu_position_cmd([path_shoulder_lf[i], path_shoulder_rf[i], path_shoulder_lh[i], path_shoulder_rh[i]], send_command=False)
            if self.bool_dxl_connected:
                # self.eventWaitUntilWrite()
                # self.dxl_communicator.sentAllCmd()
                self.bool_need_write = True
                # self.eventStartRead()
                time.sleep(dt)
                # self.updateAllMotorData()
        # self.eventStartRead()


    def switchMotorMode(self, mode='position'):
        '''
            Switch motor command mode : "position" or "velocity"
        '''
        if self.bool_dxl_connected:
            self.eventWaitUntilWrite()
            for system in self.lower_system:
                if system.real_motor_mode:
                    system.switchMotorMode(mode)
            self.eventStartRead()

        if mode == "position":
            self.str_motor_operating_mode = "p"
        else:
            self.str_motor_operating_mode = "v"


    def _threadUpdateAllMotorData(self):
        '''
            This handler will keep getting real motor data when activated.\n
            And will block if there is other things to do. \n
            The control method is based on event driven mode for efficiency. \n

            control logic:
            if not write & read is true -> read
            if want to write: clear read -> write will set and  real will wait until read is set
        '''
        event_trigger = threading.Event()
        # last_trigger_time = time.time()
        # def trigger():
            # while not self.event_terminate_thread.is_set():
                # event_trigger.set()
                # time.sleep(0.005)

        # thread_trigger = threading.Thread(target=trigger, daemon=True)
        # thread_trigger.start()
        count_time = time.time()
        iteration = 0
        while not self.event_terminate_thread.is_set():
            if not self.event_write.is_set() and self.event_read.is_set():
                event_trigger.wait(0.005)
                # time.sleep(0.009)
                # if (time.time() - last_trigger_time) >= 0.02:
                if self.bool_need_write:
                    self.dxl_communicator.sentAllCmd()
                    self.bool_need_write = False
                self.updateAllMotorData()
                iteration += 1
                event_trigger.clear()
                
            else:
                self.event_write.set()
                self.event_read.wait()
                self.event_write.clear()
            

            if time.time() - count_time >= 1:
                print("freq : {:0.2f}".format(iteration/(time.time()-count_time)))
                iteration = 0
                count_time = time.time()






    def updateAllMotorData(self, delay=-1):
        '''
        delay = Extra delay for communication, -1 means no delay.
        Update real motor info and check for interference and calculate end point location
        
            Modified by Clement, add update_noew option. if false means we do update data manually.
        '''
        if self.bool_dxl_connected:
            self.dxl_communicator.updateMotorData(delay=delay)
            for system in self.lower_system:
                if system.real_motor_mode:
                    system.updateRealMotorsInfo(update_now=False)
            self.endPointCalculation()
        else:
            self.log.warning("Not connect to DXL yet", self.updateAllMotorData)
            self.log.warning("Use activateDXLConnection() first", self.updateAllMotorData)



    def eventWaitUntilWrite(self):
        if hasattr(self, "thread_update_motor_data"):
            if self.thread_update_motor_data.is_alive():
                # self.log.debug("Start wait for write", self.eventWaitUntilWrite)
                # start_t = time.time()
                self.event_read.clear()
                self.event_write.wait(0.5)
                # if self.event_write.wait(0.5):
                    # self.log.debug("Stop wait for write {:0.3f}".format(time.time() - start_t), self.eventWaitUntilWrite)
                # else:
                    # self.log.debug("Event_write.wait timeout ! Thread is probably dead or DXL communication has problems", self.eventWaitUntilWrite)


    def eventStartRead(self):
        # self.log.debug("Start read", self.eventStartRead)
        self.event_write.clear()
        self.event_read.set()


