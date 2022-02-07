import math, datetime
import numpy as np
import robot_leg_model as RFModel
import sys, os
import pickle
import time
import gait


class TrainingConfiguration():

    def __init__(self):
        self.object_type = "TrainingConfiguration"
        self.configuration_version = "1.0.1"
        self.configuration_discription = "enter discreption"
        self.self_checked = None
        self.learning_start_at   = None

        #Operation Configuration
        self.operating_frequency = 100
        self.state_number        = None
        self.action_range        = [None, None]   #radius per second
        self.action_resolution   = None           #must be odd

        #Environment Configuration
        self.foot_name           = None
        self.gait_name           = None
        self.off_ground_loading  = [0, 0]  #kg
        self.on_ground_loading   = [0, 0]  #kg
        self.step_long           = 15      #cm
        self.ground_level        = -30     #cm
        self.gait_resolution     = 30

        #QLearning Configuration
        self.network_shape = [
            None
        ]
        self.reward_factor = dict()
        self.learning_duration = None
        self.target_passed = None
        self.activated_reward = list()

        #information for continuous training
        self.target_passed = None



    def selfCheck(self):
        check_list = dict()

        if self.isNumber(self.operating_frequency):
            check_list['operating_frequency'] = True if self.operating_frequency > 0 else False

        check_list['state_number'] = isinstance(self.state_number,int)

        check_list['action_range'] = self.isNumber(self.action_range[0])
        check_list['action_range'] = self.isNumber(self.action_range[1])

        if self.isNumber(self.action_resolution):
            check_list['action_resolution'] = True if self.action_resolution % 2 == 1 else False
        else:
            check_list['action_resolution'] = False

        check_list['gait_name'] = True if self.gait_name is not None else False

        check_list['off_ground_loading'] = True if (
            self.isNumber(self.off_ground_loading[0]) and self.isNumber(self.off_ground_loading[1])
        ) else False

        check_list['on_ground_loading'] = True if (
            self.isNumber(self.on_ground_loading[0]) and self.isNumber(self.on_ground_loading[1])
        ) else False

        check_list['step_long'] = self.isNumber(self.step_long)

        check_list['ground_level'] = self.isNumber(self.ground_level)

        check_list['gait_resolution'] = isinstance(self.gait_resolution,int)

        check_list['network_shape'] = True if self.network_shape is not None else False

        result = True
        for key, value in check_list.items():
            result = result and value
            if not value: print("Configuration Problem with {0}".format(key))
        if result: print("Learning Configuration Check Passed")
        self.self_checked = result

    @staticmethod
    def isNumber(var):
        return isinstance(var,int) or isinstance(var,float)

def loadOldConfig(file_name):
    path = "./model/{0}.lrconf".format(file_name)
    with open(path, 'rb') as f:
        old_config = pickle.load(f)
    return old_config

def saveConfig(file_name, conf):
    path = "./model/{0}.lrconf".format(file_name)
    with open(path, 'wb+') as f:
        pickle.dump(conf,f)

def selectGait(learning_conf):
    selected_gait = None
    if learning_conf.gait_name == "Cycloid_Gait":
        selected_gait = gait.Cycloid_Gait(
            learning_conf.ground_level,
            learning_conf.step_long,
            learning_conf.gait_resolution
        )
    elif hasattr(gait, learning_conf.gait_name):
        selected_gait = getattr(gait, learning_conf.gait_name)(
            learning_conf.ground_level,
            learning_conf.step_long,
            learning_conf.gait_resolution
        )
    # elif learning_conf.gait_name == "Circle":
    #     selected_gait = gait.Circle_Gait(
    #         learning_conf.ground_level,
    #         learning_conf.step_long,
    #         learning_conf.step_long,
    #         learning_conf.gait_resolution
    #     )
    # elif learning_conf.gait_name == "Line":
    #     selected_gait = gait.Line_Gait(
    #         learning_conf.ground_level,
    #         learning_conf.step_long,
    #         learning_conf.gait_resolution
    #     )
    # elif learning_conf.gait_name == "Line_obs":
    #     selected_gait = gait.ObstacleGait_Line(
    #         learning_conf.ground_level,
    #         learning_conf.step_long,
    #         learning_conf.gait_resolution
    #     )
    return selected_gait

def selectFoot(lr_info):
    name = lr_info.foot_name if hasattr(lr_info,'foot_name') else lr_info
    print("{0} foot is selected".format(name))
    #lr_conf and data struc has sttr 'foot_name' else see as string raw name
    if name == "Foot":
        return RFModel.Foot()
    elif name == 'Foot10_10_15_Real':
        return RFModel.Foot10_10_15_Real()
    elif name == 'Foot14_14_7_Real':
        return RFModel.Foot14_14_7_Real()
    else:
        return RFModel.Foot()

def setActionSpace(config):
    if hasattr(config, "action_type"):
        if config.action_type == 'linear':
            MOTOR_ACTION_SPACE = np.linspace(
                config.action_range[0], config.action_range[1],
                num=config.action_resolution
            )
        elif config.action_type == "triple":
            # Tend to use x**3
            temp = np.linspace(-1, 1, config.action_resolution)
            MOTOR_ACTION_SPACE = (temp**3) * config.action_range[1]
        
        elif config.action_type == "ununiform" or config.action_type == '13':
            if config.action_resolution == 11:
                MOTOR_ACTION_SPACE = np.array([
                    -1, -0.7, -0.4, -0.2 , -0.1, 0 , 0.1, 0.2, 0.4 ,0.7, 1
                ]) * config.action_range[1]
            elif config.action_resolution == 13:
                MOTOR_ACTION_SPACE = np.array([
                    -1, -0.7, -0.4, -0.2 , -0.1, -0.05, 0 , 0.05, 0.1, 0.2, 0.4 ,0.7, 1
                ]) * config.action_range[1]
            elif config.action_resolution == 21:
                MOTOR_ACTION_SPACE = np.array([
                    -0.984, -0.912, -0.864, -0.768, -0.6, -0.48, -0.312, -0.216, -0.120, -0.048, -0.096,
                    0, 0.048, 0.096, 0.216, 0.264, 0.312, 0.48, 0.6, 0.768, 0.864, 0.984 
                ]) * config.action_range[1]
        else:
            MOTOR_ACTION_SPACE = np.linspace(
                config.action_range[0], config.action_range[1],
                num=config.action_resolution
            )
    else:
        MOTOR_ACTION_SPACE = np.linspace(
            config.action_range[0], config.action_range[1],
            num=config.action_resolution
        )
    return MOTOR_ACTION_SPACE

def processAction(Action, MOTOR_ACTION_SPACE):
    alen = len(MOTOR_ACTION_SPACE)
    m3c = Action // (alen**2)
    m2c = (Action - m3c*alen**2) // alen
    m1c = Action % alen
    v_cmd = [
        MOTOR_ACTION_SPACE[m3c],
        MOTOR_ACTION_SPACE[m2c],
        MOTOR_ACTION_SPACE[m1c]
    ]
    return v_cmd

def porcessMonitor(monitor_status, RFunc, QL, EXCU_MODE, learning_conf, total_steps, timer, **kwargs):
    update_steps = 100 # how many steps for show learning configuration and update command
    proxy_activated = False
    if monitor_status.proxy_server is not None:
        proxy_activated = monitor_status.proxy_server.activate_data_flow
    if monitor_status.status_cmd.check_learning or proxy_activated:

        #Register Data
        data_package = monitor_status.data_stru
        for key, value in kwargs.items():
            setattr(data_package,key,value)
        monitor_status.UpdateSendObjectUDP()
        
        monitor_status.StatusReceive()
        #Notification whether server is sending data
        if total_steps % int(update_steps/2) == 0: monitor_status.printSocketMsg()
        #Receivd factor change and update
        if monitor_status.status_cmd.reward_factor is not None:
            # RFunc.reward_factor = monitor_status.status_cmd.reward_factor
            RFunc.chengeFactor(monitor_status.status_cmd.reward_factor)
            print("Chenging Reward Factor to:")
            print("    " + str(RFunc.reward_factor))
            learning_conf.reward_factor = RFunc.reward_factor
            monitor_status.status_cmd.reward_factor = None
        #Back to simple Learning
        if not (monitor_status.status_cmd.check_learning or proxy_activated):
            print("\nStop Sending data: Learning Server return Normal")
            _ = timer.countDuration()
            monitor_status.data_stru.current_count = 0
            monitor_status.status_cmd.under_learning = True
            monitor_status.status_cmd.real_time = False
    
    elif total_steps%update_steps == 0:
        if kwargs['loss'] != None:
            msg = "Learning at {0:.2f} Hz, loss: {1:.10f}, under_learning: {2}, real_time: {3} port: {4}".format(
                update_steps/timer.countDuration(),
                kwargs['loss'],
                monitor_status.status_cmd.under_learning,
                monitor_status.status_cmd.real_time,
                monitor_status.default_port
            )
            print(msg)
        elif EXCU_MODE:
            msg = "Excute at {0:.2f} Hz, loss: None, under_learning: False, real_time: {1} port: {2}".format(
                update_steps/timer.countDuration(),
                monitor_status.status_cmd.real_time,
                monitor_status.default_port
            )
            print(msg)

        #Saveing TempFile for unpredictable learning fail
        if timer.saveTemp():
            print("Backuping temparary file")
            filename = 'temp/temp_{0}'.format(os.getpid())
            learning_conf.learning_start_at = str(timer.learning_begin)
            learning_conf.target_passed = kwargs['target_passed']
            QL.SaveModel(filename, learning_conf)
            print("Model Backuped")

        monitor_status.StatusReceive()
    
    if monitor_status.status_cmd.NeedSave[0]:
        QL.SaveModel(monitor_status.status_cmd.NeedSave[1],learning_conf,remove_temp=True)
        print("Model Saved")
        monitor_status.status_cmd.NeedSave[0] = False

def statelize(rl_foot, target):
    err_ang_flag, err_rad_seg = CurrentTargetErr(target, rl_foot.EndPoint)
    #relative angles
    m1_sAng = float(Ang2Seg(rl_foot.Upper_Motor.GetPosition("deg")))
    m2_sAng = float(Ang2Seg(rl_foot.Middle_Motor.GetPosition("deg")))
    m3_sAng = float(Ang2Seg(rl_foot.Lower_Motor.GetPosition("deg")))
    state = [err_ang_flag, err_rad_seg, m1_sAng, m2_sAng, m3_sAng]
    return state

def CurrentTargetErr(Target_point, CurrentEndPoint):
    X = CurrentEndPoint[0] - Target_point[0]
    Y = CurrentEndPoint[1] - Target_point[1]

    Deg = math.atan2(Y, X)
    err_ang_flag = 0
    if Deg < 0:
        Deg = Deg + 2*math.pi
    pSeg = 2*math.pi/50
    err_ang_flag = int(Deg/pSeg)

    dist = math.hypot(X,Y)  #cm
    if dist < 3:
        err_rad_seg = round(dist/0.1)
    elif dist < 5:
        err_rad_seg = round((dist-3)/0.2) + 30
    else:
        err_rad_seg = 40

    return float(err_ang_flag), float(err_rad_seg)

def Ang2Seg(Ang_Value):
    seg_ang = int(Ang_Value/(360/50))
    return seg_ang

class RLTimer():
    def __init__(self):
        self.learning_begin = datetime.datetime.now()
        self.learning_substart = time.time()
        self.old_time = 0
        self.each_target_dtime = 0.1
        self.each_target_otime = time.time()
        self.temp_file_duration = 60 #minutes
        self.save_temp_flag = False

    def updateOldTime(self):
        self.old_time = time.time()

    def countDuration(self, auto_update=True):
        duration = time.time() - self.old_time
        if duration == 0:
            duration = 0.00001
        self.old_time = time.time() if auto_update else self.old_time
        return duration

    def timeUnderLearning(self):
        duration = datetime.datetime.now() - self.learning_begin
        duration = duration - datetime.timedelta(microseconds = duration.microseconds)
        return str(duration)

    def underCurrentTarget(self):
        if time.time() - self.each_target_otime < self.each_target_dtime:
            return True
        else:
            self.each_target_otime = time.time()
            return False

    def saveTemp(self):
        if not self.save_temp_flag and datetime.datetime.now().minute % self.temp_file_duration == 0:
            self.save_temp_flag = True
            return not self.save_temp_flag
        elif self.save_temp_flag and datetime.datetime.now().minute % self.temp_file_duration != 0:
            self.save_temp_flag = False
            return not self.save_temp_flag
        else:
            pass

    def resetAllTimer(self):
        self.old_time = time.time()
        self.each_target_otime = time.time()

class TraceLogger(object):

    def __init__(self, folder_name):
        self.trace_data    = list()
        self.trace_index   = 1
        self.old_index     = None
        self.time_activate = False
        self.start_logging = False
        self.need_save     = True
        self.folder_path   = "./excuResult/trainingTrace/{0}".format(folder_name)
        os.mkdir(self.folder_path)

    def activate(self):
        self.time_activate = True

    def clearTrace(self):
        self.trace_data = list()
    
    def appendData(self, current_index, motor_info, time):
        if self.time_activate:
            if self.old_index is None and current_index == 0:
                self.start_logging = True
            elif self.old_index != 0 and current_index == 0:
                #Cycle Finish
                self.old_index = None
                self.start_logging = False
                self.time_activate = False
                self.need_save     = True
                self.saveData()
                return
            if self.start_logging:
                self.trace_data.append((motor_info.copy(), time, current_index))
                self.old_index = current_index
        else:
            pass

    def saveData(self):
        with open("{0}/{1:03d}.trace".format(self.folder_path, self.trace_index),"wb+") as f:
            pickle.dump(self.trace_data, f)
        self.clearTrace()
        print("Trace {0:03d} Saved".format(self.trace_index))
        self.trace_index += 1
