# import ROS_quadruped_control as ROS
# import matplotlib.pyplot as plt 
import numpy as np
import math
import quadruped_robot_model
import time,datetime
import traceback
import log
import pickle
import camera as cam
import cv2
try:
    import rospy
except ModuleNotFoundError:
    pass

class CPG_controller(quadruped_robot_model.QuadrupedRobot):

    def __init__(self,log_level="info", log_file_level="debug"):
        self.log_level = log_level
        self.log_file_level = log_file_level
        self.log = log.LogHandler(self.__class__.__name__, __name__, self.log_level, self.log_file_level)
        super().__init__(self.log_level, self.log_file_level)
        ### CPG parameter ###
        self.f_b1 = 0 #CPG feedback
        self.f_b2 = 0
        self.f_b3 = 0
        self.f_b4 = 0
        self.tau = 0.224
        self.tau_prime = 0.28
        self.w_0 = 2
        self.beta = 2.5
        self.c = 2.36
        self.o1_list, self.o2_list = list(), list()
        self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1, self.gain_1, self.bias_1 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
        self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2, self.gain_2, self.bias_2 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0

        ### quadruped robot parameter ###

        self.dt = 0.05

        self.lf_x, self.lf_y, self.lf_z = 0.015, 0, 0
        self.rf_x, self.rf_y, self.rf_z = -0.05, 0, 0
        self.lb_x, self.lb_y, self.lb_z = -0.05, 0, 0
        self.rb_x, self.rb_y, self.rb_z = 0.015, 0, 0

        self.z0 = 0.35
        self.Ax = 0.008 #0.006
        self.Az = 0.02 #0.015

        self.excu_step = 0
        self.steps = 0
        self.first_flag = True
        self.l_step_flag = False
        self.r_step_flag = False
        self.excu_mode = True

        self.numberOfSteps = 30

        ### Sensor ###

        self.cam = cam.Camera()
        self.offset = 0
        self.offset_list = list()

        

    
    def oscillator_next(self, u1, u2, v1, v2, y1, y2, s1, s2, bias, gain,f_b, dt):

        # The extensor neuron
        d_u1_dt = (-u1 - self.w_0*y2 -self.beta*v1 + self.c + f_b*s1)/self.tau
        d_v1_dt = (-v1 + y1)/self.tau_prime
        y1 = max([0.0, u1])

        # The flexor neuron
        d_u2_dt = (-u2 - self.w_0*y1 -self.beta*v2 + self.c - f_b*s2)/self.tau
        d_v2_dt = (-v2 + y2)/self.tau_prime
        y2 = max([0.0, u2])

        u1 += d_u1_dt * dt
        u2 += d_u2_dt * dt
        v1 += d_v1_dt * dt
        v2 += d_v2_dt * dt

        o = bias + gain*(y1 - y2)

        return u1, u2, v1, v2, y1, y2, o
    
    
    def CPGStep(self):
        """Run 10 iteration for a time step"""

        self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1 = self.oscillator_next(u1=self.u1_1, u2=self.u2_1,
                                                                                                            v1=self.v1_1, v2=self.v2_1,
                                                                                                            y1=self.y1_1, y2=self.y2_1,
                                                                                                            s1=1.0, s2=1.0,
                                                                                                            bias=0.0, gain=1.0,f_b=self.f_b1,
                                                                                                            dt=self.dt)
        self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2 = self.oscillator_next(u1=self.u1_2, u2=self.u2_2,
                                                                                                             v1=self.v1_2, v2=self.v2_2,
                                                                                                             y1=self.y1_2, y2=self.y2_2,
                                                                                                             s1=1.0, s2=1.0,
                                                                                                             bias=0.0, gain=1.0,f_b=self.f_b2,
                                                                                                             dt=self.dt)
        lf_cmd = self.trajectoryLF()
        rf_cmd = self.trajectoryRF()
        lb_cmd = self.trajectoryLB()
        rb_cmd = self.trajectoryRB()
        if self.excu_step%5 == 0 and self.steps != 0:
            lf_pos, rf_pos, lb_pos, rb_pos = 270, 270, 270, 270
            self.executeIK(lf_cmd=lf_cmd,rf_cmd=rf_cmd,lh_cmd=lb_cmd,rh_cmd=rb_cmd,lf_pos=lf_pos,rf_pos=rf_pos,lh_pos=lb_pos,rh_pos=rb_pos) 
            time.sleep(0.02) 
        self.excu_step += 1
    def trajectoryLF(self):
        if self.o_1 < 0:
            self.l_step_flag = False
            self.lf_z = self.z0 + self.Az * self.o_1
        else:
            if not self.l_step_flag:
                self.l_step_flag = True
                self.lf_x = 0.015
                self.rb_x = 0.015
            self.lf_z = self.z0
        self.lf_x -= self.Ax * self.o_1
        lf_x_real = self.lf_x * 100    # m -> cm
        lf_z_real = self.lf_z * 100
        cmd_list = [lf_x_real,-lf_z_real]
        return cmd_list


    
        
    def trajectoryRF(self):
        if self.o_2 > 0:
            if not self.r_step_flag:
                self.r_step_flag = True
                self.steps += 1
                self.rf_x = -0.05
                self.lb_x = -0.05
            self.rf_z = self.z0 - self.Az * self.o_2
        else:
            self.r_step_flag = False
            self.rf_z = self.z0
        self.rf_x += self.Ax * self.o_2
        rf_x_real = self.rf_x * 100
        rf_z_real = self.rf_z * 100
        cmd_list = [rf_x_real,-rf_z_real]
        return cmd_list


        
    def trajectoryRB(self):
        if self.o_1 < 0:
            self.rb_z = self.z0 + self.Az * self.o_1
        else:
            self.rb_z = self.z0
        self.rb_x -= self.Ax * self.o_1
        rb_x_real = self.rb_x * 100
        rb_z_real = self.rb_z * 100
        cmd_list = [rb_x_real,-rb_z_real]
        return cmd_list


    
    def trajectoryLB(self):
        if self.o_2 > 0:
            self.lb_z = self.z0 - self.Az * self.o_2
        else:
            self.lb_z = self.z0
        self.lb_x += self.Ax * self.o_2
        lb_x_real = self.lb_x * 100
        lb_z_real = self.lb_z * 100
        cmd_list = [lb_x_real,-lb_z_real]
        return cmd_list

    def updateSteps(self):
        return self.steps - 1


    
    def excuFbCmd(self,fb_cmd,fb_mode = True):
        current_steps = self.updateSteps()
        if self.first_flag:
            new_steps = current_steps + 2
        else:
            new_steps = current_steps + 1 
        self.first_flag = False
        if fb_cmd is not None and fb_mode is True:
            while True:
                if not current_steps == new_steps:
                    self.f_b1 = fb_cmd[0]
                    self.f_b2 = fb_cmd[1]
                    self.CPGStep()
                    current_steps = self.updateSteps()
                else:
                    self.offset = self.cam.updateOffset()
                    self.offset_list.append(self.offset)
                    break
    
    def updateOffset(self):
        return self.offset
    
    def excuCPGGait(self,cycle=30):
        self.excuFbCmd([0,0])
        for step in range(cycle):
            fb_cmd = [0,0]
            self.excuFbCmd(fb_cmd)
        self.saveData(excu_mode=True)
        self.standupFromGround()
            

    
    def standupFromGround(self):
        self.slowlySetAllMotorPositionIK(8, -16, 270, resolution=40)
        self.slowlyPosition(
            lf_cmd=[0, -20],
            rf_cmd=[0, -20],
            lh_cmd=[0, -30],
            rh_cmd=[0, -30],
        )
        self.slowlySetAllMotorPositionIK(-3.5, -35, 270)

    def sitdownFromStand(self):
        self.slowlyPosition(
            lf_cmd=[10, -20],
            rf_cmd=[10, -20],
            lh_cmd=[10, -30],
            rh_cmd=[10, -30]
        )
        self.slowlyPosition(
            lh_cmd=[10, -20],
            rh_cmd=[10, -20],
        )
        self.slowlySetAllMotorPositionIK(8, -16)

    def saveData(self,excu_mode=False):
        if excu_mode:
            fileName = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            fileName = "./data/" + fileName
            data = [self.offset_list]
            with open(fileName + ".aldata", 'wb+') as f:
                pickle.dump(data,f)
            self.offset_list = list()
    
    
                

    


        








if __name__ == '__main__':
    device = "com3"
    # device = "/dev/ttyUSB0"
    robot = CPG_controller()
    robot.activateDXLConnection(device)
    robot.activateAllRealMotor()
    robot.armAllMotor()
    robot.standupFromGround()
    

    command_dict = {
        "standup":robot.standupFromGround,
        "sitdown":robot.sitdownFromStand,
        "cpg":robot.excuCPGGait,
        "correct":robot.cam.camCorrection,
    }


    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                robot.cam.closeCam()
                break
        except Exception as e:
            traceback.print_exc()
            robot.cam.closeCam()
            break
    
    robot.sitdownFromStand()
    robot.disarmAllMotor()
    robot.deactivateDXLConnection()
    