import ROS_quadruped_control as ROS
import matplotlib.pyplot as plt 
import numpy as np
import rospy
import math
import time, datetime
import traceback
import pickle


class CPG_controller(object):

    def __init__(self):
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
        self.o1_list = list()
        self.o2_list = list()
        self.y1_list = list()
        self.y2_list = list()
        self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1, self.gain_1, self.bias_1 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
        self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2, self.gain_2, self.bias_2 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
        self.u1_3, self.u2_3, self.v1_3, self.v2_3, self.y1_3, self.y2_3, self.o_3, self.gain_3, self.bias_3 = 0.4432, 0.4334, 0.1885, 0.69, 0.0, 0.0, 0.0, 1.0, 0.0
        self.u1_4, self.u2_4, self.v1_4, self.v2_4, self.y1_4, self.y2_4, self.o_4, self.gain_4, self.bias_4 = 0.4432, 0.4334, 0.1885, 0.69, 0.0, 0.0, 0.0, 1.0, 0.0

        ### quadruped robot parameter ###

        self.dt = 0.05 #0.05

        self.r_x_list, self.r_z_list = list(), list()
        self.l_x_list, self.l_z_list = list(), list()
        self.distance_list = list()
        self.offset_list = list()
        self.f_b1_list, self.f_b2_list = list(), list()
        # self.rf_hip_data, self.rf_leg_data, self.rf_knee_data = list(), list(), list()
        # self.lf_hip_data, self.lf_leg_data, self.lf_knee_data = list(), list(), list()
        # self.lb_hip_data, self.lb_leg_data, self.lb_knee_data = list(), list(), list()
        # self.rb_hip_data, self.rb_leg_data, self.rb_knee_data = list(), list(), list()
        self.yaw_list = list()
        self.cog_x, self.cog_y, self.cog_z = list(), list(), list()
        self.rf_contact, self.lf_contact, self.lb_contact, self.rb_contact = list(), list(), list(), list()

        

        self.lf_x, self.lf_y, self.lf_z = 0.015, 0, 0
        self.rf_x, self.rf_y, self.rf_z = -0.05, 0, 0
        self.lb_x, self.lb_y, self.lb_z = -0.05, 0, 0
        self.rb_x, self.rb_y, self.rb_z = 0.015, 0, 0

        self.z0 = 0.35
        self.Ax = 0.05 #0.05
        self.A1x = 0.006
        self.A2x = 0.006
        self.Az = 0.015
        self.x0 = -0.03

        self.excu_step = 0
        self.steps = 0
        self.first_flag = True
        self.l_step_flag = False
        self.r_step_flag = False
        self.excu_mode = True

        self.numberOfSteps = 30



        self.RF_foot = ROS.Foot('RF')
        self.LF_foot = ROS.Foot('LF')
        self.RB_foot = ROS.Foot('RB')
        self.LB_foot = ROS.Foot('LB')

        self.ROS_mode = False
    
    def resetAllPos(self):
        if self.ROS_mode:
            self.RF_foot.resetPos()
            self.RB_foot.resetPos()
            self.LF_foot.resetPos()
            self.LB_foot.resetPos()
        else:
            print("fail")
    

    
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
        # self.excuFbCmd()
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
        if self.excu_step%5 == 0 and self.excu_mode is True and self.steps != 0:   ##step for control command
            self.LF_foot.sendAllCommand(lf_cmd)
            self.LB_foot.sendAllCommand(lb_cmd)
            self.RF_foot.sendAllCommand(rf_cmd)
            self.RB_foot.sendAllCommand(rb_cmd)
            self.r_x_list.append(self.rf_x)
            self.r_z_list.append(-self.rf_z)
            self.l_x_list.append(self.lf_x)
            self.l_z_list.append(-self.lf_z)
            cog = self.RF_foot.ros_comunicator.updateCOG()
            self.cog_x.append(cog[0])
            self.cog_y.append(cog[1])
            self.cog_z.append(cog[2])
            # contact = self.RF_foot.ros_comunicator.updateContactData()
            # rf_contact, lf_contact, lb_contact, rb_contact = contact[0], contact[1], contact[2], contact[3]
            # self.rf_contact.append(rf_contact)
            # self.lf_contact.append(lf_contact)
            # self.lb_contact.append(lb_contact)
            # self.rb_contact.append(rb_contact) 
        self.excu_step += 1  


    


    def trajectoryLF(self):
        if self.o_1 < 0:  # <
            self.l_step_flag = False
            self.lf_z = self.z0 + self.Az * self.o_1 # +
        else:
            # lf_z = self.z0
            if not self.l_step_flag:
                self.l_step_flag = True
                self.lf_x = 0.015
                self.rb_x = 0.015
            self.lf_z = self.z0
        self.lf_x -= self.A2x * self.o_1 #-=
        p_cmd = self.LF_foot.leg_ikine(self.lf_x,0,self.lf_z)
        # p_cmd[0] = lf_y
        return p_cmd 

    
        
    def trajectoryRF(self):
        if self.o_2 > 0: #  >
            # rf_z = self.z0 - self.Az*self.o_2
            if not self.r_step_flag:
                self.r_step_flag = True
                self.steps += 1
                self.rf_x = -0.05
                self.lb_x = -0.05
            self.rf_z = self.z0 - self.Az * self.o_2 # -
        else:
            self.r_step_flag = False
            self.rf_z = self.z0
            # rf_z = self.z0
        # rf_y = self.Ay*self.o_4
        # rf_x = self.x0 + self.Ax*self.o_2
        self.rf_x += self.A1x * self.o_2 #+=
        p_cmd = self.RF_foot.leg_ikine(self.rf_x,0,self.rf_z)
        # p_cmd[0] = rf_y
        return p_cmd 

        
    def trajectoryRB(self):
        if self.o_1 < 0: # <
            # rb_z = self.z0 + self.Az*self.o_1
            self.rb_z = self.z0 + self.Az * self.o_1 # +
        else:
            # rb_z = self.z0
            self.rb_z = self.z0
        # rb_y = -self.Ay*self.o_3
        # rb_x = self.x0 - self.Ax*self.o_1
        self.rb_x -= self.A2x * self.o_1 #-=
        p_cmd = self.RB_foot.leg_ikine(self.rb_x,0,self.rb_z)
        # p_cmd[0] = rb_y
        return p_cmd 

    
    def trajectoryLB(self):
        if self.o_2 > 0: # >
            # lb_z = self.z0 - self.Az*self.o_2
            self.lb_z = self.z0 - self.Az * self.o_2 # -
        else:
            # lb_z = self.z0
            self.lb_z = self.z0
        # lb_y = -self.Ay*self.o_4
        # lb_x = self.x0 + self.Ax*self.o_2 
        self.lb_x += self.A1x * self.o_2 # +=
        p_cmd = self.LB_foot.leg_ikine(self.lb_x,0,self.lb_z)
        # p_cmd[0] = lb_y
        return p_cmd 



    
    def updateSteps(self):
        return self.steps - 1



    def activateRobot(self):
        if self.ROS_mode is not True:
            self.RF_foot.activateROSFoot()
            self.LF_foot.activateROSFoot()
            self.RB_foot.activateROSFoot()
            self.LB_foot.activateROSFoot()
            self.ROS_mode = True
        else:
            print("Robot already acivate")
    
    def resetAllSteps(self):
        if self.excu_mode:
            self.steps = 0
            self.rf_x,self.lb_x,self.lf_x,self.rb_x = -0.0325, -0.0325, 0.0325, 0.0325
            self.u1_1, self.u2_1, self.v1_1, self.v2_1, self.y1_1, self.y2_1, self.o_1, self.gain_1, self.bias_1 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
            self.u1_2, self.u2_2, self.v1_2, self.v2_2, self.y1_2, self.y2_2, self.o_2, self.gain_2, self.bias_2 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
            self.u1_3, self.u2_3, self.v1_3, self.v2_3, self.y1_3, self.y2_3, self.o_3, self.gain_3, self.bias_3 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
            self.u1_4, self.u2_4, self.v1_4, self.v2_4, self.y1_4, self.y2_4, self.o_4, self.gain_4, self.bias_4 = 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
            self.resetAllData()
            print("reset success")
        
    

    
    def excuteCPG(self):
        if self.ROS_mode:
            self.CPGStep()
            self.o1_list.append(self.o_1)
            self.o2_list.append(self.o_2)
            self.f_b1_list.append(self.f_b1)
            self.f_b2_list.append(self.f_b2)
        else:
            print("Robot is not acivate")
    
    def excuFbCmd(self,fb_cmd,dt=0.015,fb_mode = True):
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
                    self.excuteCPG()
                    rospy.sleep(dt)
                    current_steps = self.updateSteps()
                    # self.storeJointData()
                else:
                    rpy_angle = self.RF_foot.ros_comunicator.updateRpyAngle()
                    distance = self.RF_foot.ros_comunicator.updateDistance()
                    yaw = rpy_angle[2]
                    offset = self.RF_foot.ros_comunicator.updateOffset()
                    self.offset_list.append(offset)
                    self.yaw_list.append(yaw)
                    self.distance_list.append(distance)
                    break
    
    def excuteCPGGait(self,fb_cmd,steps=30):
        if fb_cmd is not None:
            self.excuFbCmd(fb_cmd)
            for step in range(steps):
                fb_cmd = [0,0]
                self.excuFbCmd(fb_cmd)
        else:
            print("fail to excute gait")
    
    def plotTrajectory(self):
        if self.x_list is not None and self.z_list is not None:
            t_index = len(self.r_z_list)
            t_list = np.linspace(0,10,t_index)
            fig = plt.figure()
            ax = fig.subplots(1)
            # ax.plot(self.r_x_list,self.r_z_list,'r',label='RF LR toe')
            # ax.plot(self.l_x_list,self.l_z_list,'b',label='LF RR toe')
            ax.plot(t_list,self.r_z_list,'r',label='RF LR toe')
            ax.plot(t_list,self.l_z_list,'b',label='LF RR toe')
            plt.xlabel('x axis (m)')
            plt.ylabel('z axis (m)')
            plt.legend(loc='upper left')
            plt.show()
        else:
            print("Without Data to plot")
    
    def plotOffset(self):
        if not len(self.offset_list) == 0:
            offset_index = len(self.offset_list)
            output_index = len(self.o1_list)
            t_list = np.linspace(0,10,offset_index)
            t1_list = np.linspace(0,10,output_index)
            fig = plt.figure()
            ax = fig.subplots(1)
            # ax.plot(t_list,self.offset_list,'r')
            # ax.plot(t1_list,self.o1_list,'b')
            ax.plot(t1_list,self.o2_list,color='blue',label='CPG(2) output')
            ax.plot(t1_list,self.o1_list,color='red',label='CPG(1) output')
            plt.ylabel('CPG output (no unit)')
            plt.xlabel('Time(sec)')
            plt.legend(loc='upper left')
            plt.show()
        else:
            print("Without Data to plot")
    
    def plotContact(self):
        if len(self.rf_contact) != 0:
            index = len(self.rf_contact)
            t_list = np.linspace(0,self.numberOfSteps,index)
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(t_list,self.rf_contact,color='r',label='RF contact')
            ax.plot(t_list,self.lf_contact,color='b',label='LF contact')
            ax.plot(t_list,self.lb_contact,color='g',label='LB contact')
            ax.plot(t_list,self.rb_contact,color='y',label='RB contact')
            plt.ylabel('Force (N)')
            plt.xlabel('Steps')
            ax.legend(loc='upper left')
            plt.show()
    
    def plotCOG(self):
        if len(self.cog_x) != 0:
            # index = len(self.cog_x)
            # t_list = np.linspace(0,self.numberOfSteps,index)
            fig = plt.figure()
            y_lim = np.linspace(-1,1,5)
            z_lim = np.linspace(0,0.5,9)
            # ax = fig.subplots()
            ax = fig.gca(projection='3d')
            ax.plot(self.cog_x,self.cog_y,self.cog_z,color='Red',label='robot COG')
            # ax.plot(t_list,self.cog_x,color='r',label='COG (x)')
            # ax.plot(t_list,self.cog_y,color='b',label='COG (y)')
            # ax.plot(t_list,self.cog_z,color='g',label='COG (z)')
            # plt.ylabel('COG (cm)')
            # plt.xlabel('Steps')
            ax.legend(loc='upper left')
            ax.set_yticks(y_lim)
            ax.set_zticks(z_lim)
            ax.set_xlabel("X-position (m)")
            ax.set_ylabel("Y-position (m)")
            ax.set_zlabel("Z-position (m)")
            plt.show()
    


        


    def resetAllData(self):
        self.r_x_list, self.l_x_list, self.r_z_list, self.l_z_list = list(), list(), list(), list()
        self.offset_list, self.distance_list = list(), list()
        self.yaw_list = list()
        self.o1_list, self.o2_list = list(), list()
        self.cog_x, self.cog_y, self.cog_z = list(), list(), list()
        self.rf_contact, self.lf_contact, self.lb_contact, self.rb_contact = list(), list(), list(), list()
        self.f_b1_list, self.f_b2_list = list(), list()

    def test(self):
        for i in range(2):
            fb_cmd = [0,0]
            self.excuFbCmd(fb_cmd)
        for i in range(2):
            fb_cmd = [0,-0.1]
            self.excuFbCmd(fb_cmd)
    
    def saveData(self,excu_mode=False):
        if excu_mode:
            fileName = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            fileName = "./data/" + fileName
            cog_data = [self.cog_x, self.cog_y, self.cog_z]
            data = [self.r_z_list, self.l_z_list, self.o1_list, self.o2_list, self.offset_list, self.yaw_list, self.distance_list, cog_data, self.f_b1_list, self.f_b2_list]
            with open(fileName + ".aldata", 'wb+') as f:
                pickle.dump(data,f)











        
            

        


    




        








if __name__ == '__main__':
    ros = ROS.ROS_Comunication()
    cpg = CPG_controller()

    command_dict = {
        "act":cpg.activateRobot,
        "plotTraj":cpg.plotTrajectory,
        "plotOffset":cpg.plotOffset,
        "test":cpg.test,
        "plotData":cpg.plotJointData,
        "plotCont":cpg.plotContact,
        "plotCOG":cpg.plotCOG
    }

    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "cpg":
                mode = input("mode : ")
                if mode == "standard":
                    fb_cmd = [0,0,0,0]
                    cpg.excuteCPGGait(fb_cmd)
                    cpg.saveData(excu_mode=True)
                else:
                    fb_cmd = [0,0,0,0]
                    fb_cmd[0] = float(input("fb1: "))
                    fb_cmd[1] = float(input("fb2: "))
                    cpg.excuteCPGGait(fb_cmd)
            elif cmd == "exit":
                cpg.resetAllPos()
                ros.resetWorld()
                break
            elif cmd == "reset":
                cpg.resetAllData()
                cpg.resetAllSteps()
                cpg.resetAllPos()
                ros.resetWorld()
        except Exception as e:
            traceback.print_exc()
            break