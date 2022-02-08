#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
from std_msgs.msg import Float64
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ContactsState
import pickle
import numpy as np
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import time
import random
from math import cos,sin,atan2,sqrt,acos
import math
import gait as Gait
from std_srvs.srv import Empty
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
class Foot(object):

    def __init__(self,foot='LF'):
        self.ros_comunicator = ROS_Comunication()
        self.shoulder_joint = None
        self.upper_joint = None
        self.lower_joint = None
        self.foot = foot
        self.p_cmd = np.zeros(3)
        self.activate_mode = False
        self.l1 = 0
        self.l2 = 0.2 
        self.l3 = 0.2

    
    def activateROSFoot(self):
        self.ros_comunicator.checkROSConnection()
        if self.foot == 'LF':
            self.shoulder_joint = self.ros_comunicator.addROSJoint(12)
            self.upper_joint = self.ros_comunicator.addROSJoint(4)
            self.lower_joint = self.ros_comunicator.addROSJoint(5) 
            print("Left Back foot ready")
        elif self.foot == 'LB':
            self.shoulder_joint = self.ros_comunicator.addROSJoint(14)
            self.upper_joint = self.ros_comunicator.addROSJoint(8)
            self.lower_joint = self.ros_comunicator.addROSJoint(9)
            print("Right Back foot ready")
        elif self.foot == 'RB':
            self.shoulder_joint = self.ros_comunicator.addROSJoint(15)
            self.upper_joint = self.ros_comunicator.addROSJoint(10)
            self.lower_joint = self.ros_comunicator.addROSJoint(11)
            print("Right Front foot ready")
        elif self.foot == 'RF':
            self.shoulder_joint = self.ros_comunicator.addROSJoint(13)
            self.upper_joint = self.ros_comunicator.addROSJoint(6)
            self.lower_joint = self.ros_comunicator.addROSJoint(7)
            print("Left Front foot ready")
        self.activate_mode = True
        
    def sendAllCommand(self,p_cmd): 
        if self.activate_mode:
            self.shoulder_joint.moveToPosition(np.rad2deg(p_cmd[0]))
            self.upper_joint.moveToPosition(np.rad2deg(p_cmd[1]))
            self.lower_joint.moveToPosition(np.rad2deg(p_cmd[2]))
        else:
            print("foot is not activate")
    
    def sendSingalCommand(self,cmd,joint='shoulder'): 
        if self.activate_mode:
            if joint == 'shoulder':
                self.shoulder_joint.moveToPosition(cmd)
            if joint == 'hip':
                self.upper_joint.moveToPosition(cmd)
            if joint == 'knee':
                self.lower_joint.moveToPosition(cmd)
        else:
            print("ROS is not activate")
        
    def resetPos(self):
        if self.activate_mode:
            if self.foot == 'LF' or self.foot == 'RB':
                p_cmd = self.leg_ikine(0,0,0.35)
                self.sendAllCommand(p_cmd)
            else:
                p_cmd = self.leg_ikine(0,0,0.35)
                self.sendAllCommand(p_cmd)
        else:
            print("foot is not activate")
    
    def leg_ikine(self, x, y, z):
        theta1 = math.atan2(y, z) + math.atan2(self.l1, -(z**2 + y**2 - self.l1**2)**0.5)
        c1 = math.cos(theta1)    
        s1 = math.sin(theta1)
        c3 = (x**2 + y**2 + z**2 - self.l1**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        s3 = (1 - c3**2)**0.5
        theta3 = math.atan2(s3, c3)
        s2p = (self.l3 * s3) / ((y * s1 + x * c1)**2 + z**2)**0.5
        c2p = (1 - s2p**2)**0.5
        theta2p = -math.atan2(s2p, c2p)
        thetap = -math.atan2(x, -(y * s1 +z * c1))
        theta2 = theta2p - thetap
        theta1 = theta1 - math.pi
        self.p_cmd[0] = theta1
        self.p_cmd[1] = theta2
        self.p_cmd[2] = theta3
        if self.activate_mode:
            if self.foot == 'RB' or self.foot == 'RF': 
                # self.sendAllCommand(self.p_cmd)
                return self.p_cmd 
            if self.foot == 'LF' or self.foot == 'LB':
                # self.sendAllCommand(-self.p_cmd)
                return -self.p_cmd
    
    def getFootData(self):
        if self.activate_mode:
            pos_data = [0, 0, 0]
            pos_data[0] = self.shoulder_joint.getJointPosition()
            pos_data[1] = self.upper_joint.getJointPosition()
            pos_data[2] = self.lower_joint.getJointPosition()
            return pos_data
            

    

class ROS_Joint_Info(object):
    def __init__(self,angle,velocity,torque,DXL_ID):
        self.angle = angle
        self.velocity = velocity
        self.torque = torque
        self.DXL_ID = DXL_ID



class ROS_Comunication(object):
    def __init__(self):
        rospy.init_node('ros_pos_control', anonymous=True)
        self.joint_list = list()
        self.info_list = list()
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        rospy.Subscriber("/imu", Imu, self.imu_cb)
        rospy.Subscriber("/ground_truth/state", Odometry, self.odometry_cb)
        rospy.Subscriber("/cog/robot", PointStamped, self.cog_cb)
        # rospy.Subscriber("/RF_foot_contactsensor_state", ContactsState, self.rf_cb)
        # rospy.Subscriber("/LF_foot_contactsensor_state", ContactsState, self.lf_cb)
        # rospy.Subscriber("/RB_foot_contactsensor_state", ContactsState, self.rb_cb)
        # rospy.Subscriber("/LB_foot_contactsensor_state", ContactsState, self.lb_cb)
        self.rf_hip_joint_cmd = rospy.Publisher("/quadruped_robot/RF_hip_joint_position_controller/command", Float64, queue_size=1)
        self.rb_hip_joint_cmd = rospy.Publisher("/quadruped_robot/RB_hip_joint_position_controller/command", Float64, queue_size=1)
        self.lf_hip_joint_cmd = rospy.Publisher("/quadruped_robot/LF_hip_joint_position_controller/command", Float64, queue_size=1)
        self.lb_hip_joint_cmd = rospy.Publisher("/quadruped_robot/LB_hip_joint_position_controller/command", Float64, queue_size=1)
        self.rf_leg_joint_cmd = rospy.Publisher("/quadruped_robot/RF_leg_joint_position_controller/command", Float64, queue_size=1)
        self.rb_leg_joint_cmd = rospy.Publisher("/quadruped_robot/RB_leg_joint_position_controller/command", Float64, queue_size=1)
        self.lf_leg_joint_cmd = rospy.Publisher("/quadruped_robot/LF_leg_joint_position_controller/command", Float64, queue_size=1)
        self.lb_leg_joint_cmd = rospy.Publisher("/quadruped_robot/LB_leg_joint_position_controller/command", Float64, queue_size=1)
        self.rf_knee_joint_cmd = rospy.Publisher("/quadruped_robot/RF_knee_joint_position_controller/command", Float64, queue_size=1)
        self.rb_knee_joint_cmd = rospy.Publisher("/quadruped_robot/RB_knee_joint_position_controller/command", Float64, queue_size=1)
        self.lf_knee_joint_cmd = rospy.Publisher("/quadruped_robot/LF_knee_joint_position_controller/command", Float64, queue_size=1)
        self.lb_knee_joint_cmd = rospy.Publisher("/quadruped_robot/LB_knee_joint_position_controller/command", Float64, queue_size=1)
        self.MAX_DXL_ID = 15
        self.init_pos = [0,0,0]
        self.activate = True

    
    def deactivateConnection(self):
        self.activate = False

    
    def checkROSConnection(self):
        self.unpauseSim()
        rate = rospy.Rate(10)  
        while (self.rf_hip_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to shoulder_joint1_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("cmd Publisher Connected")
        while (self.rb_hip_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to shoulder_joint2_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lf_hip_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to shoulder_joint3_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lb_hip_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to shoulder_joint4_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.rf_leg_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint1_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.rb_leg_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint2_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lf_leg_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint3_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lb_leg_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint4_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.rf_knee_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint5_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.rb_knee_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint6_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lf_knee_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint7_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        while (self.lb_knee_joint_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logwarn("No susbribers to foot_joint8_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher Connected")
        rospy.logdebug("All Publishers READY")
    
    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed")

    def pauseSim(self):
        """
        pause gazebo system
        """
        #rospy.logwarn("PAUSING START")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        """
        unpause gazebo system
        """
        #rospy.logwarn("UNPAUSING START")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")
        

    def addROSJoint(self, joint_number=1):
        if joint_number <= self.MAX_DXL_ID:
            if joint_number not in [joint.DXL_ID for joint in self.joint_list]:
                joint = ROS_Joint(joint_number,self.rf_hip_joint_cmd,
                                               self.rb_hip_joint_cmd,
                                               self.lf_hip_joint_cmd,
                                               self.lb_hip_joint_cmd,
                                               self.rf_leg_joint_cmd,
                                               self.rb_leg_joint_cmd,
                                               self.lf_leg_joint_cmd,
                                               self.lb_leg_joint_cmd,
                                               self.rf_knee_joint_cmd,
                                               self.rb_knee_joint_cmd,
                                               self.lf_knee_joint_cmd,
                                               self.lb_knee_joint_cmd)
                self.joint_list.append(joint)
                print("joint {0} create complete".format(joint_number))
                return joint
            else:
                print("joint {0} already exist".format(joint_number))
                for joint in self.joint_list:
                    if joint.DXL_ID == joint_number:
                        return joint
    
        else:
            print("joint number out of range")

    def selfCheck(self):
        rpy_angle = self.updateRpyAngle()
        pos = self.updateDistance()
        roll = rpy_angle[0]
        pitch = rpy_angle[1]
        if abs(roll) < 0.5 or pos < -0.1 or abs(pitch) < 0.5:
            print("robot pose check success")
            return False
        else:
            print("pose check fail")
            return True
    
    def rf_cb(self,msg):
        self.rf_force_z = msg.states[0].total_wrench.force.z
    
    def rb_cb(self,msg):
        self.rb_force_z = msg.states[0].total_wrench.force.z
    
    def lf_cb(self,msg):
        self.lf_force_z = msg.states[0].total_wrench.force.z
    
    def lb_cb(self,msg):
        self.lb_force_z = msg.states[0].total_wrench.force.z
    

    def cog_cb(self,msg):
        self.robot_cog_x, self.robot_cog_y, self.robot_cog_z = msg.point.x, msg.point.y, msg.point.z
    
    def odometry_cb(self,msg):
        self.x_offset = msg.pose.pose.position.y
        self.move_distance = msg.pose.pose.position.x
        self.robot_z = msg.pose.pose.position.z
    
    def updateContactData(self):
        contact_list = [self.rf_force_z, self.lf_force_z, self.lb_force_z, self.rb_force_z]
        return contact_list
    
    def updateOffset(self,mode='m'):
        if mode == 'm':
            return self.x_offset
        elif mode == 'cm':
            return self.x_offset * 100
    
    def updateDistance(self):
        return self.move_distance 
    
    def updateLocation(self):
        location = [self.move_distance, self.x_offset, self.robot_z]
        return location
    
    def updateCOG(self,mode='world'):
        if mode == 'local':
            robot_cog = [self.robot_cog_x, self.robot_cog_y, self.robot_cog_z]
            return robot_cog
        elif mode == 'world':
            location = self.updateLocation()
            self.robot_cog_x += location[0]
            self.robot_cog_y += location[1]
            self.robot_cog_z += location[2]
            robot_cog = [self.robot_cog_x, self.robot_cog_y, self.robot_cog_z]
            return robot_cog

    
    
    def imu_cb(self,imu_data):
        # Read the quaternion of the robot IMU
        x = imu_data.orientation.x
        y = imu_data.orientation.y
        z = imu_data.orientation.z
        w = imu_data.orientation.w
    
        # Read the angular velocity of the robot IMU
        w_x = imu_data.angular_velocity.x
        w_y = imu_data.angular_velocity.y
        w_z = imu_data.angular_velocity.z
    
        # Read the linear acceleration of the robot IMU
        a_x = imu_data.linear_acceleration.x
        a_y = imu_data.linear_acceleration.y
        a_z = imu_data.linear_acceleration.z
    
        # Convert Quaternions to Euler-Angles
        self.rpy_angle = [0, 0, 0]
        self.rpy_angle[0] = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        self.rpy_angle[1] = math.asin(2 * (w * y - z * x))
        self.rpy_angle[2] = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    def updateRpyAngle(self,mode='rad'):
        if mode == 'rad':
            return self.rpy_angle
        elif mode == 'deg':
            return np.rad2deg(self.rpy_angle)
        

    def printInfo(self):
        for info in self.info_list:
            print([info.angle,info.velocity,info.torque,info.DXL_ID])
    
    def getROSInfo(self):
        if self.joint_list is not None:
            for joint in self.joint_list:
                joint.updateJointData()
                info =  ROS_Joint_Info(
                joint.getJointPosition(),
                joint.getJointVelocity(),
                joint.getJointTorque(),
                joint.DXL_ID
                )
                self.info_list.append(info)
        
    def sentROSCommand(self,p_cmd,joint_number,time_delay=0):
        if self.activate:
            self.checkROSConnection()
            if len(self.joint_list) == joint_number:
                #self.unpauseSim()
                for joint in self.joint_list:
                    joint.moveToPosition(p_cmd[joint.DXL_ID-1])
                rospy.sleep(time_delay)
                #self.pauseSim()
                #self.getROSInfo()
            else:
                print("joint is not enough for command")
        else:
            print("connnection not activate")
    
    def robotOffsetCheck(self):
        offset = self.updateOffset(mode='cm')
        if abs(offset) > 30:
            return True
        else:
            return False
    
    def robotYawCheck(self):
        yaw_angle = self.updateRpyAngle(mode='deg')
        if abs(yaw_angle) > 10:
            return True
        else:
            return False


        

    
class ROS_Joint(object):
    def __init__(self, DXL_ID,rf_hip,rb_hip,lf_hip,lb_hip,rf_leg,rb_leg,lf_leg,lb_leg,rf_knee,rb_knee,lf_knee,lb_knee):
        self.DXL_ID = DXL_ID
        self.__position = 0
        self.__velocity = 0
        self.__torque = 0
        self.joint_state = None
        self.odometry_data = None
        self.rf_hip_joint_cmd = rf_hip
        self.rb_hip_joint_cmd = rb_hip
        self.lf_hip_joint_cmd = lf_hip
        self.lb_hip_joint_cmd = lb_hip
        self.rf_leg_joint_cmd = rf_leg
        self.rb_leg_joint_cmd = rb_leg
        self.lf_leg_joint_cmd = lf_leg
        self.lb_leg_joint_cmd = lb_leg
        self.rf_knee_joint_cmd = rf_knee
        self.rb_knee_joint_cmd = rb_knee
        self.lf_knee_joint_cmd = lf_knee
        self.lb_knee_joint_cmd = lb_knee
        self.rate = rospy.Rate(10)
        rospy.Subscriber("/quadruped_robot/joint_states", JointState, self.callbackJointState)



    def moveToPosition(self,Angle):
        """Angle Unit as degree"""
        if Angle is not None:
            if self.DXL_ID == 13:
                self.rf_hip_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 15:
                self.rb_hip_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 12:
                self.lf_hip_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 14:
                self.lb_hip_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 4:
                self.lf_leg_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 5:
                self.lf_knee_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 6:
                self.rf_leg_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 7:
                self.rf_knee_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 8:
                self.lb_leg_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 9:
                self.lb_knee_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 10:
                self.rb_leg_joint_cmd.publish(np.deg2rad(Angle))
            elif self.DXL_ID == 11:
                self.rb_knee_joint_cmd.publish(np.deg2rad(Angle))
    def writeVelocity(self,velocity):
        if velocity is not None:
            if self.DXL_ID == 13:
                self.rf_hip_joint_cmd.publish(velocity)
            elif self.DXL_ID == 15:
                self.rb_hip_joint_cmd.publish(velocity)
            elif self.DXL_ID == 12:
                self.lf_hip_joint_cmd.publish(velocity)
            elif self.DXL_ID == 14:
                self.lb_hip_joint_cmd.publish(velocity)
            elif self.DXL_ID == 4:
                self.lf_leg_joint_cmd.publish(velocity)
            elif self.DXL_ID == 5:
                self.lf_knee_joint_cmd.publish(velocity)
            elif self.DXL_ID == 6:
                self.rf_leg_joint_cmd.publish(velocity)
            elif self.DXL_ID == 7:
                self.rf_knee_joint_cmd.publish(velocity)
            elif self.DXL_ID == 8:
                self.lf_leg_joint_cmd.publish(velocity)
            elif self.DXL_ID == 9:
                self.lf_knee_joint_cmd.publish(velocity)
            elif self.DXL_ID == 10:
                self.rb_leg_joint_cmd.publish(velocity)
            elif self.DXL_ID == 11:
                self.rb_knee_joint_cmd.publish(velocity)

    def callbackJointState(self,msg):
        self.joint_state = msg
    

            
    def updateJointData(self):
        self.__position = [self.joint_state.position[0], 
                           self.joint_state.position[1], 
                           self.joint_state.position[2], 
                           self.joint_state.position[3], 
                           self.joint_state.position[4], 
                           self.joint_state.position[5], 
                           self.joint_state.position[6], 
                           self.joint_state.position[7], 
                           self.joint_state.position[8], 
                           self.joint_state.position[9], 
                           self.joint_state.position[10], 
                           self.joint_state.position[11]
                           ]
        self.__velocity = [self.joint_state.velocity[0], 
                           self.joint_state.velocity[1], 
                           self.joint_state.velocity[2], 
                           self.joint_state.velocity[3], 
                           self.joint_state.velocity[4], 
                           self.joint_state.velocity[5], 
                           self.joint_state.velocity[6], 
                           self.joint_state.velocity[7], 
                           self.joint_state.velocity[8], 
                           self.joint_state.velocity[9], 
                           self.joint_state.velocity[10], 
                           self.joint_state.velocity[11]
                           ]

    def getJointPosition(self):
        """Unit as degree"""
        self.updateJointData()
        if self.DXL_ID == 14: 
            return np.rad2deg(self.__position[0])
        elif self.DXL_ID == 9:
            return np.rad2deg(self.__position[1])
        elif self.DXL_ID == 8:
            return np.rad2deg(self.__position[2])
        elif self.DXL_ID == 12:
            return np.rad2deg(self.__position[3])
        elif self.DXL_ID == 5:
            return np.rad2deg(self.__position[4])
        elif self.DXL_ID == 4:
            return np.rad2deg(self.__position[5])
        elif self.DXL_ID == 15:
            return np.rad2deg(self.__position[6])
        elif self.DXL_ID == 11:
            return np.rad2deg(self.__position[7])
        elif self.DXL_ID == 10:
            return np.rad2deg(self.__position[8])
        elif self.DXL_ID == 13:
            return np.rad2deg(self.__position[9])
        elif self.DXL_ID == 7:
            return np.rad2deg(self.__position[10])
        elif self.DXL_ID == 6:
            return np.rad2deg(self.__position[11])
    def getJontVelocity(self):
        """Unit as rad/s"""
        if self.DXL_ID == 13:
            return self.__velocity[0]
        elif self.DXL_ID == 15:
            return self.__velocity[1]
        elif self.DXL_ID == 12:
            return self.__velocity[2]
        elif self.DXL_ID == 14:
            return self.__velocity[3]
        elif self.DXL_ID == 4:
            return self.__velocity[4]
        elif self.DXL_ID == 5:
            return self.__velocity[5]
        elif self.DXL_ID == 6:
            return self.__velocity[6]
        elif self.DXL_ID == 7:
            return self.__velocity[7]
        elif self.DXL_ID == 8:
            return self.__velocity[8]
        elif self.DXL_ID == 9:
            return self.__velocity[9]
        elif self.DXL_ID == 10:
            return self.__velocity[10]
        elif self.DXL_ID == 11:
            return self.__velocity[11]
    def getJointTorque(self):
        """Unit as N*m"""
        pass

def ROS_test():
    foot_front_left_ros = Foot('LF')
    foot_front_right_ros = Foot('RF')
    foot_rear_left_ros = Foot('LB')
    foot_rear_right_ros = Foot('RB')
    foot_front_left_ros.activateROSFoot()
    foot_front_right_ros.activateROSFoot()
    foot_rear_left_ros.activateROSFoot()
    foot_rear_right_ros.activateROSFoot()
    # foot_front_left_ros.leg_ikine(0,-0.1,0.3)
    # foot_front_right_ros.leg_ikine(0,-0.1,0.3)
    # foot_rear_left_ros.resetPos()
    # foot_rear_right_ros.resetPos()

    




if __name__ == '__main__':
    try:
        ROS_test()
    except rospy.ROSInterruptException:
        pass