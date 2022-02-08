import math, datetime
import numpy as np
import robot_foot_model as RFModel
import sys, os
import pickle
import time
import gait
import ROS_test_controller as ROS





def Ang2Seg(Ang_Value):
    seg_ang = int(Ang_Value/(360/50))
    return seg_ang

def statelize(rl_foot, target):
    err_ang_flag, err_rad_seg = CurrentTargetErr(target, rl_foot.EndPoint)
    rl_foot.Upper_Motor.updateROSJointData()
    rl_foot.Middle_Motor.updateROSJointData()
    rl_foot.Lower_Motor.updateROSJointData()
    print(rl_foot.Upper_Motor.GetPosition("deg"))
    print(rl_foot.Middle_Motor.GetPosition("deg"))
    print(rl_foot.Lower_Motor.GetPosition("deg"))
    print("joint1 vel:{0}".format(rl_foot.Upper_Motor.GetVelocity("deg")))
    print("joint2 vel:{0}".format(rl_foot.Middle_Motor.GetVelocity("deg")))
    print("joint3 vel:{0}".format(rl_foot.Lower_Motor.GetVelocity("deg")))
    m1_sAng = float(Ang2Seg(rl_foot.Upper_Motor.GetPosition("deg")))
    m2_sAng = float(Ang2Seg(rl_foot.Middle_Motor.GetPosition("deg")))
    m3_sAng = float(Ang2Seg(rl_foot.Lower_Motor.GetPosition("deg")))
    state = [err_ang_flag, err_rad_seg, m1_sAng, m2_sAng, m3_sAng]
    print(state)
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
