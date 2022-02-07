import numpy as np
import time
import threading
import socket
import traceback

import quadruped_robot_model
import quadruped_robot_gait
import log
import gait


GROUND_LEVEL = -37


class QuadrupedRobotController(quadruped_robot_model.QuadrupedRobot):
    def __init__(self, log_level="info", log_file_level="debug"):
        self.log_level = log_level
        self.log_file_level = log_file_level
        self.log = log.LogHandler(self.__class__.__name__, __name__, self.log_level, self.log_file_level)
        self.gait_phase = 0.5 # 50%
        self.gait = None
        super().__init__(self.log_level, self.log_file_level)


    def executeGaitCycle(self, gait, cycle=1):
        gait_count = 0
        self.slowlyPosition(
            lf_cmd=gait.start_point_left_front,
            rf_cmd=gait.start_point_right_front,
            lh_cmd=gait.start_point_left_hind,
            rh_cmd=gait.start_point_right_hind,
            lf_pos=gait.start_point_hip_lf,
            rf_pos=gait.start_point_hip_rf,
            lh_pos=gait.start_point_hip_lh,
            rh_pos=gait.start_point_hip_rh,
        )
        gait_count = 0
        if gait.bool_has_init_gait:
            while gait_count != gait.number_of_init_point:
                fl_cmd, fr_cmd, rl_cmd, rr_cmd, fl_pos, fr_pos, rl_pos, rr_pos, cycle_passed = \
                    gait.initGaitLookUp(gait_count)
                self.executeIK(
                    fl_cmd, fr_cmd, rl_cmd, rr_cmd, fl_pos, fr_pos, rl_pos, rr_pos
                )
                gait_count+=1


        i = 0
        while i < cycle+1:
            lf_cmd, rf_cmd, lh_cmd, rh_cmd, lf_pos, rf_pos, lh_pos, rh_pos, cycle_passed = \
                gait.gaitLookUp(gait_count)
            self.executeIK(
                lf_cmd, rf_cmd, lh_cmd, rh_cmd, lf_pos, rf_pos, lh_pos, rh_pos, gait.dt
            )
            if cycle_passed:
                i += 1
            gait_count+=1

        gait_count = 0
        if gait.bool_has_end_gait:
            while gait_count != gait.number_of_end_point:
                fl_cmd, fr_cmd, rl_cmd, rr_cmd, fl_pos, fr_pos, rl_pos, rr_pos, cycle_passed = \
                    gait.endGaitLookUp(gait_count)
                self.executeIK(
                    fl_cmd, fr_cmd, rl_cmd, rr_cmd, fl_pos, fr_pos, rl_pos, rr_pos
                )
                gait_count+=1

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

    def standupFromGround(self):
        self.slowlySetAllMotorPositionIK(8, -16, 270, resolution=40)
        self.slowlyPosition(
            lf_cmd=[0, -20],
            rf_cmd=[0, -20],
            lh_cmd=[0, -30],
            rh_cmd=[0, -30],
        )
        self.slowlySetAllMotorPositionIK(-3.5, -35, 270)


    def trotStandingInPlace(self, ground_level=-37, step_long=3, height=5,
        phase=120, resolution=16, dt=0.015, cycle=20, origin=-3):
        gait = quadruped_robot_gait.TrotStandingInPlaceGait(
            ground_level=ground_level,
            step_long=step_long,
            resolution=resolution,
            height=height,
            phase=phase,
            origin=origin,
            dt=dt
        )

        gait.changeDt(0.05)
        self.executeGaitCycle(gait=gait, cycle=cycle)    

    def trotOutwardStandingInPlace(self, phase=120, resolution=6, cycle=20):
        gait = quadruped_robot_gait.TrotOutwardStandingInPlaceGait(phase=phase, hip=[280, 280, 280, 280])
        gait.changeDt(0.05)
        self.executeGaitCycle(gait=gait, cycle=cycle)    


        

if __name__ == "__main__":
    device = "com3"
    robot = QuadrupedRobotController()
    robot.activateDXLConnection(device)
    robot.activateAllRealMotor()
    robot.armAllMotor()
    robot.standupFromGround()

    command_dict = {
        "standup":robot.standupFromGround,
        "sitdown":robot.sitdownFromStand,
        "trot":robot.trotStandingInPlace,
        "trot2":robot.trotOutwardStandingInPlace,

    }


    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                break
        except Exception as e:
            traceback.print_exc()
            break
    
    robot.sitdownFromStand()
    robot.disarmAllMotor()
    robot.deactivateDXLConnection()