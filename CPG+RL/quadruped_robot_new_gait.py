import numpy as np
import gait
import sys
import log

#============== Abbreviation ===========
# lf : left front
# rf : right front
# lh : left hip
# rh : right hip

# Robot geometry calculation
ROBOT_WIDTH = 26.2
ROBOT_HALF_WIDTH = ROBOT_WIDTH/2
ROBOT_LENGTH = 36.4
ROBOT_HALF_LENGTH = ROBOT_LENGTH/2
ROBOT_R = (ROBOT_HALF_WIDTH**2 + ROBOT_HALF_LENGTH**2)**0.5
DIAGONAL_ANGLE_W_RAD = np.arccos(ROBOT_HALF_WIDTH/ROBOT_R)
DIAGONAL_ANGLE_L_RAD = np.arccos(ROBOT_HALF_LENGTH/ROBOT_R)

class Gait():

    def __init__(self):
        self.number_of_gait_point = 1
        self.number_of_init_point = 1
        self.number_of_end_point = 1
        self.bool_has_init_gait = False
        self.bool_has_end_gait = False
        self.gait_end_lf = np.array([[0., -30.]])
        self.gait_end_rh = np.array([[0., -30.]])
        self.gait_end_rf = np.array([[0., -30.]])
        self.gait_end_lh = np.array([[0., -30.]])
        self.gait_end_hip_lf = np.array([280])
        self.gait_end_hip_rf = np.array([280])
        self.gait_end_hip_lh = np.array([280])
        self.gait_end_hip_rh = np.array([280])
        self.gait_init_lf = np.array([[0., -30.]])
        self.gait_init_lh = np.array([[0., -30.]])
        self.gait_init_rf = np.array([[0., -30.]])
        self.gait_init_rh = np.array([[0., -30.]])
        self.gait_init_hip_lf = np.array([280])
        self.gait_init_hip_rf = np.array([280])
        self.gait_init_hip_lh = np.array([280])
        self.gait_init_hip_rh = np.array([280])

        self.gait_lf = np.array([[0., -30.]])
        self.gait_rf = np.array([[0., -30.]])
        self.gait_lh = np.array([[0., -30.]])
        self.gait_rh = np.array([[0., -30.]])
        self.gait_hip_lf = np.array([280])
        self.gait_hip_rf = np.array([280])
        self.gait_hip_lh = np.array([280])
        self.gait_hip_rh = np.array([280])
        self.dt = 0.1 #0.1
        self.start_point_lf = self.gait_lf[0]
        self.start_point_rf = self.gait_rf[0]
        self.start_point_lh = self.gait_lh[0]
        self.start_point_rh = self.gait_rh[0]
        self.start_point_hip_lf = self.gait_hip_lf[0]
        self.start_point_hip_rf = self.gait_hip_rf[0]
        self.start_point_hip_lh = self.gait_hip_lh[0]
        self.start_point_hip_rh = self.gait_hip_rh[0]
        self.gait_name = None
        self.whole_gait_time = len(self.gait_lf) * self.dt
        self.current_index = 0
        self.current_target = [0, -35]
        self.ground_info = None
        self.obstacle_info = None
        #Foot Moving Speed, should be calculated!!!
        self.Ground_Step = 0
        self.gait_on_ground = np.array([True])
        self.target_on_ground = False
        self.current_ground_angle = 0

    def changeDt(self, dt):
        self.dt = dt

    def endGaitLookUp(self, index):
        self.current_index = index % self.number_of_end_point
        self.current_target_lf = (
            self.gait_end_lf[self.current_index, 0],
            self.gait_end_lf[self.current_index, 1]
        )
        self.current_target_rf = (
            self.gait_end_rf[self.current_index, 0],
            self.gait_end_rf[self.current_index, 1]
        )
        self.current_target_lh = (
            self.gait_end_lh[self.current_index, 0],
            self.gait_end_lh[self.current_index, 1]
        )
        self.current_target_rh = (
            self.gait_end_rh[self.current_index, 0],
            self.gait_end_rh[self.current_index, 1]
        )
        self.hip_target_lf = self.gait_end_hip_lf[self.current_index]
        self.hip_target_rf = self.gait_end_hip_rf[self.current_index]
        self.hip_target_lh = self.gait_end_hip_lh[self.current_index]
        self.hip_target_rh = self.gait_end_hip_rh[self.current_index]
        if self.ground_info is not None:self.target_on_ground = self.gait_on_ground[self.current_index]

        if self.ground_info is not None and self.target_on_ground:
            ground_index = np.argmin(np.hypot(
                self.ground_info.ground_x - self.current_target[0],
                self.ground_info.ground_y - self.current_target[1]
            ))
            self.current_ground_angle = self.ground_info.ground_angle_rad[ground_index]

        if self.current_index == 0:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    True]
        else:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    False]

    def generateGate(self, **kwargs):
        print("Gait not defined")
        sys.exit()

    def gaitLookUp(self, index, checking=False):
        """return (x(float) ,y(float) ,new_cycle(bool))"""
        self.current_index = index % self.number_of_gait_point
        self.current_target_lf = (
            self.gait_lf[self.current_index, 0],
            self.gait_lf[self.current_index, 1]
        )
        self.current_target_rf = (
            self.gait_rf[self.current_index, 0],
            self.gait_rf[self.current_index, 1]
        )
        self.current_target_lh = (
            self.gait_lh[self.current_index, 0],
            self.gait_lh[self.current_index, 1]
        )
        self.current_target_rh = (
            self.gait_rh[self.current_index, 0],
            self.gait_rh[self.current_index, 1]
        )
        self.hip_target_lf = self.gait_hip_lf[self.current_index]
        self.hip_target_rf = self.gait_hip_rf[self.current_index]
        self.hip_target_lh = self.gait_hip_lh[self.current_index]
        self.hip_target_rh = self.gait_hip_rh[self.current_index]
        if self.ground_info is not None:self.target_on_ground = self.gait_on_ground[self.current_index]

        if self.ground_info is not None and self.target_on_ground:
            ground_index = np.argmin(np.hypot(
                self.ground_info.ground_x - self.current_target[0],
                self.ground_info.ground_y - self.current_target[1]
            ))
            self.current_ground_angle = self.ground_info.ground_angle_rad[ground_index]

        if self.current_index == 0:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    True]
        else:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    False]

    def initGaitLookUp(self, index):
        self.current_index = index % self.number_of_init_point
        self.current_target_lf = (
            self.gait_init_lf[self.current_index, 0],
            self.gait_init_lf[self.current_index, 1]
        )
        self.current_target_rf = (
            self.gait_init_rf[self.current_index, 0],
            self.gait_init_rf[self.current_index, 1]
        )
        self.current_target_lh = (
            self.gait_init_lh[self.current_index, 0],
            self.gait_init_lh[self.current_index, 1]
        )
        self.current_target_rh = (
            self.gait_init_rh[self.current_index, 0],
            self.gait_init_rh[self.current_index, 1]
        )
        self.hip_target_lf = self.gait_init_hip_lf[self.current_index]
        self.hip_target_rf = self.gait_init_hip_rf[self.current_index]
        self.hip_target_lh = self.gait_init_hip_lh[self.current_index]
        self.hip_target_rh = self.gait_init_hip_rh[self.current_index]
        if self.ground_info is not None:self.target_on_ground = self.gait_on_ground[self.current_index]

        if self.ground_info is not None and self.target_on_ground:
            ground_index = np.argmin(np.hypot(
                self.ground_info.ground_x - self.current_target[0],
                self.ground_info.ground_y - self.current_target[1]
            ))
            self.current_ground_angle = self.ground_info.ground_angle_rad[ground_index]

        if self.current_index == 0:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    True]
        else:
            return [self.current_target_lf, self.current_target_rf, self.current_target_lh, self.current_target_rh, 
                self.hip_target_lf, self.hip_target_rf, self.hip_target_lh, self.hip_target_rh,
                    False]

    def generateVelocityLog(self):
        current_x = self.gait_rf[:, 0]
        next_x = np.roll(current_x, -1)
        delta_x = next_x - current_x
        self.velocity_log = -delta_x / self.dt
    
    def getVelocity(self):
        if hasattr(self, "velocity_log"):
            return self.velocity_log[self.current_index]
        else:
            return 0

    def renewBasicInfo(self):
        self.number_of_gait_point = len(self.gait_lf)
        self.number_of_end_point = len(self.gait_end_lf)
        self.number_of_init_point = len(self.gait_init_lf)
        if self.bool_has_init_gait:
            self.start_point_lf = self.gait_init_lf[0]
            self.start_point_rf = self.gait_init_rf[0]
            self.start_point_lh = self.gait_init_lh[0]
            self.start_point_rh = self.gait_init_rh[0]
            self.start_point_hip_lf = self.gait_init_hip_lf[0]
            self.start_point_hip_rf = self.gait_init_hip_rf[0]
            self.start_point_hip_lh = self.gait_init_hip_lh[0]
            self.start_point_hip_rh = self.gait_init_hip_rh[0]
        else:
            self.start_point_lf = self.gait_lf[0]
            self.start_point_rf = self.gait_rf[0]
            self.start_point_lh = self.gait_lh[0]
            self.start_point_rh = self.gait_rh[0]
            self.start_point_hip_lf = self.gait_hip_lf[0]
            self.start_point_hip_rf = self.gait_hip_rf[0]
            self.start_point_hip_lh = self.gait_hip_lh[0]
            self.start_point_hip_rh = self.gait_hip_rh[0]

class TrotStandingInPlaceGait(Gait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.generateGate()

    def generateGate(self):
        initial_x = self.origin
        initial_y = self.ground_level
        holder_x = np.zeros(round(self.resolution*self.phase/100)) + initial_x
        holder_y = np.zeros(round(self.resolution*self.phase/100)) + initial_y
        updown_x = np.zeros(self.resolution) + initial_x
        updown_y = np.append(
            np.linspace(initial_y, initial_y + self.height, round(self.resolution/2)),
            np.linspace(initial_y + self.height, initial_y, round(self.resolution/2))
        )
        self.gait_lf = np.stack(
            (np.append(updown_x, holder_x), np.append(updown_y, holder_y)),
            axis=-1
        )
        self.gait_rf = np.stack(
            (np.append(holder_x, updown_x), np.append(holder_y, updown_y)),
            axis=-1
        )
        self.gait_lh = np.stack(
            (np.append(holder_x, updown_x), np.append(holder_y, updown_y)),
            axis=-1
        )
        self.gait_rh = np.stack(
            (np.append(updown_x, holder_x), np.append(updown_y, holder_y)),
            axis=-1
        )
        self.gait_hip_lf = np.zeros(len(self.gait_lf)) + self.hip[0]
        self.gait_hip_rf = np.zeros(len(self.gait_rf)) + self.hip[1]
        self.gait_hip_lh = np.zeros(len(self.gait_lh)) + self.hip[2]
        self.gait_hip_rh = np.zeros(len(self.gait_rh)) + self.hip[3]
        self.renewBasicInfo()


class TrotOutwardStandingInPlaceGait(Gait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [280, 280, 280, 280]
        self.generateGate()

    def generateGate(self):
        initial_x = self.origin + self.step_long
        initial_y = self.ground_level
        holder_x = np.zeros(round(self.resolution*self.phase/100)) + initial_x
        holder_y = np.zeros(round(self.resolution*self.phase/100)) + initial_y
        updown_x = np.append(
            np.linspace(initial_x, 0, round(self.resolution/2)),
            np.linspace(0, initial_x, round(self.resolution/2))
        )
        updown_y = np.append(
            np.linspace(initial_y, initial_y + self.height, round(self.resolution/2)),
            np.linspace(initial_y + self.height, initial_y, round(self.resolution/2))
        )
        self.gait_lf = np.stack(
            (np.append(updown_x, holder_x), np.append(updown_y, holder_y)),
            axis=-1
        )
        self.gait_rf = np.stack(
            (np.append(holder_x, updown_x), np.append(holder_y, updown_y)),
            axis=-1
        )
        self.gait_lh = np.stack(
            (np.append(-holder_x, -updown_x), np.append(holder_y, updown_y)),
            axis=-1
        )
        self.gait_rh = np.stack(
            (np.append(-updown_x, -holder_x), np.append(updown_y, holder_y)),
            axis=-1
        )
        self.gait_hip_lf = np.zeros(len(self.gait_lf)) + self.hip[0]
        self.gait_hip_rf = np.zeros(len(self.gait_rf)) + self.hip[1]
        self.gait_hip_lh = np.zeros(len(self.gait_lh)) + self.hip[2]
        self.gait_hip_rh = np.zeros(len(self.gait_rh)) + self.hip[3]
        self.renewBasicInfo()

class TrotForwardGait(Gait):
    def __init__(self, ground_level=-37, step_long=5, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.forward_flag = True
        self.generateGate()

    def endGait(self):
        initial_lf_rh_x = self.gait_init_lf[-1][0]
        initial_rf_lh_x = self.gait_init_rf[-1][0]
        initial_y = self.ground_level

        # lf_rh_x = np.linspace(initial_lf_rh_x, initial_lf_rh_x-(0.5*self.step_long), round(self.resolution))
        lf_rh_x = np.linspace(initial_lf_rh_x, self.origin, round(self.resolution))
        lf_rh_y = np.zeros(round(self.resolution)) + initial_y

        # rf_lh_x = np.linspace(initial_rf_lh_x, initial_rf_lh_x+(0.5*self.step_long), round(self.resolution))
        rf_lh_x = np.linspace(initial_rf_lh_x, self.origin, round(self.resolution))
        rf_lh_y = np.append(
                np.linspace(initial_y, initial_y+self.height, round(self.resolution/2)),
                np.linspace(initial_y+self.height, initial_y, round(self.resolution/2))
            )
        self.gait_end_lf = np.stack((lf_rh_x, lf_rh_y), axis=-1)
        self.gait_end_rh = np.stack((lf_rh_x, lf_rh_y), axis=-1)
        self.gait_end_rf = np.stack((rf_lh_x, rf_lh_y), axis=-1)
        self.gait_end_lh = np.stack((rf_lh_x, rf_lh_y), axis=-1)
        self.gait_end_hip_lf = np.zeros(len(self.gait_end_lf)) + self.hip[0]
        self.gait_end_hip_rf = np.zeros(len(self.gait_end_lf)) + self.hip[1]
        self.gait_end_hip_lh = np.zeros(len(self.gait_end_lf)) + self.hip[2]
        self.gait_end_hip_rh = np.zeros(len(self.gait_end_lf)) + self.hip[3]
        self.bool_has_end_gait = True

    def generateGate(self):
        if not self.forward_flag: self.step_long = -self.step_long
        self.hold_times = round(self.resolution*0)
        self.initialGait()
        self.normalGait()
        self.endGait()
        self.renewBasicInfo()


    def initialGait(self):
        initial_x = self.origin
        initial_y = self.ground_level
        initial_lf_rh_x_1 = np.zeros(round(self.resolution/2)) + initial_x
        initial_lf_rh_x_2 = np.linspace(initial_x, self.origin + self.step_long*0.5, round(self.resolution/2))
        initial_lf_rh_y_1 = np.linspace(self.ground_level, self.ground_level+self.height, round(self.resolution/2))
        initial_lf_rh_y_2 = np.linspace(self.ground_level+self.height, self.ground_level , round(self.resolution/2))
        lf_rh_hold_x = np.zeros(self.hold_times) + initial_lf_rh_x_2[-1]
        lf_rh_hold_y = np.zeros(self.hold_times) + initial_lf_rh_y_2[-1]

        initial_lf_rh_x = np.concatenate((initial_lf_rh_x_1, initial_lf_rh_x_2, lf_rh_hold_x))
        initial_lf_rh_y = np.concatenate((initial_lf_rh_y_1, initial_lf_rh_y_2, lf_rh_hold_y))

        initial_rf_lh_x_1 = np.linspace(initial_x, self.origin - self.step_long*0.5, round(self.resolution))
        initial_rf_lh_y_1 = np.zeros(round(self.resolution)) + initial_y
        rf_lh_hold_x = np.zeros(self.hold_times) + initial_rf_lh_x_1[-1]
        rf_lh_hold_y = np.zeros(self.hold_times) + initial_rf_lh_y_1[-1]

        initial_rf_lh_x = np.concatenate((initial_rf_lh_x_1, rf_lh_hold_x))
        initial_rf_lh_y = np.concatenate((initial_rf_lh_y_1, rf_lh_hold_y))


        self.gait_init_lf = np.stack((initial_lf_rh_x, initial_lf_rh_y), axis=-1)
        self.gait_init_rh = np.stack((initial_lf_rh_x, initial_lf_rh_y), axis=-1)
        self.gait_init_rf = np.stack((initial_rf_lh_x, initial_rf_lh_y), axis=-1)
        self.gait_init_lh = np.stack((initial_rf_lh_x, initial_rf_lh_y), axis=-1)
        self.gait_init_hip_lf = np.zeros(len(self.gait_init_lf) + self.hold_times) + self.hip[0]
        self.gait_init_hip_rf = np.zeros(len(self.gait_init_lf) + self.hold_times) + self.hip[1]
        self.gait_init_hip_lh = np.zeros(len(self.gait_init_lf) + self.hold_times) + self.hip[2]
        self.gait_init_hip_rh = np.zeros(len(self.gait_init_lf) + self.hold_times) + self.hip[3]
        self.bool_has_init_gait = True

    def normalGait(self):
        initial_lf_rh_x = self.gait_init_lf[-1][0]
        initial_rf_lh_x = self.gait_init_rf[-1][0]
        initial_y = self.ground_level

        lf_rh_x = np.concatenate((
            np.linspace(initial_lf_rh_x, initial_lf_rh_x-self.step_long, round(self.resolution)),
            np.zeros(self.hold_times) + (initial_lf_rh_x-self.step_long),
            np.linspace(initial_lf_rh_x - self.step_long, initial_lf_rh_x, round(self.resolution)),
            np.zeros(self.hold_times) + initial_lf_rh_x
        ))
        lf_rh_y = np.concatenate((
            np.zeros(round(self.resolution)) + initial_y,
            np.zeros(self.hold_times) + initial_y,
            np.linspace(initial_y, initial_y+self.height, round(self.resolution/2)),
            np.linspace(initial_y+self.height, initial_y, round(self.resolution/2)),
            np.zeros(self.hold_times) + initial_y
        ))

        rf_lh_x = np.concatenate((
            np.linspace(initial_rf_lh_x, initial_rf_lh_x + self.step_long, round(self.resolution)),
            np.zeros(self.hold_times) + initial_rf_lh_x + self.step_long,
            np.linspace(initial_rf_lh_x + self.step_long, initial_rf_lh_x, round(self.resolution)),
            np.zeros(self.hold_times) + initial_rf_lh_x,
        ))
        rf_lh_y = np.concatenate((
            np.linspace(initial_y, initial_y + self.height, round(self.resolution/2)),
            np.linspace(initial_y+self.height, initial_y, round(self.resolution/2)),
            np.zeros(self.hold_times) + initial_y,
            np.zeros(round(self.resolution)) + initial_y,
            np.zeros(self.hold_times) + initial_y,

        ))
        self.gait_lf = np.stack((lf_rh_x, lf_rh_y), axis=-1)
        self.gait_rh = np.stack((lf_rh_x, lf_rh_y), axis=-1)
        self.gait_rf = np.stack((rf_lh_x, rf_lh_y), axis=-1)
        self.gait_lh = np.stack((rf_lh_x, rf_lh_y), axis=-1)
        self.gait_hip_lf = np.zeros(len(self.gait_lf)) + self.hip[0]
        self.gait_hip_rf = np.zeros(len(self.gait_lf)) + self.hip[1]
        self.gait_hip_lh = np.zeros(len(self.gait_lf)) + self.hip[2]
        self.gait_hip_rh = np.zeros(len(self.gait_lf)) + self.hip[3]



class TrotBackwardGait(TrotForwardGait):
    def __init__(self, ground_level=-37, step_long=5, resolution=20, **kwargs):
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        super().__init__(
            ground_level = self.ground_level,
            step_long=self.step_long,
            resolution=self.resolution,
            origin=self.origin,
            height=self.height,
            cycle=self.cycle,
            phase=self.phase,
            dt=self.dt,
            hip=self.hip,
        )
        self.forward_flag = False
        self.generateGate()



class TrotRotateRightGait(Gait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.turn_right = True
        self.generateGate()

    def generateGate(self):
        if not self.turn_right: self.step_long = -self.step_long
        hold_times = round(self.resolution/2)
        angle_R_step_rad = np.arccos(self.step_long/2/ROBOT_R)
        angle_L_step_rad = np.pi - angle_R_step_rad - DIAGONAL_ANGLE_L_RAD
        angle_hip_lf_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[0])))))
        angle_hip_rf_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[1])))))
        angle_hip_lh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[2])))))
        angle_hip_rh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[3])))))

        lf_x_1 = np.linspace(self.origin, self.origin + self.step_long*np.cos(angle_L_step_rad), self.resolution)
        lf_hold_x_1 = np.zeros(hold_times) + lf_x_1[-1]
        lf_x_2 = np.linspace(self.origin + self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        lf_hold_x_2 = np.zeros(hold_times) + lf_x_2[-1]

        rf_x_1 = np.linspace(self.origin, self.origin + self.step_long*np.cos(angle_L_step_rad), self.resolution)
        rf_hold_x_1 = np.zeros(hold_times) + rf_x_1[-1]
        rf_x_2 = np.linspace(self.origin + self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        rf_hold_x_2 = np.zeros(hold_times) + rf_x_2[-1]

        lh_x_1 = np.linspace(self.origin, self.origin - self.step_long*np.cos(angle_L_step_rad), self.resolution)
        lh_hold_x_1 = np.zeros(hold_times) + lh_x_1[-1]
        lh_x_2 = np.linspace(self.origin - self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        lh_hold_x_2 = np.zeros(hold_times) + lh_x_2[-1]

        rh_x_1 = np.linspace(self.origin, self.origin - self.step_long*np.cos(angle_L_step_rad), self.resolution)
        rh_hold_x_1 = np.zeros(hold_times) + rh_x_1[-1]
        rh_x_2 = np.linspace(self.origin - self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        rh_hold_x_2 = np.zeros(hold_times) + rh_x_2[-1]

        lf_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        lf_y_down = np.linspace(self.ground_level + self.height, -(self.ground_level**2 + self.step_long**2)**0.5 ,round(self.resolution/2))
        lf_hold_y_1 = np.zeros(hold_times) + lf_y_down[-1]
        lf_y_back = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level ,self.resolution)
        lf_hold_y_2 = np.zeros(hold_times) + lf_y_back[-1]

        rh_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        rh_y_down = np.linspace(self.ground_level + self.height, -(self.ground_level**2 + self.step_long**2)**0.5 ,round(self.resolution/2))
        rh_hold_y_1 = np.zeros(hold_times) + rh_y_down[-1]
        rh_y_back = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level ,self.resolution)
        rh_hold_y_2 = np.zeros(hold_times) + lf_y_back[-1]

        rf_y_hold = np.zeros(self.resolution) + self.ground_level
        rf_hold_y_2 = np.zeros(hold_times) + rf_y_hold[-1]
        rf_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        rf_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        rf_hold_y_3 = np.zeros(hold_times) + rf_y_down[-1]
        
        lh_y_hold = np.zeros(self.resolution) + self.ground_level
        lh_hold_y_2 = np.zeros(hold_times) + lh_y_hold[-1]
        lh_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        lh_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        lh_hold_y_3 = np.zeros(hold_times) + lh_y_down[-1]

        lf_hip_go = np.linspace(self.hip[0], self.hip[0] - angle_hip_lf_deg, self.resolution)
        lf_hip_hold_1 = np.zeros(hold_times) + lf_hip_go[-1]
        lf_hip_back = np.linspace(self.hip[0] - angle_hip_lf_deg, self.hip[0], self.resolution)
        lf_hip_hold_2 = np.zeros(hold_times) + lf_hip_back[-1]

        rh_hip_go = np.linspace(self.hip[0], self.hip[0] - angle_hip_rh_deg, self.resolution)
        rh_hip_hold_1 = np.zeros(hold_times) + rh_hip_go[-1]
        rh_hip_back = np.linspace(self.hip[0] - angle_hip_rh_deg, self.hip[0], self.resolution)
        rh_hip_hold_2 = np.zeros(hold_times) + lf_hip_back[-1]


        lf_x = np.concatenate((lf_x_1, lf_hold_x_1, lf_x_2, lf_hold_x_2))
        rf_x = np.zeros(len(lf_x)) + self.origin
        lh_x = np.zeros(len(lf_x)) + self.origin
        rh_x = np.concatenate((rh_x_1, rh_hold_x_1, rh_x_2, rh_hold_x_2))
        
        lf_y = np.concatenate((lf_y_up, lf_y_down, lf_hip_hold_1, lf_y_back, lf_hold_y_2))
        rf_y = np.concatenate((rf_y_hold, rf_hold_y_2, rf_y_up, rf_y_down, rf_hold_y_3))
        lh_y = np.concatenate((lh_y_hold, lh_hold_y_2, lh_y_up, lh_y_down, lh_hold_y_3))
        rh_y = np.concatenate((rh_y_up, rh_y_down, rh_hip_hold_1, rh_y_back, rh_hold_y_2))

        self.gait_lf = np.stack((lf_x, lf_y), axis=-1)
        self.gait_rf = np.stack((rf_x, rf_y), axis=-1)
        self.gait_lh = np.stack((lh_x, lh_y), axis=-1)
        self.gait_rh = np.stack((rh_x, rh_y), axis=-1)
        self.gait_hip_lf = np.concatenate((lf_hip_go, lf_hip_hold_1, lf_hip_back, lf_hip_hold_2))
        self.gait_hip_rf = np.zeros(len(self.gait_rf)) + self.hip[1]
        self.gait_hip_lh = np.zeros(len(self.gait_lh)) + self.hip[2]
        self.gait_hip_rh = np.concatenate((rh_hip_go, rh_hip_hold_1, rh_hip_back, rh_hip_hold_2))
        self.renewBasicInfo()


class TrotRotateLeftGait(TrotRotateRightGait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        super().__init__(
            ground_level = self.ground_level,
            step_long=self.step_long,
            resolution=self.resolution,
            origin=self.origin,
            height=self.height,
            cycle=self.cycle,
            phase=self.phase,
            dt=self.dt,
            hip=self.hip,
        )
        self.turn_right = False
        self.generateGate()




class TrotSmoothRotateRightGait(Gait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.turn_right = True
        self.generateGate()

    def generateGate(self):
        # if turn left, just need to negative the trun_right flag
        if not self.turn_right: self.step_long = -self.step_long
        hold_times = round(self.resolution*0.7)
        angle_R_step_rad = np.arccos(self.step_long/2/ROBOT_R)
        angle_L_step_rad = np.pi - angle_R_step_rad - DIAGONAL_ANGLE_L_RAD
        angle_hip_lf_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[0])))))
        angle_hip_rf_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[1])))))
        angle_hip_lh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[2])))))
        angle_hip_rh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[3])))))

        lf_x_1 = np.linspace(self.origin, self.origin + self.step_long*np.cos(angle_L_step_rad), self.resolution)
        lf_hold_x_1 = np.zeros(hold_times) + lf_x_1[-1]
        lf_x_2 = np.linspace(self.origin + self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        lf_hold_x_2 = np.zeros(hold_times) + lf_x_2[-1]

        rf_x_1 = np.linspace(self.origin, self.origin + self.step_long*np.cos(angle_L_step_rad), self.resolution)
        rf_hold_x_1 = np.zeros(hold_times) + rf_x_1[-1]
        rf_x_2 = np.linspace(self.origin + self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        rf_hold_x_2 = np.zeros(hold_times) + rf_x_2[-1]

        lh_x_1 = np.linspace(self.origin, self.origin - self.step_long*np.cos(angle_L_step_rad), self.resolution)
        lh_hold_x_1 = np.zeros(hold_times) + lh_x_1[-1]
        lh_x_2 = np.linspace(self.origin - self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        lh_hold_x_2 = np.zeros(hold_times) + lh_x_2[-1]

        rh_x_1 = np.linspace(self.origin, self.origin - self.step_long*np.cos(angle_L_step_rad), self.resolution)
        rh_hold_x_1 = np.zeros(hold_times) + rh_x_1[-1]
        rh_x_2 = np.linspace(self.origin - self.step_long*np.cos(angle_L_step_rad), self.origin, self.resolution)
        rh_hold_x_2 = np.zeros(hold_times) + rh_x_2[-1]

        lf_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        lf_y_down = np.linspace(self.ground_level + self.height, -(self.ground_level**2 + self.step_long**2)**0.5 ,round(self.resolution/2))
        lf_hold_y_1 = np.zeros(hold_times) + lf_y_down[-1]
        lf_y_back = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level ,self.resolution)
        lf_hold_y_2 = np.zeros(hold_times) + lf_y_back[-1]

        rf_y_go = np.linspace(self.ground_level, -(self.ground_level**2 + self.step_long**2)**0.5 ,self.resolution)
        rf_hold_y_1 = np.zeros(hold_times) + rf_y_go[-1]
        rf_y_up = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level + self.height, round(self.resolution/2))
        rf_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        rf_hold_y_2 = np.zeros(hold_times) + rf_y_down[-1]

        lh_y_go = np.linspace(self.ground_level, -(self.ground_level**2 + self.step_long**2)**0.5 ,self.resolution)
        lh_hold_y_1 = np.zeros(hold_times) + lh_y_go[-1]
        lh_y_up = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level + self.height, round(self.resolution/2))
        lh_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        lh_hold_y_2 = np.zeros(hold_times) + lh_y_down[-1]

        rh_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        rh_y_down = np.linspace(self.ground_level + self.height, -(self.ground_level**2 + self.step_long**2)**0.5 ,round(self.resolution/2))
        rh_hold_y_1 = np.zeros(hold_times) + rh_y_down[-1]
        rh_y_back = np.linspace(-(self.ground_level**2 + self.step_long**2)**0.5, self.ground_level ,self.resolution)
        rh_hold_y_2 = np.zeros(hold_times) + lf_y_back[-1]


        lf_hip_go = np.linspace(self.hip[0], self.hip[0] - angle_hip_lf_deg, self.resolution)
        lf_hip_hold_1 = np.zeros(hold_times) + lf_hip_go[-1]
        lf_hip_back = np.linspace(self.hip[0] - angle_hip_lf_deg, self.hip[0], self.resolution)
        lf_hip_hold_2 = np.zeros(hold_times) + lf_hip_back[-1]

        rf_hip_go = np.linspace(self.hip[1], self.hip[1] - angle_hip_rf_deg, self.resolution)
        rf_hip_hold_1 = np.zeros(hold_times) + rf_hip_go[-1]
        rf_hip_back = np.linspace(self.hip[1] - angle_hip_rf_deg, self.hip[1], self.resolution)
        rf_hip_hold_2 = np.zeros(hold_times) + rf_hip_back[-1]

        lh_hip_go = np.linspace(self.hip[2], self.hip[2] - angle_hip_lh_deg, self.resolution)
        lh_hip_hold_1 = np.zeros(hold_times) + lh_hip_go[-1]
        lh_hip_back = np.linspace(self.hip[2] - angle_hip_lh_deg, self.hip[2], self.resolution)
        lh_hip_hold_2 = np.zeros(hold_times) + lh_hip_back[-1]

        rh_hip_go = np.linspace(self.hip[3], self.hip[3] - angle_hip_rh_deg, self.resolution)
        rh_hip_hold_1 = np.zeros(hold_times) + rh_hip_go[-1]
        rh_hip_back = np.linspace(self.hip[3] - angle_hip_rh_deg, self.hip[3], self.resolution)
        rh_hip_hold_2 = np.zeros(hold_times) + rh_hip_back[-1]


        lf_x = np.concatenate((lf_x_1, lf_hold_x_1, lf_x_2, lf_hold_x_2))
        rf_x = np.concatenate((rf_x_1, rf_hold_x_1, rf_x_2, rf_hold_x_2))
        lh_x = np.concatenate((lh_x_1, lh_hold_x_1, lh_x_2, lh_hold_x_2))
        rh_x = np.concatenate((rh_x_1, rh_hold_x_1, rh_x_2, rh_hold_x_2))
        
        lf_y = np.concatenate((lf_y_up, lf_y_down, lf_hold_y_1, lf_y_back, lf_hold_y_2))
        rf_y = np.concatenate((rf_y_go, rf_hold_y_1, rf_y_up, rf_y_down, rf_hold_y_2))
        lh_y = np.concatenate((lh_y_go, lh_hold_y_1, lh_y_up, lh_y_down, lh_hold_y_2))
        rh_y = np.concatenate((rh_y_up, rh_y_down, rh_hold_y_1, rh_y_back, rh_hold_y_2))

        self.gait_lf = np.stack((lf_x, lf_y), axis=-1)
        self.gait_rf = np.stack((rf_x, rf_y), axis=-1)
        self.gait_lh = np.stack((lh_x, lh_y), axis=-1)
        self.gait_rh = np.stack((rh_x, rh_y), axis=-1)
        self.gait_hip_lf = np.concatenate((lf_hip_go, lf_hip_hold_1, lf_hip_back, lf_hip_hold_2))
        self.gait_hip_rf = np.concatenate((rf_hip_go, rf_hip_hold_1, rf_hip_back, rf_hip_hold_2))
        self.gait_hip_lh = np.concatenate((lh_hip_go, lh_hip_hold_1, lh_hip_back, lh_hip_hold_2))
        self.gait_hip_rh = np.concatenate((rh_hip_go, rh_hip_hold_1, rh_hip_back, rh_hip_hold_2))
        self.renewBasicInfo()


class TrotSmoothRotateLeftGait(TrotSmoothRotateRightGait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        super().__init__(
            ground_level = self.ground_level,
            step_long=self.step_long,
            resolution=self.resolution,
            origin=self.origin,
            height=self.height,
            cycle=self.cycle,
            phase=self.phase,
            dt=self.dt,
            hip=self.hip,
        )
        self.turn_right = False
        self.generateGate()



class TrotMoveRightGait(Gait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        super().__init__()
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        self.move_right = True
        self.generateGate()


    def endGaitGenerate(self):
        lf_x = np.zeros(self.resolution) + self.origin
        lf_x_wait_1 = np.zeros(self.hold_times) + self.origin
        lf_x_hold = np.zeros(self.resolution) + self.origin
        lf_x_wait_2 = np.zeros(self.hold_times) + self.origin
        
        rf_x = np.zeros(self.resolution) + self.origin
        rf_x_wait_1 = np.zeros(self.hold_times) + self.origin
        rf_x_hold = np.zeros(self.resolution) + self.origin
        rf_x_wait_2 = np.zeros(self.hold_times) + self.origin

        lh_x = np.zeros(self.resolution) + self.origin
        lh_x_wait_1 = np.zeros(self.hold_times) + self.origin
        lh_x_hold = np.zeros(self.resolution) + self.origin
        lh_x_wait_2 = np.zeros(self.hold_times) + self.origin

        rh_x = np.zeros(self.resolution) + self.origin
        rh_x_wait_1 = np.zeros(self.hold_times) + self.origin
        rh_x_hold = np.zeros(self.resolution) + self.origin
        rh_x_wait_2 = np.zeros(self.hold_times) + self.origin

        lf_y_up = np.linspace(self.virtual_height_lf, self.ground_level + self.height, round(self.resolution/2))
        lf_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        lf_y_wait_1 = np.zeros(self.hold_times) + lf_y_down[-1]
        lf_y_hold = np.zeros(self.resolution) + lf_y_down[-1]
        lf_y_wait_2 = np.zeros(self.hold_times) + lf_y_down[-1]

        rf_y_hold = np.zeros(self.resolution) + self.virtual_height_rf
        rf_y_wait_1 = np.zeros(self.hold_times) + rf_y_hold[-1]
        rf_y_up = np.linspace(self.virtual_height_rf, self.ground_level + self.height, round(self.resolution/2))
        rf_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        rf_y_wait_2 = np.zeros(self.hold_times) + rf_y_down[-1]

        lh_y_hold = np.zeros(self.resolution) + self.virtual_height_lh
        lh_y_wait_1 = np.zeros(self.hold_times) + lh_y_hold[-1]
        lh_y_up = np.linspace(self.virtual_height_lh, self.ground_level + self.height, round(self.resolution/2))
        lh_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        lh_y_wait_2 = np.zeros(self.hold_times) + lh_y_down[-1]

        rh_y_up = np.linspace(self.virtual_height_rh, self.ground_level + self.height, round(self.resolution/2))
        rh_y_down = np.linspace(self.ground_level + self.height, self.ground_level, round(self.resolution/2))
        rh_y_wait_1 = np.zeros(self.hold_times) + rh_y_down[-1]
        rh_y_hold = np.zeros(self.resolution) + rh_y_down[-1]
        rh_y_wait_2 = np.zeros(self.hold_times) + rh_y_down[-1]


        lf_hip = np.linspace(270, self.hip[0], self.resolution)
        lf_hip_wait_1 = np.zeros(self.hold_times) + lf_hip[-1]
        lf_hip_hold = np.zeros(self.resolution) + lf_hip[-1]
        lf_hip_wait_2 = np.zeros(self.hold_times) + lf_hip_hold[-1]

        rf_hip_hold = np.zeros(self.resolution) + 270
        rf_hip_wait_1 = np.zeros(self.hold_times) + rf_hip_hold[-1]
        rf_hip = np.linspace(270, self.hip[1], self.resolution)
        rf_hip_wait_2 = np.zeros(self.hold_times) + rf_hip[-1]

        lh_hip_hold = np.zeros(self.resolution) + 270
        lh_hip_wait_1 = np.zeros(self.hold_times) + lh_hip_hold[-1]
        lh_hip = np.linspace(270, self.hip[2], self.resolution)
        lh_hip_wait_2 = np.zeros(self.hold_times) + lh_hip[-1]

        rh_hip = np.linspace(270, self.hip[3], self.resolution)
        rh_hip_wait_1 = np.zeros(self.hold_times) + rh_hip[-1]
        rh_hip_hold = np.zeros(self.resolution) + rh_hip[-1]
        rh_hip_wait_2 = np.zeros(self.hold_times) + rh_hip_hold[-1]

        lf_x = np.concatenate((lf_x, lf_x_wait_1, lf_x_hold, lf_x_wait_2))
        lf_y = np.concatenate((lf_y_up, lf_y_down, lf_y_wait_1, lf_y_hold, lf_y_wait_2))
        rf_x = np.concatenate((rf_x_hold, rf_x_wait_1, rf_x, rf_x_wait_2))
        rf_y = np.concatenate((rf_y_hold, rf_y_wait_1, rf_y_up, rf_y_down, rf_y_wait_2))
        lh_x = np.concatenate((lh_x_hold, lh_x_wait_1, lh_x, lh_x_wait_2))
        lh_y = np.concatenate((lh_y_hold, lh_y_wait_1, lh_y_up, lh_y_down, lh_y_wait_2))
        rh_x = np.concatenate((rh_x, rh_x_wait_1, rh_x_hold, rh_x_wait_2))
        rh_y = np.concatenate((rh_y_up, rh_y_down, rh_y_wait_1, rh_y_hold, rh_y_wait_2))

        self.gait_end_lf = np.stack((lf_x, lf_y), axis=-1)
        self.gait_end_rf = np.stack((rf_x, rf_y), axis=-1)
        self.gait_end_lh = np.stack((lh_x, lh_y), axis=-1)
        self.gait_end_rh = np.stack((rh_x, rh_y), axis=-1)
        self.gait_end_hip_lf = np.concatenate((lf_hip, lf_hip_wait_1, lf_hip_hold, lf_hip_wait_2))
        self.gait_end_hip_rf = np.concatenate((rf_hip_hold, rf_hip_wait_1, rf_hip, rf_hip_wait_2))
        self.gait_end_hip_lh = np.concatenate((lh_hip_hold, lh_hip_wait_1, lh_hip, lh_hip_wait_2))
        self.gait_end_hip_rh = np.concatenate((rh_hip, rh_hip_wait_1, rh_hip_hold, rh_hip_wait_2))

    def generateGate(self):
        self.virtual_height_lf = self.ground_level*np.cos(np.deg2rad(self.hip[0] - 270))
        self.virtual_height_rf = self.ground_level*np.cos(np.deg2rad(self.hip[1] - 270))
        self.virtual_height_lh = self.ground_level*np.cos(np.deg2rad(self.hip[2] - 270))
        self.virtual_height_rh = self.ground_level*np.cos(np.deg2rad(self.hip[3] - 270))
        # if turn left, just need to negative the trun_right flag
        if not self.move_right: self.step_long = -self.step_long
        # self.step_long = 0.5*self.step_long
        self.hold_times = round(self.resolution*0.2)
        self.normalGaitGenerate()
        if self.hip[0] != 270:
            self.initGaitGenerate()
            self.endGaitGenerate()
            self.bool_has_end_gait = True
            self.bool_has_init_gait = True
        self.renewBasicInfo()

    def normalGaitGenerate(self):
        # if self.move_right:
            # lf, lh negative so that lf, lh foot move right
        angle_hip_lf_deg = - np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[0])))))
        angle_hip_lh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[2])))))
        angle_hip_rf_deg = - np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[1])))))
        angle_hip_rh_deg = np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[3])))))
        # else:
        #     angle_hip_lf_deg = np.rad2deg(np.arctan((self.step_long + abs(self.ground_level)*np.sin(np.deg2rad(270 - self.hip[0])))/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[0])))))
        #     angle_hip_lh_deg = - np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[2])))))
        #     angle_hip_rf_deg = np.rad2deg(np.arctan((self.step_long + abs(self.ground_level)*np.sin(np.deg2rad(270 - self.hip[1])))/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[1])))))
        #     angle_hip_rh_deg = - np.rad2deg(np.arctan(self.step_long/(abs(self.ground_level)*np.cos(np.deg2rad(270 - self.hip[3])))))

        dynamic_y_lf = self.virtual_height_lf/np.cos(np.deg2rad(angle_hip_lf_deg))
        dynamic_y_rf = self.virtual_height_rf/np.cos(np.deg2rad(angle_hip_rf_deg))
        dynamic_y_lh = self.virtual_height_lh/np.cos(np.deg2rad(angle_hip_lh_deg))
        dynamic_y_rh = self.virtual_height_rh/np.cos(np.deg2rad(angle_hip_rh_deg))

        lf_x_1 = np.zeros(self.resolution) + self.origin
        lf_hold_x_1 = np.zeros(self.hold_times) + lf_x_1[-1]
        lf_x_2 = np.zeros(self.resolution) + self.origin
        lf_hold_x_2 = np.zeros(self.hold_times) + lf_x_2[-1]

        rf_x_1 = np.zeros(self.resolution) + self.origin
        rf_hold_x_1 = np.zeros(self.hold_times) + rf_x_1[-1]
        rf_x_2 = np.zeros(self.resolution) + self.origin
        rf_hold_x_2 = np.zeros(self.hold_times) + rf_x_2[-1]

        lh_x_1 = np.zeros(self.resolution) + self.origin
        lh_hold_x_1 = np.zeros(self.hold_times) + lh_x_1[-1]
        lh_x_2 = np.zeros(self.resolution) + self.origin
        lh_hold_x_2 = np.zeros(self.hold_times) + lh_x_2[-1]

        rh_x_1 = np.zeros(self.resolution) + self.origin
        rh_hold_x_1 = np.zeros(self.hold_times) + rh_x_1[-1]
        rh_x_2 = np.zeros(self.resolution) + self.origin
        rh_hold_x_2 = np.zeros(self.hold_times) + rh_x_2[-1]

        lf_y_up = np.linspace(self.virtual_height_lf, self.virtual_height_lf + self.height, round(self.resolution/2))
        lf_y_down = np.linspace(self.virtual_height_lf + self.height, dynamic_y_lf,round(self.resolution/2))
        lf_hold_y_1 = np.zeros(self.hold_times) + lf_y_down[-1]
        lf_y_back = np.linspace(dynamic_y_lf, self.virtual_height_lf ,self.resolution)
        lf_hold_y_2 = np.zeros(self.hold_times) + lf_y_back[-1]

        rf_y_go = np.linspace(self.virtual_height_rf, dynamic_y_rf,self.resolution)
        rf_hold_y_1 = np.zeros(self.hold_times) + rf_y_go[-1]
        rf_y_up = np.linspace(dynamic_y_rf, self.virtual_height_rf + self.height, round(self.resolution/2))
        rf_y_down = np.linspace(self.virtual_height_rf + self.height, self.virtual_height_rf, round(self.resolution/2))
        rf_hold_y_2 = np.zeros(self.hold_times) + rf_y_down[-1]

        lh_y_go = np.linspace(self.virtual_height_lh, dynamic_y_lh ,self.resolution)
        lh_hold_y_1 = np.zeros(self.hold_times) + lh_y_go[-1]
        lh_y_up = np.linspace(dynamic_y_lh, self.virtual_height_lh + self.height, round(self.resolution/2))
        lh_y_down = np.linspace(self.virtual_height_lh + self.height, self.virtual_height_lh, round(self.resolution/2))
        lh_hold_y_2 = np.zeros(self.hold_times) + lh_y_down[-1]

        rh_y_up = np.linspace(self.virtual_height_rh, self.virtual_height_rh + self.height, round(self.resolution/2))
        rh_y_down = np.linspace(self.virtual_height_rh + self.height, dynamic_y_rh, round(self.resolution/2))
        rh_hold_y_1 = np.zeros(self.hold_times) + rh_y_down[-1]
        rh_y_back = np.linspace(dynamic_y_rh, self.virtual_height_rh ,self.resolution)
        rh_hold_y_2 = np.zeros(self.hold_times) + lf_y_back[-1]


        lf_hip_go = np.linspace(270, 270 + angle_hip_lf_deg, self.resolution)
        lf_hip_hold_1 = np.zeros(self.hold_times) + lf_hip_go[-1]
        lf_hip_back = np.linspace(270 + angle_hip_lf_deg, 270, self.resolution)
        lf_hip_hold_2 = np.zeros(self.hold_times) + lf_hip_back[-1]

        rf_hip_go = np.linspace(270, 270 + angle_hip_rf_deg, self.resolution)
        rf_hip_hold_1 = np.zeros(self.hold_times) + rf_hip_go[-1]
        rf_hip_back = np.linspace(270 + angle_hip_rf_deg, 270, self.resolution)
        rf_hip_hold_2 = np.zeros(self.hold_times) + rf_hip_back[-1]

        lh_hip_go = np.linspace(270, 270 + angle_hip_lh_deg, self.resolution)
        lh_hip_hold_1 = np.zeros(self.hold_times) + lh_hip_go[-1]
        lh_hip_back = np.linspace(270 + angle_hip_lh_deg, 270, self.resolution)
        lh_hip_hold_2 = np.zeros(self.hold_times) + lh_hip_back[-1]

        rh_hip_go = np.linspace(270, 270 + angle_hip_rh_deg, self.resolution)
        rh_hip_hold_1 = np.zeros(self.hold_times) + rh_hip_go[-1]
        rh_hip_back = np.linspace(270 + angle_hip_rh_deg, 270, self.resolution)
        rh_hip_hold_2 = np.zeros(self.hold_times) + rh_hip_back[-1]


        lf_x = np.concatenate((lf_x_1, lf_hold_x_1, lf_x_2, lf_hold_x_2))
        rf_x = np.concatenate((rf_x_1, rf_hold_x_1, rf_x_2, rf_hold_x_2))
        lh_x = np.concatenate((lh_x_1, lh_hold_x_1, lh_x_2, lh_hold_x_2))
        rh_x = np.concatenate((rh_x_1, rh_hold_x_1, rh_x_2, rh_hold_x_2))
        
        lf_y = np.concatenate((lf_y_up, lf_y_down, lf_hold_y_1, lf_y_back, lf_hold_y_2))
        rf_y = np.concatenate((rf_y_go, rf_hold_y_1, rf_y_up, rf_y_down, rf_hold_y_2))
        lh_y = np.concatenate((lh_y_go, lh_hold_y_1, lh_y_up, lh_y_down, lh_hold_y_2))
        rh_y = np.concatenate((rh_y_up, rh_y_down, rh_hold_y_1, rh_y_back, rh_hold_y_2))

        self.gait_lf = np.stack((lf_x, lf_y), axis=-1)
        self.gait_rf = np.stack((rf_x, rf_y), axis=-1)
        self.gait_lh = np.stack((lh_x, lh_y), axis=-1)
        self.gait_rh = np.stack((rh_x, rh_y), axis=-1)
        self.gait_hip_lf = np.concatenate((lf_hip_go, lf_hip_hold_1, lf_hip_back, lf_hip_hold_2))
        self.gait_hip_rf = np.concatenate((rf_hip_go, rf_hip_hold_1, rf_hip_back, rf_hip_hold_2))
        self.gait_hip_lh = np.concatenate((lh_hip_go, lh_hip_hold_1, lh_hip_back, lh_hip_hold_2))
        self.gait_hip_rh = np.concatenate((rh_hip_go, rh_hip_hold_1, rh_hip_back, rh_hip_hold_2))

    def initGaitGenerate(self):
        lf_x = np.zeros(self.resolution) + self.origin
        lf_x_wait_1 = np.zeros(self.hold_times) + self.origin
        lf_x_hold = np.zeros(self.resolution) + self.origin
        lf_x_wait_2 = np.zeros(self.hold_times) + self.origin
        
        rf_x = np.zeros(self.resolution) + self.origin
        rf_x_wait_1 = np.zeros(self.hold_times) + self.origin
        rf_x_hold = np.zeros(self.resolution) + self.origin
        rf_x_wait_2 = np.zeros(self.hold_times) + self.origin

        lh_x = np.zeros(self.resolution) + self.origin
        lh_x_wait_1 = np.zeros(self.hold_times) + self.origin
        lh_x_hold = np.zeros(self.resolution) + self.origin
        lh_x_wait_2 = np.zeros(self.hold_times) + self.origin

        rh_x = np.zeros(self.resolution) + self.origin
        rh_x_wait_1 = np.zeros(self.hold_times) + self.origin
        rh_x_hold = np.zeros(self.resolution) + self.origin
        rh_x_wait_2 = np.zeros(self.hold_times) + self.origin

        lf_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        lf_y_down = np.linspace(self.ground_level + self.height, self.virtual_height_lf, round(self.resolution/2))
        lf_y_wait_1 = np.zeros(self.hold_times) + lf_y_down[-1]
        lf_y_hold = np.zeros(self.resolution) + lf_y_down[-1]
        lf_y_wait_2 = np.zeros(self.hold_times) + lf_y_down[-1]

        rf_y_hold = np.zeros(self.resolution) + self.ground_level
        rf_y_wait_1 = np.zeros(self.hold_times) + rf_y_hold[-1]
        rf_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        rf_y_down = np.linspace(self.ground_level + self.height, self.virtual_height_rf, round(self.resolution/2))
        rf_y_wait_2 = np.zeros(self.hold_times) + rf_y_down[-1]

        lh_y_hold = np.zeros(self.resolution) + self.ground_level
        lh_y_wait_1 = np.zeros(self.hold_times) + lh_y_hold[-1]
        lh_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        lh_y_down = np.linspace(self.ground_level + self.height, self.virtual_height_lh, round(self.resolution/2))
        lh_y_wait_2 = np.zeros(self.hold_times) + lh_y_down[-1]

        rh_y_up = np.linspace(self.ground_level, self.ground_level + self.height, round(self.resolution/2))
        rh_y_down = np.linspace(self.ground_level + self.height, self.virtual_height_rh, round(self.resolution/2))
        rh_y_wait_1 = np.zeros(self.hold_times) + rh_y_down[-1]
        rh_y_hold = np.zeros(self.resolution) + rh_y_down[-1]
        rh_y_wait_2 = np.zeros(self.hold_times) + rh_y_down[-1]

        lf_hip = np.linspace(self.hip[0], 270, self.resolution)
        lf_hip_wait_1 = np.zeros(self.hold_times) + lf_hip[-1]
        lf_hip_hold = np.zeros(self.resolution) + lf_hip[-1]
        lf_hip_wait_2 = np.zeros(self.hold_times) + lf_hip_hold[-1]

        rf_hip_hold = np.zeros(self.resolution) + self.hip[1]
        rf_hip_wait_1 = np.zeros(self.hold_times) + rf_hip_hold[-1]
        rf_hip = np.linspace(self.hip[1], 270, self.resolution)
        rf_hip_wait_2 = np.zeros(self.hold_times) + rf_hip[-1]

        lh_hip_hold = np.zeros(self.resolution) + self.hip[2]
        lh_hip_wait_1 = np.zeros(self.hold_times) + lh_hip_hold[-1]
        lh_hip = np.linspace(self.hip[2], 270, self.resolution)
        lh_hip_wait_2 = np.zeros(self.hold_times) + lh_hip[-1]

        rh_hip = np.linspace(self.hip[3], 270, self.resolution)
        rh_hip_wait_1 = np.zeros(self.hold_times) + rh_hip[-1]
        rh_hip_hold = np.zeros(self.resolution) + rh_hip[-1]
        rh_hip_wait_2 = np.zeros(self.hold_times) + rh_hip_hold[-1]

        lf_x = np.concatenate((lf_x, lf_x_wait_1, lf_x_hold, lf_x_wait_2))
        lf_y = np.concatenate((lf_y_up, lf_y_down, lf_y_wait_1, lf_y_hold, lf_y_wait_2))
        rf_x = np.concatenate((rf_x_hold, rf_x_wait_1, rf_x, rf_x_wait_2))
        rf_y = np.concatenate((rf_y_hold, rf_y_wait_1, rf_y_up, rf_y_down, rf_y_wait_2))
        lh_x = np.concatenate((lh_x_hold, lh_x_wait_1, lh_x, lh_x_wait_2))
        lh_y = np.concatenate((lh_y_hold, lh_y_wait_1, lh_y_up, lh_y_down, lh_y_wait_2))
        rh_x = np.concatenate((rh_x, rh_x_wait_1, rh_x_hold, rh_x_wait_2))
        rh_y = np.concatenate((rh_y_up, rh_y_down, rh_y_wait_1, rh_y_hold, rh_y_wait_2))

        self.gait_init_lf = np.stack((lf_x, lf_y), axis=-1)
        self.gait_init_rf = np.stack((rf_x, rf_y), axis=-1)
        self.gait_init_lh = np.stack((lh_x, lh_y), axis=-1)
        self.gait_init_rh = np.stack((rh_x, rh_y), axis=-1)
        self.gait_init_hip_lf = np.concatenate((lf_hip, lf_hip_wait_1, lf_hip_hold, lf_hip_wait_2))
        self.gait_init_hip_rf = np.concatenate((rf_hip_hold, rf_hip_wait_1, rf_hip, rf_hip_wait_2))
        self.gait_init_hip_lh = np.concatenate((lh_hip_hold, lh_hip_wait_1, lh_hip, lh_hip_wait_2))
        self.gait_init_hip_rh = np.concatenate((rh_hip, rh_hip_wait_1, rh_hip_hold, rh_hip_wait_2))



class TrotMoveLeftGait(TrotMoveRightGait):
    def __init__(self, ground_level=-37, step_long=3, resolution=20, **kwargs):
        self.ground_level = ground_level
        self.step_long = step_long
        self.resolution = resolution
        self.origin = kwargs["origin"] if "origin" in kwargs else -3
        self.height = kwargs["height"] if "height" in kwargs else 5
        self.cycle = kwargs["cycle"] if "cycle" in kwargs else 5
        self.phase = kwargs["phase"] if "phase" in kwargs else 120
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.015
        self.hip = kwargs["hip"] if "hip" in kwargs else [270, 270, 270, 270]
        super().__init__(
            ground_level = self.ground_level,
            step_long=self.step_long,
            resolution=self.resolution,
            origin=self.origin,
            height=self.height,
            cycle=self.cycle,
            phase=self.phase,
            dt=self.dt,
            hip=self.hip,
        )
        self.move_right = False
        self.generateGate()


if __name__ == "__main__":
    gait = TrotStandingInPlaceGait()
    print(gait.gaitLookUp(20))