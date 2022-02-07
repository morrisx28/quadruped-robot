#position based reinforcement learning
#State [ERR_Ang_seg, ERR_Rad_Seg, M1_Ang, M2_Ang, M3_Ang]
# ERR_Ang_seg => [0~49]
#Saperate Q Network
# import robot_foot_model as rf_model
import QLearning as QL
import LearningUtility as LUtil
import ROSUtility as ROStil
import numpy as np
import math
import LearningConmunication as LCM
from LearningConmunication import StatusCmd
import RewardCPG as RewardFunc
import time, datetime
from collections import namedtuple
import sys, os
import quadruped_CPG_controller as CPG
import ROS_quadruped_control as ROS
import rospy

# global EXCU_MODE, CONTINUOUS_MODE
EXCU_MODE, CONTINUOUS_MODE = False, False
learning_conf = None


class TrainingConfiguration(LUtil.TrainingConfiguration):

    def __init__(self):
        super(TrainingConfiguration, self).__init__()
        self.name = None
        self.object_type = "TrainingConfiguration"
        self.configuration_version = "1.0.1"
        self.configuration_discription = "basic gait learning"

        #Operation Configuration
        self.operating_frequency = 20
        self.state_number        = 3
        self.action_range        = [-0.12, 0.12]       # feedback signal limit
        self.action_resolution   = 21            # must be odd
        self.action_type         = 'linear'      # linear, triple, ununiform


        #Environment Configuration


        #QLearning Configuration

        
        self.network_shape = [
            self.state_number,
            100,400,600,600,1000,2000,
            self.action_resolution**2    
        ]

        self.dropout = None

        self.activated_reward = [
            'Fail'
        ]



def main():
    global EXCU_MODE, CONTINUOUS_MODE, learning_conf
    if len(sys.argv) > 1:
        if sys.argv[1] == 'excu': 
            EXCU_MODE = True
            print("Server Running with: EXCU_MODE = " + str(EXCU_MODE))
        elif sys.argv[1] in ['-c', '--continuous']:
            CONTINUOUS_MODE = True
            if len(sys.argv) > 2:
                try:
                    old_config = LUtil.loadOldConfig(sys.argv[2])
                except Exception as e:
                    print(e)
                    sys.exit(0)
            else:
                print("input Error")
                sys.exit(0)
    
    


    ##Training Configuration
    QL.initialQLearning()
    if CONTINUOUS_MODE:
        if old_config.configuration_version == "1.0.1":
            learning_conf = old_config
        else:
            print("Config Version Error")
            sys.exit()
        QL.InitialQNetwork(
            learning_conf.network_shape,
            designative_mode=True,
            model_name=sys.argv[2]
        )
        rl_robot = ROStil.selectRobot(learning_conf)
        ros_comunicator = ROS.ROS_Comunication()
    else:
        learning_conf       = TrainingConfiguration()
        # rl_foot = rf_model.Foot()
        rl_robot = CPG.CPG_controller()
        ros_comunicator = ROS.ROS_Comunication()
        learning_conf.robot_name = rl_robot.__class__.__name__

        QL.InitialQNetwork(learning_conf.network_shape, dropout_rate=learning_conf.dropout)
        # QL.InitialQNetwork(learning_conf.network_shape)
    #Configuration Finished
    learning_conf.selfCheck()    
    if not learning_conf.self_checked: return #Stop Learning if Error


    ACTION_SPACE = LUtil.setActionSpace(learning_conf)
    n_action = learning_conf.action_resolution**2

    training_memory = QL.LearningMemory(
        5 * rl_robot.numberOfSteps 
    )
    QL.chengeBatchSize(rl_robot.numberOfSteps)
    QL.changeEpsDecay(rl_robot.numberOfSteps * 20)

    #Setting model
    rl_robot.activateRobot()
    fail_flag = ros_comunicator.selfCheck()
    

    #Setting reward function
    RFunc = RewardFunc.RewardMachine(learning_conf.activated_reward)
    if bool(learning_conf.reward_factor):
        if RFunc.reward_factor.keys() == learning_conf.reward_factor.keys():
            #Reward factor key match, update factor by config file
            RFunc.reward_factor = learning_conf.reward_factor
        else:
            #Mismatch
            ans = input("Reward Factor Key Missmatch, Continu as defalt(all 1)?")
            if ans in ['y', 'yes']:
                learning_conf.reward_factor = RFunc.reward_factor
    else:
        #New Learning
        learning_conf.reward_factor = RFunc.reward_factor

    
    targetNet_update = 5000
    total_steps = 0
    target_steps = 0
    ros_fail_flag = False
    robot_fail_flag = False
    # timer = LUtil.RLTimer()
    # timer.resetAllTimer()
    while True:
        ros_comunicator.unpauseSim()   #ros warm up
        ros_comunicator.pauseSim()
        current_state = ROStil.statelize(ros_comunicator,rl_robot)
        # continue_flag = True
        while True:
            ###Main Learning Code
            action, random_flag, explorer_rate = QL.choseAction(
                current_state,
                n_action,
                total_steps,
                not EXCU_MODE
            )
            
            ros_comunicator.unpauseSim()
            fb_cmd = LUtil.processTwoAction(action, ACTION_SPACE)
            rl_robot.excuFbCmd(
                fb_cmd
            )
            
            print(f"fb_cmd: {fb_cmd}")
            ros_comunicator.pauseSim()
            new_state = ROStil.statelize(ros_comunicator,rl_robot)
            steps = rl_robot.updateSteps()
            fail_flag = ros_comunicator.robotOffsetCheck()

            if fail_flag is True:  
                ros_comunicator.unpauseSim()
                rl_robot.resetAllSteps()
                new_state = None
                rl_robot.resetAllPos()
                ros_comunicator.resetWorld()
                ros_fail_flag = ros_comunicator.selfCheck()
                rospy.sleep(1)
        
            reward = RFunc.excuReward(current_state, new_state, rl_robot)
            transition = (current_state, action, new_state, reward)
            print(f"Reward: {reward}")
            
            if not EXCU_MODE and not ros_fail_flag:
                training_memory.push(*transition)
                loss = QL.Learning(training_memory)
                if total_steps % targetNet_update == 0: 
                    print("====Update Target Net====")
                    QL.updateTarget()
            else:
                loss = None
            current_state = new_state if new_state is not None else ROStil.statelize(ros_comunicator,rl_robot)

            if steps == rl_robot.numberOfSteps or ros_fail_flag is True:
                if steps == rl_robot.numberOfSteps:
                    rl_robot.saveData(excu_mode=EXCU_MODE)
                ros_comunicator.unpauseSim()
                rl_robot.resetAllSteps()
                new_state = None
                rl_robot.resetAllPos()
                ros_comunicator.resetWorld()
                ros_fail_flag = ros_comunicator.selfCheck()
                rospy.sleep(1)
            
            
            
            total_steps += 1
            ros_fail_flag = ros_comunicator.selfCheck()
            if not ros_fail_flag:
                print("step end")
                break
            else:
                print("fail")
                pass
        



if __name__ == "__main__":
    # testGround()
    try:
        main()
    except KeyboardInterrupt as _:
        # while main.tsne_generator.process:
        #     pass
        if not EXCU_MODE:
            print("Learning Stop")
            reply = input("Need Saving This Model?:")
            if reply in ["y", "yes"]:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                QL.SaveModel(filename, learning_conf,remove_temp=True)
                print("Model Saved")
            
        else:
            print("Stop Excution")