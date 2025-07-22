import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

import numpy as np
from scipy.optimize import root

A0 = 0.07106
A1 = 0.275437
A2 = 0.232857
A3 = 0.054
G = 0.2618
Z = 0.0896
Y = 0.2868
X = 0.3346

H1_2_NUM_MOTOR = 27

def equations(vars):
    x0, x1, x2 = vars
    eq1 = (-A1 * np.cos(x0) + A2 * np.sin(x0 - x2)) * np.cos(G + x1) + A0 * np.sin(G) - Z
    eq2 = (A1 * np.sin(x0) + A2 * np.cos(x0 - x2) + A3) - Y
    eq3 = (A1 * np.cos(x0) - A2 * np.sin(x0 - x2)) * np.sin(G + x1) + A0 * np.cos(G) - X
    return [eq1, eq2, eq3]



class H1_2_JointIndex:
    # legs
    LeftHipYaw = 0
    LeftHipPitch = 1
    LeftHipRoll = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipYaw = 6
    RightHipPitch = 7
    RightHipRoll = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    # torso
    WaistYaw = 12
    # arms
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    LeftWristPitch = 18
    LeftWristYaw = 19
    RightShoulderPitch = 20
    RightShoulderRoll = 21
    RightShoulderYaw = 22
    RightElbow = 23
    RightWristRoll = 24
    RightWristPitch = 25
    RightWristYaw = 26


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            print(self.low_state.imu_state.rpy)

    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(H1_2_NUM_MOTOR):
            ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
            self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
            self.low_cmd.motor_cmd[i].tau = 0.0 
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 100.0 if i < 13 else 50.0
            self.low_cmd.motor_cmd[i].kd = 1.0

        if self.time_ < self.duration_ :
            # [Stage 1]: set robot to zero posture
            for i in range(H1_2_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q 
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = 100.0 if i < 13 else 50.0
                self.low_cmd.motor_cmd[i].kd = 1.0
        else :
            # [Stage 2]: swing ankle using PR mode
            # max_P = 0.25
            # max_R = 0.25
            # t = self.time_ - self.duration_ 
            # L_P_des = max_P * np.cos(2.0 * np.pi * t)
            # L_R_des = max_R * np.sin(2.0 * np.pi * t)
            # R_P_des = max_P * np.cos(2.0 * np.pi * t)
            # R_R_des = -max_R * np.sin(2.0 * np.pi * t)

            # Kp_Pitch = 80
            # Kd_Pitch = 1
            # Kp_Roll = 80
            # Kd_Roll = 1

            # self.low_cmd.mode_pr = Mode.PR
            # self.low_cmd.mode_machine = self.mode_machine_
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnklePitch].q = L_P_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnklePitch].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnklePitch].kp = Kp_Pitch
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnklePitch].kd = Kd_Pitch
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnkleRoll].q = L_R_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnkleRoll].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnkleRoll].kp = Kp_Roll
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftAnkleRoll].kd = Kd_Roll
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnklePitch].q = R_P_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnklePitch].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnklePitch].kp = Kp_Pitch
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnklePitch].kd = Kd_Pitch
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnkleRoll].q = R_R_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnkleRoll].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnkleRoll].kp = Kp_Roll
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnkleRoll].kd = Kd_Roll

            # max_wrist_roll_angle = 0.5;  # [rad]
            # WristRoll_des = max_wrist_roll_angle * np.sin(2.0 * np.pi * t)
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftWristRoll].q = WristRoll_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftWristRoll].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftWristRoll].kp = 50
            # self.low_cmd.motor_cmd[H1_2_JointIndex.LeftWristRoll].kd = 1
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightWristRoll].q = WristRoll_des
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightWristRoll].dq = 0
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightWristRoll].kp = 50
            # self.low_cmd.motor_cmd[H1_2_JointIndex.RightWristRoll].kd = 1

            # 初始猜测值
            initial_guess = [0.5, 0.5, 0.5]
            solution = root(equations, initial_guess)
            if solution.success:
                x0_num, x1_num, x2_num = solution.x
                j3_num = x0_num - x2_num
                if -3.14 <= x0_num <= 1.57 and -0.38 <= x1_num <= 3.4 and -0.471 <= x2_num <= 0.349 and -1.012 <= j3_num <= 1.012:
                    print(f"解为: x0 = {x0_num}, x1 = {x1_num}, x2 = {x2_num}, x3 = {j3_num}")
                else:
                    print("超过关节限制")
            else:
                print("方程组无解")


            right_arm_target_angles = [x0_num, x1_num, 0, x2_num, 0, 0, j3_num]

            ratio = np.clip((self.time_ - self.duration_) / self.duration_, 0.0, 1.0)

            right_arm_indices = [
                H1_2_JointIndex.RightShoulderPitch,
                H1_2_JointIndex.RightShoulderRoll,
                H1_2_JointIndex.RightShoulderYaw,
                H1_2_JointIndex.RightElbow,
                H1_2_JointIndex.RightWristRoll,
                H1_2_JointIndex.RightWristPitch,
                H1_2_JointIndex.RightWristYaw,
            ]
            # 设置右臂关节角度
            self.low_cmd.mode_pr = Mode.AB
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightHipYaw].mode = 1  # Enable
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightHipPitch].mode = 1  # Enable
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightHipRoll].mode = 1  # Enable
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightKnee].mode = 1  # Enable
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnklePitch].mode = 1  # Enable
            self.low_cmd.motor_cmd[H1_2_JointIndex.RightAnkleRoll].mode = 1  # Enable
            for idx, target_q in zip(right_arm_indices, right_arm_target_angles):
                current_q = self.low_state.motor_state[idx].q  # 当前角度
                self.low_cmd.motor_cmd[idx].q = (1 - ratio) * current_q + ratio * target_q
                self.low_cmd.motor_cmd[idx].dq = 0
                self.low_cmd.motor_cmd[idx].kp = 50
                self.low_cmd.motor_cmd[idx].kd = 1

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)