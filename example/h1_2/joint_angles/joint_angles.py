import PyKDL
import math

pi = math.pi
 
def create_panda_chain():
    # DH参数
    a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
    alpha = [0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
    d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
    theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    chain = PyKDL.Chain()
    for i in range(7):
        chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotZ),
                                       PyKDL.Frame(PyKDL.Rotation.RotZ(theta[i]) * PyKDL.Rotation.RotX(alpha[i]),
                                                   PyKDL.Vector(a[i], -d[i] * math.sin(alpha[i]),
                                                                d[i] * math.cos(alpha[i])))))
    return chain


def compute_inverse_kinematics(chain, target_pose):
    '''
    正运动学
    '''             

    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    pos = PyKDL.Frame()
    q = PyKDL.JntArray(7)
    qq = [-0.917812, -0.917812, 43.2983, 21.2432, 16.8387, -27.4167, 19.5677]

    for i in range(7):
        q[i] = qq[i]
    fk_flag = fk.JntToCart(q, pos)
    print("fk_flag", fk_flag)
    print("pos", pos)

    '''
    逆运动学
    '''
    ikv = PyKDL.ChainIkSolverVel_pinv(chain)
    ik = PyKDL.ChainIkSolverPos_NR(chain, fk, ikv)
    # 创建目标位姿
    target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(target_pose[3], target_pose[4], target_pose[5]),
                               PyKDL.Vector(target_pose[0], target_pose[1], target_pose[2]))
    # 创建起始关节角度
    initial_joint_angles = PyKDL.JntArray(chain.getNrOfJoints())
    result = PyKDL.JntArray(chain.getNrOfJoints())
    # print(target_frame)
    # 调用逆运动学求解器
    ik.CartToJnt(initial_joint_angles, target_frame, result)
    print('result: ', result)
    return result


if __name__ == "__main__":
    # 创建机器人链
    chain = create_panda_chain()
    # 设置目标位姿
    target_pose = [0.5, 0.3, 0.4, 0.1, 0.0, 0.0]
    # 调用逆运动学求解函数
    joint_angles = compute_inverse_kinematics(chain, target_pose)
    print("关节角度: ", joint_angles)
    