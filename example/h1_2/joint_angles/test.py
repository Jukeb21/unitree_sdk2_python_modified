import PyKDL
from kdl_parser.urdf import treeFromFile, treeFromUrdfModel
from urdf_parser_py.urdf import URDF
import numpy as np
import os

def load_urdf_to_kdl(urdf_file_path):
    """
    从URDF文件加载KDL树结构
    参数:
        urdf_file_path: URDF文件的路径
    返回:
        success: 加载是否成功
        tree: KDL树对象
    """
    if not os.path.exists(urdf_file_path):
        print(f"urdf not exists - {urdf_file_path}")
        return False, None
    
    try:
        success, tree = treeFromFile(urdf_file_path)
        if success:
            print(f"load sucess: {urdf_file_path}")
            print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
            return success, tree
    except Exception as e:
        print(f"methods1 failed: {e}")
    
    # 使用二进制模式读取文件并移除编码声明
    try:
        with open(urdf_file_path, 'rb') as f:
            # 读取二进制数据
            xml_content = f.read()
            # 尝试解码为字符串
            try:
                xml_str = xml_content.decode('utf-8')
            except UnicodeDecodeError:
                # 尝试其他编码
                xml_str = xml_content.decode('latin-1')
            # 移除XML编码声明
            if xml_str.startswith('<?xml'):
                xml_str = xml_str[xml_str.find('?>') + 2:]
            # 从字符串创建URDF模型
            robot_model = URDF.from_xml_string(xml_str)
            # 从URDF模型创建KDL树
            success, tree = treeFromUrdfModel(robot_model)
            if success:
                print(f"method2 sucess: {urdf_file_path}")
                print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
                return success, tree
            else:
                print("method2 failed")
    except Exception as e:
        print(f"method2 failed: {e}")
    
    try:
        robot_model = URDF.from_xml_file(urdf_file_path)
        success, tree = treeFromUrdfModel(robot_model)
        
        if success:
            print(f"method3 sucess: {urdf_file_path}")
            print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
            return success, tree
        else:
            print("method3 failed")
    except Exception as e:
        print(f"method3 failed: {e}")
    
    print(f"can not load: {urdf_file_path}")
    return False, None

def create_chain_from_tree(tree, base_link, tip_link):
    """
    从KDL树创建运动链
    参数:
        tree: KDL树对象
        base_link: 基链接名称
        tip_link: 末端执行器链接名称
    返回:
        chain: KDL链对象
    """
    try:
        chain = tree.getChain(base_link, tip_link)
        print(f"create chain from {base_link} to {tip_link}")
        print(f"chain has {chain.getNrOfSegments()} segs and {chain.getNrOfJoints()} joints")
        return chain
    except Exception as e:
        print(f"create chain failed: {e}")
        return None

def setup_kinematics(chain):
    """
    设置正逆运动学求解器
    参数:
        chain: KDL链对象
    返回:
        fk_solver: 正向运动学求解器
        ik_vel_solver: 逆速度运动学求解器
        ik_pos_solver: 逆位置运动学求解器
        chain: 运动链对象 (添加此返回值以便在其他函数中使用)
    """
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    max_iterations = 100
    eps = 1e-6
    ik_pos_solver = PyKDL.ChainIkSolverPos_NR(
        chain, fk_solver, ik_vel_solver, max_iterations, eps
    )
    
    return fk_solver, ik_vel_solver, ik_pos_solver, chain

def perform_forward_kinematics(fk_solver, joint_positions):
    """
    执行正向运动学计算
    参数:
        fk_solver: 正向运动学求解器
        joint_positions: 关节位置列表
    返回:
        frame: KDL Frame对象，表示末端执行器的位姿
    """
    q = PyKDL.JntArray(len(joint_positions))
    for i in range(len(joint_positions)):
        q[i] = joint_positions[i]
    
    frame = PyKDL.Frame()
    status = fk_solver.JntToCart(q, frame)
    
    if status >= 0:
        position = [frame.p.x(), frame.p.y(), frame.p.z()]
        rotation = frame.M.GetQuaternion() 
        print(f"fk sucess:")
        print(f"pos: {position}")
        print(f"quat: {rotation}")
        return frame
    else:
        print("fk failed")
        return None

def perform_inverse_kinematics(ik_pos_solver, chain, target_frame, initial_joints=None):
    """
    执行逆运动学计算
    参数:
        ik_pos_solver: 逆位置运动学求解器
        chain: 运动链对象
        target_frame: 目标位姿 (KDL Frame)
        initial_joints: 初始关节配置 (可选)
    返回:
        joint_positions: 关节位置列表
    """
    num_joints = chain.getNrOfJoints()
    
    if initial_joints is None:
        q_init = PyKDL.JntArray(num_joints)
    else:
        q_init = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            q_init[i] = initial_joints[i]
    
    q_out = PyKDL.JntArray(num_joints)
    status = ik_pos_solver.CartToJnt(q_init, target_frame, q_out)
    
    if status >= 0:
        joint_positions = [q_out[i] for i in range(num_joints)]
        print(f"ik sucess:")
        print(f"Joints: {joint_positions}")
        return joint_positions
    else:
        print("ik failed")
        return None

def main():
    urdf_file = r"D:\Documents\unitree_sdk2\unitree_sdk2_python_modified\example\h1_2\h1_2.urdf"
    
    success, tree = load_urdf_to_kdl(urdf_file)
    if not success:
        return
    
    base_link = "Base"
    tip_link = "Moving Jaw"
    chain = create_chain_from_tree(tree, base_link, tip_link)
    if chain is None:
        return
    
    fk_solver, ik_vel_solver, ik_pos_solver, chain = setup_kinematics(chain)
    
    print("\n=== FK ===")
    joint_positions = [0.2, 0.2, 0.3, 0.4, 0.5, 0.6]
    if len(joint_positions) != chain.getNrOfJoints():
        print(f"joints ({len(joint_positions)}) not match ({chain.getNrOfJoints()})")
        joint_positions = [0.0] * chain.getNrOfJoints()
    
    end_effector_frame = perform_forward_kinematics(fk_solver, joint_positions)
    
    print("\n=== IK ===")
    if end_effector_frame is not None:
        # 将FK得到的末端执行器位姿作为目标
        inverse_joints = perform_inverse_kinematics(ik_pos_solver, chain, end_effector_frame, joint_positions)
        if inverse_joints is not None:
            verified_frame = perform_forward_kinematics(fk_solver, inverse_joints)
            if verified_frame is not None:
                # 计算位置误差
                error_pos = np.sqrt(
                    (verified_frame.p.x() - end_effector_frame.p.x())**2 +
                    (verified_frame.p.y() - end_effector_frame.p.y())**2 +
                    (verified_frame.p.z() - end_effector_frame.p.z())**2
                )
                print(f"error: {error_pos}")

if __name__ == "__main__":
    main()