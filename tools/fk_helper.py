import tools.torch_fk as fk
from tools.robot_hands_helper import CommonJoints
import torch


def get_chains(urdf_path, use_gpu=False, device=None):
    with open(urdf_path, "r") as f:
        robot_info = f.read()
    device = device if use_gpu else "cpu"
    assert device is not None, "Device should not be None"
    chains = [
        fk.build_serial_chain_from_urdf(robot_info, ele).to(device=device) for ele in
        [CommonJoints.THUMB_TIP, CommonJoints.INDEX_TIP, CommonJoints.MIDDLE_TIP, CommonJoints.RING_LITTLE_TIP]]
    return chains


def robot_joint_angle_to_coordinate(chains, robot_joint_angles):
    root_position_from_base = -1 * torch.tensor([list(CommonJoints.OR_POSITION)] * robot_joint_angles.shape[0])
    joint_coordinates, joint_count = [root_position_from_base.view(-1, 3).to(device=robot_joint_angles.device)], 0
    for chain_index in range(len(chains)):
        results = chains[chain_index].forward_kinematics(
            robot_joint_angles[:, chain_index * 4:(chain_index + 1) * 4], end_only=False
        )
        for _ in range(len(results)):
            joint_coordinates.append(
                results[CommonJoints.get_joints(is_right=True).LINK_INFOS[joint_count]["name"]].get_matrix()[:, :3, 3]
            )
            joint_count += 1
    joint_coordinates = torch.stack(joint_coordinates, dim=1)
    return joint_coordinates
