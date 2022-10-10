from tools.torch_canonical.trans import transform_to_canonical
from tools.common import get_key_vectors
from tools.fk_helper import robot_joint_angle_to_coordinate


def energy_loss(human_key_vectors, robot_joint_angles, chains, scale=0.625):
    assert len(robot_joint_angles.shape) == 2, "joints' shape should be BS x 16"

    th_joint_coordinates = robot_joint_angle_to_coordinate(chains=chains, robot_joint_angles=robot_joint_angles)
    robot_joints_position_from_canonical, _ = transform_to_canonical(th_joint_coordinates, is_human=False)

    robot_key_vectors = get_key_vectors(robot_joints_position_from_canonical, is_human=False)  # BS x 10 x 3
    energy = ((human_key_vectors - scale * robot_key_vectors) ** 2).sum(dim=-1).sum(dim=-1).mean()
    return energy
