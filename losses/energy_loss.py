from tools.common import get_key_vectors


def cal_loss(
        human_key_vectors, robot_joints_position_from_canonical, robot_joint_angles,
        with_self_collision, collision_epsilon):
    if with_self_collision:
        return energy_loss(
            human_key_vectors=human_key_vectors,
            robot_joints_position_from_canonical=robot_joints_position_from_canonical,
        )
    else:
        return energy_loss_collision(
            human_key_vectors=human_key_vectors,
            robot_joints_position_from_canonical=robot_joints_position_from_canonical,
            robot_joint_angles=robot_joint_angles,
            epsilon=collision_epsilon
        )


def energy_loss(human_key_vectors, robot_joints_position_from_canonical, scale=0.625):
    assert len(robot_joints_position_from_canonical.shape) == 3, "joints' shape should be BS x 21 * 3"

    robot_key_vectors = get_key_vectors(robot_joints_position_from_canonical, is_human=False)  # BS x 10 x 3
    energy = ((human_key_vectors - scale * robot_key_vectors) ** 2).sum(dim=-1).sum(dim=-1).mean()
    return energy


def energy_loss_collision(human_key_vectors, robot_joints_position_from_canonical, robot_joint_angles, epsilon):
    assert len(robot_joints_position_from_canonical.shape) == 3, "joints' shape should be BS x 21 * 3"

    robot_key_vectors = get_key_vectors(robot_joints_position_from_canonical, is_human=False)  # BS x 10 x 3
    cost_1 = 0
    cost_2 = 0
    for human_key_vector, robot_key_vector, robot_joint_angle in zip(human_key_vectors, robot_key_vectors,
                                                                     robot_joint_angles):
        for i in range(0, 10):
            d = ((human_key_vector[i]) ** 2).sum() ** 0.5
            cost_1 += 1 / 2 * _s_fn(d, epsilon=epsilon, index=i) * (
                    (robot_key_vector[i] - _f_fn(d, epsilon=epsilon, index=i) * human_key_vector[i] / d) ** 2).sum()

        cost_2 += 5 * 10 ** (-3) * ((robot_joint_angle ** 2).sum()) ** 0.5  # change 2.5*10**(-3) to 5*10**(-3)
    return (cost_1 + cost_2) / human_key_vectors.shape[0]


def _s_fn(d, epsilon, index):
    if d > epsilon or index <= 3:  # add Palm root to finger
        return 1
    elif d <= epsilon and 4 <= index <= 6:
        return 200
    elif d <= epsilon and 7 <= index <= 9:
        return 400
    else:
        assert False, "error: s(d_i) illegal!!!!!!!!!!"


def _f_fn(d, epsilon, index, beta=1.6):
    if d > epsilon or index <= 3:  # add Palm root to finger
        return beta * d
    elif d <= epsilon and 4 <= index <= 6:
        return 10 ** (-4)
    elif d <= epsilon and 7 <= index <= 9:
        return 3 * 10 ** (-2)
    else:
        assert False, "error: f(d_i) illegal!!!!!!!!!!"
