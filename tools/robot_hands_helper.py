import time

import pybullet as p
import pybullet_data
import os


class LinkState:
    def __init__(self, state):
        self.world_position = state[0]
        self.world_rotation = state[1]
        self.local_inertial_frame_position = state[2]
        self.local_inertial_frame_rotation = state[3]
        self.world_frame_position = state[4]
        self.world_frame_rotation = state[5]
        # lines below should only be used when computing velocity
        # self.world_linear_velocity = state[6]
        # self.world_angular_velocity = state[7]

    def get_position_rotation(self):
        link_from_world_position, link_from_world_rotation = self.world_position, self.world_rotation
        return link_from_world_position, link_from_world_rotation

    @classmethod
    def link_state(cls, robot_id, link_id):
        return cls(p.getLinkState(robot_id, link_id))


class JointInfo:
    def __init__(self, robot_id, info, is_right=True):
        self.robot_id = robot_id
        self.is_right = is_right
        self.index = info[0]
        self.name = info[1]
        self.type = info[2]
        self.first_position_index = info[3]
        self.first_velocity_index = info[4]
        self.flags = info[5]
        self.damping = info[6]
        self.friction = info[7]
        self.lower_limit = info[8]
        self.upper_limit = info[9]
        self.max_force = info[10]
        self.max_velocity = info[11]
        self.child_link_name = info[12]
        self.axis = info[13]
        self.parent_frame_position = info[14]
        self.parent_frame_rotation = info[15]
        self.parent_index = info[16]

    def get_position_rotation(self):
        joint_from_frame_position, joint_from_frame_rotation = self.parent_frame_position, self.parent_frame_rotation
        if self.parent_index == -1:
            frame_from_world_position, frame_from_world_rotation = p.getBasePositionAndOrientation(self.robot_id)
        else:
            frame_from_world_position, frame_from_world_rotation = LinkState \
                .link_state(self.robot_id, self.parent_index) \
                .get_position_rotation()
        joint_from_world_position, joint_from_world_rotation = p.multiplyTransforms(
            frame_from_world_position, frame_from_world_rotation,
            joint_from_frame_position, joint_from_frame_rotation
        )
        return joint_from_world_position, joint_from_world_rotation

    def get_root_related_position_rotation(self):
        joint_from_world_position, joint_from_world_rotation = self.get_position_rotation()
        base_from_world_position, base_from_world_rotation = p.getBasePositionAndOrientation(self.robot_id)
        if self.is_right:
            base_from_root_position, base_from_root_rotation = RightJoints.OR_POSITION, RightJoints.OR_ROTATION
        else:
            base_from_root_position, base_from_root_rotation = LeftJoints.OR_POSITION, LeftJoints.OR_ROTATION
        world_from_base_position, world_from_base_rotation = p.invertTransform(
            base_from_world_position, base_from_world_rotation
        )
        joint_from_base_position, joint_from_base_rotation = p.multiplyTransforms(
            world_from_base_position, world_from_base_rotation,
            joint_from_world_position, joint_from_world_rotation
        )
        joint_from_root_position, joint_from_root_rotation = p.multiplyTransforms(
            base_from_root_position, base_from_root_rotation,
            joint_from_base_position, joint_from_base_rotation
        )
        return joint_from_root_position, joint_from_root_rotation

    @classmethod
    def joint_info(cls, robot_id, joint_id):
        return cls(robot_id, p.getJointInfo(robot_id, joint_id))


class JointState:
    def __init__(self, state):
        self.position = state[0]  # position of joint angle
        self.velocity = state[1]
        self.reaction_forces = state[2]
        self.joint_motor_torque = state[3]

    @classmethod
    def joint_state(cls, robot_id, joint_id):
        return cls(p.getJointState(robot_id, joint_id))


class CommonJoints:
    THUMB, INDEX, MIDDLE, RING_LITTLE = None, None, None, None
    TIP, DIP, PIP_U, PIP_D, MCP = None, None, None, None, None
    LINK_INFOS = None
    OR_POSITION = (0, 0, 0.095)
    OR_ROTATION = (0, 0, 0, 1)
    T_OR = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.095],
        [0., 0., 0., 1.]
    ]
    THUMB_TIP = "link_15.0_tip"
    INDEX_TIP = "link_3.0_tip"
    MIDDLE_TIP = "link_7.0_tip"
    RING_LITTLE_TIP = "link_11.0_tip"

    @classmethod
    def get_sequence(cls):
        return [i + 1 for i in [-1] + cls.THUMB + cls.INDEX + cls.MIDDLE + cls.RING_LITTLE]

    @classmethod
    def get_mapping_to_sim(cls):
        mapping_to_sim = list()
        for finger in [cls.THUMB, cls.INDEX, cls.MIDDLE, cls.RING_LITTLE]:
            finger_without_tip = finger[:-1]
            mapping_to_sim += list(map(lambda x: x // 5 * 4 + x % 5, finger_without_tip))
        return mapping_to_sim

    @staticmethod
    def get_joints(is_right=True):
        if is_right:
            return RightJoints
        else:
            return LeftJoints


class LeftJoints(CommonJoints):
    THUMB = [15, 16, 17, 18, 19]
    INDEX = [10, 11, 12, 13, 14]
    MIDDLE = [5, 6, 7, 8, 9]
    RING_LITTLE = [0, 1, 2, 3, 4]
    TIP = [19, 14, 9, 4]
    DIP = [18, 13, 8, 3]
    PIP_U = [17, 12, 7, 2]
    PIP_D = [16, 11, 6, 1]
    MCP = [15, 10, 5, 0]
    LINK_INFOS = [
        "link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip",
        "link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip",
        "link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip",
        "link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip",
    ]


class RightJoints(CommonJoints):
    THUMB = [15, 16, 17, 18, 19]
    INDEX = [0, 1, 2, 3, 4]
    MIDDLE = [5, 6, 7, 8, 9]
    RING_LITTLE = [10, 11, 12, 13, 14]
    TIP = [19, 4, 9, 14]
    DIP = [18, 3, 8, 13]
    PIP_U = [17, 2, 7, 12]
    PIP_D = [16, 1, 6, 11]
    MCP = [15, 0, 5, 10]
    LINK_INFOS = [
        {'name': 'link_12.0', 'lower_bound': 0.263, 'upper_bound': 1.396},
        {'name': 'link_13.0', 'lower_bound': -0.105, 'upper_bound': 1.163},
        {'name': 'link_14.0', 'lower_bound': -0.189, 'upper_bound': 1.644},
        {'name': 'link_15.0', 'lower_bound': -0.162, 'upper_bound': 1.719},
        {'name': 'link_15.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0},
        {'name': 'link_0.0', 'lower_bound': -0.47, 'upper_bound': 0.47},
        {'name': 'link_1.0', 'lower_bound': -0.196, 'upper_bound': 1.61},
        {'name': 'link_2.0', 'lower_bound': -0.174, 'upper_bound': 1.709},
        {'name': 'link_3.0', 'lower_bound': -0.227, 'upper_bound': 1.618},
        {'name': 'link_3.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0},
        {'name': 'link_4.0', 'lower_bound': -0.47, 'upper_bound': 0.47},
        {'name': 'link_5.0', 'lower_bound': -0.196, 'upper_bound': 1.61},
        {'name': 'link_6.0', 'lower_bound': -0.174, 'upper_bound': 1.709},
        {'name': 'link_7.0', 'lower_bound': -0.227, 'upper_bound': 1.618},
        {'name': 'link_7.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0},
        {'name': 'link_8.0', 'lower_bound': -0.47, 'upper_bound': 0.47},
        {'name': 'link_9.0', 'lower_bound': -0.196, 'upper_bound': 1.61},
        {'name': 'link_10.0', 'lower_bound': -0.174, 'upper_bound': 1.709},
        {'name': 'link_11.0', 'lower_bound': -0.227, 'upper_bound': 1.618},
        {'name': 'link_11.0_tip', 'lower_bound': 0.0, 'upper_bound': -1.0}
    ]


if __name__ == "__main__":
    physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setGravity(0, 0, 0)
    # p.setRealTimeSimulation(1)

    PLANE_ID = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    ROBOT_ID = p.loadURDF(
        os.path.join("..", "robots", "allegro_hand_description", "allegro_hand_description_right.urdf"))
    # set the center of mass frame (load URDF sets base line frame)
    p.resetBasePositionAndOrientation(ROBOT_ID, [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))

    print("=" * 20)
    print(p.getNumJoints(ROBOT_ID))
    for joint_index in range(p.getNumJoints(ROBOT_ID)):
        print(JointState.joint_state(ROBOT_ID, joint_index))
        print(
            JointInfo.joint_info(ROBOT_ID, joint_index).index,
            JointInfo.joint_info(ROBOT_ID, joint_index).name,
            {
                "name": JointInfo.joint_info(ROBOT_ID, joint_index).child_link_name,
                "lower_bound": JointInfo.joint_info(ROBOT_ID, joint_index).lower_limit,
                "upper_bound": JointInfo.joint_info(ROBOT_ID, joint_index).upper_limit
            }
        )
        print()
    print("=" * 20)

    # 可以使用的关节
    movable_joint_ids = [i for i in range(p.getNumJoints(ROBOT_ID))
                         if i not in RightJoints.TIP]
    joint_position = [1.1, 0.6, 0.6, 0.4, 0, 1.2, 0.6, 0.4, 0, 0, 0.2, 0, -0.6, 0, 0.2, 0]
    print(movable_joint_ids)

    cube_pos, cube_orn = None, None
    joint_from_root_position = None

    time_step = 1 / 240
    point_id_list = list()
    while True:
        if not p.isConnected():
            break
        try:
            p.stepSimulation()

            root_position, root_rotation = p.getBasePositionAndOrientation(ROBOT_ID)
            cube_pos, cube_orn = root_position, root_rotation
            # print joints
            joint_points = [root_position] + [
                tuple(JointInfo.joint_info(ROBOT_ID, i).get_position_rotation())[0]
                for i in range(p.getNumJoints(ROBOT_ID))
            ]
            point_id_list.append(p.addUserDebugPoints(joint_points, [(255, 0, 0)] * (p.getNumJoints(ROBOT_ID) + 1), 3))

            joint_related_position = [(0, 0, 0)] + [
                tuple(JointInfo.joint_info(ROBOT_ID, i).get_root_related_position_rotation())[0]
                for i in range(p.getNumJoints(ROBOT_ID))
            ]
            joint_from_root_position = joint_related_position

            # p.getCameraImage(480, 320)

            target_v = 0
            max_force = 5
            p.setJointMotorControlArray(
                bodyUniqueId=ROBOT_ID,
                jointIndices=movable_joint_ids,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0, 1.2, 0.6, 0.4, 0, 0, 0.2, 0, -0.6, 0, 0.2, 0, 1.1, 0.6, 0.6, 0.4],
                # targetPositions=[0] * 16,
                targetVelocities=[target_v for _ in movable_joint_ids],
                forces=[max_force for _ in movable_joint_ids]
            )

            time.sleep(time_step)
            for point_id in point_id_list:
                p.removeUserDebugItem(point_id)
            point_id_list = list()
        except p.error as e:
            print(e)
            pass
    import numpy as np
    np.savetxt("../joints.txt", np.array(joint_from_root_position))
