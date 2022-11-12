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
def s(d,ebsilon,index):
    if d>ebsilon or index<=3: #add Palm root to finger
        return 1
    elif d<=ebsilon and index>=4 and index<=6:
        return 200
    elif  d<=ebsilon and index>=7 and index<=9:
        return 400
    else:
        print("error: s(d_i) illegal!!!!!!!!!!")
        assert(0)
def f(d,ebsilon,index,beta=1.6):
    if d>ebsilon or index<=3:  #add Palm root to finger
        return beta*d
    elif d<=ebsilon and index>=4 and index<=6:
        return 10**(-4)
    elif  d<=ebsilon and index>=7 and index<=9:
        return 3*10**(-2)
    else:
        print("error: f(d_i) illegal!!!!!!!!!!")
        assert(0)     


def anergy_loss_collision(human_key_vectors, robot_joint_angles, chains,e):
    assert len(robot_joint_angles.shape) == 2, "joints' shape should be BS x 16"
    th_joint_coordinates = robot_joint_angle_to_coordinate(chains=chains, robot_joint_angles=robot_joint_angles)
    robot_joints_position_from_canonical, _ = transform_to_canonical(th_joint_coordinates, is_human=False)
    robot_key_vectors = get_key_vectors(robot_joints_position_from_canonical, is_human=False)  # BS x 10 x 3
    cost_1=0
    cost_2=0
    for human_key_vector,robot_key_vector,robot_joint_angle in zip(human_key_vectors,robot_key_vectors,robot_joint_angles):
        for i in range(0,10):
            d=((human_key_vector[i])**2).sum()**0.5            
            cost_1+=1/2*s(d,ebsilon=e,index=i)*((robot_key_vector[i]-f(d,ebsilon=e,index=i)*human_key_vector[i]/d)**2).sum()

        cost_2+=5*10**(-3)*((robot_joint_angle**2).sum())**0.5#  change 2.5*10**(-3) to 5*10**(-3)
    return (cost_1+cost_2)/human_key_vectors.shape[0]