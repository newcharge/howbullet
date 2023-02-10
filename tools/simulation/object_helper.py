import pybullet as p


def load_single_object(obj_path, base_position=None, base_orientation=None, global_scaling=1, from_urdf=True):

    if base_position is None:
        base_position = [0, 0, 0]
    if base_orientation is None:
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])

    if from_urdf:
        load_object_from_urdf(obj_path, base_position, base_orientation, global_scaling)
    else:
        load_object_from_shape(obj_path, base_position, base_orientation, global_scaling)


def load_object_from_urdf(obj_path, base_position, base_orientation, global_scaling):
    obj_id = p.loadURDF(
        obj_path,
        basePosition=base_position,
        baseOrientation=base_orientation,
        globalScaling=global_scaling
    )
    return obj_id


def load_object_from_shape(obj_path, base_position, base_orientation, global_scaling):

    vis_param = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=[0, 0, 0],
        meshScale=[0.001, 0.001, 0.001] * global_scaling  # mm -> m
    )

    coli_param = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        collisionFramePosition=[0, 0, 0],
        meshScale=[0.001, 0.001, 0.001] * global_scaling  # mm -> m
    )

    obj_id = p.createMultiBody(
        baseMass=global_scaling,
        baseCollisionShapeIndex=coli_param,
        baseVisualShapeIndex=vis_param,
        basePosition=base_position,
        baseOrientation=base_orientation,
        useMaximalCoordinates=True
    )

    return obj_id


if __name__ == "__main__":
    # position: [-0.1, 0.2, 0.8]
    # orientation: [0.106318520017424, -0.8584643803487468, -0.039311689853897366, 0.500189834977259]
    pass
