import torch
import tools.torch_canonical.convert as cvt


def cal_cononical_transform(kp3d, is_human):
    assert len(kp3d.shape) == 3, "kp3d need to be BS x 21 x 3"
    dev = kp3d.device
    bs = kp3d.shape[0]
    kp3d = kp3d.clone().detach()

    # Align root
    tx = kp3d[:, 0, 0]
    ty = kp3d[:, 0, 1]
    tz = kp3d[:, 0, 2]
    # Translation
    T_t = torch.zeros((bs, 3, 4)).to(device=dev)
    T_t[:, 0, 3] = -tx
    T_t[:, 1, 3] = -ty
    T_t[:, 2, 3] = -tz
    T_t[:, 0, 0] = 1
    T_t[:, 1, 1] = 1
    T_t[:, 2, 2] = 1
    # Align index root bone with -y-axis
    y_axis = torch.tensor([[0.0, -1.0, 0.0]]).to(device=dev).expand(bs, 3)
    if is_human:
        v_mrb = cvt.normalize(kp3d[:, 9] - kp3d[:, 0])
    else:
        v_mrb = cvt.normalize(kp3d[:, 11] - kp3d[:, 0])
    R_1 = cvt.get_alignment_mat(v_mrb, y_axis)
    # Align x-y plane along plane spanned by index and middle root bone of the hand
    # after R_1 has been applied to it
    if is_human:
        v_irb = cvt.normalize(kp3d[:, 5] - kp3d[:, 0])
    else:
        v_irb = cvt.normalize(kp3d[:, 6] - kp3d[:, 0])
    normal = cvt.cross(v_mrb, v_irb).view(-1, 1, 3)
    normal_rot = torch.matmul(normal, R_1.transpose(1, 2)).view(-1, 3)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]]).to(dtype=torch.float32, device=dev).expand(bs, 3)
    R_2 = cvt.get_alignment_mat(normal_rot, z_axis)
    # Compute the canonical transform
    T = torch.bmm(R_2, torch.bmm(R_1, T_t))
    return T


def transform_to_canonical(kp3d, is_human=True):
    """Undo global translation and rotation
    """
    normalization_mat = cal_cononical_transform(kp3d, is_human)
    kp3d = cvt.xyz_to_xyz1(kp3d)
    # import pdb
    # pdb.set_trace()
    kp3d_canonical = torch.matmul(normalization_mat.unsqueeze(1), kp3d.unsqueeze(-1))
    kp3d_canonical = kp3d_canonical.squeeze(-1)
    # Pad T from 3x4 mat to 4x4 mat
    normalization_mat = cvt.pad34_to_44(normalization_mat)
    return kp3d_canonical, normalization_mat
