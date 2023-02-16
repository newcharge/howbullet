def cal_kl_loss(m, v):
    kl = -0.5 * (1 + v - m.pow(2) - v.exp()).sum().clamp(max=0)
    kl /= v.numel()
    return kl


def cal_recon_loss(pred_y, y):
    assert pred_y.shape == y.shape, "The shape of predicted Y should be same as the shape of Y !"
    recon = (y - pred_y).pow(2).mean()
    return recon
