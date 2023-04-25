import logging
import os

import hydra
import wandb
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from datasets.arctic_trajectory_dataset import ARCTICTrajectoryDataset
from networks.intention_refining import SiMLPe


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train_simlpe_config")
def main(cfg):
    if cfg.enable_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=OmegaConf.to_container(cfg, resolve=True))

    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        log.info("using GPU ...")
        device = torch.device("cuda:0")
    else:
        log.info("using CPU ...")
        device = torch.device("cpu")

    dct_mat, idct_mat = get_dct_matrix(cfg.motion.input_length_dct)
    dct_mat = torch.tensor(dct_mat).float().to(device=device).unsqueeze(0)
    idct_mat = torch.tensor(idct_mat).float().to(device=device).unsqueeze(0)

    model = SiMLPe(cfg).to(device=device)
    model.train()

    dataset = ARCTICTrajectoryDataset(
        cfg.ARCTIC_dataset_dir,
        is_training=True, data_aug=cfg.data_aug, split=cfg.motion.train_split, use_norm=cfg.motion.normalize_train_data,
        frame_step=cfg.motion.step, condition_num=cfg.motion.input_length, future_num=cfg.motion.train_target_length
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, drop_last=True,
                            shuffle=True, pin_memory=False)

    eval_dataset = ARCTICTrajectoryDataset(
        cfg.ARCTIC_dataset_dir, is_training=False, split=cfg.motion.val_split,
        frame_step=cfg.motion.step, condition_num=cfg.motion.input_length, future_num=cfg.motion.eval_target_length
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                                 num_workers=cfg.num_workers, drop_last=False,
                                 shuffle=False, pin_memory=False)
    log.info(f"train dataset: {len(dataset)} items, eval dataset: {len(eval_dataset)} items.")

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr_max,
                                 weight_decay=cfg.weight_decay)

    if cfg.model_pth != "":
        state_dict = torch.load(cfg.model_pth)
        model.load_state_dict(state_dict, strict=True)

    # ------ training ------- #
    if cfg.enable_wandb:
        wandb.watch(model)

    output_dir = os.path.join(cfg.save_dir, f"output_{cfg.exp_id}")
    os.makedirs(output_dir, exist_ok=True)

    n_iter, fine_list, max_dicts_num = 0, list(), 10
    with tqdm(total=cfg.total_iters) as loop:
        while (n_iter + 1) < cfg.total_iters or cfg.total_iters == 0:
            total_loss, iter_count = 0., 0

            for roi in dataloader:
                motion_input, motion_target = roi["condition_frame"], roi["future_frame"]
                loss, optimizer, current_lr = train_step(
                    cfg, motion_input, motion_target, model, optimizer, n_iter, dct_mat, idct_mat, device
                )

                total_loss += loss
                iter_count += 1
                loop.set_description(f"Iter [{n_iter}]")
                loop.set_postfix(loss=total_loss / iter_count)

                log_dict = {"train_loss": loss, "lr": current_lr}

                if 0 < cfg.save_every and (n_iter + 1) % cfg.save_every == 0:
                    output_path = os.path.join(output_dir, f"model_{n_iter + 1}.pth")
                    torch.save(model, output_path)

                if 0 < cfg.fine_every and (n_iter + 1) % (cfg.fine_every // max_dicts_num) == 0:
                    dicts_dir = os.path.join(output_dir, "dicts")
                    os.makedirs(dicts_dir, exist_ok=True)
                    dict_path = os.path.join(dicts_dir, f"dict_{n_iter + 1}.pth")
                    torch.save(model.state_dict(), dict_path)

                    fine_list.append({
                        "model_path": dict_path,
                        "train_loss": log_dict["train_loss"]
                    })
                    if len(fine_list) == max_dicts_num:
                        fine_list = sorted(fine_list, key=lambda d: d["train_loss"])
                        log.info(fine_list[0]["train_loss"])
                        model.load_state_dict(torch.load(fine_list[0]["model_path"]))
                        model.train()
                        fine_list = list()

                if 0 < cfg.eval_every and (n_iter + 1) % cfg.eval_every == 0:
                    model.eval()

                    st_total_error, lt_total_error = [0.] * len(cfg.metrics), [0.] * len(cfg.metrics)
                    eval_iter_count = 0

                    with torch.no_grad():
                        for eval_roi in eval_dataloader:
                            eval_motion_input = eval_roi["condition_frame"]
                            eval_motion_target = eval_roi["future_frame"]
                            errors = eval_step(
                                cfg, eval_motion_input, eval_motion_target, model, dct_mat, idct_mat, device
                            )
                            st_total_error = [
                                st_total_error[i] + errors["short_term"][i] for i in range(len(st_total_error))
                            ]
                            if "long_term" in errors:
                                lt_total_error = [
                                    lt_total_error[i] + errors["long_term"][i] for i in range(len(lt_total_error))
                                ]
                            eval_iter_count += 1

                    st_avg_error = [st_total_error[i] / eval_iter_count for i in range(len(st_total_error))]
                    lt_avg_error = [lt_total_error[i] / eval_iter_count for i in range(len(lt_total_error))]
                    log.info(
                        f"Eval st_error: {st_avg_error[0]:.6f} {st_avg_error[1]:.6f}"
                        f" lt_error: {lt_avg_error[0]:.6f} {lt_avg_error[1]:.6f}"
                    )
                    log_dict["eval_st_error"] = st_avg_error[0]
                    log_dict["eval_st_relative_error"] = st_avg_error[1]
                    if lt_avg_error[0] != 0.:
                        log_dict["eval_lt_error"] = lt_avg_error[0]
                        log_dict["eval_lt_relative_error"] = lt_avg_error[1]

                    model.train()
                n_iter += 1
                if n_iter == cfg.total_iters and cfg.total_iters != 0:
                    break

                if cfg.enable_wandb:
                    wandb.log(log_dict)

                loop.update()


def get_dct_matrix(mat_size):
    dct_mat = np.eye(mat_size)
    for k in np.arange(mat_size):
        for i in np.arange(mat_size):
            w = np.sqrt(2 / mat_size)
            if k == 0:
                w = np.sqrt(1 / mat_size)
            dct_mat[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / mat_size)
    idct_mat = np.linalg.inv(dct_mat)
    return dct_mat, idct_mat


def update_lr_multistep(curr_iter, lr_max, lr_min, total_iters, opt, factor=0.5):
    lr = lr_max if curr_iter < total_iters * factor else lr_min
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return opt, lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


def feed_net(cfg, motion_input, target_length, model, dct_mat, idct_mat, device):
    _motion_input = torch.matmul(dct_mat[:, :, :cfg.motion.input_length], motion_input.to(device=device))
    motion_pred = model(_motion_input)
    motion_pred = torch.matmul(idct_mat[:, :cfg.motion.input_length, :], motion_pred)

    if cfg.deriv_output:
        offset = motion_input[:, -1:].to(device=device)
        motion_pred = motion_pred[:, :target_length] + offset
    else:
        motion_pred = motion_pred[:, :target_length]
    return motion_pred


def cal_loss(cfg, motion_pred, motion_target, device):
    b, n, c = motion_target.shape
    motion_pred = motion_pred.reshape(b, n, -1, 3)
    motion_target = motion_target.to(device=device).reshape(b, n, -1, 3)
    loss = l2_loss(motion_pred, motion_target)

    if cfg.use_relative_loss:
        motion_pred = motion_pred.reshape(b, n, -1, 3)
        d_motion_pred = gen_velocity(motion_pred)
        motion_gt = motion_target.reshape(b, n, -1, 3)
        d_motion_gt = gen_velocity(motion_gt)
        d_loss = l2_loss(d_motion_pred, d_motion_gt)
        loss = loss + d_loss
    else:
        loss = loss.mean()
    return loss


def l2_loss(motion_pred, motion_target):
    return torch.mean(torch.norm((motion_pred - motion_target).reshape(-1, 3), 2, 1))


def root_related_l2_loss(motion_pred, motion_target):
    motion_pred = motion_pred - motion_pred[:, :, [0], :]
    motion_target = motion_target - motion_target[:, :, [0], :]
    return l2_loss(motion_pred, motion_target)


def cal_error(cfg, motion_pred, motion_target, device):
    sep_x = min(int(0.5 / cfg.motion.step * 25), cfg.motion.eval_target_length) - 1
    errors_dict = {"names": cfg.metrics}

    b, n, c = motion_target.shape
    motion_pred = motion_pred.reshape(b, n, -1, 3)
    motion_target = motion_target.to(device=device).reshape(b, n, -1, 3)

    errors_dict["short_term"] = [
        l2_loss(motion_pred[:, :sep_x], motion_target[:, :sep_x]).item(),
        root_related_l2_loss(motion_pred[:, :sep_x], motion_target[:, :sep_x]).item()
    ]

    if sep_x == cfg.motion.eval_target_length:
        return errors_dict

    errors_dict["long_term"] = [
        l2_loss(motion_pred[:, sep_x:], motion_target[:, sep_x:]).item(),
        root_related_l2_loss(motion_pred[:, sep_x:], motion_target[:, sep_x:]).item(),
    ]

    return errors_dict


def train_step(cfg, motion_input, motion_target, model, optimizer, nb_iter, dct_mat, idct_mat, device):
    motion_pred = feed_net(cfg, motion_input, cfg.motion.train_target_length, model, dct_mat, idct_mat, device)
    loss = cal_loss(cfg, motion_pred, motion_target, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, cfg.lr_max, cfg.lr_min, cfg.total_iters, optimizer)

    return loss.item(), optimizer, current_lr


def eval_step(cfg, motion_input, motion_target, model, dct_mat, idct_mat, device):
    motion_pred = feed_net(cfg, motion_input, cfg.motion.eval_target_length, model, dct_mat, idct_mat, device)
    return cal_error(cfg, motion_pred, motion_target, device)


if __name__ == "__main__":
    main()
