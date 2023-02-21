import logging

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from datasets.trajectory_dataset_step import TrajectoryDataset
from losses.mvae_loss import cal_recon_loss, cal_kl_loss
from networks.pose_refining import PoseMixtureVAE

log = logging.getLogger(__name__)


def get_feed_input(net, roi, normalize):
    flatten_future = roi["future_frame"].flatten(start_dim=1, end_dim=2)
    flatten_condition = roi["condition_frame"].flatten(start_dim=1, end_dim=2)
    x_input = flatten_future
    c_input = flatten_condition
    if normalize:
        x_input = net.normalize(x_input)
        c_input = net.normalize(c_input)
    return x_input, c_input


def feed_net(net, x_input, c_input, device, normalize):
    x_input = x_input.to(device=device)
    c_input = c_input.to(device=device)
    output, m, v = net(x_input, c_input)
    if normalize:
        former_output = net.denormalize(output)
    else:
        former_output = output
    return former_output, output, m, v


def cal_losses(output, m, v, gt, beta, frame_size):
    output_shape = (-1, frame_size)
    output = output.view(output_shape)
    recon_loss = cal_recon_loss(output.cpu(), gt)
    kl_loss = cal_kl_loss(m, v)
    loss = recon_loss + beta * kl_loss
    return kl_loss, recon_loss, loss, output.shape[0]


@hydra.main(version_base=None, config_path="configs", config_name="mvae_config")
def main(cfg):
    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=OmegaConf.to_container(cfg, resolve=True))

    frame_size = 21 * 3
    if torch.cuda.is_available():
        log.info("using GPU ...")
        device = torch.device("cuda:0")
    else:
        log.info("using CPU ...")
        device = torch.device("cpu")

    train_dataset = TrajectoryDataset(
        cfg.dataset_dir, is_training=True, claimed_sequences=cfg.train_sequences,
        step=cfg.data_frame_step
    )
    train_data_max, train_data_min = train_dataset.get_max_min()
    validation_dataset = TrajectoryDataset(
        cfg.dataset_dir, is_training=False, claimed_sequences=cfg.val_sequences,
        data_max=train_data_max, data_min=train_data_min, step=cfg.data_frame_step
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False)
    net = PoseMixtureVAE(
        normalization={"max": train_data_max, "min": train_data_min}, frame_size=frame_size
    ).to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    wandb.watch(net)

    if cfg.beta_increment:
        beta = np.linspace(0, cfg.max_kl_beta, cfg.epoch_num, endpoint=False)
    else:
        beta = np.array([cfg.max_kl_beta] * cfg.epoch_num)
    train_loop = tqdm(range(cfg.epoch_num))
    log_dict = dict()
    for i in train_loop:
        net.train()
        total_recon_loss = 0
        total_kl_loss = 0
        total_loss = 0
        passed_num = 0
        for roi in train_dataloader:
            x_input, c_input = get_feed_input(net, roi, cfg.normalize)
            _, output, m, v = feed_net(net, x_input, c_input, device, cfg.normalize)
            kl_loss, recon_loss, loss, batch_size = cal_losses(output, m, v, x_input, beta[i], frame_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_loss += loss.item() * batch_size
            passed_num += batch_size
        log_dict["train_recon"] = total_recon_loss / passed_num
        log_dict["train_kl"] = total_kl_loss / passed_num
        log_dict["train_loss"] = total_loss / passed_num
        train_loop.set_description(f"Epoch [{i + 1}/{cfg.epoch_num}]")
        train_loop.set_postfix(loss=log_dict["train_loss"])
        if 0 < cfg.eval_every_epoch and (i + 1) % cfg.eval_every_epoch == 0:
            with torch.no_grad():
                total_recon_loss = 0
                total_kl_loss = 0
                total_loss = 0
                passed_num = 0
                eval_loop = tqdm(validation_dataloader)
                for roi in eval_loop:
                    x_input, c_input = get_feed_input(net, roi, cfg.normalize)
                    _, output, m, v = feed_net(net, x_input, c_input, device, cfg.normalize)
                    kl_loss, recon_loss, loss, batch_size = cal_losses(
                        output, m, v, x_input, cfg.max_kl_beta, frame_size
                    )

                    total_recon_loss += recon_loss.item() * batch_size
                    total_kl_loss += kl_loss.item() * batch_size
                    total_loss += loss.item() * batch_size
                    passed_num += batch_size
                log_dict["val_recon"] = total_recon_loss / passed_num
                log_dict["val_kl"] = total_kl_loss / passed_num
                log_dict["val_loss"] = total_loss / passed_num
                eval_loop.set_postfix(loss=log_dict["val_loss"])
        # logging
        wandb.log(log_dict)

    # dump results
    torch.save(net, cfg.output_model_path)


if __name__ == "__main__":
    main()
