import torch
from torch.utils.data import DataLoader
from networks.retargeting_mlp import RetargetingMLP
from datasets.human_hand_dataset import HumanHandDataset
from datasets.common import split_dataset
from losses.energy_loss import cal_loss
import tools.fk_helper as fk
import tqdm
import hydra
import wandb
import logging
from omegaconf import OmegaConf


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):

    wandb.init(project="test-project", entity="howbullet", config=OmegaConf.to_container(cfg, resolve=True))

    if torch.cuda.is_available():
        log.info("using GPU ...")
        device = torch.device("cuda:0")
        chains = fk.get_chains(
            cfg.robotic_hand_urdf_path, use_gpu=True, device=device
        )
    else:
        log.info("using CPU ...")
        device = torch.device("cpu")
        chains = fk.get_chains(cfg.robotic_hand_urdf_path, use_gpu=False)

    train_val_dataset, test_dataset = split_dataset(HumanHandDataset(cfg.hand_dataset_dir), keep=cfg.train_keep)
    train_dataset, validation_dataset = split_dataset(train_val_dataset, keep=cfg.train_keep)
    torch.save(test_dataset, cfg.output_test_dataset_path)
    log.info(f"{len(train_dataset)} {len(validation_dataset)} {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False)

    net = RetargetingMLP(chains=chains).to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    wandb.watch(net)

    for i in range(cfg.epoch_num):
        net.train()
        loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        total_energy = 0
        passed_num = 0
        for _, roi in loop:
            net_input = roi["xyz_input"].to(device=device)
            human_key_vectors = roi["key_vectors"].to(device=device)
            robot_joint_angles, _, robot_joints_position_from_canonical = net(net_input)
            loss = cal_loss(
                human_key_vectors=human_key_vectors,
                robot_joints_position_from_canonical=robot_joints_position_from_canonical,
                robot_joint_angles=robot_joint_angles,
                with_self_collision=cfg.with_self_collision,
                collision_epsilon=cfg.collision_epsilon
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_energy += loss.item() * net_input.shape[0]
            passed_num += net_input.shape[0]
            loop.set_description(f"Epoch [{i + 1}/{cfg.epoch_num}]")
            loop.set_postfix(loss=total_energy / passed_num)
            # wandb.log({"train_energy_per_iter": loss.item()})
        train_energy = total_energy / passed_num
        net.eval()
        with torch.no_grad():
            loop = tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader))
            total_energy = 0
            passed_num = 0
            for _, roi in loop:
                net_input = roi["xyz_input"].to(device=device)
                human_key_vectors = roi["key_vectors"].to(device=device)
                robot_joint_angles, _, robot_joints_position_from_canonical = net(net_input)
                loss = cal_loss(
                    human_key_vectors=human_key_vectors,
                    robot_joints_position_from_canonical=robot_joints_position_from_canonical,
                    robot_joint_angles=robot_joint_angles,
                    with_self_collision=cfg.with_self_collision,
                    collision_epsilon=cfg.collision_epsilon
                )
                total_energy += loss.item() * net_input.shape[0]
                passed_num += net_input.shape[0]
                loop.set_description(f"Epoch [{i + 1}/{cfg.epoch_num}]")
                loop.set_postfix(loss=total_energy / passed_num)
            val_energy = total_energy / passed_num

        # logging
        wandb.log({"train_energy": train_energy, "val_energy": val_energy})

    # dump results
    torch.save(net, cfg.output_model_path)


if __name__ == "__main__":
    main()
