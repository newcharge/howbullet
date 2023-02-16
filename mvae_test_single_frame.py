import hydra
import torch
from torch.utils.data import DataLoader

from datasets.trajectory_dataset import TrajectoryDataset, mapping_to_simple
from mvae_train import get_feed_input, feed_net

from tools.plot_helper import get_plot_graph, Skeleton, plot_pts, Color


@hydra.main(version_base=None, config_path="configs", config_name="mvae_config")
def main(cfg):
    net = torch.load(cfg.model_path)
    net.cpu().eval()

    joint_size = 21
    validation_dataset = TrajectoryDataset(
        cfg.dataset_dir, is_training=False, claimed_sequences=cfg.val_sequences,
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True)
    for roi in validation_dataloader:
        with torch.no_grad():
            x_input, c_input = get_feed_input(net, roi, cfg.normalize)
            output, _, _, _ = feed_net(net, x_input, c_input, torch.device("cpu"), cfg.normalize)

            output_shape = (-1, joint_size, 3)
            output = output.view(output_shape)
            former = mapping_to_simple(roi["condition_frame"][0])
            pred, gt = mapping_to_simple(output[0]), mapping_to_simple(roi["future_frame"][0])
            pred_pcd, pred_lines = get_plot_graph(pred, Skeleton.HUMAN, Color.BLUE, uniform_color=True)
            former_pcd, former_lines = get_plot_graph(former, Skeleton.HUMAN, Color.GREEN, uniform_color=True)
            gt_pcd, gt_lines = get_plot_graph(gt, Skeleton.HUMAN, Color.RED, uniform_color=True)
            plot_pts([pred_pcd, pred_lines, gt_pcd, gt_lines, former_pcd, former_lines])


if __name__ == "__main__":
    main()
