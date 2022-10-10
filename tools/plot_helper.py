import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os


class Color:
    RED = [1, 0, 0]
    GREEN = [0, 1, 0]
    BLUE = [0, 0, 1]


class Skeleton:
    HUMAN = np.array([
        [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
        [5, 9], [9, 13], [13, 17],
        [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20]
    ])
    ROBOT = np.array([
        [0, 1], [0, 6], [0, 11], [0, 16],
        [6, 11], [11, 16],
        [1, 2], [2, 3], [3, 4], [4, 5],
        [6, 7], [7, 8], [8, 9], [9, 10],
        [11, 12], [12, 13], [13, 14], [14, 15],
        [16, 17], [17, 18], [18, 19], [19, 20]
    ])


def get_plot_points(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(color)
    return point_cloud


def get_plot_lines(lines, points, color):
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.paint_uniform_color(color)
    return line_set


def get_plot_graph(points, lines, color):
    hand_point_cloud = get_plot_points(points, color)
    hand_line_set = get_plot_lines(lines, points, color)
    return hand_point_cloud, hand_line_set


def plot_energy(result_dir):
    train_energy_per_iter = np.loadtxt(os.path.join(result_dir, "train_energy_per_iter.txt"))
    train_energy_per_epoch = np.loadtxt(os.path.join(result_dir, "train_energy_per_epoch.txt"))
    val_energy_per_epoch = np.loadtxt(os.path.join(result_dir, "val_energy_per_epoch.txt"))
    plt.clf()
    _, ax1 = plt.subplots()
    t_train_loss_per_iter = [i + 1 for i in range(len(train_energy_per_iter))]
    ax1.plot(t_train_loss_per_iter, train_energy_per_iter)
    ax1.set_xlabel("train loss per iter")
    plt.show()
    plt.clf()
    _, ax2 = plt.subplots()
    t_train_loss_per_epoch = [i + 1 for i in range(len(train_energy_per_epoch))]
    t_val_loss_per_epoch = [i + 1 for i in range(len(val_energy_per_epoch))]
    ax2.plot(t_train_loss_per_epoch, train_energy_per_epoch, label="train")
    ax2.plot(t_val_loss_per_epoch, val_energy_per_epoch, label="val")
    ax2.set_xlabel("loss per epoch")
    ax2.legend()
    plt.show()
    plt.close()
