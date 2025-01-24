import os

import numpy as np

# ==========================================================================
exp_data_dir = "data/experiment"

# print(os.path.exists(exp_data_dir))
exp_collected = os.path.join(
    exp_data_dir,
    "collect/Mnist+Rand_pix28x28_image(1000+1000)x2_sig2500_4wave_newPD.npz",
)
exp_target = os.path.join(
    exp_data_dir, "target/Mnist+Rand_pix28x28_image(1000+1000)x2.npz"
)
print(os.path.exists(exp_collected))
print(os.path.exists(exp_target))
print("# ==========================================================================")
target = np.load(exp_target)["arr_0"]
collect = np.load(exp_collected)["arr_0"]
print("Target Image Shape: ", target.shape)
print("Collect signal Shape:", collect.shape)
print("# ==========================================================================")


# ==========================================================================
def load_data(target_path, collect_path):
    target = np.load(target_path)["arr_0"]
    collect = np.load(collect_path)["arr_0"]
    X_all = target_diff(target)
    Y_all = collect_diff_and_skip(collect)
    return X_all, Y_all


def target_diff(target):
    # target = np.load(exp_target)["arr_0"]
    # print("Target Image Shape: ", target.shape)
    diff_target = target[::2, :] - target[1::2, :]
    # print("diff_target Shape:", diff_target.shape)
    return diff_target


def collect_diff_and_skip(collect):
    # collect = np.load(exp_collected)["arr_0"]
    diff_collect = collect[::2, :] - collect[1::2, :]
    diff_and_skip_collect = diff_collect[:, ::2]
    return diff_and_skip_collect
    # print("Collect signal Shape:", collect.shape)


def load_mnist(target_path, collect_path):
    X_all, Y_all = load_data(target_path, collect_path)
    Y_mnist = Y_all[:1000, :]
    X_mnist = X_all[:1000, :]
    return X_mnist, Y_mnist


# Y_all, X_all = load_data(exp_target, exp_collected)
X_mnist, Y_mnist = load_mnist(target_path=exp_target, collect_path=exp_collected)
print("Y_mnist Shape:", Y_mnist.shape)
print("X_mnist Shape:", X_mnist.shape)
