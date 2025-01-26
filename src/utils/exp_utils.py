import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==========================================================================
# exp_data_dir = "../data/experiment"

# # print(os.path.exists(exp_data_dir))
# exp_collected = os.path.join(
#     exp_data_dir,
#     "collect/Mnist+Rand_pix28x28_image(1000+1000)x2_sig2500_4wave_newPD.npz",
# )
# exp_target = os.path.join(
#     exp_data_dir, "target/Mnist+Rand_pix28x28_image(1000+1000)x2.npz"
# )
# print(os.path.exists(exp_collected))
# print(os.path.exists(exp_target))
# print("# ==========================================================================")
# target = np.load(exp_target)["arr_0"]
# collect = np.load(exp_collected)["arr_0"]
# print("Target Image Shape: ", target.shape)
# print("Collect signal Shape:", collect.shape)
# print("# ==========================================================================")


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
    skip_collect = collect[:, ::2]
    diff_and_skip_collect = skip_collect[0::2, :] - skip_collect[1::2, :]
    return diff_and_skip_collect
    # print("Collect signal Shape:", collect.shape)


def load_mnist(target_path, collect_path):
    X_all, Y_all = load_data(target_path, collect_path)
    Y_mnist = Y_all[:1000, :]
    X_mnist = X_all[:1000, :]
    return X_mnist, Y_mnist


def load_random(target_path, collect_path):
    X_all, Y_all = load_data(target_path, collect_path)
    Y_rand = Y_all[:1000, :]
    X_rand = X_all[:1000, :]
    return X_rand, Y_rand


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)


def standardize(y):
    mean = y.mean()
    std = y.std()
    # 万が一 std が 0 になった場合の対策 (すべて同じ値の場合など)
    if std.item() < 1e-12:
        std = torch.tensor(1e-12, device=y.device)
    y_std = (y - mean) / std

    return y_std


def speckle_pred(target_path, collect_path, alpha=1.0):
    X_rand, Y_rand = load_random(target_path=target_path, collect_path=collect_path)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_rand, Y_rand, test_size=0.2, random_state=42
    )
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred_test)
    print("Test MSE:", mse)
    S_est = model.coef_.T
    print("S_est shape:", S_est.shape)
    return S_est


def image_display(j, xx, yy, size=28, num=1):
    print("MSE =", mean_squared_error(xx, yy))
    # print("PSNR =", psnr_calc(xx, yy))
    # print("SSIM =", calculate_average_ssim(xx, yy, num_images=10))
    fig = plt.figure(figsize=(4, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(xx.reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(yy.reshape(size, size), cmap="gray")
    plt.savefig(f"img_{num}.png")


# Y_all, X_all = load_data(exp_target, exp_collected)
# X_mnist, Y_mnist = load_mnist(target_path=exp_target, collect_path=exp_collected)
# print("Y_mnist Shape:", Y_mnist.shape)
# print("X_mnist Shape:", X_mnist.shape)
# Y_mnist_tensor = np_to_torch(Y_mnist)
# print("X_mnist_tensor Shape:", Y_mnist_tensor.shape)
# print("X_mnist:", X_mnist.min(), X_mnist.max())
# X_rand, Y_rand = load_random(target_path=exp_target, collect_path=exp_collected)
# print("Y_rand Shape:", Y_rand.shape)
# print("X_rand Shape:", X_rand.shape)

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_rand, Y_rand, test_size=0.2, random_state=42
# )
