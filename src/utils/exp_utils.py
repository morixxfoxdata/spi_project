import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==========================================================================
# exp_data_dir = "data/experiment"

# # print(os.path.exists(exp_data_dir))
# exp_collected = os.path.join(
#     exp_data_dir,
#     "collect/HP+mosaic+rand_image64+10+500_size8x8_alternate_200x20020240618_collect.npz",
# )
# exp_target = os.path.join(
#     exp_data_dir, "target/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
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
    # print("target shape:", target.shape)
    # print("collect shape:", collect.shape)
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


def load_mnist(target_path, collect_path, pixel):
    X_all, Y_all = load_data(target_path, collect_path)
    if pixel == 28:
        Y_mnist = Y_all[:1000, :]
        X_mnist = X_all[:1000, :]
    elif pixel == 8:
        Y_mnist = Y_all[64:74, :]
        X_mnist = X_all[64:74, :]
    return X_mnist, Y_mnist


def load_hadamard(target_path, collect_path, pixel):
    X_all, Y_all = load_data(target_path, collect_path)
    Y_hadamard = Y_all[: pixel**2, :]
    X_hadamard = X_all[: pixel**2, :]
    return X_hadamard, Y_hadamard


def load_random(target_path, collect_path, pixel):
    X_all, Y_all = load_data(target_path, collect_path)
    if pixel == 28:
        Y_rand = Y_all[:1000, :]
        X_rand = X_all[:1000, :]
    elif pixel == 8:
        Y_rand = Y_all[74:574, :]
        X_rand = X_all[74:574, :]
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


def speckle_pred(target_path, collect_path, pixel=28, alpha=1.0):
    X_rand, Y_rand = load_random(
        target_path=target_path, collect_path=collect_path, pixel=pixel
    )
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


def speckle_pred_8(target_path, collect_path, pixel=8, alpha=1.0):
    X_rand, Y_rand = load_random(target_path, collect_path, pixel=8)
    X_hadamard, Y_hadamard = load_hadamard(target_path, collect_path, pixel=8)
    H_inv = X_hadamard.T // pixel
    print("H_inv:", H_inv.shape)
    S_h = np.matmul(H_inv, Y_hadamard)
    print("S_h:", S_h.shape)
    # n = int(X_hadamard.shape[0])
    delta = Y_rand - np.matmul(X_rand, S_h)
    delta_Ridge = Ridge(alpha=alpha)
    delta_Ridge.fit(X_rand, delta)
    delta_ridge_coef = delta_Ridge.coef_
    predicted_speckle = S_h + delta_ridge_coef.T
    print("X_had, Y_had:", X_hadamard.shape, Y_hadamard.shape)
    print("pred_speckle:", predicted_speckle.shape)
    return predicted_speckle


def image_display(j, xx, yy, model, size=28, num=1):
    # MSEとSSIMを計算
    mse_val = mean_squared_error(xx, yy)
    ssim_val = ssim_score(xx, yy)

    # ターミナルにも表示
    print("MSE =", mse_val)
    print("SSIM=", ssim_val)

    # 保存先のディレクトリを決定
    save_dir = os.path.join("results", f"pix{size}", str(model))
    if not os.path.exists(save_dir):  # 存在しなければ作る
        os.makedirs(save_dir)

    # 図の設定
    fig = plt.figure(figsize=(4, 4 * j))
    # 図全体のタイトルに MSE と SSIM を表示
    fig.suptitle(f"MSE: {mse_val:.4f}  SSIM: {ssim_val:.4f}", fontsize=12)

    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        # それぞれの画像を描画
        ax1.imshow(xx.reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(yy.reshape(size, size), cmap="gray", vmin=-1, vmax=1)

        # 軸の目盛りを非表示
        ax1.axis("off")
        ax2.axis("off")

    # 図を保存
    save_file = os.path.join(save_dir, f"img_{num}.png")
    plt.savefig(save_file)
    plt.close(fig)


def ssim_score(img1, img2):
    score = ssim(img1, img2, data_range=img1.max() - img1.min())
    return score


# X_all, Y_all = load_data(exp_target, exp_collected)
# print("X_all:", X_all.shape)
# print("Y_all:", Y_all.shape)
# speckle_pred_8(exp_target, exp_collected)
