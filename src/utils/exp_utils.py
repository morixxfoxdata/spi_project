import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
# print(os.path.exists(exp_collected))
# print(os.path.exists(exp_target))
# print("# ==========================================================================")
# target = np.load(exp_target)["arr_0"]
# collect = np.load(exp_collected)["arr_0"]
# print("Target Image Shape: ", target.shape)
# print("Collect signal Shape:", collect.shape)
# print("# ==========================================================================")


# ==========================================================================
# LOADING TOOLS
# ==========================================================================
def load_data(target_path, collect_path, pixel, region_indices):
    target = np.load(target_path)["arr_0"]
    collect = np.load(collect_path)["arr_0"]
    # print("target shape:", target.shape)
    # print("collect shape:", collect.shape)
    X_all = target_diff(target)
    Y_all = collect_diff_and_skip(collect)
    if pixel == 28:
        Y_all = wavelength_selection(Y_all, region_indices=region_indices)
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


def load_mnist(target_path, collect_path, pixel, region_indices):
    X_all, Y_all = load_data(
        target_path, collect_path, pixel, region_indices=region_indices
    )
    if pixel == 28:
        Y_mnist = Y_all[990:1000, :]
        X_mnist = X_all[990:1000, :]
    elif pixel == 8:
        Y_mnist = Y_all[64:74, :]
        X_mnist = X_all[64:74, :]
    return X_mnist, Y_mnist


def load_hadamard(target_path, collect_path, pixel, region_indices):
    X_all, Y_all = load_data(target_path, collect_path, pixel, region_indices)
    Y_hadamard = Y_all[: pixel**2, :]
    X_hadamard = X_all[: pixel**2, :]
    return X_hadamard, Y_hadamard


def load_random(target_path, collect_path, pixel, region_indices):
    X_all, Y_all = load_data(
        target_path, collect_path, pixel, region_indices=region_indices
    )
    if pixel == 28:
        Y_rand = Y_all[1000:, :]
        X_rand = X_all[1000:, :]
    elif pixel == 8:
        Y_rand = Y_all[74:574, :]
        X_rand = X_all[74:574, :]
    else:
        # pixelの値が想定外の場合はエラーを投げる
        raise ValueError(f"Unexpected value for pixel: {pixel}")

    return X_rand, Y_rand


def wavelength_selection(data, region_indices):
    block_width = 2500
    # region_indicesが単一のintの場合はリストに変換して扱う
    if isinstance(region_indices, int):
        region_indices = [region_indices]
    # インデックスの範囲チェック
    for idx in region_indices:
        if not (0 <= idx <= 3):
            raise ValueError("region_index は0〜3の範囲で指定してください。")

    # 指定された各ブロックを取得して結合
    sub_blocks = []
    for idx in region_indices:
        start_col = idx * block_width
        end_col = start_col + block_width
        sub_blocks.append(data[:, start_col:end_col])

    # 列方向(axis=1)に結合
    return np.concatenate(sub_blocks, axis=1)


# ==========================================================================
# UTILS
# ==========================================================================


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


# ==========================================================================
# SPECKLE PREDICTION
# ==========================================================================


def speckle_pred(target_path, collect_path, region_indices, pixel=28, alpha=1.0):
    X_rand, Y_rand = load_random(
        target_path=target_path,
        collect_path=collect_path,
        pixel=pixel,
        region_indices=region_indices,
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


def speckle_pred_8(target_path, collect_path, region_indices, pixel, alpha=1.0):
    X_rand_, Y_rand_ = load_random(target_path, collect_path, pixel, region_indices)
    X_hadamard, Y_hadamard = load_hadamard(
        target_path, collect_path, pixel, region_indices
    )
    H_inv = X_hadamard.T // pixel
    print("H_inv:", H_inv.shape)
    S_h = np.matmul(H_inv, Y_hadamard)
    print("S_h:", S_h.shape)
    # n = int(X_hadamard.shape[0])
    delta = Y_rand_ - np.matmul(X_rand_, S_h)
    delta_Ridge = Ridge(alpha=alpha)
    delta_Ridge.fit(X_rand_, delta)
    delta_ridge_coef = delta_Ridge.coef_
    predicted_speckle = S_h + delta_ridge_coef.T
    print("X_had, Y_had:", X_hadamard.shape, Y_hadamard.shape)
    print("pred_speckle:", predicted_speckle.shape)
    return predicted_speckle


# ==========================================================================
# IMAGE PLOT and SAVE
# ==========================================================================
def image_display(j, xx, yy, model, epochs, lr, size=28, num=1):
    # MSEとSSIMを計算
    mse_val = mean_squared_error(xx, yy)
    ssim_val = ssim_score(xx, yy)

    # ターミナルにも表示
    print("MSE =", mse_val)
    print("SSIM=", ssim_val)

    # 保存先のディレクトリを決定
    save_dir = os.path.join("../results", f"pix{size}", str(model))
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
    save_file = os.path.join(save_dir, f"img_{num}_iter{epochs}_lr{lr}.png")
    plt.savefig(save_file)
    print(f"saved! {save_file}")
    # plt.close(fig)


# ==========================================================================
# OTHERS
# ==========================================================================
def ssim_score(img1, img2):
    score = ssim(img1, img2, data_range=img1.max() - img1.min())
    return score


# X_all, Y_all = load_data(exp_target, exp_collected)
# print("X_all:", X_all.shape)
# print("Y_all:", Y_all.shape)
# speckle_pred_8(exp_target, exp_collected)
# S_estimated = speckle_pred(
#     exp_target, exp_collected, region_indices=[0, 1, 2, 3], pixel=28, alpha=1.0
# )
# print(S_estimated.shape)
# plt.imshow(S_estimated[:, 0].reshape((28, -1)))
# plt.show()
