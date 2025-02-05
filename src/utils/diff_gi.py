import os

import numpy as np
import torch
from exp_utils import load_mnist, np_to_torch, speckle_pred
from matplotlib import pyplot as plt

pixel = 28


# ==========================================================================
# DATA _ PATH
# ==========================================================================
exp_data_dir = "data/experiment"
save_dir = "results/"

# print(os.path.exists(exp_data_dir))
if pixel == 28:
    exp_collected = os.path.join(
        exp_data_dir,
        "collect/Mnist+Rand_pix28x28_image(1000+1000)x2_sig2500_4wave_newPD.npz",
    )
    exp_target = os.path.join(
        exp_data_dir, "target/Mnist+Rand_pix28x28_image(1000+1000)x2.npz"
    )
elif pixel == 8:
    exp_collected = os.path.join(
        exp_data_dir,
        "collect/HP+mosaic+rand_image64+10+500_size8x8_alternate_200x20020240618_collect.npz",
    )
    exp_target = os.path.join(
        exp_data_dir, "target/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
    )
region_indices = [0, 1, 2, 3]
S_0 = speckle_pred(
    target_path=exp_target,
    collect_path=exp_collected,
    region_indices=region_indices,
    pixel=pixel,
    alpha=1.0,
)
# print("S_0 shape:", S_0.shape)
X_mnist, Y_mnist = load_mnist(
    target_path=exp_target,
    collect_path=exp_collected,
    pixel=pixel,
    region_indices=region_indices,
)
S_0_tensor = np_to_torch(S_0).float()

X_mnist_tensor = np_to_torch(X_mnist).float()

# (1000, 10000)
Y_mnist_tensor = np_to_torch(Y_mnist).float()

print("S:", S_0_tensor.shape)
print("X:", X_mnist_tensor.shape)
print("Y:", Y_mnist_tensor.shape)

# ====================================
# パラメータ設定
# ====================================
num = 9
N = S_0_tensor.shape[1]  # 測定回数（照明パターン数）→ 10000
# ====================================
# 1. バケット信号の計算
# ====================================
# 【技術用語】
# バケット信号 (Bucket Signal): 対象物を照射した際、全画素の光強度の総和を表す信号。
#
# 各オブジェクト（画像）に対して、各パターンとの内積（要素ごとの積和）を計算します。
# X_mnist_tensor のサイズは [10, 784]、S_0_tensor のサイズは [784, 10000] なので、
# 行列積（dot product）によって、B は [10, 10000] のサイズになります。
B = torch.matmul(X_mnist_tensor, S_0_tensor)
# ====================================
# 2. 参照信号の計算
# ====================================
# 【技術用語】
# 参照信号 (Reference Signal): 各照明パターン自体の全画素（ここでは 784 ピクセル）の輝度和。
#
# S_0_tensor の各パターン（各列）の総和を計算します。結果はサイズ [10000] となります。
R = torch.sum(S_0_tensor, dim=0)  # 参照信号：各パターンの光強度の総和
# ====================================
# 3. 各信号の統計平均の計算
# ====================================
# 各画像ごとのバケット信号の平均（ensemble average）を計算します。
# また、全パターンに対する参照信号の平均も計算します。
avg_B = torch.mean(B, dim=1, keepdim=True)  # サイズ: [10, 1]
avg_R = torch.mean(R)  # スカラー
# ====================================
# 4. 差分ゴーストイメージング (DGI) による再構成
# ====================================
# 【DGI の再構成式】
#   I_DGI = (1/N) * Σ_{i=1}^{N} [B_i - (avg_B/avg_R)*R_i] * P_i
#
# 各画像について、各パターンに対して補正項を計算し、その重み付け和をとります。
#
# weight は各画像・各パターンに対して、
#    weight[j, i] = B[j, i] - (avg_B[j] / avg_R) * R[i]
# と定義され、これを用いて再構成画像を求めます。
weight = B - (avg_B / avg_R) * R  # 自動的にブロードキャストされ、サイズは [10, 10000]

# 各画像の再構成結果は、照明パターンとの線形結合で求められます。
# ここでは、weight と S_0_tensor の積を計算します。
# 注意: S_0_tensor のサイズは [784, 10000] であるため、weight (サイズ [10, 10000]) との
# 行列積を行う場合、以下のように S_0_tensor.T（サイズ [10000, 784]）を用います。
I_rec = torch.matmul(weight, S_0_tensor.T) / N  # 再構成画像のサイズは [10, 784]

# ====================================
# 5. 結果の可視化（例として最初の画像を表示）
# ====================================
# MNIST は通常 28×28 の画像なので、再構成画像も 28×28 に変形して表示します。

original_img = X_mnist_tensor[num].reshape(28, 28).detach().cpu().numpy()
reconstructed_img = I_rec[num].reshape(28, 28).detach().cpu().numpy()

np.savez(f"{exp_data_dir}/pix{pixel}_{num}_dgi.npz", reconstructed_img)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Original MNIST Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap="gray")
plt.title("Reconstructed Image (DGI)")
plt.axis("off")

plt.tight_layout()
plt.show()
