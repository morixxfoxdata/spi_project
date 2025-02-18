import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.GIDC import GIDC28
from utils.exp_utils import (
    image_display,
    load_mnist,
    np_to_torch,
    speckle_pred_inv,
    total_variation_loss_v2,
)

seed = 42
np.random.seed(seed)

# PyTorchのCPU用シードを固定
torch.manual_seed(seed)

# PyTorchのGPU用シードを固定（複数GPUがある場合は全てに対して設定）
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# cuDNNの非決定性を防ぐための設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================================================
# PIXELS
# ==========================================================================
pixel = 28
num = 1

# ==========================================================================
# DATA _ PATH
# ==========================================================================
exp_data_dir = "../data/experiment"
save_dir = "../results/"

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

# ※実際には既に与えられたデータを使用してください。
num_images = 1000  # 画像枚数
num_pixels = 784  # 1枚の画像のピクセル数（例：28×28）
num_patterns = 10000  # 照明パターン枚数

learning_rate = 0.04
num_epochs = 5000
TV_strength = 3e-9

X_mnist, Y_mnist = load_mnist(
    target_path=exp_target,
    collect_path=exp_collected,
    pixel=pixel,
    region_indices=region_indices,
)
S_0 = speckle_pred_inv(
    target_path=exp_target,
    collect_path=exp_collected,
    region_indices=region_indices,
    pixel=pixel,
)
print("X_mnist, Y_mnist, S_0 shape:", X_mnist.shape, Y_mnist.shape, S_0.shape)

# CUDAが使えるかどうかの判定
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

S_0_tensor = np_to_torch(S_0).float().to(device)
X_mnist_tensor = np_to_torch(X_mnist).float()
Y_mnist_tensor = np_to_torch(Y_mnist).float()
S_0_pinv = np.linalg.pinv(S_0)
rec_mnist = np.dot(Y_mnist, S_0_pinv)
rec_tensor = np_to_torch(rec_mnist).float()

model = GIDC28(kernel_size=7, name="GIDC_tv_sch").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習率を徐々に下げるためのスケジューラ設定（指数減衰: 各エポックごとに学習率を gamma 倍に）
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

criterion = nn.MSELoss()

rec = rec_tensor[num].reshape((1, 1, 28, 28)).to(device)
y_ = Y_mnist_tensor[num].to(device)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(rec)
    Y_dash = torch.mm(output.reshape(1, 784), S_0_tensor)
    tv = total_variation_loss_v2(output, tv_strength=TV_strength)
    loss = criterion(Y_dash, y_.unsqueeze(0)) + tv
    loss.backward()
    optimizer.step()

    # 各エポックの終わりに学習率を更新
    scheduler.step()

    if (epoch + 1) % 1000 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}, LR: {current_lr:.8f}"
        )
        model.eval()
        with torch.no_grad():
            reconstucted_target = model(rec).squeeze(0).squeeze(0).reshape(784)
            print(reconstucted_target.shape)
        image_display(
            1,
            X_mnist[num, :],
            reconstucted_target.cpu().numpy(),
            model=model.model_name,
            epochs=epoch + 1,
            lr=current_lr,
            size=pixel,
            num=num,
            alpha=0,
            tv=TV_strength,
        )
        plt.close()
