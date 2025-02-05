import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.GIDC import GIDC28

# from models.skip import skip
from utils.exp_utils import (
    image_display,
    load_mnist,
    np_to_torch,
    speckle_pred,
    total_variation_loss,
)

# ==========================================================================
# PIXELS
# ==========================================================================
pixel = 28
num = 1

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
# num_images = 10
learning_rate = 0.005
num_epochs = 10000
TV_strength = 7e-8
X_mnist, Y_mnist = load_mnist(
    target_path=exp_target,
    collect_path=exp_collected,
    pixel=pixel,
    region_indices=region_indices,
)
S_0 = speckle_pred(
    target_path=exp_target,
    collect_path=exp_collected,
    region_indices=region_indices,
    pixel=pixel,
    alpha=20,
)
print("S_0 shape:", S_0.shape)

# ※実際には既に与えられたデータを使用してください。
num_images = 1000  # 画像枚数
num_pixels = 784  # 1枚の画像のピクセル数（例：28×28）
num_patterns = 10000  # 照明パターン枚数

S_0_tensor = np_to_torch(S_0).float()
# (10, 784)
X_mnist_tensor = np_to_torch(X_mnist).float()

# (10, 10000)
Y_mnist_tensor = np_to_torch(Y_mnist).float()

dgi_path = os.path.join(exp_data_dir, f"pix28_{num}_dgi.npz")
# CUDAが使えるかどうかの判定
if torch.cuda.is_available():
    device = "cuda"
# MPSが使えるかどうかの判定（Appleシリコン環境など）
elif torch.backends.mps.is_available():
    device = "mps"
# 上記がどちらも使えない場合はCPU
else:
    device = "cpu"

print("Using device:", device)
# S_0_tensor = S_0_tensor.to(device)


# R = torch.sum(S_0_tensor, dim=0)  # R[i] = sum(P_i)
# avg_Y = torch.mean(Y_mnist_tensor, dim=1, keepdim=True)
# avg_R = torch.mean(R)
# weight = Y_mnist_tensor - (avg_Y / avg_R) * R
# I_rec = torch.matmul(weight, S_0_tensor.T) / num_patterns


# plt.show()
S_0_tensor = S_0_tensor.to(device)

model = GIDC28(kernel_size=5, name="GIDC_7e8").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

y_ = Y_mnist_tensor[num, :].unsqueeze(0).unsqueeze(0).to(device)
print(X_mnist.max())
print(dgi_path)
dgi = np.load(dgi_path)["arr_0"]
print(dgi.shape)
dgi_tensor = np_to_torch(dgi).unsqueeze(0).unsqueeze(0).to(device)
print(dgi_tensor.shape)
I_rec = dgi_tensor.reshape((28, 28)).unsqueeze(0).unsqueeze(0).to(device)
print(I_rec.shape)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(I_rec)
    # print(output.shape)
    # print(S_0_tensor.shape)
    Y_dash = torch.mm(output.reshape((1, 784)), S_0_tensor)
    tv = TV_strength * total_variation_loss(output)
    loss = criterion(Y_dash, y_.squeeze(0)) + tv
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
        )
        model.eval()
        with torch.no_grad():
            reconstucted_target = model(I_rec).squeeze(0).squeeze(0).reshape(784)
            print(reconstucted_target.shape)
        image_display(
            1,
            X_mnist[num, :],
            reconstucted_target.cpu().numpy(),
            model=model.model_name,
            epochs=epoch + 1,
            lr=learning_rate,
            size=pixel,
            num=num,
        )
        plt.close()
