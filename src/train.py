import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from models.skip import skip
from models.unet_ad import UNet1DShallow

# from models.fcmodel import FCModel
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
TV_strength = 1e-8
ALPHA = 10
# ==========================================================================
# DATA _ PATH
# ==========================================================================
exp_data_dir = "../data/experiment"
save_dir = "../results/"

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

# ==========================================================================
# SELECT WAVE_LENGTH
# region_indices=0： 0から2500点を利用
# region_indices=[0, 1]: 0から5000点を利用
# region_indices=2：5000から7500点を利用
# ==========================================================================
region_indices = [0, 1, 2, 3]
num_images = 10
learning_rate = 1e-4
num_epochs = 2000
# ==========================================================================
# Y_mnist Shape: (10, 2500)
# X_mnist Shape: (10, 784)
X_mnist, Y_mnist = load_mnist(
    target_path=exp_target,
    collect_path=exp_collected,
    pixel=pixel,
    region_indices=region_indices,
)
# if pixel == 8:
#     S_0 = speckle_pred_8(
#         target_path=exp_target,
#         collect_path=exp_collected,
#         region_indices=region_indices,
#         pixel=8,
#     )

S_0 = speckle_pred(
    target_path=exp_target,
    collect_path=exp_collected,
    region_indices=region_indices,
    pixel=pixel,
    alpha=ALPHA,
)
print("S_0 shape:", S_0.shape)


S_0_tensor = np_to_torch(S_0).float()

X_mnist_tensor = np_to_torch(X_mnist).float()

# (1000, 10000)
Y_mnist_tensor = np_to_torch(Y_mnist).float()


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

# save_dir Exists
# print(os.path.exists("../results/pix28"))

######################################################
# Training function
######################################################

loss_total = []
reconstructed_total = []
S_0_tensor = S_0_tensor.to(device)

print("S max, S min:", S_0_tensor.max(), S_0_tensor.min())
print("Y_mnist max, min:", Y_mnist_tensor.max(), Y_mnist_tensor.min())

for i in range(num_images):
    # initialize model and params
    # model = FCModel(
    #     input_size=10000, hidden_size=4096, output_size=784, name="FC_default_4096"
    # ).to(device)
    # model = UNet1D(name="CV1_conv_tv").to(device)
    model = UNet1DShallow(name="CV_shal_tv_ver2").to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    y_i = Y_mnist_tensor[i].unsqueeze(0).unsqueeze(0)  # (1, 2500)
    y_i = y_i.to(device)
    print("y_i shape:", y_i.shape)
    # y_i = standardize(y=y_i)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # output shape: (1, 784)
        out = model(y_i)
        output = out.squeeze(0)
        # print(output.shape)
        # print(S_0_tensor.shape)
        Y_dash = torch.mm(output, S_0_tensor)
        tv = TV_strength * total_variation_loss(out.reshape((1, 1, 28, 28)))
        loss = criterion(Y_dash, y_i.squeeze(0)) + tv
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                f"Image {i}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
            )
    model.eval()
    with torch.no_grad():
        reconstucted_target = model(y_i).squeeze(0).squeeze(0)
        print(reconstucted_target.shape)
    image_display(
        1,
        X_mnist[i, :],
        reconstucted_target.cpu().numpy(),
        model=model.model_name,
        epochs=num_epochs,
        lr=learning_rate,
        size=pixel,
        num=i,
        alpha=ALPHA,
        tv=TV_strength,
    )
    reconstructed_total.append(reconstucted_target.cpu().numpy())


np.savez(
    f"{save_dir}pix{pixel}_npz/{model.model_name}_img_09_iter{num_epochs}_lr{learning_rate}_a{ALPHA}_tv{TV_strength}.npz",
    reconstructed_total,
)
