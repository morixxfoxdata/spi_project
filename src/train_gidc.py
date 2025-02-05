import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.GIDC import GIDC28

# from models.skip import skip
from utils.exp_utils import image_display, load_mnist, np_to_torch, speckle_pred

# ==========================================================================
# PIXELS
# ==========================================================================
pixel = 28
num = 0

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
num_images = 10
learning_rate = 0.0006
num_epochs = 5000

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
    alpha=1.0,
)
print("S_0 shape:", S_0.shape)


S_0_tensor = np_to_torch(S_0).float()

X_mnist_tensor = np_to_torch(X_mnist).float()

# (1000, 10000)
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
S_0_tensor = S_0_tensor.to(device)
model = GIDC28(name="GIDC_tanh").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

y_ = Y_mnist_tensor[num].unsqueeze(0).unsqueeze(0).to(device)
print(X_mnist.max())
print(dgi_path)
dgi = np.load(dgi_path)["arr_0"]
print(dgi.shape)
dgi_tensor = np_to_torch(dgi).unsqueeze(0).unsqueeze(0).to(device)
print(dgi_tensor.shape)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(dgi_tensor).squeeze(0)
    # print(output.shape)
    # print(S_0_tensor.shape)
    Y_dash = torch.mm(output.reshape((1, 784)), S_0_tensor)
    loss = criterion(Y_dash, y_.squeeze(0))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
        )
model.eval()
with torch.no_grad():
    reconstucted_target = model(dgi_tensor).squeeze(0).squeeze(0).reshape(784)
    print(reconstucted_target.shape)
image_display(
    1,
    X_mnist[num, :],
    reconstucted_target.cpu().numpy(),
    model=model.model_name,
    epochs=num_epochs,
    lr=learning_rate,
    size=pixel,
    num=num,
)
