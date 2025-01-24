import os

import numpy as np

from utils.exp_utils import load_mnist

exp_data_dir = "data/experiment"

# print(os.path.exists(exp_data_dir))
exp_collected = os.path.join(
    exp_data_dir,
    "collect/Mnist+Rand_pix28x28_image(1000+1000)x2_sig2500_4wave_newPD.npz",
)
exp_target = os.path.join(
    exp_data_dir, "target/Mnist+Rand_pix28x28_image(1000+1000)x2.npz"
)

# Y_mnist Shape: (1000, 784)
# X_mnist Shape: (1000, 10000)

X_mnist, Y_mnist = load_mnist(target_path=exp_target, collect_path=exp_collected)

inv_X = np.linalg.pinv(X_mnist)
print("inv_X Shape:", inv_X.shape)

S_0 = np.matmul(inv_X, Y_mnist)
print("S_0 Shape:", S_0.shape)
