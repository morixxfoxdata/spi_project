import os

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

import wandb
from models.unet import Conv1DModel
from utils.exp_utils import image_display, load_mnist, np_to_torch, speckle_pred

pixel = 8
# CUDAが使えるかどうかの判定
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ==========================================================================
# DATA _ PATH
# ==========================================================================
exp_data_dir = "../data/experiment"
save_dir = "../results/"
os.makedirs(save_dir, exist_ok=True)  # 保存先ディレクトリを作成
print(os.path.exists(save_dir))
num_images = 10

if pixel == 8:
    exp_collected = os.path.join(
        exp_data_dir,
        "collect/HP+mosaic+rand_image64+10+500_size8x8_alternate_200x20020240618_collect.npz",
    )
    exp_target = os.path.join(
        exp_data_dir, "target/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
    )
elif pixel == 28:
    exp_collected = os.path.join(
        exp_data_dir,
        "collect/Mnist+Rand_pix28x28_image(1000+1000)x2_sig2500_4wave_newPD.npz",
    )
    exp_target = os.path.join(
        exp_data_dir, "target/Mnist+Rand_pix28x28_image(1000+1000)x2.npz"
    )

# MODEL = FCModel(input_size=500, hidden_size=256, output_size=64, name="FC_trial_2")
# MODEL = UNet1D(name="TRIAL_UNet1D_v1")
MODEL = Conv1DModel(name="Conv1d")
region_indices = [0, 1, 2, 3]

# グローバル変数でベストトライアルの結果を格納
best_reconstructed_total = None  # ベストトライアルの画像データ
best_trial_number = None  # ベストトライアルの番号
X_mnist, Y_mnist = load_mnist(
    target_path=exp_target,
    collect_path=exp_collected,
    pixel=pixel,
    region_indices=region_indices,
)


def objective(trial):
    global best_reconstructed_total, best_trial_number  # ベストトライアル用のグローバル変数

    # WandBでトライアルごとにセッションを開始
    wandb.init(project="Conv1d_8px", name=f"trial_{trial.number}")

    num_epochs = trial.suggest_int("num_epochs", 13000, 15000)
    learning_rate = trial.suggest_float("learning_rate", 7e-4, 9e-4, log=True)

    model = MODEL
    S_0 = speckle_pred(
        target_path=exp_target,
        collect_path=exp_collected,
        region_indices=region_indices,
        pixel=pixel,
        alpha=1.0,
    )
    S_0_tensor = np_to_torch(S_0).float().to(device)

    Y_mnist_tensor = np_to_torch(Y_mnist).float()

    reconstructed_total = []
    for i in range(num_images):
        print("====================================")
        print(f"NUMBER : {i}")
        y_i = Y_mnist_tensor[i].unsqueeze(0).unsqueeze(0)
        y_i = y_i.to(device)
        print("y_i shape:", y_i.shape)
        model = MODEL.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            output = model(y_i).squeeze(0)

            Y_dash = torch.mm(output, S_0_tensor)
            loss = criterion(Y_dash, y_i.squeeze(0))
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
        reconstructed_total.append(reconstucted_target.cpu().numpy())

    reconstructed_total = np.vstack(reconstructed_total)
    mse_val = mean_squared_error(X_mnist, reconstructed_total)

    # ベストトライアルを更新
    if trial.number == 0 or mse_val < trial.study.best_value:
        best_reconstructed_total = reconstructed_total  # 最良の結果を保存
        best_trial_number = trial.number  # 最良トライアル番号を記録

    # WandBにトライアルごとの結果を記録
    wandb.log(
        {
            "trial_number": trial.number,
            "mse": mse_val,
            "reconstructed_images": [
                wandb.Image(img.reshape((pixel, -1))) for img in reconstructed_total
            ],
        }
    )

    # WandBセッション終了
    wandb.finish()

    return mse_val


# Optunaの実験開始
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)

# ベストトライアル結果を保存
if best_reconstructed_total is not None:
    save_path = os.path.join(save_dir, f"best_trial_{best_trial_number}_results.npz")
    np.savez(save_path, reconstructed_total=best_reconstructed_total)
    print(f"Best trial results saved to {save_path}")

# 結果を確認
print(f"Best trial: {study.best_trial.number}")
print(f"Best mse: {study.best_value}")

for i in range(num_images):
    image_display(
        1,
        X_mnist[i, :],
        best_reconstructed_total[i],
        model=MODEL.model_name,
        epochs=study.best_params["num_epochs"],
        lr=study.best_params["learning_rate"],
        size=pixel,
        num=i,
    )
