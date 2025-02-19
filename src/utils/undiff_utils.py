import os

import matplotlib.pyplot as plt
import numpy as np

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


# ==========================================================================
def load_data_undiff(target_path, collect_path, color):
    target = np.load(target_path)["arr_0"]
    collect = np.load(collect_path)["arr_0"]
    # 重複を削除
    collect = collect[:, ::2]
    # 文字色を選択
    if color == "white":
        target = target[::2, :]
        collect = collect[::2, :]
    elif color == "black":
        target = target[1::2, :]
        collect = collect[1::2, :]
    else:
        raise ValueError("Color must be black or white.")
    return target, collect


def load_mnist_undiff(target_path, collect_path, color):
    target, collect = load_data_undiff(target_path, collect_path, color)
    Y_mnist = collect[900:910, :]
    X_mnist = target[900:910, :]
    return X_mnist, Y_mnist


def load_random_undiff(target_path, collect_path, color):
    target, collect = load_data_undiff(target_path, collect_path, color)
    Y_rand = collect[1000:, :]
    X_rand = target[1000:, :]
    return X_rand, Y_rand


def speckle_pred_inv_diff(target_path, collect_path, color):
    X_rand, Y_rand = load_random_undiff(
        target_path=target_path, collect_path=collect_path, color=color
    )
    X_pinv = np.linalg.pinv(X_rand)
    S = np.dot(X_pinv, Y_rand)  # S: (784, 10000)
    print("S shape:", S.shape)
    return S


if __name__ == "__main__":
    # target, collect = load_data_undiff(
    #     target_path=exp_target,
    #     collect_path=exp_collected,
    #     color="white",
    # )
    X_mnist, Y_mnist = load_mnist_undiff(
        target_path=exp_target,
        collect_path=exp_collected,
        color="white",
    )
    X_random, Y_random = load_random_undiff(
        target_path=exp_target,
        collect_path=exp_collected,
        color="white",
    )
    print("MNIST shape:", X_mnist.shape, Y_mnist.shape)
    print("random shape:", X_random.shape, Y_random.shape)
    # print("target_1 val:", target[1])
    # print("collect_1 val:", collect[1])
    # print("collect_0 val:", collect[0])
    plt.imshow(X_mnist[1].reshape((28, -1)), cmap="gray")
    plt.show()
    # fig, ax = plt.subplots()
    # ax.hist(X_mnist[1])
    # plt.show()
