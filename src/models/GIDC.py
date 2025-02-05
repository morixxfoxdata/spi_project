import torch
import torch.nn as nn


class GIDC(nn.Module):
    def __init__(self):
        super(GIDC, self).__init__()
        self.down1_1 = self.down_sample(1, 16, kernel_size=5, stride=1, padding="same")
        self.down1_2 = self.down_sample(16, 16, kernel_size=5, stride=1, padding="same")
        self.down1_3 = self.down_sample(16, 32, kernel_size=5, stride=2, padding=2)
        self.down2_1 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")
        self.down2_2 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")
        self.down2_3 = self.down_sample(32, 64, kernel_size=5, stride=2, padding=2)
        self.down3_1 = self.down_sample(64, 64, kernel_size=5, stride=1, padding="same")
        self.down3_2 = self.down_sample(64, 64, kernel_size=5, stride=1, padding="same")
        self.down3_3 = self.down_sample(64, 128, kernel_size=5, stride=2, padding=2)
        self.down4_1 = self.down_sample(
            128, 128, kernel_size=5, stride=1, padding="same"
        )
        self.down4_2 = self.down_sample(
            128, 128, kernel_size=5, stride=1, padding="same"
        )
        self.down4_3 = self.down_sample(128, 256, kernel_size=5, stride=2, padding=2)
        self.down5_1 = self.down_sample(
            256, 256, kernel_size=5, stride=1, padding="same"
        )
        self.down5_2 = self.down_sample(
            256, 256, kernel_size=5, stride=1, padding="same"
        )
        self.up1_1 = self.up_sample(256, 128, kernel_size=5, stride=2, padding=2)
        self.up1_2 = self.down_sample(256, 128, kernel_size=5, stride=1, padding="same")
        self.up1_3 = self.down_sample(128, 128, kernel_size=5, stride=1, padding="same")
        self.up2_1 = self.up_sample(128, 64, kernel_size=5, stride=2, padding=2)
        self.up2_2 = self.down_sample(128, 64, kernel_size=5, stride=1, padding="same")
        self.up2_3 = self.down_sample(64, 64, kernel_size=5, stride=1, padding="same")
        self.up3_1 = self.up_sample(64, 32, kernel_size=5, stride=2, padding=2)
        self.up3_2 = self.down_sample(64, 32, kernel_size=5, stride=1, padding="same")
        self.up3_3 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")
        self.up4_1 = self.up_sample(32, 16, kernel_size=5, stride=2, padding=2)
        self.up4_2 = self.down_sample(32, 16, kernel_size=5, stride=1, padding="same")
        self.up4_3 = self.down_sample(16, 16, kernel_size=5, stride=1, padding="same")
        # self.up5 = self.down_sample(16, 1, kernel_size=5, stride=1, padding="same")
        self.up5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(self.dropout),
        )

    def up_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        d1_1 = self.down1_1(x)  # d1:(1, 16, 32, 32)
        # print(d1_1.shape)
        d1_2 = self.down1_2(d1_1)  # res
        # print(d1_2.shape)
        d1_3 = self.down1_3(d1_2)
        # print(d1_3.shape)
        d2_1 = self.down2_1(d1_3)
        # print(d2_1.shape)
        d2_2 = self.down2_2(d2_1)  # res
        # print(d2_2.shape)
        d2_3 = self.down2_3(d2_2)
        # print(d2_3.shape)
        d3_1 = self.down3_1(d2_3)
        # print(d3_1.shape)
        d3_2 = self.down3_2(d3_1)  # res
        # print(d3_2.shape)
        d3_3 = self.down3_3(d3_2)
        # print(d3_3.shape)
        d4_1 = self.down4_1(d3_3)
        # print(d4_1.shape)
        d4_2 = self.down4_2(d4_1)  # res
        # print("res:", d4_2.shape)
        d4_3 = self.down4_3(d4_2)
        # print(d4_3.shape)
        d5_1 = self.down5_1(d4_3)
        # print(d5_1.shape)
        d5_2 = self.down5_2(d5_1)
        # print(d5_2.shape)
        up1_1 = self.up1_1(d5_2)
        # print(up1_1.shape)
        cat1 = torch.cat([d4_2, up1_1], dim=1)
        # print(cat1.shape)
        up1_2 = self.up1_2(cat1)
        # print(up1_2.shape)
        up1_3 = self.up1_3(up1_2)
        # print(up1_3.shape)
        up2_1 = self.up2_1(up1_3)
        # print("up21:", up2_1.shape)
        cat2 = torch.cat([d3_2, up2_1], dim=1)
        # print(cat2.shape)
        up2_2 = self.up2_2(cat2)
        # print(up2_2.shape)
        up2_3 = self.up2_3(up2_2)
        # print(up2_3.shape)
        up3_1 = self.up3_1(up2_3)
        # print(up3_1.shape)
        cat3 = torch.cat([d2_2, up3_1], dim=1)
        # print(cat3.shape)
        up3_2 = self.up3_2(cat3)
        # print(up3_2.shape)
        up3_3 = self.up3_3(up3_2)
        # print(up3_3.shape)
        up4_1 = self.up4_1(up3_3)
        # print(up4_1.shape)
        cat4 = torch.cat([d1_2, up4_1], dim=1)
        # print(cat4.shape)
        up4_2 = self.up4_2(cat4)
        # print(up4_2.shape)
        up4_3 = self.up4_3(up4_2)
        # print(up4_3.shape)
        up5 = self.up5(up4_3)
        # print(up5.shape)
        return up5


# import torch.nn as nn


class GIDC28(nn.Module):
    def __init__(self, name="GIDC28"):
        super(GIDC28, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        # エンコーダ（ダウンサンプリング部分）
        # ※padding="same"はPyTorchの最新バージョンでサポートされているか、
        #    もしくは同等のパディング処理を行っている前提です。
        self.down1_1 = self.down_sample(1, 16, kernel_size=5, stride=1, padding="same")
        self.down1_2 = self.down_sample(16, 16, kernel_size=5, stride=1, padding="same")
        # ここでストライド2：28x28 -> 14x14
        self.down1_3 = self.down_sample(16, 32, kernel_size=5, stride=2, padding=2)

        self.down2_1 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")
        self.down2_2 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")
        # ここでストライド2：14x14 -> 7x7
        self.down2_3 = self.down_sample(32, 64, kernel_size=5, stride=2, padding=2)

        # ボトム層（必要に応じて追加の畳み込みを入れる）
        self.bottom = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        # デコーダ（アップサンプリング部分）
        # アップサンプリングで7x7 -> 14x14
        self.up1_1 = self.up_sample(64, 32, kernel_size=5, stride=2, padding=2)
        # スキップ接続で、down1_3 (14x14)と連結するのでチャネル数が 32+32 = 64 になる
        self.up1_2 = self.down_sample(64, 32, kernel_size=5, stride=1, padding="same")
        self.up1_3 = self.down_sample(32, 32, kernel_size=5, stride=1, padding="same")

        # 次にアップサンプリングで14x14 -> 28x28
        self.up2_1 = self.up_sample(32, 16, kernel_size=5, stride=2, padding=2)
        # スキップ接続として、最初の層の特徴（down1_2：28x28, チャネル数16）と連結
        self.up2_2 = self.down_sample(32, 16, kernel_size=5, stride=1, padding="same")
        self.up2_3 = self.down_sample(16, 16, kernel_size=5, stride=1, padding="same")

        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def up_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # エンコーダ
        d1_1 = self.down1_1(x)  # 入力: 28x28, 出力: 28x28, チャネル数16
        d1_2 = self.down1_2(d1_1)  # 28x28, 16チャネル（スキップ用）
        d1_3 = self.down1_3(d1_2)  # 28x28 -> 14x14, チャネル数32

        d2_1 = self.down2_1(d1_3)  # 14x14, 32チャネル
        d2_2 = self.down2_2(d2_1)  # 14x14, 32チャネル
        d2_3 = self.down2_3(d2_2)  # 14x14 -> 7x7, チャネル数64

        # ボトム層
        b = self.bottom(d2_3)  # 7x7, 64チャネル

        # デコーダ
        up1_1 = self.up1_1(b)  # 7x7 -> 14x14, チャネル数32
        # スキップ接続: down1_3の出力 (14x14, 32チャネル) と連結
        cat1 = torch.cat([d1_3, up1_1], dim=1)  # 14x14, 64チャネル
        up1_2 = self.up1_2(cat1)  # 14x14, 32チャネル
        up1_3 = self.up1_3(up1_2)  # 14x14, 32チャネル

        up2_1 = self.up2_1(up1_3)  # 14x14 -> 28x28, チャネル数16
        # スキップ接続: down1_2の出力 (28x28, 16チャネル) と連結
        cat2 = torch.cat([d1_2, up2_1], dim=1)  # 28x28, 32チャネル
        up2_2 = self.up2_2(cat2)  # 28x28, 16チャネル
        up2_3 = self.up2_3(up2_2)  # 28x28, 16チャネル

        out = self.final(up2_3)  # 28x28, 1チャネル（最終出力）
        return out


# 使用例
if __name__ == "__main__":
    net = GIDC28()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = net(dummy_input)
    print(output.shape)  # torch.Size([1, 1, 28, 28])
