import torch
import torch.nn as nn


class GIDC(nn.Module):
    def __init__(self):
        super(GIDC, self).__init__()
        # 入力サイズ (1,1,32,32) を保つため、stride=1の畳み込みではpadding=2（= (5-1)/2）を用いる
        self.down1_1 = self.down_sample(1, 16, kernel_size=5, stride=1, padding=2)
        self.down1_2 = self.down_sample(16, 16, kernel_size=5, stride=1, padding=2)
        self.down1_3 = self.down_sample(16, 32, kernel_size=5, stride=2, padding=2)
        self.down2_1 = self.down_sample(32, 32, kernel_size=5, stride=1, padding=2)
        self.down2_2 = self.down_sample(32, 32, kernel_size=5, stride=1, padding=2)
        self.down2_3 = self.down_sample(32, 64, kernel_size=5, stride=2, padding=2)
        self.down3_1 = self.down_sample(64, 64, kernel_size=5, stride=1, padding=2)
        self.down3_2 = self.down_sample(64, 64, kernel_size=5, stride=1, padding=2)
        self.down3_3 = self.down_sample(64, 128, kernel_size=5, stride=2, padding=2)
        self.down4_1 = self.down_sample(128, 128, kernel_size=5, stride=1, padding=2)
        self.down4_2 = self.down_sample(128, 128, kernel_size=5, stride=1, padding=2)
        self.down4_3 = self.down_sample(128, 256, kernel_size=5, stride=2, padding=2)
        self.down5_1 = self.down_sample(256, 256, kernel_size=5, stride=1, padding=2)
        self.down5_2 = self.down_sample(256, 256, kernel_size=5, stride=1, padding=2)
        self.up1_1 = self.up_sample(256, 128, kernel_size=5, stride=2, padding=2)
        self.up1_2 = self.down_sample(256, 128, kernel_size=5, stride=1, padding=2)
        self.up1_3 = self.down_sample(128, 128, kernel_size=5, stride=1, padding=2)
        self.up2_1 = self.up_sample(128, 64, kernel_size=5, stride=2, padding=2)
        self.up2_2 = self.down_sample(128, 64, kernel_size=5, stride=1, padding=2)
        self.up2_3 = self.down_sample(64, 64, kernel_size=5, stride=1, padding=2)
        self.up3_1 = self.up_sample(64, 32, kernel_size=5, stride=2, padding=2)
        self.up3_2 = self.down_sample(64, 32, kernel_size=5, stride=1, padding=2)
        self.up3_3 = self.down_sample(32, 32, kernel_size=5, stride=1, padding=2)
        self.up4_1 = self.up_sample(32, 16, kernel_size=5, stride=2, padding=2)
        self.up4_2 = self.down_sample(32, 16, kernel_size=5, stride=1, padding=2)
        self.up4_3 = self.down_sample(16, 16, kernel_size=5, stride=1, padding=2)
        self.up5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        # 【注釈】nn.Conv2d: 2次元畳み込み層。paddingはカーネルサイズが奇数の場合、(kernel_size-1)/2にすると入力と同じ空間サイズとなる
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
        # 【注釈】nn.ConvTranspose2d: 転置畳み込み層。output_padding=1 を加えることで、出力サイズを調整している
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
        d1_1 = self.down1_1(x)  # (1, 16, 32, 32)
        d1_2 = self.down1_2(d1_1)  # (1, 16, 32, 32)
        d1_3 = self.down1_3(d1_2)  # (1, 32, 16, 16)
        d2_1 = self.down2_1(d1_3)  # (1, 32, 16, 16)
        d2_2 = self.down2_2(d2_1)  # (1, 32, 16, 16)
        d2_3 = self.down2_3(d2_2)  # (1, 64, 8, 8)
        d3_1 = self.down3_1(d2_3)  # (1, 64, 8, 8)
        d3_2 = self.down3_2(d3_1)  # (1, 64, 8, 8)
        d3_3 = self.down3_3(d3_2)  # (1, 128, 4, 4)
        d4_1 = self.down4_1(d3_3)  # (1, 128, 4, 4)
        d4_2 = self.down4_2(d4_1)  # (1, 128, 4, 4)
        d4_3 = self.down4_3(d4_2)  # (1, 256, 2, 2)
        d5_1 = self.down5_1(d4_3)  # (1, 256, 2, 2)
        d5_2 = self.down5_2(d5_1)  # (1, 256, 2, 2)
        up1_1 = self.up1_1(d5_2)  # (1, 128, 4, 4)
        cat1 = torch.cat([d4_2, up1_1], dim=1)  # (1, 256, 4, 4)
        up1_2 = self.up1_2(cat1)  # (1, 128, 4, 4)
        up1_3 = self.up1_3(up1_2)  # (1, 128, 4, 4)
        up2_1 = self.up2_1(up1_3)  # (1, 64, 8, 8)
        cat2 = torch.cat([d3_2, up2_1], dim=1)  # (1, 128, 8, 8)
        up2_2 = self.up2_2(cat2)  # (1, 64, 8, 8)
        up2_3 = self.up2_3(up2_2)  # (1, 64, 8, 8)
        up3_1 = self.up3_1(up2_3)  # (1, 32, 16, 16)
        cat3 = torch.cat([d2_2, up3_1], dim=1)  # (1, 64, 16, 16)
        up3_2 = self.up3_2(cat3)  # (1, 32, 16, 16)
        up3_3 = self.up3_3(up3_2)  # (1, 32, 16, 16)
        up4_1 = self.up4_1(up3_3)  # (1, 16, 32, 32)
        cat4 = torch.cat([d1_2, up4_1], dim=1)  # (1, 32, 32, 32)
        up4_2 = self.up4_2(cat4)  # (1, 16, 32, 32)
        up4_3 = self.up4_3(up4_2)  # (1, 16, 32, 32)
        up5 = self.up5(up4_3)  # (1, 1, 32, 32)
        return up5


# import torch.nn as nn


class GIDC28(nn.Module):
    def __init__(self, kernel_size=5, name="GI"):
        """
        kernel_size: 畳み込み層・転置畳み込み層で使用するカーネルサイズ。
                     "same" パディングの場合、内部で (kernel_size-1)//2 を計算します。
        """
        super(GIDC28, self).__init__()
        self.kernel_size = kernel_size
        self.model_name = name if name else self.__class__.__name__
        # エンコーダ部分
        self.down1_1 = self.down_sample(
            1, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_2 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_3 = self.down_sample(
            16, 32, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.down2_1 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_2 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_3 = self.down_sample(
            32, 64, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, "same"),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        # デコーダ部分
        self.up1_1 = self.up_sample(
            64, 32, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up1_2 = self.down_sample(
            64, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up1_3 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.up2_1 = self.up_sample(
            32, 16, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up2_2 = self.down_sample(
            32, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up2_3 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                16,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, "same"),
            ),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

    def _get_padding(self, kernel_size, padding):
        """
        padding引数が "same" の場合、(kernel_size-1)//2 を返す。
        それ以外の場合はそのままの値を返す。
        """
        if padding == "same":
            return (kernel_size - 1) // 2
        else:
            return padding

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def up_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        # 出力サイズを (input_size * stride) にするための output_padding を自動計算
        output_padding = stride + 2 * pad - kernel_size
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # エンコーダ
        d1_1 = self.down1_1(x)  # 28x28 -> 28x28, チャネル数16
        d1_2 = self.down1_2(d1_1)  # 28x28, チャネル数16（スキップ接続用）
        d1_3 = self.down1_3(d1_2)  # 28x28 -> 14x14, チャネル数32

        d2_1 = self.down2_1(d1_3)  # 14x14, チャネル数32
        d2_2 = self.down2_2(d2_1)  # 14x14, チャネル数32
        d2_3 = self.down2_3(d2_2)  # 14x14 -> 7x7, チャネル数64

        b = self.bottom(d2_3)  # ボトム層：7x7, チャネル数64

        # デコーダ
        up1_1 = self.up1_1(b)  # 7x7 -> 14x14, チャネル数32
        cat1 = torch.cat([d1_3, up1_1], dim=1)  # スキップ接続：14x14, チャネル数64
        up1_2 = self.up1_2(cat1)  # 14x14, チャネル数32
        up1_3 = self.up1_3(up1_2)  # 14x14, チャネル数32

        up2_1 = self.up2_1(up1_3)  # 14x14 -> 28x28, チャネル数16
        cat2 = torch.cat([d1_2, up2_1], dim=1)  # スキップ接続：28x28, チャネル数32
        up2_2 = self.up2_2(cat2)  # 28x28, チャネル数16
        up2_3 = self.up2_3(up2_2)  # 28x28, チャネル数16

        out = self.final(up2_3)  # 28x28, チャネル数1（最終出力）
        return out


class GIDC28_for_notdiff(nn.Module):
    def __init__(self, kernel_size=5, name="GI"):
        """
        kernel_size: 畳み込み層・転置畳み込み層で使用するカーネルサイズ。
        "same" パディングの場合、内部で (kernel_size-1)//2 を計算します。
        """
        super(GIDC28_for_notdiff, self).__init__()
        self.kernel_size = kernel_size
        self.model_name = name if name else self.__class__.__name__
        # エンコーダ部分
        self.down1_1 = self.down_sample(
            1, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_2 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down1_3 = self.down_sample(
            16, 32, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.down2_1 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_2 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.down2_3 = self.down_sample(
            32, 64, kernel_size=kernel_size, stride=2, padding="same"
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, "same"),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        # デコーダ部分
        self.up1_1 = self.up_sample(
            64, 32, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up1_2 = self.down_sample(
            64, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up1_3 = self.down_sample(
            32, 32, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.up2_1 = self.up_sample(
            32, 16, kernel_size=kernel_size, stride=2, padding="same"
        )
        self.up2_2 = self.down_sample(
            32, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.up2_3 = self.down_sample(
            16, 16, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                16,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=self._get_padding(kernel_size, "same"),
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def _get_padding(self, kernel_size, padding):
        """
        padding引数が "same" の場合、(kernel_size-1)//2 を返す。
        それ以外の場合はそのままの値を返す。
        """
        if padding == "same":
            return (kernel_size - 1) // 2
        else:
            return padding

    def down_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def up_sample(self, in_channels, out_channels, kernel_size, stride, padding):
        pad = self._get_padding(kernel_size, padding)
        # 出力サイズを (input_size * stride) にするための output_padding を自動計算
        output_padding = stride + 2 * pad - kernel_size
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # エンコーダ
        d1_1 = self.down1_1(x)  # 28x28 -> 28x28, チャネル数16
        d1_2 = self.down1_2(d1_1)  # 28x28, チャネル数16（スキップ接続用）
        d1_3 = self.down1_3(d1_2)  # 28x28 -> 14x14, チャネル数32

        d2_1 = self.down2_1(d1_3)  # 14x14, チャネル数32
        d2_2 = self.down2_2(d2_1)  # 14x14, チャネル数32
        d2_3 = self.down2_3(d2_2)  # 14x14 -> 7x7, チャネル数64

        b = self.bottom(d2_3)  # ボトム層：7x7, チャネル数64

        # デコーダ
        up1_1 = self.up1_1(b)  # 7x7 -> 14x14, チャネル数32
        cat1 = torch.cat([d1_3, up1_1], dim=1)  # スキップ接続：14x14, チャネル数64
        up1_2 = self.up1_2(cat1)  # 14x14, チャネル数32
        up1_3 = self.up1_3(up1_2)  # 14x14, チャネル数32

        up2_1 = self.up2_1(up1_3)  # 14x14 -> 28x28, チャネル数16
        cat2 = torch.cat([d1_2, up2_1], dim=1)  # スキップ接続：28x28, チャネル数32
        up2_2 = self.up2_2(cat2)  # 28x28, チャネル数16
        up2_3 = self.up2_3(up2_2)  # 28x28, チャネル数16

        out = self.final(up2_3)  # 28x28, チャネル数1（最終出力）
        return out


if __name__ == "__main__":
    # kernel_sizeの値を変えても動作する例（ここでは kernel_size=7）
    # net = GIDC28(kernel_size=5)
    net = GIDC28_for_notdiff(kernel_size=3)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = net(dummy_input)
    print(output.shape)  # 例: torch.Size([1, 1, 28, 28])
