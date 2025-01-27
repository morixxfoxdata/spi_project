import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Unet-based original model
    """

    def __init__(self, dropout, name="UNet"):
        super(UNet, self).__init__()
        self.dropout = dropout
        self.model_name = name if name else self.__class__.__name__
        # Encoder
        self.down1 = self.down_sample(1, 32)
        self.down2 = self.down_sample(32, 64)
        self.down3 = self.down_sample(64, 128)

        # Decoder
        self.up3 = self.up_sample(128, 64)
        self.up2 = self.up_sample(128, 32)
        self.up1 = self.up_sample(64, 16)
        self.fc = nn.Sequential(
            nn.Linear(16 * 500, 512),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
        )
        self.activation = nn.Tanh()

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        # x = x.unsqueeze(1)  # (batch_size, 1, 500)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 250)
        d2 = self.down2(d1)  # (batch_size, 64, 125)
        d3 = self.down3(d2)  # (batch_size, 128, 63)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 126)
        u3 = torch.cat([u3[:, :, :125], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 250)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500)
        # print(u1.shape)
        # Flatten and pass through fully connected layers
        out = self.fc(u1.view(1, 1, -1))  # (batch_size, 64)
        out_activated = self.activation(out)
        return out_activated


class MultiscaleSpeckleNet(nn.Module):
    def __init__(self, outdim, name="MultiNet"):
        super(MultiscaleSpeckleNet, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        # Encoder
        self.down1 = self.down_sample(1, 32)
        self.down2 = self.down_sample(32, 64)
        self.down3 = self.down_sample(64, 128)

        # Decoder
        self.up3 = self.up_sample(128, 64)
        self.up2 = self.up_sample(128, 32)
        self.up1 = self.up_sample(64, 16)
        self.activation = nn.Tanh()
        # self.activation = BinaryActivation()
        # self.activation = BinarySTEActivation()
        # self.activation = SmoothBinaryActivation()

        # グローバルプーリングを追加
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 出力サイズを1に固定

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16, outdim),  # 入力次元を16に変更
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim, outdim),
            # nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim, outdim),
            # nn.Sigmoid(),
        )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def up_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # batch_size=1(1枚分だから)
        # x shape: (batch_size, 65536)
        # x = x.unsqueeze(1)  # (batch_size, 1, 65536)

        # Encoder
        d1 = self.down1(x)  # (batch_size, 32, 32768)
        d2 = self.down2(d1)  # (batch_size, 64, 16384)
        d3 = self.down3(d2)  # (batch_size, 128, 8192)

        # Decoder
        u3 = self.up3(d3)  # (batch_size, 64, 16384)
        # 入力サイズに応じてトリミング
        u3 = torch.cat([u3[:, :, : d2.size(2)], d2], dim=1)  # (batch_size, 128, 125)

        u2 = self.up2(u3)  # (batch_size, 32, 32768)
        u2 = torch.cat([u2, d1], dim=1)  # (batch_size, 64, 250)

        u1 = self.up1(u2)  # (batch_size, 16, 500) ※入力サイズに応じて変更されます
        # print(u1.shape)
        # グローバルプーリングで固定サイズに
        pooled = self.global_pool(u1).reshape(1, 1, -1)  # (batch_size, 16)
        # print(pooled.shape)
        # Flatten and pass through fully connected layers
        out = self.fc(pooled)  # (batch_size, 64)
        activated_out = self.activation(out)
        return activated_out


if __name__ == "__main__":
    x = torch.randn(1, 1, 500)
    print(x.shape)
    # model = UNet(dropout=0.1)
    model = MultiscaleSpeckleNet(outdim=64)
    print(model(x).shape)
