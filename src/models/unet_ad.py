import torch
import torch.nn as nn

from .common import conv


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class UNet1D(nn.Module):
    """
    1次元用のUNet構造
    upsample_mode in ['deconv', 'nearest', 'linear']
    pad in ['zero', 'replication', 'none']
    """

    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
        feature_scale=4,
        more_layers=0,
        concat_x=False,
        upsample_mode="deconv",
        pad="zero",
        norm_layer=nn.InstanceNorm1d,
        need_sigmoid=True,
        need_bias=True,
        name="UNet1D",
        time_length=10000,
    ):
        super(UNet1D, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        filters = [64, 128, 256, 512, 1024]
        # filters = [16, 32, 64, 128, 256]
        # filters = [64, 128, 256]
        filters = [x // self.feature_scale for x in filters]
        self.before = nn.Sequential(nn.Linear(time_length, 4096), nn.LeakyReLU())
        self.start = unetConv1(
            num_input_channels,
            filters[0] if not concat_x else filters[0] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )

        self.down1 = unetDown1(
            filters[0],
            filters[1] if not concat_x else filters[1] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )
        self.down2 = unetDown1(
            filters[1],
            filters[2] if not concat_x else filters[2] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )
        self.down3 = unetDown1(
            filters[2],
            filters[3] if not concat_x else filters[3] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )
        self.down4 = unetDown1(
            filters[3],
            filters[4] if not concat_x else filters[4] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown1(
                    filters[4],
                    filters[4] if not concat_x else filters[4] - num_input_channels,
                    norm_layer,
                    need_bias,
                    pad,
                )
                for i in range(self.more_layers)
            ]
            self.more_ups = [
                unetUp1(filters[4], upsample_mode, need_bias, pad, same_num_filt=True)
                for i in range(self.more_layers)
            ]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = unetUp1(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp1(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp1(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp1(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid:
            self.final = nn.Sequential(
                self.final,
                nn.AdaptiveAvgPool1d(4096),
                nn.Linear(4096, 784),
                nn.Tanh(),
            )

    def forward(self, inputs):
        # Downsample
        downs = [inputs]
        down = nn.AvgPool1d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))
        b64 = self.before(inputs)
        in64 = self.start(b64)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], dim=1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], dim=1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], dim=1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], dim=1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], dim=1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 5]], dim=1)
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                lay = self.more_ups[self.more_layers - idx - 2]
                up_ = lay(up_, prevs[self.more_layers - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        # print(up4.shape)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        # print(up2.shape)
        up1 = self.up1(up2, in64)
        # print(up1.shape[2])
        return self.final(up1)


class unetConv1(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv1, self).__init__()

        if norm_layer is not None:
            self.conv1 = nn.Sequential(
                conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                norm_layer(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                norm_layer(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(
                conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                nn.ReLU(),
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown1(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown1, self).__init__()
        self.conv = unetConv1(in_size, out_size, norm_layer, need_bias, pad)
        self.down = nn.MaxPool1d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class unetUp1(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp1, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == "deconv":
            self.up = nn.ConvTranspose1d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv = unetConv1(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode in ["linear", "nearest"]:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                conv(num_filt, out_size, 3, bias=need_bias, pad=pad),
            )
            self.conv = unetConv1(out_size * 2, out_size, None, need_bias, pad)
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if inputs2.size(2) != in1_up.size(2):
            diff = inputs2.size(2) - in1_up.size(2)
            if diff > 0:
                inputs2_ = inputs2[:, :, diff // 2 : diff // 2 + in1_up.size(2)]
            else:
                pad = (-diff) // 2
                inputs2_ = nn.functional.pad(inputs2, (pad, pad))
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], dim=1))

        return output


class UNet1DShallow(nn.Module):
    """
    1次元用のUNet構造を浅くした版（downとupを各1段ずつ減らした）
    upsample_mode in ['deconv', 'nearest', 'linear']
    pad in ['zero', 'replication', 'none']
    """

    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
        feature_scale=4,
        more_layers=0,
        concat_x=False,
        upsample_mode="deconv",
        pad="zero",
        norm_layer=nn.InstanceNorm1d,
        need_sigmoid=True,
        need_bias=True,
        name="cv_shal",
        time_length=10000,
    ):
        super(UNet1DShallow, self).__init__()
        self.model_name = name if name else self.__class__.__name__
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        # down/up を 1 つずつ減らしたため、フィルタ数のリストは 4 要素に
        filters = [16, 32, 64, 128]
        filters = [x // self.feature_scale for x in filters]

        self.before = nn.Sequential(nn.Linear(time_length, 4096), nn.LeakyReLU())

        # 最初のConv
        self.start = unetConv1(
            num_input_channels,
            filters[0] if not concat_x else filters[0] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )

        # Down (3段)
        self.down1 = unetDown1(
            filters[0],
            filters[1] if not concat_x else filters[1] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )
        self.down2 = unetDown1(
            filters[1],
            filters[2] if not concat_x else filters[2] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )
        self.down3 = unetDown1(
            filters[2],
            filters[3] if not concat_x else filters[3] - num_input_channels,
            norm_layer,
            need_bias,
            pad,
        )

        # more_layers > 0 の場合の追加 Down/Up
        if self.more_layers > 0:
            self.more_downs = [
                unetDown1(
                    filters[3],
                    filters[3] if not concat_x else filters[3] - num_input_channels,
                    norm_layer,
                    need_bias,
                    pad,
                )
                for _ in range(self.more_layers)
            ]
            self.more_ups = [
                unetUp1(filters[3], upsample_mode, need_bias, pad, same_num_filt=True)
                for _ in range(self.more_layers)
            ]
            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        # Up (3段)
        #   unetUp1 は in_channels を受け取るが，4段→3段に減ったので
        #   それぞれ対応するフィルタ数にする
        self.up3 = unetUp1(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp1(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp1(filters[0], upsample_mode, need_bias, pad)

        # 出力
        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid:
            self.final = nn.Sequential(
                self.final,
                nn.AdaptiveAvgPool1d(4096),
                nn.Linear(4096, 784),
                nn.Tanh(),
            )

    def forward(self, inputs):
        # Downsample 用テンソル（concat_x=Trueの場合に使用）
        # もともと4階層(＋more_layers)→ダウンサンプリングは 4+more_layers 回
        # 今回は3階層なので 3+more_layers 回
        downs = [inputs]
        down = nn.AvgPool1d(2, 2)
        for i in range(3 + self.more_layers):
            downs.append(down(downs[-1]))

        # 最初のConv
        b64 = self.before(inputs)
        in64 = self.start(b64)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], dim=1)

        # Down 1
        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], dim=1)

        # Down 2
        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], dim=1)

        # Down 3
        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], dim=1)

        # more_layers がある場合の追加ダウン/アップ
        if self.more_layers > 0:
            prevs = [down3]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out, downs[kk + 4]], dim=1)
                prevs.append(out)

            # Up
            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                lay = self.more_ups[self.more_layers - idx - 2]
                up_ = lay(up_, prevs[self.more_layers - idx - 2])
        else:
            up_ = down3

        # Up の合成
        up3 = self.up3(up_, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


if __name__ == "__main__":
    x = torch.randn(1, 1, 10000)
    print(x.shape)
    model = UNet1D()
    print(model(x).shape)  # (1, 1, 64)
