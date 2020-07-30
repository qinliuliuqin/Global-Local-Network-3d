import torch.nn.functional as F
import torch
import torch.nn as nn

from networks.module.weight_init import kaiming_weight_init, gaussian_weight_init


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


class InputBlock(nn.Module):
  """ input block of vb-net """

  def __init__(self, in_channels, out_channels):
    super(InputBlock, self).__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    self.gn = nn.GroupNorm(1, num_channels=out_channels)
    self.act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.act(self.gn(self.conv(input)))
    return out


class OutputBlock(nn.Module):
  """ output block of v-net
      The output is a list of foreground-background probability vectors.
      The length of the list equals to the number of voxels in the volume
  """

  def __init__(self, in_channels, out_channels):
    super(OutputBlock, self).__init__()
    self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    self.gn1 = nn.GroupNorm(1, out_channels)
    self.act1 = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.act1(self.gn1(self.conv1(input)))

    return out


class ConvGnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, stride, padding, do_act=True, bias=True):
        super(ConvGnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride=stride, padding=padding, groups=1, bias=bias)
        self.gn = nn.GroupNorm(1, out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.gn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvGnRelu3(nn.Module):
    """Bottle neck structure"""

    def __init__(self, in_channels, out_channels, ksize, stride, padding, ratio, do_act=True, bias=True):
        super(BottConvGnRelu3, self).__init__()
        self.conv1 = ConvGnRelu3(in_channels, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv2 = ConvGnRelu3(in_channels//ratio, in_channels//ratio, ksize, stride, padding, do_act=True, bias=bias)
        self.conv3 = ConvGnRelu3(in_channels//ratio, out_channels, ksize, stride, padding, do_act=do_act, bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, stride, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=True))
            else:
                layers.append(ConvGnRelu3(channels, channels, ksize, stride, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        output = self.act(input + output)

        return output


class BottResidualBlock3(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ksize, stride, padding, ratio, num_convs):
        super(BottResidualBlock3, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=True))
            else:
                layers.append(BottConvGnRelu3(channels, channels, ksize, stride, padding, ratio, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)


class DownBlock(nn.Module):
  """ downsample block of v-net """

  def __init__(self, in_channels, out_channels, num_convs, compression=False, ratio=4):
    super(DownBlock, self).__init__()
    self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
    self.down_gn = nn.GroupNorm(1, num_channels=out_channels)
    self.down_act = nn.ReLU(inplace=True)
    if compression:
      self.rblock = BottResidualBlock3(out_channels, 3, 1, 1, ratio, num_convs)
    else:
      self.rblock = ResidualBlock3(out_channels, 3, 1, 1, num_convs)

  def forward(self, input):
    out = self.down_act(self.down_gn(self.down_conv(input)))
    out = self.rblock(out)
    return out


class UpBlock(nn.Module):
  """ Upsample block of v-net """

  def __init__(self, in_channels, out_channels, num_convs, compression=False, ratio=4):
    super(UpBlock, self).__init__()
    if compression:
      self.rblock = BottResidualBlock3(in_channels, 3, 1, 1, ratio, num_convs)
    else:
      self.rblock = ResidualBlock3(in_channels, 3, 1, 1, num_convs)

    self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
    self.up_gn = nn.GroupNorm(1, out_channels)
    self.up_act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.rblock(input)
    out = self.up_act(self.up_gn(self.up_conv(out)))
    return out


# backbone
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.num_layers = 5
        assert len(in_channels) == len(out_channels) == self.num_layers

        self.layer1 = InputBlock(in_channels[0], out_channels[0])
        self.layer2 = DownBlock(in_channels[1], out_channels[1], 1, compression=False)
        self.layer3 = DownBlock(in_channels[2], out_channels[2], 2, compression=True)
        self.layer4 = DownBlock(in_channels[3], out_channels[3], 3, compression=True)
        self.layer5 = DownBlock(in_channels[4], out_channels[4], 3, compression=True)

    def forward(self, input, ext_fms=None):
        assert isinstance(input, torch.Tensor)

        if ext_fms is None:
            out1 = self.layer1(input)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)

        else:
            out1 = self.layer1(input + ext_fms[0])
            out2 = self.layer2(out1 + ext_fms[1])
            out3 = self.layer3(out2 + ext_fms[2])
            out4 = self.layer4(out3 + ext_fms[3])
            out5 = self.layer5(out4 + ext_fms[4])

        return [out1, out2, out3, out4, out5]


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.num_layers = 5
        assert len(in_channels) == len(out_channels) == 5

        self.layer1 = OutputBlock(in_channels[0], out_channels[0])
        self.layer2 = UpBlock(in_channels[1], out_channels[1], 1, compression=False)
        self.layer3 = UpBlock(in_channels[2], out_channels[2], 2,  compression=False)
        self.layer4 = UpBlock(in_channels[3], out_channels[3], 3, compression=True)
        self.layer5 = UpBlock(in_channels[4], out_channels[4], 3, compression=True)

    def forward(self, input, skip, ext_fms=None):
        assert isinstance(input, torch.Tensor)

        if ext_fms is None:
            out5 = self.layer5(input)
            out4 = self.layer4(torch.cat((out5, skip[3]), 1))
            out3 = self.layer3(torch.cat((out4, skip[2]), 1))
            out2 = self.layer2(torch.cat((out3, skip[1]), 1))
            out1 = self.layer1(torch.cat((out2, skip[0]), 1))

        else:
            out5 = self.layer5(input + ext_fms[4])
            out4 = self.layer4(torch.cat((out5, skip[3]), 1) + ext_fms[3])
            out3 = self.layer3(torch.cat((out4, skip[2]), 1) + ext_fms[2])
            out2 = self.layer2(torch.cat((out3, skip[1]), 1) + ext_fms[1])
            out1 = self.layer1(torch.cat((out2, skip[0]), 1) + ext_fms[0])

        return [out1, out2, out3, out4, out5]


# Header
class SegmentationHeader(nn.Module):
    """ Segmentation header """

    def __init__(self, in_channels, out_channels):
        super(SegmentationHeader, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gn = nn.GroupNorm(1, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.gn(self.conv(input))
        out = self.softmax(out)

        return out


class GlobalLocalNetwork(nn.Module):
    """ global local neural network """

    def __init__(self, num_in_channels, num_out_channels):
        super(GlobalLocalNetwork, self).__init__()

        self.global_encoder = Encoder([num_in_channels, 16, 32, 64, 128], [16, 32, 64, 128, 256])
        self.global_decoder = Decoder([32, 64, 128, 256, 256], [32, 16, 32, 64, 128])
        self.global_header = SegmentationHeader(32, num_out_channels)

        self.local_encoder = Encoder([num_in_channels, 16, 32, 64, 128], [16, 32, 64, 128, 256])
        self.local_decoder = Decoder([32, 64, 128, 256, 256], [32, 16, 32, 64, 128])
        self.local_header = SegmentationHeader(32, num_out_channels)

        self.ensemble_header = SegmentationHeader(64, num_out_channels)

    def max_stride(self):
        return 16

    def _crop_and_upsample_global(self, fms_global, start_coords, ratio):
        """
        Crop global patches and then up-sample these cropped patches
        """
        batch, ch, dim_z, dim_y, dim_x = fms_global.shape
        _batch, _dim = start_coords.shape
        assert batch == _batch and _dim == 3

        cropped_patches = []
        cropped_size = [dim_z // ratio, dim_y // ratio, dim_x // ratio]

        for idx in range(batch):
            s_z, s_y, s_x = int(start_coords[idx][2] // ratio), int(start_coords[idx][1]) // ratio, int(start_coords[idx][0] // ratio)
            e_z, e_y, e_x = int(s_z + cropped_size[2]), int(s_y + cropped_size[1]), int(s_x + cropped_size[0])
            cropped_patch = fms_global[idx, :, s_z:e_z, s_y:e_y, s_x:e_x]
            cropped_patches.append(torch.unsqueeze(cropped_patch, 0))
        cropped_patches = torch.cat(cropped_patches, 0)
        upsampled_cropped_patches = \
            F.interpolate(cropped_patches, (dim_z, dim_y, dim_x), mode='trilinear', align_corners=True)

        return upsampled_cropped_patches

    def forward(self, input_global, input_local=None, mode=3, coords=None, ratio=None):

        if mode == 1:
            # train global model only
            fms_g_e = self.global_encoder(input_global)
            fms_g_d_skip = [fms_g_e[0], fms_g_e[1], fms_g_e[2], fms_g_e[3]]
            fms_g_d = self.global_decoder(fms_g_e[4], fms_g_d_skip)

            return self.global_header(fms_g_d[0])

        elif mode == 2:
            # train local model only
            fms_l_e = self.local_encoder(input_local)
            fms_l_d_skip = [fms_l_e[0], fms_l_e[1], fms_l_e[2], fms_l_e[3]]
            fms_l_d = self.local_decoder(fms_l_e[4], fms_l_d_skip)

            return self.local_header(fms_l_d[0])

        elif mode == 3:
            # train global to local model
            fms_g_e = self.global_encoder(input_global)
            fms_g_d_skip = [fms_g_e[0], fms_g_e[1], fms_g_e[2], fms_g_e[3]]
            fms_g_d = self.global_decoder(fms_g_e[4], fms_g_d_skip)

            fms_l_e = self.local_encoder(input_local)
            fms_l_d_skip = [fms_l_e[0], fms_l_e[1], fms_l_e[2], fms_l_e[3]]
            fms_l_d = self.local_decoder(fms_l_e[4], fms_l_d_skip)

            # crop from the global patch
            cropped_upsampled_fms_g_d = self._crop_and_upsample_global(fms_g_d[0], coords, ratio)
            fms_ensemble = torch.cat((cropped_upsampled_fms_g_d, fms_l_d[0]), 1)

            return self.global_header(fms_g_d[0]), self.local_header(fms_l_d[0]), self.ensemble_header(fms_ensemble)

        else:
            raise ValueError('Unsupported value type.')