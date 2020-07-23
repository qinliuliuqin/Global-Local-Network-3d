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

    def forward(self, input, feature_maps=None):
        assert isinstance(input, torch.Tensor)

        if feature_maps is None:
            out1 = self.layer1(input)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)

        else:
            if feature_maps[0] is None:
                out1 = self.layer1(input)
            else:
                out1 = self.layer1(torch.cat((input, feature_maps[0]), 1))
            out2 = self.layer2(torch.cat((out1, feature_maps[1]), 1))
            out3 = self.layer3(torch.cat((out2, feature_maps[2]), 1))
            out4 = self.layer4(torch.cat((out3, feature_maps[3]), 1))
            out5 = self.layer5(torch.cat((out4, feature_maps[4]), 1))

        return [out1, out2, out3, out4, out5]


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.num_layers = 5
        assert len(in_channels) == len(out_channels) == 5

        self.layer1 = UpBlock(in_channels[0], out_channels[0], 3, compression=True)
        self.layer2 = UpBlock(in_channels[1], out_channels[1], 3, compression=True)
        self.layer3 = UpBlock(in_channels[2], out_channels[2], 2,  compression=False)
        self.layer4 = UpBlock(in_channels[3], out_channels[3], 1, compression=False)
        self.layer5 = OutputBlock(in_channels[4], out_channels[4])

    def forward(self, input, feature_maps=None):
        assert isinstance(input, torch.Tensor)
        assert len(feature_maps) == self.num_layers

        if feature_maps is None:
            out1 = self.layer1(input)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)

        else:
            if feature_maps[0] is None:
                out1 = self.layer1(input)
            else:
                out1 = self.layer1(torch.cat((input, feature_maps[0]), 1))
            out2 = self.layer2(torch.cat((out1, feature_maps[1]), 1))
            out3 = self.layer3(torch.cat((out2, feature_maps[2]), 1))
            out4 = self.layer4(torch.cat((out3, feature_maps[3]), 1))
            out5 = self.layer5(torch.cat((out4, feature_maps[4]), 1))

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
        self.global_decoder = Decoder([256, 256, 128, 64, 32], [128, 64, 32, 16, 32])
        self.global_header = SegmentationHeader(32, num_out_channels)

        self.local_encoder = Encoder([num_in_channels, 32, 64, 128, 256], [16, 32, 64, 128, 256])
        self.local_decoder = Decoder([512, 384, 192, 96, 48], [128, 64, 32, 16, 32])
        self.local_header = SegmentationHeader(32, num_out_channels)


    def forward(self, input_global, input_local, coords):

        # global branch
        fms_global_encoder = self.global_encoder(input_global)
        fms_global_skip = [None, fms_global_encoder[3], fms_global_encoder[2], fms_global_encoder[1],
                           fms_global_encoder[0]]
        fms_global_decoder = self.global_decoder(fms_global_encoder[4], fms_global_skip)

        # global to local upsampling
        # TO BE DONE

        # concatenate global and local feature maps
        fms_g2l_encoder_skip = [None, fms_global_encoder[0], fms_global_encoder[1], fms_global_encoder[2],
                        fms_global_encoder[3]]
        fms_local_encoder = self.local_encoder(input_local, fms_g2l_encoder_skip)
        fms_local_skip = [fms_global_encoder[-1],
                          torch.cat((fms_global_decoder[0], fms_local_encoder[3]), 1),
                          torch.cat((fms_global_decoder[1], fms_local_encoder[2]), 1),
                          torch.cat((fms_global_decoder[2], fms_local_encoder[1]), 1),
                          torch.cat((fms_global_decoder[3], fms_local_encoder[0]), 1)]

        fms_local_decoder = self.local_decoder(fms_local_encoder[4], fms_local_skip)

        return fms_local_decoder[-1]