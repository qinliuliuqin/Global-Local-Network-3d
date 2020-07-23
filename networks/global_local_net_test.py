import torch

from networks.global_local_net import GlobalLocalNetwork


def test_global_local_network():

    batch, in_ch, dim_z, dim_y, dim_x = 2, 1, 32, 32, 32
    in_global_images = torch.randn([batch, in_ch, dim_z, dim_y, dim_x])
    in_local_images = torch.randn([batch, in_ch, dim_z, dim_y, dim_x])

    out_ch = 4
    mode = 3
    model = GlobalLocalNetwork(in_ch, out_ch)
    seg = model(in_global_images, in_local_images, mode, None)

    print(seg.shape)


if __name__ == '__main__':

    test_global_local_network()
