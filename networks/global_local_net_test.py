import torch

from networks.global_local_net import GlobalLocalNetwork


def test_global_local_network():

    batch, ch, dim_z, dim_y, dim_x = 2, 1, 32, 32, 32
    in_global_images = torch.randn([batch, ch, dim_z, dim_y, dim_x])
    in_local_images = torch.randn([batch, ch, dim_z, dim_y, dim_x])


    model = GlobalLocalNetwork(1, 4)
    res = model(in_global_images, in_local_images, None)

    print(res.shape)


if __name__ == '__main__':

    test_global_local_network()
