import numpy as np
import torch
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, down_sample_ratio, loss_func, loss_weight, use_gpu):
        self.loss_func = loss_func
        assert len(loss_weight) == 3
        self.loss_weight = torch.FloatTensor(loss_weight)
        self.down_sample_ratio = down_sample_ratio
        self.use_gpu = use_gpu

        self.global_patches = None
        self.global_masks = None
        self.global_to_local_coords = []

    def generate_global_patches(self, images, masks):
        assert isinstance(images, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert images.shape == masks.shape

        batch, ch, dim_z, dim_y, dim_x = images.shape
        self.down_sampled_size = [val // self.down_sample_ratio for val in [dim_z, dim_y, dim_x]]
        self.global_patches = F.interpolate(images, self.down_sampled_size, mode='trilinear', align_corners=True)
        self.global_masks = F.interpolate(masks, self.down_sampled_size, mode='nearest')

        for batch_idx in range(batch):
            start_coord = [np.random.randint(self.down_sampled_size[idx] - self.down_sampled_size[idx] // self.down_sample_ratio) for idx in range(3)]
            self.global_to_local_coords.append(start_coord)

    def train_global_to_local(self, images, masks, model):
        model.train()
        self.generate_global_patches(images, masks)

        # crop local patches
        local_patches, local_masks = [], []
        for idx, start_coords in enumerate(self.global_to_local_coords):
            local_start_coords = [start_coords[idy] * self.down_sample_ratio for idy in range(3)]
            local_end_coords = [local_start_coords[idy] + self.down_sampled_size[idy] for idy in range(3)]

            local_patch = images[idx, :, local_start_coords[0]:local_end_coords[0], local_start_coords[1]:local_end_coords[1], local_start_coords[2]:local_end_coords[2]]
            local_patches.append(torch.unsqueeze(local_patch, dim=0))
            local_mask = masks[idx, :, local_start_coords[0]:local_end_coords[0], local_start_coords[1]:local_end_coords[1], local_start_coords[2]:local_end_coords[2]]
            local_masks.append(torch.unsqueeze(local_mask, dim=0))

        local_patches, local_masks = torch.cat(local_patches), torch.cat(local_masks)

        global_to_local_coords = torch.FloatTensor(self.global_to_local_coords)
        self.global_to_local_coords = []

        if self.use_gpu:
            self.global_patches, self.global_masks, global_to_local_coords, local_patches, local_masks = \
                self.global_patches.cuda(), self.global_masks.cuda(), global_to_local_coords.cuda(), \
                local_patches.cuda(), local_masks.cuda()

        out_global, out_local, out_global2local = \
            model(self.global_patches, local_patches, 3, global_to_local_coords, self.down_sample_ratio)

        loss_global = self.loss_func(out_global, self.global_masks)
        loss_local = self.loss_func(out_local, local_masks)
        loss_g2l = self.loss_func(out_global2local, local_masks)

        loss = loss_global * self.loss_weight[0] + loss_local * self.loss_weight[1] + loss_g2l * self.loss_weight[2]
        loss.backward()

        return loss
