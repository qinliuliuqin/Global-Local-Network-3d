import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from utils.image_tools import convert_image_to_tensor, convert_tensor_to_image, bbox_partition_by_fixed_voxel_size


def down_sample_image_tensor(image_t, down_sample_ratio=4, mode='trilinear', align_corners=None):
    """ Down-sample the input image tensor along the spatial axises"""
    assert isinstance(image_t, torch.Tensor)
    assert image_t.dim() == 5
    batch, ch, dim_z, dim_y, dim_x = image_t.shape
    down_sampled_size = [int(dim // down_sample_ratio) for dim in [dim_z, dim_y, dim_x]]
    down_sampled_image_t = F.interpolate(image_t, down_sampled_size, mode=mode, align_corners=align_corners)

    return down_sampled_image_t


def generate_global_and_local_patches(images, masks=None, down_sample_ratio=4):
    """ Generate the down-sampled global images and the cropped local images for training. """
    assert isinstance(images, torch.Tensor)
    if masks is not None:
        assert isinstance(masks, torch.Tensor)
        assert images.dim() == masks.dim() == 5
        assert np.all([images.shape[idx] - masks.shape[idx] == 0] for idx in range(5))

    # get the down-sampled global patches
    global_patches = down_sample_image_tensor(images, down_sample_ratio, 'trilinear', True)

    global_masks = None
    if masks is not None:
        global_masks = down_sample_image_tensor(masks, down_sample_ratio, 'nearest')

    batch, _, dim_z, dim_y, dim_x = global_patches.shape
    down_sampled_size = [dim_z, dim_y, dim_x]

    global_to_local_coords = []
    for batch_idx in range(batch):
        start_coord = [
            np.random.randint(down_sampled_size[idx] - down_sampled_size[idx] // down_sample_ratio) for
            idx in range(3)]
        global_to_local_coords.append(start_coord)

    # get local patches
    local_patches, local_masks = [], []
    for idx, start_coords in enumerate(global_to_local_coords):
        sp = [start_coords[idy] * down_sample_ratio for idy in range(3)]
        ep = [sp[idy] + down_sampled_size[idy] for idy in range(3)]

        local_patch = images[idx, :, sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        local_patches.append(torch.unsqueeze(local_patch, dim=0))

        if masks is not None:
            local_mask = masks[idx, :, sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
            local_masks.append(torch.unsqueeze(local_mask, dim=0))

    local_patches = torch.cat(local_patches)
    local_masks = torch.cat(local_masks) if local_masks is not None else None

    global_to_local_coords = torch.FloatTensor(global_to_local_coords)

    return global_patches, global_masks, local_patches, local_masks, global_to_local_coords


class Trainer(object):
    def __init__(self, model, optimizer, down_sample_ratio, loss_func, loss_weight, use_gpu):
        self.model = model
        self.model.train()

        self.optimizer = optimizer
        self.loss_func = loss_func

        assert len(loss_weight) == 3
        self.loss_weight = torch.FloatTensor(loss_weight)
        self.loss_weight = self.loss_weight / torch.sum(self.loss_weight)

        self.down_sample_ratio = down_sample_ratio
        self.use_gpu = use_gpu

    def train(self, images, masks, mode=3):
        assert isinstance(images, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert images.dim() == masks.dim() == 5

        global_patches, global_masks, local_patches, local_masks, global_to_local_coords = \
            generate_global_and_local_patches(images, masks, self.down_sample_ratio)

        if self.use_gpu:
            global_patches, global_masks = global_patches.cuda(), global_masks.cuda()
            local_patches, local_masks = local_patches.cuda(), local_masks.cuda()
            global_to_local_coords = global_to_local_coords.cuda()

        # clear previous gradients
        self.optimizer.zero_grad()

        out_global, out_local, out_global2local = \
            self.model(global_patches, local_patches, mode, global_to_local_coords, self.down_sample_ratio)

        loss_global = self.loss_func(out_global, global_masks)
        loss_local = self.loss_func(out_local, local_masks)
        loss_g2l = self.loss_func(out_global2local, local_masks)

        loss = loss_global * self.loss_weight[0] + loss_local * self.loss_weight[1] + loss_g2l * self.loss_weight[2]
        loss.backward()

        # update weights
        self.optimizer.step()

        return loss.item()


class Evaluator(object):
    def __init__(self, model, metrics, crop_size, down_sample_ratio, normalizer):
        self.model = model
        self.model.eval()

        self.metrics = metrics
        self.crop_size = crop_size
        self.down_sample_ratio = down_sample_ratio
        self.normalizer = normalizer

    def evaluate(self, images, masks):
        assert isinstance(images, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert images.dim() == masks.dim() == 5
        assert np.all([images.shape[idx] == masks.shape[idx] for idx in range(3)])
        assert images.shape[0] == 1

        dim_z, dim_y, dim_x = images.shape[2], images.shape[3], images.shape[4]

        # get the start voxels
        bbox_start_voxel, bbox_end_voxel = [0, 0, 0], [images.shape[4], images.shape[3], images.shape[2]]
        start_voxels, end_voxels = bbox_partition_by_fixed_voxel_size(bbox_start_voxel, bbox_end_voxel, self.crop_size)

        # crop image and mask into patches
        cropped_images, cropped_masks = [], []
        for idx in range(len(start_voxels)):

            start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]
            if end_voxel[0] <= dim_x and end_voxel[1] < dim_y and end_voxel[2] < dim_x:
                cropped_image = images[:, :, start_voxel[2]:end_voxel[2], start_voxel[1]:end_voxel[1], start_voxel[0]:end_voxel[0]]
                cropped_mask = masks[:, :, start_voxel[2]:end_voxel[2], start_voxel[1]:end_voxel[1], start_voxel[0]:end_voxel[0]]

            else:
                real_end_voxel = end_voxel[:]
                real_end_voxel[0] = min(end_voxel[0], dim_x)
                real_end_voxel[1] = min(end_voxel[1], dim_y)
                real_end_voxel[2] = min(end_voxel[2], dim_z)
                cropped_image = images[0, 0, start_voxel[2]:real_end_voxel[2], start_voxel[1]:real_end_voxel[1], start_voxel[0]:real_end_voxel[0]]
                cropped_mask = masks[0, 0, start_voxel[2]:real_end_voxel[2], start_voxel[1]:real_end_voxel[1], start_voxel[0]:real_end_voxel[0]]

                cropped_image = convert_tensor_to_image(cropped_image, dtype=np.float32)
                cropped_mask = convert_tensor_to_image(cropped_mask, dtype=np.float32)

                padding_size = [end_voxel[0] - real_end_voxel[0], end_voxel[1] - real_end_voxel[1], end_voxel[2] - real_end_voxel[2]]
                cropped_image = sitk.ConstantPad(cropped_image, [0, 0, 0], padding_size)
                cropped_mask = sitk.ConstantPad(cropped_mask, [0, 0, 0], padding_size)

                # convert image to tensor
                cropped_image = convert_image_to_tensor(cropped_image)
                cropped_image = torch.unsqueeze(cropped_image, 0)
                cropped_mask = convert_image_to_tensor(cropped_mask)
                cropped_mask = torch.unsqueeze(cropped_mask, 0)

            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)

        # test the cropped patches
        with torch.no_grad():
            for idx in range(len(cropped_images)):
                image, mask = cropped_images[idx], cropped_masks[idx]
                global_patches, global_masks, local_patches, local_masks, global_to_local_coords = \
                    generate_global_and_local_patches(image, mask, self.down_sample_ratio)

                out_global, out_local, out_g2l = \
                    self.model(global_patches, local_patches, 3, global_to_local_coords, self.down_sample_ratio)
                # res_g = self.metrics(out_global, global_masks)
                # res_l = self.metrics(out_local, local_masks)
                # res_g2l = self.metrics(out_g2l, local_masks)

        # return res_g2l
        return 0