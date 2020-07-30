import SimpleITK as sitk
import torch

from utils.helper import down_sample_image_tensor


def test():
    image_path = '/mnt/projects/CT_Dental/data_v2/case_1_cbct_trauma/seg_head.nii.gz'
    image = sitk.ReadImage(image_path)

    image_size = image.GetSize()
    cropped_image = image[:image_size[0] + 20, :image_size[1] + 1, :image_size[2] + 1]
    cropped_image = sitk.ConstantPad(cropped_image, [12, 12, 12], [24, 24, 32])

    cropped_image_path = '/home/ql/debug/cropped_image.nii.gz'
    sitk.WriteImage(cropped_image, cropped_image_path, True)

def test_down_sample_image_tensor():
    image = torch.randn(1, 2, 4, 8, 16)
    down_sample_ratio = 4
    down_sampled_image = down_sample_image_tensor(image, down_sample_ratio, 'trilinear', True)

    assert down_sampled_image.dim() == image.dim()
    assert down_sampled_image.shape[0] == image.shape[0]
    assert down_sampled_image.shape[1] == image.shape[1]
    assert down_sampled_image.shape[2] == image.shape[2] // down_sample_ratio
    assert down_sampled_image.shape[3] == image.shape[3] // down_sample_ratio
    assert down_sampled_image.shape[4] == image.shape[4] // down_sample_ratio


if __name__ == '__main__':

    test_down_sample_image_tensor()