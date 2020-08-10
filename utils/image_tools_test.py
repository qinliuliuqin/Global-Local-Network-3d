from utils.image_tools import bbox_partition_by_fixed_voxel_size

def test_bbox_partition_case_1():
    bbox_start_voxel, bbox_end_voxel = [0, 0, 0], [96, 96, 96]
    partition_size = [164, 64, 64]

    start_voxels, end_voxels = bbox_partition_by_fixed_voxel_size(bbox_start_voxel, bbox_end_voxel, partition_size)
    print(start_voxels, end_voxels)

def test_bbox_partition_case_2():
    bbox_start_voxel, bbox_end_voxel = [0, 0, 0], [400, 400, 560]
    partition_size = [512, 512, 512]

    start_voxels, end_voxels = bbox_partition_by_fixed_voxel_size(bbox_start_voxel, bbox_end_voxel, partition_size)
    print(start_voxels, end_voxels)

test_bbox_partition_case_1()
test_bbox_partition_case_2()