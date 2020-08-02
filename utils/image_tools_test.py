from utils.image_tools import bbox_partition_by_fixed_voxel_size

bbox_start_voxel, bbox_end_voxel = [0, 0, 0], [96, 96, 96]
partition_size = [164, 64, 64]

start_voxels, end_voxels = bbox_partition_by_fixed_voxel_size(bbox_start_voxel, bbox_end_voxel, partition_size)
print(start_voxels, end_voxels)