# Point Segmentation for Semantic-KITTI
* This is very initial code for point segmentation. Now, the codes are not organized and hard-coded in many parts.
* The purpose of this code is to investigate parts where we can improve upon. For now, I am thinking about adding...
    1. Multi-resolution in/out - U-Net architecture is spatially hierarchical design for feature extraction. I think we can add multi-resolution input/output guidance
    2. Have an unique definition of voxel (like gaussian thing from Hiro). For now, I have (1) aggregation and (2) offest. I think aggregation doesn't help much. But offset looks like it is helping (not sure by how much in number though)
    3. pre-training - implement gaussian spaltting render for pre-training the feature extraction
* Now this only supports Semantic-KITTI

## How to use
```
CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 4 --num_workers 16 --data_config_file ./data_config/semantic-kitti.yaml --data_dir /nfs/wattrel/data/md0/datasets/kitti --save_dir ./test
```