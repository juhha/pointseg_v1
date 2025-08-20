import argparse, os, glob, numpy as np, yaml, tqdm, json

from codes.model_utils import VoxelUNet, PVSeg
from codes.data_utils import SimpleKittiDataset, collate_fn
from codes.loss_utils import calc_loss

import torch
import torch.nn as nn

def main(args):
    # get user arguments
    data_dir = args.data_dir
    data_config_file = args.data_config_file
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok = True)
    #######
    # Load data loader
    # data_dir = '/nfs/wattrel/data/md0/datasets/kitti'
    odometry_dir = os.path.join(data_dir, 'odometry', 'dataset', 'sequences')
    seq_train = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    seq_val = ["08"]
    seq_test = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    
    list_files = sorted(glob.glob(os.path.join(odometry_dir, '**', 'velodyne', '*.bin')))
    list_sequences = sorted(set([f.split('/')[-3] for f in list_files]))
    list_sequences_train = [seq for seq in list_sequences if seq in seq_train]
    list_sequences_val = [seq for seq in list_sequences if seq in seq_val]
    list_sequences_test = [seq for seq in list_sequences if seq in seq_test]

    # train files
    list_files_label_train = [f for seq in list_sequences_train for f in sorted(glob.glob(os.path.join(odometry_dir, seq, 'labels', '*.label')))]
    list_files_vel_train = [f.replace('labels', 'velodyne').replace('label', 'bin') for f in list_files_label_train]
    list_files_vox_label_train = [f.replace('labels', 'voxels') for f in list_files_label_train]
    list_files_vox_bin_train = [f.replace('labels', 'voxels').replace('label', 'bin') for f in list_files_label_train]
    # validation files
    list_files_label_val = [f for seq in list_sequences_val for f in sorted(glob.glob(os.path.join(odometry_dir, seq, 'labels', '*.label')))]
    list_files_vel_val = [f.replace('labels', 'velodyne').replace('label', 'bin') for f in list_files_label_val]
    list_files_vox_label_val = [f.replace('labels', 'voxels') for f in list_files_label_val]
    list_files_vox_bin_val = [f.replace('labels', 'voxels').replace('label', 'bin') for f in list_files_label_val]
    # test files
    list_files_label_test = [f for seq in list_sequences_test for f in sorted(glob.glob(os.path.join(odometry_dir, seq, 'labels', '*.label')))]
    list_files_vel_test = [f.replace('labels', 'velodyne').replace('label', 'bin') for f in list_files_label_test]
    list_files_vox_label_test = [f.replace('labels', 'voxels') for f in list_files_label_test]
    list_files_vox_bin_test = [f.replace('labels', 'voxels').replace('label', 'bin') for f in list_files_label_test]
    
    list_files_train = [
        {
            'lidar': f_vel,
            'label': f_label
        }
        for f_vel, f_label in zip(list_files_vel_train, list_files_label_train)
    ]

    list_files_val = [
        {
            'lidar': f_vel,
            'label': f_label
        }
        for f_vel, f_label in zip(list_files_vel_val, list_files_label_val)
    ]
    # Define data loaders
    ds = SimpleKittiDataset(list_files_train, data_config_file)
    dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, num_workers = num_workers, collate_fn = collate_fn, shuffle = True)

    ds_val = SimpleKittiDataset(list_files_val, data_config_file)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size = 1, num_workers = num_workers, collate_fn = collate_fn, shuffle = False)
    
    # FIX: for now, I have hard-coded network. Later, will have to add customizable definition
    net_v = VoxelUNet(4, 19, [32,64,128,256],3,2).cuda()
    net_p = nn.Linear(4, 16).cuda()
    net_out = nn.Linear(48, 20).cuda()
    vox_unit = [0.05, 0.05, 0.05]
    net = PVSeg(net_v, net_p, net_out, vox_unit).cuda()

    num_params = 0
    for p in net.parameters():
        num_params += np.prod(p.shape)
    print(f"{num_params:,}")
    # FIX: tune this part later as well
    optimizer = torch.optim.Adam(net.parameters())

    class_map = {i:c for i, c in enumerate(['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle','person','bicyclist','motorcyclist','road','parking','sidewalk','other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'])}
    
    curr_epoch = 0
    progress = {}

    max_epochs = 200
    for epoch in range(curr_epoch, max_epochs):
        pbar = tqdm.tqdm(total = len(dl), position = 0, desc = f'Train ({epoch+1}/{max_epochs})')
        list_loss = []
        for idx, batch in enumerate(dl):
            lidar, label, list_fdict = batch
            lidar = lidar.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = net(lidar)
            # loss_fn = nn.CrossEntropyLoss()
            loss = calc_loss(out, label)
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'loss': f'{np.mean(list_loss):.2f}'})
        pbar.close()
        progress[epoch] = {'loss_train': np.mean(list_loss)}
        count_intersections = np.zeros(20)
        count_union = np.zeros(20)
        if True: #(epoch+1) % 10 == 0: # for now, simply run evaluation for each epoch
            pbar = tqdm.tqdm(total = len(dl_val), position = 0)
            for batch in dl_val:
                lidar, label, list_fdict = batch
                lidar = lidar.cuda()
                label = label.cuda()
                with torch.no_grad():
                    out = net(lidar)
                    pred = torch.argmax(out, dim = 1)
                    for i in range(20):
                        gt = label == i
                        p = pred == i
                        # collect metrics
                        count_intersections[i] += (gt & p).sum().item()
                        count_union[i] += (gt | p).sum().item()
                pbar.update(1)
                ious = count_intersections / count_union
            pbar.close()
            print(ious)
            print(ious[1:].mean())
            progress[epoch]['iou'] = list(ious)
        json.dump(progress, open(os.path.join(save_dir, 'progress.json'), 'w'))
        curr_epoch = epoch + 1
        torch.save(net.state_dict(), os.path.join(save_dir, 'latest.pt'))

if __name__ == '__main__':
    
    # data_dir = args.data_dir
    # data_config_file = args.data_config_file
    # batch_size = args.batch_size
    # num_workers = args.num_workers
    # save_dir = args.save_dir
    # collect user arguments
    parser = argparse.ArgumentParser(description="Configuration for PAIRNET model training.")
    parser.add_argument('--data_dir', type=str, default='/nfs/wattrel/data/md0/datasets/kitti', help='data directory')
    parser.add_argument('--data_config_file', type=str, default='./data_config/semantic-kitti.yaml', help='data configuration yaml file')
    parser.add_argument('--num_workers', type=int, default=16, help='# of workers for processing') 
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 4)')
    parser.add_argument("--save_dir", dest="save_dir", type=str, default='./temp', help="checkpoint directory")
    args = parser.parse_args()
    
    main(args)
    