import numpy as np
import torch
import torch.nn as nn

from torchsparse.nn import functional as F
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate
import torchsparse.nn as spnn

F.set_kmap_mode('hashmap')
F.set_downsample_mode('minkowski') 
conv_config = F.conv_config.get_default_conv_config()
F.conv_config.set_global_conv_config(conv_config)

def groupby_scatter(indices: torch.Tensor, values: torch.Tensor, agg = 'max'):
    """
    Perform a group-by-min in PyTorch for:
      - indices: shape (N,)  (each entry is a group ID, e.g. 0..K-1)
      - values: shape (N, D) (the data we want to aggregate per group)
      - add: str (aggregation type)
    
    Returns:
        out: shape (num_groups, D), where out[g] = min(...) of all values[i]
             with indices[i] == g, applied elementwise across the D dimensions.
    """
    device = values.device
    num_groups = int(indices.max().item()) + 1     # e.g., groups from 0..max
    D = values.shape[1]
    index_expanded = indices[:,None].expand(-1, D)  # shape (N, D)
    if agg == 'min':
        out = torch.full((num_groups, D), float('inf'), device=values.device).to(device)
        out.scatter_reduce_(dim=0, index=index_expanded.long(), src=values, reduce="min")
    elif agg == 'max':
        out = torch.full((num_groups, D), float('-inf'), device=values.device).to(device)
        out.scatter_reduce_(dim=0, index=index_expanded.long(), src=values, reduce="max")
    elif agg == 'count':
        ones = torch.ones_like(values, device=device)
        count_out = torch.zeros((num_groups, D), device=device)
        count_out.scatter_reduce_(0, index_expanded, ones, reduce="sum")
        out = count_out[:,:1] # return a vector
    elif agg == 'sum':
        sum_out = torch.zeros((num_groups, D), device=device)
        sum_out.scatter_reduce_(0, index_expanded, values, reduce="sum")
        out = sum_out
    elif agg == 'avg':
        # We'll compute population std, i.e. sqrt(E[x^2] - (E[x])^2)
        # 1) sum of x
        sum_out = torch.zeros((num_groups, D), device=device)
        sum_out.scatter_reduce_(0, index_expanded, values, reduce="sum")
        # 2) count
        ones = torch.ones_like(values, device=device)
        count_out = torch.zeros((num_groups, D), device=device)
        count_out.scatter_reduce_(0, index_expanded, ones, reduce="sum")
        # average
        denom    = count_out.clamp_min(1.0)    # avoid div-by-zero
        mean     = sum_out / denom
        out = mean
    elif agg == 'std':
        # We'll compute population std, i.e. sqrt(E[x^2] - (E[x])^2)
        # 1) sum of x
        sum_out = torch.zeros((num_groups, D), device=device)
        sum_out.scatter_reduce_(0, index_expanded, values, reduce="sum")

        # 2) sum of x^2
        sum_sq_out = torch.zeros((num_groups, D), device=device)
        sum_sq_out.scatter_reduce_(0, index_expanded, values*values, reduce="sum")

        # 3) count
        ones = torch.ones_like(values, device=device)
        count_out = torch.zeros((num_groups, D), device=device)
        count_out.scatter_reduce_(0, index_expanded, ones, reduce="sum")

        # E[x]   = sum_out / count_out
        # E[x^2] = sum_sq_out / count_out
        denom    = count_out.clamp_min(1.0)    # avoid div-by-zero
        mean     = sum_out / denom
        mean_sq  = sum_sq_out / denom
        var      = mean_sq - mean*mean
        var      = torch.clamp(var, min=0.0)   # numerical safety
        std_     = var.sqrt()
        # If a group had count=0, let's keep its std=0 (or could do NaN)
        zero_mask = (count_out == 0)
        std_ = torch.where(zero_mask, torch.zeros_like(std_), std_)
        out = std_
    return out

class BaseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_channels,
        kernel_size: int = 3,
        num_layers: int = 2,
        act: tuple = ('leaky_relu', {'negative_slope': 0.2, 'inplace': True}),
        add_dilation: bool = False,
        norm: tuple = ('instance', {})
    ):
        super().__init__()
        # act = ('relu', {})
        # norm = ('batch', {})
        act_fn_map = {
            'leaky_relu': spnn.LeakyReLU,
            'relu': spnn.ReLU,
        }
        act_fn = act_fn_map[act[0]]
        act_param = act[1]
        norm_fn_map = {
            'instance': spnn.InstanceNorm,
            'batch': spnn.BatchNorm,
        }
        if norm is None:
            norm_fn = None
        else:
            norm_fn = norm_fn_map[norm[0]]
            norm_param = norm[1]
        layers = []
        inc = in_channels
        outc = feature_channels
        dilation = 1
        for i in range(num_layers-1):
            padding = ((kernel_size-1) // 2)  * dilation
            layers.append(spnn.Conv3d(inc, outc, kernel_size, 1, padding, dilation))
            if norm is not None:
                layers.append(norm_fn(outc, **norm_param))
            layers.append(act_fn(**act_param))
            if add_dilation:
                dilation += 1
            inc = outc
        padding = ((kernel_size-1) // 2)  * dilation
        layers.append(spnn.Conv3d(inc, out_channels, kernel_size, 1, padding, dilation))
        layers.append(act_fn(**act_param))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class UpsampleSparse(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self, x, scale = 2):
        device = x.F.device
        mask = torch.zeros(x.spatial_range).to(device)
        mask[x.coords[:,0], x.coords[:,1], x.coords[:,2], x.coords[:,3]] = 1
        mask = mask.unsqueeze(1)
        mask = torch.nn.functional.interpolate(mask, scale_factor = scale, mode = 'nearest')
        new_coord = torch.stack(torch.where(mask[:,0])).T.to(device, dtype = torch.int32)
        ref_coord = new_coord.clone()
        ref_coord[:,1:] = ref_coord[:,1:]//2
        feat_sparse = x.dense()[ref_coord[:,0], ref_coord[:,1], ref_coord[:,2], ref_coord[:,3]]
        if hasattr(x, 'spatial_range'):
            return SparseTensor(feats = feat_sparse, coords = new_coord, spatial_range = (x.spatial_range[0], x.spatial_range[1]*scale, x.spatial_range[2]*scale, x.spatial_range[3]*scale))
        else:
            return SparseTensor(feats = feat_sparse, coords = new_coord)

class UpsampleSparseCoord(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self, x, scale = 2):
        """
        Spatially upsample 3D coordinates.

        Parameters:
        - coords: NumPy array of shape (N, 3), where N is the number of coordinates.
        - scale: Integer scaling factor for upsampling.

        Returns:
        - upsampled_coords: NumPy array of shape (N * 2^3, 3), containing upsampled coordinates.
        """
        device = x.F.device
        coords = x.C[:,1:]
        # Step 1: Scale the original coordinates
        base = coords * scale

        # Step 2: Generate all possible combinations of offsets for each dimension
        offsets = torch.Tensor(list(torch.arange(scale)) * 3).reshape(3, scale).int().to(device)

        # Use meshgrid to create a grid of offsets
        grids = torch.meshgrid(*offsets, indexing='ij')  # List of 3 arrays, each of shape (scale, scale, scale)

        # Flatten each grid and stack them to get all combinations
        offset_combinations = torch.stack([g.flatten() for g in grids], axis=1)  # Shape: (scale**3, 3)

        # Step 3: Add offset combinations to each base coordinate
        # Expand dimensions for broadcasting
        # base[:, np.newaxis, :] has shape (N, 1, 3)
        # offset_combinations[np.newaxis, :, :] has shape (1, 8, 3)
        # Resulting upsampled will have shape (N, 8, 3)
        upsampled = base[:, None, :] + offset_combinations[None, :, :]  # Shape: (N, 8, 3)

        # Step 4: Reshape to (N * 8, 3) for a flat list of coordinates
        upsampled_coords = upsampled.reshape(-1,3)

        # Step 5: repeat batch index
        batch_coord = torch.repeat_interleave(x.C[:,:1], scale ** 3, dim=0)
        # Step 6: concatenate the batch index and coordinate
        upsampled_coords = torch.cat([batch_coord, upsampled_coords], dim = -1)

        # Step 7: get upsampled features
        upsampled_feat = torch.repeat_interleave(x.F, scale ** 3,dim=0)
        if hasattr(x, 'spatial_range'):
            return SparseTensor(feats = upsampled_feat, coords = upsampled_coords, spatial_range = (x.spatial_range[0], x.spatial_range[1]*2, x.spatial_range[2]*2, x.spatial_range[3]*2))
        else:
            return SparseTensor(feats = upsampled_feat, coords = upsampled_coords)

class PointVoxelNet(nn.Module):
    def __init__(
        self,
        net_v,
        net_p,
        vox_size: (torch.Tensor) = torch.Tensor([0.05, 0.05, 0.05])[:,None]
    ):
        super().__init__()
        self.net_v = net_v
        self.net_p = net_p
        self.vox_size = vox_size
    def forward(self, lidar, index = None):
        if index is not None:
            groupby_min = groupby_min_scatter(index, lidar[:,:3])
            
        return lidar
        
class VoxelUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: (list, tuple),
        in_channel_point: int,
        feat_channel_point: int,
        kernel_size: (list, tuple, int) = 3,
        num_layers: (int, list, tuple) = 2,
        act: tuple = ('leaky_relu', {'negative_slope': 0.2, 'inplace': True}),
        norm: tuple = ('instance', {}),
        upsample_method = 'conv_transpose'
    ):
        super().__init__()
        # initial feature layer
        self.init_feature = spnn.Conv3d(in_channels, channels[0], 3, 1, 1, bias = True)
        # setting up parameters
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in channels]
        if isinstance(num_layers, int):
            num_layers = [num_layers for _ in channels]
        # down blocks
        inc = channels[0]
        blocks = []
        downsamples = []
        for c, k, n_layer in zip(channels, kernel_size, num_layers):
            blocks.append(BaseBlock(inc, c, c, k, n_layer, act = act, norm = norm))
            downsamples.append(spnn.Conv3d(c,c,2,2,0))
            inc = c
        self.down_blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList(downsamples[:-1])
        # up blocks
        inc = channels[-1] + channels[-2] # 96
        blocks = []
        upsamples = []
        if upsample_method == 'interpolate':
            upsamples.append(UpsampleSparse())
        elif upsample_method == 'interpolate_coord':
            upsamples.append(UpsampleSparseCoord())
        elif upsample_method == 'conv_transpose':
            upsamples.append(spnn.Conv3d(channels[-1], channels[-1], 2, 2, 0, transposed = True, generative = False))
        for i in range(len(channels)-1): # middle block doesn't have counterpart
            c = channels[::-1][i+1]
            k = kernel_size[::-1][i+1]
            n_layer = num_layers[::-1][i+1]
            blocks.append(BaseBlock(inc, c, c, k, n_layer, act = act, norm = norm))
            if i < len(channels)-2: # if this is not last up_block
                next_skip_c = channels[::-1][i+2]
                inc = c + next_skip_c
                upsamples.append(spnn.Conv3d(c, c, 2, 2, 0, transposed = True, generative = False))
        self.up_blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        # output layers
        layers = []
        for i in range(len(channels)):
            c = channels[-i-1]
            layers.append(spnn.Conv3d(c, out_channels, 3, 1, 1))
        self.out_layers = nn.ModuleList(layers)
        # point layers
        self.point_layer = nn.Linear(in_channel_point, feat_channel_point)
    def forward(self, x: SparseTensor):
        outputs = []
        feats = []
        # initial feature
        x = self.init_feature(x)
        res = []
        # down blocks
        for block, downsample in zip(self.down_blocks, self.downsamples):
            x = block(x)
            res.append(x)
            x = downsample(x)
        x = self.down_blocks[-1](x)
        feats.append(x)
        outputs.append(self.out_layers[0](x))
        # up blocks
        res = res[::-1] # middle block doesn't have skip connection
        for block, upsample, r, out_layer in zip(self.up_blocks, self.upsamples, res, self.out_layers[1:]):
            x = upsample(x)
            x.F = torch.cat([r.F, x.F], dim = -1)
            x = block(x)
            feats.append(x)
            outputs.append(out_layer(x))
        return outputs, feats
    
class PVSeg(nn.Module):
    def __init__(
        self,
        net_v: nn.Module,
        net_p: nn.Module,
        net_out: nn.Module,
        vox_unit: list = [0.05, 0.05, 0.05]
    ):
        super().__init__()
        self.net_v = net_v
        self.net_p = net_p
        self.net_out = net_out
        self.register_buffer("vox_unit", torch.tensor(vox_unit)[None,:])
    def process_lidar(self, lidar):
        # voxelize
        xyz = lidar[:,1:-1] / self.vox_unit
        intensity = lidar[:,-1:]
        batch_index = lidar[:,:1]
        coord = xyz.floor()
        offset = coord + 0.5 - xyz
        values_agg = torch.cat([xyz, offset, intensity], dim = -1)
        # coord_voxelized, indices, inverse, counts = np.unique(np.concatenate([batch_index.cpu().numpy(), coord.cpu().numpy()], axis = 1), return_index=True, return_counts=True, return_inverse=True, axis = 0)
        
        coord_voxelized, inverse, counts = torch.unique(torch.cat([batch_index, coord], dim = -1), dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(inverse, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(cum_sum.device), cum_sum[:-1]))
        indices = ind_sorted[cum_sum]
        
        # coord_voxelized = torch.Tensor(coord_voxelized).to(coord.device, dtype = torch.long)
        coord_voxelized = coord_voxelized.long()
        coord_voxelized -= coord_voxelized.min(dim=0)[0]
        # indices = torch.Tensor(indices).to(coord.device, dtype = torch.long)
        # inverse = torch.Tensor(inverse).to(coord.device, dtype = torch.long)
        # counts = torch.Tensor(counts).to(coord.device, dtype = torch.long)

        # agg_max = groupby_scatter(inverse, values_agg, 'max')
        # agg_min = groupby_scatter(inverse, values_agg, 'min')
        agg_avg = groupby_scatter(inverse, values_agg, 'avg')
        # agg_sum = groupby_scatter(inverse, values_agg, 'sum')
        # agg_std = groupby_scatter(inverse, values_agg, 'std')
        agg_count = groupby_scatter(inverse, values_agg, 'count')

        coord_sp = coord_voxelized.int()
        feat_sp = agg_avg[:,3:]

        xsp = SparseTensor(coords = coord_sp, feats = feat_sp).cuda()
        xp = values_agg[:,-4:]
        return xsp, xp, inverse
    def forward(self, lidar):
        xsp, xp, inverse = self.process_lidar(lidar)
        vox_out, vox_feat = self.net_v(xsp)
        vox_feat = vox_feat[-1].F[inverse]
        p_feat = self.net_p(xp)
        feat = torch.cat([vox_feat, p_feat], dim = -1)
        out = self.net_out(feat)
        return out
    
    