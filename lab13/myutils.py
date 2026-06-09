import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, Module
import matplotlib.pyplot as plt
from torch.nn import    Linear, \
                        BatchNorm1d, \
                        Module
import torch.nn.functional as F
import math
from snntorch import utils
from tqdm import tqdm

##########################################
# DRAW EDGES
##########################################
def draw_graph(pos, edges):
    pos_np = pos.numpy()
    edges_np = edges.numpy()
    
    # Extract coordinates
    x, y, t = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(t, x, 32-y, c=t, cmap='plasma', s=5, alpha=0.8) # 32-y to flip Y-axis
    
    # Draw edges
    for (src, dst) in edges_np:
        if src < len(pos_np) and dst < len(pos_np):
            x_line = [pos_np[src, 0], pos_np[dst, 0]]
            y_line = [32-pos_np[src, 1], 32-pos_np[dst, 1]] # 32-y to flip Y-axis
            t_line = [pos_np[src, 2], pos_np[dst, 2]]
            ax.plot(t_line, x_line, y_line, color='gray', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("3D Event Graph")
    fig.colorbar(sc, ax=ax, label='Normalized Time')
    
    # ax.view_init(elev=0, azim=0)
    ax.view_init(elev=30, azim=35)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()

##########################################
# READ DATASET
##########################################

def read_dataset(filename):
    with open(filename, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | raw_data[4::5]

    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    td_indices = np.where(all_y != 240)[0]
    x, y, ts, p = all_x[td_indices], all_y[td_indices], all_ts[td_indices], all_p[td_indices]
    w, h = x.max() + 1, y.max() + 1

    return ts, x, y, p, h, w


def collate_fn(data_list):
    x = torch.cat([d['x'] for d in data_list], dim=0)
    pos = torch.cat([d['pos'] for d in data_list], dim=0)

    edge_index, offset = [], 0
    for d in data_list:
        edge_index.append(d['edge_index'] + offset)
        offset += d['x'].shape[0]
    edge_index = torch.cat(edge_index, dim=0)

    y = torch.stack([d['y'] for d in data_list], dim=0)
    batch = torch.cat([
        torch.full((d['x'].shape[0],), i, dtype=torch.long)
        for i, d in enumerate(data_list)
    ], dim=0)

    return {"x": x, "pos": pos, "edge_index": edge_index, "y": y, "batch": batch}


##########################################
# POINTNET CONVOLUTION
##########################################

class PointNetConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        return self.message(x, pos, edge_index)

    def message(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)
        msg = self.linear(msg)

        unique_positions, indices = torch.unique(edge_index[:, 0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim),
                                      dtype=msg.dtype, device=x.device)
        pooled_features = pooled_features.scatter_reduce(
            0, expanded_indices, msg, reduce="amax", include_self=False
        )
        return pooled_features

    def __repr__(self):
        return f"{self.__class__.__name__}(linear={self.linear})"


##########################################
# GRAPH POOL OUT 2D
##########################################

class GraphPoolOut2D(Module):
    def __init__(self, pool_size: int = 4, max_dimension: int = 256):
        super().__init__()
        self.pool_size = pool_size
        self.max_dimension = max_dimension
        self.grid_size = max_dimension // pool_size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        max_batch = batch.max() + 1
        qpos = torch.div(pos[:, :2], self.pool_size, rounding_mode='floor').long()
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)

        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        new_batch, uniq_qpos = unique_keys[:, 0], unique_keys[:, 1:]

        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        pooled_x = pooled_x.scatter_reduce(0, inv.unsqueeze(1).expand(-1, x.size(1)), x,
                                           reduce="amax", include_self=False)

        indices_1d = uniq_qpos[:, 0] * self.grid_size + uniq_qpos[:, 1]
        output_x = torch.zeros((max_batch, self.grid_size ** 2, x.size(1)),
                               dtype=x.dtype, device=x.device)
        output_x[new_batch, indices_1d] = pooled_x
        return output_x.flatten(start_dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size}, max_dimension={self.max_dimension})"


##########################################
# MATRIX NEIGHBOR GRAPH GENERATOR
##########################################

class GraphGen(Module):
    def __init__(self, r, dimension_XY=32, self_loop=True, device='cpu'):
        super().__init__()
        self.r = r
        self.dimension_XY = dimension_XY
        self.self_loop = self_loop
        self.device = device

        offsets = torch.arange(-r, r + 1, device=device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')
        mask = (dx ** 2 + dy ** 2) <= r ** 2
        self.ctx_offsets = torch.stack([dx[mask], dy[mask]], dim=1)

        self.pos_list, self.feat_list, self.edge_list = [], [], []
        self.neighbour_matrix = torch.full(
            (dimension_XY, dimension_XY), -1, dtype=torch.int32, device=device
        )
        self.index = 0

    def forward(self, events):
        for i in range(events.shape[0]):
            x, y, t, feature = events[i]
            x, y, t, feature = int(x), int(y), float(t), float(feature)

            if not (0 <= x < self.dimension_XY and 0 <= y < self.dimension_XY):
                continue

            idx_existing = self.neighbour_matrix[x, y].item()
            if idx_existing != -1 and float(self.pos_list[idx_existing][2]) == t:
                continue

            self.pos_list.append((x, y, t))
            self.feat_list.append(feature)
            if self.self_loop:
                self.edge_list.append((self.index, self.index))

            ctx_xy = self.ctx_offsets + torch.tensor([x, y], device=self.device)
            ctx_xy = torch.clamp(ctx_xy, 0, self.dimension_XY - 1)
            idxes = self.neighbour_matrix[ctx_xy[:, 0], ctx_xy[:, 1]]
            valid_mask = idxes != -1
            idxes = idxes[valid_mask]

            if idxes.numel() > 0:
                neighbor_pos = torch.tensor([self.pos_list[j] for j in idxes.tolist()],
                                            device=self.device)
                diffs = neighbor_pos - torch.tensor([x, y, t], device=self.device)
                mask = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2 + diffs[:, 2] ** 2) <= self.r ** 2
                for j in idxes[mask].tolist():
                    self.edge_list.append((self.index, j))

            self.neighbour_matrix[x, y] = self.index
            self.index += 1

        pos = torch.tensor(self.pos_list, dtype=torch.int32, device=self.device)
        x = torch.tensor(self.feat_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        edges = torch.tensor(self.edge_list, dtype=torch.int32, device=self.device) \
            if self.edge_list else torch.empty((0, 2), dtype=torch.int32, device=self.device)
        self.reset()
        return x, pos, edges

    def reset(self):
        self.pos_list.clear()
        self.feat_list.clear()
        self.edge_list.clear()
        self.neighbour_matrix.fill_(-1)
        self.index = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dimension_XY}, r={self.r}, device={self.device})"

##########################################
# GCN
##########################################

class GCN(Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetConv(1+3, 32, bias=True)
        self.norm1 = BatchNorm1d(32)

        self.conv2 = PointNetConv(32+3, 32, bias=True)
        self.norm2 = BatchNorm1d(32)

        self.conv3 = PointNetConv(32+3, 32, bias=True)
        self.norm3 = BatchNorm1d(32)
        
        self.pool_out = GraphPoolOut2D(pool_size=8, max_dimension=32)

        self.linear = Linear(4*4*32, 10)

    def forward(self, x, pos, edges, batch):
        x = self.conv1(x, pos, edges)
        x = F.relu(x)
        x = self.norm1(x)

        x = self.conv2(x, pos, edges)
        x = F.relu(x)
        x = self.norm2(x)

        x = self.conv3(x, pos, edges)
        x = F.relu(x)
        x = self.norm3(x)

        x = self.pool_out(x, pos, batch)

        x = self.linear(x)

        return x

##########################################
# SNN
##########################################

def events_to_spike_tensor(events, H, W, dt_us):
    # events: list of (x,y,t,p) with t in microseconds
    xs = np.array([e[0] for e in events], dtype=int)
    ys = np.array([e[1] for e in events], dtype=int)
    ts = np.array([e[2] for e in events], dtype=float)
    ps = np.array([e[3] for e in events], dtype=int)

    min_t = ts.min()
    # normalize timestamps so first event maps to bin 0
    ts_rel = ts - min_t
    duration = ts_rel.max()

    # number of bins so that bins cover [0, duration)
    T = int(math.ceil(duration / dt_us))   # dt_us in same units as ts_rel

    if T == 0:
        T = 1

    spike = torch.zeros((T, 1, 2, H, W), dtype=torch.float32)

    for x, y, t, p in zip(xs, ys, ts_rel, ps):
        t_idx = int(t // dt_us)        # floor division -> 0..T-1
        if t_idx >= T:
            t_idx = T - 1              # clamp edge case due to rounding
        spike[t_idx, 0, int(p), int(y), int(x)] = 1.0

    return spike

def forward_pass_snn(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in tqdm(range(data.size(0))):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)
