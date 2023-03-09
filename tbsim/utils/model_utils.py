import torch
import numpy as np
from tbsim.models.base_models import MLP
import torch.nn as nn
from tbsim.utils.geometry_utils import round_2pi
import community as community_louvain
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import torch.distributions as td
import tbsim.utils.tensor_utils as TensorUtils

class PED_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(PED_PED_encode, self).__init__()
        self.FC = MLP(10, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        input = torch.cat((deltax, x1[..., 2:4], x2[..., 2:4], size1, size2), dim=-1)
        return self.FC(input)


class PED_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(PED_VEH_encode, self).__init__()
        self.FC = MLP(10, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        veh_vel = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(x2[..., 3]), dim=-1),
                torch.unsqueeze(x2[..., 2] * torch.sin(x2[..., 3]), dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat((deltax, x1[..., 2:4], veh_vel, size1, size2), dim=-1)
        return self.FC(input)


class VEH_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(VEH_PED_encode, self).__init__()
        self.FC = MLP(9, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dv = torch.cat(
            (
                torch.unsqueeze(
                    x2[..., 2] * torch.cos(theta)
                    + torch.sin(theta) * x2[..., 3]
                    - x1[..., 2],
                    dim=-1,
                ),
                torch.unsqueeze(
                    x2[..., 3] * torch.cos(theta) - torch.sin(theta) * x2[..., 2],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        input = torch.cat(
            (dx, torch.unsqueeze(x1[..., 2], dim=-1), dv, size1, size2), dim=-1
        )
        return self.FC(input)


class VEH_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, hidden_dim=[64]):
        super(VEH_VEH_encode, self).__init__()
        self.FC = MLP(11, obs_enc_dim, hidden_dim)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dtheta = x2[..., 3] - x1[..., 3]
        dv = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(dtheta) - x1[..., 2], dim=-1),
                torch.unsqueeze(torch.sin(dtheta) * x2[..., 2], dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat(
            (
                dx,
                torch.unsqueeze(x1[..., 2], dim=-1),
                dv,
                torch.unsqueeze(torch.cos(dtheta), dim=-1),
                torch.unsqueeze(torch.sin(dtheta), dim=-1),
                size1,
                size2,
            ),
            dim=-1,
        )
        return self.FC(input)


def PED_rel_state(x, x0):
    rel_x = torch.clone(x)
    rel_x[..., 0:2] -= x0[..., 0:2]
    return rel_x


def VEH_rel_state(x, x0):
    rel_XY = x[..., 0:2] - x0[..., 0:2]
    theta = x0[..., 3]
    rel_x = torch.stack(
        [
            rel_XY[..., 0] * torch.cos(theta) + rel_XY[..., 1] * torch.sin(theta),
            rel_XY[..., 1] * torch.cos(theta) - rel_XY[..., 0] * torch.sin(theta),
            x[..., 2],
            x[..., 3] - x0[..., 3],
        ],
        dim=-1,
    )
    rel_x[..., 3] = round_2pi(rel_x[..., 3])
    return rel_x


class PED_pre_encode(nn.Module):
    def __init__(self, enc_dim, hidden_dim=[64], use_lane_info=False):
        super(PED_pre_encode, self).__init__()
        self.FC = MLP(4, enc_dim, hidden_dim)

    def forward(self, x):
        return self.FC(x)


class VEH_pre_encode(nn.Module):
    def __init__(self, enc_dim, hidden_dim=[64], use_lane_info=False):
        super(VEH_pre_encode, self).__init__()
        self.use_lane_info = use_lane_info
        if use_lane_info:
            self.FC = MLP(8, enc_dim, hidden_dim)
        else:
            self.FC = MLP(5, enc_dim, hidden_dim)

    def forward(self, x):
        if self.use_lane_info:
            input = torch.cat(
                (
                    x[..., 0:3],
                    torch.cos(x[..., 3:4]),
                    torch.sin(x[..., 3:4]),
                    x[..., 4:5],
                    torch.cos(x[..., 5:6]),
                    torch.sin(x[..., 5:6]),
                ),
                dim=-1,
            )
        else:
            input = torch.cat(
                (x[..., 0:3], torch.cos(x[..., 3:]), torch.sin(x[..., 3:])), dim=-1
            )
        return self.FC(input)

def break_graph(M, resol=1.0):
    if isinstance(M, np.ndarray):
        resol = resol * np.max(M)
        G = nx.Graph()
        for i in range(M.shape[0]):
            G.add_node(i)
        for i in range(M.shape[0]):
            for j in range(i + 1, M.shape[0]):
                if M[i, j] > 0:
                    G.add_edge(i, j, weight=M[i, j])
        partition = community_louvain.best_partition(G, resolution=resol)
    elif isinstance(M, nx.classes.graph.Graph):
        G = M
        partition = community_louvain.best_partition(G, resolution=resol)

    while max(partition.values()) == 0 and resol >= 0.1:
        resol = resol * 0.9
        partition = community_louvain.best_partition(G, resolution=resol)
    return partition


def break_graph_recur(M, max_num):
    n_components, labels = connected_components(
        csgraph=csr_matrix(M), directed=False, return_labels=True
    )
    idx = 0

    while idx < n_components:
        subset = np.where(labels == idx)[0]
        if subset.shape[0] <= max_num:
            idx += 1
        else:
            partition = break_graph(M[np.ix_(subset, subset)])
            added_partition = 0
            for i in range(subset.shape[0]):
                if partition[i] > 0:
                    labels[subset[i]] = n_components + partition[i] - 1
                    added_partition = max(added_partition, partition[i])

            n_components += added_partition
            if added_partition == 0:
                idx += 1

    return n_components, labels

def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))

class Normal:

    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu
    def pseudo_sample(self,n_sample):
        sigma_points = torch.stack((self.mu,self.mu-self.sigma,self.mu+self.sigma),1)
        if n_sample<=1:
            return sigma_points[:,:n_sample]
        else:
            remain_n = n_sample-3
            sigma_tiled = self.sigma.unsqueeze(1).repeat_interleave(remain_n,1)
            mu_tiled = self.mu.unsqueeze(1).repeat_interleave(remain_n,1)
            sample = torch.randn_like(sigma_tiled)*sigma_tiled+mu_tiled
            return torch.cat([sigma_points,sample],1)

class Categorical:

    def __init__(self, probs=None, logits=None, temp=0.01):
        super().__init__()
        self.logits = logits
        self.temp = temp
        if probs is not None:
            self.probs = probs
        else:
            assert logits is not None
            self.probs = torch.softmax(logits, dim=-1)
        self.dist = td.OneHotCategorical(self.probs)

    def rsample(self,n_sample=1):
        relatex_dist = td.RelaxedOneHotCategorical(self.temp, self.probs)
        return relatex_dist.rsample((n_sample,)).transpose(0,-2)

    def sample(self):
        return self.dist.sample()
    
    def pseudo_sample(self,n_sample):
        D = self.probs.shape[-1]
        idx = self.probs.argsort(-1,descending=True)
        assert n_sample<=D
        return TensorUtils.to_one_hot(idx[...,:n_sample],num_class=D)



    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            p = Categorical(logits=torch.zeros_like(self.probs))
        kl = td.kl_divergence(self.dist, p.dist)
        return kl

    def mode(self):
        argmax = self.probs.argmax(dim=-1)
        one_hot = torch.zeros_like(self.probs)
        one_hot.scatter_(1, argmax.unsqueeze(1), 1)
        return one_hot

        

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

class AFMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())        

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

def rotation_2d_torch(x, theta, origin=None):
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x


class ExpParamAnnealer(nn.Module):

    def __init__(self, start, finish, rate, cur_epoch=0):
        super().__init__()
        self.register_buffer('start', torch.tensor(start))
        self.register_buffer('finish', torch.tensor(finish))
        self.register_buffer('rate', torch.tensor(rate))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.finish - (self.finish - self.start) * (self.rate ** self.cur_epoch)