import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
import torch

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]
def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids
def calculate_local_density(xyz, N):
    B, total_points, C = xyz.shape
    dists = torch.cdist(xyz, xyz)  # 计算距离矩阵
    knn_dists, _ = torch.topk(dists, N + 1, dim=-1, largest=False)  
    knn_dists = knn_dists[:, :, 1:]  # 排除自身
    density = 1.0 / (torch.mean(knn_dists, dim=-1) + 1e-6)
    return density

def assign_density_levels(density):
    min_density = torch.min(density, dim=1, keepdim=True)[0]
    max_density = torch.max(density, dim=1, keepdim=True)[0]
    normalized_density = (density - min_density) / (max_density - min_density)
    return normalized_density



def modified_fps(xyz, M, n):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    density = calculate_local_density(xyz, 8)
    normalized_density = assign_density_levels(density)
#    print(normalized_density)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :]
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        # Adjust the distances based on the normalized density
        adjusted_dists = cur_dist / (torch.cos(n * normalized_density[batchlists, inds]).unsqueeze(1)).float()
        dists[adjusted_dists < dists] = adjusted_dists[adjusted_dists < dists]
        inds = torch.max(dists, dim=1)[1] 
    return centroids


def ball_query(xyz, new_xyz, radius, K):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def knn1(x, k):
    x_coords = x[:, :3, :]  # 假设坐标是前三个特征
    x_features = x[:, 3:, :]  # 假设其他的是点的特征
    
    C = x_features.size()[1]

    # 计算坐标的内积
    coords_inner = -2 * torch.matmul(x_coords.transpose(2, 1), x_coords)
    # 计算特征的内积
    features_inner = -2 * torch.matmul(x_features.transpose(2, 1), x_features)
    # 计算坐标的平方和
    coords_sq = torch.sum(x_coords ** 2, dim=1, keepdim=True)
    # 计算特征的平方和
    features_sq = torch.sum(x_features ** 2, dim=1, keepdim=True)
    
    # 计算坐标空间和特征空间的距离
    coords_distance = -coords_sq - coords_inner - coords_sq.transpose(2, 1)
    features_distance = -features_sq - features_inner - features_sq.transpose(2, 1)
    # 最终距离是坐标空间距离加上特征空间距离
    
    pairwise_distance = coords_distance + features_distance
    # 获取 top k 最近邻居的索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  
    return feature
def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)
    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds

def three_interpolate(xyz1, xyz2, points2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = points2.shape
    dists, inds = three_nn(xyz1, xyz2)
    inversed_dists = 1.0 / (dists + 1e-8)
    weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
    interpolated_points = gather_points(points2, inds)  # shape=(B, N1, 3, C2)
    interpolated_points = torch.sum(weight * interpolated_points, dim=2)
    return interpolated_points