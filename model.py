import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaGCN.util import modified_fps, gather_points
import MinkowskiEngine as ME
from MambaGCN.mamba import Mamba2
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
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
class OverlapAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(OverlapAttentionBlock, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        '''
        :param x: (B, C, N)
        :param ol: (B, N)
        :return: (B, C, N)
        '''
        B, C, N = x.size()
        x_q = self.q_conv(x).permute(0, 2, 1).contiguous() # B, N, C
        x_k = self.k_conv(x) # B, C, N
        x_v = self.v_conv(x)
        attention = torch.bmm(x_q, x_k) # B, N, N
        
        attention = torch.softmax(attention, dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # B, C, N
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
class OverlapAttention(nn.Module):
    def __init__(self, dim):
        super(OverlapAttention, self).__init__()
        self.overlap_attention1 = OverlapAttentionBlock(dim)
        self.overlap_attention2 = OverlapAttentionBlock(dim)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim*2, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim//2, dim//4, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim//4, 1, kernel_size=1, bias=True))
    def forward(self, x):
        x1 = self.overlap_attention1(x)
        x2 = self.overlap_attention2(x1)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv_fuse(x)
        return x
    
class OverlapAttention1(nn.Module):
    def __init__(self, dim):
        super(OverlapAttention1, self).__init__()
        self.overlap_attention1 = OverlapAttentionBlock(dim)
        self.overlap_attention2 = OverlapAttentionBlock(dim)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim*2, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim//2, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 1, kernel_size=1, bias=True))
    def forward(self, x):
        x1 = self.overlap_attention1(x)
        x2 = self.overlap_attention2(x1)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv_fuse(x)
        return x
class OverlapAttention2(nn.Module):
    def __init__(self, dim):
        super(OverlapAttention2, self).__init__()
        self.overlap_attention1 = OverlapAttentionBlock(dim)
        self.overlap_attention2 = OverlapAttentionBlock(dim)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim*2, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(16, 1, kernel_size=1, bias=True))
    def forward(self, x):
        x1 = self.overlap_attention1(x)
        x2 = self.overlap_attention2(x1)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv_fuse(x)
        return x

class Fused1(nn.Module):
    def __init__(self):
        super(Fused1, self).__init__()
        self.weight_a = nn.Parameter(torch.ones(1))
        self.weight_b = nn.Parameter(torch.ones(1))
        self.weight_c = nn.Parameter(torch.ones(1))
        self.weight_d = nn.Parameter(torch.ones(1))
    def forward(self, a, b, c, d):
        a = self.weight_a * a
        b = self.weight_b * b 
        c = self.weight_c * c
        d = self.weight_d * d
        out = torch.cat([a, b, c, d], dim=1)
        return out
    
class Fused2(nn.Module):
    def __init__(self):
        super(Fused2, self).__init__()
        self.weight_a = nn.Parameter(torch.ones(1))
        self.weight_b = nn.Parameter(torch.ones(1))
        self.weight_c = nn.Parameter(torch.ones(1))
        self.weight_d = nn.Parameter(torch.ones(1))
    def forward(self, a, b, c, d):
        a = self.weight_a * a
        b = self.weight_b * b 
        c = self.weight_c * c
        d = self.weight_d * d
        out = torch.cat([a, b, c, d], dim=1)
        return out
class Fused3(nn.Module):
    def __init__(self):
        super(Fused3, self).__init__()
        self.weight_a = nn.Parameter(torch.ones(1))
        self.weight_b = nn.Parameter(torch.ones(1))
        self.weight_c = nn.Parameter(torch.ones(1))
        self.weight_d = nn.Parameter(torch.ones(1))
    def forward(self, a, b, c, d):
        a = self.weight_a * a
        b = self.weight_b * b 
        c = self.weight_c * c
        d = self.weight_d * d
        out = torch.cat([a, b, c, d], dim=1)
        return out
    
class PointNet_SA_Module1(nn.Module):
    def __init__(self,k):
        super(PointNet_SA_Module1, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2+6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2+6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2+6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.overlap_attention1 = Mamba2(64)
        self.overlap_attention2 = Mamba2(64)
        self.conv_fuse1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(8, 1, kernel_size=1, bias=True))
        self.conv_fuse2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(8, 1, kernel_size=1, bias=True))
    def forward(self, x):    #x.shape, B*C*N
        batch_size = x.size(0)
        
        point = get_graph_feature(x, k=self.k)
        point = self.conv1(point)
        point1 = point.max(dim=-1, keepdim=False)[0]
        point1_, _ = torch.max(point1, dim=2)
        point1_mean = torch.mean(point1, dim=2)

        point = torch.cat([x, point1], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv2(point)
        point2 = point.max(dim=-1, keepdim=False)[0]
        point2_, _ = torch.max(point2, dim=2)
        point2_mean = torch.mean(point2, dim=2)

        point = torch.cat([x, point2], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv3(point)
        point3 = point.max(dim=-1, keepdim=False)[0]
        point3_, _ = torch.max(point3, dim=2)
        point3_mean = torch.mean(point3, dim=2)

        point = torch.cat([x, point3], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv4(point)
        point4 = point.max(dim=-1, keepdim=False)[0]
        point4_, _ = torch.max(point4, dim=2)
        point4_mean = torch.mean(point4, dim=2)

        stack = torch.stack([point1_.detach(), point2_.detach(), point3_.detach(), point4_.detach()], dim=1) 
        weights = self.overlap_attention1(stack.contiguous())
        weights = self.conv_fuse1(weights.permute(0, 2, 1).contiguous())
        weights = torch.split(weights, 1, dim=2)
        weights = [w.squeeze(2) for w in weights]
        weights = [w.mean(dim=0) for w in weights]
        stack2 = torch.stack([point1_mean.detach(), point2_mean.detach(), point3_mean.detach(), point4_mean.detach()], dim=1) 
        weights2 = self.overlap_attention2(stack2.contiguous())
        weights2 = self.conv_fuse2(weights2.permute(0, 2, 1).contiguous())
        weights2 = torch.split(weights2, 1, dim=2)
        weights2 = [w.squeeze(2) for w in weights2]
        weights2 = [w.mean(dim=0) for w in weights2]
        point = torch.cat(((weights[0]+weights2[0])*point1, (weights[1]+weights2[1])*point2, (weights[2]+weights2[2])*point3, (weights[3]+weights2[3])*point4), dim=1)
        point = self.conv5(point)
        return point
    
class PointNet_SA_Module2(nn.Module):
    def __init__(self, M, k):
        super(PointNet_SA_Module2, self).__init__()
        self.M = M
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(2048+6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(256+6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256+6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256+6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.overlap_attention1 = Mamba2(128)
        self.overlap_attention2 = Mamba2(128)
        self.conv_fuse1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(8, 1, kernel_size=1, bias=True))
        self.conv_fuse2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(8, 1, kernel_size=1, bias=True))
    def forward(self, x, point):
        batch_size = x.size(0)
        inds =  modified_fps(x.permute(0, 2, 1), self.M, 0.9)
        new_xyz = gather_points(x.permute(0, 2, 1), inds).permute(0, 2, 1)
        point = gather_points(point.permute(0, 2, 1), inds).permute(0, 2, 1)

        point = torch.cat([new_xyz, point], dim=1)

        point = get_graph_feature(point, k=self.k)
        point = self.conv1(point)
        point1 = point.max(dim=-1, keepdim=False)[0]
        point1_, _ = torch.max(point1, dim=2)
        point1_mean = torch.mean(point1, dim=2)
  
        point = torch.cat([new_xyz, point1], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv2(point)
        point2 = point.max(dim=-1, keepdim=False)[0]
        point2_, _ = torch.max(point2, dim=2)
        point2_mean = torch.mean(point2, dim=2)

        point = torch.cat([new_xyz, point2], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv3(point)
        point3 = point.max(dim=-1, keepdim=False)[0]
        point3_, _ = torch.max(point3, dim=2)
        point3_mean = torch.mean(point3, dim=2)

        point = torch.cat([new_xyz, point3], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv4(point)
        point4 = point.max(dim=-1, keepdim=False)[0]
        point4_, _ = torch.max(point4, dim=2)
        point4_mean = torch.mean(point4, dim=2)
        
        stack = torch.stack([point1_.detach(), point2_.detach(), point3_.detach(), point4_.detach()], dim=1) 
        weights = self.overlap_attention1(stack.contiguous())
        weights = self.conv_fuse1(weights.permute(0, 2, 1).contiguous())
        weights = torch.split(weights, 1, dim=2)
        weights = [w.squeeze(2) for w in weights]
        weights = [w.mean(dim=0) for w in weights]
        stack2 = torch.stack([point1_mean.detach(), point2_mean.detach(), point3_mean.detach(), point4_mean.detach()], dim=1) 
        weights2 = self.overlap_attention2(stack2.contiguous())
        weights2 = self.conv_fuse2(weights2.permute(0, 2, 1).contiguous())
        weights2 = torch.split(weights2, 1, dim=2)
        weights2 = [w.squeeze(2) for w in weights2]
        weights2 = [w.mean(dim=0) for w in weights2]
        point = torch.cat(((weights[0]+weights2[0])*point1, (weights[1]+weights2[1])*point2, (weights[2]+weights2[2])*point3, (weights[3]+weights2[3])*point4), dim=1)
        point = self.conv5(point)
        return new_xyz, point   
     
class PointNet_SA_Module3(nn.Module):
    def __init__(self, M, k):
        super(PointNet_SA_Module3, self).__init__()
        self.M = M
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(2048+6, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(512+6, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(512+6, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(512+6, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.overlap_attention1 = Mamba2(256)
        self.overlap_attention2 = Mamba2(256)
        self.conv_fuse1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(16, 1, kernel_size=1, bias=True))
        self.conv_fuse2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(16, 1, kernel_size=1, bias=True))
    def forward(self, x, point):
        batch_size = x.size(0)
#        print('x', x.shape)
#        print('point', point.shape)
        inds =  modified_fps(x.permute(0, 2, 1), self.M, 0.9)
        new_xyz = gather_points(x.permute(0, 2, 1), inds).permute(0, 2, 1)
        point = gather_points(point.permute(0, 2, 1), inds).permute(0, 2, 1)
#        print('new_xyz', new_xyz.shape)
#        print('point', point.shape)

        point = torch.cat([new_xyz, point], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv1(point)
        point1 = point.max(dim=-1, keepdim=False)[0]
        point1_, _ = torch.max(point1, dim=2)
        point1_mean = torch.mean(point1, dim=2)
  
        point = torch.cat([new_xyz, point1], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv2(point)
        point2 = point.max(dim=-1, keepdim=False)[0]
        point2_, _ = torch.max(point2, dim=2)
        point2_mean = torch.mean(point2, dim=2)

        point = torch.cat([new_xyz, point2], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv3(point)
        point3 = point.max(dim=-1, keepdim=False)[0]
        point3_, _ = torch.max(point3, dim=2)
        point3_mean = torch.mean(point3, dim=2)

        point = torch.cat([new_xyz, point3], dim=1)
        point = get_graph_feature(point, k=self.k)
        point = self.conv4(point)
        point4 = point.max(dim=-1, keepdim=False)[0]
        point4_, _ = torch.max(point4, dim=2)
        point4_mean = torch.mean(point4, dim=2)
        
        stack = torch.stack([point1_.detach(), point2_.detach(), point3_.detach(), point4_.detach()], dim=1) 
        weights = self.overlap_attention1(stack.contiguous())
        weights = self.conv_fuse1(weights.permute(0, 2, 1).contiguous())
        weights = torch.split(weights, 1, dim=2)
        weights = [w.squeeze(2) for w in weights]
        weights = [w.mean(dim=0) for w in weights]
        stack2 = torch.stack([point1_mean.detach(), point2_mean.detach(), point3_mean.detach(), point4_mean.detach()], dim=1) 
        weights2 = self.overlap_attention2(stack2.contiguous())
        weights2 = self.conv_fuse2(weights2.permute(0, 2, 1).contiguous())
        weights2 = torch.split(weights2, 1, dim=2)
        weights2 = [w.squeeze(2) for w in weights2]
        weights2 = [w.mean(dim=0) for w in weights2]
        point = torch.cat(((weights[0]+weights2[0])*point1, (weights[1]+weights2[1])*point2, (weights[2]+weights2[2])*point3, (weights[3]+weights2[3])*point4), dim=1)
        point = self.conv5(point)
        return new_xyz, point   
    
class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.pt_sa1 = PointNet_SA_Module1(k=20)
        self.pt_sa2 = PointNet_SA_Module2(M=512, k=40)
        self.pt_sa3 = PointNet_SA_Module3(M=256, k=50)

        self.linear1 = nn.Linear(args.emb_dims*2, 2048, bias=False)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(2048, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(512, output_channels)

        self.linear4 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.linear5 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=args.dropout)
        self.linear6 = nn.Linear(256, output_channels)

        self.linear7 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.dp5 = nn.Dropout(p=args.dropout)
        self.linear8 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp6 = nn.Dropout(p=args.dropout)
        self.linear9 = nn.Linear(256, output_channels)

        self.linear10 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn7 = nn.BatchNorm1d(512)
        self.dp7 = nn.Dropout(p=args.dropout)
        self.linear11 = nn.Linear(512, 256, bias=False)
        self.bn8 = nn.BatchNorm1d(256)
        self.dp8 = nn.Dropout(p=args.dropout)
        self.linear12 = nn.Linear(256, output_channels)
        
        self.overlap_attention1 = OverlapAttention1(2048)
        self.overlap_attention2 = OverlapAttention2(40)
   
    def forward(self, xyz):   
        batch = xyz.size()[0]
        batch_size = xyz.size(0)
        new_point1 = self.pt_sa1(xyz)                  #new_point(B, C, N)
        new_xyz2, new_point2 = self.pt_sa2(xyz, new_point1)
        new_xyz3, new_point3 = self.pt_sa3(new_xyz2, new_point2)
        
        x1 = F.adaptive_max_pool1d(new_point1, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(new_point1, 1).view(batch_size, -1)
        x_2048 = torch.cat((x1, x2), dim = 1)

        x3 = F.adaptive_max_pool1d(new_point2, 1).view(batch_size, -1)
        x4 = F.adaptive_avg_pool1d(new_point2, 1).view(batch_size, -1)
        x_512 = torch.cat((x3, x4), dim = 1)
        
        x5 = F.adaptive_max_pool1d(new_point3, 1).view(batch_size, -1)
        x6 = F.adaptive_avg_pool1d(new_point3, 1).view(batch_size, -1)
        x_256 = torch.cat((x5, x6), dim = 1)

        x = x_2048 + x_512 + x_256
        #classifier1
        x_ = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x_ = self.dp1(x_)
        x_ = F.leaky_relu(self.bn2(self.linear2(x_)), negative_slope=0.2)
        x_ = self.dp2(x_)
        x_ = self.linear3(x_)
        #classifier2
        x_2048_ = F.leaky_relu(self.bn3(self.linear4(x_2048)), negative_slope=0.2)
        x_2048_ = self.dp3(x_2048_)
        x_2048_ = F.leaky_relu(self.bn4(self.linear5(x_2048_)), negative_slope=0.2)
        x_2048_ = self.dp4(x_2048_)
        x_2048_ = self.linear6(x_2048_)
        #classifier3
        x_512_ = F.leaky_relu(self.bn5(self.linear7(x_512)), negative_slope=0.2)
        x_512_ = self.dp5(x_512_)
        x_512_ = F.leaky_relu(self.bn6(self.linear8(x_512_)), negative_slope=0.2)
        x_512_ = self.dp6(x_512_)
        x_512_ = self.linear9(x_512_)
        #classifier4
        x_256_ = F.leaky_relu(self.bn7(self.linear10(x_256)), negative_slope=0.2)
        x_256_ = self.dp7(x_256_)
        x_256_ = F.leaky_relu(self.bn8(self.linear11(x_256_)), negative_slope=0.2)
        x_256_ = self.dp8(x_256_)
        x_256_ = self.linear12(x_256_)

        
        stack1 = torch.stack([x, x_2048, x_512, x_256], dim=1)    
        weights1 = self.overlap_attention1(stack1.permute(0, 2, 1).contiguous())
        weights1 = torch.split(weights1, 1, dim=2)
        weights1 = [w.squeeze(2) for w in weights1]
        
        stack2 = torch.stack([x_.detach(), x_2048_.detach(), x_512_.detach(), x_256_.detach()], dim=1)    
        weights2 = self.overlap_attention2(stack2.permute(0, 2, 1).contiguous())
        weights2 = torch.split(weights2, 1, dim=2)
        weights2 = [w.squeeze(2) for w in weights2]
        out = (weights1[0] + 0.5*weights2[0]) * x_.detach() + (weights1[1]+ 0.5*weights2[1]) * x_2048_.detach() + (weights1[2]+ 0.5*weights2[2]) * x_512_.detach() +  (weights1[3]+ 0.5*weights2[3]) * x_256_.detach()
        
        return x_, x_2048_, x_512_, x_256_, out