import json
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import math
import torch
from pathlib import Path
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/home/raozf/data'
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
def download_scanobjectnn():
    DATA_DIR = '/home/raozf/data'
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        os.mkdir(os.path.join(DATA_DIR, 'h5_files'))
        www = 'https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
def load_scanobjectnn( partition):
    data_dir =  '/home/raozf/data'
    download_scanobjectnn()
    h5_name = os.path.join(data_dir, 'h5_files/main_split_nobg', '%s_objectdataset.h5'%partition)         #objectdataset_augmentedrot_scale75
    f = h5py.File(h5_name, 'r')
    all_data = f['data'][:].astype('float32')
    all_label = f['label'][:].astype('int64')
    f.close()
    return all_data, all_label
def download_shapenetpart():
    DATA_DIR = '/home/raozf/data'
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))
        
def load_data_partseg(partition):
    download_shapenetpart()
    DATA_DIR = '/home/raozf/data'
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def load_data(partition):
#    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/home/raozf/data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def random_point_dropout(pc, max_dropout_ratio=0.5):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        
        self.dp = True
        self.noise = False
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            if self.dp:
                pointcloud = random_point_dropout(pointcloud)
            if self.noise:
                pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label
    def __len__(self):
        return self.data.shape[0]
    

class ModelNet_O(Dataset):
    def __init__(self, num_points, partition='train'):
        self.root_dir = Path('/home/raozf/data/ModelNet_O/')  # 确保 root_dir 是一个 Path 对象
        self.num_points = num_points
        self.split = partition
        self.samples = []
        # 遍历所有类别文件夹
        for category in os.listdir(self.root_dir):
            category_dir = self.root_dir / category / self.split
            if category_dir.is_dir():
                # 遍历每个类别下的点云文件
                for file in os.listdir(category_dir):
                    if file.endswith('.xyz'):
                        self.samples.append((category_dir / file, category))
    def __len__(self):
        return len(self.samples)
    
    def _encode_label(self, category):
        return torch.tensor(sorted(os.listdir(self.root_dir)).index(category), dtype=torch.long)

    def __getitem__(self, idx):

        file_path, category = self.samples[idx]
        # 加载点云文件
        pointcloud = np.loadtxt(file_path, delimiter=' ')  # 根据实际文件格式调整
        if self.split == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = random_point_dropout(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        # 对类别标签进行编码
        label = self._encode_label(category)
        label = torch.tensor([label], dtype=torch.long)  # 包装为一维张量
        return {
            "coordinates": pointcloud.to(torch.float32),
            "features": pointcloud.to(torch.float32),
            "label": label,
        }
class ScanObjectNN(Dataset):
    def __init__(self,  num_points=2048, partition='training'):
        self.data, self.label = load_scanobjectnn(partition)
        self.num_points = num_points
        self.partition = partition
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        label = np.array([label]) if np.isscalar(label) else label
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = random_point_dropout(pointcloud)
            np.random.shuffle(pointcloud)
        xyz = torch.from_numpy(pointcloud)
        label = torch.from_numpy(label)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }
    def __len__(self):
        return self.data.shape[0]
    
class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = 2048
        self.partition = partition        
        self.class_choice = class_choice
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
#            pointcloud = translate_pointcloud(pointcloud)
            seg = seg[indices]
            
        xyz = torch.from_numpy(pointcloud)
        label = torch.from_numpy(label)
        seg = torch.from_numpy(seg)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
            "seg": seg,
          }
    def __len__(self):
        return self.data.shape[0]

    
class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data_root = '/home/raozf/data/ShapeNetCore'
        self.split= partition
        self.npoints =  num_points
        self.cat = {}
        with open(os.path.join(self.data_root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f.readlines():
                self.cat[line.strip().split()[0]] = line.strip().split()[1]
        train_json_path = os.path.join(self.data_root, 'train_test_split', 'shuffled_train_file_list.json')
        val_json_path = os.path.join(self.data_root, 'train_test_split', 'shuffled_val_file_list.json')
        test_json_path = os.path.join(self.data_root, 'train_test_split', 'shuffled_test_file_list.json')
        train_lists = self.decode_json(self.data_root, train_json_path)
        val_lists = self.decode_json(self.data_root, val_json_path)
        test_lists = self.decode_json(self.data_root, test_json_path)

        self.file_lists = []
        if self.split == 'train':
            self.file_lists.extend(train_lists)
        elif self.split == 'val':
            self.file_lists.extend(val_lists)
        elif self.split == 'trainval':
            self.file_lists.extend(train_lists)
            self.file_lists.extend(val_lists)
        elif self.split == 'test':
            self.file_lists.extend(test_lists)
        self.seg_classes = {'Earphone': [16, 17, 18],
                            'Motorbike': [30, 31, 32, 33, 34, 35],
                            'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11],
                            'Laptop': [28, 29], 'Cap': [6, 7],
                            'Skateboard': [44, 45, 46], 'Mug': [36, 37],
                            'Guitar': [19, 20, 21], 'Bag': [4, 5],
                            'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.caches = {}
    def decode_json(self, data_root, path):
        with open(path, 'r') as f:
            l = json.load(f)
        l = [os.path.join(data_root, item.split('/')[1], item.split('/')[2] + '.txt') for item in l]
        return l
    def __getitem__(self, index):
        if index in self.caches:
            xyz_points, labels = self.caches[index]
        else:
            pc = np.loadtxt(self.file_lists[index]).astype(np.float32)
            xyz_points = pc[:, :6]
            labels = pc[:, -1].astype(np.int32)
            xyz_points = random_point_dropout(xyz_points)
            self.caches[index] = xyz_points, labels

        # resample
        choice = np.random.choice(len(xyz_points), 2500, replace=True)
        xyz_points = xyz_points[choice, :3]
        labels = labels[choice]
        xyz = torch.from_numpy(xyz_points)
        label = torch.from_numpy(labels)
        
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }
    def __len__(self):
        return len(self.file_lists)
if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
