import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from parameters import get_parameters

parameters = get_parameters()
cuda = parameters['cuda']
device = parameters['device']
lat, lon, sin, cos, velocity = 0, 1, 2, 3, 4
row, column, day = 0, 1, 2
'''
构建dataloader类加载数据
'''


class TrajDataset(Dataset):
    """
    下面这三个是必备函数，必须要写
    __init__负责读取
    __getitem__负责获取数据编号
    __len__返回总长度
    """

    def __init__(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(current_dir, '../../../datasets/'))
        self.traj = torch.from_numpy(np.load(f'{file_path}/{name}/traj.npy').astype(np.float32))
        self.heat = torch.from_numpy(np.load(f'{file_path}/{name}/Heat.npy').astype(np.float32))
        self.ssh = torch.from_numpy(np.load(f'{file_path}/{name}/SSH.npy').astype(np.float32))
        self.sst = torch.from_numpy(np.load(f'{file_path}/{name}/SST.npy').astype(np.float32))
        self.sss = torch.from_numpy(np.load(f'{file_path}/{name}/SSS.npy').astype(np.float32))
        self.curr = torch.from_numpy(np.load(f'{file_path}/{name}/UV.npy').astype(np.float32))

    def __getitem__(self, index):

        return self.traj[index].to(device), \
               self.heat[index].unsqueeze(0).to(device), \
               self.ssh[index].unsqueeze(0).to(device), \
               self.sst[index].unsqueeze(0).to(device), \
               self.sss[index].unsqueeze(0).to(device), \
               self.curr[index].to(device)

    def __len__(self):
        return self.traj.shape[0]


def get_dataloader(name, batch_size=32, shuffle=True, drop_last=True):
    """
    必备函数，改对应参数即可
    """
    dataset = TrajDataset(name)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last) # num_workers=8, pin_memory=True
    return data_loader


if __name__ == '__main__':
    dataLoader = get_dataloader('test', batch_size=32, shuffle=False, drop_last=True)
    print(len(dataLoader))
    exit()
    dataLoader = get_dataloader('valid', batch_size=32, shuffle=False, drop_last=True)
    print(len(dataLoader))
    for traj, heat, ssh, sst, sss, curr in dataLoader:
        exit()
