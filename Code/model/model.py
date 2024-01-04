import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parameters import get_parameters

np.set_printoptions(suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# 参数读取
parameters = get_parameters()
cuda = parameters['cuda']
device = parameters['device']
learning_rate = parameters['learning_rate']
epochs = parameters['epochs']
early_stop_patients = parameters['early_stop_patients']
batch_size = parameters['batch_size']
mlp_features = parameters['mlp_features']
lstm_unit = parameters['lstm_unit']
lstm_layers = parameters['lstm_layers']
d_ff = parameters['d_ff']
dropout = parameters['dropout']
bidirectional = parameters['bidirectional']
conv_channel = parameters['conv_channel']
traj_num = parameters['traj_num']
traj_features = parameters['traj_features']
output_num = parameters['output_num']
output_fea = parameters['output_fea']

heat_channel = parameters['heat_channel']
heat_width = parameters['heat_width']
heat_height = parameters['heat_height']

envs_channel = parameters['envs_channel']
envs_width = parameters['envs_width']
envs_height = parameters['envs_height']

curr_channel = parameters['curr_channel']
curr_width = parameters['curr_width']
curr_height = parameters['curr_height']


class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,  # mlp_features + envs_height * envs_width + curr_width * curr_height + heat_width * heat_height
                            hidden_size=lstm_unit,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, inputs):
        h0 = torch.zeros(lstm_layers, inputs.shape[0], lstm_unit).to(device)
        c0 = torch.zeros(lstm_layers, inputs.shape[0], lstm_unit).to(device)
        output, (_, _) = self.lstm(inputs, (h0, c0))
        return output  # [batch, 19, lstm_unit] 如果是双向，此处lstm_unit=2*lstm_unit


class InputTrajFeatureExtraction(nn.Module):
    def __init__(self):
        super(InputTrajFeatureExtraction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(traj_num * traj_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_num * mlp_features),
        )

    def forward(self, traj):
        # traj = 轨迹数量 * 特征数量 [batch, 4, 5]
        traj_features = self.mlp(traj.view(traj.size(0), -1)).view(traj.size(0), output_num, -1)
        return traj_features  # [batch, 21, mlp_features]


class OutputTrajProjection(nn.Module):
    def __init__(self, input_size):
        super(OutputTrajProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, features):
        # features=[batch, 19, lstm_unit]  如果是双层，此处lstm_unit=2*lstm_unit
        out = self.projection(features)
        return out  # features=[batch, 19, 2] 最终轨迹


class SST_FeatureExtraction(nn.Module):
    def __init__(self):
        super(SST_FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=envs_channel, out_channels=conv_channel * 2, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_num, kernel_size=(1, 1), bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features.view(inputs.size(0), output_num, -1)


class SSS_FeatureExtraction(nn.Module):
    def __init__(self):
        super(SSS_FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=envs_channel, out_channels=conv_channel * 2, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_num, kernel_size=(1, 1), bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features.view(inputs.size(0), output_num, -1)


class SSH_FeatureExtraction(nn.Module):
    def __init__(self):
        super(SSH_FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=envs_channel, out_channels=conv_channel * 2, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_num, kernel_size=(1, 1), bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features.view(inputs.size(0), output_num, -1)


class Heatmap_FeatureExtraction(nn.Module):
    def __init__(self):
        super(Heatmap_FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=envs_channel, out_channels=conv_channel * 2, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_num, kernel_size=(1, 1), bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features.view(inputs.size(0), output_num, -1)


class Current_FeatureExtraction(nn.Module):
    def __init__(self):
        super(Current_FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=curr_channel, out_channels=conv_channel * 2, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_num, kernel_size=(1, 1), bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features.view(inputs.size(0), output_num, -1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # 输入输出维度一样[batch, channel, width, height]
        out = self.resblock(x)
        return out + x


class Interpolation(nn.Module):

    def __init__(self, use_sst_block=True, use_sss_block=True, use_ssh_block=True, use_current_block=True, use_heatmap_block=True, use_lstm=True):
        super(Interpolation, self).__init__()
        self.traj_block = InputTrajFeatureExtraction()

        self.feature_size = mlp_features
        if use_sst_block:
            self.sst_block = SST_FeatureExtraction()
            self.feature_size += envs_height * envs_width
        if use_sss_block:
            self.sss_block = SSS_FeatureExtraction()
            self.feature_size += envs_height * envs_width
        if use_ssh_block:
            self.ssh_block = SSH_FeatureExtraction()
            self.feature_size += envs_height * envs_width
        if use_current_block:
            self.current_block = Current_FeatureExtraction()
            self.feature_size += envs_height * envs_width
        if use_heatmap_block:
            self.fishing_heatmap_block = Heatmap_FeatureExtraction()
            self.feature_size += envs_height * envs_width
        if use_lstm:
            self.lstm = LSTM(self.feature_size)
            self.projection = OutputTrajProjection(lstm_unit)
        else:
            self.projection = OutputTrajProjection(self.feature_size)

    def forward(self, traj, heat, ssh, sst, sss, curr):
        # Concatenate the extracted features together
        features = self.traj_block(traj)

        if hasattr(self, 'sst_block'):
            features = torch.cat([features, self.sst_block(sst)], dim=-1)
        if hasattr(self, 'sss_block'):
            features = torch.cat([features, self.sss_block(sss)], dim=-1)
        if hasattr(self, 'ssh_block'):
            features = torch.cat([features, self.ssh_block(ssh)], dim=-1)
        if hasattr(self, 'current_block'):
            features = torch.cat([features, self.current_block(curr)], dim=-1)
        if hasattr(self, 'fishing_heatmap_block'):
            features = torch.cat([features, self.fishing_heatmap_block(heat)], dim=-1)
        if hasattr(self, 'lstm'):
            features = self.lstm(features)
        results = self.projection(features)
        return results  # [batch, 21, 2]


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    device = torch.device(f'cuda:0' if cuda else "cpu")
    traj = torch.rand(size=[8, 1, traj_num, traj_features]).to(device)
    ssh = torch.rand(size=[8, 1, envs_height, envs_width]).to(device)
    sst = torch.rand(size=[8, 1, envs_height, envs_width]).to(device)
    sss = torch.rand(size=[8, 1, envs_height, envs_width]).to(device)
    curr = torch.rand(size=[8, 3, curr_height, curr_width]).to(device)
    heat = torch.rand(size=[8, 1, heat_height, heat_width]).to(device)
    model = Interpolation().to(device)
    # input_traj, sub_heat, sub_ssh, sub_sst, sub_sss, sub_u, sub_v, sub_magnitude
    out = model(traj, heat, ssh, sst, sss, curr)
    print(out.size())
