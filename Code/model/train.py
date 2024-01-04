import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn as nn
from model import Interpolation
from parameters import get_parameters
from data_loader import get_dataloader
from test import global_coordinate_distance
from tqdm import tqdm

lat, lon, sin, cos, velocity = 0, 1, 2, 3, 4
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
# 参数读取
parameters = get_parameters()
cuda = parameters['cuda']
device = parameters['device']
learning_rate = parameters['learning_rate']
early_stop_patients = parameters['early_stop_patients']
batch_size = parameters['batch_size']
epochs = parameters['epochs']
output_num = parameters['output_num']
output_fea = parameters['output_fea']

if torch.cuda.is_available():
    print(f'GPU available  : {torch.cuda.is_available()}')
    print(f'GPU count      : {torch.cuda.device_count()}')
    print(f'GPU index      : {torch.cuda.current_device()}')
    print(f'GPU name       : {torch.cuda.get_device_name()}')
    print('Training on GPU!')
else:
    print('Training on CPU!')


def train(model, train_dataloader, valid_dataloader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5, last_epoch=-1)
    if os.path.exists('checkpoints/checkpoint.pth'):
        print('there has a well-trained model.\n'
              'loading and continue training\n')
        checkpoint = torch.load('checkpoints/checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
    best_epoch = 0
    train_loss = []
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for i, (traj, heat, ssh, sst, sss, curr) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_traj = traj[:, [0, 1, -2, -1]]
            target_traj = traj[:, 1:-1:2, :2]
            results = model(input_traj, heat, ssh, sst, sss, curr)
            loss = criterion(results, target_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        lr_scheduler.step()
        tr_loss /= len(train_dataloader)
        train_loss.append(tr_loss)# .cpu().detach().numpy()
        print(f'[Epoch {epoch + 1}]  [Tloss {round(tr_loss * 10000, 2)}]  ', end='')

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for traj, heat, ssh, sst, sss, curr in valid_dataloader:
                input_traj = traj[:, [0, 1, -2, -1]]
                target_traj = traj[:, 1:-1:2, :2]
                results = model(input_traj, heat, ssh, sst, sss, curr)
                loss = criterion(results, target_traj)
                va_loss += loss.item()
            va_loss /= len(valid_dataloader)
            print(f'[Vloss {round(va_loss * 10000, 2)}]  ', end='')
        if va_loss < best_loss:
            best_epoch = epoch
            best_loss = va_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': lr_scheduler.state_dict()}
            torch.save(checkpoint, 'checkpoints/checkpoint.pth')
        print(f'[Best Loss at {best_epoch + 1}]')
        if best_epoch - epoch > early_stop_patients:
            exit(f'model has no decreased in {early_stop_patients} epochs, stop training......')
    train_loss = np.array(train_loss).reshape(-1)
    np.save('checkpoints/loss.npy', train_loss)

if __name__ == '__main__':
    print('start training...')
    train(model=Interpolation(use_sst_block=True,
                              use_sss_block=True,
                              use_ssh_block=True,
                              use_current_block=True,
                              use_heatmap_block=True,
                              use_lstm=True).to(device),
          train_dataloader=get_dataloader('train', batch_size=batch_size, shuffle=True, drop_last=True),
          valid_dataloader=get_dataloader('valid', batch_size=batch_size * 4, shuffle=False, drop_last=False))
