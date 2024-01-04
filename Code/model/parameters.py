# 用字典存储模型所需参数
# 字典用dict生成，里面key=value
import torch
device_id = 1 if torch.cuda.device_count() > 1 else 0


def get_parameters():
    cuda = True if torch.cuda.is_available() else False
    device = torch.device(f'cuda:{device_id}' if cuda else "cpu")
    # 训练参数设置
    learning_rate = 1e-3
    early_stop_patients = 200
    batch_size = 256
    epochs = 1200

    # 模型参数设置
    traj_num = 4
    traj_features = 5
    output_num = 21
    output_fea = 2
    mlp_features = 512

    heat_channel = 1
    heat_width = 9
    heat_height = 9

    envs_channel = 1
    envs_width = 9
    envs_height = 9

    curr_channel = 3
    curr_width = 9
    curr_height = 9

    lstm_unit = 256
    lstm_layers = 3
    d_ff = 2048
    dropout = 0.1
    bidirectional = False
    conv_channel = 64

    parameters = dict(cuda=cuda,
                      device=device,
                      learning_rate=learning_rate,
                      epochs=epochs,
                      early_stop_patients=early_stop_patients,
                      batch_size=batch_size,
                      mlp_features=mlp_features,
                      lstm_unit=lstm_unit,
                      lstm_layers=lstm_layers,
                      d_ff=d_ff,
                      dropout=dropout,
                      bidirectional=bidirectional,
                      conv_channel=conv_channel,
                      traj_num=traj_num,
                      traj_features=traj_features,
                      output_num=output_num,
                      output_fea=output_fea,

                      heat_channel=heat_channel,
                      heat_width=heat_width,
                      heat_height=heat_height,

                      envs_channel=envs_channel,
                      envs_width=envs_width,
                      envs_height=envs_height,

                      curr_channel=curr_channel,
                      curr_width=curr_width,
                      curr_height=curr_height,
                      )
    return parameters
