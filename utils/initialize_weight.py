from torch import nn
from torch.nn import init


def xavier_init(m):
    # 判断是否为卷积层或全连接层，并使用Xavier均匀分布初始化权重
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, (nn.Linear,)):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    # 对于循环层，比如LSTM、GRU等，同样进行初始化
    elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in m.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)
    # 对于归一化层，比如BatchNorm等，通常权重初始化为1，偏置初始化为0
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.constant_(m.weight.data, 1)
        init.zeros_(m.bias.data)
    return m
