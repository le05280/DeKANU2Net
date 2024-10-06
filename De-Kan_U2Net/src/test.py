import torch
import torch.nn as nn
from model1 import u2net_full


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
        return param_count


if __name__ == '__main__':
    # 创建网络模型实例。假设输入图像有3个通道，且我们的目标是二分类。则n_classes=2
    model = u2net_full()
    # 计算并打印模型参数数量
    param = count_param(model)

    print('U2Net total parameters: %.2fM (%d)' % (param / 1e6, param))
    input = torch.randn(50, 512, 7, 7)

    ca = ChannelAttention(dim=512, reduction=8)
    output = ca(input)
    print(output.shape)


