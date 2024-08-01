import torch
import torch.nn as nn


def init_parameters(input_size, hidden_size, bias=True):
    """初始化GRU单元的参数

    Args:
        input_size (int): 输入特征的维度
        hidden_size (int): 隐藏状态的维度
        bias (bool): 是否使用偏置项, 默认为True

    Returns:
        tuple: 包含W, U和b(如果bias为True)的张量
    """
    W = nn.Parameter(torch.Tensor(input_size, hidden_size))
    U = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    if bias:
        b = nn.Parameter(torch.Tensor(hidden_size))
        return W, U, b
    else:
        return W, U, None


class GRUCell(nn.Module):
    """实现GRU单元。

    Attributes:
        input_size (int): 输入特征的维度
        hidden_size (int): 隐藏状态的维度
        bias (bool) : 是否使用偏置项
        W_r (nn.Parameter): 重置门的权重矩阵
        U_r (nn.Parameter): 重置门的递归权重矩阵
        b_r (nn.Parameter): 重置门的偏置项
        W_z (nn.Parameter): 更新门的权重矩阵
        U_z (nn.Parameter): 更新门的递归权重矩阵
        b_z (nn.Parameter): 更新门的偏置项
        W_h (nn.Parameter): 新记忆内容的权重矩阵
        U_h (nn.Parameter): 新记忆内容的递归权重矩阵
        b_h (nn.Parameter): 新记忆内容的偏置项
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """初始化GRU单元

        Args:
            input_size (int): 输入特征的维度
            hidden_size (int): 隐藏状态的维度
            bias (bool): 是否使用偏置项。默认为True
        """
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # 初始化参数
        self.W_r, self.U_r, self.b_r = init_parameters(self.input_size, self.hidden_size, self.bias)
        self.W_z, self.U_z, self.b_z = init_parameters(self.input_size, self.hidden_size, self.bias)
        self.W_h, self.U_h, self.b_h = init_parameters(self.input_size, self.hidden_size, self.bias)

        self.init_weights()

    def init_weights(self):
        """初始化权重。"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    @staticmethod
    def linear(x, h, W, U, b):
        """线性变换

        Args:
            x (torch.Tensor): 输入张量
            h (torch.Tensor): 上一时间步的隐藏状态
            W (nn.Parameter): 输入到隐藏的权重矩阵
            U (nn.Parameter): 隐藏到隐藏的权重矩阵
            b (nn.Parameter): 偏置项(如果有)

        Returns:
            torch.Tensor: 线性变换后的张量
        """
        if b is not None:
            return x @ W + h @ U + b
        else:
            return x @ W + h @ U

    def forward(self, x, h_prev):
        """前向传播

        Args:
            x (torch.Tensor): 当前时间步的输入
            h_prev (torch.Tensor): 上一时间步的隐藏状态

        Returns:
            torch.Tensor: 当前时间步的隐藏状态
        """
        # 计算重置门
        r = torch.sigmoid(self.linear(x, h_prev, self.W_r, self.U_r, self.b_r))

        # 计算更新门
        z = torch.sigmoid(self.linear(x, h_prev, self.W_z, self.U_z, self.b_z))

        # 计算新记忆内容
        h_hat = torch.tanh(self.linear(x, r * h_prev, self.W_h, self.U_h, self.b_h))

        # 计算最终隐藏状态
        h_next = (1 - z) * h_hat + z * h_prev

        return h_next
