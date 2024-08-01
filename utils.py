"""
这里我们完成一些其他的方法
"""
import numpy as np
import torch


# 均匀分布词嵌入
def init_embedding(tensor):  # (vocab_num, dim)
    """
    :param tensor: embedding tensor
    :return: None
    """
    # 计算偏差值, 均匀分布的上下界
    bias = np.sqrt(3.0 / tensor.size(1))
    # 使用均匀分布初始化嵌入张量
    torch.nn.init.uniform_(tensor, -bias, bias)


# x = torch.empty(100, 16)
# print(x)
# init_embedding(x)
# print(x)

def load_embeddings(emb_file, word_map):
    # 如果我们有一个预先的词嵌入模型文件, 我们可以使用这个文件为我们训练的词表做词嵌入
    """
    :param emb_file: 词嵌入模型文件
    :param word_map: 词表
    :return: 返回嵌入张量, 嵌入维度
    """
    with open(emb_file, 'r') as e:
        emb_dim = len(e.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # 创建一个张量embeddings来存储嵌入向量，并使用init_embedding函数进行初始化
    embeddings = torch.FloatTensor(len(vocab), emb_dim)  # (vocab_num, dim)
    init_embedding(embeddings)

    # 读取词嵌入文件
    print("\n正在进行embeddings......")

    # 一次性读取整个词嵌入文件到内存
    for line in open(emb_file, 'r', encoding='utf8'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # 如果词不在词表，则不进行嵌入
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


# word_map = {'hello': 0, 'world': 1, 'python': 2, 'I': 3}
# emb_file = './glove.6B/glove.6B.50d.txt'
#
# embeddings, emb_dim = load_embeddings(emb_file, word_map)
#
# print(embeddings)
# print(f'嵌入维度: {emb_dim}')


def clip_gradient(optimizer, grad_clip):
    # 在反向传播过程中对梯度进行裁剪，以避免梯度爆炸
    """
    :param optimizer: 优化器对象
    :param grad_clip: 裁剪梯度的阈值
    :return: None
    """
    for group in optimizer.param_groups:  # 参数组
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


#
# import torch
# import torch.nn as nn
# import torch.optim as optim


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc1 = nn.Linear(2, 5)  # 输入维度为2，输出维度为10
#         self.fc2 = nn.Linear(5, 2)  # 输入维度为10，输出维度为2，二类分类问题
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # 模拟输入数据和目标数据
# inputs = torch.randn(5, 2)  # 5个样本，每个样本2维
# targets = torch.randint(0, 2, (5,))  # 5个样本的目标标签，类别为0、1
#
# # 初始化模型、损失函数和优化器
# model = MyModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 前向传播
# outputs = model(inputs)
# loss = criterion(outputs, targets)
#
# # 反向传播
# loss.backward()
#
# # 梯度裁剪
# clip_gradient(optimizer, grad_clip=0.01)
#
# # 更新参数
# optimizer.step()
#
# # 打印模型参数
# for name, param in model.named_parameters():
#     print(f"参数{name}, type: {param}")
#     print(f"参数的额梯度 of {name}:")
#     print(param.grad)
#     print("-----------------------")


class AverageMeter(object):
    # 这是一个数据跟踪类, 用于记录训练过程中的一些统计量指标

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()  # 内部变量重置为零!!!!!!

    def reset(self):
        self.val = 0  # 之前记录的最近值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    # 更新统计量，传入新的值val和对应值的权重n, 然后根据此计算sum和avg
    def update(self, val, n=1):
        """
        :param val: 新的值
        :param n: 对应值的权重（通常是该值所代表的样本数）
        :return: None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    # 一个收缩学习率的函数
    """
    :param optimizer: 需要收缩学习率的优化器
    :param shrink_factor: 乘以学习率的因子(区间为 (0, 1))
    """

    print("\n收缩学习率ing......")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("现在新的学习率是 %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    # 一个模型精确度评估的方法, 该函数计算模型预测的前 ( k ) 个标签(词)中的准确度
    """
    :param scores: 模型评估得分
    :param targets: 真实标签s(批量)
    :param k: 用于计算 top-k accuracy 的 ( k ) 值
    :return: top-k 精度
    """

    batch_size = targets.size(0)  # 获取批量大小
    _, ind = scores.topk(k, 1, True, True)  # 获取每个输入的前 k 个最高分的索引
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))  # 检查这些索引是否与真实标签匹配
    correct_total = correct.view(-1).float().sum()  # 计算匹配的总数
    return correct_total.item() * (100.0 / batch_size)  # 返回 top-k 准确度的百分比


# import torch

# # 模拟的预测分数 (batch_size=10, num_classes=5)
# scores = torch.tensor([
#     [0.1, 0.2, 0.3, 0.2, 0.2],
#     [0.1, 0.2, 0.1, 0.5, 0.1],
#     [0.7, 0.1, 0.1, 0.05, 0.05],
#     [0.3, 0.3, 0.2, 0.1, 0.1],
#     [0.1, 0.4, 0.1, 0.3, 0.1],
#     [0.05, 0.05, 0.8, 0.05, 0.05],
#     [0.2, 0.3, 0.4, 0.05, 0.05],
#     [0.5, 0.1, 0.2, 0.1, 0.1],
#     [0.2, 0.5, 0.1, 0.15, 0.05],
#     [0.1, 0.3, 0.4, 0.1, 0.1]
# ])
#
# # 真实标签
# # targets = torch.tensor([2, 3, 0, 1, 1, 2, 2, 0, 1, 2])  # 100%
# targets = torch.tensor([0, 3, 0, 1, 1, 2, 2, 0, 1, 2])  # 90%
#
# # 计算 top-3 准确度
# k = 3
# top_k_accuracy = accuracy(scores, targets, k)
# print(f'Top-{k} accuracy: {top_k_accuracy:.2f}%')


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    # 保存模型训练的检查点
    """
    :param data_name: 处理过的数据集的基本名称
    :param epoch: 当前的训练轮数(epoch)
    :param epochs_since_improvement: 自上次 BLEU-4 分数改善以来的 epoch 数量
    :param encoder: 编码器模型
    :param decoder: 解码器模型
    :param encoder_optimizer: 用于更新编码器权重的优化器，如果在微调(find_tuning)编码器的话
    :param decoder_optimizer: 用于更新解码器权重的优化器
    :param bleu4: 当前 epoch 的{验!证!集!} BLEU-4 分数
    :param is_best: 当前检查点是否是最佳的
    :return: None
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # 如果 is_best 为真，则将当前检查点另存为一个特殊文件，以免被后续的较差检查点覆盖
    if is_best:
        torch.save(state, 'BEST_' + filename)
