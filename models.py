import torch
import torch.nn as nn
import os
import random
from module.ResNet import resnet50, resnet18
from module.DenseNet import densenet121
from module.GRUCell import GRUCell
from torch.nn import LSTMCell
import torchvision
import torchvision.transforms as transforms

# 选择设备 (GPU或CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    编码器类，使用ResNet对图像进行编码
    """

    def __init__(self, path=None, dtype='res', *, encoded_image_size=14):
        """初始化编码器.

        Args:
            encoded_image_size (int): 编码图像的尺寸. 默认为14
        """
        # global net
        global modules
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.dtype = dtype

        if dtype == 'res':
            net = resnet50()
            modules = list(net.children())[:-2]

        if dtype == "dense":
            net = densenet121()
            modules = list(net.children())

            if path:
                net.load_state_dict(torch.load(path), strict=False)

        self.net = nn.Sequential(*modules)
        # print(self.net)

        # 自适应平均池化，将特征图调整到指定的尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # 允许微调部分卷积层
        self.fine_tune()

    def forward(self, images):
        """前向传播

        Args:
            images (torch.Tensor): 图像张量, 尺寸为 (batch_size, 3, image_size, image_size)

        Returns:
            torch.Tensor: 编码后的图像
        """
        out = self.net(images)  # (batch_size, 2048 or 1024, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print(out.size())
        return out

    def fine_tune(self, fine_tune=True):
        """
        允许或禁止对指定模型的卷积层进行梯度计算

        Args:
            model_name (str): 指定的模型名称，支持 'res' 或 'dens'
            fine_tune (bool): 允许微调
        """
        supported_models = ['res', 'dense']

        if self.dtype not in supported_models:
            raise ValueError(f"Unsupported model '{self.dtype}'. Supported models are: {', '.join(supported_models)}.")

        for p in self.net.parameters():
            p.requires_grad = False  # 冻结所有参数

        # 根据模型名称解冻特定层级
        if fine_tune:
            if self.dtype == 'res':
                # 解冻 ResNet-50 中的第5到末尾的所有层
                for c in list(self.net.children())[5:]:
                    for p in c.parameters():
                        p.requires_grad = True
            elif self.dtype == 'dense':
                # 解冻 DenseNet-121 中的'denseblock'和'transition'
                for name, param in self.net.named_parameters():
                    if 'denseblock' in name or 'transition' in name:
                        param.requires_grad = True


class Attention(nn.Module):
    """
    注意力网络类
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """初始化注意力网络

        Args:
            encoder_dim (int): 编码图像的特征维度
            decoder_dim (int): 解码器RNN的尺寸
            attention_dim (int): 注意力网络的尺寸
        """
        super(Attention, self).__init__()
        # 线性层将编码后的图像特征转化为注意力空间
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # 线性层将解码器的隐藏状态转化为注意力空间
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # 线性层计算需要进行softmax的值
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # softmax层计算注意力权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """前向传播

        Args:
            encoder_out (torch.Tensor): 编码图像, 尺寸为 (batch_size, num_pixels, encoder_dim)
            decoder_hidden (torch.Tensor): 解码器上一个时间步的输出, 尺寸为 (batch_size, decoder_dim)

        Returns:
            tuple: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # 计算注意力得分并通过relu激活
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        # 计算加权后的编码图像
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    使用循环神经网络和注意力机制的解码器类
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """初始化解码器

        Args:
            attention_dim (int): 注意力网络的尺寸
            embed_dim (int): 嵌入层的尺寸
            decoder_dim (int): 解码器RNN的尺寸
            vocab_size (int): 词汇表的大小
            encoder_dim (int): 编码图像的特征尺寸
            dropout (float): dropout率
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # 注意力网络
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        # 解码单元
        self.decode_step = LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        # 初始化隐藏状态的线性层
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # 创建一个sigmoid激活的门控
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # 线性层，计算词汇表上的得分
        self.fc = nn.Linear(decoder_dim, vocab_size)
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """用均匀分布初始化一些参数, 以便更容易收敛
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """使用预训练的嵌入层加载嵌入层

        Args:
            embeddings (torch.Tensor): 预训练的嵌入层.
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        基于编码图像创建解码器的初始隐藏状态

        Args:
            encoder_out (torch.Tensor): 编码图像, 尺寸为 (batch_size, num_pixels, encoder_dim) 的张量.

        Returns:
            torch.Tensor: 初始隐藏状态, 尺寸为 (batch_size, decoder_dim).
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, teacher_forcing_ratio=0.5):
        """前向传播

        Args:
            encoder_out (torch.Tensor): 编码图像, 尺寸为 (batch_size, enc_image_size, enc_image_size, encoder_dim) 的张量.
            encoded_captions (torch.Tensor): 编码后的字幕, 尺寸为 (batch_size, max_caption_length) 的张量.
            caption_lengths (torch.Tensor): 字幕长度, 尺寸为 (batch_size, 1) 的张量.

        Returns:
            tuple: 词汇表得分, 排序后的编码描述, 解码长度, 权重, 排序索引
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 展平图像
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # 按长度降序排序输入数据
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # 嵌入层
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # print(f"Embeddings size: {embeddings.size()}")

        # 初始化解码器状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 不在<end>位置解码，因为生成<end>就结束了
        # 所以，解码长度为实际长度 - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # 创建张量以存储词预测得分和alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # 在每个时间步，解码过程为：
        # 基于解码器的前一个隐藏状态输出对编码器的输出进行注意力加权
        # 然后在解码器中使用前一个单词和注意力加权编码生成一个新单词
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            if use_teacher_forcing and t < embeddings.size(1):
                teacher_force = embeddings[:batch_size_t, t, :]
                # print(f"Teacher forcing at step {t}, size: {teacher_force.size()}")
            else:
                if t == 0:  # 在第一步时没有以前的预测值
                    teacher_force = torch.zeros(batch_size_t, self.embed_dim).to(encoder_out.device)
                else:
                    teacher_force = self.embedding(predictions[:batch_size_t, t - 1, :].argmax(dim=1))
                    # print(f"No teacher forcing at step {t}, size: {teacher_force.size()}")

            # 计算注意力加权编码
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # 门控标量, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # 输入尺寸 print(f"Step {t}: teacher_force size = {teacher_force.size()}, attention_weighted_encoding size =
            # {attention_weighted_encoding.size()}")

            # 解码一步
            h, c = self.decode_step(
                torch.cat([teacher_force, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
