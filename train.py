import time

from collections import OrderedDict
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from dataset import *
import os
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# 数据参数
data_folder = './archive/h5'  # 包含数据文件的文件夹
data_name = 'flickr8k_10_cap_per_img_3_min_word_freq'  # 数据文件的基本名称

# 模型参数
emb_dim = 200  # 词嵌入的维度
attention_dim = 512  # 注意力线性层的维度
decoder_dim = 512  # 解码器RNN的维度
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
cudnn.benchmark = True  # 仅当模型输入为固定尺寸时设为True；否则会有大量计算开销

# 训练参数
start_epoch = 0
epochs = 120  # 训练的轮数（如果没有触发早停）
epochs_since_improvement = 0  # 记录自上次验证BLEU分数改进以来的轮数
batch_size = 32
workers = 0  # 数据加载的worker数量，目前只有1进程(主进程)个用于h5py
encoder_lr = 1e-4  # 编码器的学习率（如果进行微调1e-4）
decoder_lr = 4e-4  # 解码器的学习率
grad_clip = 5.  # 梯度裁剪的绝对值
alpha_c = 1.  # “双随机注意力”的正则化参数，如论文中所述
best_bleu4 = 0.  # 当前的BLEU-4分数
print_freq = 10  # 每隔__批次打印训练/验证状态
fine_tune_encoder = True  # 微调编码器
checkpoint_path = None  # 查点路径，如果没有则为None
encoder_t = 'dense'  # 设置编码器网络
pre_model = './premodel/densenet121.pth'
encoder_dim = 2048
if encoder_t == 'dense':
    encoder_dim = 1024


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # 读取词映射文件
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 初始化或加载检查点
    if checkpoint_path is None:
        # 初始化解码器
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=encoder_dim,
                                       dropout=dropout)
        em = torch.load('./archive/h5/EMBEDDINGS_flickr8k_10_cap_per_img_3_min_word_freq.pth')
        decoder.load_pretrained_embeddings(em)
        # decoder.fine_tune_embeddings()
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        # 初始化编码器
#         if encoder_t == "res":

#             pretrained_model = torchvision.models.resnet50(pretrained=True)
#             pretrained_state_dict = pretrained_model.state_dict()
#             new_state_dict = OrderedDict()
#             for k, v in pretrained_state_dict.items():
#                 new_key = k.replace('conv1', 'conv0.0') \
#                     .replace('bn1', 'conv0.1') \
#                     .replace('downsample', 'shortcut') \
#                     .replace('layer', 'layer')
#                 new_state_dict[new_key] = v

        encoder = Encoder(path=pre_model, dtype=encoder_t)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # 移动到GPU，如果可用
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 自定义数据加载器
    # 数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', ima_transform=data_transforms['train']),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', ima_transform=data_transforms['val']),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # 训练和验证的轮次
    for epoch in range(start_epoch, epochs):

        # 如果连续4个epoch没有提升，则降低学习率，并在总计20个epoch后终止训练
        if epochs_since_improvement == 12:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.9)

        # 一个epoch的训练
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # 一个epoch的验证
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # 检查是否有改进
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # 保存检查点
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    执行一个epoch的训练

    Args:
        train_loader: 训练数据的DataLoader
        encoder: 编码器模型
        decoder: 解码器模型
        criterion: 损失函数
        encoder_optimizer: 优化器，用于更新编码器的权重（如果进行微调）
        decoder_optimizer: 优化器，用于更新解码器的权重
        epoch: 当前的epoch数
    """
    decoder.train()  # 训练模式（使用dropout和batchnorm）
    encoder.train()

    batch_time = AverageMeter()  # 前向传播和反向传播时间
    data_time = AverageMeter()  # 数据加载时间
    losses = AverageMeter()  # 每个词的损失
    top5accs = AverageMeter()  # top5准确率

    start = time.time()

    # 批次
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # 移动到GPU，如果可用
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # print(caps.shape)
        # print(caplens.shape)

        # 前向传播
        imgs = encoder(imgs)
        # print(imgs.size())
        # print(caps.size())
        # print(caplens.size())
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # 由于解码是从<start>开始的，因此目标是<start>之后的所有词，直到<end>
        targets = caps_sorted[:, 1:]

        # 移除未解码的时间步或填充的时间步
        # pack_padded_sequence是一个简单的方法来做到这一点
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # 计算损失
        loss = criterion(scores, targets)

        # 添加双随机注意力正则化
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # 反向传播
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # 更新权重
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # 记录指标
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # 打印状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    执行一个epoch的验证

    Args:
        val_loader: 验证数据的DataLoader
        encoder: 编码器模型
        decoder: 解码器模型
        criterion: 损失函数
    Returns:
        float: BLEU-4分数
    """
    decoder.eval()  # 验证模式（不使用dropout和batchnorm）
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # 用于计算BLEU-4分数的参考
    hypotheses = list()  # 预测

    # 禁用梯度计算
    with torch.no_grad():
        # 批次
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            # 移动到设备，如果可用
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 前向传播
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # 由于解码是从<start>开始的，因此目标是<start>之后的所有词，直到<end>
            targets = caps_sorted[:, 1:]

            # 移除未解码的时间步或填充的时间步
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # 计算损失
            loss = criterion(scores, targets)

            # 添加双随机注意力正则化
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # 记录指标
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # 存储参考用于每张图像
            sort_ind = sort_ind.to(allcaps.device)
            allcaps = allcaps[sort_ind]  # 因为图像在解码器中被排序
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # 移除<start>和填充
                references.append(img_captions)

            # 假设
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # 移除填充
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # 计算BLEU-4分数
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
