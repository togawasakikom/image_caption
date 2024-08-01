import random
import torch
import math
import torch.nn.functional as F
from models import *
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的编码器和解码器模型
encoder = Encoder(dtype='res')
encoder.eval()
encoder.to(device)

decoder = DecoderWithAttention(attention_dim=512,
                               embed_dim=200,
                               decoder_dim=512,
                               vocab_size=3444,
                               encoder_dim=2048,
                               dropout=0.5)
decoder.to(device)
decoder.eval()


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    使用束搜索为图像生成描述。

    :param encoder: 编码器模型
    :param decoder: 解码器模型
    :param image_path: 图像路径
    :param word_map: 单词映射
    :param beam_size: 每个解码步骤考虑的序列数
    :return: 生成的描述，权重用于可视化
    """

    k = beam_size
    vocab_size = len(word_map)

    # 转换图像
    global attention_weight

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    img = np.array(img)

    img = img.transpose((2, 0, 1))

    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    image = transform(img)  # (3, 256, 256)
    # 编码
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # 展平编码
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # 我们将问题视为具有 k 的批处理大小
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # 张量存储每步的前 k 个单词；现在它们只是 <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # 张量存储前 k 个序列；现在它们只是 <start>
    seqs = k_prev_words  # (k, 1)

    # 张量存储前 k 个序列的分数；现在它们只是 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # 张量存储前 k 个序列的 alphas；现在它们只是 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # 列表存储完成的序列、它们的 alphas 和分数
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # 开始解码
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s 是一个小于或等于 k 的数字，因为一旦序列达到 <end> 就会从这个过程中移除
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # 门控标量, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # 加分
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # 对于第一步，所有 k 点将具有相同的分数（因为相同的 k 个前一个单词、h、c）
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # 展开并找到最高分及其展开索引
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # 将展开的索引转换为实际的分数索引
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # 将新单词添加到序列和 alphas 中
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        # 检查哪些序列是完整的（没有达到<end>）
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # 将完成的序列放到一边
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # 相应地减少 beam 长度

        # 如果没有完成的序列，选择分数最高的未完成序列
        if k == 0:
            if len(complete_seqs_scores) == 0:
                complete_seqs.extend(seqs.tolist())
                complete_seqs_alpha.extend(seqs_alpha.tolist())
                complete_seqs_scores.extend(top_k_scores)

            break

        # 继续处理未完成的序列
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # 如果过程持续时间过长则中断
        if step > 50:
            if len(complete_seqs_scores) == 0:
                complete_seqs.extend(seqs.tolist())
                complete_seqs_alpha.extend(seqs_alpha.tolist())
                complete_seqs_scores.extend(top_k_scores)
            break
        step += 1

    # 选择分数最高的序列
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    # 将索引转换为单词
    rev_word_map = {v: k for k, v in word_map.items()}
    sentence = ' '.join(
        [rev_word_map[idx] for idx in seq if idx not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    return sentence, alphas


# 打开 JSON 文件
with open('./archive/flickr8k.json', 'r') as f:
    image_data = json.load(f)
    test_data_test = []
    test_data_train = []
    for image in image_data["images"]:
        if image["split"] == "test":
            test_data_test.append(image['filename'])
        if image["split"] == "train":
            test_data_train.append(image['filename'])

img_name_test = random.choice(test_data_test)
ima_name_train = random.choice(test_data_train)

path_test = os.path.join('./archive/Images', img_name_test)
path_train = os.path.join('./archive/Images', ima_name_train)

word_map_file = './archive/h5/WORDMAP_flickr8k_10_cap_per_img_3_min_word_freq.json'

with open(word_map_file, 'r') as f:
    word_map = json.load(f)

s1, a1 = caption_image_beam_search(encoder=encoder, decoder=decoder, word_map=word_map,
                                   image_path=path_test)
s2, a2 = caption_image_beam_search(encoder=encoder, decoder=decoder, word_map=word_map,
                                   image_path=path_train)
print(img_name_test)
print(s1)

print(ima_name_train)
print(s2)


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    可视化每个单词对应的权重

    :param image_path: 被描述的图像路径
    :param seq: 描述
    :param alphas: 权重
    :param rev_word_map: 反向单词映射，即 ix2word
    :param smooth: 是否平滑权重？
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.Resampling.LANCZOS)

    words = []
    for ind in set(seq.split()):
        if ind in rev_word_map:
            words.append(rev_word_map[ind])
        else:
            words.append(0)

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(math.ceil(len(words) / 5), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        alphas = np.array(alphas)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.get_cmap())
        plt.axis('off')
    plt.show()


def main():
    # 加载预训练的编码器和解码器模型
    encoder = Encoder(dtype='res')
    encoder.eval()
    encoder.to(device)

    decoder = DecoderWithAttention(attention_dim=512,
                                   embed_dim=200,
                                   decoder_dim=512,
                                   vocab_size=3444,
                                   encoder_dim=2048,
                                   dropout=0.5)
    decoder.to(device)
    decoder.eval()

    # 加载 word map
    word_map_file = './archive/h5/WORDMAP_flickr8k_10_cap_per_img_3_min_word_freq.json'
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    # 反向 word map
    rev_word_map = {v: k for k, v in word_map.items()}

    # 打开 JSON 文件
    with open('./archive/flickr8k.json', 'r') as f:
        image_data = json.load(f)
        test_data = [image['filename'] for image in image_data['images'] if image['split'] == 'test']

    # 随机选择一个测试图像
    img_name = random.choice(test_data)
    path = os.path.join('./archive/Images', img_name)

    # 使用束搜索生成描述
    sentence, alphas = caption_image_beam_search(encoder, decoder, path, word_map, beam_size=3)

    # 可视化
    visualize_att(path, sentence, alphas, rev_word_map, smooth=True)

# if __name__ == '__main__':
#     main()
