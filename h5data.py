"""
这里把数据集封装成h5格式
"""
import os
import json
import h5py
import tqdm
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from random import seed, sample, choice
from collections import Counter
import torch
import torch.nn as nn


def init_embedding(embeddings):
    """
    使用均匀分布初始化词嵌入
    :param embeddings: 词嵌入张量
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    从预训练的词嵌入文件加载词嵌入
    :param emb_file: 词嵌入模型文件路径
    :param word_map: 词汇表
    :return: 嵌入张量, 嵌入维度
    """
    # 读取词嵌入维度
    with open(emb_file, 'r', encoding='utf8') as e:
        emb_dim = len(e.readline().split()) - 1

    vocab = set(word_map.keys())

    # 创建一个张量embeddings来存储嵌入向量，并使用init_embedding函数进行初始化
    embeddings = torch.FloatTensor(len(vocab), emb_dim)  # (vocab_num, dim)
    init_embedding(embeddings)

    # 读取词嵌入文件
    print("\n正在进行embeddings......")

    # 一次性读取整个词嵌入文件到内存
    for line in open(emb_file, 'r', encoding='utf8'):
        line = line.split()

        emb_word = line[0]
        embedding = list(map(float, line[1:]))

        # 如果词不在词表，则不进行嵌入
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def create_h5_data(json_path, image_folder, min_word_freq, max_len, data_name, captions_per_image, output_folder,
                   emb_file=None):
    """创建h5数据集合处理原数据集"""

    # 读取json文件
    with open(json_path, 'r') as j:
        data = json.load(j)

    # 存储图像读取路径和描述
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []

    word_frequency = Counter()

    for item in data['images']:
        captions = []
        for s in item['sentences']:
            # 更新词频
            word_frequency.update(s['tokens'])
            if len(s['tokens']) <= max_len:
                captions.append(s['tokens'])

        # 如果当前图像没有一句有效的描述，则跳过此图像
        if len(captions) == 0:
            continue

        # 图像路径
        path = os.path.join(image_folder, item['filename'])

        # 根据数据分割类型将图像路径和描述添加到对应列表中
        if item['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif item['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif item['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # 检查路径和描述列表长度是否一致
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # 创建词汇表：只保留出现频率大于 min_word_freq 的单词
    words = [w for w in word_frequency.keys() if word_frequency[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 为所有输出文件创建基础名称
    base_filename = data_name + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # 将词汇表保存为JSON文件
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # 加载预训练词嵌入（如果提供）
    if emb_file is not None:
        embeddings, emb_dim = load_embeddings(emb_file, word_map)
        torch.save(embeddings, os.path.join(output_folder, 'EMBEDDINGS_' + base_filename + '.pth'))
        print(f"Embeddings saved with dimension {emb_dim}")

    # 设置随机种子以确保结果可重复
    seed(123)

    for impaths, imcaps, split_type in [(train_image_paths, train_image_captions, "TRAIN"),
                                        (val_image_paths, val_image_captions, "VAL"),
                                        (test_image_paths, test_image_captions, "TEST")]:
        with h5py.File(os.path.join(output_folder, split_type + "_IMAGES_" + base_filename + '.hdf5'), 'a') as h:

            h.attrs['captions_per_image'] = captions_per_image

            images = h.create_dataset("images", (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split_type)

            encode_captions = []
            caplens = []

            # 图像描述采样
            for i, path in enumerate(tqdm.tqdm(impaths)):
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # 检查每次采样是否符合规范
                assert len(captions) == captions_per_image

                # 读取图像
                img = imageio.imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # 调整图像大小
                img = np.array(Image.fromarray(img).resize((256, 256)))
                img = img.transpose((2, 0, 1))
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # 保存图像到数据文件
                images[i] = img

                # 填充描述词向量
                for j, cap in enumerate(captions):
                    encode_cap = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in cap] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(cap))

                    # 查找描述长度
                    cap_len = len(cap) + 2

                    encode_captions.append(encode_cap)
                    caplens.append(cap_len)

            # 验证数据一致性
            assert images.shape[0] * captions_per_image == len(encode_captions) == len(caplens)

            # 将编码后的描述和它们的长度保存为JSON文件
            with open(os.path.join(output_folder, split_type + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(encode_captions, j)

            with open(os.path.join(output_folder, split_type + '_CAP_LENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


flickr8k_path = "./archive/flickr8k.json"
image_path = "./archive/Images"
output = "./archive/h5/"
emb_file = "./glove.6B/glove.6B.200d.txt"

create_h5_data(json_path=flickr8k_path, image_folder=image_path, min_word_freq=3, max_len=40,
               data_name="flickr8k", captions_per_image=10, output_folder=output, emb_file=emb_file)

# em = torch.load('./archive/h5/EMBEDDINGS_flickr8k_10_cap_per_img_3_min_word_freq.pth')
# print(len(em))
