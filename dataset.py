import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    在这里实现torch的dataset类用来加载批量的数据
    """

    def __init__(self, data_folder, data_name, split, ima_transform=None):
        """
        :param data_folder: 数据集的文件路径
        :param data_name: 数据集的名称
        :param split: 划分数据集的种类，一般包括 'TRAIN'、'VAL' 或 'TEST'
        :param ima_transform: 对图像使用的变换管道
        """

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # 打开HDF5文件, 并加载图像数据
        self.h5data = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h5data['images']

        # 获取每张图片的描述数量
        self.cpi = self.h5data.attrs['captions_per_image']

        # 把词编码后的描述数据文件加载到内存中
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # 把词编码后的描述长度文件加载到内存中
        with open(os.path.join(data_folder, self.split + '_CAP_LENS_' + data_name + '.json'), 'r') as j:
            self.cap_lens = json.load(j)

        # 设置图像变换方法
        self.transform = ima_transform

        # 记录数据样本数
        self.data_size = len(self.captions)

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        # 获取数据样本, img: 从 HDF5 文件中加载并标准化的图像
        img = torch.FloatTensor(self.imgs[item // self.cpi] / 255.)

        # caption: 当前索引对应的描述
        caption = torch.LongTensor(self.captions[item])

        # cao_lens当前索引对用的描述长度
        # cap_lens = torch.LongTensor(self.cap_lens[item])
        # 如果直接传递一个整形, 结果会返回一个self.cap_lens[item]数值长度大小的张量列表,我也不知道为什么
        # 可以使用torch.tensor或者用列表包裹解决这个问题
        # cap_lens = torch.tensor(self.cap_lens[item], dtype=torch.long)
        cap_lens = torch.LongTensor([self.cap_lens[item]])

        if self.split == 'TRAIN':
            return img, caption, cap_lens
        else:
            # 在验证或测试模式下, 需要返回所有描述以计算 BLEU-4 分数
            all_captions = torch.LongTensor(
                self.captions[((item // self.cpi) * self.cpi):(((item // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, cap_lens, all_captions


# # 验证数据加载器是否正常工作
# # 训练集加载
# data_folder = './archive/h5/'
# data_name = 'flickr8k_10_cap_per_img_3_min_word_freq'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# batch_size = 2
# train_loader = torch.utils.data.DataLoader(
#     CaptionDataset(data_folder, data_name, 'TRAIN', ima_transform=transforms.Compose([normalize])),
#     batch_size=batch_size, shuffle=True, pin_memory=True)
#
# for i, (imgs, caps, caplens) in enumerate(train_loader):
#     imgs = imgs
#     caps = caps
#     caplens = caplens
#     print(imgs.size())
#     print(caps.size())
#     print(caplens.size())
#     break


# 测试集部分
# data_folder = './archive/h5/'
# data_name = 'flickr8k_10_cap_per_img_3_min_word_freq'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# batch_size = 2
# train_loader = torch.utils.data.DataLoader(
#     CaptionDataset(data_folder, data_name, 'TEST', ima_transform=transforms.Compose([normalize])),
#     batch_size=batch_size, shuffle=True, pin_memory=True)
#
# for i, (imgs, caps, caplens, all_captions) in enumerate(train_loader):
#     imgs = imgs
#     caps = caps
#     caplens = caplens
#     all_captions = all_captions
#     break
