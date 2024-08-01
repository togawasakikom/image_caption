"""
在这里读取txt文本转成json格式方便后续的处理
"""
import json
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 设定文件路径
input_file = './archive/captions.txt'
output_file = './archive/flickr8k.json'

images = []
captions_dict = defaultdict(list)

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头行
    for row in reader:
        image_filename, caption = row
        captions_dict[image_filename].append(caption)

sentid = 0
imgid = 0

for filename, captions in captions_dict.items():
    img_entry = {
        "sentids": [],
        "imgid": imgid,
        "sentences": [],
        "filename": filename
    }

    for caption in captions:
        tokens = caption.strip('.').lower().split()
        sentence_entry = {
            "tokens": tokens,
            "raw": caption,
            "imgid": imgid,
            "sentid": sentid
        }
        img_entry["sentences"].append(sentence_entry)
        img_entry["sentids"].append(sentid)
        sentid += 1

    images.append(img_entry)
    imgid += 1

# 划分数据集
train_images, test_images = train_test_split(images, test_size=0.05, random_state=42)
train_images, val_images = train_test_split(train_images, test_size=0.05, random_state=42)

# 添加split字段
for img in train_images:
    img["split"] = "train"
for img in val_images:
    img["split"] = "val"
for img in test_images:
    img["split"] = "test"

# 写入json文件
with open(output_file, 'w') as f:
    json.dump({"images": train_images + val_images + test_images}, f, indent=4)

print(f" 数据成功写入 to {output_file}")
