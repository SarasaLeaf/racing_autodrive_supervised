# src/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

class RacingDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, keys=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        assert len(self.image_files) == len(self.label_files), "画像とラベルの数が一致しません。"

        # キー入力の種類を定義
        self.keys = keys if keys is not None else ['left', 'right', 'up', 'down', 'space']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像の読み込み
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # ラベルの読み込み
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            keys = f.read().strip().split()

        # ラベルをバイナリベクトルに変換
        label = [0] * len(self.keys)
        for key in keys:
            if key in self.keys:
                label[self.keys.index(key)] = 1
        label = torch.tensor(label, dtype=torch.float)

        return image, label
