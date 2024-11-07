# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import RacingDataset
from model import ResNetModel, SimpleCNN
import torchvision.transforms as transforms
import copy

def main():
    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNetの平均
                             std=[0.229, 0.224, 0.225])   # ImageNetの標準偏差
    ])

    # キーの定義
    # keys = ['left', 'right', 'up', 'down', 'space']
    keys = ['w', 'a', 's', 'd', 'space']

    # データセットとデータローダーの作成
    dataset = RacingDataset(images_dir='../data/images',
                            labels_dir='../data/labels',
                            transform=transform,
                            keys=keys)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # モデルのインスタンス化
    model = ResNetModel(num_keys=len(keys), pretrained=True)  # または SimpleCNN(num_keys=len(keys))
    model = model.to(device)

    # 損失関数とオプティマイザーの設定
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # トレーニングと検証のループ
    num_epochs = 25
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 各エポックでトレーニングと検証を行う
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            # データをイテレート
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 勾配の初期化
                optimizer.zero_grad()

                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # トレーニング時のみ逆伝播とオプティマイザー
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # ベストモデルの保存
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # ベストモデルのロード
    model.load_state_dict(best_model_wts)
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/best_model.pth')
    print('トレーニング完了。ベストモデルを保存しました。')

if __name__ == "__main__":
    main()
