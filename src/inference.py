# src/inference.py

import os
import torch
import cv2
import numpy as np
from pynput.keyboard import Controller, Key
import mss
import time
from PIL import Image
import torchvision.transforms as transforms
from model import ResNetModel, SimpleCNN
import pygetwindow as gw
import logging
from datetime import datetime

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_window_bounds(window_title):
    """
    指定したウインドウタイトルに一致するウインドウの位置とサイズを取得します。
    """
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        logging.error(f"ウインドウ '{window_title}' が見つかりませんでした。")
        exit()
    window = windows[0]
    if window.isMinimized:
        window.restore()
        time.sleep(0.5)  # ウインドウが復元されるのを待つ
    window.activate()
    time.sleep(0.5)  # ウインドウがアクティブになるのを待つ
    left, top, width, height = window.left, window.top, window.width, window.height
    return {"left": left, "top": top, "width": width, "height": height}

def generate_timestamp():
    """
    現在の日時を 'YYYYMMDD_HHMMSS_mmm' 形式の文字列として返します。
    ミリ秒まで含めることで、1秒内に複数回キャプチャした場合でも一意性を保ちます。
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
    return timestamp

def main():
    # ウインドウタイトルを指定
    window_title = "Forza Motorsport"  # ここをキャプチャしたいウインドウのタイトルに変更

    # ウインドウの位置とサイズを取得
    monitor = get_window_bounds(window_title)
    logging.info(f"キャプチャ対象のウインドウ位置: {monitor}")

    # キーボードコントローラーの初期化
    keyboard = Controller()

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルのロード
    keys = ['w', 'a', 's', 'd', 'space']
    model = ResNetModel(num_keys=len(keys), pretrained=False)  # トレーニング済みモデルを使用
    model_path = './models/best_model.pth'
    if not os.path.exists(model_path):
        logging.error(f"モデルファイル '{model_path}' が見つかりません。トレーニングが必要です。")
        exit()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    time.sleep(0.5)  # ウインドウがアクティブになるのを待つ

    # スクリーンキャプチャの設定
    sct = mss.mss()

    # 画像の前処理
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # キーとモデルの出力をマッピング
    key_map = keys  # ラベルと一致させる

    # 押されているキーを管理
    pressed_keys = set()

    try:
        while True:
            # スクリーンショットの取得
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            image = Image.fromarray(img)

            # 画像の前処理
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)

            # 推論
            with torch.no_grad():
                output = model(input_tensor)
                output = output.cpu().numpy()[0]

            # 閾値を設定してキー入力を決定
            threshold = 0.5
            keys_to_press = [key_map[i] for i in range(len(key_map)) if output[i] > threshold]

            # キーの押下と解放
            # まず、以前押されていたが現在押すべきでないキーを解放
            keys_to_release = pressed_keys - set(keys_to_press)
            for key in keys_to_release:
                if key == 'space':
                    keyboard.release(Key.space)
                elif key == 'a':
                    keyboard.release("a")
                elif key == 'd':
                    keyboard.release("d")
                elif key == 'w':
                    keyboard.release("w")
                elif key == 's':
                    keyboard.release("s")
                pressed_keys.discard(key)

            # 現在押すべきキーを押下
            for key in keys_to_press:
                if key not in pressed_keys:
                    if key == 'space':
                        keyboard.press(Key.space)
                    elif key == 'a':
                        keyboard.press("a")
                    elif key == 'd':
                        keyboard.press("d")
                    elif key == 'w':
                        keyboard.press("w")
                    elif key == 's':
                        keyboard.press("s")
                    pressed_keys.add(key)

            # フレームレートの調整
            time.sleep(0.05)  # 20 FPS
    except KeyboardInterrupt:
        # すべてのキーを解放
        for key in pressed_keys:
            if key == 'space':
                keyboard.release(Key.space)
            elif key == 'a':
                keyboard.release("a")
            elif key == 'd':
                keyboard.release("d")
            elif key == 'w':
                keyboard.release("w")
            elif key == 's':
                keyboard.release("s")
        logging.info("リアルタイム制御を終了します。")

if __name__ == "__main__":



    main()
