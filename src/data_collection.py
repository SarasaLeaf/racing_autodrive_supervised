# src/data_collection.py

import os
import cv2
import numpy as np
from pynput import keyboard
import mss
import time

# 保存ディレクトリの設定
os.makedirs('../data/images', exist_ok=True)
os.makedirs('../data/labels', exist_ok=True)

# キー入力の記録用変数
current_keys = set()

def on_press(key):
    try:
        current_keys.add(key.char)
    except AttributeError:
        current_keys.add(key.name)

def on_release(key):
    try:
        current_keys.discard(key.char)
    except AttributeError:
        current_keys.discard(key.name)
    if key == keyboard.Key.esc:
        # Escキーで終了
        return False

def main():
    # キーボードリスナーの開始
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # スクリーンキャプチャの設定
    sct = mss.mss()
    monitor = sct.monitors[1]  # メインモニターを指定

    count = 0
    try:
        while True:
            # スクリーンショットの取得
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            # 画像の保存
            img_path = f'../data/images/frame_{count:05d}.png'
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 現在のキー入力を保存
            label = ' '.join(sorted(current_keys))
            label_path = f'../data/labels/frame_{count:05d}.txt'
            with open(label_path, 'w') as f:
                f.write(label)

            count += 1
            time.sleep(0.05)  # 20 FPS
    except KeyboardInterrupt:
        print("データ収集を終了します。")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
