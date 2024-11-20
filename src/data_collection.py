# src/data_collection.py

import os
import cv2
import numpy as np
from pynput import keyboard
import mss
import time
import pygetwindow as gw
import logging
from datetime import datetime

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_window_bounds(window_title):
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
    
    for ii in range( 3, 0, -1 ):
        print(f"{ii}", end = "\r")
        time.sleep(1)
    print("Start")

    # キーボードリスナーの開始
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # スクリーンキャプチャの設定
    sct = mss.mss()

    try:
        while True:
            # スクリーンショットの取得
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            # タイムスタンプの生成
            timestamp = generate_timestamp()

            # 画像の保存
            img_path = f'../data/images/frame_{timestamp}.png'
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 現在のキー入力を保存
            label = ' '.join(sorted(current_keys))
            label_path = f'../data/labels/frame_{timestamp}.txt'
            with open(label_path, 'w') as f:
                f.write(label)

            time.sleep(0.05)  # 20 FPS
            # time.sleep(0.5) # 2 FPS
    except KeyboardInterrupt:
        logging.info("データ収集を終了します。")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
