import cv2
import numpy as np
import os
import math

from modules import *

def main():
    input_folder_path = "./input"
    output_folder_path = "./output"
    # inputからjpgファイルを取得
    files = os.listdir(input_folder_path)
    files_jpg = [f for f in files if os.path.isfile(os.path.join(input_folder_path, f)) and f.endswith(".jpg")]
    # ファイルの名前順にソート
    files_jpg.sort()

    panel = Framedetect()

    twoPageImg = []

    src_panel_img = [] # １ページ画像
    src_panel_imgs = [] # 全てのページ画像
    page_type = [False, False] # 0-white 1-black # ページタイプ
    speech_balloon_img = [] # 吹き出し画像
    speech_balloon_imgs = [] # 全ての吹き出し画像
    speechballoon_max = 99


    # 画像一枚ずつ処理
    for file in files_jpg:
        img = cv2.imread(os.path.join(input_folder_path, file))
        if img is None:
            print("Not open:", file)
            continue
        print("Open:", file)
        pageImg = PageCut(img)
        for i, page in enumerate(pageImg):
            twoPageImg.append(page)

        # page_typeのの取得
        for i in twoPageImg:
            count = 0
            page_type[count] = get_page_type(i)
            print('ページタイプまできたよ')
            count += 1

        # コマ分割
        for i in range(len(twoPageImg)):
            print(len(twoPageImg))
            print(len(page_type))
            # if page_type[i] == 1:
            #     print('コマ分割前でcontinue')
            #     continue
            src_panel_img = panel.frame_detect(twoPageImg[i])
            print(src_panel_img)
            print('コマ分割まできたよ')

        for i in range(len(src_panel_img)):
            src_panel_imgs.append(src_panel_img[i])
            print('コマ分割保存まできたよ')

        # 吹き出し分割
        for i in range(len(src_panel_imgs)):
            print('吹き出し分割手前まできたよ')
            speech_balloon_img = detect_speech_balloons(src_panel_imgs[i])
            print('吹き出し分割まできたよ')
            if speech_balloon_img.empty(): continue
            if len(speech_balloon_img) > speechballoon_max: continue
            for j in range(len(speech_balloon_img)):
                speech_balloon_imgs.append(speech_balloon_img[j])
                print('吹き出し分割保存まできたよ')
                cv2.imwrite(os.path.join(output_folder_path, "./speech_balloon_speech_balloon_" + str(j) + ".png"), speech_balloon_img[j])


if __name__ == '__main__':
    main()
