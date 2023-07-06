import cv2
import numpy as np
import sys
from modules import *

def PageCut(input_img):
    pageImg = []
    if input_img.shape[1] > input_img.shape[0]:  # 縦 < 横の場合: 見開きだと判断し真ん中で切断
        cut_img_left = input_img[:, :input_img.shape[1]//2]  # 右ページ
        cut_img_right = input_img[:, input_img.shape[1]//2:]  # 左ページ
        pageImg.append(cut_img_right)
        pageImg.append(cut_img_left)
    else:  # 縦 > 横の場合: 単一ページ画像だと判断しそのまま保存
        pageImg.append(input_img)
    return pageImg

def extractSpeechBalloon(fukidashi_contours,hierarchy2,gaussian_img):
    for i in range(len(fukidashi_contours)):
        area = cv2.contourArea(fukidashi_contours[i])
        length = cv2.arcLength(fukidashi_contours[i], True)
        en = 0.0
        if gaussian_img.shape[0] * gaussian_img.shape[1] * 0.005 <= area and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.05:
            en = 4.0 * np.pi * area / (length * length)
        if en > 0.4:
            cv2.drawContours(gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1)

    return gaussian_img

def main():
    # コマンドライン引数からディレクトリ名を取得
    args = sys.argv
    if len(args) != 2:
        print('Usage: python koma.py [directory]')
        sys.exit()
    directory = args[1]
    # ディレクトリから画像リストを取得
    img_list = get_imgList_form_dir(directory)

    # 画像リストから画像を1枚ずつ読み込み
    for img in img_list:
        # 画像が見開きだったら分割
        pageImg = PageCut(img)
        # 分割した画像を1枚ずつ処理
        for i in range(len(pageImg)):
            img = pageImg[i]
            # 画像がカラーの場合グレースケールに変換
            if len(img.shape) == 3:
                color_img = img
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)







if __name__ == '__main__':
    main()
