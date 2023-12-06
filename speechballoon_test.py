import cv2
import numpy as np
import os

from modules import *


def main():
    input = cv2.imread("./../src/frame_1.jpg")
    if input is None:
        print("Not open:", input)
        return
    # srcがカラーの場合グレースケールに変換
    if len(input.shape) == 3:
        color_img = input
        src_img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    else:
        color_img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    speechballoon_imgs = []
    speechballoon_imgs_bin = []

    imageSize = src_img.shape[0:2]

    alpha_img = np.zeros(imageSize, dtype=np.uint8)
    gray_img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)[1]

    # オープニング
    bin_img = cv2.erode(bin_img, None, iterations=1)
    bin_img = cv2.dilate(bin_img, None, iterations=1)

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    back_150_img = np.full(imageSize, 150, dtype=np.uint8)
    mask_img = np.full(imageSize, 255, dtype=np.uint8)

    alpha_img = cv2.cvtColor(alpha_img, cv2.COLOR_GRAY2BGRA)

    alpha_img[:, :, 3] = 0

    # 輪郭の描画
    for i in range(0, len(contours)):
        mask_result_1_img = gray_img.copy()
        panel_area = src_img.shape[0] * src_img.shape[1]
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        en = (4 * math.pi * area) / (length * length)

        B, W, G = 0, 0, 0
        TH = 255 / 3

        # 極端に小さい or 大きい or 吹き出しぽくない形を除外
        if area < panel_area / 100 or area > panel_area / 2 or en < 0.7:
            continue
        bounding_box = cv2.boundingRect(contours[i])
        bounding_box_size = bounding_box[2] * bounding_box[3]

        # 吹き出し画像の中心座標
        center_x = bounding_box[0] + bounding_box[2] / 2
        center_y = bounding_box[1] + bounding_box[3] / 2

        # マスク画像生成
        cv2.drawContours(mask_img, contours, i, (0, 0, 0), 4, cv2.LINE_AA, hierarchy)
        cv2.drawContours(mask_img, contours, i, (0, 0, 0), -1, cv2.LINE_AA, hierarchy)

        # マスク処理で背景をグレーに変換
        mask_result_1_img = cv2.bitwise_and(back_150_img, back_150_img, mask=mask_img)

        # 吹き出し部分だけを抽出
        resize_img = mask_result_1_img[
            bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0] : bounding_box[0] + bounding_box[2]
        ]

        cv2.imshow("resize_img", resize_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # B,W,Gのカウント
        for y in range(resize_img.shape[0]):
            for x in range(resize_img.shape[1]):
                s = resize_img[y, x]
                if s != 150:
                    if s > 255 - TH:
                        W += 1
                    elif s < TH:
                        B += 1
                    else:
                        G += 1
        print("B:", B, "W:", W, "G:", G)

        # 矩形度
        maxrect = bounding_box_size * 0.95

        # 吹き出しの白黒比率判定（吹き出しらしい比率）
        if B / W > 0.01 and B / W < 0.7:
            if B >= 10:
                # 吹き出しの形判定
                setType = 0
                if area >= maxrect:  # 矩形
                    setType = 1
                elif en >= 0.7:  # 円
                    setType = 0
                else:
                    setType = 2

        mask_result_2_img = color_img.copy()
        # 4チャンネル化
        mask_result_2_img = cv2.cvtColor(mask_result_2_img, cv2.COLOR_BGR2BGRA)

        # マスク処理で吹き出し以外を透過
        alpha_img = cv2.copyTo(mask_result_2_img, mask_img)

        if bounding_box_size < panel_area * 0.9:
            # 吹き出し部分だけ切り取る
            resize_alpha_img = mask_result_2_img[
                bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0] : bounding_box[0] + bounding_box[2]
            ]
            speechballoon_imgs.append(resize_alpha_img)
            cv2.imshow("resize_alpha_img", resize_alpha_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 二値処理用
            mask_result_bin_img = mask_result_2_img.copy()
            mask_result_bin_img = cv2.threshold(mask_result_bin_img, 150, 255, cv2.THRESH_BINARY)[1]
            resize_alpha_img_bin = mask_result_bin_img[
                bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0] : bounding_box[0] + bounding_box[2]
            ]
            speechballoon_imgs_bin.append(resize_alpha_img_bin)

    # 吹き出し誤抽出除去手法
    # balloon_px_th = 5  # 抽出した吹き出し画像で何ピクセル内側までを吹き出しとするか
    # unspeechballoon_flag = False
    # black_count = 0
    # white_count = 0
    # index = 0

    # for speechballoon_img in speechballoon_imgs_bin:
    #     if speechballoon_img is None:
    #         continue
    #     unspeechballoon_flag = False
    #     black_count = 0
    #     for y in range(speechballoon_img.shape[0]):
    #         for x in range(speechballoon_img.shape[1]):
    #             if speechballoon_img[y, x] == 0:
    #                 black_count += 1
    #             else:
    #                 white_count += 1
    #     if black_count < white_count:
    #         unspeechballoon_flag = True
    #     else:
    #         unspeechballoon_flag = False
    #     if unspeechballoon_flag:
    #         speechballoon_imgs.pop(index)
    #         speechballoon_imgs_bin.pop(index)
    #     index += 1
    #     black_count = 0
    #     white_count = 0


if __name__ == "__main__":
    main()
