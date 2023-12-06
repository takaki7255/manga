import cv2
import numpy as np
import os
import math
import re

from modules import *


def main():
    # フォルダから画像を読み込み
    folder = "./input/Belmondo/"
    img_files = get_imgs_from_folder_sorted(folder)
    # print(img_files)
    for img_file in img_files:
        if img_file != "004.jpg":
            continue
        input_img = cv2.imread(folder + img_file)
        # print(input_img)
        twoPage = PageCut(input_img)
        for pagenum in range(len(twoPage)):
            print(img_file + "_" + str(pagenum))
            src_img = twoPage[pagenum]
            # 画像が黒画像の場合はスキップ
            if is_black_image(src_img, 10):
                continue
            if src_img is None:
                print("Not open:", src_img)
                return
            # srcがカラーの場合グレースケールに変換
            if len(src_img.shape) == 3:
                color_img = src_img
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            else:
                color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

            crop_imgs = []
            # color_imgをcrop_imgにコピー
            crop_img = color_img.copy()
            result_img = np.zeros_like(src_img)

            height, width = src_img.shape[:2]

            binForSpeechBalloon_img = cv2.threshold(
                src_img, 230, 255, cv2.THRESH_BINARY
            )[1]

            # 膨張収縮
            kernel = np.ones((3, 3), np.uint8)
            binForSpeechBalloon_img = cv2.erode(
                binForSpeechBalloon_img, kernel, (-1, -1), iterations=1
            )
            binForSpeechBalloon_img = cv2.dilate(
                binForSpeechBalloon_img, kernel, (-1, -1), iterations=1
            )

            hierarchy2 = []  # cv::Vec4i のリスト
            hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

            # 輪郭抽出
            hukidashi_contours, hierarchy2 = cv2.findContours(
                binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

            # 吹き出し検出　塗りつぶし
            gaussian_img = extractSpeechBalloon(
                hukidashi_contours, hierarchy2, gaussian_img
            )

            inverse_bin_img = cv2.threshold(
                gaussian_img, 150, 255, cv2.THRESH_BINARY_INV
            )[1]

            canny_img = cv2.Canny(inverse_bin_img, 50, 110, apertureSize=3)

            # 確率的ハフ変換
            lines = cv2.HoughLinesP(
                canny_img,
                rho=1,
                theta=np.pi / 360,
                threshold=50,
                minLineLength=70,
                maxLineGap=10,
            )

            lines_img = np.zeros(src_img.shape, dtype=np.uint8)
            for line in lines:
                # 検出した直線を画像端まで描画する
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.1:
                    continue
                if abs(slope) > 10:
                    continue
                if x1 == x2:
                    continue
                if y1 == y2:
                    continue
                if x1 < 0 or x1 > width or x2 < 0 or x2 > width:
                    continue
                if y1 < 0 or y1 > height or y2 < 0 or y2 > height:
                    continue
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imshow("lines_img", lines_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def extractSpeechBalloon(fukidashi_contours, hierarchy2, gaussian_img):
    for i in range(len(fukidashi_contours)):
        area = cv2.contourArea(fukidashi_contours[i])
        length = cv2.arcLength(fukidashi_contours[i], True)
        en = 0.0
        if (
            gaussian_img.shape[0] * gaussian_img.shape[1] * 0.001 <= area
            and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.05
        ):
            en = 4.0 * np.pi * area / (length * length)
        if en > 0.4:
            cv2.drawContours(
                gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1
            )

    return gaussian_img


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


def get_imgs_from_folder_sorted(folder):
    image_files = []
    all_files = os.listdir(folder)
    for file in all_files:
        if os.path.isfile(os.path.join(folder, file)):
            extension = os.path.splitext(file)[1].lower()
            if extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                image_files.append(file)
    return sorted(image_files, key=natural_keys)


def is_black_image(img, threshold=0):
    if len(img.shape) == 3:  # もしカラー画像なら、グレースケールに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return np.all(avg_color <= threshold)


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # 重複していない場合、IoUは0になります
    if union_area == 0:
        return 0

    return intersection_area / union_area


def is_Overlap(box1, bounding_boxes):
    for box2 in bounding_boxes:
        if compute_iou(box1, box2) != 0:
            return True
    return False


if __name__ == "__main__":
    main()
