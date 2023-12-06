import cv2
import numpy as np
from modules import *


def main():
    input = cv2.imread("./../Manga109_released_2021_12_30/images/AosugiruHaru/010.jpg")
    input = PageCut(input)[0]
    # 画像がカラーの場合グレースケールに変換
    if len(input.shape) == 3:
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    cv2.imshow("input", input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    crop_pixel = 10
    cropped_image = input[:, crop_pixel:-crop_pixel]
    cropped_image = cropped_image[crop_pixel:-crop_pixel, :]
    bin_img = cv2.threshold(cropped_image, 230, 255, cv2.THRESH_BINARY)[1]
    if is_all_black_image(bin_img, 10):
        print("all black")
        return
    # 画像端から5ピクセルの範囲の黒画素以外の画素値を取得
    page_type = 1
    top, left, right, bottom = 1, 1, 1, 1
    # top
    for y in range(5):
        for x in range(bin_img.shape[1]):
            if bin_img[y, x] != 0:
                top = 0
                break
    # left
    for y in range(bin_img.shape[0]):
        for x in range(5):
            if bin_img[y, x] != 0:
                left = 0
                break
    # right
    for y in range(bin_img.shape[0]):
        for x in range(bin_img.shape[1] - 5, bin_img.shape[1]):
            if bin_img[y, x] != 0:
                right = 0
                break
    # bottom
    for y in range(bin_img.shape[0] - 5, bin_img.shape[0]):
        for x in range(bin_img.shape[1]):
            if bin_img[y, x] != 0:
                bottom = 0
                break
    if top == 0 and left == 0 and right == 0 and bottom == 0:
        page_type = 0
        print(top, left, right, bottom)
    if page_type == 1:
        print("page_type: 1")
    cv2.imshow("cropped_image", bin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
