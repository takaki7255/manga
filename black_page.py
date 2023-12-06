import cv2
import numpy as np
from modules import *
import os


def main():
    folder = "./../Manga109_released_2021_12_30/images/"
    # フォルダー内のフォルダー名を取得
    folder_list = os.listdir(folder)
    # print(folder_list)
    for folder_name in folder_list:
        # print(folder_name)
        # .DS_Storeをスキップ
        if folder_name == ".DS_Store":
            continue
        # フォルダー内の画像を取得
        img_files = get_imgs_from_folder_sorted(folder + folder_name)
        for img_file in img_files:
            # print(img_file)
            img = cv2.imread(folder + folder_name + "/" + img_file)
            two_img = PageCut(img)
            for page_img in two_img:
                crop_pixel = 10
                cropped_image = page_img[:, crop_pixel:-crop_pixel]
                cropped_image = cropped_image[crop_pixel:-crop_pixel, :]
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                if is_all_black_image(cropped_image, 10):
                    continue
                bin_img = cv2.threshold(cropped_image, 230, 255, cv2.THRESH_BINARY)[1]
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
                if page_type == 1:
                    # with open("black_page.txt", "a") as f:
                    #     f.write(folder_name + "/" + img_file + "\n")
                    # 画像を保存
                    if not os.path.exists("./black_img/" + folder_name):
                        os.makedirs("./black_img/" + folder_name)
                    cv2.imwrite("./black_img/" + folder_name + "/" + img_file, page_img)


if __name__ == "__main__":
    main()
