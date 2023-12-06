import cv2
import numpy as np
import os

from modules import *


def main():
    imgs = get_imgs_from_folder_sorted("./black_img")
    for img in imgs:
        input = cv2.imread("./black_img/" + img)
        cut_imgs = PageCut(input)
        count = 0
        for cut_img in cut_imgs:
            # 番号つけて保存
            cv2.imwrite("./black_img/" + img + "_" + str(count) + ".jpg", cut_img)
            count += 1


if __name__ == "__main__":
    main()
