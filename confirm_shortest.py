import cv2
import numpy as np

from modules import *


def main():
    input = cv2.imread("./../Manga109_released_2021_12_30/images/HarukaRefrain/014.jpg")
    # xmin="765" ymin="346" xmax="772" ymax="348"に矩形を描画
    cv2.rectangle(input, (765, 346), (772, 348), (0, 0, 255), 3)
    cv2.imshow("input", input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
