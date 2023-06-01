from modules import *
import cv2
import numpy as np

class Points:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def main():
    input_img = cv2.imread('./input/006.jpg')
    twoPage = PageCut(input_img)
    print(twoPage)
    for i in range(len(twoPage)):
        cv2.imwrite('./output/cut/006_{}.jpg'.format(i), twoPage[i])
        page_type = get_page_type(twoPage[i])
        print(page_type)


if __name__ == '__main__':
    main()
