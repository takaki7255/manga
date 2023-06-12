from modules import *
import cv2
import numpy as np
from pylsd.lsd import lsd

def main():
    input_img = cv2.imread('./output/0604/004_inverse_bin_img.jpg',cv2.IMREAD_GRAYSCALE)
    bin_img = cv2.imread('./output/0604/004_inverse_bin_img.jpg',cv2.IMREAD_GRAYSCALE)
    pylsd_img = cv2.imread('./input/Belmondo/004.jpg',cv2.IMREAD_GRAYSCALE)
    cut_img = PageCut(pylsd_img)[1]
    pygau_img = cv2.GaussianBlur(cut_img,(5,5),0)
    gaussian_img = cv2.GaussianBlur(input_img,(5,5),0)

    edge = cv2.Canny(gaussian_img, 50, 110, apertureSize=3)
    
    
    lines = cv2.HoughLines(edge, 1, np.pi / 180.0, 50)
    # print(lines)

    #pylsdを使ってみる
    linesL = lsd(pygau_img)


    result_img = np.zeros_like(input_img)
    resultL1_img = np.zeros_like(input_img) # 黒に白線引くやつ
    resultL_img = cut_img.copy()
    resultL_img = cv2.cvtColor(resultL_img, cv2.COLOR_GRAY2BGR) # 画像に対して赤直線引くやつ

    # pylsdのやつ
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        # resultL_img = cv2.line(resultL_img, (x1,y1), (x2,y2), (255,255,255), 3)
        if (x2-x1)**2 + (y2-y1)**2 > 9000:# 今のところ9000が最適
        # 赤線を引く
            resultL_img = cv2.line(resultL_img, (x1,y1), (x2,y2), (0,0,255), 3)
            resultL1_img = cv2.line(resultL1_img, (x1,y1), (x2,y2), (255,255,255), 3)

    
    # # hough変換のやつ最大で100本の直線を描画する
    # for i in range(min(len(lines), 100)):
    #     line = lines[i]
    #     rho, theta = line[0]
    #     theta_degrees = np.degrees(theta)
    #     if (theta_degrees < 70 or theta_degrees > 110):
    #         continue
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 2000 * (-b))
    #     y1 = int(y0 + 2000 * (a))
    #     x2 = int(x0 - 2000 * (-b))
    #     y2 = int(y0 - 2000 * (a))

    #     cv2.line(result_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # cv2.imshow('edge', edge)
    # cv2.imshow('Hough Lines', result_img)
    cv2.imshow('pylsd', resultL_img)
    cv2.imshow('pylsd1', resultL1_img)
    # cv2.imwrite('./output/0604/004_0_pylsd.jpg', resultL_img)
    cv2.imwrite('./output/0604/004_0_pylsd1.jpg', resultL1_img)
    cv2.imwrite('./output/0604/004_0_hough_lines.jpg', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    and_img = cv2.bitwise_and(bin_img, result_img)
    cv2.imshow('and_img', and_img)
    cv2.imwrite('./output/0604/004_0_and_img.jpg', and_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
