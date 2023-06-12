from modules import *
import cv2
import numpy as np
from pylsd.lsd import lsd

def main():
    input_img = cv2.imread('./input/Belmondo/004.jpg')
    twoPage = PageCut(input_img)
    src_img = twoPage[1]
    if src_img is None:
        print("Not open:", src_img)
        return
    # srcがカラーの場合グレースケールに変換
    if len(src_img.shape) == 3:
        color_img = src_img
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    else:
        color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

    height, width = src_img.shape[:2]

    gausForLsd_img = cv2.GaussianBlur(src_img, (5, 5), 0)

    binForSpeechBalloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1]
    
    # 膨張収縮
    kernel = np.ones((3, 3), np.uint8)
    binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)
    binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)
    

    hierarchy2 = []  # cv::Vec4i のリスト
    hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

    # 輪郭抽出
    hukidashi_contours, hierarchy2 = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

    # 吹き出し検出　塗りつぶし
    gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)

    inverse_bin_img = cv2.threshold(gaussian_img,100,255,cv2.THRESH_BINARY_INV)[1]

    # lsdで直線検出
    lines = lsd(gausForLsd_img)

    result_img = color_img.copy()
    lines_img = np.zeros_like(src_img)
    # pylsdのやつ
    for line in lines:
        x1, y1, x2, y2 = map(int,line[:4])
        if (x2-x1)**2 + (y2-y1)**2 > 9000:# 今のところ9000が最適
        # 赤線を引く
            # result_img = cv2.line(result_img, (x1,y1), (x2,y2), (0,0,255), 3)
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            # 線分の傾きと切片を計算
            if x2 - x1 != 0:
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
            else:
                a = 0
                b = x1

            if a != 0:
                y1 = 0
                try:
                    x1 = int((y1 - b) / a)
                except ZeroDivisionError:
                    print('zerodivision',a)
                y2 = src_img.shape[0]
                x2 = int((y2 - b) / a)
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                cv2.line(lines_img, (x1, 0), (x1, src_img.shape[0]), (255, 255, 255), 3)
                cv2.line(result_img, (x1, 0), (x1, src_img.shape[0]), (0, 0, 255), 3)
        else:
            continue

    and_img = cv2.bitwise_and(lines_img,inverse_bin_img)
    
    # 画像端に直線を描画する
    cv2.line(and_img, (0, 0), (width-1, 0), (255,255,255), 3)  # 上辺
    cv2.line(and_img, (0, height-1), (width-1, height-1), (255,255,255), 3)  # 下辺
    cv2.line(and_img, (0, 0), (0, height-1), (255,255,255), 3)  # 左辺
    cv2.line(and_img, (width-1, 0), (width-1, height-1), (255,255,255), 3)  # 右辺


    cv2.imshow('result_img', result_img)
    cv2.imshow('lines_img', lines_img)
    cv2.imshow('gaussian_img', inverse_bin_img)
    cv2.imshow('and_img', and_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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

if __name__ == '__main__':
    main()
