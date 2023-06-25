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
    lines_senbun = np.zeros_like(src_img)
    result_senbub = np.zeros_like(src_img)
    # pylsdのやつ
    for line in lines:
        x1, y1, x2, y2 = map(int,line[:4])
        if (x2-x1)**2 + (y2-y1)**2 > 9000:# 今のところ9000が最適
        # 赤線を引く
            cv2.line(lines_senbun, (x1,y1), (x2,y2), (255,255,255), 3)

    for i in range(len(lines)):
        x1, y1, x2, y2 = map(int,lines[i][:4])
        if (x2-x1)**2 + (y2-y1)**2 > 15000:
            pt1 = (int(lines[i][0]), int(lines[i][1]))
            pt2 = (int(lines[i][2]), int(lines[i][3]))

            # 線分の傾きと切片を計算します。
            if pt2[0] - pt1[0] != 0:  # 垂直な線分を除きます。
                slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
                intercept = pt1[1] - slope * pt1[0]

                # 画像の左右の端でのy座標を計算します。
                y_start = int(slope * 0 + intercept)
                y_end = int(slope * lines_img.shape[1] + intercept)
                cv2.line(lines_img, (0, y_start), (lines_img.shape[1], y_end), (255, 255, 255), 3)
            #     y2 = src_img.shape[0]
            #     x2 = int((y2 - b) / a)
            #     cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            #     cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # else:
            #     cv2.line(lines_img, (x1, 0), (x1, src_img.shape[0]), (255, 255, 255), 3)
            #     cv2.line(result_img, (x1, 0), (x1, src_img.shape[0]), (0, 0, 255), 3)
        else:
            continue


    and_img = cv2.bitwise_and(lines_img,inverse_bin_img)
    and_senbun = cv2.bitwise_and(lines_senbun,inverse_bin_img)
    
    # # 画像端に直線を描画する　最後にやる
    # cv2.line(and_img, (0, 0), (width-1, 0), (255,255,255), 3)  # 上辺
    # cv2.line(and_img, (0, height-1), (width-1, height-1), (255,255,255), 3)  # 下辺
    # cv2.line(and_img, (0, 0), (0, height-1), (255,255,255), 3)  # 左辺
    # cv2.line(and_img, (width-1, 0), (width-1, height-1), (255,255,255), 3)  # 右辺

    # cv2.line(and_senbun, (0, 0), (width-1, 0), (255,255,255), 3)  # 上辺
    # cv2.line(and_senbun, (0, height-1), (width-1, height-1), (255,255,255), 3)  # 下辺
    # cv2.line(and_senbun, (0, 0), (0, height-1), (255,255,255), 3)  # 左辺
    # cv2.line(and_senbun, (width-1, 0), (width-1, height-1), (255,255,255), 3)  # 右辺

    # # 直線検出
    # lines_s = cv2.HoughLines(lines_senbun, 1, np.pi / 180, 200)
    # # 直線を描画
    # for i in range(len(lines_s)):
    #     rho = lines[i][0]
    #     theta = lines[i][1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))

    #     cv2.line(result_senbub, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 膨張・収縮処理
    # kernel = np.ones((3,3),np.uint8)
    # and_senbun = cv2.dilate(and_senbun,kernel,iterations = 4)
    # and_senbun = cv2.erode(and_senbun,kernel,iterations = 3)

    # and_senbunに輪郭抽出
    contours = []
    hierarchy = []
    contours,hierarchy = cv2.findContours(and_senbun, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    # 極端に小さい輪郭を削除
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # 輪郭を描画
    result_senbub = cv2.drawContours(result_senbub, contours, -1, (255, 255, 255), 3)

    #ここにさらに直線検出
    lines_s = cv2.HoughLines(and_senbun, 1, np.pi / 180, 250)
    # 直線を描画
    for i in range(len(lines_s)):
        rho = lines_s[i][0][0]
        theta = lines_s[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(result_senbub, (x1, y1), (x2, y2), (255, 255, 255), 2)

    result_senbub = cv2.bitwise_and(result_senbub,inverse_bin_img)

    cv2.imshow('lines_img', lines_img)
    cv2.imwrite('./output/0604/lines_img.jpg', lines_img)
    cv2.imshow('lines_senbun', lines_senbun)
    cv2.imwrite('./output/0604/lines_senbun.jpg', lines_senbun)
    cv2.imshow('gaussian_img', inverse_bin_img)
    cv2.imshow('and_img', and_img)
    cv2.imshow('result_senbub', result_senbub)
    cv2.imwrite('./output/0604/result_senbub.jpg', result_senbub)
    cv2.imwrite('./output/0604/and_img.jpg', and_img)
    cv2.imshow('and_senbun',and_senbun)
    cv2.imwrite('./output/0604/and_senbun.jpg', and_senbun)
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
