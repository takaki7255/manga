import cv2
import numpy as np
from modules import *
from pylsd.lsd import lsd

def main():
    input = cv2.imread('./input/Belmondo/005.jpg')
    cut_images = PageCut(input)
    src_img = cut_images[0]
    if src_img is None:
        print("Not open:", src_img)
        return
    # srcがカラーの場合グレースケールに変換
    if len(src_img.shape) == 3:
        color_img = src_img
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    else:
        color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

    # 使う画像の確保
    crop_imgs = [] # 切り出したコマ画像を格納するリスト
    crop_img = color_img.copy() # 切り出したコマ画像
    result_img = np.zeros_like(src_img) # 結果画像

    height, width = src_img.shape[:2] # 画像の高さと幅を取得
    
    # ここから吹き出し検出，二値化
    binForSpeechBalloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1] # 吹き出し抽出の二値化

    # 膨張収縮
    kernel = np.ones((3, 3), np.uint8)
    binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)
    binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, kernel,(-1,-1), iterations = 1)

    hukidashi_contours, hierarchy = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0) #  ガウシアンフィルタ

    # 吹き出し検出　塗りつぶし
    gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy, gaussian_img)

    inverse_bin_img = cv2.threshold(gaussian_img,100,255,cv2.THRESH_BINARY_INV)[1] # 二値化
    # ここまで吹き出し検出，二値化

    # ここからLSD
    gausForLsd_img = cv2.GaussianBlur(src_img, (5, 5), 0) # LSDのためにガウシアンフィルタ
    lines = lsd(gausForLsd_img) # LSD

    # pylsdのやつ
    senbun_img = np.zeros_like(src_img)
    for line in lines:
        x1, y1, x2, y2 = map(int,line[:4])
        if (x2-x1)**2 + (y2-y1)**2 > 9000:# 今のところ9000が最適
        # 赤線を引く
            cv2.line(senbun_img, (x1,y1), (x2,y2), (255,255,255), 3)
    # ここまでLSD

    and_img = cv2.bitwise_and(inverse_bin_img, senbun_img) # LSDと吹き出しのand
    cv2.imshow("and_img", and_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
