import cv2
import numpy as np
from modules import *
from pylsd.lsd import lsd
import re


def main():
    folder = "./black_img/"
    imgs = get_imgs_from_folder_sorted(folder)
    for img in imgs:
        # cv2.imshow("img", cv2.imread(folder + img))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img)
        # if img != "belmond044のコピー.jpg_1.jpg":
        #     continue
        src_img = cv2.imread(folder + img)
        if len(src_img.shape) == 3:
            color_img = src_img
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        else:
            color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        result_img = np.zeros_like(src_img)
        height, width = src_img.shape[:2]

        gauss_for_lsd_img = cv2.GaussianBlur(src_img, (5, 5), 0)

        senbun = np.zeros_like(src_img)
        lines = lsd(gauss_for_lsd_img)
        for line in lines:
            x1, y1, x2, y2 = map(int, line[:4])
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
                cv2.line(senbun, (x1, y1), (x2, y2), (255, 255, 255), 3)
        # cv2.imshow("senbun", senbun)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 確率的ハフ変換
        lines = cv2.HoughLinesP(senbun, rho=1, theta=np.pi / 180, threshold=100, minLineLength=300, maxLineGap=10)
        # 直線の描画
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(senbun, (x1, y1), (x2, y2), (255, 255, 255), 3)

        bin_for_speech_balloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1]

        # 膨張収縮
        kernel = np.ones((3, 3), np.uint8)
        bin_for_speech_balloon_img = cv2.erode(bin_for_speech_balloon_img, kernel, (-1, -1), iterations=1)
        bin_for_speech_balloon_img = cv2.dilate(bin_for_speech_balloon_img, kernel, (-1, -1), iterations=1)

        hierarchy2 = []
        hukidashi_contours = []
        # 輪郭抽出
        hukidashi_contours, hierarchy2 = cv2.findContours(
            bin_for_speech_balloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)
        gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)

        inverse_bin_img = cv2.threshold(gaussian_img, 150, 255, cv2.THRESH_BINARY_INV)[1]
        # 膨張処理
        kernel = np.ones((3, 3), np.uint8)
        inverse_bin_img = cv2.dilate(inverse_bin_img, kernel, (-1, -1), iterations=3)
        # cv2.imshow("inverse_bin_img", inverse_bin_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        and_img = cv2.bitwise_and(senbun, inverse_bin_img)
        # cv2.imshow("and_img", and_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 輪郭抽出
        contours = []
        hierarchy = []
        contours, hierarchy = cv2.findContours(senbun, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭の描画
        # cv2.drawContours(color_img, contours, -1, (0, 0, 255), 3)
        # cv2.imshow("contours", color_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # バウンディングボックスのリストと面積判定
        bounding_boxes = []
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            min_area = width * height * 0.048
            max_area = width * height * 0.5
            w, h = bounding_rect[2], bounding_rect[3]
            area = w * h
            if area > max_area:
                continue
            if area < min_area:
                continue
            bounding_boxes.append(bounding_rect)

        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("result", color_img)
        cv2.imshow("senbun", senbun)
        cv2.moveWindow("senbun", 450, 0)
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
            cv2.drawContours(gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1)

    return gaussian_img


if __name__ == "__main__":
    main()
