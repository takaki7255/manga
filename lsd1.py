import cv2
import numpy as np
from modules import *
from pylsd.lsd import lsd
import re


def main():
    # フォルダから画像を読み込み
    folder = "./black_img/"
    img_files = get_imgs_from_folder_sorted(folder)
    # print(img_files)
    for img_file in img_files:
        # if img_file != "buraritessenmonogatari088のコピー.jpg_1.jpg":
        #     continue
        input_img = cv2.imread(folder + img_file)
        # print(input_img)
        twoPage = PageCut(input_img)
        for pagenum in range(len(twoPage)):
            print(img_file + "_" + str(pagenum))
            src_img = twoPage[pagenum]
            # 画像が黒画像の場合はスキップ
            if is_black_image(src_img, 10):
                continue
            if src_img is None:
                print("Not open:", src_img)
                return
            # srcがカラーの場合グレースケールに変換
            if len(src_img.shape) == 3:
                color_img = src_img
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            else:
                color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

            crop_imgs = []
            # color_imgをcrop_imgにコピー
            crop_img = color_img.copy()
            result_img = np.zeros_like(src_img)

            height, width = src_img.shape[:2]

            gausForLsd_img = cv2.GaussianBlur(src_img, (5, 5), 0)

            binForSpeechBalloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1]

            # 膨張収縮
            kernel = np.ones((3, 3), np.uint8)
            binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, kernel, (-1, -1), iterations=1)
            binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, kernel, (-1, -1), iterations=1)

            hierarchy2 = []  # cv::Vec4i のリスト
            hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

            # 輪郭抽出
            hukidashi_contours, hierarchy2 = cv2.findContours(
                binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

            # 吹き出し検出　塗りつぶし
            gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)

            inverse_bin_img = cv2.threshold(gaussian_img, 150, 255, cv2.THRESH_BINARY_INV)[1]
            cv2.imshow("inverse_bin_img", inverse_bin_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            senbun = np.zeros_like(src_img)
            lines = lsd(gausForLsd_img)
            for line in lines:
                x1, y1, x2, y2 = map(int, line[:4])
                if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 2000:  # 今のところ9000が最適
                    # 線を引く
                    cv2.line(senbun, (x1, y1), (x2, y2), (255, 255, 255), 3)
            # cv2.imshow("senbun", senbun)
            # cv2.imshow("input", src_img)
            # cv2.moveWindow("input", 500, 0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            hough_lines = cv2.HoughLines(senbun, 1, np.pi / 180, 300)
            if hough_lines is None:
                continue
            hough_img = np.zeros_like(src_img)
            for i in range(len(hough_lines)):
                rho = hough_lines[i][0][0]
                theta = hough_lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))
                cv2.line(hough_img, (x1, y1), (x2, y2), (255, 255, 255), 5)
            # cv2.imshow("hough", hough_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 二値画像に膨張収縮
            kernel = np.ones((3, 3), np.uint8)
            forAndInverseImg = cv2.dilate(inverse_bin_img, kernel, iterations=2)
            # forAndInverseImg = cv2.erode(forAndInverseImg, kernel, iterations=2)
            # 画像の黒と白を入れ替える
            # forAndInverseImg = cv2.bitwise_not(forAndInverseImg)
            # cv2.imshow("forAndInverseImg", forAndInverseImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            and_img = np.zeros_like(src_img)
            and_img = cv2.bitwise_and(forAndInverseImg, hough_img)
            # クロージング
            kernel = np.ones((3, 3), np.uint8)
            and_img = cv2.morphologyEx(and_img, cv2.MORPH_CLOSE, kernel)
            # 白と黒を入れ替える
            and_inverse_img = cv2.bitwise_not(and_img)
            # クロージング
            kernel = np.ones((3, 3), np.uint8)
            and_inverse_img = cv2.morphologyEx(and_inverse_img, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow("and", and_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # # 画像端に直線を引く
            # for i in range(height):
            #     and_img[i][0] = 255
            #     and_img[i][width - 1] = 255
            # for i in range(width):
            #     and_img[0][i] = 255
            #     and_img[height - 1][i] = 255
            # cv2.imshow("and", and_img)

            # 輪郭抽出
            contours = []
            hierarchy = []
            contours, hierarchy = cv2.findContours(forAndInverseImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
                # 輪郭を描画
                cv2.drawContours(crop_img, contour, -1, (0, 255, 0), 3)
            # cv2.imshow("contour", crop_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for box in bounding_boxes:
                x, y, w, h = box
                cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("result", color_img)
            cv2.imshow("input", src_img)
            cv2.imshow("and_img", and_img)
            cv2.imshow("senbun", senbun)
            cv2.imshow("hough", hough_img)
            cv2.imshow("forAndInverseImg", forAndInverseImg)
            cv2.moveWindow("forAndInverseImg", 0, 500)
            cv2.moveWindow("hough", 750, 500)
            cv2.moveWindow("senbun", 250, 500)
            cv2.moveWindow("input", 500, 0)
            cv2.moveWindow("and_img", 1000, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite(
            #     "./output/0821/" + img_file + "_" + str(pagenum) + ".jpg", result_img
            # )

            # contours2 = []
            # hierarchy2 = []
            # result_img2 = np.zeros_like(src_img)
            # contours2, hierarchy2 = cv2.findContours(
            #     result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            # )
            # for contour2 in contours2:
            #     cv2.drawContours(result_img2, contour2, -1, (255, 255, 255), 3)
            # cv2.imshow("result2", result_img2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


def get_imgs_from_folder_sorted(folder):
    image_files = []
    all_files = os.listdir(folder)
    for file in all_files:
        if os.path.isfile(os.path.join(folder, file)):
            extension = os.path.splitext(file)[1].lower()
            if extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                image_files.append(file)
    return sorted(image_files, key=natural_keys)


def is_black_image(img, threshold=0):
    if len(img.shape) == 3:  # もしカラー画像なら、グレースケールに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return np.all(avg_color <= threshold)


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # 重複していない場合、IoUは0になります
    if union_area == 0:
        return 0

    return intersection_area / union_area


def is_Overlap(box1, bounding_boxes):
    for box2 in bounding_boxes:
        if compute_iou(box1, box2) != 0:
            return True
    return False


if __name__ == "__main__":
    main()
