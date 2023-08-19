import cv2
import numpy as np
from modules import *
from pylsd.lsd import lsd
import re


def main():
    # フォルダから画像を読み込み
    folder = "./input/Belmondo/"
    img_files = get_imgs_from_folder_sorted(folder)
    # print(img_files)
    for img_file in img_files:
        if img_file != "012.jpg":
            continue
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

            binForSpeechBalloon_img = cv2.threshold(
                src_img, 230, 255, cv2.THRESH_BINARY
            )[1]

            # 膨張収縮
            kernel = np.ones((3, 3), np.uint8)
            binForSpeechBalloon_img = cv2.erode(
                binForSpeechBalloon_img, kernel, (-1, -1), iterations=1
            )
            binForSpeechBalloon_img = cv2.dilate(
                binForSpeechBalloon_img, kernel, (-1, -1), iterations=1
            )

            hierarchy2 = []  # cv::Vec4i のリスト
            hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

            # 輪郭抽出
            hukidashi_contours, hierarchy2 = cv2.findContours(
                binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

            # 吹き出し検出　塗りつぶし
            gaussian_img = extractSpeechBalloon(
                hukidashi_contours, hierarchy2, gaussian_img
            )

            inverse_bin_img = cv2.threshold(
                gaussian_img, 100, 255, cv2.THRESH_BINARY_INV
            )[1]

            senbun = np.zeros_like(src_img)
            lines = lsd(gausForLsd_img)
            for line in lines:
                x1, y1, x2, y2 = map(int, line[:4])
                if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 9000:  # 今のところ9000が最適
                    # 線を引く
                    cv2.line(senbun, (x1, y1), (x2, y2), (255, 255, 255), 3)

            cv2.imwrite(
                "./output/0818" + img_file + "_" + str(pagenum) + "_lsd1senbun.jpg",
                senbun,
            )

            hough_lines = cv2.HoughLines(senbun, 1, np.pi / 180, 250)
            if hough_lines is None:
                continue
            hough_img = np.zeros_like(src_img)
            for rho, theta in hough_lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))
                cv2.line(hough_img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.imshow("hough", hough_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 二値画像に膨張収縮
            kernel = np.ones((3, 3), np.uint8)
            forAndInverseImg = cv2.dilate(inverse_bin_img, kernel, iterations=2)
            forAndInverseImg = cv2.erode(forAndInverseImg, kernel, iterations=2)

            and_img = np.zeros_like(src_img)
            and_img = cv2.bitwise_and(forAndInverseImg, hough_img)

            # 画像端に直線を引く
            for i in range(height):
                and_img[i][0] = 255
                and_img[i][width - 1] = 255
            for i in range(width):
                and_img[0][i] = 255
                and_img[height - 1][i] = 255
            cv2.imshow("and", and_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 輪郭抽出
            contours = []
            hierarchy = []
            contours, hierarchy = cv2.findContours(
                and_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            # 極端に小さい輪郭を削除
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
            # 画像サイズの輪郭を削除
            contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) < (width * height) / 2
            ]
            for contour in contours:
                bounding_rect = cv2.boundingRect(contour)
                cv2.rectangle(
                    result_img,
                    (bounding_rect[0], bounding_rect[1]),
                    (
                        bounding_rect[0] + bounding_rect[2],
                        bounding_rect[1] + bounding_rect[3],
                    ),
                    (255, 255, 255),
                    3,
                )

            cv2.imshow("result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 輪郭を描画
            count = 0
            for contour in contours:
                count += 1
                cv2.drawContours(result_img, contour, -1, (255, 255, 255), 3)
            cv2.imshow("result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def extractSpeechBalloon(fukidashi_contours, hierarchy2, gaussian_img):
    for i in range(len(fukidashi_contours)):
        area = cv2.contourArea(fukidashi_contours[i])
        length = cv2.arcLength(fukidashi_contours[i], True)
        en = 0.0
        if (
            gaussian_img.shape[0] * gaussian_img.shape[1] * 0.005 <= area
            and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.05
        ):
            en = 4.0 * np.pi * area / (length * length)
        if en > 0.4:
            cv2.drawContours(
                gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1
            )

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


if __name__ == "__main__":
    main()
