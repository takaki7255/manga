import cv2
import numpy as np
import os
import math

from modules import *

def main():
    input_img = cv2.imread('./input/006.jpg')
    twoPage = PageCut(input_img)
    src_img = twoPage[0]
    if src_img is None:
        print("Not open:", src_img)
        return
    # srcがカラーの場合グレースケールに変換
    if len(src_img.shape) == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        color_img = src_img
    else:
        color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

    binForSpeechBalloon_img = cv2.threshold(src_img, 230, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('binForSpeechBalloon_img', binForSpeechBalloon_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 膨張収縮
    kernel = np.ones((3, 3), np.uint8)
    binForSpeechBalloon_img = cv2.dilate(binForSpeechBalloon_img, None, iterations=1)
    binForSpeechBalloon_img = cv2.erode(binForSpeechBalloon_img, None, iterations=1)

    hierarchy2 = []  # cv::Vec4i のリスト
    hukidashi_contours = []  # cv::Point のリストのリスト（輪郭情報）

    # 輪郭抽出
    hukidashi_contours, hierarchy2 = cv2.findContours(binForSpeechBalloon_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    gaussian_img = cv2.GaussianBlur(src_img, (3, 3), 0)

    # 吹き出し検出　塗りつぶし
    gaussian_img = extractSpeechBalloon(hukidashi_contours, hierarchy2, gaussian_img)
    # cv2.imshow('gaussian_img', gaussian_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # inverse_bin_img = cv2.bitwise_not(binForSpeechBalloon_img)
    inverse_bin_img = cv2.threshold(gaussian_img,210,255,cv2.THRESH_BINARY_INV)[1]

    pageCorners,_, = findFrameExistenceArea(inverse_bin_img)
    # print(pageCorners)

    canny_img = cv2.Canny(src_img, 120, 130, apertureSize=3)

    lines = []
    lines2 = []

    lines = cv2.HoughLines(canny_img, 1, np.pi / 180.0, 50)
    lines2 = cv2.HoughLines(canny_img, 1, np.pi / 360.0, 50)

    lines_img = np.zeros(src_img.shape, dtype=np.uint8)

    lines_img = drawLines(lines, lines_img)
    lines_img = drawLines(lines2, lines_img)

    and_img = np.zeros(src_img.shape, dtype=np.uint8)
    and_img = cv2.bitwise_and(inverse_bin_img, lines_img)

    # cv2.imshow('and_img', and_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours = []
    tmp_img = and_img.copy()
    contours, _ = cv2.findContours(tmp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundingbox_from_and_img = and_img.copy()

    complement_and_img = createAndImgWithBoundingBox(boundingbox_from_and_img, contours,inverse_bin_img)

    # cv2.imshow('complement_and_img', complement_and_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours3 = []
    bounding_boxes = []

    contours3, _ = cv2.findContours(complement_and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours3)):
        tmp_bounding_box = cv2.boundingRect(contours3[i])

        if judgeAreaOfBoundingBox(tmp_bounding_box, complement_and_img.shape[0]*complement_and_img.shape[1]):
            bounding_boxes.append(tmp_bounding_box)

    for i in range(len(contours3)):

        approx = cv2.approxPolyDP(contours3[i], 6, True)
        print(approx)

        # Create a bounding rectangle
        brect = cv2.boundingRect(contours3[i])
        print(brect)

        # Coordinates of the top left and bottom right
        xmin = brect[0]
        ymin = brect[1]
        xmax = brect[0] + brect[2]
        ymax = brect[1] + brect[3]

        if xmin<6:xmin = 0
        if xmax>inverse_bin_img.shape[1]-6:xmax = inverse_bin_img.shape[1]
        if ymin<6:ymin = 0
        if ymax>inverse_bin_img.shape[0]-6:ymax = inverse_bin_img.shape[0]

        bbPoints = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=np.int32)

        # 大きさ４で初期化
        definitePanelPoint = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)

        flag_LT = False
        flag_LB = False
        flag_RT = False
        flag_RB = False

        bb_min_LT = src_img.shape[0]
        bb_min_RT = src_img.shape[0]
        bb_min_LB = src_img.shape[0]
        bb_min_RB = src_img.shape[0]

        isOverlap = True

        if judgeAreaOfBoundingBox(brect, src_img.shape[1] * src_img.shape[0]):
            # Check if bounding boxes overlap
            isOverlap = judgeBoundingBoxOverlap(bounding_boxes, brect)
        else:
            isOverlap = False

        # if not isOverlap:
        #     continue
        for i in range(len(approx)):
            p = approx[i][0]

            print('deffinitePanel',definitePanelPoint)
            print('pagecorners',pageCorners)
            print('bbpoints',bbPoints)

            flag_LT, bb_min_LT, definitePanelPoint[0] = definePanelCorners(flag_LT, p, bb_min_LT,  pageCorners[0],  definitePanelPoint[0], bbPoints[0])
            flag_LB, bb_min_LB, definitePanelPoint[1] = definePanelCorners(flag_LB, p, bb_min_LB,  pageCorners[1],  definitePanelPoint[1], bbPoints[1])
            flag_RT, bb_min_RT, definitePanelPoint[2] = definePanelCorners(flag_RT, p, bb_min_RT,  pageCorners[2],  definitePanelPoint[2], bbPoints[2])
            flag_RB, bb_min_RB, definitePanelPoint[3] = definePanelCorners(flag_RB, p, bb_min_RB,  pageCorners[3],  definitePanelPoint[3], bbPoints[3])
            definitePanelPoint = align2edge(definitePanelPoint, inverse_bin_img)

        top_line, bottom_line, left_line, right_line = renew_line(definitePanelPoint)

        alphaImage = cv2.cvtColor(src_img, cv2.COLOR_BGR2BGRA)

        # createAlphaImage(alphaImage, definitePanelPoint)

        cut_img = alphaImage[brect[1]:brect[1]+brect[3], brect[0]:brect[0]+brect[2]]
        # cv2.imshow("cut_img",cut_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        panel_imgs = []
        panel_imgs.append(cut_img)
        print(panel_imgs[0].shape)

        print('definitePanelPoint[0]',definitePanelPoint[0])
        print('definitePanelPoint[1]',definitePanelPoint[1])
        print('definitePanelPoint[2]',definitePanelPoint[2])
        print('definitePanelPoint[3]',definitePanelPoint[3])
        cv2.line(color_img,definitePanelPoint[0],definitePanelPoint[2],(255,0,0),thickness=2,lineType=8)
        cv2.line(color_img,definitePanelPoint[2],definitePanelPoint[3],(255,0,0),thickness=2,lineType=8)
        cv2.line(color_img,definitePanelPoint[3],definitePanelPoint[1],(255,0,0),thickness=2,lineType=8)
        cv2.line(color_img,definitePanelPoint[1],definitePanelPoint[0],(255,0,0),thickness=2,lineType=8)

    for i in range (len(panel_imgs)):
        cv2.imshow("panel"+str(i),panel_imgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()













def extractSpeechBalloon(fukidashi_contours,hierarchy2,gaussian_img):
    for i in range(len(fukidashi_contours)):
        area = cv2.contourArea(fukidashi_contours[i])
        length = cv2.arcLength(fukidashi_contours[i], True)
        en = 0.0
        if gaussian_img.shape[0] * gaussian_img.shape[1] * 0.008 <= area and area < gaussian_img.shape[0] * gaussian_img.shape[1] * 0.03:
            en = 4.0 * np.pi * area / (length * length)
        if en > 0.4:
            cv2.drawContours(gaussian_img, fukidashi_contours, i, 0, -1, cv2.LINE_AA, hierarchy2, 1)

        return gaussian_img

def findFrameExistenceArea(inverse_bin_img):
    height, width = inverse_bin_img.shape

    histogram = np.zeros(width, dtype=int)

    for y in range(height):
        for x in range(width):
            if x <= 2 or x >= width - 2 or y <= 2 or y >= height - 2:
                continue
            if inverse_bin_img[y, x] > 0:
                histogram[x] += 1

        min_x = 0
        max_x = width -1

        for x in range(width):
            if histogram[x] > 0:
                min_x = x
                break

        for x in range(width - 1, -1, -1):
            if histogram[x] > 0:
                max_x = x
            break

        if min_x < 6: min_x = 0
        if max_x > width - 6: max_x = width

        pageCorners = [(min_x, 0), (max_x, 0), (max_x, height), (min_x, height)]

        rec_img = np.zeros((height, width), dtype=np.uint8)

        return pageCorners, rec_img

def drawLines(lines, lines_img):
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0 - 2000*b), int(y0 + 2000*a))
        pt2 = (int(x0 + 2000*b), int(y0 - 2000*a))

        lines_img = cv2.line(lines_img, pt1, pt2, (255), 1, cv2.LINE_AA)

        return lines_img

def createAndImgWithBoundingBox(src_img, contours, inverse_bin_img):
    for i in range(len(contours)):
        bounding_box = cv2.boundingRect(contours[i])
        if not judgeAreaOfBoundingBox(bounding_box, src_img.shape[0]*src_img.shape[1]):
            continue
        # Draw rectangle
        cv2.rectangle(src_img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (255), 3)

    dst_img = cv2.bitwise_and(src_img, inverse_bin_img)
    return dst_img

def judgeAreaOfBoundingBox(bounding_box, page_area):
    bb_area = bounding_box[2] * bounding_box[3]
    if bb_area < 0.048 * page_area:
        return False
    return True

def judgeBoundingBoxOverlap(bounding_boxes, brect):
    for box in bounding_boxes:
    # If it's the same, skip
        if box[0] == brect[0] and box[1] == brect[1] and box[2] == brect[2] and box[3] == brect[3]:
            continue

        overlap_rect = cv2.bitwise_and(brect, box)
        if overlap_rect[0] == 0 and overlap_rect[1] == 0 and overlap_rect[2] == 0 and overlap_rect[3] == 0:
            continue

        if (overlap_rect[0] == brect[0] and overlap_rect[1] == brect[1]
            and overlap_rect[2] == brect[2] and overlap_rect[3] == brect[3]):
            isOverlap = False

    return isOverlap

import numpy as np

def definePanelCorners(definite, current_point, bounding_box_min_dist, PageCornerPoint, definite_panel_point, boundingBoxPoint):
    if not definite:
        page_corner_dist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(PageCornerPoint))
        if page_corner_dist < 8:
            definite_panel_point = PageCornerPoint
            definite = True
        else:
            bounding_box_dist = np.linalg.norm(np.array(boundingBoxPoint) - np.array(current_point))
            if bounding_box_dist < bounding_box_min_dist:
                bounding_box_min_dist = bounding_box_dist
                definite_panel_point = current_point
    print("受け渡す前",definite_panel_point)
    return definite, bounding_box_min_dist, definite_panel_point

def align2edge(definite_panel_point, inverse_bin_img):
    th_edge = 6  # If within 6px
    for i in range(4):
        x, y = definite_panel_point[i]
        if i in [0, 2] and x < th_edge:  # lt and lb
            x = 0
        if i in [0, 1] and y < th_edge:  # lt and rt
            y = 0
        if i in [1, 3] and x > inverse_bin_img.shape[1] - th_edge:  # rt and rb
            x = inverse_bin_img.shape[1]
        if i in [2, 3] and y > inverse_bin_img.shape[0] - th_edge:  # lb and rb
            y = inverse_bin_img.shape[0]
        definite_panel_point[i] = (x, y)

    return definite_panel_point

def createAlphaImage(alph_img, definitePanelPoint):
    height, width = alph_img.shape
    for i in range(height):
        for j in range(width):
            px = alph_img[i, j]
            # 領域外を指定
            # if outside(definitePanelPoint, renew_line(definitePanelPoint), a, b):
            #     alph_img[i, j] = 0


def renew_line(p):
    top_line = [p[0], p[2],False]
    bottom_line = [p[1], p[3],False]
    left_line = [p[1], p[0],True]
    right_line = [p[3], p[2],True]
    return top_line, bottom_line, left_line, right_line

def calc(line):
    if line[2]:
        a = float(line[1][0] - line[0][0]) / (line[1][1] - line[0][1])
        b = line[0][0] - a * line[0][1]
    else:
        a = float(line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        b = line[0][1] - a * line[0][0]
    return a,b


def outside(p,lines, a,b):
    return judgeArea(p,lines[0][2],a,b) == 1 or judgeArea(p,lines[1][2],a,b) == 0 or judgeArea(p,lines[2][2],a,b) == 1 or judgeArea(p,lines[3][2],a,b) == 0

def judgeArea(p,y2x,a,b):
    if y2x: p = [p[1],p[0]]

    if p[1] > a * p[0] + b:
        return 0
    elif p[1] < a * p[0] + b:
        return 1
    else:
        return 2






if __name__ == '__main__':
    main()
