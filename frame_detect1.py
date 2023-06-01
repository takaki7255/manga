import cv2
import numpy as np

# 線の定義と計算
def line_def(p1, p2, y2x):
    line = {'p1': p1, 'p2': p2, 'y2x': y2x}
    line['a'], line['b'] = calc(line)
    return line

def calc(line):
    p1, p2, y2x = line['p1'], line['p2'], line['y2x']
    if y2x:
        a = float(p2[0]-p1[0])/(p2[1]-p1[1])
        b = p1[0]-p1[1]*a
    else:
        a = float(p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1]-p1[0]*a
    return a, b

# 領域の判定
def judge_area(line, p):
    if line['y2x']: p = (p[1], p[0])
    # 直線よりも上の領域
    if p[1] > line['a']*p[0] + line['b']: return 0
    # 直線よりも下の領域
    elif p[1] < line['a']*p[0] + line['b']: return 1
    # 直線上
    else: return 2

# 四角形の定義
def points_def(lt=(0, 0), lb=(0, 0), rt=(0, 0), rb=(0, 0)):
    points = {'lt': lt, 'lb': lb, 'rt': rt, 'rb': rb}
    points['top_line'] = line_def(lt, rt, False)
    points['bottom_line'] = line_def(lb, rb, False)
    points['left_line'] = line_def(lb, lt, True)
    points['right_line'] = line_def(rb, rt, True)
    return points

# 四角形の外側(直線上は含まない)にあるかの判定
def outside(points, p):
    return judge_area(points['top_line'], p) == 1 or \
           judge_area(points['right_line'], p) == 0 or \
           judge_area(points['bottom_line'], p) == 0 or \
           judge_area(points['left_line'], p) == 1


def find_frame_existence_area(inverse_bin_img):
    histogram = np.zeros(inverse_bin_img.shape[1])
    
    for y in range(inverse_bin_img.shape[0]):
        for x in range(inverse_bin_img.shape[1]):
            if 2 < x < inverse_bin_img.shape[1] - 2 and 2 < y < inverse_bin_img.shape[0] - 2:
                if 0 < inverse_bin_img[y, x]:
                    histogram[x] += 1
                    
    min_x, max_x = 0, inverse_bin_img.shape[1] - 1
    min_x = next((i for i, x in enumerate(histogram) if x > 0), 0)
    max_x = next((i for i, x in reversed(list(enumerate(histogram))) if x > 0), 0)
    
    if min_x < 6: min_x = 0
    if max_x > inverse_bin_img.shape[1] - 6: max_x = inverse_bin_img.shape[1]
    
    page_corners = points_def((min_x, 0), (min_x, inverse_bin_img.shape[0]), (max_x, 0), (max_x, inverse_bin_img.shape[0]))
    
    rec_img = np.zeros((inverse_bin_img.shape[0], inverse_bin_img.shape[1], 3), dtype=np.uint8)
    
    return page_corners

def draw_hough_lines(lines, draw_lines_image):
    for i in range(min(len(lines), 100)):
        line = lines[i]
        rho, theta = line.x, line.y
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = a*rho, b*rho
        
        pt1 = (x0 - 2000*b, y0 + 2000*a)
        pt2 = (x0 + 2000*b, y0 - 2000*a)
        
        cv2.line(draw_lines_image, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)

def create_and_img_with_bounding_box(src_img, contours, inverse_bin_img):
    for contour in contours:
        bounding_box = cv2.boundingRect(contour)
        if not judge_area_of_bounding_box(bounding_box, src_img.shape[0]*src_img.shape[1]):
            continue
        cv2.rectangle(src_img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), (255, 0, 0), 3)
    dst_img = cv2.bitwise_and(src_img, inverse_bin_img)
    return dst_img

def judge_area_of_bounding_box(bounding_box, page_area):
    if bounding_box[2]*bounding_box[3] < 0.048 * page_area:
        return False
    return True

def judge_bounding_box_overlap(is_overlap, bounding_boxes, brect):
    for j in range(len(bounding_boxes)):
        if bounding_boxes[j] == brect:
            continue
        overlap_rect = brect & bounding_boxes[j]
        if overlap_rect == brect:
            is_overlap = False
    return is_overlap

def define_panel_corners(definite, current_point, bounding_box_min_dist, page_corner_point, bounding_box_point):
    if not definite:
        page_corner_dist = np.linalg.norm(bounding_box_point-page_corner_point)
        if page_corner_dist < 8:
            definite_panel_point = page_corner_point
            definite = True
        else:
            bounding_box_dist = np.linalg.norm(bounding_box_point - current_point)
            if bounding_box_dist < bounding_box_min_dist:
                bounding_box_min_dist = bounding_box_dist
                definite_panel_point = current_point
    return definite, bounding_box_min_dist, definite_panel_point

def align2edge(definite_panel_point, inverse_bin_img):
    th_edge = 6
    if definite_panel_point.lt.x < th_edge: definite_panel_point.lt.x = 0
    if definite_panel_point.lt.y < th_edge: definite_panel_point.lt.y = 0
    if definite_panel_point.rt.x > inverse_bin_img.shape[1] - th_edge: definite_panel_point.rt.x = inverse_bin_img.shape[1]
    if definite_panel_point.rt.y < th_edge: definite_panel_point.rt.y = 0
    if definite_panel_point.lb.x < th_edge: definite_panel_point.lb.x = 0
    if definite_panel_point.lb.y > inverse_bin_img.shape[0] - th_edge: definite_panel_point.lb.y = inverse_bin_img.shape[0]
    if definite_panel_point.rb.x > inverse_bin_img.shape[1] - th_edge: definite_panel_point.rb.x = inverse_bin_img.shape[1]
    if definite_panel_point.rb.y > inverse_bin_img.shape[0] - th_edge: definite_panel_point.rb.y = inverse_bin_img.shape[0]
    return definite_panel_point

def create_alpha_image(alpha_image, definite_panel_point):
    for y in range(alpha_image.shape[0]):
        for x in range(alpha_image.shape[1]):
            if definite_panel_point.outside((x, y)):
                alpha_image[y, x, 3] = 0
    return alpha_image
