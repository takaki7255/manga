import cv2
import numpy as np
import os


# 15色を生成する関数
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = int(255 * i / num_colors)  # 色相を変化
        saturation = 255
        value = 255
        colors.append(cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0])
    return colors


colors = generate_colors(15)


result_list = os.listdir("./result/")

# for result in result_list:
#     file = result.split(".")[0]
#     with open("result/" + result, mode="r") as f:
#         lines = f.readlines()
#         for line in lines:
#             if "BoundingBox" not in line:
#                 with open("result/" + file + "seiri.txt", mode="a") as f:
#                     f.write(line)
#                 continue
#             elif "Subpanels" in line:
#                 with open("result/" + file + "seiri.txt", mode="a") as f:
#                     f.write(line)
#                 continue
#             else:
#                 parts = line.split(" ")
#                 # print(parts)
#                 for part in parts:
#                     if ":" in part:
#                         order = part.split(":")[0]
#                     if "\n" in part:
#                         frame_info = part.split("\n")
#                     if "BoundingBox" in part:
#                         frame_id = part
#                 # print(order)
#                 # print(frame_info)
#                 frame_info = frame_info[0].split(",")
#                 # print(frame_info)
#                 xmin = round(float(frame_info[0]))
#                 ymin = round(float(frame_info[1]))
#                 xmax = round(float(frame_info[2]))
#                 ymax = round(float(frame_info[3]))
#                 # print(order, xmin, ymin, xmax, ymax, frame_id)
#                 with open("result/" + file + "seiri.txt", mode="a") as f:
#                     f.write(
#                         order
#                         + ","
#                         + str(xmin)
#                         + ","
#                         + str(ymin)
#                         + ","
#                         + str(xmax)
#                         + ","
#                         + str(ymax)
#                         + ","
#                         + frame_id
#                         + "\n"
#                     )
# ---------------------------ここまでデータ整理---------------------------


manga_list = os.listdir("./../Manga109_released_2021_12_30/images/")
do_list = [
    "AppareKappore",
    "EverydayOsakanaChan",
    "KarappoHighschool",
    "MAD_STONE",
    "PrismHeart",
    "ReveryEarth",
    "SeisinkiVulnus",
    "TasogareTsushin",
    "That'sIzumiko",
    "YoumaKourin",
]

import os

result_seiri_list = [
    "./result/AppareKappore_seiri.txt",
    "./result/EverydayOsakanaChanseiri.txt",
    "./result/KarappoHighschoolseiri.txt",
    "./result/MAD_STONEseiri.txt",
    "./result/PrismHeartseiri.txt",
    "./result/ReveryEarthseiri.txt",
    "./result/SeisinkiVulnusseiri.txt",
    "./result/TasogareTsushinseiri.txt",
    "./result/That'sIzumikoseiri.txt",
    "./result/YoumaKourinseiri.txt",
]
for result in result_seiri_list:
    with open(result, mode="r") as f:
        # 最初の一行はタイトル
        lines = f.readlines()
        title = lines[0]
        title = title.split("\n")[0]
        print(title)
        pages = {}
        for line in lines:
            if "Page" in line:
                page_num = line.split(" ")[1].split(":")[0]
                # print(page_num)
                page_num = int(page_num)
                if page_num < 10:
                    page_num = "00" + str(page_num)
                elif page_num < 100:
                    page_num = "0" + str(page_num)
                else:
                    page_num = str(page_num)
                current_page = page_num
                pages[current_page] = []
            if "Subpanels" in line:
                continue
            if "BoundingBox" in line:
                order = line.split(",")[0]
                xmin = line.split(",")[1]
                ymin = line.split(",")[2]
                xmax = line.split(",")[3]
                ymax = line.split(",")[4]
                box = [order, xmin, ymin, xmax, ymax]
                pages[current_page].append(box)
    # print(pages)
    for page_num, boxes in pages.items():
        print(page_num)
        print(boxes)
        img_path = "./../Manga109_released_2021_12_30/images/" + title + "/" + page_num + ".jpg"
        print(img_path)
        img = cv2.imread(img_path)
        if img is not None:
            color = generate_colors(len(boxes))
            for box in boxes:
                order = int(box[0])
                xmin = int(box[1])
                ymin = int(box[2])
                xmax = int(box[3])
                ymax = int(box[4])
                color = colors[order % 15]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
                # cv2.putText(
                #     img,
                #     str(order),
                #     (xmin, ymin - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9,
                #     color[int(order) - 1],
                #     2,
                #     cv2.LINE_AA,
                # )
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("画像がありません")
