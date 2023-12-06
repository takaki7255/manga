import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from manga109utils import *

import order_estimator


def show_panel(panel, labeltext, linestyle="solid", edgecolor="red"):
    fontsize = 40
    plt.axes().add_patch(
        patches.Rectangle(
            xy=(panel.xmin, panel.ymin),
            width=panel.width,
            height=panel.height,
            linewidth=3,
            linestyle=linestyle,
            ec=edgecolor,
            fill=False,
        )
    )
    g_sub_bb_x = panel.xmin + panel.width / 2
    g_sub_bb_y = panel.ymin + panel.height / 2
    plt.text(g_sub_bb_x - fontsize / 2, g_sub_bb_y + fontsize / 2, labeltext, fontsize=fontsize, color=edgecolor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str)
    parser.add_argument("--page", type=int)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--dataset-root", type=str, default="./../Manga109_released_2021_12_30")
    parser.add_argument("--initial-cut", type=str, default="two-page", help="two-page-four-panel / two-page / one-page")

    args = parser.parse_args()

    order_estimator.interception_ratio_threshold = args.threshold

    book = Book(args.title, manga109_root_dir=args.dataset_root)
    for i_page, page in enumerate(book.get_page_iter()):
        if i_page != args.page:
            continue

        print(f"{book.title} p.{i_page}")
        image = page.get_image()
        pagewidth = image.size[0]

        panels = page.get_bbs()["frame"]
        boxOrderEstimator = order_estimator.BoxOrderEstimator(
            panels, pagewidth=pagewidth, initial_cut_option=args.initial_cut
        )

        plt.figure(figsize=(10, 7))
        if len(np.array(image).shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

        for i_panel, panel in enumerate(boxOrderEstimator.ordered_bbs):
            order = i_panel + 1
            if len(panel.panels) == 1:
                show_panel(panel, labeltext=f"{order}", edgecolor="red")
            else:
                for subpanel in panel.panels:
                    show_panel(subpanel, edgecolor="orange", linestyle="dotted", labeltext=f"({order})")
        plt.show()

        break
