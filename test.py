import cv2
import numpy as np

class PageFrameDetector:
    def __init__(self):
        self.page_images = []

    def pageCut(self, input_page_image):
        if input_page_image.shape[1] > input_page_image.shape[0]:
            cut_img_left = input_page_image[:, :input_page_image.shape[1]//2]
            cut_img_right = input_page_image[:, input_page_image.shape[1]//2:]
            self.page_images.append(cut_img_right)
            self.page_images.append(cut_img_left)
        else:
            self.page_images.append(input_page_image)

    def blackpageFramedetect(self, input_page_image):
        histgram_src_tb = np.zeros(input_page_image.shape[0], dtype=int)
        
        for y in range(input_page_image.shape[0]):
            for x in range(input_page_image.shape[1]):
                if input_page_image[y, x] == 0:
                    histgram_src_tb[y] += 1

        for i in range(input_page_image.shape[0]):
            print(histgram_src_tb[i])

    def findFrameArea(self, input_page_image):
        gaussian_img = cv2.GaussianBlur(input_page_image, (3,3), 0)
        _, inverse_bin_img = cv2.threshold(~gaussian_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        histgram_lr = np.zeros(inverse_bin_img.shape[1], dtype=int)
        
        for y in range(inverse_bin_img.shape[0]):
            for x in range(inverse_bin_img.shape[1]):
                if x <= 2 or x >= inverse_bin_img.shape[1] - 2 or y <= 2 or y >= inverse_bin_img.shape[0] - 2:
                    continue
                if inverse_bin_img[y, x] > 0:
                    histgram_lr[x] += 1
        
        min_x_lr = np.where(histgram_lr > 0)[0][0]
        max_x_lr = np.where(histgram_lr > 0)[0][-1]
        
        if min_x_lr < 6:
            min_x_lr = 0
        if max_x_lr > inverse_bin_img.shape[1] - 6:
            max_x_lr = inverse_bin_img.shape[1]
        
        cut_page_img_lr = input_page_image[:, min_x_lr:max_x_lr]
        
        histgram_tb = np.zeros(inverse_bin_img.shape[0], dtype=int)
        
        for y in range(inverse_bin_img.shape[0]):
            for x in range(inverse_bin_img.shape[1]):
                if x <= 2 or x >= inverse_bin_img.shape[0] - 2 or y <= 2 or y >= inverse_bin_img.shape[0] - 2:
                    continue
                if inverse_bin_img[y, x] > 0:
                    histgram_tb[y] += 1
        
        min_y_tb = np.where(histgram_tb > 0)[0][0]
        max_y_tb = np.where(histgram_tb > 0)[0][-1]
        
        if min_y_tb < 6:
            min_y_tb = 0
        if max_y_tb > cut_page_img_lr.shape[0] - 6:
            max_y_tb = cut_page_img_lr.shape[0]
        
        cut_page_img = cut_page_img_lr[min_y_tb:max_y_tb, :]
        return cut_page_img

def main():
    FILENAME = './../Manga109_released_2021_12_30/images/JijiBabaFight/003.jpg'
    # Load image in grayscale
    src_img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print("file is not found")
        return -1

    # Instantiate PageFrameDetector class
    page_frame_detector = PageFrameDetector()
    
    # Cut image if it is spread
    page_frame_detector.pageCut(src_img)
    for page_img in page_frame_detector.page_images:
        cv2.imshow("page", page_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Black page detection
    src_img = page_frame_detector.findFrameArea(src_img)
    page_frame_detector.blackpageFramedetect(src_img)
    return 0

if __name__ == "__main__":
    main()
