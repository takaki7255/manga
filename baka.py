import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread('./output/cut/006_1.jpg')

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二値化
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# 輪郭を検出
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 各輪郭に対して処理
for cnt in contours:
    # 輪郭の外接矩形を取得
    x, y, w, h = cv2.boundingRect(cnt)

    # 外接矩形の面積が一定以上のものだけを取得
    if w * h > 40 * 40:
        # 外接矩形を描画
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # コマを切り出す
        roi = image[y:y+h, x:x+w]

        # 切り出したコマを保存
        cv2.imwrite('frame_' + str(x) + '.jpg', roi)

# 結果を表示
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
