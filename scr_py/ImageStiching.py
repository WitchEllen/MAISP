from Stitcher import Stitcher
import cv2

# 读取拼接图片（注意图片左右的放置）
imageB = cv2.imread('data/left.jpg')
imageA = cv2.imread('data/right.jpg')  # 是对右边的图形做变换
# imageA = cv2.imread("test2.jpg")
# imageB = cv2.imread("test1.jpg")
# imageA = cv2.resize(imageA, (0, 0), fx=0.3, fy=0.3)
# imageB = cv2.resize(imageB, (0, 0), fx=0.3, fy=0.3)

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('result/result_v2.jpg', result)
cv2.imwrite('result/vis_v2.jpg', vis)
