import cv2

# 读取拼接图片（注意图片左右的放置）
imageB = cv2.imread('data/left2.jpg')
imageA = cv2.imread('data/right2.jpg')

left = cv2.imread('data/left.jpg')
right = cv2.imread('data/right.jpg')

# stitcher = cv2.createStitcher(False)
pano = cv2.UMat()
stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

retval_E = stitcher.estimateTransform([imageB, imageA])
retval_C, pano = stitcher.composePanorama([imageB, imageA])
# retval_C, pano = stitcher.composePanorama((left, right))
# (_result, pano) = stitcher.stitch((imageB, imageA))

print(retval_C)
# cv2.imwrite('result/pano.jpg', pano)
