import cv2
# from findRegion import findPlateNumberRegion

def process_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)  # 显示图片
    # cv2.waitKey(0)
    # 高斯平滑
    # gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # median = cv2.medianBlur(gaussian, 5)
    # # cv2.imshow('', median)
    # # cv2.waitKey(0)
    # # 边缘检测
    # sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # # cv2.imshow('', sobel)
    # # cv2.waitKey(0)
    # # 二值化
    # ret, binary = cv2.threshold(sobel, 190, 255, cv2.THRESH_BINARY)
    # # cv2.imshow('', binary)
    # # cv2.waitKey(0)
    # # 膨胀和腐蚀操作的核函数
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    # # 膨胀一次，让轮廓突出
    # dilation = cv2.dilate(binary, element2, iterations=1)
    # erosion = cv2.erode(dilation, element1, iterations=1)
    # dilation = cv2.dilate(erosion, element2, iterations=1)
    # # erosion = cv2.erode(dilation, element1, iterations=1)
    # # cv2.imshow('', erosion)
    # # cv2.waitKey(0)
    # # 再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(dilation, element2, iterations=1)
    # # cv2.imshow('', dilation2)
    # # cv2.waitKey(0)
    #
    # region = findPlateNumberRegion(dilation2)
    # # print(region)
    # # 用绿线画出这些找到的轮廓
    # cImg = img.copy()
    # for box in region:
    #     # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    #     Xs = [i[0] for i in box]
    #     Ys = [i[1] for i in box]
    #     x1 = min(Xs)
    #     x2 = max(Xs)
    #     y1 = min(Ys)
    #     y2 = max(Ys)
    #     hight = y2 - y1
    #     width = x2 - x1
    #     cropImg = gray[y1:y1 + hight, x1:x1 + width]
    #
    # cv2.imshow('1', cropImg)
    # cv2.waitKey(0)

    # 2、将灰度图像二值化，设定阈值是100
    ret, img_thre = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', img_thre)
    # cv2.waitKey(0)

    # 3、保存黑白图片
    # cv2.imwrite('F:\\PycharmProjects\\car_recognition\\thre_res.png', img_thre)

    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        # print(s)
        # print(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True


    # 分割图像
    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (
            0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    images = []
    n = 1
    start = 1
    end = 2
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = img_thre[1:height, start:end]
                cj = cv2.resize(cj,(20,20))
                # cv2.imshow('caijian', cj)
                # cv2.waitKey(0)
                images.append(cj)

    return images

