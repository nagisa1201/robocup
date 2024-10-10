import cv2
import numpy as np

# 配置摄像头捕捉框的参数
cap = cv2.VideoCapture(0)

def empty(a):
    pass

# 创建滑条窗口
'''
126 3395 335
'''
cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 640, 240)
cv2.createTrackbar('Threshold', 'Parameters', 127, 255, empty)
cv2.createTrackbar('Max Area', 'Parameters', 5000, 10000, empty)  # 最大面积阈值
cv2.createTrackbar('Min Area', 'Parameters', 500, 2000, empty)    # 最小面积阈值

# 定义感兴趣区域（ROI）
roi_start_point = (200, 100)  # ROI左上角坐标
roi_end_point = (440, 380)     # ROI右下角坐标

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在图像上绘制 ROI 矩形
        cv2.rectangle(frame, roi_start_point, roi_end_point, (255, 0, 0), 2)

        # 提取 ROI
        roi = frame[roi_start_point[1]:roi_end_point[1], roi_start_point[0]:roi_end_point[0]]

        # 转换到灰度图像
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 获取滑条的值
        thresh = cv2.getTrackbarPos('Threshold', 'Parameters')
        max_area = cv2.getTrackbarPos('Max Area', 'Parameters')
        min_area = cv2.getTrackbarPos('Min Area', 'Parameters')

        # 二值化处理
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 50:  # 需要至少5个点来拟合圆
                # 拟合圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # 计算拟合圆的面积
                area = np.pi * (radius ** 2)

                # 过滤掉过大的圆和过小的圆
                if min_area <= area <= max_area:
                    # 将圆的中心坐标转换为在原图中的坐标
                    center_in_frame = (center[0] + roi_start_point[0], center[1] + roi_start_point[1])
                    # 绘制拟合的圆
                    cv2.circle(frame, center_in_frame, radius, (0, 255, 0), 2)  # 绘制外圆
                    cv2.circle(frame, center_in_frame, 2, (0, 0, 255), 3)      # 绘制圆心

        # 显示结果
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
