import cv2
import numpy as np

def detect_lines(frame, threshold):
    # 高斯滤波和中值滤波
    guss_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    median_blur = cv2.medianBlur(guss_blur, 5)

    # 转换为灰度图像
    gray = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)

    # 应用二值化，将黑色线条提取出来
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_lines = []

    # 遍历轮廓并提取线段
    for contour in contours:
        # 过滤掉小轮廓
        if cv2.contourArea(contour) > 100:
            # 拟合多边形，简化轮廓
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 如果多边形的边数大于等于2，假设有直线
            if len(approx) >= 2:
                for i in range(len(approx) - 1):
                    x1, y1 = approx[i][0]
                    x2, y2 = approx[i + 1][0]
                    valid_lines.append((x1, y1, x2, y2))

    return gray, binary, valid_lines

def lines_intersection(line1, line2):
    # 获取直线的两个端点
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 计算直线交点
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 直线平行

    cx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    cy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return cx, cy

def update_threshold(val):
    global threshold
    threshold = val

def main():
    global threshold
    threshold = 50  # 初始化阈值

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 创建窗口
    cv2.namedWindow("Frame")

    # 创建滑条
    cv2.createTrackbar("Threshold", "Frame", threshold, 255, update_threshold)

    # 定义感兴趣区域 (ROI) 的位置和大小
    roi_x, roi_y, roi_w, roi_h = 200, 100, 300, 200  # 设置ROI的左上角坐标和宽高

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break

        # 在帧上绘制ROI矩形
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # 识别直线
        gray, binary, valid_lines = detect_lines(roi, threshold)

        # 绘制检测到的直线
        for x1, y1, x2, y2 in valid_lines:
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 如果检测到两条直线，计算交点
        if len(valid_lines) >= 2:
            line1 = valid_lines[0]
            line2 = valid_lines[1]
            intersection_point = lines_intersection(line1, line2)

            if intersection_point is not None:
                cx, cy = intersection_point
                # 画出交点
                cv2.circle(roi, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                # 计算偏移量
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                offset_x = cx - center_x
                offset_y = cy - center_y

                # 显示偏移量
                cv2.putText(roi, f'Offset: ({offset_x:.2f}, {offset_y:.2f})', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 在帧上绘制ROI矩形
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow("Frame", frame)
        cv2.imshow("Gray", gray)
        cv2.imshow("Binary", binary)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
