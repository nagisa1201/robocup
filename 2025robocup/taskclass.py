import cv2 
import numpy as np
from filterpy.kalman import KalmanFilter
import time 
class PreprocessingTask:
    """
    预处理任务类:对帧流进行预处理
    """
    def __init__(self, camera_frame):
        self.camera_frame = camera_frame
        self.frame_width = camera_frame.shape[1]
        self.frame_height = camera_frame.shape[0]
        self.isfinished = False
        self.min_thresh = 155
        self.max_thresh = 255

    def preprocess(self):
        """
        对帧流进行预处理
        :return: 预处理后的图像
        """
        blurred_frame = cv2.GaussianBlur(self.camera_frame, (5, 5), 0)
        medianBlur = cv2.medianBlur(blurred_frame, 5)
        if medianBlur.ndim == 3:  # 检查是否为3通道（BGR）
            gray = cv2.cvtColor(medianBlur, cv2.COLOR_BGR2GRAY)
        else:
            gray = medianBlur.copy()  # 已经是灰度图则直接使用
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        iteration = 5
        closeMat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element, iterations=iteration)
        calcMat = cv2.bitwise_not(cv2.subtract(closeMat, gray))
        removeShadowMat = cv2.normalize(calcMat, None, 0, 180, cv2.NORM_MINMAX)
        bilateral_blur2=cv2.bilateralFilter(removeShadowMat,9,75,75)
        
        bilateral_blur3=cv2.bilateralFilter(bilateral_blur2,9,75,75)
        _, thresh = cv2.threshold(bilateral_blur3, self.min_thresh, self.max_thresh, cv2.THRESH_BINARY_INV)
        kernel1 = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        kernel2 = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        self.camera_frame = dilated
        self.isfinished = True

class QRCodeTask:
    """
    二维码任务类:对帧流进行二维码检测和解码
    """
    def __init__(self,camera_frame):
        self.camera_frame = camera_frame
        self.qr_detector = cv2.QRCodeDetector()
        self.code_data = None # 最终引用的解码后的二维码数据
        self.isdetected = False # 最终判断二维码是否被检测到的标志位
        self.best_circle = None # 最佳圆环
    def detect_and_decode(self):
        """
        检测并解码二维码
        :return: retval表示成功与否（至少有一个二维码被检测到）, decoded_info（解码信息列表）, points（二维码角点列表）
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2GRAY)
        # 检测二维码
        retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)
        self.isdetected = retval
        if self.isdetected:
            self.code_data = decoded_info[0] if decoded_info else None
        return decoded_info, points
    def draw_qrcode_info(self, decoded_info, points):
        """
        在图像上绘制二维码信息,这个成员函数是用作该类调试使用
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        line_type = 2
        for i in range(len(decoded_info)):
            if decoded_info[i]:
                # 获取二维码的四个角点
                pts = points[i].astype(np.int32)
                cv2.polylines(self.camera_frame, [pts], True, (0, 255, 0), 2)
                text_pos = (pts[0][0], pts[0][1] - 10)
                cv2.putText(self.camera_frame, decoded_info[i], text_pos, 
                           font, font_scale, font_color, line_type)
                # 当有多个二维码时，显示数量
                cv2.putText(self.camera_frame, f"QRCodes Numbers: {len(decoded_info)}", (10, 30), 
                           font, font_scale, font_color, line_type)
        return self.camera_frame
    
class CircleTask:
    """
    圆形检测任务类:对帧流进行多圆环检测
    """
    def __init__(self, frame):
        self.processed_frame = frame
        self.isdetected = False  # 最终判断圆环是否被检测到的标志位
        self.best_circle = None  # 最佳圆环
        self.min_area = 1000
        self.max_area = 50000
        self.min_circularity = 0.5
        self.min_convexity = 0.6
    def detect_best_circle(self):
        """检测最佳圆环"""
        mask = self.processed_frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_score = 0
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤面积不满足条件的轮廓
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算轮廓的圆度
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue  # 避免除零错误
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 计算轮廓的凸度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue  # 避免除零错误
            convexity = area / hull_area
            
            # 过滤圆度和凸度不满足条件的轮廓
            if circularity < self.min_circularity or convexity < self.min_convexity:
                continue
            
            # 计算最小包围圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 计算轮廓矩以获取重心（更精确的中心估计）
            M = cv2.moments(contour)
            if M["m00"] > 0:  # 避免除以零
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)
            else:
                center = (int(x), int(y))
            
            # 综合评分
            score = 0.8 * circularity + 0.2 * convexity
            
            # 筛选最佳圆环
            if score > max_score:
                max_score = score
                self.best_circle = {
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'score': score,
                    'area': area,
                    'circularity': circularity,
                    'convexity': convexity
                }
    def draw_best_circle(self):
        """在图像上绘制最佳圆环"""
        if self.best_circle is not None:
            center = self.best_circle['center']
            radius = self.best_circle['radius']
            cv2.circle(self.processed_frame, center, radius, (0, 255, 0), 2)
            cv2.putText(self.processed_frame, f"Best Circle: {self.best_circle['score']:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return self.processed_frame




if __name__ == "__main__":
    cv2.namedWindow("Processed Frame")

    cap = cv2.VideoCapture(0)   

    
    while True:
        ret, frame = cap.read() 
        if not ret:
            print("无法读取相机帧")
            break

        preprocessing_task = PreprocessingTask(frame)
        preprocessing_task.preprocess()
        process_frame = preprocessing_task.camera_frame
        qr_task = QRCodeTask(frame)
        circle_task = CircleTask(process_frame)
        circle_task.detect_best_circle()
        process_frame = circle_task.draw_best_circle()

        decoded_info, points = qr_task.detect_and_decode()
        frame = qr_task.draw_qrcode_info(decoded_info, points)
        print(f"检测到二维码: {qr_task.code_data}")
        cv2.imshow("QR Code Detection", frame)
        cv2.imshow("Processed Frame", process_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()