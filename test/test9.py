import cv2
import numpy as np

class CircleDetector:
    def __init__(self):
        # 初始化参数
        self.adapt_block = 15
        self.adapt_c = 2
        self.kernel_size = 3
        self.open_iter = 1
        self.close_iter = 2
        self.min_area = 1000
        self.max_area = 50000
        self.min_circularity = 0.5
        self.min_convexity = 0.6
        self.processing_images = {}

    def preprocess_frame(self, frame):
        """预处理帧图像"""
        # 高斯模糊
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 中值滤波
        medianBlur = cv2.medianBlur(blurred_frame, 5)
        
        # 转换为灰度图
        if medianBlur.ndim == 3:  # 检查是否为3通道（BGR）
            gray = cv2.cvtColor(medianBlur, cv2.COLOR_BGR2GRAY)
        else:
            gray = medianBlur.copy()  # 已经是灰度图则直接使用
        
        # 形态学闭运算
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        iteration = 21
        closeMat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element, iterations=iteration)
        
        # 背景减除
        calcMat = cv2.bitwise_not(cv2.subtract(closeMat, gray))
        
        # 归一化
        removeShadowMat = cv2.normalize(calcMat, None, 0, 200, cv2.NORM_MINMAX)
        
        # 双边滤波
        bilateral_blur2 = cv2.bilateralFilter(removeShadowMat, 9, 75, 75)
        bilateral_blur3 = cv2.bilateralFilter(bilateral_blur2, 9, 75, 75)
        
        # 二值化
        _, thresh = cv2.threshold(bilateral_blur3, 155, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作：腐蚀+膨胀
        kernel1 = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        kernel2 = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        
        return dilated

    def detect_circle(self, frame):
        """检测圆形目标"""
        # 预处理图像
        mask = self.preprocess_frame(frame)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_circle = None
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
                best_circle = {
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'score': score,
                    'area': area,
                    'circularity': circularity,
                    'convexity': convexity
                }
        
        return best_circle

    def visualize_results(self, frame, circle):
        """可视化检测结果"""
        if circle is None:
            return frame
        
        display_frame = frame.copy()
        center = circle['center']
        radius = circle['radius']
        
        # 绘制圆和中心标记
        cv2.circle(display_frame, center, radius, (0, 255, 0), 2)
        cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
        
        # 显示中心坐标和半径
        info_text = f"中心: ({center[0]}, {center[1]})"
        radius_text = f"半径: {radius}"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, radius_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示检测分数
        score_text = f"检测分数: {circle['score']:.2f}"
        cv2.putText(display_frame, score_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return display_frame

    def run(self):
        """运行圆环检测程序"""
        cap = cv2.VideoCapture(1)  # 使用默认摄像头1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 检测圆环
            circle = self.detect_circle(frame)
            
            # 可视化结果
            display_frame = self.visualize_results(frame, circle)
            
            # 显示结果
            cv2.imshow('Circle Detection', display_frame)
            cv2.imshow('Mask', self.preprocess_frame(frame))
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CircleDetector()
    detector.run()