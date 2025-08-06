import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import time

class CircleDetector:
    def __init__(self, camera_id=0):
        # 检测并列出可用摄像头
        self.available_cameras = self.find_available_cameras()
        print(f"可用摄像头列表: {self.available_cameras}")
        
        # 选择摄像头
        self.camera_id = camera_id if camera_id in self.available_cameras else (self.available_cameras[0] if self.available_cameras else 0)
        
        # 初始化相机并设置帧率上限为30
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 {self.camera_id}")
        
        # 设置相机参数：帧率上限30
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.target_fps = 30
        self.last_frame_time = time.time()
        self.fps = 0
        
        # 获取相机帧尺寸
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        # 确保帧尺寸有效，避免后续处理越界
        if self.frame_width <= 0 or self.frame_height <= 0:
            self.frame_width, self.frame_height = 640, 480
        
        # 创建参数调节窗口（仅保留模式2相关参数）
        cv2.namedWindow('Settings')
        self.min_thresh = 155
        self.max_thresh = 255
        self.create_trackbars()
        
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.array([self.frame_width/2, self.frame_height/2, 50, 0, 0, 0])
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.kf.P *= 1000.0
        self.kf.R = np.array([[10, 0, 0],
                             [0, 10, 0],
                             [0, 0, 20.0]])
        self.kf.Q = np.eye(6) * 0.1
        
        # 初始化变量
        self.last_detected = False
        self.last_color = (0, 0, 0)
        self.color_history = []
        self.processing_images = {}
        self.last_valid_state = None
        self.detection_failure_count = 0
        self.max_failure_count = 5
        self.debug_mode = 1
        self.detection_confidence_threshold = 0.5


    def find_available_cameras(self, max_check=10):
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available
    
    def switch_camera(self, camera_id):
        if camera_id in self.available_cameras and camera_id != self.camera_id:
            self.cap.release()
            self.camera_id = camera_id
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # 切换摄像头后重新设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 更新相机尺寸并确保有效
            self.frame_width = max(640, int(self.cap.get(3)))
            self.frame_height = max(480, int(self.cap.get(4)))
            self.kf.x = np.array([self.frame_width/2, self.frame_height/2, 50, 0, 0, 0])
            self.last_valid_state = None
            self.detection_failure_count = 0
            
            print(f"已切换到摄像头 {camera_id}，帧率上限保持为 {self.target_fps}")
            return True
        return False
    
    def create_trackbars(self):
        # 仅保留自适应阈值法（模式2）相关参数
        cv2.createTrackbar('ADAPT_BLOCK', 'Settings', 15, 50, lambda x: None)
        cv2.createTrackbar('ADAPT_C', 'Settings', 2, 20, lambda x: None)
        cv2.createTrackbar('BLUR_SIZE', 'Settings', 5, 21, lambda x: None)  # 高斯模糊尺寸
        
        # 形态学参数
        cv2.createTrackbar('KERNEL_SIZE', 'Settings', 3, 10, lambda x: None)
        cv2.createTrackbar('OPEN_ITER', 'Settings', 1, 5, lambda x: None)
        cv2.createTrackbar('CLOSE_ITER', 'Settings', 2, 5, lambda x: None)
        
        # 轮廓筛选参数
        cv2.createTrackbar('MIN_AREA', 'Settings', 1000, 10000, lambda x: None)
        cv2.createTrackbar('MAX_AREA', 'Settings', 50000, 100000, lambda x: None)
        cv2.createTrackbar('MIN_CIRCULARITY', 'Settings', 50, 100, lambda x: None)  # 0-100%
        cv2.createTrackbar('MIN_CONVEXITY', 'Settings', 60, 100, lambda x: None)    # 0-100%
        
        # 卡尔曼滤波器参数
        cv2.createTrackbar('KF_PROC_NOISE', 'Settings', 10, 100, lambda x: None)  # 0.1-10
        cv2.createTrackbar('KF_MEAS_NOISE', 'Settings', 20, 100, lambda x: None)  # 0.1-10
        
        # 检测灵敏度参数
        cv2.createTrackbar('DETECT_SENSITIVITY', 'Settings', 50, 100, lambda x: None)  # 0-100%
        
        # 新增的二值化阈值参数
        cv2.createTrackbar('MIN_THRESH', 'Settings', self.min_thresh, 255, lambda x: None)
        cv2.createTrackbar('MAX_THRESH', 'Settings', self.max_thresh, 255, lambda x: None)

    def get_current_settings(self):
        # 自适应二值化参数（确保块大小为有效奇数）
        adapt_block = cv2.getTrackbarPos('ADAPT_BLOCK', 'Settings')
        adapt_block = max(3, adapt_block if adapt_block % 2 == 1 else adapt_block + 1)  # 最小3，确保奇数
        
        adapt_c = max(0, cv2.getTrackbarPos('ADAPT_C', 'Settings'))  # 确保非负
        
        # 高斯模糊尺寸（确保有效奇数）
        blur_size = cv2.getTrackbarPos('BLUR_SIZE', 'Settings')
        blur_size = max(1, blur_size if blur_size % 2 == 1 else blur_size + 1)  # 最小1，确保奇数
        
        # 形态学参数（确保有效）
        kernel_size = max(1, min(10, cv2.getTrackbarPos('KERNEL_SIZE', 'Settings')))  # 限制在1-10
        open_iter = max(1, min(10, cv2.getTrackbarPos('OPEN_ITER', 'Settings')))  # 限制在1-10
        close_iter = max(1, min(10, cv2.getTrackbarPos('CLOSE_ITER', 'Settings')))  # 限制在1-10
        
        # 轮廓筛选参数（确保范围有效）
        min_area = max(100, cv2.getTrackbarPos('MIN_AREA', 'Settings'))  # 最小100
        max_area = max(min_area + 1000, cv2.getTrackbarPos('MAX_AREA', 'Settings'))  # 确保max > min
        min_circularity = max(0.1, cv2.getTrackbarPos('MIN_CIRCULARITY', 'Settings') / 100.0)  # 最小0.1
        min_convexity = max(0.1, cv2.getTrackbarPos('MIN_CONVEXITY', 'Settings') / 100.0)  # 最小0.1
        
        # 卡尔曼滤波参数（确保有效范围）
        kf_proc_noise = max(0.1, cv2.getTrackbarPos('KF_PROC_NOISE', 'Settings') / 10.0)
        kf_meas_noise = max(0.1, cv2.getTrackbarPos('KF_MEAS_NOISE', 'Settings') / 10.0)
        
        # 检测灵敏度
        detect_sensitivity = max(0.1, cv2.getTrackbarPos('DETECT_SENSITIVITY', 'Settings') / 100.0)
        
        # 更新二值化阈值
        self.min_thresh = cv2.getTrackbarPos('MIN_THRESH', 'Settings')
        self.max_thresh = cv2.getTrackbarPos('MAX_THRESH', 'Settings')
        
        return {
            'adapt_block': adapt_block,
            'adapt_c': adapt_c,
            'blur_size': blur_size,
            'kernel_size': kernel_size,
            'open_iter': open_iter,
            'close_iter': close_iter,
            'min_area': min_area,
            'max_area': max_area,
            'min_circularity': min_circularity,
            'min_convexity': min_convexity,
            'kf_proc_noise': kf_proc_noise,
            'kf_meas_noise': kf_meas_noise,
            'detect_sensitivity': detect_sensitivity
        }

    def preprocess_frame(self, frame):
        """使用您提供的预处理流程"""
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
        iteration = 5
        closeMat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element, iterations=iteration)
        
        # 背景减除
        calcMat = cv2.bitwise_not(cv2.subtract(closeMat, gray))
        
        # 归一化
        removeShadowMat = cv2.normalize(calcMat, None, 0, 180, cv2.NORM_MINMAX)
        
        # 双边滤波
        bilateral_blur2 = cv2.bilateralFilter(removeShadowMat, 9, 75, 75)
        bilateral_blur3 = cv2.bilateralFilter(bilateral_blur2, 9, 75, 75)
        
        # 二值化
        _, thresh = cv2.threshold(bilateral_blur3, self.min_thresh, self.max_thresh, cv2.THRESH_BINARY_INV)
        
        # 形态学操作：腐蚀+膨胀
        kernel1 = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        kernel2 = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(eroded, kernel2, iterations=1)
        
        # 保存中间结果用于调试
        self.processing_images['blurred'] = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
        self.processing_images['median'] = cv2.cvtColor(medianBlur, cv2.COLOR_BGR2RGB) if medianBlur.ndim == 3 else cv2.cvtColor(medianBlur, cv2.COLOR_GRAY2RGB)
        self.processing_images['gray'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        self.processing_images['close'] = cv2.cvtColor(closeMat, cv2.COLOR_GRAY2RGB)
        self.processing_images['calc'] = cv2.cvtColor(calcMat, cv2.COLOR_GRAY2RGB)
        self.processing_images['removeShadow'] = cv2.cvtColor(removeShadowMat, cv2.COLOR_GRAY2RGB)
        self.processing_images['bilateral2'] = cv2.cvtColor(bilateral_blur2, cv2.COLOR_GRAY2RGB)
        self.processing_images['bilateral3'] = cv2.cvtColor(bilateral_blur3, cv2.COLOR_GRAY2RGB)
        self.processing_images['thresh'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        self.processing_images['eroded'] = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)
        self.processing_images['dilated'] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
        
        return dilated

    def detect_circle(self, frame):
        """检测圆形目标，增加边界检查避免越界"""
        settings = self.get_current_settings()
        
        # 确保帧有效
        if frame is None or frame.size == 0:
            return None, None, settings
        
        # 使用您提供的预处理方法
        mask = self.preprocess_frame(frame)
        
        # 查找轮廓，处理空掩码情况
        if mask is None or mask.size == 0:
            return None, mask, settings
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓并保存（检查图像尺寸避免越界）
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            contour_img = frame.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
            self.processing_images['contours'] = contour_img
        
        best_circle = None
        max_score = 0
        center_candidates = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤面积不满足条件的轮廓
            if area < settings['min_area'] or area > settings['max_area']:
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
            if circularity < settings['min_circularity'] or convexity < settings['min_convexity']:
                continue
            
            # 计算最小包围圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 检查是否在图像范围内，避免越界
            if x < 0 or x >= self.frame_width or y < 0 or y >= self.frame_height:
                continue
            
            # 计算轮廓矩以获取重心（更精确的中心估计）
            M = cv2.moments(contour)
            if M["m00"] > 0:  # 避免除以零
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # 确保中心在图像范围内
                cX = max(0, min(self.frame_width - 1, cX))
                cY = max(0, min(self.frame_height - 1, cY))
                center = (cX, cY)
                center_candidates.append(center)
            else:
                # 确保中心在图像范围内
                x_clamped = max(0, min(self.frame_width - 1, int(x)))
                y_clamped = max(0, min(self.frame_height - 1, int(y)))
                center = (x_clamped, y_clamped)
            
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
        
        # 调试信息：显示所有候选圆心（检查图像是否存在）
        if self.debug_mode and center_candidates and 'contours' in self.processing_images:
            temp_img = self.processing_images['contours'].copy()
            for (cx, cy) in center_candidates:
                # 确保绘制坐标在图像范围内
                if 0 <= cx < temp_img.shape[1] and 0 <= cy < temp_img.shape[0]:
                    cv2.circle(temp_img, (cx, cy), 3, (255, 0, 0), -1)
            self.processing_images['candidates'] = temp_img
        
        return best_circle, mask, settings

    def process_frame(self, frame):
        """处理帧并检测圆形，增加边界检查"""
        # 检查帧有效性
        if frame is None or frame.size == 0:
            return None
            
        circle, mask, settings = self.detect_circle(frame)
        
        # 更新卡尔曼滤波器参数
        self.kf.Q = np.eye(6) * settings['kf_proc_noise']
        self.kf.R = np.array([
            [settings['kf_meas_noise'], 0, 0],
            [0, settings['kf_meas_noise'], 0],
            [0, 0, settings['kf_meas_noise'] * 50]
        ])
        
        circle_info = None
        
        # 预测当前状态并确保有效性
        try:
            prediction = self.kf.predict()
            # 检查预测结果是否有效
            if prediction is None or not np.all(np.isfinite(prediction)):
                if self.last_valid_state is not None:
                    prediction = self.last_valid_state
                else:
                    prediction = np.array([self.frame_width/2, self.frame_height/2, 50, 0, 0, 0])
        except:
            if self.last_valid_state is not None:
                prediction = self.last_valid_state
            else:
                prediction = np.array([self.frame_width/2, self.frame_height/2, 50, 0, 0, 0])
        
        # 处理检测结果
        if circle is not None and circle['score'] > self.detection_confidence_threshold:
            self.detection_failure_count = 0
            center = circle['center']
            radius = circle['radius']
            contour = circle['contour']
            
            # 估计圆环颜色
            color = self.estimate_color(frame, contour)
            
            # 更新颜色历史
            filtered_color = self.filter_color(color)
            
            # 计算检测与预测之间的距离
            predicted_center = (int(prediction[0]), int(prediction[1]))
            # 确保预测中心在有效范围内
            predicted_center = (
                max(0, min(self.frame_width - 1, predicted_center[0])),
                max(0, min(self.frame_height - 1, predicted_center[1]))
            )
            
            dist = np.sqrt((center[0] - predicted_center[0])**2 + 
                          (center[1] - predicted_center[1])** 2)
            
            # 放宽更新条件
            max_dist_threshold = max(100, radius * 2)
            
            # 更新滤波器
            self.kf.update(np.array([center[0], center[1], radius]))
            self.last_valid_state = np.copy(self.kf.x)
            
            # 确保输出中心在图像范围内
            output_center = (
                max(0, min(self.frame_width - 1, int(self.kf.x[0]))),
                max(0, min(self.frame_height - 1, int(self.kf.x[1])))
            )
            
            circle_info = {
                'center': output_center,
                'radius': max(1, int(self.kf.x[2])),  # 确保半径有效
                'color': filtered_color,
                'contour': contour,
                'mask': mask,
                'settings': settings,
                'detection': circle,
                'status': 'detected'
            }
            self.last_detected = True
            self.last_color = filtered_color
            
        else:
            # 检测失败处理
            self.detection_failure_count += 1
            # 确保预测中心在有效范围内
            predicted_center = (
                max(0, min(self.frame_width - 1, int(prediction[0]))),
                max(0, min(self.frame_height - 1, int(prediction[1])))
            )
            predicted_radius = max(1, int(prediction[2]))
            
            # 确定使用的中心
            if self.last_valid_state is not None and self.detection_failure_count < self.max_failure_count:
                center = (
                    max(0, min(self.frame_width - 1, int(self.last_valid_state[0]))),
                    max(0, min(self.frame_height - 1, int(self.last_valid_state[1])))
                )
                radius = max(1, int(self.last_valid_state[2]))
                status = 'predicted'
            else:
                center = predicted_center
                radius = predicted_radius
                status = 'lost'
                if self.detection_failure_count >= self.max_failure_count:
                    print("检测失败次数过多，正在重新初始化...")
                    self.last_valid_state = None
            
            circle_info = {
                'center': center,
                'radius': radius,
                'color': self.last_color if self.last_detected else (0, 0, 0),
                'mask': mask,
                'settings': settings,
                'status': status
            }
            self.last_detected = False
        
        return circle_info

    def visualize_results(self, frame, circle_info):
        """可视化结果，增加边界检查避免越界绘制"""
        if circle_info is None or frame is None or frame.size == 0:
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 复制原始帧
        display_frame = frame.copy()
        settings = circle_info['settings']
        h, w = display_frame.shape[:2]
        
        # 显示帧率信息
        fps_text = f"FPS: {self.fps:.1f} (上限 {self.target_fps})"
        if 0 <= 30 < h and 0 <= 10 < w:  # 检查文本位置是否有效
            cv2.putText(display_frame, fps_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 准备预览图像
        preview_size = (max(100, w//5), max(75, h//5))  # 确保预览尺寸有效
        processed_previews = {}
        
        for name, img in self.processing_images.items():
            if img is None or img.size == 0:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 确保调整后的尺寸有效
            processed_previews[name] = cv2.resize(img, preview_size)
        
        # 预览图像布局
        preview_images = [
            ('original', '原图'),
            ('gray', '灰度'),
            ('close', '闭运算'),
            ('dilated', '最终掩膜')
        ]
        
        # 创建预览网格（检查尺寸避免越界）
        if processed_previews and preview_images:
            try:
                # 确保有足够的预览图
                valid_previews = []
                for name, title in preview_images:
                    if name in processed_previews:
                        valid_previews.append((name, title))
                
                if len(valid_previews) >= 4:
                    row1 = np.hstack([
                        processed_previews[valid_previews[0][0]],
                        processed_previews[valid_previews[1][0]]
                    ])
                    row2 = np.hstack([
                        processed_previews[valid_previews[2][0]],
                        processed_previews[valid_previews[3][0]]
                    ])
                    
                    preview_grid = np.vstack([row1, row2])
                    grid_h, grid_w = preview_grid.shape[:2]
                    
                    # 确保网格位置在图像范围内
                    if grid_h <= h and grid_w <= w:
                        display_frame[h-grid_h:h, w-grid_w:w] = preview_grid
                        
                        # 添加预览图标题
                        for i, (_, title) in enumerate(valid_previews):
                            x = w - grid_w + 10 if i % 2 == 0 else w - grid_w//2 + 10
                            y = h - grid_h + 20 if i < 2 else h - grid_h//2 + 20
                            # 检查文本位置是否有效
                            if 0 <= y < h and 0 <= x < w:
                                cv2.putText(display_frame, title, (x, y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception as e:
                if self.debug_mode:
                    print(f"预览图绘制错误: {e}")
        
        # 显示检测状态
        status_text = f"检测状态: {circle_info['status']}"
        if 0 <= h - 40 > 0 and 0 <= 10 < w:
            cv2.putText(display_frame, status_text, 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示摄像头信息
        cam_text = f"摄像头: {self.camera_id}"
        if 0 <= h - 10 > 0 and 0 <= 10 < w:
            cv2.putText(display_frame, cam_text, 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示阈值信息
        thresh_text = f"阈值: {self.min_thresh}-{self.max_thresh}"
        if 0 <= h - 90 > 0 and 0 <= 10 < w:
            cv2.putText(display_frame, thresh_text, 
                       (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)
        
        # 绘制圆环和圆心（确保在图像范围内）
        center = circle_info['center']
        radius = circle_info['radius']
        color = circle_info['color']
        
        # 确保半径有效
        radius = max(1, min(radius, max(h, w) // 2))
        
        # 选择标记颜色
        if circle_info['status'] == 'detected':
            marker_color = (0, 255, 0)  # 绿色：成功检测
        elif circle_info['status'] == 'predicted':
            marker_color = (0, 255, 255)  # 黄色：预测
        else:
            marker_color = (0, 0, 255)  # 红色：丢失
        
        # 绘制圆和中心标记（检查是否在图像范围内）
        if 0 <= center[0] < w and 0 <= center[1] < h:
            cv2.circle(display_frame, center, radius, color, 2)
            cv2.circle(display_frame, center, 5, marker_color, -1)
            
            # 显示中心坐标
            info_text = f"中心: ({center[0]}, {center[1]})"
            radius_text = f"半径: {radius}"
            
            if 0 <= 60 < h and 0 <= 10 < w:
                cv2.putText(display_frame, info_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
                cv2.putText(display_frame, radius_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
        
        # 调试模式显示更多信息
        if self.debug_mode and 'detection' in circle_info:
            circle = circle_info['detection']
            score_text = f"检测分数: {circle['score']:.2f}"
            circ_text = f"圆度: {circle['circularity']:.2f}"
            area_text = f"面积: {circle['area']:.0f}"
            
            if 0 <= 120 < h and 0 <= 10 < w:
                cv2.putText(display_frame, score_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, circ_text, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, area_text, (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return display_frame

    def filter_color(self, color):
        """过滤颜色，去除异常值"""
        self.color_history.append(color)
        if len(self.color_history) > 10:
            self.color_history.pop(0)
        
        # 计算平均值
        if len(self.color_history) > 0:
            avg_color = np.mean(self.color_history, axis=0).astype(int)
            return tuple(avg_color)
        return color

    def estimate_color(self, frame, contour):
        """从轮廓区域估计颜色"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        mean_val = cv2.mean(frame, mask=mask)
        return (int(mean_val[2]), int(mean_val[1]), int(mean_val[0]))  # BGR转RGB

    def run(self):
        """运行圆环检测程序主循环"""
        print(f"圆环检测程序已启动（使用自定义预处理流程）")
        print(f"帧率上限设置为 {self.target_fps} 帧/秒")
        print("控制键:")
        print("  q: 退出程序")
        print("  d: 切换调试模式")
        print("  s: 保存当前参数到文件")
        print("  数字键0-9: 切换到对应ID的摄像头")
        
        while True:
            # 控制帧率
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            # 确保帧率不超过目标值
            if elapsed < 1.0 / self.target_fps:
                time.sleep((1.0 / self.target_fps) - elapsed)
            
            # 更新帧率计算
            self.fps = 1.0 / (time.time() - self.last_frame_time)
            self.last_frame_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print(f"无法从摄像头 {self.camera_id} 获取帧，尝试切换摄像头...")
                current_idx = self.available_cameras.index(self.camera_id) if self.camera_id in self.available_cameras else 0
                next_idx = (current_idx + 1) % len(self.available_cameras) if self.available_cameras else 0
                if not self.switch_camera(self.available_cameras[next_idx] if self.available_cameras else 0):
                    break
                continue
                
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            self.processing_images['original'] = frame.copy()
            
            # 处理当前帧
            circle_info = self.process_frame(frame)
            
            # 可视化结果
            display_frame = self.visualize_results(frame, circle_info)
            
            # 显示结果
            cv2.imshow('Circle Detection (Custom Preprocessing)', display_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = 1 - self.debug_mode
                print(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
            elif key == ord('s'):
                if circle_info:
                    settings = circle_info['settings']
                    self.save_settings(settings)
                    print("参数已保存到 settings.txt")
            elif ord('0') <= key <= ord('9'):
                cam_id = int(chr(key))
                self.switch_camera(cam_id)
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

    def save_settings(self, settings):
        """保存当前参数到文件"""
        with open('custom_preprocess_settings.txt', 'w') as f:
            f.write("# 自定义预处理参数配置\n")
            f.write(f"MIN_THRESH={self.min_thresh}\n")
            f.write(f"MAX_THRESH={self.max_thresh}\n")
            f.write(f"MIN_AREA={settings['min_area']}\n")
            f.write(f"MAX_AREA={settings['max_area']}\n")
            f.write(f"MIN_CIRCULARITY={settings['min_circularity']}\n")
            f.write(f"MIN_CONVEXITY={settings['min_convexity']}\n")
            f.write(f"KF_PROC_NOISE={settings['kf_proc_noise']}\n")
            f.write(f"KF_MEAS_NOISE={settings['kf_meas_noise']}\n")
            f.write(f"DETECT_SENSITIVITY={settings['detect_sensitivity']}\n")

if __name__ == "__main__":
    try:
        detector = CircleDetector()
        detector.run()
    except Exception as e:
        print(f"程序运行错误: {e}")