import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import time  # 用于计算实际帧率

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
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 设置目标帧率
        self.target_fps = 30
        self.last_frame_time = time.time()  # 用于计算实际帧率
        self.fps = 0  # 实际帧率
        
        # 获取相机帧尺寸
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        
        # 创建参数调节窗口
        cv2.namedWindow('Settings')
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
        self.min_contour_area = 500
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
            
            self.frame_width = int(self.cap.get(3))
            self.frame_height = int(self.cap.get(4))
            self.kf.x = np.array([self.frame_width/2, self.frame_height/2, 50, 0, 0, 0])
            self.last_valid_state = None
            self.detection_failure_count = 0
            
            print(f"已切换到摄像头 {camera_id}，帧率上限保持为 {self.target_fps}")
            return True
        return False
    
    def create_trackbars(self):
        cv2.createTrackbar('THRESH_METHOD', 'Settings', 0, 3, lambda x: None)
        cv2.createTrackbar('H_MIN', 'Settings', 90, 179, lambda x: None)
        cv2.createTrackbar('H_MAX', 'Settings', 130, 179, lambda x: None)
        cv2.createTrackbar('S_MIN', 'Settings', 100, 255, lambda x: None)
        cv2.createTrackbar('S_MAX', 'Settings', 255, 255, lambda x: None)
        cv2.createTrackbar('V_MIN', 'Settings', 100, 255, lambda x: None)
        cv2.createTrackbar('V_MAX', 'Settings', 255, 255, lambda x: None)
        cv2.createTrackbar('ADAPT_BLOCK', 'Settings', 15, 50, lambda x: None)
        cv2.createTrackbar('ADAPT_C', 'Settings', 2, 20, lambda x: None)
        cv2.createTrackbar('KERNEL_SIZE', 'Settings', 3, 10, lambda x: None)
        cv2.createTrackbar('OPEN_ITER', 'Settings', 1, 5, lambda x: None)
        cv2.createTrackbar('CLOSE_ITER', 'Settings', 2, 5, lambda x: None)
        cv2.createTrackbar('MIN_AREA', 'Settings', 1000, 10000, lambda x: None)
        cv2.createTrackbar('MAX_AREA', 'Settings', 50000, 100000, lambda x: None)
        cv2.createTrackbar('MIN_CIRCULARITY', 'Settings', 50, 100, lambda x: None)
        cv2.createTrackbar('MIN_CONVEXITY', 'Settings', 60, 100, lambda x: None)
        cv2.createTrackbar('KF_PROC_NOISE', 'Settings', 10, 100, lambda x: None)
        cv2.createTrackbar('KF_MEAS_NOISE', 'Settings', 20, 100, lambda x: None)
        cv2.createTrackbar('DETECT_SENSITIVITY', 'Settings', 50, 100, lambda x: None)

    def get_current_settings(self):
        thresh_method = cv2.getTrackbarPos('THRESH_METHOD', 'Settings')
        h_min = cv2.getTrackbarPos('H_MIN', 'Settings')
        h_max = cv2.getTrackbarPos('H_MAX', 'Settings')
        s_min = cv2.getTrackbarPos('S_MIN', 'Settings')
        s_max = cv2.getTrackbarPos('S_MAX', 'Settings')
        v_min = cv2.getTrackbarPos('V_MIN', 'Settings')
        v_max = cv2.getTrackbarPos('V_MAX', 'Settings')
        
        adapt_block = cv2.getTrackbarPos('ADAPT_BLOCK', 'Settings')
        adapt_block = adapt_block if adapt_block % 2 == 1 else adapt_block + 1
        adapt_c = cv2.getTrackbarPos('ADAPT_C', 'Settings')
        
        kernel_size = cv2.getTrackbarPos('KERNEL_SIZE', 'Settings')
        kernel_size = max(1, kernel_size) if kernel_size % 2 == 1 else kernel_size + 1
        open_iter = max(1, cv2.getTrackbarPos('OPEN_ITER', 'Settings'))
        close_iter = max(1, cv2.getTrackbarPos('CLOSE_ITER', 'Settings'))
        
        min_area = cv2.getTrackbarPos('MIN_AREA', 'Settings')
        max_area = max(min_area + 1000, cv2.getTrackbarPos('MAX_AREA', 'Settings'))
        min_circularity = cv2.getTrackbarPos('MIN_CIRCULARITY', 'Settings') / 100.0
        min_convexity = cv2.getTrackbarPos('MIN_CONVEXITY', 'Settings') / 100.0
        
        kf_proc_noise = cv2.getTrackbarPos('KF_PROC_NOISE', 'Settings') / 10.0
        kf_meas_noise = cv2.getTrackbarPos('KF_MEAS_NOISE', 'Settings') / 10.0
        detect_sensitivity = cv2.getTrackbarPos('DETECT_SENSITIVITY', 'Settings') / 100.0
        
        return {
            'thresh_method': thresh_method,
            'color_lower': np.array([h_min, s_min, v_min], dtype=np.uint8),
            'color_upper': np.array([h_max, s_max, v_max], dtype=np.uint8),
            'adapt_block': adapt_block,
            'adapt_c': adapt_c,
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

    def advanced_thresholding(self, frame, settings):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        self.processing_images['hsv'] = hsv
        self.processing_images['gray'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.processing_images['y_channel'] = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        
        if settings['thresh_method'] == 0:
            mask = cv2.inRange(hsv, settings['color_lower'], settings['color_upper'])
        elif settings['thresh_method'] == 1:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.processing_images['blurred'] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        elif settings['thresh_method'] == 2:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            mask = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                settings['adapt_block'], 
                settings['adapt_c']
            )
            self.processing_images['blurred'] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        else:
            hsv_mask = cv2.inRange(hsv, settings['color_lower'], settings['color_upper'])
            _, brightness_mask = cv2.threshold(
                y_channel, 
                settings['color_lower'][2], 
                255, 
                cv2.THRESH_BINARY
            )
            mask = cv2.bitwise_and(hsv_mask, brightness_mask)
            self.processing_images['brightness_mask'] = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)
            self.processing_images['hsv_mask'] = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        
        self.processing_images['mask_before'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        kernel = np.ones((settings['kernel_size'], settings['kernel_size']), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=settings['open_iter'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=settings['close_iter'])
        
        self.processing_images['mask_after'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        return mask

    def detect_circle(self, frame):
        settings = self.get_current_settings()
        self.processing_images['original'] = frame.copy()
        mask = self.advanced_thresholding(frame, settings)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = frame.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
        self.processing_images['contours'] = contour_img
        
        best_circle = None
        max_score = 0
        center_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < settings['min_area'] or area > settings['max_area']:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            if circularity < settings['min_circularity'] or convexity < settings['min_convexity']:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)
                center_candidates.append(center)
            else:
                center = (int(x), int(y))
            
            score = 0.8 * circularity + 0.2 * convexity
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
        
        if self.debug_mode and center_candidates:
            temp_img = self.processing_images['contours'].copy()
            for (cx, cy) in center_candidates:
                cv2.circle(temp_img, (cx, cy), 3, (255, 0, 0), -1)
            self.processing_images['candidates'] = temp_img
        
        return best_circle, mask, settings

    def process_frame(self, frame):
        circle, mask, settings = self.detect_circle(frame)
        
        self.kf.Q = np.eye(6) * settings['kf_proc_noise']
        self.kf.R = np.array([
            [settings['kf_meas_noise'], 0, 0],
            [0, settings['kf_meas_noise'], 0],
            [0, 0, settings['kf_meas_noise'] * 50]
        ])
        
        circle_info = None
        
        # 修复卡尔曼滤波预测可能返回None的问题
        try:
            prediction = self.kf.predict()
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
        
        if circle is not None and circle['score'] > self.detection_confidence_threshold:
            self.detection_failure_count = 0
            center = circle['center']
            radius = circle['radius']
            contour = circle['contour']
            
            color = self.estimate_color(frame, contour)
            filtered_color = self.filter_color(color)
            
            predicted_center = (int(prediction[0]), int(prediction[1]))
            dist = np.sqrt((center[0] - predicted_center[0])**2 + 
                          (center[1] - predicted_center[1])** 2)
            
            max_dist_threshold = max(100, radius * 2)
            self.kf.update(np.array([center[0], center[1], radius]))
            self.last_valid_state = np.copy(self.kf.x)
            
            circle_info = {
                'center': (int(self.kf.x[0]), int(self.kf.x[1])),
                'radius': int(self.kf.x[2]),
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
            self.detection_failure_count += 1
            predicted_center = (int(prediction[0]), int(prediction[1]))
            predicted_radius = int(prediction[2])
            
            if self.last_valid_state is not None and self.detection_failure_count < self.max_failure_count:
                center = (int(self.last_valid_state[0]), int(self.last_valid_state[1]))
                radius = int(self.last_valid_state[2])
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

    def estimate_color(self, frame, contour):
        mask = np.zeros_like(frame[:,:,0])
        cv2.drawContours(mask, [contour], -1, 255, -1)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        mean_color = cv2.mean(masked, mask=mask)[:3]
        return tuple([int(c) for c in mean_color])

    def filter_color(self, new_color):
        self.color_history.append(new_color)
        if len(self.color_history) > 10:
            self.color_history.pop(0)
        avg_color = np.mean(self.color_history, axis=0)
        return tuple([int(c) for c in avg_color])

    def visualize_results(self, frame, circle_info):
        if circle_info is None:
            return frame
        
        display_frame = frame.copy()
        settings = circle_info['settings']
        h, w = display_frame.shape[:2]
        
        # 显示帧率信息
        cv2.putText(display_frame, f"FPS: {self.fps:.1f} (上限 {self.target_fps})", 
                   (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        preview_size = (w//5, h//5)
        processed_previews = {}
        
        for name, img in self.processing_images.items():
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            processed_previews[name] = cv2.resize(img, preview_size)
        
        method = settings['thresh_method']
        method_names = ["HSV阈值", "Otsu自动阈值", "自适应阈值", "组合阈值"]
        
        if method == 0:
            preview_images = [
                ('original', '原图'),
                ('hsv', 'HSV'),
                ('mask_before', '掩码(前)'),
                ('mask_after', '掩码(后)')
            ]
        elif method == 1:
            preview_images = [
                ('original', '原图'),
                ('gray', '灰度'),
                ('blurred', '模糊'),
                ('mask_after', '二值化结果')
            ]
        elif method == 2:
            preview_images = [
                ('original', '原图'),
                ('gray', '灰度'),
                ('blurred', '模糊'),
                ('mask_after', '二值化结果')
            ]
        else:
            preview_images = [
                ('original', '原图'),
                ('hsv_mask', 'HSV掩码'),
                ('brightness_mask', '亮度掩码'),
                ('mask_after', '组合结果')
            ]
        
        if processed_previews and preview_images:
            try:
                row1 = np.hstack([
                    processed_previews.get(preview_images[0][0], np.zeros((preview_size[1], preview_size[0], 3), dtype=np.uint8)),
                    processed_previews.get(preview_images[1][0], np.zeros((preview_size[1], preview_size[0], 3), dtype=np.uint8))
                ])
                
                row2 = np.hstack([
                    processed_previews.get(preview_images[2][0], np.zeros((preview_size[1], preview_size[0], 3), dtype=np.uint8)),
                    processed_previews.get(preview_images[3][0], np.zeros((preview_size[1], preview_size[0], 3), dtype=np.uint8))
                ])
                
                preview_grid = np.vstack([row1, row2])
                grid_h, grid_w = preview_grid.shape[:2]
                display_frame[h-grid_h:h, w-grid_w:w] = preview_grid
                
                for i, (_, title) in enumerate(preview_images):
                    x = w - grid_w + 10 if i % 2 == 0 else w - grid_w//2 + 10
                    y = h - grid_h + 20 if i < 2 else h - grid_h//2 + 20
                    cv2.putText(display_frame, title, (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except:
                pass
        
        cv2.putText(display_frame, f"二值化方法: {method_names[method]}", 
                   (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display_frame, f"检测状态: {circle_info['status']}", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display_frame, f"摄像头: {self.camera_id}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        center = circle_info['center']
        radius = circle_info['radius']
        color = circle_info['color']
        
        if circle_info['status'] == 'detected':
            marker_color = (0, 255, 0)
        elif circle_info['status'] == 'predicted':
            marker_color = (0, 255, 255)
        else:
            marker_color = (0, 0, 255)
        
        cv2.circle(display_frame, center, radius, color, 2)
        cv2.circle(display_frame, center, 5, marker_color, -1)
        
        info_text = f"中心: ({center[0]}, {center[1]})"
        radius_text = f"半径: {radius}"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
        cv2.putText(display_frame, radius_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2)
        
        if self.debug_mode and 'detection' in circle_info:
            circle = circle_info['detection']
            score_text = f"检测分数: {circle['score']:.2f}"
            circ_text = f"圆度: {circle['circularity']:.2f}"
            area_text = f"面积: {circle['area']:.0f}"
            
            cv2.putText(display_frame, score_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, circ_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, area_text, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return display_frame

    def run(self):
        """运行圆环检测程序主循环，限定帧率为30"""
        print(f"圆环检测程序已启动，帧率上限设置为 {self.target_fps} 帧/秒")
        print("控制键:")
        print("  q: 退出程序")
        print("  d: 切换调试模式")
        print("  s: 保存当前参数到文件")
        print("  数字键0-9: 切换到对应ID的摄像头")
        
        while True:
            # 计算时间差，控制帧率不超过30
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            # 如果时间间隔不足（超过目标帧率），等待
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
                
            frame = cv2.flip(frame, 1)
            circle_info = self.process_frame(frame)
            display_frame = self.visualize_results(frame, circle_info)
            
            cv2.imshow('Circle Detection', display_frame)
            
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
        
        self.cap.release()
        cv2.destroyAllWindows()

    def save_settings(self, settings):
        with open('settings.txt', 'w') as f:
            f.write("# 圆环检测参数配置\n")
            f.write(f"THRESH_METHOD={settings['thresh_method']}\n")
            f.write(f"H_MIN={settings['color_lower'][0]}\n")
            f.write(f"H_MAX={settings['color_upper'][0]}\n")
            f.write(f"S_MIN={settings['color_lower'][1]}\n")
            f.write(f"S_MAX={settings['color_upper'][1]}\n")
            f.write(f"V_MIN={settings['color_lower'][2]}\n")
            f.write(f"V_MAX={settings['color_upper'][2]}\n")
            f.write(f"ADAPT_BLOCK={settings['adapt_block']}\n")
            f.write(f"ADAPT_C={settings['adapt_c']}\n")
            f.write(f"KERNEL_SIZE={settings['kernel_size']}\n")
            f.write(f"OPEN_ITER={settings['open_iter']}\n")
            f.write(f"CLOSE_ITER={settings['close_iter']}\n")
            f.write(f"MIN_AREA={settings['min_area']}\n")
            f.write(f"MAX_AREA={settings['max_area']}\n")
            f.write(f"MIN_CIRCULARITY={settings['min_circularity']}\n")
            f.write(f"MIN_CONVEXITY={settings['min_convexity']}\n")
            f.write(f"KF_PROC_NOISE={settings['kf_proc_noise']}\n")
            f.write(f"KF_MEAS_NOISE={settings['kf_meas_noise']}\n")
            f.write(f"DETECT_SENSITIVITY={settings['detect_sensitivity']}\n")

if __name__ == "__main__":
    detector = CircleDetector()
    detector.run()