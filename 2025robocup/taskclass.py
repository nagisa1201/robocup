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
        self.min_thresh = 50
        self.max_thresh = 166

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
        iteration = 21
        closeMat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element, iterations=iteration)
        calcMat = cv2.bitwise_not(cv2.subtract(closeMat, gray))
        removeShadowMat = cv2.normalize(calcMat, None, 0, 200, cv2.NORM_MINMAX)
        bilateral_blur2=cv2.bilateralFilter(removeShadowMat,9,75,75)
        
        bilateral_blur3=cv2.bilateralFilter(bilateral_blur2,9,75,75)
        _, thresh = cv2.threshold(bilateral_blur3, self.min_thresh, self.max_thresh, cv2.THRESH_BINARY_INV)
        kernel1 = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        kernel2 = np.ones((5, 5), np.uint8)
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
    def __init__(self, camera_frame):
        self.camera_frame = camera_frame
        self.isdetected = False  # 最终判断圆环是否被检测到的标志位
        self.frame_width = camera_frame.shape[1]
        self.frame_height = camera_frame.shape[0]




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

        decoded_info, points = qr_task.detect_and_decode()
        frame = qr_task.draw_qrcode_info(decoded_info, points)
        print(f"检测到二维码: {qr_task.code_data}")
        cv2.imshow("QR Code Detection", frame)
        cv2.imshow("Processed Frame", process_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()