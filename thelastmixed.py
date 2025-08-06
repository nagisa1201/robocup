import cv2
import numpy as np
import math
import serial
import pyzbar.pyzbar as pyzbar
import time
import struct
from filterpy.kalman import KalmanFilter
# import easyocr

# reader = easyocr.Reader(['en'])

width = 640
height = 480
# # 定义状态转移矩阵 A，观测矩阵 H，过程噪声协方差 Q，测量噪声协方差 R，初始状态协方差 P
last_valid_angle = 0  # 初始化为默认值或其他有效值

kf = KalmanFilter(dim_x=1, dim_z=1)
kf.F = np.array([[1]])    # State transition matrix
kf.H = np.array([[1]])    # Measurement function
kf.Q = np.array([[1]])    # Process uncertainty
kf.R = np.array([[10]])   # Measurement uncertainty
kf.x = np.array([last_valid_angle])  # 初始化卡尔曼滤波器状态
kf.P = np.array([[1]])    # Initial covariance estimate
##########任务二函数##########
def correct_rotation(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_characters(frame):
    detected_text = []
    preprocessed_frame = preprocess_image(frame)
    
    for angle in range(0, 360, 25):
        rotated_frame = correct_rotation(preprocessed_frame, angle)
        results = reader.readtext(rotated_frame)

        for result in results:
            text = result[1]
            if text in ['A', 'B', 'C']:
                detected_text.append(text)
                print(f"Detected: {text}")
                points = result[0]
                if len(points) >= 4:
                    x_coords = [point[0] for point in points]
                    y_coords = [point[1] for point in points]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    print(f"Detected at {angle} degrees: {text}")
    return detected_text
############任务二函数结束############
# def send_data(ser,Head,x:float,y:float,color:float,tail):
#     data_to_send = bytearray()

#     # 添加header
#     data_to_send.extend(struct.pack('<B', Head))

#     # 添加每个one_place结构的数据
#     # for place in places:
#     data_to_send.extend(struct.pack('<f', x))  # x
#     data_to_send.extend(struct.pack('<f', y))  # y
#     data_to_send.extend(struct.pack('<f', color))  # color

#     # 添加tail
#     data_to_send.extend(struct.pack('<B', tail))

#     # # 通过串口发送数据
#     ser.write(data_to_send)
# # 串口配置
# SERIAL_PORT = '/dev/ttyS6'  # 替换为你的串口设备
# BAUD_RATE = 115200
# TIMEOUT = 1

# # 初始化串口
# ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# 参数
min_area = 2500
max_area = 20000
min_thresh = 80
max_thresh = 255

def callback(value):
    pass

def get_aligned_images(pipeline, align, profile):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    color_sensor = profile.get_device().query_sensors()[1]
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))
    color_image = np.asanyarray(color_frame.get_data())
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
    return rotated_image

def cross_angle(frame):
    global min_thresh, max_thresh
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    medianBlur = cv2.medianBlur(blurred_frame, 5)
    gray = cv2.cvtColor(medianBlur, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iteration = 21
    closeMat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element, iterations=iteration)
    calcMat = cv2.bitwise_not(cv2.subtract(closeMat, gray))
    removeShadowMat = cv2.normalize(calcMat, None, 0, 200, cv2.NORM_MINMAX)
    bilateral_blur2=cv2.bilateralFilter(removeShadowMat,9,75,75)
    
    bilateral_blur3=cv2.bilateralFilter(bilateral_blur2,9,75,75)
    _, thresh = cv2.threshold(bilateral_blur3, min_thresh, max_thresh, cv2.THRESH_BINARY_INV)
    kernel1 = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(thresh, kernel1, iterations=1)
    kernel2 = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(eroded, kernel2, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    center_x = width // 2 + 6
    center_y = height // 2 + 20
    offset_x = None
    offset_y = None
    best_line = None
    min_angle = float('inf')
    best_angle = 0
    best_y = -float('inf')  # 初始化为最小值

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 12:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if cY > best_y:  # 选择 y 值更大的那个
                        best_y = cY
                        offset_x = cX - center_x
                        offset_y = cY - center_y
                        rect = cv2.minAreaRect(approx)
                        angle = rect[2]
                        if angle < -45:
                            angle += 90
                        elif angle > 45:
                            angle -= 90
                        min_angle = float('inf')
                        best_angle = angle
                        best_line = None
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                        lines = []
                        for i in range(4):
                            p1 = tuple(box[i])
                            p2 = tuple(box[(i + 2) % 4])
                            lines.append((p1, p2))
                            cv2.line(frame, p1, p2, (255, 0, 0), 2)
                        vertical_line_x1 = center_x
                        vertical_line_y1 = 0
                        vertical_line_x2 = center_x
                        vertical_line_y2 = height
                        cv2.line(frame, (vertical_line_x1, vertical_line_y1), (vertical_line_x2, vertical_line_y2), (0, 255, 255), 2)
                        for p1, p2 in lines:
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            line_angle = math.atan2(dy, dx) * 180 / np.pi
                            if line_angle > 0:
                                line_angle -= 180
                            vertical_line_angle = 90
                            angle_diff = abs(vertical_line_angle - line_angle)
                            angle_diff = min(angle_diff, 180 - angle_diff)
                            if angle_diff < min_angle:
                                min_angle = angle_diff
                                best_line = (p1, p2)
                    
    rotated_frame = rotate_image(frame, best_angle)
    if best_line:
        cv2.line(rotated_frame, best_line[0], best_line[1], (0, 0, 255), 2)
    if(min_angle<-45):
        return offset_x, offset_y, 90+min_angle, thresh,removeShadowMat
    return offset_x, offset_y, -min_angle, thresh,removeShadowMat


def barcode_reader(frame):
    previous_data = ['_'] 
    QRcode = pyzbar.decode(frame)
    QR_data=None
    for obj in QRcode:   
        QR_data = obj.data.decode('utf-8')
#每次有新的二维码只给information赋值一次，并打印一次，避免重复打印
        if QR_data != previous_data:
            previous_data = QR_data        
        point = obj.rect
        cv2.rectangle(frame, (point.left, point.top), (point.left + point.width, point.top + point.height), (0, 255, 0), 5)
    return QR_data

def main():
    crossflag=True
    QRflag=False
    anpfalg=False
    global last_valid_angle  # 确保能够更新全局变量
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        _,frame=cap.read()
        # if ser.in_waiting > 0:
        #     data = ser.read().decode('utf-8').strip()
            # print(data)
            # if data == '1':
            #     crossflag=True
            #     QRflag=False
            #     anpfalg=False
            # elif data == '2':
            #     crossflag=False
            #     QRflag=True
            #     anpfalg=False
            # elif data == '3':
            #     crossflag=False
            #     QRflag=False
            #     anpfalg=True
        # if crossflag==True:
        #     offset_x,offset_y,min_angle,thresh,removeShadowMat = cross_angle(frame)
        #     if np.isfinite(min_angle):  # 如果 min_angle 有效
        #         kf.predict()
        #         kf.update(np.array([min_angle]))
        #         last_valid_angle = min_angle  # 更新最后的有效值
        #         filtered_angle = kf.x
        #         print(offset_x,offset_y,min_angle,filtered_angle)
        #     else:
        #         kf.predict()  # 只进行预测，不更新状态
        #     if offset_x is not None:
        #         send_data(ser, 0xFA, offset_x, filtered_angle, offset_y, 0xAF)
        # if QRflag==True:
        #     QRcode=None
        #     QRcode=barcode_reader(frame)
        #     print(QRcode)
            
        #     if QRcode!=None:
        #         QRcode=float(QRcode)
        #         send_data(ser, 0xFB, 0, 0, QRcode, 0xBF)
        # if anpfalg==True:
        #     detected_text = detect_characters(frame)
        #     if detected_text:
        #         print(f"Detected text: {detected_text}")
        #         for text in detected_text:
        #             if text == 'A':
        #                 send_data(ser, 0xFC, 0, 0, 0, 0xCF)
        #             elif text == 'B':
        #                 send_data(ser, 0xFC, 0, 0, 1, 0xCF)
        #             elif text == 'C':
        #                 send_data(ser, 0xFC, 0, 0, 2, 0xCF)
        # cv2.imshow('output', frame)
        # if thresh is not None:
        #     cv2.imshow('thresh',thresh)
        #     cv2.imshow('removeShadowMat',removeShadowMat)
        cv2.imshow('output', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
