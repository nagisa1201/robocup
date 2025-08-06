import cv2
import numpy as np

class QRCodeDetector:
    def __init__(self, camera_id=0):
        # 初始化相机
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("无法打开相机")
        
        # 初始化二维码检测器
        self.qr_detector = cv2.QRCodeDetector()
        
        # 显示参数
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (0, 255, 0)
        self.line_type = 2
        
    def detect_and_decode(self, frame):
        """检测并解码二维码"""
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测二维码
        retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)
        
        return retval, decoded_info, points
    
    def draw_qrcode_info(self, frame, decoded_info, points):
        """在图像上绘制二维码信息"""
        for i in range(len(decoded_info)):
            if decoded_info[i]:
                # 获取二维码的四个角点
                pts = points[i].astype(np.int32)
                
                # 绘制边界框
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                
                # 获取文本位置（二维码上方）
                text_pos = (pts[0][0], pts[0][1] - 10)
                
                # 显示解码信息
                cv2.putText(frame, decoded_info[i], text_pos, 
                           self.font, self.font_scale, self.font_color, self.line_type)
                
                # 显示二维码数量
                cv2.putText(frame, f"QR Codes: {len(decoded_info)}", (10, 30), 
                           self.font, self.font_scale, self.font_color, self.line_type)
        
        return frame
    
    def run(self):
        """运行二维码检测程序"""
        print("二维码检测程序已启动")
        print("控制键:")
        print("  q: 退出程序")
        print("  s: 保存当前帧及检测结果")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 镜像翻转，使显示更直观
            frame = cv2.flip(frame, 1)
            
            # 检测并解码二维码
            retval, decoded_info, points = self.detect_and_decode(frame)
            
            # 如果检测到二维码，绘制信息
            if retval:
                frame = self.draw_qrcode_info(frame, decoded_info, points)
            
            # 显示结果
            cv2.imshow('QR Code Detection', frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and retval:
                # 保存当前帧及检测结果
                cv2.imwrite('qrcode_result.png', frame)
                print("已保存结果到 qrcode_result.png")
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建并运行检测器
    detector = QRCodeDetector(camera_id=0)
    detector.run()    