# utils/yolo_detector.py
import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, device='cuda'):
        print(f"加载YOLO模型: {model_path}，设备: {device}")

        # 定义需要处理的类别
        self.target_classes = ['bottle']  # 只处理药瓶
        self.class_names = None

        try:
            self.detector = YOLO(model_path)
            # 设置设备
            if device and device != 'cpu':
                if device.isdigit():
                    device = f'cuda:{device}'
                try:
                    self.detector.to(device)
                    print(f"YOLO模型已加载到设备: {device}")
                except Exception as e:
                    print(f"无法将模型加载到设备 {device}: {e}，使用默认设备")

            # 获取类别名称
            if hasattr(self.detector, 'names'):
                self.class_names = self.detector.names
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"YOLO模型加载失败: {str(e)}")
            raise

    def detect(self, frame):
        try:
            # 启用详细输出以显示标签
            results = self.detector(frame, verbose=False)
            if not results:
                return None, frame

            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])

            # 过滤目标类别
            filtered_boxes = []
            if result.boxes is not None and result.boxes.cls is not None and self.class_names is not None:
                classes = result.boxes.cls.cpu().numpy()
                for i, class_id in enumerate(classes):
                    class_name = self.class_names.get(int(class_id), f"class_{int(class_id)}")
                    if class_name in self.target_classes:
                        filtered_boxes.append(boxes[i])

            # 显示标签和置信度
            plotted_frame = result.plot(labels=True, conf=True)
            return np.array(filtered_boxes), plotted_frame
        except Exception as e:
            print(f"目标检测过程中发生错误: {str(e)}")
            return None, frame
