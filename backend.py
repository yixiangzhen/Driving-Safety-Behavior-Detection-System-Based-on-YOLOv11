import cv2
from ultralytics import YOLO


class DriverBehaviorDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cpu")  # 强制用CPU
        self.class_names = [
            "awake",
            "distracted",
            "drowsy",
            "head drop",
            "phone",
            "smoking",
            "yawn",
        ]

        self.class_conf = {
            "awake": 0.10,
            "distracted": 0.35,
            "drowsy": 0.45,
            "head drop": 0.35,
            "phone": 0.25,
            "smoking": 0.35,
            "yawn": 0.35,
        }

        self.frame_counters = {
            "distracted": 0,
            "drowsy": 0,
            "head drop": 0,
            "phone": 0,
            "smoking": 0,
            "yawn": 0,
        }

        self.thresholds = {
            "distracted": 15,
            "drowsy": 20,
            "head drop": 8,
            "phone": 12,
            "smoking": 12,
            "yawn": 10,
        }

        self.enabled_classes = set(self.class_names)
        self.imgsz = 480
        self.device = 0
        self.half = True
        self.show_boxes = True
        self.show_conf = True

    def reset_state(self):
        for key in self.frame_counters:
            self.frame_counters[key] = 0

    def set_enabled_class(self, class_name, enabled):
        if enabled:
            self.enabled_classes.add(class_name)
        else:
            self.enabled_classes.discard(class_name)
            if class_name in self.frame_counters:
                self.frame_counters[class_name] = 0

    def set_class_conf(self, class_name, value):
        if class_name in self.class_conf:
            self.class_conf[class_name] = float(value)

    def set_sequence_threshold(self, class_name, value):
        if class_name in self.thresholds:
            self.thresholds[class_name] = max(1, int(float(value)))

    def set_preset(self, preset):
        presets = {
            "灵敏模式": {
                "awake": 0.08,
                "distracted": 0.25,
                "drowsy": 0.35,
                "head drop": 0.25,
                "phone": 0.18,
                "smoking": 0.22,
                "yawn": 0.25,
            },
            "平衡模式": {
                "awake": 0.10,
                "distracted": 0.35,
                "drowsy": 0.45,
                "head drop": 0.35,
                "phone": 0.25,
                "smoking": 0.35,
                "yawn": 0.35,
            },
            "严格模式": {
                "awake": 0.18,
                "distracted": 0.50,
                "drowsy": 0.58,
                "head drop": 0.50,
                "phone": 0.38,
                "smoking": 0.48,
                "yawn": 0.50,
            },
        }
        if preset in presets:
            self.class_conf.update(presets[preset])

    def detect_frame(self, frame, use_sequence=True, show_fps_text=None):
        min_conf = min(self.class_conf.values()) if self.class_conf else 0.1
        min_conf = max(0.01, min_conf)

        try:
            results = self.model(
                frame,
                conf=min_conf,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception:
            results = self.model(frame, conf=min_conf, imgsz=self.imgsz, verbose=False)

        result = results[0]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            if class_name not in self.enabled_classes:
                continue
            if conf < self.class_conf.get(class_name, 0.35):
                continue

            item = {
                "class_name": class_name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            }
            detections.append(item)

            if self.show_boxes:
                color = self.get_color(class_name)
                label = class_name if not self.show_conf else f"{class_name} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                text_w, text_h = text_size
                top_y = max(0, y1 - text_h - 10)
                cv2.rectangle(frame, (x1, top_y), (x1 + text_w, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(22, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 0),
                    2,
                )

        if show_fps_text:
            cv2.putText(
                frame,
                show_fps_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 80, 0),
                2,
            )

        if use_sequence:
            state, is_alert = self.update_sequence_state(detections)
        else:
            state, is_alert = self.get_image_state(detections)

        return frame, detections, state, is_alert

    def get_image_state(self, detections):
        detected_classes = {item["class_name"] for item in detections}

        if "head drop" in detected_classes:
            return "疲劳驾驶：检测到点头/低头", True
        if "phone" in detected_classes and "distracted" in detected_classes:
            return "分心驾驶：侧头看手机", True
        if "phone" in detected_classes:
            return "分心驾驶：使用手机", True
        if "smoking" in detected_classes:
            return "危险驾驶：抽烟", True
        if "yawn" in detected_classes:
            return "疲劳风险：打哈欠", True
        if "drowsy" in detected_classes:
            return "疲劳驾驶：困倦", True
        if "distracted" in detected_classes:
            return "分心驾驶：注意力偏移", True
        if "awake" in detected_classes:
            return "正常驾驶：清醒", False

        return "未检测到驾驶状态", False

    def update_sequence_state(self, detections):
        detected_classes = {item["class_name"] for item in detections}

        for cls in self.frame_counters:
            if cls in detected_classes:
                self.frame_counters[cls] += 1
            else:
                self.frame_counters[cls] = max(0, self.frame_counters[cls] - 1)

        phone_alert = self.frame_counters["phone"] >= self.thresholds["phone"]
        distracted_alert = self.frame_counters["distracted"] >= self.thresholds["distracted"]

        if self.frame_counters["head drop"] >= self.thresholds["head drop"]:
            return "疲劳驾驶：检测到点头/低头", True
        if phone_alert and distracted_alert:
            return "分心驾驶：侧头看手机", True
        if phone_alert:
            return "分心驾驶：使用手机", True
        if self.frame_counters["smoking"] >= self.thresholds["smoking"]:
            return "危险驾驶：抽烟", True
        if self.frame_counters["yawn"] >= self.thresholds["yawn"]:
            return "疲劳风险：打哈欠", True
        if self.frame_counters["drowsy"] >= self.thresholds["drowsy"]:
            return "疲劳驾驶：困倦", True
        if distracted_alert:
            return "分心驾驶：注意力偏移", True
        if "awake" in detected_classes:
            return "正常驾驶：清醒", False

        return "未稳定识别状态", False

    def detect_image(self, image_path):
        self.reset_state()
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("图片路径无效，无法读取图片")
        return self.detect_frame(frame, use_sequence=False)

    def detect_video(self, video_path, output_path, should_stop=None, progress_callback=None, alert_callback=None):
        self.reset_state()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("视频路径无效，无法打开视频")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        final_frame = None
        final_state = "未稳定识别状态"
        final_alert = False
        frame_index = 0

        while True:
            if should_stop is not None and should_stop():
                break

            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, detections, state, is_alert = self.detect_frame(frame, use_sequence=True)
            writer.write(annotated_frame)

            final_frame = annotated_frame
            final_state = state
            final_alert = is_alert
            frame_index += 1

            if alert_callback is not None:
                alert_callback(state, is_alert)

            if progress_callback is not None and frame_index % 10 == 0:
                progress_callback(frame_index, total_frames)

        cap.release()
        writer.release()

        return final_frame, final_state, final_alert, output_path

    @staticmethod
    def get_color(class_name):
        colors = {
            "awake": (0, 255, 0),
            "distracted": (255, 180, 0),
            "drowsy": (0, 165, 255),
            "head drop": (0, 0, 255),
            "phone": (255, 0, 255),
            "smoking": (128, 0, 255),
            "yawn": (0, 255, 255),
        }
        return colors.get(class_name, (0, 255, 0))
