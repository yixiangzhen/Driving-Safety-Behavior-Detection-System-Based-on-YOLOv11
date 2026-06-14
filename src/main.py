# industrial image OCR/src/main.py
import cv2
import threading
import argparse
import time
import os
from utils.yolo_detector import YoloDetector
from utils.ocr_processor import OCRProcessor
from utils.text_stitcher import TextStitcher
from utils.common import draw_chinese_text

# 全局变量
latest_frames = {0: None, 1: None}
locks = {0: threading.Lock(), 1: threading.Lock()}
stop_event = threading.Event()
OCR_LOG_PATH = "logs/ocr_recognized_trace.log"  # 带追踪的日志路径


def init_log_file():
    """初始化日志文件（带启动信息）"""
    log_dir = os.path.dirname(OCR_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(OCR_LOG_PATH):
        with open(OCR_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(f"OCR日志（带追踪） - 启动时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
    print(f"日志文件路径：{os.path.abspath(OCR_LOG_PATH)}")
    return OCR_LOG_PATH


def write_ocr_log(camera_id, obj_id, display_text, full_text):
    """硬写入日志（带追踪标记）"""
    try:
        log_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[追踪][{log_time}] 摄像头{camera_id} | {obj_id} | 显示：{display_text} | 完整：{full_text}\n"
        with open(OCR_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(log_line)
        print(f"[日志已写] {log_line.strip()}")  # 强制打印写入标记
    except Exception as e:
        print(f"[日志失败] 原因：{str(e)} | 内容：{log_line.strip()}")


class MedicineBottleOCR:
    def __init__(self, yolo_model_path):
        self.detector = YoloDetector(yolo_model_path)
        self.ocr_processor = OCRProcessor()
        self.text_stitcher = TextStitcher()
        self.font_path = "simhei.ttf"
        self.debug_counter = 0  # 调试计数器，避免打印过多

    def process_video(self, source=0):
        """处理视频流（带调试）"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[错误] 摄像头{source}无法打开")
            return
        print(f"[成功] 摄像头{source}已启动")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            processed_frame = self.process_frame(frame, source)
            with locks[source]:
                latest_frames[source] = processed_frame.copy()
            time.sleep(0.01)
        cap.release()

    def process_frame(self, frame, camera_id=None):
        """帧处理（逐行打印中间变量）"""
        self.debug_counter += 1
        # 每10帧打印一次，避免刷屏
        if self.debug_counter % 10 == 0:
            print(f"\n[调试] 处理第{self.debug_counter}帧（摄像头{camera_id}）")

        try:
            # 1. YOLO检测
            boxes, processed_frame = self.detector.detect(frame)
            if boxes is None:
                if self.debug_counter % 10 == 0:
                    print(f"[调试] YOLO未检测到物体（摄像头{camera_id}）")
                return frame
            print(f"[调试] YOLO检测到{len(boxes)}个物体（摄像头{camera_id}）")

            h, w = frame.shape[:2]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = max(0, min(w-1, x1)), max(0, min(w-1, x2))
                y1, y2 = max(0, min(h-1, y1)), max(0, min(h-1, y2))
                if x1 >= x2 or y1 >= y2:
                    continue

                # 2. 提取ROI
                margin = 10
                roi = frame[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
                print(f"[调试] 提取ROI（大小：{roi.shape[1]}x{roi.shape[0]}，摄像头{camera_id}）")

                # 3. OCR识别（核心追踪点）
                raw_text = self.ocr_processor.recognize_text(roi).strip()
                print(f"[调试] OCR原始文本（摄像头{camera_id}）：'{raw_text}'（长度：{len(raw_text)}）")

                obj_id = f"obj{i}"
                # 4. 文本拼接
                self.text_stitcher.update_dual_camera_history(obj_id, camera_id, raw_text)
                stitched_text = self.text_stitcher.dual_camera_stitch_text(obj_id)
                print(f"[调试] 拼接后文本（摄像头{camera_id}）：'{stitched_text}'（长度：{len(stitched_text) if stitched_text else 0}）")

                # 5. 确定最终文本（与界面显示关联）
                full_text = stitched_text if stitched_text else raw_text
                print(f"[调试] 最终判断文本（摄像头{camera_id}）：'{full_text}'（长度：{len(full_text)}）")

                # 6. 界面显示
                display_text = f"{obj_id}: {full_text[:15]}"
                processed_frame = draw_chinese_text(
                    processed_frame, display_text,
                    (max(0, x1-margin), max(0, y1-margin-30)), self.font_path
                )
                print(f"[调试] 界面显示文本（摄像头{camera_id}）：'{display_text}'")

                # 7. 强制写入日志（即使文本看似为空，也尝试写入）
                # 🔥 关键修改：移除len(full_text)判断，强制记录所有情况
                print(f"[调试] 准备写入日志（摄像头{camera_id}）：full_text是否为空？{len(full_text) == 0}")
                write_ocr_log(
                    camera_id=camera_id,
                    obj_id=obj_id,
                    display_text=display_text,
                    full_text=full_text
                )

        except Exception as e:
            print(f"[错误] 帧处理异常：{str(e)}")
        return processed_frame


def reco_worker(recognizer, source_index):
    recognizer.process_video(source_index)


def main():
    init_log_file()
    parser = argparse.ArgumentParser(description='OCR日志追踪')
    parser.add_argument('--model', type=str, default='../model/best.pt', help='YOLO模型路径')
    parser.add_argument('--camera1', type=int, default=0, help='摄像头1索引')
    args = parser.parse_args()

    # 模型校验
    if not os.path.exists(args.model):
        print(f"[错误] 模型不存在：{args.model}")
        return
    print(f"[成功] 模型路径：{os.path.abspath(args.model)}")

    # 启动单摄像头测试（排除多线程干扰）
    recognizer = MedicineBottleOCR(args.model)
    print(f"[启动] 按 'q' 退出 | 日志路径：{os.path.abspath(OCR_LOG_PATH)}")
    threading.Thread(target=reco_worker, args=(recognizer, args.camera1)).start()

    # 显示画面
    while True:
        with locks[0]:
            if latest_frames[0] is not None:
                cv2.imshow('Camera 1', latest_frames[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()
    print(f"[退出] 日志已保存至：{os.path.abspath(OCR_LOG_PATH)}")


if __name__ == "__main__":
    main()