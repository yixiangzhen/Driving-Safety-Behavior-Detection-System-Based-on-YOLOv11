# medicine_bottle_ocr.py
import cv2
import threading
import time
import logging
from utils.yolo_detector import YoloDetector
from utils.ocr_processor import OCRProcessor
from utils.text_stitcher import TextStitcher
from utils.common import draw_chinese_text, extract_batch_no, extract_expiry_date

latest_frames = {0: None, 1: None}
locks = {0: threading.Lock(), 1: threading.Lock()}
stop_event = threading.Event()


class MedicineBottleOCR:
    def __init__(self, yolo_model_path):
        # 初始化各模块
        self.yolo_model_path = yolo_model_path
        self.detector = YoloDetector(yolo_model_path)
        # 使用线程本地存储来避免多线程冲突
        self.local_ocr_processor = threading.local()
        self.local_text_stitcher = threading.local()

        # 显示设置
        self.font_path = "simhei.ttf"
        self.display_size = (800, 600)

    @property
    def ocr_processor(self):
        """为每个线程创建独立的OCR处理器"""
        if not hasattr(self.local_ocr_processor, 'instance'):
            self.local_ocr_processor.instance = OCRProcessor()
        return self.local_ocr_processor.instance

    @property
    def text_stitcher(self):
        """为每个线程创建独立的文本拼接器"""
        if not hasattr(self.local_text_stitcher, 'instance'):
            self.local_text_stitcher.instance = TextStitcher()
        return self.local_text_stitcher.instance

    def process_video(self, source=0, logger=None):
        """处理视频流"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            if logger:
                logger.error(f"无法打开视频源 {source}")
            print(f"无法打开视频源 {source}")
            return

        frame_count = 0
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if logger:
                    logger.warning(f"无法从视频源 {source} 读取到帧，视频结束。")
                break

            processed_frame = self.process_frame(frame, source)

            # 线程安全地更新帧
            with locks.get(source, threading.Lock()):
                latest_frames[source] = processed_frame.copy()

            frame_count += 1
            if frame_count % 2 == 0:
                if logger:
                    logger.info(f"已处理 {frame_count} 帧来自视频源 {source}")

        cap.release()
        if logger:
            logger.info(f"视频源 {source} 的处理线程已停止。")
        print(f"视频源 {source} 的处理线程已停止。")

    def process_frame(self, frame, source_id=0):
        """处理单帧"""
        # 1. YOLO检测药瓶
        boxes, processed_frame = self.detector.detect(frame)
        if boxes is None:
            return frame

        # 2. 对每个检测到的药瓶进行OCR
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # 确保坐标在图像范围内
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]

            # 检查ROI是否有效
            if roi.size == 0:
                print(f"警告: 摄像头 {source_id} 药瓶 {i} ROI区域为空")
                continue

            # 3. OCR识别
            text = self.ocr_processor.recognize_text(roi)
            print(f"摄像头 {source_id} 药瓶 {i} 识别文本: '{text}'")

            if text:
                # 4. 更新识别历史（使用当前线程的text_stitcher）
                self.text_stitcher.update_history(i, text)

                # 5. 智能拼接文本
                stitched_text = self.text_stitcher.stitch_text(i)
                if stitched_text:
                    self.text_stitcher.update_stitched_result(i, stitched_text)

                # 6. 在图像上显示结果
                display_text = stitched_text if stitched_text else text[:15]
                processed_frame = draw_chinese_text(
                    processed_frame,
                    f"{i}: {display_text}",
                    (x1, y1 - 30),
                    font_path=self.font_path
                )
            else:
                # 即使没有识别到文本，也显示提示信息
                processed_frame = draw_chinese_text(
                    processed_frame,
                    f"{i}: 未识别到文本",
                    (x1, y1 - 30),
                    font_path=self.font_path,
                    color=(0, 0, 255)  # 红色显示未识别
                )

        return processed_frame

    def print_results(self):
        """打印识别结果"""
        print("\n" + "=" * 50)
        print(f"药瓶识别结果 (时间: {time.strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 50)

        # 打印当前线程的结果
        try:
            for bottle_id, text in self.text_stitcher.stitched_results.items():
                print(f"药瓶 #{bottle_id}:")
                print(f"  识别文本: {text}")

                # 提取关键信息
                batch_no = extract_batch_no(text)
                expiry_date = extract_expiry_date(text)

                if batch_no:
                    print(f"  批号: {batch_no}")
                if expiry_date:
                    print(f"  有效期至: {expiry_date}")

                print("-" * 45)
        except Exception as e:
            print(f"打印结果出错: {e}")
        print("=" * 50)
        return 1


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def reco_worker(yolo_model_path, logger_name, source_index):
    """工作线程函数"""
    # 为每个线程创建独立的识别器实例
    recognizer = MedicineBottleOCR(yolo_model_path)
    logger = logging.getLogger(logger_name)
    logger.info(f"线程启动，处理视频源 {source_index}")
    recognizer.process_video(source_index, logger=logger)


def main():
    """主函数，启动双摄像头识别"""
    # YOLO模型路径
    yolo_model_path = r"D:\A_Python\跟着黑马学ai\workworkwrokwrok\工业图像识别1\ultralytics\yolo11n.pt"

    # 设置日志
    reco1_logger = setup_logger('reco1_logger', 'reco1.log')
    reco2_logger = setup_logger('reco2_logger', 'reco2.log')

    print("启动双摄像头识别...")
    # 创建并启动线程，为每个线程创建独立的识别器实例
    reco1_thread = threading.Thread(target=reco_worker, args=(yolo_model_path, 'reco1_logger', 0))
    reco2_thread = threading.Thread(target=reco_worker, args=(yolo_model_path, 'reco2_logger', 1))

    reco1_thread.start()
    reco2_thread.start()

    try:
        while True:
            local_frame1 = None
            local_frame2 = None

            # 线程安全地获取最新帧
            with locks[0]:
                if latest_frames[0] is not None:
                    local_frame1 = latest_frames[0].copy()

            with locks[1]:
                if latest_frames[1] is not None:
                    local_frame2 = latest_frames[1].copy()

            # 显示帧
            if local_frame1 is not None:
                cv2.imshow('Camera 1', local_frame1)

            if local_frame2 is not None:
                cv2.imshow('Camera 2', local_frame2)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("用户中断程序")
        stop_event.set()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()
    finally:
        # 等待线程结束
        reco1_thread.join(timeout=5)
        reco2_thread.join(timeout=5)

        # 清理资源
        cv2.destroyAllWindows()
        print("所有线程执行完毕，程序退出。")


if __name__ == "__main__":
    main()
