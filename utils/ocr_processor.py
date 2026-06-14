import re
import threading
import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self):
        # 线程本地存储，避免多线程资源竞争
        self.local_ocr = threading.local()
        self.ocr_config = {
            'lang': 'ch',
            'use_angle_cls': True  # 初始化时启用角度分类，无需后续重复传参
        }

        # OCR常见错误文本修正字典
        self.ocr_corrections = {
            '疏软': '硫酸', '肾素': '霉素', '电产': '生产',
            '产品社号': '产品批号', '有就期至': '有效期至',
            '封用': '注射用', '耐用': '注射用', '蔬酸': '硫酸',
            '主射用': '注射用', '性射用': '注射用', '破酸': '硫酸',
            '软巟': '软骨素', '软量': '软骨素', '软具': '软骨素',
            '硫酸丝': '硫酸软骨素', '硫盟': '硫酸软骨素'
        }

    @property
    def ocr(self):
        """为每个线程创建独立OCR实例，避免多线程冲突"""
        if not hasattr(self.local_ocr, 'instance'):
            self.local_ocr.instance = PaddleOCR(**self.ocr_config)
        return self.local_ocr.instance

    def recognize_text(self, roi):
        try:
            # 1. 基础校验：ROI为空直接返回
            if roi is None or roi.size == 0:
                print("OCR警告: ROI区域为空（无图像可识别）")
                return ""

            # 2. 图像格式适配：强制转为3通道彩色图（解决模型通道数报错）
            if len(roi.shape) == 3:
                # 已为3通道，直接使用
                color_img = roi
            elif len(roi.shape) == 2:
                # 单通道灰度图 → 3通道伪彩色图（复制单通道数据到RGB三通道）
                color_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                print("OCR调试: 单通道灰度图已转为3通道彩色图")
            else:
                # 异常格式（如4通道透明图），拒绝处理
                print(f"OCR警告: ROI格式异常（形状{roi.shape}），仅支持3通道彩色/2通道灰度图")
                return ""

            # 3. 图像增强：提升文本对比度，减少OCR误识别
            # 先转灰度图增强（效果更优），再转回3通道适配模型
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray_img)
            enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            print(f"OCR调试: 增强后图像形状（通道验证）: {enhanced_img.shape}")  # 应输出 (h, w, 3)

            # 4. 调用OCR模型识别（新版本返回列表嵌套字典格式）
            ocr_result = self.ocr.ocr(enhanced_img)
            print(f"OCR调试: 原始结果关键信息: {'有数据' if ocr_result else '空'}")

            # 5. 结果格式校验（适配新版本字典返回格式）
            if not isinstance(ocr_result, list) or len(ocr_result) == 0:
                print("OCR警告: 识别结果为空（未检测到任何文本）")
                return ""

            # 取第一个结果字典（多结果场景优先取第一组）
            first_result = ocr_result[0]
            if not isinstance(first_result, dict):
                print(f"OCR警告: 结果格式异常（非字典）: {type(first_result)}")
                return ""

            # 检查字典是否包含核心键（rec_texts=识别文本，rec_scores=置信度）
            required_keys = ['rec_texts', 'rec_scores']
            if not all(key in first_result for key in required_keys):
                print(f"OCR警告: 结果字典缺少核心键，现有键: {list(first_result.keys())}")
                return ""

            # 6. 提取并过滤有效文本
            rec_texts = first_result['rec_texts']  # 文本列表
            rec_scores = first_result['rec_scores']  # 对应置信度列表

            # 校验文本与置信度列表长度一致
            if len(rec_texts) != len(rec_scores):
                print(f"OCR警告: 文本与置信度长度不匹配（文本{len(rec_texts)}条，置信度{len(rec_scores)}条）")
                return ""

            # 过滤空文本和低置信度（阈值0.1可根据需求调整）
            valid_texts = []
            valid_scores = []
            for text, score in zip(rec_texts, rec_scores):
                # 类型转换与清洗
                clean_text = str(text).strip()
                clean_score = float(score) if isinstance(score, (int, float)) else 0.0

                # 保留非空且置信度>0.1的文本
                if clean_text and clean_score > 0.1:
                    valid_texts.append(clean_text)
                    valid_scores.append(clean_score)

            # 无有效文本时返回空
            if not valid_texts:
                print("OCR警告: 无满足条件的有效文本（低置信度或空文本）")
                return ""

            # 7. 文本修正与输出
            print(f"OCR有效识别文本: {valid_texts}")
            print(f"对应置信度: {[round(s, 3) for s in valid_scores]}")  # 保留3位小数易读
            combined_text = " ".join(valid_texts)
            corrected_text = self.correct_text(combined_text)
            print(f"修正后最终文本: {corrected_text}")

            return corrected_text

        except Exception as e:
            print(f"OCR错误（捕获异常）: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印堆栈便于定位问题
            return ""

    def correct_text(self, text):
        """修正OCR识别的常见错误文本"""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        corrected_text = text
        # 遍历修正字典替换错误文本
        for wrong_text, correct_text in self.ocr_corrections.items():
            corrected_text = corrected_text.replace(wrong_text, correct_text)
        return corrected_text.strip()