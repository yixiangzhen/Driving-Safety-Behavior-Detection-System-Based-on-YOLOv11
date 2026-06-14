# utils/common.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


def draw_chinese_text(image, text, position, font_path="simhei.ttf", font_size=20, color=(0, 255, 0)):
    """在图像上绘制中文文本"""
    try:
        # 将OpenCV图像转换为PIL图像
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 创建绘图对象
        draw = ImageDraw.Draw(image_pil)

        # 加载字体
        font = ImageFont.truetype(font_path, font_size)

        # 绘制文本
        draw.text(position, text, font=font, fill=color)

        # 转换回OpenCV图像
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        return image_cv
    except Exception as e:
        print(f"绘制中文文本错误: {e}")
        # 如果出错，使用OpenCV绘制英文文本
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image


def extract_batch_no(text):
    """提取批号"""
    patterns = [
        r'批号[:：]?\s*(\w{6,12})',
        r'产品批号[:：]?\s*(\w{6,12})',
        r'Lot[:：]?\s*(\w{6,12})',
        r'\b(\d{6,12})\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def extract_expiry_date(text):
    """提取有效期"""
    patterns = [
        r'有效期至[:：]?\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
        r'有效期[:：]?\s*(\d{4}[-/年]\d{1,2}[-/月])',
        r'EXP[:：]?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            date = match.group(1)
            return date.replace('年', '-').replace('月', '-').replace('日', '')
    return ""
