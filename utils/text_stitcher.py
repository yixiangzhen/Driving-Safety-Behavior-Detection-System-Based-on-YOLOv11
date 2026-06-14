# text_stitcher.py
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher


class TextStitcher:
    def __init__(self):
        # 药品名称识别优化
        self.keyword_patterns = [
            r'注射用[\u4e00-\u9fa5]{3,8}',  # 注射用XXX
            r'[\u4e00-\u9fa5]{2,6}酸[\u4e00-\u9fa5]{0,6}',  # XX酸XX
            r'[\u4e00-\u9fa5]{3,8}素',  # XXXX素
            r'[\u4e00-\u9fa5]{2,8}剂'  # XXXX剂
        ]

        self.text_history = defaultdict(list)
        self.stitched_results = {}
        self.last_update_time = time.time()

    def update_history(self, bottle_id, text):
        self.text_history[bottle_id].append((text, time.time()))
        self.last_update_time = time.time()

    def stitch_text(self, bottle_id):
        """智能拼接文本"""
        if bottle_id not in self.text_history or not self.text_history[bottle_id]:
            return ""

        fragments = [t[0] for t in self.text_history[bottle_id][-10:]]  # 取最近10个片段

        # 1. 尝试从单个片段中提取完整药品名称
        for fragment in fragments:
            medicine_name = self.extract_medicine_name(fragment)
            if medicine_name:
                return medicine_name

        # 2. 尝试拼接多个片段
        if len(fragments) >= 3:
            # 寻找最佳拼接
            best_stitch = self.find_best_stitch(fragments)
            if best_stitch:
                return best_stitch

        return ""

    def extract_medicine_name(self, text):
        """从文本中提取药品名称"""
        for pattern in self.keyword_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(0)
                if 5 <= len(name) <= 20:  # 合理的药品名称长度
                    return name
        return ""

    def find_best_stitch(self, fragments):
        """寻找最佳拼接方式"""
        # 1. 简单拼接所有片段
        combined = " ".join(fragments)
        medicine_name = self.extract_medicine_name(combined)
        if medicine_name:
            return medicine_name

        # 2. 寻找重叠部分拼接
        if not fragments:
            return ""

        base_text = fragments[0]
        for fragment in fragments[1:]:
            overlap = self.find_overlap(base_text, fragment)
            if overlap > 0:
                base_text += fragment[overlap:]

        medicine_name = self.extract_medicine_name(base_text)
        if medicine_name:
            return medicine_name

        # 3. 返回最长的片段
        return max(fragments, key=len) if fragments else ""

    def find_overlap(self, text1, text2):
        """寻找两个文本的重叠部分"""
        if not text1 or not text2:
            return 0

        max_possible = min(len(text1), len(text2), 10)  # 最大重叠长度
        for overlap in range(max_possible, 2, -1):  # 最小重叠2个字符
            if text1.endswith(text2[:overlap]):
                return overlap

        # 模糊匹配
        for overlap in range(max_possible, 2, -1):
            end_part = text1[-overlap:]
            start_part = text2[:overlap]
            similarity = SequenceMatcher(None, end_part, start_part).ratio()
            if similarity > 0.8:
                return overlap

        return 0

    def get_stitched_result(self, bottle_id):
        return self.stitched_results.get(bottle_id, "")

    def update_stitched_result(self, bottle_id, text):
        self.stitched_results[bottle_id] = text

    def clean_history(self):
        """定期清理内存"""
        current_time = time.time()
        # 创建一个要删除的键列表，避免在迭代时修改字典
        keys_to_delete = []
        for bottle_id in list(self.text_history.keys()):
            if current_time - self.last_update_time > 10:  # 超过10秒未更新
                keys_to_delete.append(bottle_id)

        # 删除过期的记录
        for bottle_id in keys_to_delete:
            del self.text_history[bottle_id]
            if bottle_id in self.stitched_results:
                del self.stitched_results[bottle_id]
