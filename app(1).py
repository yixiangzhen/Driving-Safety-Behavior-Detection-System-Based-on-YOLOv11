import os
import cv2
import json
import time
import queue
import threading
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw

from backend import DriverBehaviorDetector


MODEL_PATH = r"F:\code\yolov11\runs\detect\train4\weights\best.pt"


CLASS_CN = {
    "awake": "清醒",
    "distracted": "分心",
    "drowsy": "困倦",
    "head drop": "点头/低头",
    "phone": "手机",
    "smoking": "抽烟",
    "yawn": "打哈欠",
}


ALERT_TYPES = {
    "手机": "手机使用",
    "抽烟": "抽烟",
    "点头": "疲劳点头",
    "低头": "疲劳点头",
    "打哈欠": "打哈欠",
    "困倦": "困倦",
    "分心": "分心",
}


class NavButton(tk.Canvas):
    def __init__(self, parent, icon, text, command, width=188, height=46):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0, cursor="hand2")
        self.icon = icon
        self.text = text
        self.command = command
        self.width_value = width
        self.height_value = height
        self.active = False
        self.draw()
        self.bind("<Button-1>", lambda event: self.command())

    def draw(self):
        self.delete("all")
        bg = "#1d4ed8" if self.active else "#0f172a"
        fg = "#ffffff" if self.active else "#cbd5e1"
        if self.active:
            self.create_rectangle(0, 8, 4, self.height_value - 8, fill="#38bdf8", outline="")
        self.create_rectangle(8, 4, self.width_value - 8, self.height_value - 4, fill=bg, outline=bg)
        self.create_text(30, self.height_value // 2, text=self.icon, fill=fg, font=("Microsoft YaHei", 14, "bold"))
        self.create_text(58, self.height_value // 2, text=self.text, anchor="w", fill=fg, font=("Microsoft YaHei", 12, "bold"))

    def set_active(self, active):
        self.active = active
        self.draw()


class ActionButton(tk.Canvas):
    def __init__(self, parent, text, command, color="#2563eb", width=118, height=38):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0, cursor="hand2")
        self.text = text
        self.command = command
        self.color = color
        self.width_value = width
        self.height_value = height
        self.draw()
        self.bind("<Button-1>", lambda event: self.command())

    def round_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def draw(self):
        self.delete("all")
        self.round_rect(5, 7, self.width_value - 1, self.height_value - 1, 14, fill="#cbd5e1", outline="")
        self.round_rect(2, 2, self.width_value - 4, self.height_value - 6, 14, fill=self.color, outline="")
        self.create_text(
            self.width_value // 2,
            self.height_value // 2 - 2,
            text=self.text,
            fill="#ffffff",
            font=("Microsoft YaHei", 10, "bold"),
        )


class DriverDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("驾驶安全行为监测控制台")
        self.root.geometry("1360x860")
        self.root.minsize(1220, 760)
        self.root.configure(bg="#eef4fb")

        self.detector = DriverBehaviorDetector(MODEL_PATH)

        self.current_page = "detect"
        self.running_camera = False
        self.video_running = False
        self.busy = False
        self.camera_thread = None
        self.camera_stop_event = threading.Event()
        self.camera_queue = queue.Queue(maxsize=1)
        self.prev_frame_time = 0
        self.fps_smooth = 0

        self.current_image_tk = None
        self.warning_image_tk = None
        self.last_alert_state = None
        self.alert_logs = []
        self.risk_counts = {
            "手机使用": 0,
            "抽烟": 0,
            "疲劳点头": 0,
            "打哈欠": 0,
            "困倦": 0,
            "分心": 0,
        }
        self.last_detections = []
        self.last_state = "等待检测"
        self.last_is_alert = False
        self.mode_var = tk.StringVar(value="平衡模式")
        self.enabled_vars = {}
        self.conf_vars = {}
        self.threshold_vars = {}
        self.option_vars = {
            "show_boxes": tk.BooleanVar(value=True),
            "show_conf": tk.BooleanVar(value=True),
            "show_fps": tk.BooleanVar(value=True),
            "auto_log": tk.BooleanVar(value=True),
        }

        self.build_layout()
        self.switch_page("detect")
        self.update_clock()

    def build_layout(self):
        self.sidebar = tk.Frame(self.root, bg="#0f172a", width=218)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="驾驶安全监测",
            bg="#0f172a",
            fg="#ffffff",
            font=("Microsoft YaHei", 18, "bold"),
        ).pack(anchor="w", padx=20, pady=(26, 2))
        tk.Label(
            self.sidebar,
            text="Driver Monitor",
            bg="#0f172a",
            fg="#7dd3fc",
            font=("Microsoft YaHei", 10, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 24))

        self.nav_buttons = {}
        nav_specs = [
            ("overview", "⌂", "首页总览"),
            ("detect", "▣", "检测中心"),
            ("settings", "⚙", "检测设置"),
            ("ai", "AI", "AI助手"),
            ("logs", "≡", "危险日志"),
            ("about", "i", "系统说明"),
        ]
        for key, icon, text in nav_specs:
            btn = NavButton(self.sidebar, icon, text, lambda k=key: self.switch_page(k))
            btn.pack(padx=14, pady=4)
            self.nav_buttons[key] = btn

        tk.Label(
            self.sidebar,
            text="YOLOv11 · best.pt",
            bg="#0f172a",
            fg="#64748b",
            font=("Microsoft YaHei", 9),
        ).pack(side="bottom", anchor="w", padx=22, pady=(0, 24))

        self.main = tk.Frame(self.root, bg="#eef4fb")
        self.main.pack(side="right", fill="both", expand=True)

        self.topbar = tk.Frame(self.main, bg="#eef4fb", height=74)
        self.topbar.pack(fill="x")
        self.topbar.pack_propagate(False)

        self.page_title = tk.Label(
            self.topbar,
            text="检测中心",
            bg="#eef4fb",
            fg="#0f172a",
            font=("Microsoft YaHei", 24, "bold"),
        )
        self.page_title.pack(side="left", padx=28)

        self.time_label = tk.Label(
            self.topbar,
            text="",
            bg="#eef4fb",
            fg="#475569",
            font=("Microsoft YaHei", 10, "bold"),
            justify="right",
        )
        self.time_label.pack(side="right", padx=28)

        self.content = tk.Frame(self.main, bg="#eef4fb")
        self.content.pack(fill="both", expand=True, padx=24, pady=(0, 24))

    def update_clock(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"{now}\n模型：best.pt  设备：CUDA/CPU")
        self.root.after(1000, self.update_clock)

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def switch_page(self, page):
        self.current_page = page
        titles = {
            "overview": "首页总览",
            "detect": "检测中心",
            "settings": "检测设置",
            "ai": "AI助手",
            "logs": "危险日志",
            "about": "系统说明",
        }
        self.page_title.config(text=titles.get(page, "检测中心"))
        for key, btn in self.nav_buttons.items():
            btn.set_active(key == page)
        self.clear_content()

        if page == "overview":
            self.build_overview_page()
        elif page == "settings":
            self.build_settings_page()
        elif page == "ai":
            self.build_ai_page()
        elif page == "logs":
            self.build_logs_page()
        elif page == "about":
            self.build_about_page()
        else:
            self.build_detect_page()

    def make_card(self, parent, bg="#ffffff"):
        card = tk.Frame(parent, bg=bg, bd=0, highlightthickness=1, highlightbackground="#d9e4f2")
        return card

    def build_overview_page(self):
        top = tk.Frame(self.content, bg="#eef4fb")
        top.pack(fill="x", pady=(0, 16))

        stats = [
            ("当前状态", self.last_state, "#2563eb"),
            ("危险次数", str(len(self.alert_logs)), "#dc2626"),
            ("启用类别", f"{len(self.detector.enabled_classes)}/7", "#059669"),
            ("实时 FPS", f"{self.fps_smooth:.1f}", "#d97706"),
        ]
        for title, value, color in stats:
            card = self.make_card(top)
            card.pack(side="left", fill="x", expand=True, padx=8)
            tk.Label(card, text=title, bg="#ffffff", fg="#64748b", font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=18, pady=(14, 2))
            tk.Label(card, text=value, bg="#ffffff", fg=color, font=("Microsoft YaHei", 20, "bold"), wraplength=220, justify="left").pack(anchor="w", padx=18, pady=(0, 14))

        body = tk.Frame(self.content, bg="#eef4fb")
        body.pack(fill="both", expand=True)
        left = self.make_card(body)
        left.pack(side="left", fill="both", expand=True, padx=(8, 8))
        right = self.make_card(body)
        right.pack(side="right", fill="both", expand=True, padx=(8, 8))

        tk.Label(left, text="风险类型占比", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 16, "bold")).pack(anchor="w", padx=20, pady=(18, 8))
        self.overview_pie = tk.Canvas(left, bg="#ffffff", highlightthickness=0, height=280)
        self.overview_pie.pack(fill="x", padx=18, pady=8)
        self.draw_pie_chart(self.overview_pie, self.risk_counts)

        tk.Label(right, text="最近不安全行为", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 16, "bold")).pack(anchor="w", padx=20, pady=(18, 8))
        self.recent_log_frame = tk.Frame(right, bg="#ffffff")
        self.recent_log_frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.refresh_recent_logs(self.recent_log_frame)

    def build_detect_page(self):
        top_controls = tk.Frame(self.content, bg="#eef4fb")
        top_controls.pack(fill="x", pady=(0, 12))
        ActionButton(top_controls, "选择图片", self.detect_image, "#6257d8").pack(side="left", padx=6)
        ActionButton(top_controls, "选择视频", self.detect_video, "#2563eb").pack(side="left", padx=6)
        ActionButton(top_controls, "实时监测", self.start_camera, "#d97706").pack(side="left", padx=6)
        ActionButton(top_controls, "停止检测", self.stop_all, "#dc2626").pack(side="left", padx=6)

        self.mode_label = tk.Label(
            top_controls,
            text="当前模式：平衡模式",
            bg="#eef4fb",
            fg="#475569",
            font=("Microsoft YaHei", 10, "bold"),
        )
        self.mode_label.pack(side="right", padx=10)

        body = tk.Frame(self.content, bg="#eef4fb")
        body.pack(fill="both", expand=True)

        self.video_card = self.make_card(body, "#0f172a")
        self.video_card.pack(side="left", fill="both", expand=True, padx=(0, 14))
        video_header = tk.Frame(self.video_card, bg="#0f172a")
        video_header.pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(video_header, text="检测画面 LIVE", bg="#0f172a", fg="#e0f2fe", font=("Microsoft YaHei", 15, "bold")).pack(side="left")
        self.source_label = tk.Label(video_header, text="来源：暂无", bg="#0f172a", fg="#94a3b8", font=("Microsoft YaHei", 10))
        self.source_label.pack(side="right")
        self.image_label = tk.Label(self.video_card, bg="#0f172a")
        self.image_label.pack(expand=True, fill="both", padx=18, pady=18)

        side = tk.Frame(body, bg="#eef4fb", width=372)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        self.status_card = self.make_card(side)
        self.status_card.pack(fill="x", pady=(0, 12))
        tk.Label(self.status_card, text="当前状态", bg="#ffffff", fg="#64748b", font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", padx=18, pady=(14, 2))
        self.state_label = tk.Label(self.status_card, text="等待检测", bg="#ffffff", fg="#2563eb", font=("Microsoft YaHei", 18, "bold"), wraplength=320, justify="left")
        self.state_label.pack(anchor="w", padx=18, pady=(0, 8))
        self.risk_label = tk.Label(self.status_card, text="风险等级：未开始", bg="#ffffff", fg="#64748b", font=("Microsoft YaHei", 12, "bold"))
        self.risk_label.pack(anchor="w", padx=18, pady=(0, 14))

        bar_card = self.make_card(side)
        bar_card.pack(fill="x", pady=(0, 12))
        tk.Label(bar_card, text="类别置信度", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 14, "bold")).pack(anchor="w", padx=18, pady=(14, 4))
        self.bar_canvas = tk.Canvas(bar_card, bg="#ffffff", highlightthickness=0, height=178)
        self.bar_canvas.pack(fill="x", padx=12, pady=(0, 12))

        pie_card = self.make_card(side)
        pie_card.pack(fill="both", expand=True)
        tk.Label(pie_card, text="风险统计", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 14, "bold")).pack(anchor="w", padx=18, pady=(14, 4))
        self.pie_canvas = tk.Canvas(pie_card, bg="#ffffff", highlightthickness=0, height=205)
        self.pie_canvas.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(pie_card, text="最近日志", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w", padx=18)
        self.mini_log_label = tk.Label(pie_card, text="暂无", bg="#ffffff", fg="#475569", font=("Microsoft YaHei", 9), justify="left", wraplength=320)
        self.mini_log_label.pack(anchor="w", padx=18, pady=(4, 12))

        self.refresh_visuals()

    def build_settings_page(self):
        container = tk.Frame(self.content, bg="#eef4fb")
        container.pack(fill="both", expand=True)

        left = self.make_card(container)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = self.make_card(container)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.configure(width=430)
        right.pack_propagate(False)

        tk.Label(left, text="检测类别与置信度", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 16, "bold")).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(left, text="勾选要检测的类别，并拖动滑块调整每类置信度阈值。阈值越低越灵敏，阈值越高越严格。", bg="#ffffff", fg="#64748b", font=("Microsoft YaHei", 10), wraplength=520, justify="left").pack(anchor="w", padx=20, pady=(0, 12))

        for cls in self.detector.class_names:
            row = tk.Frame(left, bg="#ffffff")
            row.pack(fill="x", padx=20, pady=4)
            var = tk.BooleanVar(value=cls in self.detector.enabled_classes)
            self.enabled_vars[cls] = var
            cb = tk.Checkbutton(row, text=f"{CLASS_CN.get(cls, cls)} ({cls})", variable=var, bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 10, "bold"), command=lambda c=cls: self.apply_enabled(c))
            cb.pack(side="left")
            value_label = tk.Label(row, text=f"{self.detector.class_conf[cls]:.2f}", bg="#ffffff", fg="#2563eb", font=("Microsoft YaHei", 10, "bold"), width=5)
            value_label.pack(side="right", padx=(8, 0))
            scale_var = tk.DoubleVar(value=self.detector.class_conf[cls])
            self.conf_vars[cls] = (scale_var, value_label)
            scale = tk.Scale(row, from_=0.05, to=0.90, resolution=0.01, orient="horizontal", variable=scale_var, bg="#ffffff", highlightthickness=0, length=220, command=lambda v, c=cls: self.apply_conf(c, v))
            scale.pack(side="right")

        tk.Label(right, text="检测模式与连续帧阈值", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 16, "bold")).pack(anchor="w", padx=20, pady=(18, 4))
        mode_row = tk.Frame(right, bg="#ffffff")
        mode_row.pack(fill="x", padx=20, pady=(4, 12))
        for mode in ["灵敏模式", "平衡模式", "严格模式"]:
            rb = tk.Radiobutton(mode_row, text=mode, value=mode, variable=self.mode_var, bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 10, "bold"), command=self.apply_preset)
            rb.pack(side="left", padx=(0, 18))

        tk.Label(right, text="危险行为持续帧数阈值", bg="#ffffff", fg="#64748b", font=("Microsoft YaHei", 10), wraplength=520, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        for cls in self.detector.thresholds:
            row = tk.Frame(right, bg="#ffffff")
            row.pack(fill="x", padx=20, pady=4)
            tk.Label(row, text=f"{CLASS_CN.get(cls, cls)}", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 10, "bold"), width=12, anchor="w").pack(side="left")
            value_label = tk.Label(row, text=str(self.detector.thresholds[cls]), bg="#ffffff", fg="#dc2626", font=("Microsoft YaHei", 10, "bold"), width=4)
            value_label.pack(side="right")
            scale_var = tk.IntVar(value=self.detector.thresholds[cls])
            self.threshold_vars[cls] = (scale_var, value_label)
            scale = tk.Scale(row, from_=3, to=45, resolution=1, orient="horizontal", variable=scale_var, bg="#ffffff", highlightthickness=0, length=250, command=lambda v, c=cls: self.apply_threshold(c, v))
            scale.pack(side="right")

        opt = tk.Frame(right, bg="#ffffff")
        opt.pack(fill="x", padx=20, pady=(14, 6))
        tk.Label(opt, text="显示选项", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 13, "bold")).pack(anchor="w")
        options = [
            ("show_boxes", "显示检测框"),
            ("show_conf", "显示置信度"),
            ("show_fps", "显示 FPS"),
            ("auto_log", "自动写入日志"),
        ]
        for key, text in options:
            tk.Checkbutton(opt, text=text, variable=self.option_vars[key], bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 10), command=self.apply_options).pack(anchor="w", pady=2)

        btn_row = tk.Frame(right, bg="#ffffff")
        btn_row.pack(fill="x", padx=20, pady=(12, 18))
        ActionButton(btn_row, "恢复默认", self.reset_settings, "#64748b").pack(side="left", padx=(0, 8))
        ActionButton(btn_row, "应用设置", self.apply_all_settings, "#2563eb").pack(side="left")

    def build_logs_page(self):
        panel = self.make_card(self.content)
        panel.pack(fill="both", expand=True)
        header = tk.Frame(panel, bg="#ffffff")
        header.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(header, text="不安全行为日志", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 18, "bold")).pack(side="left")
        ActionButton(header, "清空日志", self.clear_logs, "#dc2626", width=98).pack(side="right")

        columns = ("time", "state")
        self.log_table = ttk.Treeview(panel, columns=columns, show="headings", height=20)
        self.log_table.heading("time", text="开始时间")
        self.log_table.heading("state", text="不安全行为")
        self.log_table.column("time", width=260, anchor="center")
        self.log_table.column("state", width=780, anchor="w")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.log_table.yview)
        self.log_table.configure(yscrollcommand=scrollbar.set)
        self.log_table.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 20))
        scrollbar.pack(side="right", fill="y", padx=(0, 20), pady=(0, 20))
        self.refresh_log_table()

    def build_ai_page(self):
        container = tk.Frame(self.content, bg="#eef4fb")
        container.pack(fill="both", expand=True)

        left = self.make_card(container)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = self.make_card(container)
        right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        tk.Label(
            left,
            text="AI 助手",
            bg="#ffffff",
            fg="#0f172a",
            font=("Microsoft YaHei", 20, "bold"),
        ).pack(anchor="w", padx=22, pady=(20, 4))
        tk.Label(
            left,
            text="基于检测日志、风险统计、当前检测类别和阈值配置，自动生成驾驶安全分析内容。",
            bg="#ffffff",
            fg="#64748b",
            font=("Microsoft YaHei", 10),
            wraplength=610,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 18))

        btn_row = tk.Frame(left, bg="#ffffff")
        btn_row.pack(fill="x", padx=22, pady=(0, 14))
        ActionButton(btn_row, "生成检测报告", self.ai_generate_report, "#2563eb", width=132).pack(side="left", padx=(0, 10))
        ActionButton(btn_row, "风险总结建议", self.ai_generate_summary, "#16a34a", width=132).pack(side="left", padx=(0, 10))
        ActionButton(btn_row, "阈值调参建议", self.ai_generate_tuning, "#d97706", width=132).pack(side="left", padx=(0, 10))
        ActionButton(btn_row, "清空内容", self.ai_clear_output, "#64748b", width=100).pack(side="left")

        output_frame = tk.Frame(left, bg="#ffffff")
        output_frame.pack(fill="both", expand=True, padx=22, pady=(0, 22))
        self.ai_output = tk.Text(
            output_frame,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Microsoft YaHei", 11),
            wrap="word",
            bd=0,
            padx=16,
            pady=14,
        )
        self.ai_output.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.ai_output.yview)
        scroll.pack(side="right", fill="y")
        self.ai_output.configure(yscrollcommand=scroll.set)
        self.ai_output.insert("end", self.ai_welcome_text())

        tk.Label(
            right,
            text="AI 可分析的信息",
            bg="#ffffff",
            fg="#0f172a",
            font=("Microsoft YaHei", 16, "bold"),
        ).pack(anchor="w", padx=20, pady=(20, 10))

        self.ai_stats_canvas = tk.Canvas(right, bg="#ffffff", highlightthickness=0, height=215)
        self.ai_stats_canvas.pack(fill="x", padx=18, pady=(0, 8))
        self.draw_pie_chart(self.ai_stats_canvas, self.risk_counts)

        context_frame = tk.Frame(right, bg="#ffffff")
        context_frame.pack(fill="both", expand=True, padx=22, pady=(8, 16))
        self.ai_context_text = tk.Text(
            context_frame,
            bg="#f8fafc",
            fg="#334155",
            font=("Microsoft YaHei", 10),
            wrap="word",
            bd=0,
            padx=10,
            pady=10,
            height=12,
        )
        self.ai_context_text.pack(side="left", fill="both", expand=True)
        context_scroll = ttk.Scrollbar(context_frame, orient="vertical", command=self.ai_context_text.yview)
        context_scroll.pack(side="right", fill="y")
        self.ai_context_text.configure(yscrollcommand=context_scroll.set)
        self.ai_context_text.insert("end", self.get_ai_context_text())
        self.ai_context_text.configure(state="disabled")

        tk.Label(
            right,
            text="说明",
            bg="#ffffff",
            fg="#0f172a",
            font=("Microsoft YaHei", 13, "bold"),
        ).pack(anchor="w", padx=22, pady=(4, 4))
        tk.Label(
            right,
            text="当前 AI 助手为本地分析版，不需要联网。后续可以把这里生成的日志、统计和阈值信息发送给大语言模型，获得更自然、更详细的分析报告。",
            bg="#ffffff",
            fg="#64748b",
            font=("Microsoft YaHei", 10),
            justify="left",
            wraplength=470,
        ).pack(anchor="w", padx=22, pady=(0, 18))

    def ai_welcome_text(self):
        return (
            "欢迎使用 AI助手。\n\n"
            "当前页面已接入 qwen-turbo，可以基于系统日志和检测统计生成两类内容：\n"
            "1. 自动生成驾驶安全检测报告。\n"
            "2. 根据危险日志生成风险总结和安全建议。\n\n"
            "建议先完成一次图片、视频或实时检测，再点击上方按钮生成内容。"
        )

    def ai_clear_output(self):
        if hasattr(self, "ai_output"):
            self.ai_output.delete("1.0", "end")
            self.ai_output.insert("end", self.ai_welcome_text())

    def ai_set_output(self, text):
        if not hasattr(self, "ai_output"):
            return
        self.ai_output.delete("1.0", "end")
        self.ai_output.insert("end", text)

    def ai_generate_report(self):
        prompt = (
            "请根据以下驾驶安全行为检测系统数据，生成一份正式、清晰、适合作为课程大作业展示的检测报告。"
            "内容要包括：检测任务说明、YOLO检测结果概况、危险行为统计、当前风险判断、系统效果评价。"
            "语言使用中文，不要编造未提供的数据。\n\n"
            f"{self.get_ai_context_text(detail=True)}"
        )
        self.run_qwen_task("正在调用 qwen-turbo 生成驾驶安全检测报告，请稍等...", prompt)

    def ai_generate_summary(self):
        prompt = (
            "请根据以下驾驶安全行为检测日志和统计结果，生成风险总结与安全建议。"
            "请重点说明最主要的不安全行为、可能造成的驾驶风险、对驾驶员的提醒建议、对系统使用者的改进建议。"
            "语言使用中文，建议要具体，不要空泛。\n\n"
            f"{self.get_ai_context_text(detail=True)}"
        )
        self.run_qwen_task("正在调用 qwen-turbo 生成风险总结和安全建议，请稍等...", prompt)

    def ai_generate_tuning(self):
        prompt = (
            "请根据以下驾驶行为检测系统的最近检测结果、危险日志、类别置信度阈值和连续帧阈值，"
            "给出阈值调参建议。说明哪些类别可以适当降低阈值以减少漏检，哪些类别可以适当提高阈值以减少误报。"
            "语言使用中文，建议要谨慎，不要给出过激调整。\n\n"
            f"{self.get_ai_context_text(detail=True)}"
        )
        self.run_qwen_task("正在调用 qwen-turbo 生成阈值调参建议，请稍等...", prompt)

    def run_qwen_task(self, loading_text, prompt):
        self.ai_set_output(loading_text)
        thread = threading.Thread(target=self.qwen_worker, args=(prompt,), daemon=True)
        thread.start()

    def qwen_worker(self, prompt):
        try:
            result = self.call_qwen_turbo(prompt)
        except Exception as e:
            result = (
                "AI生成失败。\n\n"
                f"错误信息：{e}\n\n"
                "请检查：\n"
                "1. 项目根目录 .env 中是否配置 DASHSCOPE_API_KEY=你的Key\n"
                "2. 如果你使用的是其他变量名，也可以配置 QWEN_API_KEY=你的Key\n"
                "3. 当前电脑是否能访问 DashScope 兼容接口\n"
                "4. QWEN_MODEL 可选，默认使用 qwen-turbo"
            )
        self.root.after(0, lambda: self.ai_set_output(result))

    def call_qwen_turbo(self, prompt):
        api_key = self.get_env_value("DASHSCOPE_API_KEY") or self.get_env_value("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("没有读取到 DASHSCOPE_API_KEY 或 QWEN_API_KEY")

        url = self.get_env_value("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        model = self.get_env_value("QWEN_MODEL") or "qwen-turbo"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是驾驶安全行为检测系统的AI分析助手，回答要专业、准确、适合课程答辩和实验报告。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

        result = json.loads(body)
        return result["choices"][0]["message"]["content"].strip()

    def get_env_value(self, key):
        value = os.environ.get(key)
        if value:
            return value.strip().strip('"').strip("'")
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(env_path):
            return None
        encodings = ["utf-8", "gbk"]
        for encoding in encodings:
            try:
                with open(env_path, "r", encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        name, raw_value = line.split("=", 1)
                        if name.strip() == key:
                            return raw_value.strip().strip('"').strip("'")
                return None
            except UnicodeDecodeError:
                continue
        return None

    def get_ai_context_text(self, detail=False):
        total_alerts = len(self.alert_logs)
        risk_lines = [f"- {key}: {value}次" for key, value in self.risk_counts.items()]
        enabled = sorted(list(self.detector.enabled_classes))
        conf_lines = [f"- {cls}: {self.detector.class_conf.get(cls, 0):.2f}" for cls in self.detector.class_names]
        threshold_lines = [f"- {cls}: {value}帧" for cls, value in self.detector.thresholds.items()]
        recent_logs = self.alert_logs[-20:] if detail else self.alert_logs[-5:]
        if recent_logs:
            log_lines = [f"- {item['time']}  {item['state']}" for item in recent_logs]
        else:
            log_lines = ["- 暂无危险日志"]

        if self.last_detections:
            detection_lines = [
                f"- {item.get('class_name', 'unknown')}: {item.get('conf', 0):.2f}"
                for item in sorted(self.last_detections, key=lambda x: x.get("conf", 0), reverse=True)[:8]
            ]
        else:
            detection_lines = ["- 暂无最近检测框"]

        return (
            f"当前状态: {self.last_state}\n"
            f"当前是否报警: {'是' if self.last_is_alert else '否'}\n"
            f"危险日志总数: {total_alerts}\n"
            f"启用类别: {', '.join(enabled)}\n"
            f"实时FPS: {self.fps_smooth:.1f}\n\n"
            "风险统计:\n" + "\n".join(risk_lines) + "\n\n"
            "最近检测类别与置信度:\n" + "\n".join(detection_lines) + "\n\n"
            "最近危险日志:\n" + "\n".join(log_lines) + "\n\n"
            "当前类别置信度阈值:\n" + "\n".join(conf_lines) + "\n\n"
            "当前连续帧触发阈值:\n" + "\n".join(threshold_lines)
        )

    def build_about_page(self):
        panel = self.make_card(self.content)
        panel.pack(fill="both", expand=True)
        tk.Label(panel, text="系统说明", bg="#ffffff", fg="#0f172a", font=("Microsoft YaHei", 20, "bold")).pack(anchor="w", padx=24, pady=(24, 10))
        text = (
            "本系统基于 YOLOv11 实现驾驶安全行为检测，支持图片检测、视频检测和摄像头实时监测。\n\n"
            "检测类别包括：awake、distracted、drowsy、head drop、phone、smoking、yawn。\n\n"
            "检测设置页支持类别开关、每类置信度阈值、连续帧触发阈值、显示选项和检测模式切换。\n\n"
            "实时检测采用 camera_worker 子线程和 Queue(maxsize=1) 最新帧队列，减少 YOLO 推理造成的界面卡顿。"
        )
        tk.Label(panel, text=text, bg="#ffffff", fg="#334155", font=("Microsoft YaHei", 12), justify="left", wraplength=860).pack(anchor="w", padx=24, pady=12)

    def apply_enabled(self, cls):
        self.detector.set_enabled_class(cls, self.enabled_vars[cls].get())
        self.detector.reset_state()

    def apply_conf(self, cls, value):
        var, label = self.conf_vars[cls]
        self.detector.set_class_conf(cls, float(value))
        label.config(text=f"{float(value):.2f}")

    def apply_threshold(self, cls, value):
        var, label = self.threshold_vars[cls]
        self.detector.set_sequence_threshold(cls, int(float(value)))
        label.config(text=str(int(float(value))))

    def apply_preset(self):
        self.detector.set_preset(self.mode_var.get())
        for cls, (var, label) in self.conf_vars.items():
            var.set(self.detector.class_conf[cls])
            label.config(text=f"{self.detector.class_conf[cls]:.2f}")
        self.detector.reset_state()
        if hasattr(self, "mode_label"):
            self.mode_label.config(text=f"当前模式：{self.mode_var.get()}")

    def apply_options(self):
        self.detector.show_boxes = self.option_vars["show_boxes"].get()
        self.detector.show_conf = self.option_vars["show_conf"].get()

    def apply_all_settings(self):
        self.apply_options()
        messagebox.showinfo("设置已应用", "检测设置已更新")

    def reset_settings(self):
        self.mode_var.set("平衡模式")
        self.detector.enabled_classes = set(self.detector.class_names)
        for cls, var in self.enabled_vars.items():
            var.set(True)
        self.detector.set_preset("平衡模式")
        default_thresholds = {
            "distracted": 15,
            "drowsy": 20,
            "head drop": 8,
            "phone": 12,
            "smoking": 12,
            "yawn": 10,
        }
        self.detector.thresholds.update(default_thresholds)
        for cls, (var, label) in self.conf_vars.items():
            var.set(self.detector.class_conf[cls])
            label.config(text=f"{self.detector.class_conf[cls]:.2f}")
        for cls, (var, label) in self.threshold_vars.items():
            var.set(self.detector.thresholds[cls])
            label.config(text=str(self.detector.thresholds[cls]))
        self.detector.reset_state()

    def get_risk_info(self, state, is_alert):
        if not is_alert:
            if "正常" in state:
                return "安全", "#16a34a"
            return "未稳定", "#64748b"
        if "点头" in state or "手机" in state or "抽烟" in state:
            return "危险", "#dc2626"
        if "疲劳" in state or "分心" in state:
            return "警告", "#d97706"
        return "异常", "#f97316"

    def classify_alert_type(self, state):
        for key, value in ALERT_TYPES.items():
            if key in state:
                return value
        return "分心"

    def detect_image(self):
        self.stop_camera()
        self.video_running = False
        if self.busy:
            messagebox.showwarning("提示", "当前正在检测，请先停止或等待检测完成")
            return
        image_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if not image_path:
            return
        self.switch_page("detect")
        self.busy = True
        self.source_label.config(text=f"来源：图片")
        try:
            frame, detections, state, is_alert = self.detector.detect_image(image_path)
            self.show_frame(frame)
            self.update_state_panel(state, is_alert)
            self.update_detection_visuals(detections)
            if is_alert:
                self.add_alert_log(state)
        except Exception as e:
            messagebox.showerror("错误", str(e))
        finally:
            self.busy = False

    def detect_video(self):
        self.stop_camera()
        self.video_running = False
        if self.busy:
            messagebox.showwarning("提示", "当前正在检测，请先停止或等待检测完成")
            return
        video_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if not video_path:
            return
        self.switch_page("detect")
        self.source_label.config(text="来源：视频")
        self.state_label.config(text="视频检测中，请稍等", fg="#d97706")
        self.busy = True
        self.video_running = True
        thread = threading.Thread(target=self.detect_video_worker, args=(video_path,), daemon=True)
        thread.start()

    def detect_video_worker(self, video_path):
        output_path = os.path.join(os.path.dirname(video_path), "detect_result.mp4")
        try:
            final_frame, state, is_alert, saved_path = self.detector.detect_video(
                video_path,
                output_path,
                should_stop=lambda: not self.video_running,
                progress_callback=self.update_video_progress,
                alert_callback=lambda s, a: self.root.after(0, lambda: self.handle_alert_state(s, a)),
            )
            self.root.after(0, lambda: self.on_video_finished(final_frame, state, is_alert, saved_path))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.busy = False
            self.video_running = False

    def update_video_progress(self, current, total):
        text = f"视频检测中 {current}/{total}" if total > 0 else f"视频检测中 {current}"
        self.root.after(0, lambda: self.state_label.config(text=text, fg="#d97706"))

    def on_video_finished(self, final_frame, state, is_alert, saved_path):
        self.last_alert_state = None
        if final_frame is not None:
            self.show_frame(final_frame)
            self.update_state_panel(f"{state}\n结果视频：{saved_path}", is_alert)
        else:
            self.state_label.config(text="视频检测已停止", fg="#64748b")
            self.risk_label.config(text="风险等级：已停止", fg="#64748b")

    def start_camera(self):
        self.video_running = False
        if self.busy:
            messagebox.showwarning("提示", "当前正在检测，请先停止或等待检测完成")
            return
        if self.running_camera:
            return
        self.switch_page("detect")
        self.detector.reset_state()
        self.last_alert_state = None
        self.camera_stop_event.clear()
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break
        self.source_label.config(text="来源：摄像头实时监测")
        self.state_label.config(text="摄像头启动中", fg="#d97706")
        self.risk_label.config(text="风险等级：检测中", fg="#d97706")
        self.running_camera = True
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()
        self.poll_camera_queue()

    def camera_worker(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("错误", "无法打开摄像头，请检查摄像头是否被占用"))
            self.running_camera = False
            return
        self.prev_frame_time = time.time()
        self.fps_smooth = 0
        while not self.camera_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.time()
            diff = now - self.prev_frame_time
            if diff > 0:
                fps = 1.0 / diff
                self.fps_smooth = fps if self.fps_smooth == 0 else self.fps_smooth * 0.9 + fps * 0.1
            self.prev_frame_time = now
            fps_text = f"FPS: {self.fps_smooth:.1f}" if self.option_vars["show_fps"].get() else None
            frame, detections, state, is_alert = self.detector.detect_frame(
                frame,
                use_sequence=True,
                show_fps_text=fps_text,
            )
            data = (frame, detections, state, is_alert)
            if self.camera_queue.full():
                try:
                    self.camera_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.camera_queue.put_nowait(data)
            except queue.Full:
                pass
        cap.release()

    def poll_camera_queue(self):
        if not self.running_camera:
            return
        try:
            frame, detections, state, is_alert = self.camera_queue.get_nowait()
            self.show_frame(frame)
            self.update_state_panel(state, is_alert)
            self.update_detection_visuals(detections)
            self.handle_alert_state(state, is_alert)
        except queue.Empty:
            pass
        self.root.after(30, self.poll_camera_queue)

    def stop_camera(self):
        self.camera_stop_event.set()
        self.running_camera = False

    def stop_all(self):
        self.video_running = False
        self.stop_camera()
        self.busy = False
        self.last_alert_state = None
        if hasattr(self, "state_label"):
            self.state_label.config(text="已停止检测", fg="#64748b")
            self.risk_label.config(text="风险等级：已停止", fg="#64748b")

    def show_frame(self, frame):
        if self.current_page != "detect" or not hasattr(self, "image_label"):
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((760, 520))
        self.current_image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_image_tk)

    def update_state_panel(self, state, is_alert):
        self.last_state = state
        self.last_is_alert = is_alert
        if self.current_page != "detect" or not hasattr(self, "state_label"):
            return
        risk, color = self.get_risk_info(state, is_alert)
        self.state_label.config(text=state, fg=color)
        self.risk_label.config(text=f"风险等级：{risk}", fg=color)

    def update_detection_visuals(self, detections):
        self.last_detections = detections
        if hasattr(self, "bar_canvas"):
            self.draw_confidence_bars(self.bar_canvas, detections)
        if hasattr(self, "pie_canvas"):
            self.draw_pie_chart(self.pie_canvas, self.risk_counts)
        if hasattr(self, "ai_stats_canvas"):
            self.draw_pie_chart(self.ai_stats_canvas, self.risk_counts)
        if hasattr(self, "ai_context_text"):
            self.ai_context_text.configure(state="normal")
            self.ai_context_text.delete("1.0", "end")
            self.ai_context_text.insert("end", self.get_ai_context_text())
            self.ai_context_text.configure(state="disabled")
        if hasattr(self, "mini_log_label"):
            recent = self.alert_logs[-3:]
            if not recent:
                self.mini_log_label.config(text="暂无")
            else:
                self.mini_log_label.config(text="\n".join([f"{x['time']}  {x['state']}" for x in recent]))

    def refresh_visuals(self):
        # 👇 全局判断：如果页面控件不存在，直接退出
        if hasattr(self, 'ai_context_text') and self.ai_context_text.winfo_exists():
            self.draw_pie_chart(self.ai_stats_canvas, self.risk_counts)
            self.draw_confidence_bars(self.bar_canvas, [])

            self.ai_context_text.configure(state="normal")
            self.ai_context_text.delete(1.0, tk.END)
            self.ai_context_text.insert(tk.END, self.ai_context)
            self.ai_context_text.configure(state="disabled")

    def draw_confidence_bars(self, canvas, detections):
        canvas.delete("all")
        if not detections:
            canvas.create_text(165, 80, text="暂无检测类别", fill="#94a3b8", font=("Microsoft YaHei", 11, "bold"))
            return
        top_items = sorted(detections, key=lambda x: x["conf"], reverse=True)[:5]
        y = 18
        for item in top_items:
            name = item["class_name"]
            conf = item["conf"]
            label = CLASS_CN.get(name, name)
            canvas.create_text(18, y + 8, text=label, anchor="w", fill="#0f172a", font=("Microsoft YaHei", 9, "bold"))
            canvas.create_rectangle(92, y, 285, y + 16, fill="#e2e8f0", outline="")
            color = "#16a34a" if name == "awake" else "#dc2626" if name in ["phone", "smoking", "head drop"] else "#d97706"
            canvas.create_rectangle(92, y, 92 + int(193 * min(conf, 1)), y + 16, fill=color, outline="")
            canvas.create_text(300, y + 8, text=f"{conf:.2f}", anchor="w", fill="#475569", font=("Microsoft YaHei", 9, "bold"))
            y += 30

    def draw_pie_chart(self, canvas, counts):
        if not canvas or not canvas.winfo_exists():
            return
        canvas.delete("all")
        total = sum(counts.values())
        colors = ["#dc2626", "#f97316", "#f59e0b", "#2563eb", "#14b8a6", "#8b5cf6"]
        if total == 0:
            canvas.create_oval(45, 25, 165, 145, fill="#e2e8f0", outline="")
            canvas.create_text(105, 85, text="暂无风险", fill="#64748b", font=("Microsoft YaHei", 10, "bold"))
            return
        start = 0
        i = 0
        for key, value in counts.items():
            if value <= 0:
                continue
            extent = 360 * value / total
            canvas.create_arc(42, 22, 168, 148, start=start, extent=extent, fill=colors[i % len(colors)], outline="#ffffff")
            canvas.create_rectangle(205, 26 + i * 24, 218, 39 + i * 24, fill=colors[i % len(colors)], outline="")
            canvas.create_text(226, 32 + i * 24, text=f"{key} {value}", anchor="w", fill="#334155", font=("Microsoft YaHei", 9, "bold"))
            start += extent
            i += 1

    def handle_alert_state(self, state, is_alert):
        if is_alert:
            if state != self.last_alert_state:
                self.add_alert_log(state)
                self.last_alert_state = state
        else:
            self.last_alert_state = None

    def add_alert_log(self, state):
        if not self.option_vars["auto_log"].get():
            return
        alert_type = self.classify_alert_type(state)
        self.risk_counts[alert_type] = self.risk_counts.get(alert_type, 0) + 1
        self.alert_logs.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
        })
        if self.current_page == "logs" and hasattr(self, "log_table"):
            self.refresh_log_table()

    def refresh_log_table(self):
        if not hasattr(self, "log_table"):
            return
        for item in self.log_table.get_children():
            self.log_table.delete(item)
        for log in self.alert_logs:
            self.log_table.insert("", "end", values=(log["time"], log["state"]))

    def refresh_recent_logs(self, parent):
        for widget in parent.winfo_children():
            widget.destroy()
        if not self.alert_logs:
            tk.Label(parent, text="暂无不安全行为记录", bg="#ffffff", fg="#94a3b8", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w", pady=12)
            return
        for log in self.alert_logs[-8:][::-1]:
            row = tk.Frame(parent, bg="#ffffff")
            row.pack(fill="x", pady=5)
            tk.Label(row, text="●", bg="#ffffff", fg="#dc2626", font=("Microsoft YaHei", 14, "bold")).pack(side="left")
            tk.Label(row, text=f"{log['time']}  {log['state']}", bg="#ffffff", fg="#334155", font=("Microsoft YaHei", 10), wraplength=430, justify="left").pack(side="left", padx=8)

    def clear_logs(self):
        self.alert_logs.clear()
        for key in self.risk_counts:
            self.risk_counts[key] = 0
        self.refresh_log_table()
        messagebox.showinfo("已清空", "危险日志已清空")

    def on_close(self):
        self.video_running = False
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DriverDashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
