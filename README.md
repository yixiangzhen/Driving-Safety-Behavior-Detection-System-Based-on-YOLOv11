# Driving-Safety-Behavior-Detection-System-Based-on-YOLOv11
Driving Safety Behavior Detection System Based on YOLOv11


## 项目简介

本项目是一个基于 Python、OpenCV、Tkinter 和 YOLOv11 实现的驾驶员安全行为监测系统。系统可以对图片、视频和摄像头实时画面进行驾驶行为识别，并根据检测结果判断驾驶员是否存在疲劳驾驶、分心驾驶、低头、使用手机、抽烟、打哈欠等不安全行为。

项目不仅包含模型训练和摄像头测试脚本，还实现了一个完整的桌面端可视化系统，支持图片检测、视频检测、实时监测、风险日志、检测参数调节和 AI 辅助分析等功能。

## 功能特点

- 支持驾驶员状态识别
- 支持图片检测
- 支持视频检测并保存检测结果
- 支持摄像头实时监测
- 支持检测框、类别名称、置信度显示
- 支持 FPS 显示
- 支持不同类别单独设置置信度阈值
- 支持连续帧风险判断，减少单帧误报
- 支持启用或关闭指定检测类别
- 支持灵敏、平衡、严格等检测模式
- 支持风险日志记录和清空
- 支持 AI 生成检测报告、风险总结和阈值调参建议

## 识别类别

系统共识别 7 类驾驶员行为状态：

| 类别名称 | 含义 |
| --- | --- |
| `awake` | 清醒驾驶 |
| `distracted` | 分心驾驶 |
| `drowsy` | 困倦状态 |
| `head drop` | 点头或低头 |
| `phone` | 使用手机 |
| `smoking` | 抽烟 |
| `yawn` | 打哈欠 |

## 项目结构

```text
yolov11/
├── app(1).py      # 桌面端主程序，包含界面、检测入口、日志和 AI 助手
├── backend.py     # 检测核心逻辑，封装 YOLO 模型加载、图片/视频/实时帧检测
├── train.py       # 模型训练脚本
├── test.py        # 摄像头实时检测测试脚本
├── tool.py        # YOLO 标签类别批量转换工具
├── data.yaml      # 数据集配置文件
├── yolo11n.pt     # YOLOv11 n 预训练权重
├── yolo11s.pt     # YOLOv11 s 预训练权重
├── datasets/      # 数据集目录
├── runs/          # 训练和检测输出目录
├── ultralytics/   # YOLO/Ultralytics 相关代码
└── README.md      # 项目说明文档
