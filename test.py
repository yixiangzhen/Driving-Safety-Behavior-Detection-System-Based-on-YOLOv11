from ultralytics import YOLO
import cv2

# 加载你训练好的模型，优先用 best.pt
model = YOLO(r"F:\code\yolov11\runs\detect\train3\weights\best.pt")

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

# 如果摄像头打不开，可以把 0 改成 1 或 2
if not cap.isOpened():
    print("无法打开摄像头，请检查摄像头是否被占用")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # YOLO 实时检测
    results = model(frame, conf=0.1)

    # 取第一张图的检测结果
    result = results[0]

    for box in result.boxes:
        # 坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 类别和置信度
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        # 显示文字
        label = f"{class_name} {conf:.2f}"

        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画标签背景
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)

        # 写类别和置信度
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    # 显示画面
    cv2.imshow("Driver Behavior Detection", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()