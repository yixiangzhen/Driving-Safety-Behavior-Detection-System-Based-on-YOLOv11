from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"F:\code\yolov11\runs\detect\train2\weights\best.pt")

    model.train(
        data=r"F:\code\yolov11\data.yaml",
        epochs=40,
        batch=32,
        workers=0,
        imgsz=640,
        pretrained=True
    )