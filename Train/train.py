from ultralytics import YOLO

model = YOLO('yolo11s-seg.pt')

hist = model.train(
    data="/content/file/data.yaml",
    imgsz=640,
    epochs=150,
    patience = 20
)