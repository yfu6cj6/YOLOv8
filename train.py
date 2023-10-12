from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8x.pt")
    model.train(data="data.yaml",
                mode="detect",
                epochs=100,
                imgsz=640,
                device='0')