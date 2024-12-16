from ultralytics import YOLO

def main():
    model = YOLO('models/yolo11n.pt')
    result = model.train(data='images/data.yaml',epochs=30,imgsz=640,workers=2)

if __name__ == '__main__':
    main()