from ultralytics import YOLO
 
model = YOLO('runs/segment/train/weights/best.pt')
results = model.predict(source='datasets/mulReference-seg/images/test/8426805_492631301.jpg', save=True, show=True, imgsz=640)