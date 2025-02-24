from ultralytics import YOLO
model = YOLO('yolov8n.pt') # yolov8n/s/m/l/xのいずれかを指定。多クラスの検出であるほど大きいパラメータが必要
model.train(data='/home/onishi/venv/karura_detection/data/data.yaml', epochs=300, batch=20)