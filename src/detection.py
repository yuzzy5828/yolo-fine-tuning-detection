import cv2
import torch
from ultralytics import YOLO

# -----------------------------------------
# 1. モデルの読み込み
# -----------------------------------------
model = YOLO("/home/onishi/venv/karura_detection/runs/detect/train3/weights/best.pt")
model.to('cuda')  # GPUがある場合にGPU上で動作させる

# -----------------------------------------
# 2. Webカメラの準備
# -----------------------------------------
cap = cv2.VideoCapture(0)  # PC 内蔵カメラ or USBカメラ
if not cap.isOpened():
    print("カメラが開けませんでした。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # 3. 推論 (物体検出)
    # -----------------------------------------
    # フレームをモデルに渡して推論
    results = model(frame, conf=0.5)

    # 1画像に対しての結果はリストの0番目に格納される
    res = results[0]

    # -----------------------------------------
    # 4. 描画済みの画像を取得
    # -----------------------------------------
    # res.plot() で、推論結果（バウンディングボックスやラベルなど）が描画された
    # NumPy配列画像を取得できる
    annotated_frame = res.plot()

    # -----------------------------------------
    # 5. 結果の表示
    # -----------------------------------------
    cv2.imshow("fine tuned Real-time Detection", annotated_frame)

    # 'q'キーが押されたらループ終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------------
# 6. 後処理
# -----------------------------------------
cap.release()
cv2.destroyAllWindows()
