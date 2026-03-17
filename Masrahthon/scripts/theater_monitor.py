"""
نظام مراقبة المسرح بالذكاء الاصطناعي
- كشف الأشخاص والجوالات
- كشف الدخان
- خريطة حرارة الجمهور
"""
import cv2
import sys
import torch
import functools
import numpy as np
from ultralytics import YOLO as YOLO_Base

# حل مشكلة الأمان في PyTorch
torch.load = functools.partial(torch.load, weights_only=False)

try:
    from ultralyticsplus import YOLO as YOLO_Plus
except ImportError:
    print("❌ خطأ: مكتبة ultralyticsplus غير مثبتة.")
    sys.exit()

# إعداد الموديلات
device = 'cuda' if torch.cuda.is_available() else 'cpu'
obj_model = YOLO_Base('yolov8s.pt')
smoke_model = YOLO_Plus('kittendev/YOLOv8m-smoke-detection')

obj_model.to(device)
smoke_model.to(device)

# إعداد الكاميرا
cap = cv2.VideoCapture(0)
W, H = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

heatmap = np.zeros((H, W), dtype=np.float32)
frame_count = 0
smoke_duration_frames = 0
obj_results = None
smoke_results = None

print(f"🚀 نظام المراقبة المتطور يعمل.. (خريطة الحرارة + تحليل التفاعل)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    display_frame = frame.copy()
    smoke_detected_now = False

    if frame_count % 2 == 0:
        obj_results = obj_model.predict(frame, imgsz=960, conf=0.25, classes=[0, 67], half=(device == 'cuda'), verbose=False)

    if frame_count % 3 == 0:
        smoke_results = smoke_model.predict(frame, imgsz=480, conf=0.4, verbose=False)

    current_person_count = 0
    current_phones = 0

    if obj_results is not None:
        for box in obj_results[0].boxes:
            c = int(box.cls[0])
            b = box.xyxy[0].cpu().numpy().astype(int)

            if c == 0:
                current_person_count += 1
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                cv2.circle(heatmap, (cx, cy), 50, 2, -1)
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (255, 200, 0), 1)

            elif c == 67:
                current_phones += 1
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(display_frame, "PHONE", (b[0], b[1]-10), 0, 0.6, (0, 0, 255), 2)

    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    display_frame = cv2.addWeighted(display_frame, 0.8, heatmap_color, 0.2, 0)

    if smoke_results is not None and len(smoke_results[0].boxes) > 0:
        smoke_detected_now = True
        smoke_duration_frames += 1
        for box in smoke_results[0].boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 2)

    if current_person_count > 0:
        engagement = max(0, int(((current_person_count - (current_phones * 1.5)) / current_person_count) * 100))
    else:
        engagement = 0

    cv2.rectangle(display_frame, (10, 10), (380, 200), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Theater Analysis", (20, 45), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Audience Count: {current_person_count}", (20, 80), 0, 0.6, (0, 255, 0), 1)
    cv2.putText(display_frame, f"Engagement: {engagement}%", (20, 115), 0, 0.6, (255, 150, 0), 2)

    cv2.rectangle(display_frame, (20, 130), (220, 145), (50, 50, 50), -1)
    cv2.rectangle(display_frame, (20, 130), (20 + int(engagement * 2), 145), (0, 255, 0), -1)

    smoke_sec = smoke_duration_frames / 10
    cv2.putText(display_frame, f"Smoke Timer: {smoke_sec:.1f}s", (20, 180), 0, 0.7, (0, 0, 255), 2)

    if smoke_detected_now:
        cv2.putText(display_frame, "SMOKE ALERT", (W//2 - 100, 50), 0, 1.2, (0, 0, 255), 3)

    cv2.imshow("Advanced AI Theater Monitor", display_frame)

    heatmap *= 0.98

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
