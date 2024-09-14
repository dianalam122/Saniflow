from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture("Saniflow/Videos/testVid.mp4") 

YOLOmodel = YOLO('yolov8n.pt')


classNames = ["person", "dog"]

while True:
    success, img = cap.read()
    results = YOLOmodel(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l = 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)




