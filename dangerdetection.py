from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)  # For Webcam
#ap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture(0)  # For Video

model = YOLO("yolov8mod.pt")

classNames = ['spoon-HARMLESS', 'knife-HARMFULL', 'fork-HARMFULL', 'peller-HARMFULL', 'sicssors-HARMFULL', 'plough-HARMFULL', 't-fork-HARMLESS',
              'orange-HARMLESS', 'apple-HARMLESS', 'lemon-HARMLESS', 'plants-HARMLESS', 'photos-HARMLESS', 'banana-HARMLESS', 'glass-HARMLESS', 'sofa-HARMLESS', 
              'lamp-HARMLESS', 'clock-HARMLESS', 'chair-HARMLESS', 'pillow-HARMLESS']
myColor = (0, 0, 255)

while True:
    ret, frame0 = cap.read()
    results = model(frame0, stream=True)
    Harmfull=0
    Harmless=0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.5:
                if currentClass =='knife-HARMFULL' or currentClass =='fork-HARMFULL' or currentClass == "peller-HARMFULL" or currentClass == "sicssors-HARMFULL" or currentClass == "plough-HARMFULL":
                    myColor = (0, 0, 255)
                    Harmfull=Harmfull+1
                    
                else:
                    myColor = (0, 255, 0)
                    Harmless=Harmless+1

                cvzone.putTextRect(frame0, f'{classNames[cls],} {conf,} {Harmfull,} {Harmless}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(frame0, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow('webcam', frame0) 
    k=cv2.waitKey(10)
    if k==27:
        break;
cap.release()
cv2.destroyAllWindows()


