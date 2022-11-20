from Detector import *
import cv2
import time
import numpy as np


detector = Yolo(r"Ml_model\yolov3-tiny-obj.cfg", r"Ml_model\yolov3-tiny-obj_final.weights", ["box"])
cap = cv2.VideoCapture(r"video\Conveyor.mp4")
#cap = cv2.VideoCapture(r"video\Conveyor.mp4")    #for live cam streaming
height =540
width = 960
fpss = 10
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
videosave = cv2.VideoWriter(r'Inference.mp4',fourcc, fpss, (width, height))


FPS = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
ft = 10
num = 0


def draw_on_frame(frame, results):
    global num
    a, b, c, d = 534, 632, 554, 735  #ROI

    for cls, objs in results.items():
        for x1, y1, x2, y2 in objs:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=2)
            cv2.putText(frame, 'Current No of Detected boxes in Conveyor ' + ': ' + str(len(objs)), (1170, 150),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0xFF, 0xFF), 3, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, 'FPS ' + ': ' + str(FPS), (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, 'Total No of Detected boxes in Conveyor ' + ': ' + str(num), (1170, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0xFF, 0xFF), 3, cv2.FONT_HERSHEY_SIMPLEX)
            

            # FINDING CENTROID OF BBOX LIES UNDER ROI
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            if a < x_center < c and b < y_center < d:
                num += 1



    return frame

while (cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    ret, frame = cap.read()
    results = detector.detect(frame, conf=0.4)
    print(results.items())
    frame = draw_on_frame(frame, results)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    print(frame.shape)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(ft) & 0xFF
    if key == 113:
        break
    videosave.write(frame)
cap.release()
videosave.release()
cv2.destroyAllWindows()
