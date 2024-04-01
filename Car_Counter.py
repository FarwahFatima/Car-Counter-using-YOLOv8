import cv2
import math
from ultralytics import YOLO
import numpy as np 
import sort

model = YOLO("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/.venv/yolo-weights/yolov8n.pt")
cap = cv2.VideoCapture("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/car_counter/Cars Driving.mp4")

mask = cv2.imread("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/car_counter/2.png") # image mask (the part of frame use to detect)
limits = [350, 400, 700 , 400]  # limits of red line (if car passes this line consider as counted) 

tracker = sort.Sort(max_age=20, min_hits=4, iou_threshold=0.3) # tracking car so that each carr count individually
 
counter = [] # list of counting car crossed
while True:
    success, frame = cap.read()
    if not success:
        break

    img_reg = cv2.bitwise_and(frame, mask)
    results = model(img_reg, stream = True)

    detections = np.empty((0, 5)) 

    for r in results:
        result = r.boxes 
        for box in result:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            if cls in [2] and conf> 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2) # red line 
    
    tracking_result = tracker.update(detections) # tracked car
    
    for result in tracking_result: # looping through tracked cars

        x1, y1, x2, y2, id= result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cv2.rectangle(frame , (x1, y1), (x2, y2), (255, 0, 0), 2)  # bounding box around car
        cv2.putText(frame, f' {int(id)}', (max(0, x1), max(35, y1)), # id of that detect car
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        x_circle, y_circle = x1 + w // 2, y1 + h // 2 
        cv2.circle(frame, (x_circle, y_circle), 4, (200, 0, 200), cv2.FILLED) # center circle inside bounding box

        if limits[0] < x_circle < limits[2] and limits[1] - 24 < y_circle < limits[3] + 24: # x position of circle point is lesser than limits x coordinate and y position of circle point is in between 24 of limits y
            if counter.count(id) == 0:  # if id count is 0 in counter list
                counter.append(id) 

    cv2.rectangle(frame, (40,30), (250, 70), (255, 200, 255), -1)  # backgound of Count: (light pink)
    cv2.putText(frame, f' Count: {len(counter)}', (55, 55), # Count car crosses
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    img = cv2.resize(frame, (416, 416))
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total count of Counted Cars: {len(counter)}")