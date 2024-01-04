from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')     # pretrarined model for detecting cars in an image/video
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('sample.mp4')
vehicles = [2, 3, 5, 7]

# read frames from the video
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # if frame_nmr > 10:
        #     break
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []       # to save all the bonding boxes of the vehicles
        # print(detections)
        for detection in detections.boxes.data.tolist():
            # print(detection)
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # assign each license plate to a car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # crop the license plate
                license_plate_crop = frame[int(y1): int(y2), int(x1): int(x2), :]
                
                # process the license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 244, cv2.THRESH_BINARY_INV)      # pixels lower than 64 goes to 255 and the pixels higher than 64 goes to 0
                
                """
                cv2.imshow('original_crop', license_plate_crop)
                cv2.imshow('threshold', license_plate_crop_thresh)
                cv2.waitKey(0)
                """
                    
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score
                                                                    }}
                
# write results
write_csv(results, './test.csv')