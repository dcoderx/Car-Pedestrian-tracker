import cv2
from random import randrange

video = cv2.VideoCapture('cars.mp4')

car_classifier_file = 'haarcascade_car.xml'
pedestrian_classifier = 'Fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

while True:
    rs, frame = video.read()

    if rs:
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    cars = car_tracker.detectMultiScale(frame)
    pedestrian = pedestrian_tracker.detectMultiScale(frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0 , 255), 4)

    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

        cv2.imshow('tracker', frame)

        cv2.waitKey(1) & 0xFF




