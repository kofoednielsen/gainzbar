# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pathlib import Path
from faces import who_is
from tqdm import tqdm
from itertools import chain
import requests
import numpy as np
import threading
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments

def ping_niels(name):
    headers = {
        'Content-Type': 'application/json'
    }
    data = {'name': name}
    response = requests.post('https://us-central1-pull-ups-a9ce0.cloudfunctions.net/countPullups', headers=headers, json=data)
    print(response)

def face_detection(motion_images):
    faces = [who_is(f) for f in tqdm(motion_images)]
    # get the ones that where there was a face
    face_detections = list(filter(lambda f: f[0], faces))
    if len(face_detections) > 0:
        # Get the index of the best face detections.
        face_best_index = np.argmin([f[1] for f in face_detections])
        # get the best face
        face_best = face_detections[face_best_index]
        print(face_best)
        # threshold in case a hand looked like niels
        if face_best[1] < 0.95:
            ping_niels(face_best[2])

# movemen detection stuff
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture()
while not vs.open('http://192.168.1.80:8080/video'):
    pass

#vs = VideoStream(src=0).start()
background = None

time.sleep(2.0)
last_motion_time = 0
motion_images = []
# loop over the frames from the video stream
while True:
    start = time.time()
    (ret, frame) = vs.read()
    if not ret:
        print('skip')
        continue
    frame = imutils.resize(frame, width=(160))
    
    fg = fgbg.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    # convert to 1d array of all pixels
    pixels = np.concatenate(fg, axis=0)
    # get non black
    nonblack = sum(pixels)
    cv2.imshow('org', frame)
    # random number i found to nicely indicate
    # a pullup happening
    if nonblack > 500000:
        motion_images.append(frame)
        last_motion_time = time.time()
    # After half a second of no motion, if there was motion_images then do
    if time.time()-last_motion_time > 0.5 and len(motion_images) > 0:
        print('pullup!')
        x = threading.Thread(target=face_detection, args=(motion_images,))
        x.start()
        #face_detection(motion_images)
        motion_images = []

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
