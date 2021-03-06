
# deps
from os import getenv
from pathlib import Path
import argparse
import cv2
import imutils
import numpy as np
import requests
import threading
import time

# app imports
from face_recogniton import check_faces

PULLUP_ENDPOINT = getenv('PULLUP_ENDPOINT')
assert PULLUP_ENDPOINT, 'environment variable PULLUP_ENDPOINT must be set.'

PULLUP_ENDPOINT_SECRET = getenv('PULLUP_ENDPOINT_SECRET')
assert PULLUP_ENDPOINT, 'environment variable PULLUP_ENDPOINT_SECRET must be set.'

def ping_niels(name):
    """ send http POST request to update the
        web interface """
    headers = {
        'Content-Type': 'application/json'
    }
    data = {'name': name}
    response = requests.post(PULLUP_ENDPOINT, headers=headers, json=data)
    print(response)

def face_detection(motion_images):
    # check for faces in all motion images
    face_checks = check_faces(motion_images)
    # get the ones that had a face in them
    faces = list(filter(lambda f: f['got_face'], face_checks))
    if len(faces) > 0:
        # Get the index of the best face detections.
        # best is the lowest distance
        face_best_index = np.argmin([face['dist'] for face in faces])
        # get the best face
        face_best = faces[face_best_index]
        print(f'Best was distance {round(float(face_best["dist"]),3)} for {face_best["name"]}')
        # threshold in case a hand looked like niels
        if face_best['dist'] < 0.90:
            ping_niels(face_best['name'])


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
    (ret, frame) = vs.read()
    if not ret:
        print('network broke, reconnecting!')
        vs.open('http://192.168.1.80:8080/video')
        continue
            
    frame = imutils.resize(frame, width=(160))
    
    fg = fgbg.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    # convert to 1d array of all pixels
    pixels = np.concatenate(fg, axis=0)
    # get non black
    nonblack = sum(pixels)
    #cv2.imshow('org', frame)
    # random number i found to nicely indicate
    # a pullup happening
    if nonblack > 500000:
        motion_images.append(frame)
        last_motion_time = time.time()
    # After half a second of no motion, if there was motion_images then do
    if time.time()-last_motion_time > 0.5 and len(motion_images) > 0:
        print('pullup!')
        start = time.time()
        face_detection(motion_images)
        end = time.time()
        print(f'processing took {round(end-start, 3)}')
        motion_images = []

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
