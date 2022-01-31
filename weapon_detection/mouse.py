import numpy as np
import cv2
import time
import os
# mouse callback function
drawing = False # true if mouse is pressed
ix,iy = -1,-1
img = None
current_class = -1
flag = False
coordinates = {}
# mouse callback function
def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img, current_class
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y


    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            train_capture.set(cv2.CAP_PROP_POS_MSEC, current_video_position * 1000)
            _, img_tmp = train_capture.read()
            cv2.rectangle(img_tmp, pt1=(ix,iy), pt2=(x, y),color=(0,255,0),thickness=4)
            img = img_tmp

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        train_capture.set(cv2.CAP_PROP_POS_MSEC, current_video_position * 1000)
        _, img_tmp = train_capture.read()
        cv2.rectangle(img_tmp, pt1=(ix,iy), pt2=(x, y),color=(0,255,0),thickness=4)
        current_class = 1
        img = img_tmp
    elif event == cv2.EVENT_FLAG_RBUTTON:
        current_class = 0


DATA_PATH = './data/1/'
VIDEO_PATH = os.path.join(DATA_PATH, 'video.mp4')
train_capture = cv2.VideoCapture(VIDEO_PATH)
SECONDS_PER_FRAME = 1
FPS = int(train_capture.get(cv2.CAP_PROP_FPS))
DURATION = int(train_capture.get(cv2.CAP_PROP_FRAME_COUNT)) // FPS
for current_video_position in range(0, DURATION + SECONDS_PER_FRAME, SECONDS_PER_FRAME):
    train_capture.set(cv2.CAP_PROP_POS_MSEC, current_video_position * 1000)
    _, img = train_capture.read()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_reactangle_with_drag)
    while(current_class == -1):
        cv2.imshow('image',img)
        print(current_class)
        k = cv2.waitKey(1) & 0xFF
    print(current_class)
    current_class = -1
cv2.destroyAllWindows()
    
