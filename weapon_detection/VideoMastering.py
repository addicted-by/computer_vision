import cv2
from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
from cv2 import CAP_PROP_POS_MSEC
import numpy as np
import torch
import os
import csv
def parse_file(path_file):
    result = []

    with open(path_file, 'r') as f:
        for line in f.readlines():
            data = line.split('\n')[0].split(' ')
            result.append([int(data[0]), int(data[1]), int(data[2])])
            assert result[-1][0] <= result[-1][1], result[-1]
            if len(result) > 1:
                assert result[-1][0] == result[-2][1], (result[-1], result[-2])
    return result


DATA_PATH = './data/1/'
path_mp4 = os.path.join(DATA_PATH, 'video.mp4')
path_txt = os.path.join(DATA_PATH, 'predictions.txt')
file = parse_file(path_txt)
cap = cv2.VideoCapture(path_mp4)
FPS = int(cap.get(CAP_PROP_FPS))
print("FPS: ", FPS)
duration = int(cap. get(CAP_PROP_FRAME_COUNT) / FPS)
print("duration", duration)

sec_per_frame = 0.24
cur_segment_id = 0
cur_picture_id = 1
cur_video_pos = 0
       
with open("./data/data.csv", mode="w", newline = '', encoding='utf-8') as w_file:
        file_writer = csv.DictWriter(w_file, fieldnames=['image_name', 'class_id'])
        file_writer.writeheader() 
        while cap.isOpened():
                cap.set(CAP_PROP_POS_MSEC, cur_video_pos * 1000)
                ret, frame = cap.read()
                while cur_video_pos > file[cur_segment_id][1]:
                        cur_segment_id += 1
                #cv2.imshow('video', frame)
                image_name = str(cur_picture_id) + '.jpg'
                print(image_name, ' saved with ', file[cur_segment_id][2])
                cv2.imwrite(os.path.join('./data', image_name), frame)
                file_writer.writerow({'image_name': image_name, 'class_id': file[cur_segment_id][2]})
                #while
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cur_video_pos += sec_per_frame
                cur_picture_id += 1

cap.release()
cv2.destroyAllWindows()