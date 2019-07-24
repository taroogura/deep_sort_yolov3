#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import argparse
import warnings
import sys
import subprocess
from glob import glob
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
writeVideo_flag = True 

def track(yolo, video_path, image_output_dir):
    tracker = Tracker(metric)
    
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        out = cv2.VideoWriter(video_path + '.avi', fourcc, 15, (w, h))
        list_file = open(video_path + 'detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    frame_id = 0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # person_output_dir = os.path.join(root_image_output_dir, os.path.splitext(os.path.basename(video_path))[0])

            person_output_dir = os.path.join(image_output_dir, "%04d" % track.track_id)
            try:
                subprocess.call(["mkdir", "-p", person_output_dir])
            except OSError as e:
                print("couldn't make ", person_output_dir)
                print(e)
                pass


            bbox = track.to_tlbr()
            
            image_filename = "{}_{:04d}-{:04d}.jpg".format(os.path.basename(video_path), track.track_id, frame_id)
            output_image_path = os.path.join(person_output_dir, image_filename)
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            cv2.imwrite(output_image_path, cropped_image)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        # cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        frame_id += 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def crpp_images(video_dir, root_image_output_dir):
    video_paths = glob(os.path.join(video_dir, "*.mkv"))    # いったんmkvだけ
    try:
        subprocess.call(["mkdir", "-p", root_image_output_dir])
    except OSError as e:
        print("couldn't make ", root_image_output_dir)
        print(e)
        pass
    for video_path in video_paths:
        if os.path.isfile(video_path + ".avi"):
            print(video_path + " already processed!")
            continue
        
        image_output_dir = os.path.join(root_image_output_dir, os.path.splitext(os.path.basename(video_path))[0])
        try:
            subprocess.call(["mkdir", "-p", image_output_dir])
        except OSError as e:
            print("couldn't make ", image_output_dir)
            print(e)
            pass

        track(YOLO(), video_path, image_output_dir)


if __name__ == '__main__':
    '''
    python demo1.py --videodir video --imagedir output/detected_images
    '''
    parser = argparse.ArgumentParser(description='Human Detection using YOLO in OPENCV')
    parser.add_argument('--videodir', help='Path to video file.')
    parser.add_argument('--imagedir', help='Path to output image file.')
    args = parser.parse_args()
    crpp_images(args.videodir, args.imagedir)
    
