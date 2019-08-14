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
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from mtcnn.mtcnn import MTCNN
import pandas as pd
import re

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

# mtcnn test
detector = MTCNN()

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
writeVideo_flag = True 

def sharpness_lap(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)

    return lap.var()

def square_padding(img):
    """
    長い方の辺の長さの正方形にパディングする
    """
    size = img.shape
    long_len = size[0] if (size[0] > size[1]) else size[1]
    
    square_img = np.zeros((long_len, long_len, 3), np.uint8)

    oy_idx = int((long_len - size[0]) / 2)
    ox_idx = int((long_len - size[1]) / 2)
    square_img[oy_idx:oy_idx + size[0], ox_idx:ox_idx + size[1]] = img

    return square_img

def rectangle_padding(img, ratio=2.0):
    """
    縦横比がratioの長方形となるようにパディングする
    """
    size = img.shape
    if (size[0] >= size[1] * 2):
        hei_len = int(size[0]) if (size[0] % 2 == 0) else int(size[0] + 1)
        wid_len = int(hei_len / 2)
    else:
        wid_len = int(size[1])
        hei_len = int(wid_len * 2)
    
    rectangle_img = np.zeros((hei_len, wid_len, 3), np.uint8)

    oy_idx = int((hei_len - size[0]) / 2)
    ox_idx = int((wid_len - size[1]) / 2)
    rectangle_img[oy_idx:oy_idx + size[0], ox_idx:ox_idx + size[1]] = img

    return rectangle_img

def black_padding(img, ratio=1.5):
    """
    両辺の長さをratio倍にして、広げた文を黒くしておく
    """
    size = img.shape
    long_l = int(size[0] * ratio)
    long_w = int(size[1] * ratio)

    padded_img = np.zeros((long_l, long_w, 3), np.uint8)

    oy_idx = int((long_l - size[0]) / 2)
    ox_idx = int((long_w - size[1]) / 2)
    padded_img[oy_idx:oy_idx + size[0], ox_idx:ox_idx + size[1]] = img

    return padded_img, oy_idx, ox_idx
    
def track(yolo, video_path, image_output_dir):
    tracker = Tracker(metric)
    
    # video_capture = cv2.VideoCapture(0)
    video_name = os.path.basename(video_path)
    splitted_videoname = re.split("_|-", os.path.splitext(video_name)[0])

    camera_name = splitted_videoname[0]
    # 1/10 秒の桁までを時間ラベルとして保持しておきたい
    base_time_label = int(splitted_videoname[1][:-1]) + int((int(splitted_videoname[1][-1]) + 5) / 10)
    
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
    pre_frames_num = 5  # motionの設定で、motion検知前の5フレームくらいも保存しておくようにしておく
    ave_pre_frame = None
    ave_pre_image = None
    
    # image_label_df = pd.DataFrame()
    images_per_video_df = pd.DataFrame()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        write_frame = frame.copy()

        # 背景差分テスト
        if (frame_id <= pre_frames_num):
            if (frame_id == 0):
                ave_pre_frame = np.float32(frame)
            elif (frame_id < pre_frames_num):
                ave_pre_frame += np.float32(frame)
            elif (frame_id == pre_frames_num):
                ave_pre_frame = ave_pre_frame / pre_frames_num
                ave_pre_frame = ave_pre_frame.astype(np.uint8)
                ave_pre_image = Image.fromarray(ave_pre_frame[...,::-1])  #bgr to rgb
                # cv2.imwrite("./ave_frame.jpg", ave_pre_frame)

            frame_id += 1
            continue


       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs, scores = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxs, scores, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if(len(boxes) > 0):
            indices, overlaps = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        else:
            indices = []
        detections = [detections[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
            
        # 動画は10fpsという前提
        time_label = str(base_time_label + frame_id)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # person_output_dir = os.path.join(root_image_output_dir, os.path.splitext(os.path.basename(video_path))[0])
            person_output_dir = os.path.join(image_output_dir, os.path.basename(image_output_dir) + "_%04d" % track.track_id)
            try:
                subprocess.call(["mkdir", "-p", person_output_dir])
            except OSError as e:
                print("couldn't make ", person_output_dir)
                print(e)
                pass

            bbox = track.to_tlbr()
            body_t, body_l, body_b, body_r = (int(max(0, bbox[1])), int(max(0, bbox[0])), int(min(h, bbox[3])), int(min(w, bbox[2])))
            body_w, body_h = (body_r - body_l, body_b - body_t)

            # image_filename = "{}_{:04d}-{:04d}.jpg".format(video_name, track.track_id, frame_id)
            image_filename = "{}_{:04d}_{:04d}.jpg".format(os.path.basename(person_output_dir), track.age, frame_id)
            # image_filename = os.path.basename(person_output_dir) + "-%04d" % track.age)÷

            output_image_path = os.path.join(person_output_dir, image_filename)

            # cropped_image = frame[body_t:body_b, body_l:body_r]
            cropped_image = frame[body_t:body_b, body_l:body_r].copy()

            # 背景差分テスト
            # cropped_ave_pre_image = ave_pre_frame[body_l:body_r, body_t:body_b]
            # blurred_diff = cv2.GaussianBlur(cropped_image, (3, 3), 0) - cv2.GaussianBlur(cropped_ave_pre_image, (3, 3), 0)
            # blurred_diff = cropped_image.astype(np.float32) - cropped_ave_pre_image.astype(np.float32)

            # blurred_diff = cv2.GaussianBlur(blurred_diff, (3, 3), 0)

            # blurred_diff = np.linalg.norm(blurred_diff, axis=2)
            # blurred_diff = cv2.GaussianBlur(blurred_diff, (5, 5), 0)
            # blurred_diff = cv2.normalize(blurred_diff, None , 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # ret, th = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # ret, th = cv2.threshold(blurred_diff, 16, 255,cv2.THRESH_BINARY)
            
            # masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask = th)

            if (body_l + 1 > body_r or body_t + 1 > body_b):
                continue


            # square_padded_img = square_padding(cropped_image)
            # black_padded_img = black_padding(square_padded_img)
            # result = detector.detect_faces(black_padded_img)
            black_padded_img, padded_len, padded_wid = black_padding(cropped_image)
            result = detector.detect_faces(black_padded_img)
            face_confidence = 0
            face_lapvar = 0
            face_output_image_path = ""
            face_l, face_r, face_t, face_b = (0,0,0,0)
            real_face_l, real_face_r, real_face_t, real_face_b = (0,0,0,0)
            is_face_detected = False
            if(len(result) > 0):
                face_bbox = result[0]['box']
                face_confidence = result[0]['confidence']

                face_l, face_r, face_t, face_b = (int(face_bbox[0] - 0.15 * face_bbox[2]), int(face_bbox[0] + 1.15 * face_bbox[2]), \
                                                  int(face_bbox[1] - 0.15 * face_bbox[3]), int(face_bbox[1] + 1.15 * face_bbox[3]))
                real_face_l, real_face_r, real_face_t, real_face_b = (body_l + face_l - padded_wid, body_l + face_r - padded_wid, \
                                                                      body_t + face_t - padded_len, body_t + face_b - padded_len)

                face_clipped_image = black_padded_img.copy()[face_t : face_b, face_l : face_r]
                square_padded_img = square_padding(face_clipped_image)
                if (face_clipped_image.size > 0):
                    is_face_detected = True
                    face_output_image_path = output_image_path + "_face.jpg"
                    cv2.imwrite(face_output_image_path, square_padded_img)
                    # face_lapvar = sharpness_lap(face_clipped_image)
                    face_lapvar = sharpness_lap(black_padded_img[int(face_bbox[1]):int(face_bbox[1] + face_bbox[3]), \
                                                                 int(face_bbox[0]):int(face_bbox[0] + face_bbox[2])] )
            padded_cropped_image = rectangle_padding(cropped_image, 2.0)
            cv2.imwrite(output_image_path, padded_cropped_image)
            lapvar = sharpness_lap(cropped_image)
            
            # cv2.imwrite(output_image_path, masked_image)
            # cv2.imwrite(output_image_path + "_mask.jpg", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
            # cv2.imwrite(output_image_path + "_blurred.jpg", cv2.cvtColor(blurred_diff, cv2.COLOR_GRAY2BGR))

            cv2.rectangle(write_frame, (body_l, body_t), (body_r, body_b),(255,255,255), 2)
            cv2.putText(write_frame, str(track.track_id), (body_l, body_t), 0, 5e-3 * 200, (0, 255, 0), 2)

            body_confidence = 0.0
            body_overlap = 0.0
            if (len(scores) > 0):
                # cv2.putText(frame, str('%.6f' % lapvar), (body_l, body_t + 40), 0, 5e-3 * 100, (255, 255, 0), 2)
                cv2.putText(write_frame, str('%.6f' % face_confidence), (body_l, body_t + 40), 0, 5e-3 * 100, (255, 255, 0), 2)
                cv2.putText(write_frame, str('%.6f' % face_lapvar), (body_l, body_t + 60), 0, 5e-3 * 100, (255, 0, 255), 2)
                if (len(scores) > track.current_detection_idx):
                    body_confidence = scores[track.current_detection_idx]
                    cv2.putText(write_frame, str('%.6f' % body_confidence), (body_l, body_t + 20), 0, 5e-3 * 100, (0, 255, 255), 2)
                    body_overlap = overlaps[track.current_detection_idx]
                    cv2.putText(write_frame, str('%.6f' % body_overlap), (body_l, body_t + 80), 0, 5e-3 * 100, (0, 0, 255), 2)
                if (len(result) > 0 and face_clipped_image.size > 0):
                    cv2.rectangle(write_frame, (real_face_l, real_face_t), (real_face_r, real_face_b), (0, 0, 255), 2)

            d = {"camera_name":camera_name,  "video_name": video_name, "image_filename": image_filename, "is_face_detected": is_face_detected, "time_label":time_label,
                 "track_id": os.path.basename(person_output_dir), "track_age": track.age, "frame_id": os.path.basename(image_output_dir) + "_%04d" % frame_id,
                 "body_t": body_t, "body_l": body_l, "body_w": body_w, "body_h": body_h, "face_t": real_face_t, "face_l": real_face_l, "face_w": int(real_face_r - real_face_l), "face_h": int(real_face_b - real_face_t),
                 "body_confidence": body_confidence, "body_overlap": body_overlap, "face_confidence": face_confidence, "body_lapvar":lapvar, "face_lapvar":face_lapvar}
            images_per_video_df = images_per_video_df.append(d, ignore_index=True)

        for det in detections:
            bbox = det.to_tlbr()
            body_t, body_l, body_b, body_r = (int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
            cv2.rectangle(write_frame,(body_t, body_l), (body_b, body_r),(255,0,0), 2)
                    
        if writeVideo_flag:
            # save a frame
            out.write(write_frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_id += 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    csv_path = video_path + '.csv'
    images_per_video_df.to_csv(csv_path)

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def crop_images(video_dir, root_image_output_dir):
    Y = YOLO()
    video_paths = glob(os.path.join(video_dir, "*.mkv"))  # いったんmkvだけ
    try:
        subprocess.call(["mkdir", "-p", root_image_output_dir])
    except OSError as e:
        print("couldn't make ", root_image_output_dir)
        print(e)
        pass
    for video_path in tqdm(video_paths):
        # if os.path.isfile(video_path + ".avi"):
        #     print(video_path + " already processed!")
        #     continue
        
        image_output_dir = os.path.join(root_image_output_dir, os.path.splitext(os.path.basename(video_path))[0])
        try:
            subprocess.call(["mkdir", "-p", image_output_dir])
        except OSError as e:
            print("couldn't make ", image_output_dir)
            print(e)
            pass

        track(Y, video_path, image_output_dir)


if __name__ == '__main__':
    '''
    python demo1.py --videodir video --imagedir output/detected_images
    '''
    parser = argparse.ArgumentParser(description='Human Detection using YOLO in OPENCV')
    # parser.add_argument('--videodir', help='Path to video file.')
    parser.add_argument('--cameradir', help='Path to camera(video files) directory')
    parser.add_argument('--imagedir', help='Path to output image file.')
    args = parser.parse_args()
    # crpp_images(args.videodir, args.imagedir)
    crop_images(args.cameradir, args.imagedir)
    
