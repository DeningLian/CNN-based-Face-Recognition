# -*- coding: utf-8 -*-
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import shutil

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

def extract_face_from_img(source):
    img = cv2.imread(source)

    # run detector
    results = detector.detect_face(img)

    if results is not None:

        total_boxes = results[0]
        points = results[1]
        
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 144, 0.37)
        return total_boxes, enumerate(chips)
        # for i, chip in enumerate(chips):
            # cv2.imshow('chip_'+str(i), chip)
            # cv2.imwrite('faces/chip_'+str(i)+'.png', chip)
        # draw = img.copy()
        # for b in total_boxes:
        #     cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        # for p in points:
        #     for i in range(5):
        #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        
        # height, width = draw.shape[:2]
        # draw = cv2.resize(draw, (320, int(height/width*320)))
        # cv2.imshow("detection result", draw)
        # cv2.waitKey(0)

def create_output_dir(data_dir):
    """
    if os.path.exists(data_dir+"/train"):
        shutil.rmtree(data_dir+"/train")
    if os.path.exists(data_dir+"/test"):
        shutil.rmtree(data_dir+"/test")
    os.mkdir(data_dir+"/train")
    os.mkdir(data_dir+"/test")"""
    if os.path.exists(data_dir+"/faces"):
        shutil.rmtree(data_dir+"/faces")
    os.mkdir(data_dir+"/faces")


def extract_face_from_dir(data_dir):
    create_output_dir(data_dir)
    person_names = os.listdir(data_dir+'/raw_faces')
    for person_name in person_names:
        if '.' in person_name:
            continue
        os.mkdir(data_dir+"/faces/"+person_name)
        # os.mkdir(data_dir+"/test/"+person_name)
        print person_name
        face_imgs = os.listdir(data_dir+'/raw_faces/'+person_name)

        for face_img_name in face_imgs:
            if not '.jpg' in face_img_name:
                continue
            print face_img_name
            boxs, faces = extract_face_from_img(data_dir+'/raw_faces/'+person_name+'/'+face_img_name)
            # print type(faces)
            if list(faces).__len__() == 1:
                for i, face in faces:
                    face = cv2.resize(face, (64, 64))
                    cv2.imwrite(data_dir+'/faces/'+person_name+'/'+face_img_name, face)
                    # cv2.imshow('chip_'+str(i), face)
                    # cv2.waitKey(0)
                # cv2.imwrite(data_dir+'/train/'+person_name+'/'+face_img_name, face)
                # extract_face(data_dir+'/faces/'+person_name+'/'+face_img_name)
            


def extract_face():
    extract_face_from_dir('./../../data')


if __name__ == '__main__':
    extract_face()
    # os.mkdir(data_dir+"/train/123")