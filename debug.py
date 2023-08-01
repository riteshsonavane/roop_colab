#!/usr/bin/env python3
import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

assert insightface.__version__>='0.7'

def draw_faces_on_image(img, faces):
    cv2.imwrite(f'tmp_face.jpg', app.draw_on(img, faces))

def find_source_face_and_target_distances(target_faces, reference_face):
    distances = []
    if target_faces:
        for face in target_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distances.append(np.sum(np.square(face.normed_embedding - reference_face.normed_embedding)))
    return distances

def load_video_in_memoery(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}")

    frame_count = 0
    frames = []
    while True:
        # Read frame from video
        ret, frame = cap.read()
        frames.append(frame)
        # If frame is read successfully, save it
        if ret:
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        else:
            break

    # Release the video capture object
    cap.release()
    return frames

def face_similarity(face_analyser, img1, img2):
  # how to use
  # providers = ['CPUExecutionProvider']
  # app = FaceAnalysis(providers=providers, name='buffalo_l')
  # app.prepare(ctx_id=0, det_size=(640, 640))
  # app.prepare(ctx_id=0)
  # similarity_1 = face_similarity(
  #     app, 
  #     cv2.imread('/content/roop_colab/child.png'),
  #     cv2.imread('/content/roop_colab/father.png')
  # )
  # similarity_2 = face_similarity(
  #     app, 
  #     cv2.imread('/content/roop_colab/child.png'),
  #     cv2.imread('/content/roop_colab/mother.png')
  # )
  # print(f'father similarity {similarity_1}, mother similarity {similarity_2}') 
  f1 = face_analyser.get(img1)[0]
  f2 = face_analyser.get(img2)[0]
  return np.sum(np.square(f1.normed_embedding - f2.normed_embedding))

if __name__ == '__main__':
    providers = ['CPUExecutionProvider']
    app = FaceAnalysis(providers=providers, name='buffalo_l')
    # app.prepare(ctx_id=0, det_size=(640, 640))
    app.prepare(ctx_id=0)
    model_path = resolve_relative_path('./models/inswapper_128.onnx')
    swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    img = cv2.imread('/content/roop_colab/yoga.png')
    faces = app.get(img)

    # draw a square on the target photo
    draw_faces_on_image(img, faces)
    source_face = app.get(cv2.imread('/content/roop_colab/source.jpg'))[0]
    print(find_source_face_and_target_distances(faces, source_face))
    res = img.copy()
    for i, face in enumerate(faces):
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
