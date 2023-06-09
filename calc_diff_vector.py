import ast

import cv2
import numpy as np
import pandas as pd
import dlib
import os
import concurrent.futures
import random

# loading models, detectors
facerec_model = "/Users/outianyi/Computer_Vision/dlib_face_recognition_resnet_model_v1.dat"
face_recognizer = dlib.face_recognition_model_v1(facerec_model)
model = '/Users/outianyi/Computer_Vision/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model)
detector = dlib.get_frontal_face_detector()


class DiffVecHelper:
    @staticmethod
    def distance(a, b):

        # return np.linalg.norm(a - b)
        # return np.sqrt(np.sum(np.sqrt(np.subtract(np.array(a), np.array(b)))))
        return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))

    @staticmethod
    def random_choice_5vecs(vectors, batch_size=5):
        vector_list = list(vectors)
        # while True:
        # random_vectors  = random.sample(vector_list, batch_size)
        random_vectors = []
        for i in range(5):
            random_vectors.append(random.sample(vector_list, 1))
        return random_vectors

    # random_vectors: generated by @method random_choose_5paths
    # random_vectors include 5 people, several faces each
    @staticmethod
    def calc_other_moments(self_faces):
        # self_faces = ast.literal_eval(self_faces)
        self_face = random.sample(self_faces, 1)[0]
        # random_vectors = next(random_vectors)
        random_vectors = DiffVecHelper.random_choice_5vecs(all_des, 5)
        moments = []
        for random_vector in random_vectors:
            t_vector = random.sample(random_vector[0], 1)[0]
            t_dis = DiffVecHelper.distance(self_face, t_vector)
            # moments.append(DiffVecHelper.distance(self_face, t_vector))
            moments.append(t_dis)
        return np.mean(moments)


# ------------------------------------------------------------
# read all faces
# face_info = pd.read_csv('multi_threding_results.csv')
# read test faces
face_info = pd.read_csv('test_people_results.csv')
# face_info = face_info[:40]
all_des = face_info['descriptors'].values
# convert from str to list
for i in range(len(all_des)):
    all_des[i] = ast.literal_eval(all_des[i])


def processing_chunk(face_info: pd.DataFrame, all_vecs):

    print('generating random moment...')
    face_info['outer moments'] = face_info['descriptors'].apply(DiffVecHelper.calc_other_moments)

    print('reordering...')
    new_index = ['index', 'path', 'name', 'average moments', 'outer moments', 'descriptors']
    face_info = face_info.reindex(columns=new_index)

    print('a chunk finished!')
    return face_info


chunk_size = 200
chunks = [face_info[i: i + chunk_size] for i in range(0, len(face_info), chunk_size)]

# create thread pool, max cpu 6
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = executor.map(processing_chunk, chunks, face_info['descriptors'])
    # results = pd.concat(executor.map(processing_chunk, chunks, face_info['descriptors']))

# results = pd.DataFrame(results)
results = list(results)
results = pd.concat(results)
print('saving...')
# results.to_csv('face_info_inner_outer_moments.csv')
# saving test faces
results.to_csv('test_faces_inner_outer_moments.csv')
pass

