import cv2
import os
import numpy as np
import json

FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
EMBED_MODEL = "models/openface.nn4.small2.v1.t7"

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
embed_net = cv2.dnn.readNetFromTorch(EMBED_MODEL)

faces_dir = "faces"
saved_embeddings = []
saved_names = []
details_dict = {}


if os.path.exists("details.json"):
    details_dict = json.load(open("details.json"))

for person in os.listdir(faces_dir):
    person_folder = os.path.join(faces_dir, person)

    if not os.path.isdir(person_folder):
        continue

    if person not in details_dict:
        details_dict[person] = {
            "relation": "",
            "age": "",
            "gender": "",
            "nickname": ""
        }

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        if detections[0, 0, 0, 2] < 0.7:
            continue

        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        face = img[y1:y2, x1:x2]

        face_blob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, (96, 96), (0, 0, 0),
            swapRB=True
        )

        embed_net.setInput(face_blob)
        vec = embed_net.forward()

        saved_embeddings.append(vec.flatten())
        saved_names.append(person)

np.save("embeddings.npy", np.array(saved_embeddings))
np.save("names.npy", np.array(saved_names))

json.dump(details_dict, open("details.json", "w"))

print("Encoding complete with details!")
