from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
import cv2
import numpy as np
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
FACES_FOLDER = "faces"
MODELS_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_data():
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    names = np.load("names.npy", allow_pickle=True)
    details = json.load(open("details.json"))
    return embeddings, names, details


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images[]")
    name = request.form["name"]
    relation = request.form["relation"]
    age = request.form["age"]
    gender = request.form["gender"]
    nickname = request.form["nickname"]

    person_folder = os.path.join(FACES_FOLDER, name)
    os.makedirs(person_folder, exist_ok=True)

    saved_files = []

    for img in files:
        image_path = os.path.join(person_folder, img.filename)
        img.save(image_path)
        saved_files.append(image_path)

    details_path = "../details.json"
    details = json.load(open(details_path)) if os.path.exists(details_path) else {}

    details[name] = {
        "relation": relation,
        "age": age,
        "gender": gender,
        "nickname": nickname
    }

    json.dump(details, open(details_path, "w"))

    subprocess.run(["python", "encode_faces.py"])

    return jsonify({"status": "success", "saved_images": saved_files})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None:
        return jsonify({"status": "error", "message": "No image received"})

    FACE_PROTO = "models/deploy.prototxt"
    FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
    EMBED_MODEL = "models/openface.nn4.small2.v1.t7"

    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    embed_net = cv2.dnn.readNetFromTorch(EMBED_MODEL)

    embeddings = np.load("embeddings.npy", allow_pickle=True)
    names = np.load("names.npy", allow_pickle=True)
    details = json.load(open("details.json"))    # <<< IMPORTANT

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),(104, 177, 123))
    face_net.setInput(blob)
    detections = face_net.forward()

    if detections[0, 0, 0, 2] < 0.6:
        return jsonify({"status": "success", "name": "No face found"})

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)
    face = img[y1:y2, x1:x2]

    blob_face = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96),(0, 0, 0), swapRB=True)
    embed_net.setInput(blob_face)
    vec = embed_net.forward().flatten()

    distances = np.linalg.norm(embeddings - vec, axis=1)
    idx = np.argmin(distances)

    if distances[idx] > 0.9:
        name = "Unknown"
    else:
        name = names[idx].item()

    if name != "Unknown" and name in details:
        person = details[name]
    else:
        person = {
            "relation": "Unknown",
            "age": "Unknown",
            "gender": "Unknown",
            "nickname": "Unknown"
        }

    return jsonify({
        "status": "success",
        "name": name,
        "relation": person["relation"],
        "age": person["age"],
        "gender": person["gender"],
        "nickname": person["nickname"]
    })

@app.route("/camera")
def camera_page():
    return send_from_directory("website", "camera.html")

@app.route("/")
def index():
    return send_from_directory("website", "index.html")


app.run(port=5000, debug=True)
