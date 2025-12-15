import cv2
import numpy as np

FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
EMBED_MODEL = "models/openface.nn4.small2.v1.t7"

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
embed_net = cv2.dnn.readNetFromTorch(EMBED_MODEL)

embeddings = np.load("embeddings.npy", allow_pickle=True)
names = np.load("names.npy", allow_pickle=True)

def recognize_face(face):
    blob = cv2.dnn.blobFromImage(face, 1.0/255, (96,96), (0,0,0), swapRB=True)
    embed_net.setInput(blob)
    vec = embed_net.forward().flatten()

    distances = np.linalg.norm(embeddings - vec, axis=1)
    idx = np.argmin(distances)

    if distances[idx] > 0.9:
        return "Unknown"
    return names[idx]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104,177,123))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence < 0.7:
            continue

        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (x1, y1, x2, y2) = box.astype(int)

        face = frame[y1:y2, x1:x2]
        name = recognize_face(face)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
