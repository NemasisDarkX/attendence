import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
import os

# Initialize InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' if GPU works
app.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces
def load_known_faces(path="./known_faces"):
    known_embeddings = []
    known_names = []

    for file in os.listdir(path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if faces:
                known_embeddings.append(faces[0].embedding)
                name = os.path.splitext(file)[0]
                known_names.append(name)
            else:
                print(f"[WARNING] No face found in {file}")
    return known_embeddings, known_names

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Mark attendance
def mark_attendance(name, recorded):
    if name not in recorded:
        with open("attendance.csv", "a") as f:
            f.write(f"{name},Present\n")
        recorded.add(name)

# Load known faces
known_embeddings, known_names = load_known_faces()
recorded_attendance = set()

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Cannot access webcam")
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        name = "Unknown"

        # Compare with known embeddings
        similarities = [cosine_similarity(emb, k) for k in known_embeddings]
        if similarities:
            max_index = np.argmax(similarities)
            if similarities[max_index] > 0.5:  # Threshold (adjust if needed)
                name = known_names[max_index]
                mark_attendance(name, recorded_attendance)
#checkmate
        # Draw bounding box
        bbox = face.bbox.astype(int)
        left, top, right, bottom = bbox
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("InsightFace Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
