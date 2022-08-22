import face_recognition
import numpy as np


def detect_faces(rgb: np.array):
    face_recognition.face_locations(rgb)
