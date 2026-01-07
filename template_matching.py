import librosa
import numpy as np
import os

def extract_mfcc(file):
    signal, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)   

template_folder = "/kaggle/input/sounds"
templates = {}

for file in os.listdir(template_folder):
    if file.endswith(".wav"):
        word = file.replace(".wav", "")
        path = os.path.join(template_folder, file)
        templates[word] = extract_mfcc(path)

print("Templates Loaded:", templates.keys())

test_audio = "/kaggle/input/sounds/start.wav"   
test_features = extract_mfcc(test_audio)

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

scores = {}

for word in templates:
    dist = euclidean_distance(test_features, templates[word])
    scores[word] = dist

recognized_word = min(scores, key=scores.get)

print("\n--- TEMPLATE MATCHING RESULT ---")
for key, value in scores.items():
    print(f"{key} → Distance: {value:.2f}")

print("\n Recognized Word:", recognized_word)
