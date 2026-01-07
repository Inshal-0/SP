import librosa
import numpy as np
import os

def extract_mfcc(file):
    signal, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return mfcc  

template_folder = "/kaggle/input/sounds"
templates = {}

if os.path.exists(template_folder):
    for file in os.listdir(template_folder):
        if file.endswith(".wav"):
            word = file.replace(".wav", "")
            path = os.path.join(template_folder, file)
            templates[word] = extract_mfcc(path)
    print("Templates Loaded:", templates.keys())
else:
    print("Error: Template folder not found.")

test_audio = "/kaggle/input/sounds/start.wav" 
test_features = extract_mfcc(test_audio)

scores = {}

print("\nCalculating distances...")
for word, template_features in templates.items():
    
    D, wp = librosa.sequence.dtw(X=test_features, Y=template_features, metric='cosine')
    final_distance = D[-1, -1] / len(wp)
    scores[word] = final_distance

if scores:
    recognized_word = min(scores, key=scores.get)

    print("\n--- DTW TEMPLATE MATCHING RESULT ---")
    for key, value in scores.items():
        print(f"{key} → DTW Distance: {value:.4f}")

    print("\n🎉 FINAL RECOGNIZED WORD:", recognized_word)
else:
    print("No templates found to compare.")