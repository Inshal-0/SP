import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/kaggle/input/sounds/left.wav"   
signal, sr = librosa.load(audio_path)

energy = np.sum(signal**2) / len(signal)

if energy > 0.01:
    energy_label = "Loud Sound Detected"
else:
    energy_label = "Silent Sound Detected"

zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

if zcr > 0.1:
    zcr_label = "Unvoiced Consonant"
else:
    zcr_label = "Voiced Sound"

print("\n--- RULE BASED SPEECH ANALYSIS ---")
print("Energy:", energy_label)
print("ZCR:", zcr_label)

# Example Rule-Based Word Inference
if energy > 0.01 and zcr > 0.1:
    print("Rule Based Recognized Pattern: CONSONANT HEAVY WORD")
else:
    print("Rule Based Recognized Pattern: VOWEL DOMINANT WORD")
