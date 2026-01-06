dataset_path = path  
target_words = ['up', 'down', 'left', 'right', 'stop'] 

print(f"🚀 Preparing to load: {target_words}")


def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=512)
    return mfcc.T 


data = {}  

for word in target_words:
    folder = os.path.join(dataset_path, word)
    if os.path.isdir(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        data[word] = files
        print(f"   Found {len(files)} files for '{word}'")
    else:
        print(f" Warning: Folder '{word}' not found!")

train_files = {word: [] for word in target_words}
test_files = {word: [] for word in target_words}

for word, files in data.items():
    tr, te = train_test_split(files, test_size=0.2, random_state=42)
    train_files[word] = tr
    test_files[word] = te

print(f" Data Split Completed (80% Train, 20% Test)")








