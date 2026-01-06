hmm_models = {}

print("\nStarting Training...")

for word in target_words:
    print(f"   Training HMM for '{word}'...")
    
    X_concat = []
    lengths = []
    
    for file in train_files[word]:
        features = extract_mfcc(file)
        X_concat.append(features)
        lengths.append(len(features))
    
    X_concat = np.vstack(X_concat)
    
    model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100, random_state=42)
    model.fit(X_concat, lengths)
    
    hmm_models[word] = model

print(" Training Completed!")