y_true = []
y_pred = []

print("\nStarting Evaluation...")

for true_word in target_words:
    for file in test_files[true_word]:
        features = extract_mfcc(file)
        
        best_score = float('-inf')
        predicted_word = None
        
        for model_label, model in hmm_models.items():
            try:
                score = model.score(features)
                if score > best_score:
                    best_score = score
                    predicted_word = model_label
            except:
                continue 
        
        if predicted_word:
            y_true.append(true_word)
            y_pred.append(predicted_word)