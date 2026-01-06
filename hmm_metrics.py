acc = accuracy_score(y_true, y_pred)
print(f"\n Final Accuracy: {acc*100:.2f}%")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred, target_names=target_words))

cm = confusion_matrix(y_true, y_pred, labels=target_words)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_words, yticklabels=target_words)
plt.title(f'HMM Confusion Matrix (Accuracy: {acc*100:.1f}%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
