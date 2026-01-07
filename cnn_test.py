from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
  for inputs, labels in test_loader:
    inputs ,labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs,1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())
print("Evaluation Complete! ")
print("Accuracy : ",accuracy_score(y_true,y_pred))
print("Classification Report : ",classification_report(y_true,y_pred))
fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=classes,    
    xticks_rotation='vertical', 
    cmap='Blues',             
    ax=ax                      
)
plt.show()
