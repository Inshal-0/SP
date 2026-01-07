losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 20
for epoch in range(epochs):
  model.train()
  total_loss = 0
  for data,label in train_loader:
    data , label = data.to(device), label.to(device)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs,label)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  losses.append(total_loss/len(train_loader))
  print(f"Epoch {epoch+1} ended with Loss : {total_loss/len(train_loader)}")