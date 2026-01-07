class AudioCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(16,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Flatten(),
        nn.Linear(32*6*8, 128),
        nn.ReLU(),
        nn.Linear(128,10)    )
  def forward(self,x):
    return self.model(x)