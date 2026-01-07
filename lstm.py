import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self):
        super(AudioLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=26, hidden_size=64, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)        
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10) 

    def forward(self, x):
        x = x.squeeze(1)   
        x = x.permute(0, 2, 1) 
        
        out, _ = self.lstm1(x) 
        out = self.dropout(out)
        
        _, (hidden_state, _) = self.lstm2(out)
        
        out = hidden_state[-1]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out