import torch
import torch.nn as nn

class basic_model(nn.Module):
    def __init__(self, chunkSize= 8):
        super(basic_model, self).__init__()
        self.chunkSize = chunkSize                
        self.transfer = nn.ReLU()
        self.flatten = nn.Flatten()        
        self.layers = nn.Sequential(
            self.flatten,
            nn.Linear(chunkSize * chunkSize, 64),
            self.transfer,
            nn.Linear(64, 32),
            self.transfer,
            nn.Linear(32, 16),
            self.transfer,
            nn.Linear(16, 8),
            self.transfer,
            nn.Linear(8, 4),
            self.transfer,
            nn.Linear(4, 2),
            self.transfer,
            nn.Linear(2, 1),
            self.transfer,
        )

    def forward(self, x):
        out = self.layers(x)        
        return out




if __name__ == "__main__":
    model = basic_model()
    inp = torch.randn(1, 8, 8)
    out = model(inp)
    print(out)
    print(model)