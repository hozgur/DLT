import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import time
from PIL import Image
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (device)


from dataset import ImageDataset

import model


import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

dataset = ImageDataset(device = device,chunkSize=8)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = model.basic_model().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        data = data.to(device).unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
