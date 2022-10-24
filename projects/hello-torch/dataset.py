import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from images import download_image

class ImageDataset(Dataset):
    def __init__(self, device = None,chunkSize=8):
        transform = transforms.Compose([transforms.PILToTensor()])
        image = download_image("fruit.png")
        image = image.convert("L")
        self.width = image.width
        self.height = image.height
        self.chunkSize = chunkSize
        self.chunksX = self.width // self.chunkSize
        self.chunksY = self.height // self.chunkSize
        self.chunkCount = self.chunksX * self.chunksY
        imageTensor1 = transform(image)
        imageTensor2 = imageTensor1[:,0:self.chunksY*self.chunkSize, 0:self.chunksX*self.chunkSize]
        imageTensor3 = imageTensor2.view(self.chunkCount, self.chunkSize, self.chunkSize)
        if (device):
            self.imageTensor = imageTensor3.to(device)
        else:
            self.imageTensor = imageTensor3
                        
    def __len__(self):
        return self.chunkCount

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.imageTensor[idx]
        output = self.imageTensor[idx][4,4]
        return input, output


if __name__ == "__main__":
    dataset = ImageDataset()
    print(dataset[0])