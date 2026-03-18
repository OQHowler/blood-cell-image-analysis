import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------- DATA PATH -----------
data_path = "data/cell_images"

# ----------- CUSTOM DATASET -----------
class MalariaDataset(Dataset):
    def __init__(self, data_path):
        self.images = []
        self.labels = []
        
        classes = ["Parasitized", "Uninfected"]
        
        for label, cls in enumerate(classes):
            class_path = os.path.join(data_path, cls)
            
            for img_name in os.listdir(class_path)[:1000]:
                img_path = os.path.join(class_path, img_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # ✅ FIX 1: Keep RGB (no grayscale)
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                
                # Convert to (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                
                self.images.append(img)
                self.labels.append(label)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # ❌ REMOVE expand_dims (bug)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ----------- MODEL -----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # ✅ FIX 2: 3 input channels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 128 → 64 → 32 → 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 128 → 64
        x = self.pool(torch.relu(self.conv2(x)))  # 64 → 32
        x = self.pool(torch.relu(self.conv3(x)))  # 32 → 16
        
        x = x.view(-1, 64 * 16 * 16)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# ----------- LOAD DATA -----------
dataset = MalariaDataset(data_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ----------- MODEL SETUP -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------- TRAINING -----------
# ✅ FIX 3: more epochs
epochs = 20


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# ----------- EVALUATION -----------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"\nCNN Accuracy: {accuracy:.2f}")