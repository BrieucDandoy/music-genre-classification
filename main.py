import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchaudio
import numpy as np
import os

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(98880, 128)  # Adjust the input size based on your image dimensions
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x



class AudioDataset(Dataset):
    def __init__(self,min=0,max=90):
        self.labels = []
        self.genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        self.file_paths = []
        path = "Data/genres_original/"

        for label_id,genre in enumerate(self.genres):
            for i in range(min,max):
                if genre != 'jazz' and i != 54:
                    try:
                        file_path = f"{path}{genre}/{genre}.{str(i).zfill(5)}.wav"
                        for i in range(5):
                            self.file_paths.append(file_path)
                            self.labels.append(label_id)
                    except:
                        print(genre,i)
    

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mfcc = load_and_process_audio(file_path,    idx%5)
        label = self.labels[idx]
        label_tensor = torch.zeros(10)
        label_tensor[label] = 1
        return mfcc, label_tensor


def create_data_loader(batch_size, shuffle=True,min=0,max=90):
    dataset = AudioDataset(min=min,max=max)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_and_process_audio(file_path,idx):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform[:,:660000]
    list_coef = [0,0.05,0.1,0.2,0.3]
    if idx != 0:
        waveform = (waveform + list_coef[idx] * torch.rand(1,660000))/(1+list_coef[idx])
    return get_mfccs(waveform,sample_rate)



def get_mfccs(waveform,sample_rate):
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 600, "hop_length": 160, "n_mels": 15, "center": False},
        )(waveform).reshape(13,-1)
    mfcc_delta = torchaudio.transforms.ComputeDeltas()(mfcc).reshape(13,-1)
    mfcc_delta2 = torchaudio.transforms.ComputeDeltas()(mfcc_delta).reshape(13,-1)
    img = torch.stack((mfcc, mfcc_delta, mfcc_delta2), dim=0)
    return img



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
learning_rate = 0.001
num_epochs = 10
batch_size = 32 
model = AudioClassifier(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = create_data_loader( batch_size=batch_size)

print(device)
if not os.path.exists("audio_classifier_model.pth"):
    print('Début entrainement')
    for epoch in range(num_epochs):
        # model.train() 
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    print("Training completed!")

    torch.save(model.state_dict(), "audio_classifier_model.pth")
else:
    model = AudioClassifier(num_classes=10)
    state_dict = torch.load("audio_classifier_model.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


train_loader = create_data_loader( batch_size=1,min=90,max=100)
list_result = []
list_real = []
for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        list_result.append(torch.argmax(model.fc3(outputs)).item())
        list_real.append(torch.argmax(labels).item())
results = [1 if res == truth else 0 for res,truth in zip(list_result,list_real)]
print(f"Le nombre de bonne réponse est de {sum(results)} pour un total\
 de {len(results)} données soit une précision de {round(sum(results)/len(results),2) *100}%")
