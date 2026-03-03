import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess_text import df, vocab_size, max_len
from preprocess_image import transform
from FlickrDataset import FlickrDataset
from model import CNNtoLSTM

embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 30
batch_size = 32
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = FlickrDataset(dataframe=df,image_dir="Images",   transform=transform)
train_loader = DataLoader( dataset, batch_size=batch_size, shuffle=True)


model = CNNtoLSTM(embed_size,hidden_size,vocab_size,num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

  for epoch in range(num_epochs):
      model.train()
      total_loss = 0

      for images, captions in train_loader:
          images = images.to(device)
          captions = captions.to(device)

          outputs = model(images, captions[:, :-1])
          loss = criterion(
            outputs.reshape(-1, vocab_size),
            captions[:, 1:].reshape(-1)
        )

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



  torch.save(model.state_dict(), "caption_model.pth")
  print("Model kaydedildi: caption_model.pth")