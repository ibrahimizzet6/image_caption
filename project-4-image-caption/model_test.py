
import torch
from model import CNNtoLSTM
from preprocess_text import word2idx, idx2word, vocab_size


embed_size = 256
hidden_size = 512
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CNNtoLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load("caption_model.pth", map_location=device))
model.eval()

from PIL import Image
from preprocess_image import transform

img_path = r"C:\Users\BB\python_icin\CV\Projeler\project-4-image-caption\Images\3765374230_cb1bbee0cb.jpg"
image = Image.open(img_path).convert("RGB")

image = transform(image)

def generate_caption(model, image, word2idx, idx2word, max_len=20, device="cpu"):
    model.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)

        inputs = features.unsqueeze(1)
        states = None

        caption = []

        input_word = torch.tensor([[word2idx["<start>"]]]).to(device)

        for _ in range(max_len):
            embeddings = model.decoder.embedding(input_word)
            lstm_input = torch.cat((inputs, embeddings), dim=1)
            
            outputs, states = model.decoder.lstm(lstm_input, states)
            output = model.decoder.linear(outputs[:, -1, :])
            
            predicted = output.argmax(1)
            predicted_word = idx2word[predicted.item()]

            if predicted_word == "<end>":
                break

            caption.append(predicted_word)
            input_word = predicted.unsqueeze(0)

        return " ".join(caption)
    
caption = generate_caption(model, image, word2idx, idx2word, max_len=30, device=device)
print("Generated Caption:", caption)