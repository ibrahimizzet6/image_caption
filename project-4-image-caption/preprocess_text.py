import pandas as pd 
import re
from collections import Counter

df = pd.read_csv('captions.txt')
#print(df.head())


#metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_caption"] = df["caption"].apply(clean_text)
df["clean_caption"] = "<start> " + df["clean_caption"] + " <end>"

#vocabulary oluşturma
all_words = []

for cap in df["clean_caption"]:
    all_words.extend(cap.split())

counter = Counter(all_words)

#print("Toplam unique kelime:", len(counter))

vocab = list(counter.keys())
#word2idx oluşturma
word2idx = {"<pad>": 0,"<unk>": 1}

for idx, word in enumerate(vocab):
    word2idx[word] = idx + 2

idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)

#print("Vocab size:", vocab_size)

#encode
def encode_caption(caption):
    return [word2idx.get(word, word2idx["<unk>"]) for word in caption.split()]

df["encoded"] = df["clean_caption"].apply(encode_caption)

#print(df["encoded"].iloc[0])

#maxlen bul 
max_len = max(len(cap) for cap in df["encoded"])
#print("Max length:", max_len)

#padding
def pad_caption(seq, max_len):
    return seq + [0] * (max_len - len(seq))

df["padded"] = df["encoded"].apply(lambda x: pad_caption(x, max_len))

