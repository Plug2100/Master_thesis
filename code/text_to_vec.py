import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import time
import chromadb


def vectorization(model, text, id_, device, collection_mean, collection_cls):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encoded_text = inputs.to(device)

    with torch.no_grad():
        model_output = model(**encoded_text)
        embedding_mean = model_output.last_hidden_state.mean(dim=1)
        embedding_cls = model_output.pooler_output

        collection_mean.add(
            embeddings=[embedding_mean.tolist()[0]],
            ids=[str(id_)]
        )

        collection_cls.add(
            embeddings=[embedding_cls.tolist()[0]],
            ids=[str(id_)]
        )
    


csv_file = 'C:/Users/plug2/Desktop/diploma/citeulike-a-master/raw-data.csv'
data = pd.read_csv(csv_file, encoding='latin1')
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

chroma_client_mean = chromadb.PersistentClient('vectors/mean_vec')
chroma_client_cls = chromadb.PersistentClient('vector/cls_vec')
collection_mean = chroma_client_mean.create_collection(name="embedding", metadata={"hnsw:space": "cosine"})
collection_cls = chroma_client_cls.create_collection(name="embedding", metadata={"hnsw:space": "cosine"})
cls_vectors = []
mean_vectors = []
for _, row in data.iterrows():
    text = row['raw.title'] + '. ' + row['raw.abstract']
    id_ = row['doc.id']
    vectorization(model, text, id_, device, collection_mean, collection_cls)
