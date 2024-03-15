import pandas as pd
import chromadb
import json
from collections import defaultdict 
import sys
import math
from tqdm import tqdm

users = 5551
papers = 16980

# Read test data
positive_examples = defaultdict(list)
with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users_20.dat', 'r') as file:
    current_line = -1
    for line in file:
        current_line += 1
        numbers = list(map(int, line.split()))
        for i in numbers:
            positive_examples[current_line + papers].append(i)

# read ground truth to exclude it
positive_examples_gt = defaultdict(list)
with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users_80.dat', 'r') as file:
    current_line = -1
    for line in file:
        current_line += 1
        numbers = list(map(int, line.split()))
        for i in numbers:
            positive_examples_gt[current_line + papers].append(i)

chroma_client_users = chromadb.PersistentClient(path="vectors/results_clients")
collection_users = chroma_client_users.get_collection(name="embedding")

chroma_client_papers = chromadb.PersistentClient(path="vectors/results_papers")
collection_papers = chroma_client_papers.get_collection(name="embedding")


Recall = 0
NDCG = 0
IDCG_20 = sum(1 / math.log2(1 + i) for i in range(1, 21))
for i in tqdm(range(papers, papers + users)): # correct user id
    responce = collection_users.get(ids=[str(i)], include=['embeddings'])
    user_embedding = responce['embeddings'][0] # extract user emb

    ans = collection_papers.query( # extract recommendations
            query_embeddings=[user_embedding],
            n_results=len(positive_examples_gt[i]) + 20 # > len(ground truth) + 20
        )
    # Calculate the results
    ids_rec = ans['ids'][0]
    all_recs = 0
    DCG = 0
    relevant = 0
    for j in ids_rec:
        # if was not in train
        if int(j) not in positive_examples_gt[i]:# I NEED TO CHECK THE TYPES!!!!!! everyther should be str
            all_recs += 1
            # if true
            if int(j) in positive_examples[i]:# I NEED TO CHECK THE TYPES!!!!!! everyther should be str
                DCG += 1 / math.log2(1 + all_recs)
                relevant += 1
        # @20
        if(all_recs == 20):
            break
    NDCG += DCG / IDCG_20
    Recall += relevant / len(positive_examples[i])

print('NDCG@20:', NDCG / users)
print('Recall@20:', Recall / users)

