import torch
from torch.nn import LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GCN2Conv
import torch.nn.functional as F
import chromadb
import pandas as pd
import numpy as np
import time
from collections import defaultdict 
from tqdm import tqdm
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
users = 5551
vec_length = 768
papers = 16980

def distance_loss(output_vectors, pair_indices, positive_examples): # RN Bayesian Personalized Ranking (BPR) loss
    loss = 0.0
    for i in pair_indices:
        vec1 = output_vectors[i[0]]
        vec2 = output_vectors[i[1]]
        negative_example_number = random.randint(0, papers)
        while negative_example_number in positive_examples[i[1]]:
            negative_example_number = random.randint(0, papers)
        vec3 = output_vectors[negative_example_number]
        cos_distance_positive = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        cos_distance_negative = F.cosine_similarity(vec3.unsqueeze(0), vec2.unsqueeze(0))
        loss -= torch.log(F.sigmoid(cos_distance_positive - cos_distance_negative))
    loss /= len(pair_indices)
    return loss


def users_to_zero(tensor, papers = 16980):
    tensor[papers:] = 0
    return tensor


def articles_to_zero(tensor, users = 5551):
    tensor[:users] = 0
    return tensor


class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, negative_slope=0.01):
        super(SimpleGNN, self).__init__()
        
        # Initialize the first set of Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        
        # Initialize the second set of Graph Convolutional Layers
        self.conv1_2 = GCNConv(num_node_features, num_node_features)
        self.conv2_2 = GCNConv(num_node_features, num_node_features)
        self.conv3_2 = GCNConv(num_node_features, num_node_features)

        # Initialize the third set of Graph Convolutional Layers
        self.conv1_3 = GCNConv(num_node_features, num_node_features)
        self.conv2_3 = GCNConv(num_node_features, num_node_features)
        self.conv3_3 = GCNConv(num_node_features, num_node_features)

        # Initialize the third set of Graph Convolutional Layers
        self.conv1_4 = GCNConv(num_node_features, num_node_features)
        self.conv2_4 = GCNConv(num_node_features, num_node_features)
        self.conv3_4 = GCNConv(num_node_features, num_node_features)
        
        # LeakyReLU activation function
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        
        # Learnable weights for the first aggregation
        self.weight = torch.nn.Parameter(torch.randn(1))
        self.weight1 = torch.nn.Parameter(torch.randn(1))
        self.weight2 = torch.nn.Parameter(torch.randn(1))
        self.weight3 = torch.nn.Parameter(torch.randn(1))
        
        # Learnable weights for the second aggregation
        self.weight_2 = torch.nn.Parameter(torch.randn(1))
        self.weight1_2 = torch.nn.Parameter(torch.randn(1))
        self.weight2_2 = torch.nn.Parameter(torch.randn(1))
        self.weight3_2 = torch.nn.Parameter(torch.randn(1))
        self.weight4_2 = torch.nn.Parameter(torch.randn(1))

        # Learnable weights for the third aggregation
        self.weight_3 = torch.nn.Parameter(torch.randn(1))
        self.weight1_3 = torch.nn.Parameter(torch.randn(1))
        self.weight2_3 = torch.nn.Parameter(torch.randn(1))
        self.weight3_3 = torch.nn.Parameter(torch.randn(1))
        self.weight4_3 = torch.nn.Parameter(torch.randn(1))
        self.weight5_3 = torch.nn.Parameter(torch.randn(1))

        # Learnable weights for the third aggregation
        self.weight_4 = torch.nn.Parameter(torch.randn(1))
        self.weight1_4 = torch.nn.Parameter(torch.randn(1))
        self.weight2_4 = torch.nn.Parameter(torch.randn(1))
        self.weight3_4 = torch.nn.Parameter(torch.randn(1))
        self.weight4_4 = torch.nn.Parameter(torch.randn(1))
        self.weight5_4 = torch.nn.Parameter(torch.randn(1))
        self.weight6_4 = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, edge_index_1, edge_index_2, edge_index_3): # paper_to_paper_edge_index, paper_to_user_edge_index, user_to_paper_edge_index
        weights = [] # Check the weights - if smth is not important for example
        # First layer processing
        x_1 = self.conv1(x, edge_index_1)
        #x_1 = users_to_zero(x_1) # cut not connected verts
        x_2 = self.conv2(x, edge_index_2)
        #x_2 = articles_to_zero(x_2)
        x_3 = self.conv3(x, edge_index_3)
        #x_3 = users_to_zero(x_3)
        x_1 = self.leaky_relu(x_1)
        x_2 = self.leaky_relu(x_2)
        x_3 = self.leaky_relu(x_3)
        x1 = self.weight1 * x_1 + self.weight2 * x_2 + self.weight3 * x_3 + self.weight * x
        x1 = self.leaky_relu(x1)  # Apply activation
        weights.extend(['1: ',self.weight1, self.weight2, self.weight3, self.weight, '|'])
        
        # Second layer processing
        x_1 = self.conv1(x1, edge_index_1)
        #x_1 = users_to_zero(x_1)
        x_2 = self.conv2(x1, edge_index_2)
        #x_2 = articles_to_zero(x_2)
        x_3 = self.conv3(x1, edge_index_3)
        #x_3 = users_to_zero(x_3)
        x_1 = self.leaky_relu(x_1)
        x_2 = self.leaky_relu(x_2)
        x_3 = self.leaky_relu(x_3)
        x2 = self.weight1_2 * x_1 + self.weight2_2 * x_2 + self.weight3_2 * x_3 + self.weight_2 * x1 + self.weight4_2 * x
        x2 = self.leaky_relu(x2)  # Apply activation
        weights.extend(['2: ', self.weight1_2, self.weight2_2, self.weight3_2, self.weight_2, self.weight4_2, '|'])

        # Third layer processing
        x_1 = self.conv1(x2, edge_index_1)
        #x_1 = users_to_zero(x_1)
        x_2 = self.conv2(x2, edge_index_2)
        #x_2 = articles_to_zero(x_2)
        x_3 = self.conv3(x2, edge_index_3)
        #x_3 = users_to_zero(x_3)
        x_1 = self.leaky_relu(x_1)
        x_2 = self.leaky_relu(x_2)
        x_3 = self.leaky_relu(x_3)
        x3 = self.weight1_3 * x_1 + self.weight2_3 * x_2 + self.weight3_3 * x_3 + self.weight_3 * x2 + self.weight4_3 * x + self.weight5_3 * x1
        x3 = self.leaky_relu(x3)  # Apply activation
        weights.extend(['3: ', self.weight1_3, self.weight2_3, self.weight3_3, self.weight_3, self.weight4_3, self.weight5_3, '|'])
        #return x3, weights
       
        # Fourth layer processing
        x_1 = self.conv1(x3, edge_index_1)
        #x_1 = users_to_zero(x_1)
        x_2 = self.conv2(x3, edge_index_2)
        #x_2 = articles_to_zero(x_2)
        x_3 = self.conv3(x3, edge_index_3)
        #x_3 = users_to_zero(x_3)
        x_1 = self.leaky_relu(x_1)
        x_2 = self.leaky_relu(x_2)
        x_3 = self.leaky_relu(x_3)
        x4 = self.weight1_4 * x_1 + self.weight2_4 * x_2 + self.weight3_4 * x_3 + self.weight_4 * x2 + self.weight4_4 * x + self.weight5_4 * x1  + self.weight6_4 * x3
        x4 = self.leaky_relu(x4)  # Apply activation
        weights.extend(['4: ', self.weight1_4, self.weight2_4, self.weight3_4, self.weight_4, self.weight4_4, self.weight5_4, self.weight6_4, '|'])

        return x4, weights



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the model
model = SimpleGNN(vec_length)
model = model.to(device)

# vectors
chroma_client = chromadb.PersistentClient('vectors/mean_vec')
collection = chroma_client.get_collection(name='embedding')
csv_file = 'C:/Users/plug2/Desktop/diploma/citeulike-a-master/raw-data.csv'
data = pd.read_csv(csv_file, encoding='latin1')

all_vectors = []

for _, row in data.iterrows():
    id_ = row['doc.id']
    emb = collection.get(ids=[str(id_)], include=['embeddings'])
    all_vectors.append(emb['embeddings'][0])

#generate users emb
all_vectors.extend(np.random.uniform(low=-1, high=1, size=(users, vec_length)).tolist())
all_vectors = torch.tensor(all_vectors)  # Then convert to a tensor
all_vectors = all_vectors.to(device)


# Paper to paper - cytations
paper_from = []
paper_to = []
with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/citations.dat', 'r') as file:
    current_line = -1
    for line in file:
        current_line += 1
        numbers = list(map(int, line.split()))
        if numbers[0] == 0:
            continue
        for i in range(len(numbers) - 1):
            paper_from.append(current_line)
            paper_to.append(numbers[i + 1])

paper_to_paper_edge_index = torch.tensor([paper_from, paper_to], dtype=torch.long)  # Edge index
paper_to_paper_edge_index = paper_to_paper_edge_index.to(device)



# Paper to user and user to paper
paper_to_user_paper = []
paper_to_user_user = []
user_to_paper_user = []
user_to_paper_paper = []
ground_truth = []
positive_examples = defaultdict(list)
with open('C:/Users/plug2/Desktop/diploma/citeulike-a-master/users_80.dat', 'r') as file:
    current_line = -1
    for line in file:
        current_line += 1
        numbers = list(map(int, line.split()))
        for i in numbers:
            paper_to_user_paper.append(i)
            paper_to_user_user.append(current_line + papers)
            user_to_paper_user.append(current_line + papers)
            user_to_paper_paper.append(i)
            ground_truth.append([i, current_line + papers]) # paper - user
            positive_examples[current_line + papers].append(i)

paper_to_user_edge_index = torch.tensor([paper_to_user_paper, paper_to_user_user], dtype=torch.long)  # Edge index
paper_to_user_edge_index = paper_to_user_edge_index.to(device)
user_to_paper_edge_index = torch.tensor([user_to_paper_user, user_to_paper_paper], dtype=torch.long)  # Edge index
user_to_paper_edge_index = user_to_paper_edge_index.to(device)
random.shuffle(ground_truth)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
model.train()
batch = 100
previous_loss = 1
for epoch in range(12):
    start_model_time = time.time()
    loss_sum = 0
    iterations = 0
    for i in tqdm(range(0, len(ground_truth), batch)):
        pair_indices = ground_truth[i:i+batch]
        output_vectors, weights = model(all_vectors, paper_to_paper_edge_index, paper_to_user_edge_index, user_to_paper_edge_index)  
        loss = distance_loss(output_vectors, pair_indices, positive_examples)

        iterations += 1
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_model_time = time.time()
    print('Epoch:', epoch, 'Loss:', loss_sum/iterations, 'Loss diff:', previous_loss - loss_sum/iterations)
    if((previous_loss - loss_sum/iterations) < 0.001):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    previous_loss = loss_sum/iterations

# Inference
output_vectors, weights = model(all_vectors, paper_to_paper_edge_index, paper_to_user_edge_index, user_to_paper_edge_index)
chroma_client_results_papers = chromadb.PersistentClient('vectors/results_papers')
collection_results_papers = chroma_client_results_papers.create_collection(name="embedding", metadata={"hnsw:space": "cosine"})
chroma_client_results_users = chromadb.PersistentClient('vectors/results_clients')
collection_results_users = chroma_client_results_users.create_collection(name="embedding", metadata={"hnsw:space": "cosine"})
id_ = 0
print(weights)
for i in output_vectors:
    if (id_ < papers):
        collection_results_papers.add(
                embeddings=[i.tolist()],
                ids=[str(id_)]
            )
    else:
        collection_results_users.add(
                embeddings=[i.tolist()],
                ids=[str(id_)]
            )
    id_ += 1
print(id_)
