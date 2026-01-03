import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the GNN Model
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_feats, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        # Layer 1
        x = F.relu(torch.spmm(adj, self.conv1(x)))
        # Layer 2
        x = F.relu(torch.spmm(adj, self.conv2(x)))
        # Global Pooling (Mean)
        return self.classifier(torch.mean(x, dim=0))

# Helper to Load PKL Graph
def load_graph_tensor(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    
    # Feature Engineering (Centered Coordinates)
    xs = [d['x'] for _, d in G.nodes(data=True)]
    ys = [d['y'] for _, d in G.nodes(data=True)]
    
    if not xs: return None, None # Skip empty graphs
    
    x_c, y_c = np.mean(xs), np.mean(ys)
    
    # Create Features Tensor [Num_Nodes, 2]
    feats = [[d['x']-x_c, d['y']-y_c] for _, d in G.nodes(data=True)]
    x = torch.tensor(feats, dtype=torch.float32)

    # Create Adjacency Matrix
    edges = list(G.edges()) + [(i, i) for i in range(len(G))] # Add self-loops
    mapping = {n: i for i, n in enumerate(G.nodes())} # Re-map to 0..N-1
    indices = [[mapping[u], mapping[v]] for u, v in edges]
    
    i = torch.tensor(indices).t()
    v = torch.ones(i.shape[1])
    adj = torch.sparse_coo_tensor(i, v, (len(G), len(G)))
    
    # Extract Label (Only exists for training data)
    label = None
    type_map = {'organic': 0, 'grid': 1, 'hybrid': 2}
    if 'type' in G.graph and G.graph['type'] in type_map:
        label = type_map[G.graph['type']]
        
    return (x, adj), label

# Main Execution
def run_baseline():
    # Paths relative to this script
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE, 'data', 'train')
    TEST_DIR = os.path.join(BASE, 'data', 'test')
    SUBMISSION_DIR = os.path.join(BASE, 'submissions')
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, 'baseline_submission.csv')

    print(f"Checking Training Folder: {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        print("Error: Train folder not found.")
        return

    # --- LOAD TRAINING DATA ---
    print("Loading Training Graphs...")
    train_data = []
    y_train = []
    
    files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.pkl')]
    for f in files:
        (x, adj), label = load_graph_tensor(os.path.join(TRAIN_DIR, f))
        if x is not None and label is not None:
            train_data.append((x, adj))
            y_train.append(label)

    if not train_data:
        print("Error: No valid training graphs found.")
        return

    y_train = torch.tensor(y_train, dtype=torch.long)
    
    # --- TRAIN MODEL ---
    model = SimpleGCN(in_feats=2, hidden_dim=16, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Training GNN...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for i, (x, adj) in enumerate(train_data):
            optimizer.zero_grad()
            out = model(x, adj).unsqueeze(0)
            loss = criterion(out, y_train[i].unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(train_data):.4f}")

    # --- PREDICT ON TEST SET ---
    print("Predicting on Test Folder...")
    model.eval()
    results = []
    
    if os.path.exists(TEST_DIR):
        test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.pkl')]
        for f in test_files:
            (x, adj), _ = load_graph_tensor(os.path.join(TEST_DIR, f))
            if x is None: 
                pred = 0 # Fallback
            else:
                with torch.no_grad():
                    out = model(x, adj).unsqueeze(0)
                    pred = torch.argmax(out, dim=1).item()
            
            results.append({'filename': f, 'target': pred})
    
    # --- SAVE SUBMISSION ---
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved submission to {SUBMISSION_FILE}")

if __name__ == "__main__":
    run_baseline()
