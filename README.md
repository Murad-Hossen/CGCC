
<img width="900" height="800" alt="three_cities_networks" src="https://github.com/user-attachments/assets/0bf5f83b-0777-47fa-b76b-81b9d2932cca" />

# City Graph class challenge (CGCC)


🏆 View Live Leaderboard: [Open leaderboard](https://murad-hossen.github.io/CGCC/leaderboard/)

This dataset comprises street network graphs for 120 diverse cities across continents including North America, South America, Europe, Asia, Africa, Australia & Oceania, and others like the Middle East and Central Asia. The graphs are extracted from OpenStreetMap using OSMnx, focusing on driveable roads within a 500-meter buffer around each city's central point.

The dataset includes a total of 120 cities with an unbalanced distribution of street network types reflecting real-world urban patterns: 37 grid cities (such as planned orthogonal layouts like Salt Lake City, USA), 31 organic cities (such as irregular, historic winding streets like Boston, USA), and 52 hybrid cities (such as mixed elements like Atlanta, USA).

Each city's data is stored as a serialized NetworkX graph in `.pkl` format within the `data` folder, including nodes (intersections with coordinates), edges (roads with lengths and geometries), and graph attributes for layout `type` (grid/organic/hybrid) and city `name`.

This dataset is ideal for urban planning analysis, graph theory, or machine learning tasks like layout classification. It was generated via a Python script using OSMnx and NetworkX.


The goal of Task 3 is to train a model to classify each city's street layout into one of three classes:

- `0` = **organic**
- `1` = **grid**
- `2` = **hybrid**

Participants will train on the **train set** and submit predictions for the **test set** as a `submission.csv`.

---

## Environment setup

Before training or submitting, set up a Python environment and install dependencies.

### 1. Create a virtual environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

From the **project root** (where `requirements.txt` is):

**Linux / macOS:**
```bash
pip install -r requirements.txt
```

**Windows:**
```cmd
pip install -r requirements.txt
```

(If you use the starter-code baseline, also install its dependencies: `pip install -r starter_code/requirements.txt`.)

---

## Dataset Summary

- Class distribution in the full dataset:
  - **organic**: 31
  - **grid**: 37
  - **hybrid**: 52

Each city graph is stored as a serialized NetworkX graph (`.pkl`) and contains:
- **nodes**: intersections with coordinates (`x`, `y`)
- **edges**: road segments (may include attributes such as length/geometry depending on OSM)
- **graph attributes** (e.g., city name).  
For the **test set**, the label attribute is removed.

This dataset is useful for urban planning analysis, graph learning, and layout classification tasks.

---

## Data Split (Train/Test)
The dataset is split into **70/30** with **stratification by class**:

- `data/train/` : labeled graphs (70%)
- `data/test/`  : unlabeled graphs (30%)

Training labels are provided in:
- `data/train_labels.csv` with columns:
  - `filename`
  - `target`

---

## What You Need To Do (Participant)

### Step 1: Train
Train your model using:
- graphs in `data/train/`
- labels in `data/train_labels.csv`

### Step 2: Predict
Predict labels for every graph in:
- `data/test/`

### Step 3: Prepare your submission file
Create a CSV with columns `filename` and `prediction` (same format as `submissions/sample_submission.csv`):

```csv
filename,prediction
Boston_Massachusetts_USA.pkl,2
Delhi_India.pkl,0
Turin_Italy.pkl,1
...
```

Save it as a `.csv` file (e.g. `my_submission.csv`) in the **`submissions/`** folder.  
**Note:** `.csv` files in `submissions/` are git-ignored, so your raw submission will not be pushed. You will submit an **encrypted** version instead.

### Step 4: Encrypt your submission
From the project root, run the encryption script so it can find your CSV and the encryption key:

**Linux / macOS:**
```bash
cd submissions
python encrypt_submissions.py
cd ..
```

**Windows (Command Prompt):**
```cmd
cd submissions
python encrypt_submissions.py
cd ..
```

**Windows (PowerShell):**
```powershell
cd submissions
python encrypt_submissions.py
cd ..
```

This creates a `.enc` file next to each `.csv` in `submissions/` (e.g. `my_submission.csv.enc`). Only `.enc` files are tracked by git; your `.csv` stays local.

### Step 5: Push your encrypted submission
Commit and push the new `.enc` file(s) to the repository (e.g. open a Pull Request or push to the main branch, as per the challenge rules). The automated pipeline will decrypt and score the **latest** `.enc` file in `submissions/` and update the leaderboard.

---

## Project directory structure

Below is the layout of the repository (relevant to competitors). Run `tree -a -I '.git|.venv|__pycache__' --dirsfirst` from the project root to print it locally (if `tree` is installed).

```
.
├── .env.example          # Example env file; organisers use .env for secrets (e.g. test labels).
├── .github/
│   ├── scripts/
│   │   └── process_submission.py   # CI script: finds latest .enc, decrypts, updates leaderboard.
│   └── workflows/
│       └── process_submission.yml # Runs on push to main.
├── data/
│   ├── train/            # Training city graphs (.pkl).
│   ├── test/             # Test city graphs (.pkl), no labels.
│   └── train_labels.csv  # Labels for files in data/train/.
├── encryption/
│   ├── encrypt.py        # Used by encrypt_submissions.py (reads public_key.pem).
│   ├── decrypt.py        # Used by CI to decrypt .enc submissions.
│   ├── generate_keys.py  # For organisers; generates key pair.
│   └── public_key.pem    # Public key used to encrypt submissions.
├── leaderboard/          # Leaderboard logic and assets (scores, CSV, HTML).
├── starter_code/
│   ├── baseline.py       # Example GCN baseline (train + predict).
│   └── requirements.txt # Extra deps for baseline (e.g. PyTorch).
├── submissions/
│   ├── encrypt_submissions.py  # Run this (from submissions/) to turn .csv → .enc.
│   ├── sample_submission.csv  # Format reference (filename, prediction).
│   └── *.csv.enc         # Your encrypted submissions; push these.
├── leaderboard.md        # Auto-generated leaderboard markdown.
├── README.md             # This file.
├── requirements.txt      # Root Python dependencies (encryption, scoring, etc.).
├── scoring_script.py     # Local scoring helper (needs test labels).
└── index.html            # Project index / redirect if used.
```

---

## Baseline Model (GCN) — Details

The provided baseline is a **Graph Convolutional Network (GCN)** for **graph-level classification** (one label per city graph).

### Input
Each city is a graph `G` stored as a `.pkl` NetworkX file.

- Nodes: intersections with coordinate attributes `x` and `y`
- Edges: road connections between intersections

### Node Features Used (per node)
For each node, we build a 3D feature vector:

1. **Centered & scaled x-coordinate**
2. **Centered & scaled y-coordinate**
3. **Normalized node degree**

So the node feature matrix is:

- `X ∈ R^(N×3)` where `N` = number of nodes in the city graph.

### Adjacency (GCN Normalization)
The baseline builds a sparse adjacency matrix with self-loops and applies standard GCN normalization:

Normalized adjacency: D^{-1/2}(A+I)D^{-1/2}

This improves stability compared to using a raw adjacency matrix.

### Model Architecture
The baseline uses two GCN-style message passing layers (implemented with sparse matrix multiplication) and then a graph pooling step:

- Layer 1: `X -> hidden`
- Layer 2: `hidden -> hidden`
- Pooling: concatenate **mean pooling** + **max pooling** to get a graph embedding
- Classifier: linear layer to output 3 logits (organic/grid/hybrid)

Training uses:
- Adam optimizer
- Cross-entropy loss
- class weights (helps if classes are imbalanced)
- dropout + weight decay (regularization)

### Validation
To provide a baseline metric without touching the hidden test labels, the script splits the training set internally:

- 70% train
- 30% validation (stratified)

It prints:
- Validation Accuracy
- Validation Macro-F1 (main metric)

### Output
After training, the baseline predicts on the unlabeled test graphs. Save your predictions in the required format and place the CSV in `submissions/`, then encrypt with `submissions/encrypt_submissions.py` and push the resulting `.enc` file(s).

Format of the submission CSV:
```csv
filename,prediction
City1.pkl,2
City2.pkl,0
...
```
