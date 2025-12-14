#  Twitch Stream Language Prediction with Graph ML

A complete lab demonstrating **Graph Machine Learning** for predicting stream languages on Twitch using network structure and Node2Vec embeddings.

##  Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Extensions](#extensions)

##  Problem Statement

**Scenario:** You're a Data Scientist at Twitch 

Every day, new streamers join the platform. You need to identify the **language** of new streams, but audio-to-text conversion and language detection are expensive.

**Question:** Can we predict stream language using the **graph structure** instead?

### Key Hypothesis

- Users mostly chat in a **single language**
- If a user chats in two streams → both streams likely use the **same language**
- We can leverage this **shared audience** pattern to predict languages
- Exception: English (widely understood)

### Graph Representation

```
(:Stream)-[:SHARED_AUDIENCE {weight}]->(:Stream)
```

- **Nodes**: Twitch streams
- **Edges**: Shared audience (undirected)
- **Weights**: Count of shared viewers

##  Approach

1. **Build Graph**: Create monopartite graph of streams with shared audience
2. **Generate Embeddings**: Use **Node2Vec** to create vector representations
3. **Train Classifier**: Use **Random Forest** to predict language from embeddings
4. **Evaluate**: Analyze performance with confusion matrix and metrics

##  Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB RAM minimum
- Ports: 7474, 7687, 8888 available

### Installation

```bash
# Clone the repository
cd twitch-graphml-lab

# Start services
docker-compose up -d

# Wait for services to be ready (~30 seconds)
docker-compose logs -f

# Access Jupyter
# Open: http://localhost:8888
```

### Run the Lab

1. Open Jupyter: http://localhost:8888
2. Open notebook: `twitch_language_prediction.ipynb`
3. Run all cells sequentially

##  Project Structure

```
twitch-graphml-lab/
│
├── docker-compose.yml       # Services orchestration
│
├── app/
│   ├── Dockerfile          # Python environment
│   ├── requirements.txt    # Dependencies
│   │
│   ├── graph.py            # Neo4j connection utilities
│   ├── data_loader.py      # Data loading module
│   ├── ml.py               # Machine learning pipeline
│   ├── viz.py              # Visualization utilities
│   │
│   └── twitch_language_prediction.ipynb  # Main lab notebook
│
├── data/                   # Data directory (CSV imports)
│
└── README.md              # This file
```

##  Usage

### Module: `graph.py`

```python
from graph import GraphConnector

# Connect to Neo4j
graph = GraphConnector(
    uri="bolt://neo4j:7687",
    user="neo4j",
    password="graphml2024"
)

# Run queries
result = graph.run_query("MATCH (n) RETURN count(n)")
```

### Module: `data_loader.py`

```python
from data_loader import TwitchDataLoader

# Load Twitch data
loader = TwitchDataLoader(graph)
stats = loader.load_all()

print(f"Loaded {stats['streams']} streams")
```

### Module: `ml.py`

```python
from ml import GraphMLPipeline

# Initialize ML pipeline
ml = GraphMLPipeline(graph)

# Create graph projection
ml.create_graph_projection("twitch", orientation="UNDIRECTED")

# Generate Node2Vec embeddings
ml.run_node2vec(embedding_dimension=8)

# Prepare and train
df = ml.prepare_training_data()
results = ml.train_classifier(df)
```

### Module: `viz.py`

```python
from viz import *

# Plot language distribution
plot_language_distribution(df_languages)

# Plot confusion matrix
ml.plot_confusion_matrix(cm, labels)

# Create dashboard
create_analysis_dashboard(results, graph)
```

## 🔬 Methodology

### 1. Data Loading

**Sources:**
- Streams CSV: https://bit.ly/3JjgKgZ
- Relationships CSV: https://bit.ly/3S9Uyd8

**Schema:**
```cypher
CREATE (s:Stream {
    streamId: integer,
    language: string,
    views: integer,
    mature: boolean
})

CREATE (s1:Stream)-[:SHARED_AUDIENCE {weight: integer}]-(s2:Stream)
```

### 2. Graph Projection

```cypher
CALL gds.graph.project(
    'twitch',
    'Stream',
    {
        SHARED_AUDIENCE: {
            orientation: 'UNDIRECTED',
            properties: 'weight'
        }
    }
)
```

**Why UNDIRECTED?**
- Shared audience is bidirectional
- Language similarity is symmetric
- Better for capturing community structure

### 3. Node2Vec Embeddings

```cypher
CALL gds.node2vec.write('twitch', {
    embeddingDimension: 8,
    walkLength: 80,
    relationshipWeightProperty: 'weight',
    inOutFactor: 0.5,
    returnFactor: 1.0,
    writeProperty: 'embedding'
})
```

**Parameters:**
- `embeddingDimension`: 8 (balance between performance and dimensionality)
- `walkLength`: 80 (capture broader neighborhood)
- `inOutFactor`: 0.5 (balance BFS/DFS)
- `returnFactor`: 1.0 (standard random walk)

### 4. Machine Learning

**Model:** Random Forest Classifier

**Features:** 8-dimensional Node2Vec embeddings

**Evaluation:**
- Train/Test split: 80/20
- Stratified sampling (preserve language distribution)
- Metrics: Accuracy, Precision, Recall, F1-Score

##  Results

### Expected Performance

- **Accuracy**: ~90-92%
- **Weighted F1-Score**: ~0.91-0.93
- **Top Performing Languages**: English, Portuguese, Russian
- **Challenges**: Small language classes, English overlap

### Key Findings

1. **Cosine Similarity** outperforms Euclidean distance
2. **Higher degree nodes** have more reliable embeddings
3. **English streams** are sometimes misclassified (hypothesis confirmed)
4. **Strong language clusters** emerge in embedding space

### Sample Confusion Matrix Analysis

```
              precision    recall  f1-score   support
    en           0.91      0.93      0.92       384
    pt           0.96      0.93      0.94        54
    ru           0.96      0.92      0.94        59
    de           0.84      0.82      0.83        39
    ...
```

##  Discussion Points

### 1. Confusion Matrix Interpretation

**What to look for:**
- **Diagonal values**: Correct predictions (should be high)
- **Off-diagonal clusters**: Which languages are confused?
- **English column**: Often higher off-diagonal (overlaps with others)

**Insights:**
- Similar languages (Spanish/Portuguese) might cluster
- English acts as "universal language" (higher misclassification)
- Low-resource languages harder to predict

### 2. Appropriate Metrics for Manager

**Recommendation: Weighted F1-Score**

**Why not Accuracy?**
- Misleading with imbalanced classes
- English dominates (70%+) → high accuracy even with poor minority class performance

**Why F1-Score?**
- Balances Precision and Recall
- Weighted version accounts for class imbalance
- Business-relevant: Both false positives and false negatives matter

**Business Impact:**
- **False Positive**: Wrong language tag → Bad recommendations → Poor UX
- **False Negative**: Missing language tag → Lost discoverability → Reduced engagement

### 3. Improving Classifier Quality

**A. Better Embeddings:**
```python
# Try larger dimensions
ml.run_node2vec(embedding_dimension=32)

# Tune hyperparameters
ml.run_node2vec(
    walk_length=120,
    inOutFactor=0.3,
    returnFactor=1.5
)

# Try different algorithms
# GraphSAGE, GCN, FastRP
```

**B. Additional Features:**
```cypher
// Add centrality metrics
CALL gds.pageRank.write('twitch', {writeProperty: 'pagerank'})
CALL gds.betweenness.write('twitch', {writeProperty: 'betweenness'})

// Community detection
CALL gds.louvain.write('twitch', {writeProperty: 'community'})
```

**C. Model Improvements:**
```python
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Try different models
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
```

**D. Handle Class Imbalance:**
```python
# SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# Class weights
model = RandomForestClassifier(class_weight='balanced')

# Stratified k-fold CV
from sklearn.model_selection import StratifiedKFold
```

**E. Data Quality:**
- Remove dead/inactive accounts
- Filter low-degree nodes (insufficient connections)
- Add temporal features (streaming patterns)
- Incorporate viewer demographics

## 🔧 Advanced Extensions

### 1. Real-time Prediction

```python
# For new streamer
def predict_new_stream(stream_id):
    # Update graph with new connections
    # Generate embedding (incremental Node2Vec)
    # Predict language
    pass
```

### 2. Multi-language Streams

```python
# Softmax probabilities instead of hard classification
proba = model.predict_proba(embeddings)

# Threshold-based multi-label
languages = [lang for lang, p in zip(labels, proba[0]) if p > 0.3]
```

### 3. Confidence Scoring

```python
# Use prediction confidence
confidence = np.max(model.predict_proba(X), axis=1)

# Flag low-confidence predictions for human review
uncertain = confidence < 0.7
```

### 4. Temporal Analysis

```cypher
// Track language changes over time
MATCH (s:Stream)
WHERE s.createdAt > date('2024-01-01')
RETURN s.language, count(*) AS new_streams
ORDER BY new_streams DESC
```

### 5. A/B Testing

Compare Graph ML approach with:
- Audio transcription + NLP
- Manual tagging
- Baseline (most common language)

---

**Happy Graph Learning!** 

For questions or issues, check the notebook comments or Neo4j browser at http://localhost:7474
