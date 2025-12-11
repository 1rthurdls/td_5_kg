"""
Visualization Module - Advanced plotting for graph ML analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 12


def plot_language_distribution(df_languages):
    """
    Plot distribution of languages in the dataset
    
    Args:
        df_languages: DataFrame with 'language' and 'count' columns
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by count
    df_sorted = df_languages.sort_values('count', ascending=False).head(15)
    
    # Create bar plot
    bars = ax.bar(df_sorted['language'], df_sorted['count'], 
                  color=sns.color_palette("viridis", len(df_sorted)))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Top 15 Languages in Twitch Streams', fontsize=16, pad=20)
    ax.set_xlabel('Language', fontsize=14)
    ax.set_ylabel('Number of Streams', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_degree_distribution(graph_connector):
    """
    Plot degree distribution of the graph
    
    Args:
        graph_connector: GraphConnector instance
    """
    query = """
        MATCH (s:Stream)
        WITH s, size((s)-[:SHARED_AUDIENCE]-()) AS degree
        RETURN degree, count(*) AS count
        ORDER BY degree
    """
    
    df = graph_connector.run_query(query)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    axes[0].bar(df['degree'], df['count'], color='steelblue', alpha=0.7)
    axes[0].set_title('Degree Distribution (Linear Scale)', fontsize=14)
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Number of Nodes')
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].bar(df['degree'], df['count'], color='steelblue', alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_title('Degree Distribution (Log Scale)', fontsize=14)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Number of Nodes (log)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_embedding_space_2d(df, method='pca'):
    """
    Visualize embeddings in 2D using dimensionality reduction
    
    Args:
        df: DataFrame with 'embedding' and 'language' columns
        method: 'pca' or 'tsne'
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Extract embeddings
    X = np.array(df['embedding'].tolist())
    
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = 'Embedding Space Visualization (PCA)'
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 'Embedding Space Visualization (t-SNE)'
    
    X_reduced = reducer.fit_transform(X)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get unique languages and colors
    languages = df['language'].unique()
    colors = sns.color_palette("husl", len(languages))
    
    # Plot each language
    for i, lang in enumerate(languages):
        mask = df['language'] == lang
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                  c=[colors[i]], label=lang, alpha=0.6, s=50)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, top_n=10):
    """
    Plot feature importance from Random Forest model
    
    Args:
        model: Trained RandomForestClassifier
        top_n: Number of top features to show
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(range(top_n), importances[indices], color='steelblue', alpha=0.7)
    ax.set_title(f'Top {top_n} Feature Importances (Embedding Dimensions)', 
                fontsize=14)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f'Dim {i}' for i in indices])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_learning_curve(X_train, y_train, model, cv=5):
    """
    Plot learning curve to diagnose bias/variance
    
    Args:
        X_train: Training features
        y_train: Training labels
        model: Scikit-learn model
        cv: Number of cross-validation folds
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train_sizes, train_mean, label='Training score', 
           color='blue', marker='o')
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color='blue')
    
    ax.plot(train_sizes, val_mean, label='Cross-validation score',
           color='red', marker='s')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color='red')
    
    ax.set_title('Learning Curve', fontsize=14)
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(embeddings, sample_size=1000):
    """
    Plot correlation matrix of embedding dimensions
    
    Args:
        embeddings: Array of embeddings
        sample_size: Number of samples to use (for performance)
    """
    # Sample if too large
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
    
    # Compute correlation
    corr = np.corrcoef(embeddings.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr, cmap='coolwarm', center=0, 
               square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Embedding Dimensions Correlation Matrix', fontsize=14, pad=20)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    
    plt.tight_layout()
    return fig


def create_analysis_dashboard(results, graph_connector):
    """
    Create comprehensive analysis dashboard
    
    Args:
        results: Dictionary with training results
        graph_connector: GraphConnector instance
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    cm = results['confusion_matrix']
    labels = results['label_mapping'].astype(str)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, pad=10)
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # 2. Per-class F1 scores
    ax2 = fig.add_subplot(gs[0, 2])
    report = results['classification_report']
    languages = [k for k in report.keys() 
                if k not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = [report[lang]['f1-score'] for lang in languages]
    ax2.barh(languages, f1_scores, color='steelblue')
    ax2.set_xlabel('F1-Score', fontsize=10)
    ax2.set_title('F1-Score by Language', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Feature importance
    ax3 = fig.add_subplot(gs[1, 2])
    importances = results['model'].feature_importances_[:8]
    ax3.bar(range(len(importances)), importances, color='coral')
    ax3.set_xlabel('Embedding Dimension', fontsize=10)
    ax3.set_ylabel('Importance', fontsize=10)
    ax3.set_title('Top Feature Importances', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    metrics_text = f"""
    Overall Performance Metrics:
    
    Accuracy:         {report['accuracy']:.4f}
    Macro Avg F1:     {report['macro avg']['f1-score']:.4f}
    Weighted Avg F1:  {report['weighted avg']['f1-score']:.4f}
    
    Training Samples: {len(results['X_train'])}
    Test Samples:     {len(results['X_test'])}
    Number of Classes: {len(labels)}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Graph ML Analysis Dashboard', fontsize=18, y=0.98)
    
    return fig
