"""
Machine Learning Module - Node embeddings and classification
"""
from graph import GraphConnector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphMLPipeline:
    """
    Graph Machine Learning Pipeline for Twitch stream classification
    """
    
    def __init__(self, connector: GraphConnector):
        self.graph = connector
        self.model = None
        self.label_mapping = None
    
    def create_graph_projection(self, graph_name="twitch", orientation="UNDIRECTED"):
        """
        Create a graph projection for GDS algorithms
        
        Args:
            graph_name: Name for the graph projection
            orientation: 'UNDIRECTED' or 'DIRECTED'
            
        Returns:
            DataFrame with projection statistics
        """
        logger.info(f"Creating graph projection '{graph_name}'...")
        
        # Drop existing projection if it exists
        self.graph.execute(f"""
            CALL gds.graph.drop('{graph_name}', false)
            YIELD graphName
        """)
        
        # Create new projection
        query = f"""
            CALL gds.graph.project(
                '{graph_name}',
                'Stream',
                {{
                    SHARED_AUDIENCE: {{
                        orientation: '{orientation}',
                        properties: 'weight'
                    }}
                }}
            )
            YIELD graphName, nodeCount, relationshipCount, projectMillis
            RETURN graphName, nodeCount, relationshipCount, projectMillis
        """
        
        result = self.graph.run_query(query)
        logger.info(f"✓ Graph projection created: {result.to_dict('records')[0]}")
        
        return result
    
    def run_node2vec(self, graph_name="twitch", embedding_dimension=8, 
                     walk_length=80, iterations=10):
        """
        Run Node2Vec algorithm to generate node embeddings
        
        Args:
            graph_name: Name of the graph projection
            embedding_dimension: Dimension of embedding vectors
            walk_length: Length of random walks
            iterations: Number of iterations
            
        Returns:
            DataFrame with algorithm statistics
        """
        logger.info("Running Node2Vec algorithm...")
        
        query = f"""
            CALL gds.node2vec.write('{graph_name}', {{
                embeddingDimension: $embeddingDim,
                relationshipWeightProperty: 'weight',
                walkLength: $walkLength,
                iterations: $iterations,
                inOutFactor: 0.5,
                returnFactor: 1.0,
                writeProperty: 'embedding'
            }})
            YIELD nodeCount, nodePropertiesWritten, computeMillis, configuration
            RETURN nodeCount, nodePropertiesWritten, computeMillis, 
                   configuration.embeddingDimension AS embeddingDimension
        """
        
        result = self.graph.run_query(query, {
            "embeddingDim": embedding_dimension,
            "walkLength": walk_length,
            "iterations": iterations
        })
        
        logger.info(f"✓ Node2Vec complete: {result.to_dict('records')[0]}")
        
        return result
    
    def analyze_embedding_distances(self):
        """
        Analyze distances between embeddings of connected nodes
        
        Returns:
            DataFrame with distance metrics
        """
        logger.info("Analyzing embedding distances...")
        
        query = """
            MATCH (s1:Stream)-[r:SHARED_AUDIENCE]-(s2:Stream)
            WHERE s1.streamId < s2.streamId
            WITH s1, s2, r,
                 gds.similarity.euclidean(s1.embedding, s2.embedding) AS euclidean,
                 gds.similarity.cosine(s1.embedding, s2.embedding) AS cosine
            RETURN euclidean, cosine, r.weight AS weight
        """
        
        df = self.graph.run_query(query)
        logger.info(f"✓ Analyzed {len(df)} node pairs")
        
        return df
    
    def plot_distance_distributions(self, df):
        """
        Plot distribution of distances between connected nodes
        
        Args:
            df: DataFrame with euclidean and cosine distances
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Euclidean distance
        sns.histplot(df['euclidean'], bins=50, ax=axes[0], color='blue', kde=True)
        axes[0].set_title('Euclidean Distance Distribution', fontsize=14)
        axes[0].set_xlabel('Euclidean Distance')
        axes[0].set_ylabel('Frequency')
        
        # Cosine similarity
        sns.histplot(df['cosine'], bins=50, ax=axes[1], color='green', kde=True)
        axes[1].set_title('Cosine Similarity Distribution', fontsize=14)
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def analyze_degree_by_similarity(self):
        """
        Analyze average degree by cosine similarity buckets
        
        Returns:
            DataFrame with degree statistics by similarity
        """
        query = """
            MATCH (s1:Stream)-[r:SHARED_AUDIENCE]-(s2:Stream)
            WHERE s1.streamId < s2.streamId
            WITH s1, s2,
                 gds.similarity.cosine(s1.embedding, s2.embedding) AS cosine,
                 size((s1)-[:SHARED_AUDIENCE]-()) AS degree1,
                 size((s2)-[:SHARED_AUDIENCE]-()) AS degree2,
                 r.weight AS weight
            WITH round(cosine * 10) / 10 AS cosineBucket,
                 avg((degree1 + degree2) / 2.0) AS avgDegree,
                 avg(weight) AS avgWeight,
                 count(*) AS pairCount
            RETURN cosineBucket AS cosineSimilarity, 
                   avgDegree, 
                   avgWeight, 
                   pairCount
            ORDER BY cosineSimilarity
        """
        
        return self.graph.run_query(query)
    
    def plot_degree_by_similarity(self, df):
        """Plot average degree by cosine similarity"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='cosineSimilarity', y='avgDegree', 
                   color='blue', ax=ax)
        ax.set_title('Average Degree by Cosine Similarity', fontsize=14)
        ax.set_xlabel('Cosine Similarity Bucket')
        ax.set_ylabel('Average Degree')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_weight_by_similarity(self, df):
        """Plot average weight by cosine similarity"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x='cosineSimilarity', y='avgWeight', 
                   color='green', ax=ax)
        ax.set_title('Average Relationship Weight by Cosine Similarity', fontsize=14)
        ax.set_xlabel('Cosine Similarity Bucket')
        ax.set_ylabel('Average Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def prepare_training_data(self):
        """
        Prepare data for machine learning
        
        Returns:
            DataFrame with embeddings and language labels
        """
        logger.info("Preparing training data...")
        
        query = """
            MATCH (s:Stream)
            WHERE s.embedding IS NOT NULL
            RETURN s.streamId AS streamId,
                   s.language AS language,
                   s.embedding AS embedding
        """
        
        df = self.graph.run_query(query)
        
        # Encode language labels
        df['languageEncoded'], self.label_mapping = pd.factorize(df['language'])
        
        logger.info(f"✓ Prepared {len(df)} training samples")
        logger.info(f"✓ Found {len(self.label_mapping)} unique languages")
        
        return df
    
    def train_classifier(self, df, test_size=0.2, random_state=42, **rf_params):
        """
        Train a Random Forest classifier
        
        Args:
            df: DataFrame with embeddings and labels
            test_size: Proportion of test set
            random_state: Random seed
            **rf_params: Additional RandomForestClassifier parameters
            
        Returns:
            Dictionary with model and evaluation results
        """
        logger.info("Training Random Forest classifier...")
        
        # Prepare features and labels
        X = np.array(df['embedding'].tolist())
        y = df['languageEncoded'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(random_state=random_state, **rf_params)
        self.model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_mapping.astype(str),
            output_dict=True
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"✓ Model trained with accuracy: {report['accuracy']:.4f}")
        
        return {
            'model': self.model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': report,
            'confusion_matrix': cm,
            'label_mapping': self.label_mapping
        }
    
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Confusion Matrix - Language Classification', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def print_classification_report(self, report):
        """Print formatted classification report"""
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        
        # Print per-class metrics
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"\nLanguage: {label}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1-score']:.4f}")
                print(f"  Support:   {metrics['support']}")
        
        # Print aggregated metrics
        print("\n" + "-"*70)
        print(f"Accuracy:          {report['accuracy']:.4f}")
        print(f"Macro Avg F1:      {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1:   {report['weighted avg']['f1-score']:.4f}")
        print("="*70 + "\n")
