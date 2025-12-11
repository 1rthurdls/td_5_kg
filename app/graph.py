"""
Graph Module - Neo4j Connection and Query Utilities
"""
from neo4j import GraphDatabase
import pandas as pd
import os


class GraphConnector:
    """
    Neo4j Graph Database Connector
    """
    
    def __init__(self, uri=None, user=None, password=None):
        """
        Initialize connection to Neo4j
        
        Args:
            uri: Neo4j connection URI (default: from env or localhost)
            user: Neo4j username (default: from env or 'neo4j')
            password: Neo4j password (default: from env)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'graphml2024')
        
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
    
    def run_query(self, query, parameters=None):
        """
        Execute a Cypher query and return results as DataFrame
        
        Args:
            query: Cypher query string
            parameters: Dictionary of query parameters
            
        Returns:
            pandas DataFrame with query results
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return pd.DataFrame([record.data() for record in result])
    
    def execute(self, query, parameters=None):
        """
        Execute a Cypher query without returning results
        
        Args:
            query: Cypher query string
            parameters: Dictionary of query parameters
        """
        with self.driver.session() as session:
            session.run(query, parameters or {})
    
    def test_connection(self):
        """
        Test the connection to Neo4j
        
        Returns:
            Boolean indicating if connection is successful
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Connection failed: {e}")
            return False


# Convenience function for quick queries
def query(cypher, parameters=None, uri=None, user=None, password=None):
    """
    Quick query execution function
    
    Args:
        cypher: Cypher query string
        parameters: Dictionary of query parameters
        uri, user, password: Connection credentials
        
    Returns:
        pandas DataFrame with query results
    """
    connector = GraphConnector(uri, user, password)
    try:
        return connector.run_query(cypher, parameters)
    finally:
        connector.close()
