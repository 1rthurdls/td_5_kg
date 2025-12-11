"""
Data Loader Module - Load Twitch streaming data into Neo4j
"""
from graph import GraphConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitchDataLoader:
    """
    Load Twitch streaming data into Neo4j graph database
    """
    
    def __init__(self, connector: GraphConnector):
        self.graph = connector
    
    def create_constraints(self):
        """Create database constraints for data integrity"""
        logger.info("Creating constraints...")
        
        # Constraint for Stream nodes
        self.graph.execute("""
            CREATE CONSTRAINT stream_id IF NOT EXISTS
            FOR (s:Stream)
            REQUIRE s.streamId IS UNIQUE
        """)
        
        logger.info("✓ Constraints created")
    
    def load_streams(self, csv_url="https://bit.ly/3JjgKgZ"):
        """
        Load stream data from CSV
        
        Args:
            csv_url: URL to CSV file with stream data
        """
        logger.info(f"Loading streams from {csv_url}...")
        
        query = """
            LOAD CSV WITH HEADERS FROM $url AS row
            MERGE (s:Stream {streamId: toInteger(row.new_id)})
            SET s.language = row.language,
                s.deadAccount = row.dead_account = 'True',
                s.mature = row.mature = 'True',
                s.views = toInteger(row.views),
                s.createdAt = datetime(row.created_at)
            RETURN count(s) AS streamsLoaded
        """
        
        result = self.graph.run_query(query, {"url": csv_url})
        count = result.iloc[0]['streamsLoaded']
        logger.info(f"✓ Loaded {count} streams")
        
        return count
    
    def load_shared_audience(self, csv_url="https://bit.ly/3S9Uyd8", batch_size=5000):
        """
        Load shared audience relationships between streams
        
        Args:
            csv_url: URL to CSV file with relationship data
            batch_size: Number of rows to process per transaction
        """
        logger.info(f"Loading shared audience relationships from {csv_url}...")
        
        query = """
            CALL {
                LOAD CSV WITH HEADERS FROM $url AS row
                MATCH (s1:Stream {streamId: toInteger(row.node_1)})
                MATCH (s2:Stream {streamId: toInteger(row.node_2)})
                MERGE (s1)-[r:SHARED_AUDIENCE]-(s2)
                SET r.weight = toInteger(row.weight)
            } IN TRANSACTIONS OF $batchSize ROWS
            RETURN count(*) AS relationshipsLoaded
        """
        
        result = self.graph.run_query(
            query, 
            {"url": csv_url, "batchSize": batch_size}
        )
        
        count = result.iloc[0]['relationshipsLoaded'] if not result.empty else 0
        logger.info(f"✓ Loaded {count} SHARED_AUDIENCE relationships")
        
        return count
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}
        
        # Count streams
        result = self.graph.run_query("MATCH (s:Stream) RETURN count(s) AS count")
        stats['streams'] = result.iloc[0]['count']
        
        # Count relationships
        result = self.graph.run_query(
            "MATCH ()-[r:SHARED_AUDIENCE]-() RETURN count(r) AS count"
        )
        stats['relationships'] = result.iloc[0]['count']
        
        # Count languages
        result = self.graph.run_query("""
            MATCH (s:Stream)
            RETURN s.language AS language, count(*) AS count
            ORDER BY count DESC
        """)
        stats['languages'] = result.to_dict('records')
        
        return stats
    
    def load_all(self):
        """Load all data in correct order"""
        logger.info("Starting full data load...")
        
        self.create_constraints()
        self.load_streams()
        self.load_shared_audience()
        
        stats = self.get_statistics()
        logger.info(f"✓ Data load complete: {stats}")
        
        return stats


def main():
    """Main execution function"""
    connector = GraphConnector()
    
    # Test connection
    if not connector.test_connection():
        logger.error("Failed to connect to Neo4j")
        return
    
    logger.info("✓ Connected to Neo4j")
    
    # Load data
    loader = TwitchDataLoader(connector)
    loader.load_all()
    
    connector.close()


if __name__ == "__main__":
    main()
