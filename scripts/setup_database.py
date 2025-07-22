import sys
import os
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DatabaseManager
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_database():
    """Initialize the database and create all required tables."""
    try:
        logger.info("Starting database setup...")
        
        # Create database manager instance
        db_manager = DatabaseManager()
        
        # Create tables
        db_manager.create_tables()
        
        logger.info("Database setup completed successfully!")
        logger.info("Tables created:")
        logger.info("- medical_chunks (with pgvector extension and IVFFlat index)")
        
        # Close connection
        db_manager.close()
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_database()