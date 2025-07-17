import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the Video Q&A system."""
    
    def __init__(self):
        # GPU settings
        self.USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
        self.GPU_BATCH_SIZE = int(os.getenv('GPU_BATCH_SIZE', '32'))
        self.GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
        
        # Paths
        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.TRANSCRIPTS_PATH = os.getenv("TRANSCRIPTS_PATH", "./data/transcripts")
        
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Model Settings
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct-v0.1.Q4_0.gguf")
        
        # Text Processing
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Vector Database
        self.COLLECTION_NAME = "video_transcripts"
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
        
        # System Settings
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(self.TRANSCRIPTS_PATH, exist_ok=True)