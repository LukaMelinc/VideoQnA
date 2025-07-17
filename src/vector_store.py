import os
import uuid
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config.config import Config
import numpy as np
import torch

class VectorStore:
    """Handles vector database operations using ChromaDB."""
    
    def __init__(self):
        self.config = Config()
        self.config.ensure_directories()

        # Check for GPU availability (fixed typo)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Vector store using device: {self.device}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=self.device)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"description": "Video transcript embeddings"}
        )
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if overlap is None:
            overlap = self.config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                best_break = end
                
                for i in range(max(0, end - 100), end):
                    if any(text[i:i+2] == ending for ending in sentence_endings):
                        best_break = i + 2
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_transcript(self, transcript_data: Dict) -> int:
        """Add a transcript to the vector database."""
        video_id = transcript_data['video_id']
        metadata = transcript_data['metadata']
        transcript_text = transcript_data['transcript']
        segments = transcript_data.get('segments', [])
        
        # Remove existing data for this video
        self.remove_video(video_id)
        
        # Chunk the transcript
        chunks = self.chunk_text(transcript_text)
        
        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{video_id}_{i}"
            
            # Find relevant segments for this chunk
            chunk_segments = self._find_segments_for_chunk(chunk, segments)
            
            metadata_entry = {
                'video_id': video_id,
                'chunk_index': i,
                'video_title': metadata.get('title', 'Unknown'),
                'uploader': metadata.get('uploader', 'Unknown'),
                'upload_date': metadata.get('upload_date', 'Unknown'),
                'video_url': metadata.get('url', ''),
                'duration': metadata.get('duration', 0),
                'segments_count': len(chunk_segments),
                'text_length': len(chunk)
            }
            
            # Add timestamp info if available
            if chunk_segments:
                metadata_entry['start_time'] = min(seg['start'] for seg in chunk_segments)
                metadata_entry['end_time'] = max(seg['start'] + seg['duration'] for seg in chunk_segments)
            
            documents.append(chunk)
            metadatas.append(metadata_entry)
            ids.append(doc_id)
        
        # Generate embeddings
        #embeddings = self.embedding_model.encode(documents).tolist()
        # Generate embeddings with batch processing for better GPU utilization
        print(f"Generating embeddings on {self.device}...")
        embeddings = self.embedding_model.encode(
            documents, 
            batch_size=32,  # Adjust based on your GPU memory
            show_progress_bar=True,
            convert_to_tensor=True
        ).cpu().numpy().tolist()  # Move back to CPU for ChromaDB
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks for video: {metadata.get('title', video_id)}")
        return len(chunks)
    
    def _find_segments_for_chunk(self, chunk_text: str, segments: List[Dict]) -> List[Dict]:
        """Find which transcript segments correspond to a text chunk."""
        # This is a simple heuristic - find segments whose text appears in the chunk
        relevant_segments = []
        chunk_lower = chunk_text.lower()
        
        for segment in segments:
            if segment['text'].lower() in chunk_lower:
                relevant_segments.append(segment)
        
        return relevant_segments
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant transcript chunks."""
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        # Generate query embedding on GPU
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_tensor=True
        ).cpu().numpy().tolist()[0]  # Move back to CPU for ChromaDB
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def remove_video(self, video_id: str) -> int:
        """Remove all chunks for a specific video."""
        # Get all items for this video
        try:
            results = self.collection.get(
                where={"video_id": video_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Removed {len(results['ids'])} chunks for video {video_id}")
                return len(results['ids'])
        except Exception as e:
            print(f"Error removing video {video_id}: {e}")
        
        return 0
    
    def list_videos(self) -> List[Dict]:
        """List all videos in the database."""
        try:
            # Get all unique videos
            results = self.collection.get(include=['metadatas'])
            
            videos = {}
            for metadata in results['metadatas']:
                video_id = metadata['video_id']
                if video_id not in videos:
                    videos[video_id] = {
                        'video_id': video_id,
                        'title': metadata.get('video_title', 'Unknown'),
                        'uploader': metadata.get('uploader', 'Unknown'),
                        'upload_date': metadata.get('upload_date', 'Unknown'),
                        'url': metadata.get('video_url', ''),
                        'duration': metadata.get('duration', 0),
                        'chunks': 0
                    }
                videos[video_id]['chunks'] += 1
            
            return list(videos.values())
        except Exception as e:
            print(f"Error listing videos: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            videos = self.list_videos()
            
            return {
                'total_chunks': count,
                'total_videos': len(videos),
                'collection_name': self.config.COLLECTION_NAME,
                'embedding_model': self.config.EMBEDDING_MODEL
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.config.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "Video transcript embeddings"}
            )
            print("Database cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False
