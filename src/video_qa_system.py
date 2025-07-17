from typing import List, Dict, Optional
from src.transcript_extractor import TranscriptExtractor
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from src.gpu_utils import check_gpu_availability, get_gpu_memory_usage
from config.config import Config


class VideoQASystem:
    """Main orchestrator for the Video Q&A system."""
    
    def __init__(self, llm_type: str = "local"):
        """
        Initialize the Video Q&A system.
        
        Args:
            llm_type: Type of LLM to use ("local", "openai", or "fallback")
        """
        self.config = Config()
        
        # Check GPU availability
        gpu_available = check_gpu_availability()
        
        self.transcript_extractor = TranscriptExtractor()
        self.vector_store = VectorStore()
        self.llm = LLMInterface(model_type=llm_type)
        
        print("Video Q&A System initialized successfully!")
        if gpu_available:
            print(get_gpu_memory_usage())
    
    def add_video(self, video_url: str, force_refresh: bool = False) -> bool:
        """
        Add a single video to the knowledge base.
        
        Args:
            video_url: YouTube video URL
            force_refresh: Whether to re-extract transcript if it already exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract transcript
            transcript_data = self.transcript_extractor.extract_and_save_transcript(
                video_url, force_refresh
            )
            
            if not transcript_data:
                return False
            
            # Add to vector database
            chunks_added = self.vector_store.add_transcript(transcript_data)
            print(f"Added video to knowledge base: {transcript_data['metadata']['title']}")
            print(f"Created {chunks_added} searchable chunks")
            
            return True
            
        except Exception as e:
            print(f"Error adding video {video_url}: {e}")
            return False
    
    def add_videos(self, video_urls: List[str], force_refresh: bool = False) -> Dict[str, bool]:
        """
        Add multiple videos to the knowledge base.
        
        Args:
            video_urls: List of YouTube video URLs
            force_refresh: Whether to re-extract transcripts if they already exist
            
        Returns:
            Dictionary mapping URLs to success status
        """
        results = {}
        
        for url in video_urls:
            print(f"\nProcessing: {url}")
            results[url] = self.add_video(url, force_refresh)
        
        successful = sum(1 for success in results.values() if success)
        print(f"\nSummary: {successful}/{len(video_urls)} videos added successfully")
        
        return results
    
    def ask_question(self, question: str, top_k: int = None) -> str:
        """
        Ask a question about the video content.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve (default from config)
            
        Returns:
            Generated answer
        """
        try:
            # Search for relevant context
            if top_k is None:
                top_k = self.config.TOP_K_RESULTS
            
            context = self.vector_store.search(question, top_k)
            
            if not context:
                return "I couldn't find any relevant information in the video transcripts to answer your question. Please make sure you've added videos to the knowledge base."
            
            # Generate answer
            answer = self.llm.generate_answer(question, context)
            
            return answer
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def get_relevant_sources(self, question: str, top_k: int = None) -> List[Dict]:
        """
        Get the most relevant sources for a question without generating an answer.
        
        Args:
            question: The question to search for
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant sources with metadata
        """
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        return self.vector_store.search(question, top_k)
    
    def list_videos(self) -> List[Dict]:
        """List all videos in the knowledge base."""
        return self.vector_store.list_videos()
    
    def remove_video(self, video_id: str) -> bool:
        """
        Remove a video from the knowledge base.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            removed_chunks = self.vector_store.remove_video(video_id)
            if removed_chunks > 0:
                print(f"Removed video {video_id} and {removed_chunks} chunks")
                return True
            else:
                print(f"Video {video_id} not found in database")
                return False
        except Exception as e:
            print(f"Error removing video {video_id}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        stats = self.vector_store.get_collection_stats()
        stats['gpu_memory'] = get_gpu_memory_usage()
        return stats
    
    def clear_knowledge_base(self) -> bool:
        """Clear all videos from the knowledge base."""
        return self.vector_store.clear_database()
    
    def interactive_session(self):
        """Start an interactive Q&A session."""
        print("\n=== Interactive Video Q&A Session ===")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'stats' to see knowledge base statistics")
        print("Type 'videos' to list all videos in the knowledge base")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\nKnowledge Base Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                if question.lower() == 'videos':
                    videos = self.list_videos()
                    if videos:
                        print(f"\nVideos in knowledge base ({len(videos)}):")
                        for video in videos:
                            print(f"  - {video['title']} ({video['chunks']} chunks)")
                    else:
                        print("\nNo videos in knowledge base.")
                    continue
                
                if not question:
                    continue
                
                print("\nThinking...")
                answer = self.ask_question(question)
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def search_and_display_sources(self, question: str, top_k: int = 3):
        """Search for sources and display them in a readable format."""
        sources = self.get_relevant_sources(question, top_k)
        
        if not sources:
            print("No relevant sources found.")
            return
        
        print(f"\nTop {len(sources)} relevant sources for: '{question}'")
        print("=" * 60)
        
        for i, source in enumerate(sources, 1):
            metadata = source['metadata']
            similarity = source['similarity']
            
            print(f"\nSource {i} (Similarity: {similarity:.2f}):")
            print(f"Video: {metadata.get('video_title', 'Unknown')}")
            print(f"Uploader: {metadata.get('uploader', 'Unknown')}")
            
            if 'start_time' in metadata:
                minutes = int(metadata['start_time'] // 60)
                seconds = int(metadata['start_time'] % 60)
                print(f"Timestamp: {minutes}:{seconds:02d}")
            
            print(f"Content: {source['document'][:200]}...")
            print("-" * 60)
