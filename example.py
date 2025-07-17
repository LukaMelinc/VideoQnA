#!/usr/bin/env python3
"""
Example usage of the Video Transcript Q&A System
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.video_qa_system import VideoQASystem


def main():
    """Demonstrate the Video Q&A system with example videos."""
    
    print("Video Transcript Q&A System - Example Usage")
    print("=" * 50)
    
    # Initialize the system
    print("Initializing Video Q&A System...")
    qa_system = VideoQASystem(llm_type="local")  # Use local model
    
    # Example video URLs (replace with actual URLs you want to test)
    example_videos = [
        "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown neural networks
        "https://www.youtube.com/watch?v=kCc8FmEb1nY",  # Machine learning explained
    ]
    
    print(f"\nAdding {len(example_videos)} example videos...")
    print("Note: Replace these URLs with videos you want to analyze")
    
    # Add videos (this will extract transcripts and create embeddings)
    results = qa_system.add_videos(example_videos)
    
    # Check results
    successful = sum(1 for success in results.values() if success)
    if successful == 0:
        print("No videos were successfully added. Please check the URLs and try again.")
        print("You can also add videos manually using:")
        print("python main.py add-videos 'YOUR_YOUTUBE_URL_HERE'")
        return
    
    print(f"\nSuccessfully added {successful} videos to the knowledge base")
    
    # Show knowledge base stats
    stats = qa_system.get_stats()
    print(f"\nKnowledge base statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example questions
    example_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the main concepts explained?",
        "What examples are given?",
    ]
    
    print(f"\nAsking example questions...")
    print("-" * 30)
    
    for question in example_questions:
        print(f"\nQ: {question}")
        try:
            answer = qa_system.ask_question(question)
            print(f"A: {answer}")
        except Exception as e:
            print(f"A: Error generating answer: {e}")
        print("-" * 30)
    
    # Show interactive mode option
    print("\nTo continue with interactive mode, run:")
    print("python main.py interactive")


if __name__ == "__main__":
    main()
