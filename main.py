#!/usr/bin/env python3
"""
Video Transcript Q&A System - Main CLI Interface

This script provides a command-line interface for the Video Q&A system.
You can add YouTube videos, ask questions about their content, and manage the knowledge base.
"""

import click
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.video_qa_system import VideoQASystem
from src.gpu_utils import check_gpu_availability, get_gpu_memory_usage, clear_gpu_cache


@click.group()
@click.option('--llm-type', default='local', type=click.Choice(['local', 'openai', 'fallback']),
              help='Type of LLM to use (default: local)')
@click.pass_context
def cli(ctx, llm_type):
    """Video Transcript Q&A System - Ask questions about YouTube video content."""
    ctx.ensure_object(dict)
    ctx.obj['llm_type'] = llm_type

@cli.command()
@click.pass_context
def gpu_info(ctx):
    """Show GPU information and memory usage."""
    print("GPU Information:")
    print("-" * 20)
    check_gpu_availability()
    print()
    print(get_gpu_memory_usage())

@cli.command()
@click.pass_context
def clear_gpu(ctx):
    """Clear GPU cache to free up memory."""
    print("Clearing GPU cache...")
    clear_gpu_cache()
    print(get_gpu_memory_usage())

@cli.command()
@click.argument('video_urls', nargs=-1, required=True)
@click.option('--force-refresh', is_flag=True, help='Re-extract transcripts even if they exist')
@click.pass_context
def add_videos(ctx, video_urls, force_refresh):
    """Add one or more YouTube videos to the knowledge base."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    
    print(f"Adding {len(video_urls)} video(s) to the knowledge base...")
    results = qa_system.add_videos(list(video_urls), force_refresh)
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    failed = len(video_urls) - successful
    
    print(f"\nResults: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("\nFailed URLs:")
        for url, success in results.items():
            if not success:
                print(f"  - {url}")


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of relevant sources to consider')
@click.option('--show-sources', is_flag=True, help='Show the sources used for the answer')
@click.pass_context
def ask(ctx, question, top_k, show_sources):
    """Ask a question about the video content."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    
    if show_sources:
        print("Searching for relevant sources...")
        qa_system.search_and_display_sources(question, top_k)
        print("\n" + "="*60)
    
    print("Generating answer...")
    answer = qa_system.ask_question(question, top_k)
    print(f"\nAnswer: {answer}")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start an interactive Q&A session."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    qa_system.interactive_session()


@cli.command()
@click.pass_context
def list_videos(ctx):
    """List all videos in the knowledge base."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    videos = qa_system.list_videos()
    
    if not videos:
        print("No videos in the knowledge base.")
        return
    
    print(f"Videos in knowledge base ({len(videos)}):\n")
    
    for video in videos:
        print(f"Title: {video['title']}")
        print(f"Uploader: {video['uploader']}")
        print(f"Video ID: {video['video_id']}")
        print(f"Duration: {video['duration']} seconds")
        print(f"Chunks: {video['chunks']}")
        print(f"URL: {video['url']}")
        print("-" * 50)


@cli.command()
@click.argument('video_id')
@click.pass_context
def remove_video(ctx, video_id):
    """Remove a video from the knowledge base."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    
    if qa_system.remove_video(video_id):
        print(f"Video {video_id} removed successfully.")
    else:
        print(f"Failed to remove video {video_id}.")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    stats = qa_system.get_stats()
    
    print("Knowledge Base Statistics:")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")


@cli.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear(ctx, confirm):
    """Clear all videos from the knowledge base."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    
    if not confirm:
        if not click.confirm("Are you sure you want to clear all videos from the knowledge base?"):
            print("Operation cancelled.")
            return
    
    if qa_system.clear_knowledge_base():
        print("Knowledge base cleared successfully.")
    else:
        print("Failed to clear knowledge base.")


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of relevant sources to retrieve')
@click.pass_context
def search(ctx, question, top_k):
    """Search for relevant sources without generating an answer."""
    qa_system = VideoQASystem(llm_type=ctx.obj['llm_type'])
    qa_system.search_and_display_sources(question, top_k)


# Example usage function
def show_examples():
    """Show example usage commands."""
    examples = [
        "# Add a single video",
        "python main.py add-videos 'https://youtube.com/watch?v=VIDEO_ID'",
        "",
        "# Add multiple videos",
        "python main.py add-videos 'https://youtube.com/watch?v=ID1' 'https://youtube.com/watch?v=ID2'",
        "",
        "# Ask a question",
        "python main.py ask 'What is the main topic discussed?'",
        "",
        "# Ask a question with sources shown",
        "python main.py ask 'What is machine learning?' --show-sources",
        "",
        "# Start interactive session",
        "python main.py interactive",
        "",
        "# List all videos",
        "python main.py list-videos",
        "",
        "# Show statistics (including GPU info)",
        "python main.py stats",
        "",
        "# Show GPU information",
        "python main.py gpu-info",
        "",
        "# Clear GPU cache",
        "python main.py clear-gpu",
        "",
        "# Use OpenAI instead of local model",
        "python main.py --llm-type openai ask 'What is discussed in the videos?'",
    ]
    
    print("Example Usage:")
    print("=" * 50)
    for example in examples:
        print(example)


if __name__ == "__main__":
    # Show examples if no arguments provided
    if len(sys.argv) == 1:
        print("Video Transcript Q&A System")
        print("=" * 40)
        print("No command provided. Here are some examples:\n")
        show_examples()
        print("\nFor full help, run: python main.py --help")
    else:
        cli()
