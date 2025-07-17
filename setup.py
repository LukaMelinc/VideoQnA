#!/usr/bin/env python3
"""
Setup script for Video Transcript Q&A System
"""

import os
import sys
import subprocess
from pathlib import Path


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        with open(env_example) as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✓ Created .env file")
        print("Please edit .env file to add your API keys if needed")
    else:
        print("✓ .env file already exists")


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/transcripts",
        "data/chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_requirements():
    """Install Python requirements."""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")
    try:
        # Test imports
        from src.video_qa_system import VideoQASystem
        
        # Initialize system (this will test model loading)
        qa_system = VideoQASystem(llm_type="fallback")
        
        print("✓ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Video Transcript Q&A System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Test functionality
    if not test_basic_functionality():
        print("\nSetup completed with warnings. Basic functionality test failed.")
        print("You may need to check your dependencies or configuration.")
    else:
        print("\n✓ Setup completed successfully!")
    
    print("\nNext steps:")
    print("1. Edit .env file to add your API keys (optional)")
    print("2. Add videos: python main.py add-videos 'https://youtube.com/watch?v=VIDEO_ID'")
    print("3. Ask questions: python main.py ask 'What is the main topic?'")
    print("4. Start interactive mode: python main.py interactive")


if __name__ == "__main__":
    main()
