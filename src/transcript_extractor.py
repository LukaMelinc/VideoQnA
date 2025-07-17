import re
import json
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
from config.config import Config


class TranscriptExtractor:
    """Handles extraction of transcripts from YouTube videos."""
    
    def __init__(self):
        self.config = Config()
        self.config.ensure_directories()
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If it's already a video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        return None
    
    def get_video_metadata(self, video_id: str) -> Dict:
        """Get video metadata using yt-dlp."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                return {
                    'title': info.get('title', 'Unknown Title'),
                    'uploader': info.get('uploader', 'Unknown Uploader'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', 'Unknown Date'),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                }
        except Exception as e:
            print(f"Error getting metadata for {video_id}: {e}")
            return {
                'title': f'Video {video_id}',
                'uploader': 'Unknown',
                'duration': 0,
                'upload_date': 'Unknown',
                'description': '',
                'view_count': 0,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            }
    
    def extract_transcript(self, video_id: str, languages: List[str] = ['en']) -> Optional[Dict]:
        """Extract transcript from a YouTube video."""
        try:
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find transcript in preferred languages
            transcript = None
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except:
                    continue
            
            # If no transcript in preferred languages, try auto-generated English
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Try any available transcript
                    try:
                        transcript = transcript_list.find_transcript(['en'])
                    except:
                        available_transcripts = list(transcript_list)
                        if available_transcripts:
                            transcript = available_transcripts[0]
                        else:
                            raise Exception("No transcripts available")
            
            # Fetch the transcript data
            transcript_data = transcript.fetch()
            
            # Format the transcript
            formatter = TextFormatter()
            formatted_transcript = formatter.format_transcript(transcript_data)
            
            # Get metadata
            metadata = self.get_video_metadata(video_id)
            
            # Create segments with timestamps
            segments = []
            for entry in transcript_data:
                segments.append({
                    'start': entry.start,
                    'duration': entry.duration,
                    'text': entry.text
                })
            
            return {
                'video_id': video_id,
                'metadata': metadata,
                'transcript': formatted_transcript,
                'segments': segments,
                'language': transcript.language_code
            }
            
        except Exception as e:
            print(f"Error extracting transcript for {video_id}: {e}")
            return None
    
    def save_transcript(self, transcript_data: Dict, filename: Optional[str] = None) -> str:
        """Save transcript data to JSON file."""
        if filename is None:
            video_id = transcript_data['video_id']
            filename = f"{video_id}_transcript.json"
        
        filepath = os.path.join(self.config.TRANSCRIPTS_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_transcript(self, video_id: str) -> Optional[Dict]:
        """Load transcript data from JSON file."""
        filename = f"{video_id}_transcript.json"
        filepath = os.path.join(self.config.TRANSCRIPTS_PATH, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def extract_and_save_transcript(self, video_url: str, force_refresh: bool = False) -> Optional[Dict]:
        """Extract transcript from URL and save to file."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            print(f"Could not extract video ID from URL: {video_url}")
            return None
        
        # Check if transcript already exists
        if not force_refresh:
            existing_transcript = self.load_transcript(video_id)
            if existing_transcript:
                print(f"Transcript for {video_id} already exists. Use force_refresh=True to update.")
                return existing_transcript
        
        # Extract transcript
        transcript_data = self.extract_transcript(video_id)
        if transcript_data:
            filepath = self.save_transcript(transcript_data)
            print(f"Transcript saved to: {filepath}")
            return transcript_data
        else:
            print(f"Failed to extract transcript for {video_id}")
            return None
    
    def batch_extract_transcripts(self, video_urls: List[str], force_refresh: bool = False) -> List[Dict]:
        """Extract transcripts from multiple videos."""
        results = []
        
        for url in video_urls:
            print(f"Processing: {url}")
            transcript_data = self.extract_and_save_transcript(url, force_refresh)
            if transcript_data:
                results.append(transcript_data)
        
        print(f"Successfully processed {len(results)} out of {len(video_urls)} videos.")
        return results
