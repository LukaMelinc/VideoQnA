#!/usr/bin/env python3
"""
Flask web application for Video Transcript Q&A System
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import sys
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.video_qa_system import VideoQASystem
from src.gpu_utils import check_gpu_availability, get_gpu_memory_usage

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Global QA system instance
qa_system = None

def init_qa_system():
    """Initialize the QA system"""
    global qa_system
    if qa_system is None:
        try:
            qa_system = VideoQASystem(llm_type="local")
            print("QA System initialized successfully")
        except Exception as e:
            print(f"Error initializing QA system: {e}")
            qa_system = VideoQASystem(llm_type="fallback")

@app.before_first_request
def startup():
    """Initialize the system on first request"""
    init_qa_system()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/add-video', methods=['GET', 'POST'])
def add_video():
    """Add video page"""
    if request.method == 'POST':
        try:
            video_url = request.form.get('video_url', '').strip()
            force_refresh = request.form.get('force_refresh') == 'on'
            
            if not video_url:
                flash('Please provide a video URL', 'error')
                return render_template('add_video.html')
            
            # Add video to system
            success = qa_system.add_video(video_url, force_refresh)
            
            if success:
                flash('Video added successfully!', 'success')
            else:
                flash('Failed to add video. Please check the URL and try again.', 'error')
                
        except Exception as e:
            flash(f'Error adding video: {str(e)}', 'error')
    
    return render_template('add_video.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    """Ask question page"""
    answer = None
    sources = None
    
    if request.method == 'POST':
        try:
            question = request.form.get('question', '').strip()
            show_sources = request.form.get('show_sources') == 'on'
            
            if not question:
                flash('Please enter a question', 'error')
            else:
                # Get answer
                answer = qa_system.ask_question(question)
                
                # Get sources if requested
                if show_sources:
                    sources = qa_system.get_relevant_sources(question, top_k=3)
                
        except Exception as e:
            flash(f'Error processing question: {str(e)}', 'error')
    
    return render_template('ask_question.html', answer=answer, sources=sources)

@app.route('/videos')
def list_videos():
    """List videos page"""
    try:
        videos = qa_system.list_videos()
        stats = qa_system.get_stats()
        return render_template('videos.html', videos=videos, stats=stats)
    except Exception as e:
        flash(f'Error loading videos: {str(e)}', 'error')
        return render_template('videos.html', videos=[], stats={})

@app.route('/api/remove-video/<video_id>', methods=['POST'])
def remove_video_api(video_id):
    """API endpoint to remove a video"""
    try:
        success = qa_system.remove_video(video_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat interface"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'})
        
        answer = qa_system.ask_question(question)
        sources = qa_system.get_relevant_sources(question, top_k=2)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        stats = qa_system.get_stats()
        gpu_info = get_gpu_memory_usage()
        
        return jsonify({
            'status': 'healthy',
            'stats': stats,
            'gpu_info': gpu_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("Starting Video Q&A Web Application...")
    check_gpu_availability()
    
    app.run(host='0.0.0.0', port=port, debug=debug)
