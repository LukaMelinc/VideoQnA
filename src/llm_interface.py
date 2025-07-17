import os
from typing import List, Dict, Optional
from gpt4all import GPT4All
from config.config import Config
import torch


class LLMInterface:
    """Interface for interacting with Local Language Models (gpt4all Mistral)."""
    
    def __init__(self, model_type: str = "local"):
        self.config = Config()
        self.model_type = "local"
        
        # Check for GPU availability FIRST
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"LLM interface using device: {self.device}")
        
        self._initialize_local_model()
    
    def _initialize_local_model(self):
        """Initialize local gpt4all Mistral 7B model."""
        try:
            model_name = self.config.LLM_MODEL  # e.g., 'mistral-7b-instruct-v0.1.Q4_0.gguf'
            print(f"Loading local gpt4all model: {model_name}")
            
            # Initialize with GPU support if available
            if self.device == 'cuda':
                self.model = GPT4All(
                    model_name,
                    device='gpu'  # Use GPU if available
                )
                print(f"Local gpt4all model initialized successfully on GPU")
            else:
                self.model = GPT4All(model_name, device='cpu')
                print("Local gpt4all model initialized successfully on CPU")
                
        except Exception as e:
            print(f"Error initializing gpt4all model with GPU: {e}")
            try:
                # Fallback to CPU
                print("Falling back to CPU...")
                self.model = GPT4All(self.config.LLM_MODEL, device='cpu')
                self.device = 'cpu'
                print("Local gpt4all model initialized successfully on CPU")
            except Exception as cpu_error:
                print(f"Error initializing gpt4all model on CPU: {cpu_error}")
                print("Falling back to simple rule-based responses")
                self.model_type = "fallback"
    
    def generate_answer(self, question: str, context: List[Dict], max_tokens: int = None) -> str:
        """Generate an answer based on the question and retrieved context."""
        if max_tokens is None:
            max_tokens = self.config.MAX_TOKENS
        
        # Prepare context string
        context_text = self._format_context(context)
        
        if self.model_type == "local":
            return self._generate_local_answer(question, context_text, max_tokens)
        else:
            return self._generate_fallback_answer(question, context)
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format the retrieved context for the prompt."""
        if not context:
            return "No relevant context found."
        formatted_context = []
        for i, item in enumerate(context, 1):
            metadata = item['metadata']
            document = item['document']
            video_title = metadata.get('video_title', 'Unknown Video')
            uploader = metadata.get('uploader', 'Unknown')
            # Add timestamp if available
            timestamp_info = ""
            if 'start_time' in metadata:
                minutes = int(metadata['start_time'] // 60)
                seconds = int(metadata['start_time'] % 60)
                timestamp_info = f" (at {minutes}:{seconds:02d})"
            formatted_context.append(
                f"Source {i}: {video_title} by {uploader}{timestamp_info}\n"
                f"Content: {document}\n"
            )
        return "\n".join(formatted_context)
    
    def _generate_local_answer(self, question: str, context: str, max_tokens: int) -> str:
        """Generate answer using local gpt4all model."""
        try:
            prompt = self._create_prompt(question, context)
            
            # Generate with optimized parameters for GPU
            response = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=0.7,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1,
                streaming=False  # Set to True if you want streaming responses
            )
            
            answer = self._clean_response(response)
            return answer if answer else "I couldn't generate a clear answer based on the provided context."
            
        except Exception as e:
            print(f"Error generating gpt4all response: {e}")
            return self._generate_fallback_answer(question, [{'document': context, 'metadata': {}}])
    
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model."""
        prompt = f"""Based on the following video transcript excerpts, please answer the question. Be specific and cite which video the information comes from when possible.

Context from video transcripts:
{context}

Question: {question}

Answer: """
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up the model response."""
        response = response.replace("<|endoftext|>", "")
        response = response.replace("<pad>", "")
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Question:', 'Context:', 'Answer:')):
                cleaned_lines.append(line)
                if len(' '.join(cleaned_lines)) > 300:
                    break
        return ' '.join(cleaned_lines)
    
    def _generate_fallback_answer(self, question: str, context: List[Dict]) -> str:
        """Generate a simple fallback answer when models are not available."""
        if not context:
            return "I don't have enough information to answer your question. Please make sure you've added video transcripts to the database."
        videos = set()
        content_snippets = []
        for item in context:
            metadata = item.get('metadata', {})
            document = item.get('document', '')
            video_title = metadata.get('video_title', 'Unknown Video')
            videos.add(video_title)
            sentences = document.split('.')[:2]
            content_snippets.extend(sentences)
        video_list = ', '.join(videos)
        content_preview = '. '.join(content_snippets[:3])[:200] + "..."
        return f"""Based on the video transcripts from: {video_list}

Here's relevant content I found: {content_preview}

I found this information related to your question: "{question}". For more detailed analysis, consider using a local language model."""
    
    def ask_followup(self, original_question: str, followup_question: str, context: List[Dict]) -> str:
        """Handle follow-up questions with context from the original question."""
        combined_question = f"Original question: {original_question}\nFollow-up: {followup_question}"
        return self.generate_answer(combined_question, context)
