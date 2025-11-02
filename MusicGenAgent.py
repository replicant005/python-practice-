
# this agent genrates muisic, intead of using an API, 
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class MusicGenerationAgent:
    """
    Agent that generates music from text prompts using Meta's MusicGen model.
    Runs entirely locally without requiring external APIs.
    """
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize the music generation agent.
        
        Args:
            model_size: Size of model to use. Options:
                - "small" (300M params, fastest)
                - "medium" (1.5B params, balanced)
                - "large" (3.3B params, best quality)
                - "melody" (1.5B params, can condition on melody)
        """
        self.model_size = model_size
        self.model_name = f"facebook/musicgen-{model_size}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading MusicGen {model_size} model on {self.device}...")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        
        # Set default generation parameters
        self.sample_rate = self.model.config.audio_encoder.sampling_rate
        
        logger.info("Model loaded successfully!")
    
    def generate_music(
        self,
        prompt: str,
        duration: float = 8.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        output_path: Optional[str] = None
    ) -> tuple[np.ndarray, int]:
        """
        Generate music from a text prompt.
        
        Args:
            prompt: Text description of the music to generate
            duration: Length of audio to generate in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            output_path: Optional path to save the audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        logger.info(f"Generating music for prompt: '{prompt}'")
        logger.info(f"Duration: {duration}s, Temperature: {temperature}")
        
        # Process the text prompt
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Calculate max new tokens based on duration
        # MusicGen generates at ~50 tokens per second
        max_new_tokens = int(duration * 50)
        
        # Generate audio
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
            )
        
        # Convert to numpy array
        audio_array = audio_values[0, 0].cpu().numpy()
        
        # Save if output path provided
        if output_path:
            self._save_audio(audio_array, output_path)
            logger.info(f"Audio saved to: {output_path}")
        
        return audio_array, self.sample_rate
    
    def generate_batch(
        self,
        prompts: list[str],
        duration: float = 8.0,
        output_dir: str = "generated_music"
    ) -> list[tuple[np.ndarray, int]]:
        """
        Generate multiple music tracks from a list of prompts.
        
        Args:
            prompts: List of text descriptions
            duration: Length of each audio in seconds
            output_dir: Directory to save output files
            
        Returns:
            List of (audio_array, sample_rate) tuples
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        for i, prompt in enumerate(prompts):
            output_file = output_path / f"track_{i+1:03d}.wav"
            audio, sr = self.generate_music(
                prompt=prompt,
                duration=duration,
                output_path=str(output_file)
            )
            results.append((audio, sr))
        
        return results
    
    def _save_audio(self, audio_array: np.ndarray, output_path: str):
        """Save audio array to WAV file."""
        # Normalize audio to prevent clipping
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(output_path, self.sample_rate, audio_int16)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }


# Example usage
if __name__ == "__main__":
    # Initialize agent with small model (faster, less memory)
    agent = MusicGenerationAgent(model_size="small")
    
    # Print model info
    info = agent.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example prompts
    prompts = [
        "upbeat electronic dance music with heavy bass",
        "calm acoustic guitar with soft percussion",
        "epic orchestral soundtrack with strings and brass",
        "lo-fi hip hop beats for studying",
        "80s synthwave with retro drums"
    ]
    
    # Generate single track
    print("\n" + "="*60)
    print("Generating single track...")
    audio, sr = agent.generate_music(
        prompt=prompts[0],
        duration=10.0,
        output_path="output_music.wav"
    )
    print(f"Generated audio shape: {audio.shape}")
    print(f"Sample rate: {sr} Hz")
    
    # Generate multiple tracks
    print("\n" + "="*60)
    print("Generating multiple tracks...")
    results = agent.generate_batch(
        prompts=prompts[:3],
        duration=8.0,
        output_dir="music_output"
    )
    print(f"Generated {len(results)} tracks")
