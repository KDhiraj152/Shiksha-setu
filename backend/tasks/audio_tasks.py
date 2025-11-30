"""Celery tasks for audio transcription using Whisper."""
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

from celery import shared_task
import torch

from ..core.config import settings
from ..utils.logging import get_logger
from ..utils.device_manager import get_device_manager

logger = get_logger(__name__)


@shared_task(bind=True, name='audio.transcribe')
def transcribe_audio_task(
    self,
    audio_path: str,
    language: str = "hi",
    task_type: str = "transcribe",
    user_id: str = None
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper model.
    
    Args:
        audio_path: Path to audio file
        language: Audio language code (hi, en, ta)
        task_type: 'transcribe' or 'translate' (to English)
        user_id: User identifier
        
    Returns:
        Transcription result with metadata
    """
    try:
        logger.info(f"Starting audio transcription: {audio_path}")
        
        # Import here to avoid loading model on module import
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa
        
        # Get device
        device_manager = get_device_manager()
        device = device_manager.get_device()
        
        # Load Whisper model
        model_name = "openai/whisper-small"  # ~244M parameters
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # Process audio in chunks (30 seconds each)
        chunk_duration = 30
        chunk_samples = chunk_duration * sr
        transcriptions = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            
            # Prepare inputs
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            inputs = inputs.to(device)
            
            # Set language and task
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language,
                task=task_type
            )
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs.input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=448
                )
            
            # Decode
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            transcriptions.append(transcription)
            
            # Update progress
            progress = min(int((i + chunk_samples) / len(audio) * 100), 100)
            self.update_state(
                state='PROGRESS',
                meta={'progress': progress, 'message': f'Transcribing... {progress}%'}
            )
        
        # Combine transcriptions
        full_transcription = " ".join(transcriptions)
        
        # Calculate statistics
        word_count = len(full_transcription.split())
        
        result = {
            "transcription": full_transcription,
            "language": language,
            "duration": duration,
            "word_count": word_count,
            "confidence": 0.85,  # Placeholder - Whisper doesn't provide confidence
            "model": model_name,
            "device": str(device),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(
            f"Transcription completed: {word_count} words, {duration:.2f}s audio",
            extra={"user_id": user_id, "audio_path": audio_path}
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise
