# Standard imports
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

# Third party imports
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu

# LLaVA imports 
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GeneratorConfig:
    """Configuration for the video QA generator"""
    llava_model: str = "lmms-lab/llava-onevision-qwen2-0.5b-ov"  # llava_model: str = "lmms-lab/llava-onevision-qwen2-7b-ov"
    max_frames: int = 8  # Increased for better coverage
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.0
    
@dataclass 
class EvaluatorConfig:
    """Configuration for the critic"""
    llava_model: str = "lmms-lab/llava-critic-7b"
    max_frames: int = 8  # Match with generator
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.0

def extract_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Extract frames from video"""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = np.linspace(0, len(vr)-1, num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        raise

class VideoQASystem:
    """Video question answering using LLaVA-OneVision"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        logger.info(f"Loading QA model: {config.llava_model}")
        
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            config.llava_model,
            None,
            model_name="llava_qwen",
            device_map=config.device,
        )
        self.model.eval()
        
    def process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Process raw frames for model input"""
        processed = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]
        return processed.to(device=self.config.device, dtype=self.config.dtype)

    def generate_answer(self, frames: np.ndarray, question: str) -> str:
        """Generate answer for video"""
        try:
            # Process frames
            processed_frames = self.process_frames(frames)
            
            # Prepare conversation
            conv = conv_templates["qwen_1_5"].copy()
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            
            # Tokenize
            input_ids = tokenizer_image_token(
                conv.get_prompt(),
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).to(self.config.device)
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    images=[processed_frames],
                    modalities=["video"],
                    do_sample=False,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens
                )
                
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

class VideoLLaVACritic:
    """Evaluation using LLaVA-Critic"""
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        logger.info(f"Loading critic model: {config.llava_model}")
        
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            config.llava_model,
            None,
            model_name="llava_qwen",
            device_map=config.device,
        )
        self.model.eval()
        
    def process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Process raw frames for model input"""
        processed = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]
        return processed.to(device=self.config.device, dtype=self.config.dtype)

    def evaluate(self, frames: np.ndarray, question: str, answer: str, caption: str) -> Dict:
        """Evaluate model response"""
        try:
            # Process frames
            processed_frames = self.process_frames(frames)
            
            # Construct evaluation prompt
            prompt = f"""
            
            Given a video sequence and a question-answer pair, please evaluate the answer's quality.

            Video context: A sequence of frames showing: {caption}

            Question: {question}
            Model's Answer: {answer}

            Rate this answer on a scale of 0-100 considering:
            1. Accuracy in describing what's shown in the video
            2. Completeness in capturing important details
            3. Coherence with the provided caption and context

            Provide your evaluation as:
            Score: <number>
            Explanation: <detailed justification>
            """
            # Prepare conversation
            conv = conv_templates["qwen_1_5"].copy()
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            
            # Tokenize
            input_ids = tokenizer_image_token(
                conv.get_prompt(),
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).to(self.config.device)
            
            # Generate evaluation
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids,
                    images=[processed_frames],
                    modalities=["video"],
                    do_sample=False,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse result
            try:
                score_line = next(line for line in result.split('\n') if line.startswith('Score:'))
                score = int(score_line.split(':')[1].strip())
                explanation = '\n'.join(
                    line.strip() for line in result.split('\n') 
                    if not line.startswith('Score:') and line.strip()
                )
                explanation = explanation.replace("Explanation: ", "")   
            except Exception as e:
                logger.warning(f"Failed to parse evaluation result: {e}")
                return {
                    "score": 0,
                    "explanation": f"Error parsing evaluation: {result}"
                }
            
            return {
                "score": score,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {
                "score": 0,
                "explanation": f"Evaluation failed: {str(e)}"
            }

def evaluate_videos(
    video_dir: str,
    caption_file: str,
    save_path: Optional[str] = None,
    num_videos: int = 5,
    questions: Optional[List[str]] = None,
) -> Dict:
    """Run evaluation on multiple videos"""
    
    if questions is None:
        questions = [
            "What is the main activity shown in this video?",
            "Describe the sequence of events that occur.",
            "What are the key objects and people visible?"
        ]

    # Initialize models
    try:
        generator = VideoQASystem(GeneratorConfig())
        critic = VideoLLaVACritic(EvaluatorConfig())
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
        
    # Load captions
    try:
        with open(caption_file) as f:
            captions = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load caption file: {e}")
        raise
        
    # Get video files
    video_files = sorted([
        f for f in os.listdir(video_dir) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ])[:num_videos]
    
    if not video_files:
        raise ValueError(f"No video files found in {video_dir}")
    
    results = {}
    
    # Process videos
    for video_file in tqdm(video_files, desc="Evaluating videos"):
        video_id = video_file.split(".")[0]
        video_path = os.path.join(video_dir, video_file)
        logger.info(f"Processing video: {video_id}")
        
        try:
            # Extract frames
            frames = extract_frames(video_path, generator.config.max_frames)
            caption = " ".join(captions[video_id]["text"])
            
            video_results = []
            
            # Process each question
            for question in questions:
                try:
                    # Generate answer
                    answer = generator.generate_answer(frames, question)
                    
                    # Evaluate answer
                    evaluation = critic.evaluate(frames, question, answer, caption)
                    
                    video_results.append({
                        "question": question,
                        "answer": answer,
                        "score": evaluation["score"],
                        "explanation": evaluation["explanation"]
                    })
                except Exception as e:
                    logger.error(f"Error processing question '{question}' for video {video_id}: {e}")
                    continue
            
            if video_results:
                results[video_id] = video_results
                
                # Save progress
                if save_path:
                    try:
                        with open(save_path, "w") as f:
                            json.dump(results, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save results: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            continue
            
    return results

if __name__ == "__main__":
    try:
        results = evaluate_videos(
            video_dir="video/",
            caption_file="video_caption_sliced.json",
            save_path="video_evaluation_results_1000.json",
            num_videos=1000,
            questions=None,
        )
        
        # Print results
        for video_id, video_results in results.items():
            print(f"\nVideo: {video_id}")
            print("="* 80)
            
            for result in video_results:
                print(f"\nQuestion: {result['question']}")
                print(f"Answer: {result['answer']}")
                print(f"Score: {result['score']}/100")
                print(f"Explanation: {result['explanation']}")
                print("-" * 40)
                
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise