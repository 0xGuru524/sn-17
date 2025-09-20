#!/usr/bin/env python3
"""
3D Generation LoRA Server
Combines FLUX LoRA generation with Trellis 3D conversion in a FastAPI server
"""

from io import BytesIO
import os
import argparse
import torch
import numpy as np
import imageio
import random
from time import time
from typing import *

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
from omegaconf import OmegaConf
from loguru import logger
from PIL import Image

# FLUX and LoRA imports
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF

# Trellis 3D imports
from FluxTRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from FluxTRELLIS.trellis.representations import Gaussian, MeshExtractResult
from FluxTRELLIS.trellis.utils import render_utils, postprocessing_utils

NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10007)
    parser.add_argument("--config", default="configs/text_mv.yaml")
    parser.add_argument("--lora-path", default="3dgen-lora/weights/3dgen.safetensors", help="Path to LoRA weights")
    parser.add_argument("--base-model", default="camenduru/FLUX.1-dev-diffusers", help="Base FLUX model")
    return parser.parse_args()

args = get_args()
app = FastAPI()

# Global variables for models
flux_pipeline = None
trellis_pipeline = None

@app.get("/")
async def root():
    return {
        "message": "3D Generation LoRA Server is running", 
        "status": "healthy",
        "lora_path": args.lora_path,
        "base_model": args.base_model
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "cuda_available": torch.cuda.is_available(),
        "flux_loaded": flux_pipeline is not None,
        "trellis_loaded": trellis_pipeline is not None
    }

def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config

def initialize_models():
    """Initialize FLUX and Trellis pipelines"""
    global flux_pipeline, trellis_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16
    
    logger.info("Initializing FLUX pipeline with LoRA...")
    
    try:
        # Load base FLUX model
        flux_pipeline = FluxPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="balanced"
        )
        
        # Load LoRA weights if available
        if os.path.exists(args.lora_path):
            logger.info(f"Loading LoRA weights from {args.lora_path}")
            flux_pipeline.load_lora_weights(args.lora_path)
        else:
            logger.warning(f"LoRA file not found: {args.lora_path}")
        
        logger.info("Initializing Trellis 3D pipeline...")
        trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        trellis_pipeline.cuda()
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise e

@app.post("/generate_image/")
async def generate_image(
    prompt: str = Form(),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(20),
    seed: int = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
):
    """Generate 2D image using FLUX with LoRA"""
    if flux_pipeline is None:
        return {"error": "FLUX pipeline not initialized"}
    
    try:
        t0 = time()
        
        # Set seed
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate image
        image = flux_pipeline(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        
        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        generation_time = time() - t0
        logger.info(f"Image generation took: {generation_time:.2f} seconds")
        
        return Response(content=buffer.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return {"error": str(e)}

@app.post("/generate_3d_video/")
async def generate_3d_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    guidance_scale: float = Form(3.5),
    seed: int = Form(None),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(12),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12),
):
    """Generate 3D video from text prompt using FLUX LoRA + Trellis"""
    if flux_pipeline is None or trellis_pipeline is None:
        return {"error": "Models not initialized"}
    
    try:
        t0 = time()
        
        # Set seed
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate FLUX image with LoRA
        prompt_flux = "wbgmsst, " + prompt + ", 3D isometric, white background"
        logger.info(f"Generating FLUX image with prompt: {prompt_flux}")
        
        image = flux_pipeline(
            prompt=prompt_flux,
            guidance_scale=guidance_scale,
            num_inference_steps=NUM_INFERENCE_STEPS,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
        
        flux_time = time() - t0
        logger.info(f"FLUX generation took: {flux_time:.2f} seconds")
        
        # Convert image to 3D using Trellis
        logger.info("Converting image to 3D...")
        t1 = time()
        
        outputs = trellis_pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        
        trellis_time = time() - t1
        logger.info(f"Trellis 3D conversion took: {trellis_time:.2f} seconds")
        
        # Render video
        logger.info("Rendering 3D video...")
        t2 = time()
        
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        
        buffer = BytesIO()
        imageio.mimsave(buffer, video, fps=15, format='mp4')
        buffer.seek(0)
        
        render_time = time() - t2
        total_time = time() - t0
        
        torch.cuda.empty_cache()
        
        logger.info(f"Video rendering took: {render_time:.2f} seconds")
        logger.info(f"Total generation took: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return StreamingResponse(content=buffer, media_type="video/mp4")
        
    except Exception as e:
        logger.error(f"Error generating 3D video: {e}")
        return {"error": str(e)}

@app.post("/generate_trellis_video/")
async def generate_trellis_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
):
    """Legacy endpoint for compatibility"""
    return await generate_3d_video(prompt, video_res)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting 3D Generation LoRA Server...")
    initialize_models()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
