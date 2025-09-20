#!/usr/bin/env python3
"""
Simplified test script for trained FLUX LoRA using Hugging Face models directly
"""

import os
import argparse
import torch
from diffusers import FluxPipeline
from peft import PeftModel
import requests
from PIL import Image
import io

def test_lora_with_hf_models(lora_path, prompt, output_dir="test_outputs", guidance_scale=3.5, seed=42):
    """
    Test LoRA using Hugging Face models directly
    """
    print(f"Testing LoRA: {lora_path}")
    print(f"Prompt: {prompt}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load base model from Hugging Face (using open model)
        print("Loading Stable Diffusion XL from Hugging Face...")
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(lora_path)
        
        # Generate image
        print("Generating image...")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=20,  # FLUX.1-dev uses more steps
            generator=generator,
            height=1024,
            width=1024,
        ).images[0]
        
        # Save image
        output_path = os.path.join(output_dir, f"generated_{seed}.png")
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test trained FLUX LoRA with Hugging Face models")
    parser.add_argument("--lora-path", required=True, help="Path to trained LoRA file")
    parser.add_argument("--prompt", required=True, help="Prompt for image generation")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for generated images")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate LoRA file exists
    if not os.path.exists(args.lora_path):
        print(f"❌ LoRA file not found: {args.lora_path}")
        return
    
    print("="*60)
    print("FLUX LoRA TESTING (Hugging Face Models)")
    print("="*60)
    print(f"LoRA: {args.lora_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Run inference
    success = test_lora_with_hf_models(
        args.lora_path, args.prompt, 
        args.output_dir, args.guidance_scale, args.seed
    )
    
    if success:
        print(f"\n🎉 Test completed! Check {args.output_dir} for generated images.")
    else:
        print("\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()
