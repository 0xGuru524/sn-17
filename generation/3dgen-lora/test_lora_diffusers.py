#!/usr/bin/env python3
"""
Simple test script for trained FLUX LoRA using diffusers library
"""

import os
import argparse
import torch
from diffusers import FluxPipeline
from PIL import Image

def test_lora_with_diffusers(lora_path, prompt, output_dir="test_outputs", guidance_scale=3.5, seed=42, no_lora=False):
    """
    Test LoRA using diffusers library directly
    """
    if no_lora:
        print(f"Testing base model (no LoRA)")
    else:
        print(f"Testing LoRA: {lora_path}")
    print(f"Prompt: {prompt}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load base FLUX model from Hugging Face
        print("Loading FLUX.1-dev from Hugging Face...")
        pipe = FluxPipeline.from_pretrained(
            "camenduru/FLUX.1-dev-diffusers",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        
        # Load LoRA weights if not using --no-lora
        if not no_lora and lora_path:
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
    parser = argparse.ArgumentParser(description="Test trained FLUX LoRA with diffusers")
    parser.add_argument("--lora-path", help="Path to trained LoRA file (not required with --no-lora)")
    parser.add_argument("--prompt", required=True, help="Prompt for image generation")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for generated images")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-lora", action="store_true", help="Test base model without LoRA")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.no_lora and not args.lora_path:
        print("❌ Error: --lora-path is required unless using --no-lora")
        return
    
    if not args.no_lora and not os.path.exists(args.lora_path):
        print(f"❌ LoRA file not found: {args.lora_path}")
        return
    
    print("="*60)
    if args.no_lora:
        print("FLUX BASE MODEL TESTING (Diffusers)")
    else:
        print("FLUX LoRA TESTING (Diffusers)")
    print("="*60)
    if not args.no_lora:
        print(f"LoRA: {args.lora_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Run inference
    success = test_lora_with_diffusers(
        args.lora_path, args.prompt, 
        args.output_dir, args.guidance_scale, args.seed, args.no_lora
    )
    
    if success:
        print(f"\n🎉 Test completed! Check {args.output_dir} for generated images.")
    else:
        print("\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()
