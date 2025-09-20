#!/usr/bin/env python3
"""
Test script for trained FLUX LoRA
Generates images with different prompts using the trained LoRA


python test_lora.py --lora-path weights/3dgen.safetensors --prompt "3dgen, intricate gear assembly on metal plate" --guidance-scale 3.5
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "3dgen, ceramic model of monkey" --guidance-scale 3.5
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "3dgen, herringbone tweed pants feature a classic diagonal" --guidance-scale 3.5
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "3dgen, herringbone tweed pants feature a classic, diagonal twill pattern with intricate texture and subtle earthy undertones that evoke the rustic charm of traditional British countryside, front view, accurate, complete, white background" --guidance-scale 3.5
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "3dgen, herringbone tweed pants feature a classic, diagonal twill pattern with intricate texture and subtle earthy undertones that evoke the rustic charm of traditional British countryside, front view, accurate, complete, white background" --guidance-scale 3.5 --no-lora
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "herringbone tweed pants feature a classic, diagonal twill pattern with intricate texture and subtle earthy undertones that evoke the rustic charm of traditional British countryside, front view, accurate, complete, white background" --guidance-scale 3.5 --no-lora
python test_lora.py --lora-path weights/3dgen.safetensors --prompt "herringbone tweed pants feature a classic diagonal" --guidance-scale 3.5 --no-lora
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))

def test_lora_inference(lora_path, base_model_path, prompt, output_dir, num_images=1, steps=20, guidance_scale=1.0, seed=42, no_lora=False):
    """
    Test LoRA inference using the flux minimal inference script
    """
    if no_lora:
        print(f"Testing base model (no LoRA): {base_model_path}")
    else:
        print(f"Testing LoRA: {lora_path}")
    print(f"Prompt: {prompt}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model directory from base model path
    model_dir = os.path.dirname(os.path.dirname(base_model_path))
    clip_l_path = os.path.join(model_dir, "clip", "clip_l.safetensors")
    t5xxl_path = os.path.join(model_dir, "clip", "t5xxl_fp16.safetensors")
    ae_path = os.path.join(model_dir, "vae", "ae.sft")
    
    # Prepare arguments for flux_minimal_inference
    args = [
        'python', 'sd-scripts/flux_minimal_inference.py',
        '--ckpt_path', base_model_path,
        '--clip_l', clip_l_path,
        '--t5xxl', t5xxl_path,
        '--ae', ae_path,
        '--prompt', prompt,
        '--output_dir', output_dir,
        '--steps', str(steps),
        '--guidance', str(guidance_scale),
        '--seed', str(seed),
        '--width', '1024',
        '--height', '1024',
        '--dtype', 'bfloat16'
    ]
    
    # Add LoRA weights only if not using --no-lora
    if not no_lora and lora_path:
        args.extend(['--lora_weights', lora_path])
    
    print(f"Running: {' '.join(args)}")
    
    try:
        import subprocess
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        print(f"✅ Successfully generated images in {output_dir}")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during inference: {e}")
        print("Error output:", e.stderr)
        return False

def test_with_gen_img(lora_path, base_model_path, prompt, output_dir, num_images=1, steps=20, guidance_scale=1.0, seed=42, no_lora=False):
    """
    Test LoRA using the gen_img.py script
    """
    if no_lora:
        print(f"Testing base model (no LoRA) with gen_img.py: {base_model_path}")
    else:
        print(f"Testing LoRA with gen_img.py: {lora_path}")
    print(f"Prompt: {prompt}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for gen_img.py
    args = [
        'python', 'sd-scripts/gen_img.py',
        '--ckpt', base_model_path,
        '--prompt', prompt,
        '--outdir', output_dir,
        '--images_per_prompt', str(num_images),
        '--steps', str(steps),
        '--scale', str(guidance_scale),
        '--seed', str(seed),
        '--H', '1024',
        '--W', '1024',
        '--bf16'
    ]
    
    # Add LoRA weights only if not using --no-lora
    if not no_lora and lora_path:
        args.extend(['--network_weights', lora_path, '--network_module', 'networks.lora_flux'])
    
    print(f"Running: {' '.join(args)}")
    
    try:
        import subprocess
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        print(f"✅ Successfully generated images in {output_dir}")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during inference: {e}")
        print("Error output:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Test trained FLUX LoRA with different prompts")
    parser.add_argument("--lora-path", help="Path to trained LoRA file (not required with --no-lora)")
    parser.add_argument("--base-model", default="models/flux-dev/transformer/diffusion_pytorch_model.safetensors", help="Base model path")
    parser.add_argument("--prompt", required=True, help="Prompt for image generation")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for generated images")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--method", choices=["flux_minimal", "gen_img"], default="flux_minimal", help="Inference method")
    parser.add_argument("--no-lora", action="store_true", help="Test base model without LoRA")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.no_lora and not args.lora_path:
        print("❌ Error: --lora-path is required unless using --no-lora")
        return
    
    if not args.no_lora and not os.path.exists(args.lora_path):
        print(f"❌ LoRA file not found: {args.lora_path}")
        return
    
    # Check if base model exists
    if not os.path.exists(args.base_model):
        print(f"❌ Base model not found: {args.base_model}")
        return
    
    print("="*60)
    if args.no_lora:
        print("FLUX BASE MODEL TESTING")
    else:
        print("FLUX LoRA TESTING")
    print("="*60)
    if not args.no_lora:
        print(f"LoRA: {args.lora_path}")
    print(f"Base Model: {args.base_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print("="*60)
    
    # Run inference
    if args.method == "flux_minimal":
        success = test_lora_inference(
            args.lora_path, args.base_model, args.prompt, 
            args.output_dir, args.num_images, args.steps, 
            args.guidance_scale, args.seed, args.no_lora
        )
    else:
        success = test_with_gen_img(
            args.lora_path, args.base_model, args.prompt, 
            args.output_dir, args.num_images, args.steps, 
            args.guidance_scale, args.seed, args.no_lora
        )
    
    if success:
        print(f"\n🎉 Test completed! Check {args.output_dir} for generated images.")
    else:
        print("\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()
