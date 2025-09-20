#!/usr/bin/env python3
"""
Client script to test the 3D Generation LoRA Server
"""

import requests
import argparse
import os
from pathlib import Path

def test_image_generation(server_url, prompt, output_dir="test_outputs"):
    """Test 2D image generation endpoint"""
    print(f"Testing image generation with prompt: '{prompt}'")
    
    data = {
        "prompt": prompt,
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "seed": 42,
        "width": 1024,
        "height": 1024
    }
    
    try:
        response = requests.post(f"{server_url}/generate_image/", data=data)
        
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_image_{data['seed']}.png")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Image saved to: {output_path}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_3d_video_generation(server_url, prompt, output_dir="test_outputs"):
    """Test 3D video generation endpoint"""
    print(f"Testing 3D video generation with prompt: '{prompt}'")
    
    data = {
        "prompt": prompt,
        "video_res": 1088,
        "guidance_scale": 3.5,
        "seed": 42,
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 12,
        "slat_guidance_strength": 3.0,
        "slat_sampling_steps": 12
    }
    
    try:
        response = requests.post(f"{server_url}/generate_3d_video/", data=data)
        
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_video_{data['seed']}.mp4")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Video saved to: {output_path}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health(server_url):
    """Test server health"""
    print("Testing server health...")
    
    try:
        response = requests.get(f"{server_url}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server is healthy!")
            print(f"   CUDA available: {health_data.get('cuda_available', False)}")
            print(f"   FLUX loaded: {health_data.get('flux_loaded', False)}")
            print(f"   Trellis loaded: {health_data.get('trellis_loaded', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test 3D Generation LoRA Server")
    parser.add_argument("--server-url", default="http://127.0.0.1:10006", help="Server URL")
    parser.add_argument("--prompt", default="3dgen, a cute cat sitting on a chair", help="Prompt for generation")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory")
    parser.add_argument("--test-type", choices=["health", "image", "video", "all"], default="all", help="Type of test to run")
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D Generation LoRA Server Test Client")
    print("="*60)
    print(f"Server URL: {args.server_url}")
    print(f"Prompt: {args.prompt}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)
    
    success = True
    
    if args.test_type in ["health", "all"]:
        success &= test_health(args.server_url)
        print()
    
    if args.test_type in ["image", "all"]:
        success &= test_image_generation(args.server_url, args.prompt, args.output_dir)
        print()
    
    if args.test_type in ["video", "all"]:
        success &= test_3d_video_generation(args.server_url, args.prompt, args.output_dir)
        print()
    
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
