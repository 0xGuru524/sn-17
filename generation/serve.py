
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)                                          

from io import BytesIO
import imageio
from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

from diffusers import FluxPipeline
import torch
# from huggingface_hub import hf_hub_download

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

MAX_SEED = np.iinfo(np.int32).max

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
    return parser.parse_args()

args = get_args()
app = FastAPI()

def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config

@app.on_event("startup")
def startup_event() -> None:
    config = get_config()
    # initialize flux pipeline
    flux_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    # lora_path = hf_hub_download(repo_id="manbeast3b/FLUX.1-schnell-3dgen-lora", filename="3dgen.safetensors")
    flux_pipeline.load_lora_weights("./3dgen-lora/weights/3dgen.safetensors")
    apply_cache_on_pipe(flux_pipeline, residual_diff_threshold=0.08)

    # Initialize trellis pipeline
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    trellis_pipeline.cuda()

    app.state.flux_pipeline = flux_pipeline
    app.state.trellis_pipeline = trellis_pipeline


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
) -> Response:
    t0 = time()
    prompt = f"3dgen, {prompt}"
    image = app.state.flux_pipeline(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
    image.save("generated_image.jpg")
    t1 = time()
    logger.info(f" Generation took: {(t1 - t0) / 60.0} min")

    seed = get_seed(True, 1)
    outputs = app.state.trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30)

    buffer = BytesIO()
    outputs['gaussian'][0].save_ply(buffer)
    buffer.seek(0)
    buffer = buffer.getbuffer()
    t2 = time()
    logger.info(f" Saving and encoding took: {(t2 - t1) / 60.0} min")

    return Response(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
):
    t0 = time()
    prompt = f"3dgen, {prompt}"
    image = app.state.flux_pipeline(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
    image.save("generated_image.jpg")
    t1 = time()
    logger.info(f" Generation took: {(t1 - t0) / 60.0} min")

    seed = get_seed(True, 1)
    outputs = app.state.trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    t2 = time()

    buffer = BytesIO()
    imageio.mimsave(buffer, video, fps=30)
    buffer.seek(0)
    buffer = buffer.getbuffer()
    logger.info(f" It took: {(time() - t2) / 60.0} min")

    return StreamingResponse(content=buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
