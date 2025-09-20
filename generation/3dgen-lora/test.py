import torch
from diffusers import DiffusionPipeline

model_id = 'black-forest-labs/FLUX.1-schnell'
adapter_id = 'manbeast3b/flux-lora-training'
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) # loading directly in bf16
pipeline.load_lora_weights(adapter_id)

prompt = "A happy pizza."


## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)
    
pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
image = pipeline(
    prompt=prompt,
    num_inference_steps=15,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=0.0,
).images[0]
image.save("output.png", format="PNG")