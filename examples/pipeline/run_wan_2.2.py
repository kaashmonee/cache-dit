import os
import sys
import tempfile
from typing import Optional

import gradio as gr
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

sys.path.append("..")

from utils import get_args, cachify


MODEL_ID = os.environ.get("WAN_2_1_DIR", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers")
HEIGHT, WIDTH = 696, 376
DEFAULT_STEPS = 50
DEFAULT_FRAMES = 81
GUIDANCE = 5.0


args = get_args()
print(args)


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline(device: str) -> WanImageToVideoPipeline:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)

    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    ).to(device)

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )

    if pipe.scheduler is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=3.0 if HEIGHT == 480 else 5.0,
        )

    if args.cache:
        if not hasattr(pipe, "transformer") or not hasattr(pipe, "transformer_2"):
            raise AttributeError("Wan pipeline is missing expected transformer modules for caching.")
        from cache_dit import (
            ForwardPattern,
            BlockAdapter,
            ParamsModifier,
            BasicCacheConfig,
        )

        cachify(
            args,
            BlockAdapter(
                pipe=pipe,
                transformer=[pipe.transformer, pipe.transformer_2],
                blocks=[pipe.transformer.blocks, pipe.transformer_2.blocks],
                forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
                params_modifiers=[
                    ParamsModifier(
                        cache_config=BasicCacheConfig(
                            max_warmup_steps=4,
                            max_cached_steps=8,
                        ),
                    ),
                    ParamsModifier(
                        cache_config=BasicCacheConfig(
                            max_warmup_steps=2,
                            max_cached_steps=20,
                        ),
                    ),
                ],
                has_separate_cfg=True,
            ),
        )

    return pipe.to(device)


DEVICE = _select_device()
PIPELINE = _load_pipeline(DEVICE)


def generate(
    image,
    prompt: str,
    steps: int,
    frames: int,
) -> str:
    if image is None:
        raise gr.Error("Please upload an image.")

    generator: Optional[torch.Generator]
    if DEVICE == "cuda":
        generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        generator = torch.Generator().manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    result = PIPELINE(
        image=image.convert("RGB"),
        prompt=prompt or None,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=steps,
        num_frames=frames,
        guidance_scale=GUIDANCE,
        generator=generator,
    )

    frames_np = result.frames[0]
    tmpdir = tempfile.mkdtemp(prefix="wan2_2_")
    video_path = os.path.join(tmpdir, "wan2_2.mp4")
    export_to_video(frames_np, video_path, fps=16)
    return video_path


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(lines=3, placeholder="Optional prompt", label="Prompt"),
        gr.Slider(10, 80, value=DEFAULT_STEPS, step=1, label="Denoising Steps"),
        gr.Slider(16, 121, value=DEFAULT_FRAMES, step=1, label="Frames"),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Wan 2.2 I2V",
    description="Minimal Gradio wrapper for Wan 2.2 image-to-video generation.",
)


if __name__ == "__main__":
    demo.launch()
