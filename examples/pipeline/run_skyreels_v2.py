import os
import sys
import time
import tempfile
import random
from typing import Optional

import gradio as gr
import torch
from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2ImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

sys.path.append("..")

from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


MODEL_ID = os.environ.get(
    "SKYREELS_V2_DIR", "Skywork/SkyReels-V2-I2V-14B-540P-Diffusers"
)
HEIGHT = 1400
WIDTH = 740
DEFAULT_NUM_FRAMES = 97
DEFAULT_GUIDANCE = 5.0
DEFAULT_FPS = 24
FLOW_SHIFT = 5.0


vae = AutoencoderKLWan.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=torch.float32,
).to("cuda")

pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer", "text_encoder"],
    ),
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config,
    flow_shift=FLOW_SHIFT,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)


def _prepare_seed(seed: Optional[int]) -> int:
    if seed is None or seed < 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def generate_video(
    image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    num_frames: int,
    guidance_scale: float,
    seed: Optional[int],
    fps: int,
):
    if image is None:
        raise gr.Error("Please upload an input image to drive the video.")

    actual_seed = _prepare_seed(seed)
    generator = torch.Generator("cpu").manual_seed(actual_seed)

    start = time.time()
    result = pipe(
        image=image,
        prompt=prompt or None,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        height=(HEIGHT//16)*16,
        width=(WIDTH//16)*16,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        generator=generator,
    )
    frames = result.frames[0]
    elapsed = time.time() - start

    cache_dit.summary(pipe, details=True)

    output_dir = tempfile.mkdtemp(prefix="skyreels_v2_")
    save_path = os.path.join(
        output_dir, f"skyreels_v2.{strify(args, pipe)}.seed{actual_seed}.mp4"
    )
    export_to_video(frames, save_path, fps=fps, quality=8)

    status = (
        f"Saved to {save_path} — seed {actual_seed} — {elapsed:.2f}s"
    )
    return save_path, status


with gr.Blocks(title="SkyReels V2 I2V 540P") as demo:
    gr.Markdown("## SkyReels V2 Image-to-Video (540P)")

    with gr.Row():
        image_input = gr.Image(
            label="Input Image",
            type="pil",
            image_mode="RGB",
        )
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Describe motion, style, lighting…",
            lines=5,
        )

    negative_prompt_input = gr.Textbox(
        label="Negative Prompt",
        placeholder="Optional: things to avoid",
        lines=2,
    )

    with gr.Row():
        steps_slider = gr.Slider(
            minimum=10,
            maximum=80,
            value=50,
            step=1,
            label="Denoising Steps",
        )
        frames_slider = gr.Slider(
            minimum=16,
            maximum=121,
            value=DEFAULT_NUM_FRAMES,
            step=1,
            label="Frames",
        )
        guidance_slider = gr.Slider(
            minimum=0.0,
            maximum=12.0,
            value=DEFAULT_GUIDANCE,
            step=0.1,
            label="Guidance Scale",
        )

    with gr.Row():
        seed_input = gr.Number(
            label="Seed (-1 for random)",
            value=-1,
            precision=0,
        )
        fps_slider = gr.Slider(
            minimum=8,
            maximum=30,
            value=DEFAULT_FPS,
            step=1,
            label="FPS",
        )

    run_button = gr.Button("Generate Video")
    video_output = gr.Video(label="Generated Video")
    status_output = gr.Markdown(visible=True)

    run_button.click(
        fn=generate_video,
        inputs=[
            image_input,
            prompt_input,
            negative_prompt_input,
            steps_slider,
            frames_slider,
            guidance_slider,
            seed_input,
            fps_slider,
        ],
        outputs=[video_output, status_output],
        batch=False,
    )

demo.queue().launch()
