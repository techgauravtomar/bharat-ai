
import torch
import threading
import uuid

from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "storage" / "generated"
LORA_DIR = ROOT / "storage" / "checkpoints" / "bharat_architecture_lora_v01"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

_pipeline = None
_generation_lock = threading.Lock()


def load_pipeline():
    global _pipeline

    if _pipeline is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available.")

        if not (LORA_DIR / "adapter_model.safetensors").exists():
            raise FileNotFoundError("Bharat AI LoRA adapter not found.")

        print("Loading Bharat AI base model...")

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        print("Attaching Bharat AI architecture LoRA...")

        pipe.unet.requires_grad_(False)

        config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"]
        )

        pipe.unet.add_adapter(config, adapter_name="bharat_architecture")

        adapter_state = load_file(
            str(LORA_DIR / "adapter_model.safetensors"),
            device="cpu"
        )

        result = set_peft_model_state_dict(
            pipe.unet,
            adapter_state,
            adapter_name="bharat_architecture"
        )

        unexpected = getattr(result, "unexpected_keys", [])
        mismatched = getattr(result, "mismatched_keys", [])

        if unexpected or mismatched:
            raise RuntimeError(
                f"LoRA loading mismatch. Unexpected: {unexpected}; "
                f"Mismatched: {mismatched}"
            )

        pipe.unet.set_adapter("bharat_architecture")

        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()

        _pipeline = pipe

        print("Bharat AI architecture LoRA loaded.")

    return _pipeline


def generate_image(prompt, negative_prompt=None):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _generation_lock:
        pipe = load_pipeline()

        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt or None,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.0
            )

        filename = f"bharat_ai_{uuid.uuid4().hex}.png"
        output_path = OUTPUT_DIR / filename

        result.images[0].save(output_path)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return filename
