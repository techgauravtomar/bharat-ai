
import os

# MPS compatibility fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model_state_dict


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT = Path.home() / "Desktop/bharat-ai"

MODEL = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    / "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)

MANIFEST = (
    PROJECT / "dataset/metadata/temple_training_manifest.jsonl"
)

OUTPUT = PROJECT / "storage/checkpoints/mac_lora_smoke_v01"

DEVICE = torch.device("mps")
RESOLUTION = 128
LEARNING_RATE = 1e-4


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def main():

    print("\n=== BHARAT AI LORA SMOKE TEST ===")

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available.")

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not rows:
        raise RuntimeError("Training manifest is empty.")

    # One image only for technical smoke test.
    record = next(
        (
            row for row in rows
            if Path(row["image_path"]).is_file()
            and row.get("caption", "").strip()
        ),
        None
    )

    if record is None:
        raise RuntimeError("No usable image and caption found.")

    image_path = Path(record["image_path"])

    if not image_path.is_absolute():
        image_path = PROJECT / image_path

    caption = record["caption"]

    print("Image:", image_path.name)
    print("Device:", DEVICE)
    print("Resolution:", RESOLUTION)


    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    print("\nLoading training image...")

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (RESOLUTION, RESOLUTION),
        Image.Resampling.LANCZOS
    )

    import numpy as np

    pixels = np.asarray(image).copy()

    pixel_values = (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .float()
        / 127.5
        - 1.0
    )

    pixel_values = pixel_values.unsqueeze(0)


    # --------------------------------------------------
    # LOAD FROZEN COMPONENTS ON CPU
    # --------------------------------------------------

    print("\nLoading VAE on CPU...")

    vae = AutoencoderKL.from_pretrained(
        str(MODEL),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    vae.eval()
    vae.requires_grad_(False)

    print("Loading tokenizer...")

    tokenizer = CLIPTokenizer.from_pretrained(
        str(MODEL),
        subfolder="tokenizer",
        local_files_only=True
    )

    print("Loading text encoder on CPU...")

    text_encoder = CLIPTextModel.from_pretrained(
        str(MODEL),
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    text_encoder.eval()
    text_encoder.requires_grad_(False)


    # --------------------------------------------------
    # ENCODE IMAGE AND CAPTION ON CPU
    # --------------------------------------------------

    print("\nEncoding image and caption...")

    with torch.no_grad():

        latents = vae.encode(pixel_values).latent_dist.sample()

        latents = latents * vae.config.scaling_factor

        tokens = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoder_hidden_states = text_encoder(
            tokens.input_ids
        )[0]

    del vae, text_encoder, pixel_values, tokens
    gc.collect()


    # --------------------------------------------------
    # LOAD UNET
    # --------------------------------------------------

    print("\nLoading UNet on MPS in float16...")

    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL),
        subfolder="unet",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0"
        ]
    )

    unet.add_adapter(lora_config)

    unet.to(DEVICE)

    trainable_params = [
        parameter
        for parameter in unet.parameters()
        if parameter.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found.")

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable_params)
    )


    # --------------------------------------------------
    # PREPARE TRAINING TENSORS
    # --------------------------------------------------

    latents = latents.to(
        device=DEVICE,
        dtype=torch.float16
    )

    encoder_hidden_states = encoder_hidden_states.to(
        device=DEVICE,
        dtype=torch.float16
    )

    noise = torch.randn_like(latents)

    scheduler = DDPMScheduler.from_pretrained(
        str(MODEL),
        subfolder="scheduler",
        local_files_only=True
    )

    alphas = scheduler.alphas_cumprod.to(
        device=DEVICE,
        dtype=torch.float32
    )

    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=DEVICE
    ).long()

    alpha = alphas[timesteps].view(-1, 1, 1, 1)

    noisy_latents = (
        alpha.sqrt() * latents.float()
        + (1 - alpha).sqrt() * noise.float()
    ).to(torch.float16)

    if scheduler.config.prediction_type == "epsilon":
        target = noise.float()

    elif scheduler.config.prediction_type == "v_prediction":
        target = (
            alpha.sqrt() * noise.float()
            - (1 - alpha).sqrt() * latents.float()
        )

    else:
        raise ValueError(
            "Unsupported prediction type: "
            + str(scheduler.config.prediction_type)
        )


    # --------------------------------------------------
    # ONE REAL TRAINING STEP
    # --------------------------------------------------

    print("\nStarting one LoRA training step...")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE
    )

    unet.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states
    ).sample

    loss = F.mse_loss(
        prediction.float(),
        target.float()
    )

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    print("Optimizer step: SUCCESS")


    # --------------------------------------------------
    # SAVE ONLY LORA ADAPTER
    # --------------------------------------------------

    print("\nSaving LoRA adapter...")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    adapter_state = get_peft_model_state_dict(unet)

    adapter_state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter_state.items()
    }

    save_file(
        adapter_state,
        str(OUTPUT / "adapter_model.safetensors")
    )

    with open(OUTPUT / "training_info.json", "w") as f:
        json.dump(
            {
                "project": "Bharat AI",
                "experiment": "Indian Architecture LoRA smoke test",
                "base_model": "Stable Diffusion 1.5",
                "resolution": RESOLUTION,
                "training_steps": 1,
                "training_images": 1,
                "loss": loss.item(),
                "device": "mps"
            },
            f,
            indent=2
        )

    print("\n=== TRAINING SMOKE TEST PASSED ===")
    print("Checkpoint:", OUTPUT)
    print("Loss:", loss.item())

    del unet, optimizer, prediction
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
