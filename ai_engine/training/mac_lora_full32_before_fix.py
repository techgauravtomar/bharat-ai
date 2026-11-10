import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import save_file

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model_state_dict


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT = Path.home() / "Desktop/bharat-ai"

MODEL = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    / "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)

MANIFEST = PROJECT / "dataset/metadata/temple_training_manifest.jsonl"

OUTPUT = PROJECT / "storage/checkpoints/bharat_architecture_lora_v01"

DEVICE = torch.device("mps")
RESOLUTION = 128
EPOCHS = 1
LEARNING_RATE = 1e-4
SAVE_EVERY = 8
SEED = 42

torch.manual_seed(SEED)


def save_checkpoint(unet, step, loss_value):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    state = get_peft_model_state_dict(unet)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in state.items()
    }

    save_file(
        state,
        str(OUTPUT / "adapter_model.safetensors")
    )

    info = {
        "project": "Bharat AI",
        "model": "Stable Diffusion 1.5",
        "experiment": "Indian Architecture LoRA v0.1",
        "training_images": 32,
        "resolution": RESOLUTION,
        "epochs": EPOCHS,
        "completed_steps": step,
        "last_loss": loss_value,
        "device": "mps"
    }

    (OUTPUT / "training_info.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8"
    )

    print(f"Checkpoint saved at step {step}", flush=True)


def main():
    print("\n=== BHARAT AI FULL 32 IMAGE TRAINING ===", flush=True)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU unavailable.")

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != 32:
        raise RuntimeError(
            f"Expected 32 records, found {len(rows)}. "
            "Training stopped."
        )

    records = []

    for row in rows:
        path = Path(row["image_path"])

        if not path.is_absolute():
            path = PROJECT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        caption = row.get("caption", "").strip()

        if not caption:
            raise ValueError(f"Missing caption: {path.name}")

        records.append((path, caption))

    print("Validated images:", len(records), flush=True)

    # --------------------------------------------------
    # LOAD CPU ENCODERS
    # --------------------------------------------------

    print("\nLoading VAE on CPU...", flush=True)

    vae = AutoencoderKL.from_pretrained(
        str(MODEL),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    vae.eval()
    vae.requires_grad_(False)

    print("Loading tokenizer and text encoder...", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(
        str(MODEL),
        subfolder="tokenizer",
        local_files_only=True
    )

    text_encoder = CLIPTextModel.from_pretrained(
        str(MODEL),
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # --------------------------------------------------
    # PRECOMPUTE LATENTS AND TEXT EMBEDDINGS ON CPU
    # --------------------------------------------------

    cached_data = []

    print("\nEncoding all 32 images on CPU...", flush=True)

    with torch.no_grad():
        for index, (path, caption) in enumerate(records, start=1):

            image = Image.open(path).convert("RGB")
            image = ImageOps.fit(
                image,
                (RESOLUTION, RESOLUTION),
                method=Image.Resampling.LANCZOS
            )

            pixels = np.asarray(image).copy()

            pixel_values = (
                torch.from_numpy(pixels)
                .permute(2, 0, 1)
                .float()
                / 127.5
                - 1.0
            ).unsqueeze(0)

            latent = vae.encode(
                pixel_values
            ).latent_dist.sample()

            latent = latent * vae.config.scaling_factor

            tokens = tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            embedding = text_encoder(
                tokens.input_ids
            )[0]

            cached_data.append((
                latent.cpu(),
                embedding.cpu()
            ))

            print(
                f"Encoded {index}/32: {path.name}",
                flush=True
            )

            del image, pixels, pixel_values, latent
            del tokens, embedding

    del vae, text_encoder, tokenizer
    gc.collect()

    print("CPU encoders released.", flush=True)

    # --------------------------------------------------
    # LOAD UNET ON MPS
    # --------------------------------------------------

    print("\nLoading UNet on MPS...", flush=True)

    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL),
        subfolder="unet",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    unet.requires_grad_(False)

    config = LoraConfig(
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

    unet.add_adapter(config)
    unet.to(DEVICE)

    trainable = [
        p for p in unet.parameters()
        if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable),
        flush=True
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE
    )

    scheduler = DDPMScheduler.from_pretrained(
        str(MODEL),
        subfolder="scheduler",
        local_files_only=True
    )

    alphas = scheduler.alphas_cumprod.to(
        device=DEVICE,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # TRAIN ALL 32 IMAGES
    # --------------------------------------------------

    total_steps = EPOCHS * len(cached_data)
    completed = 0
    losses = []

    OUTPUT.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(EPOCHS):
        unet.train()

        order = torch.randperm(len(cached_data)).tolist()

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}",
            flush=True
        )

        for position, data_index in enumerate(order, start=1):

            latents, embeddings = cached_data[data_index]

            latents = latents.to(
                device=DEVICE,
                dtype=torch.float16
            )

            embeddings = embeddings.to(
                device=DEVICE,
                dtype=torch.float16
            )

            noise = torch.randn_like(latents)

            timestep = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (1,),
                device=DEVICE
            ).long()

            alpha = alphas[timestep].view(1, 1, 1, 1)

            noisy = (
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

            optimizer.zero_grad(set_to_none=True)

            prediction = unet(
                noisy,
                timestep,
                embeddings
            ).sample

            loss = F.mse_loss(
                prediction.float(),
                target.float()
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss. Training stopped.")

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            completed += 1

            print(
                f"Step {completed}/{total_steps} "
                f"| Image {position}/32 "
                f"| Loss {loss_value:.6f}",
                flush=True
            )

            del latents, embeddings, noise
            del noisy, prediction, loss, target

            if completed % SAVE_EVERY == 0:
                save_checkpoint(
                    unet,
                    completed,
                    losses[-1]
                )
                gc.collect()
                torch.mps.empty_cache()

    # --------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------

    save_checkpoint(
        unet,
        completed,
        losses[-1]
    )

    elapsed = time.time() - start_time

    print("\n=== BHARAT AI TRAINING COMPLETE ===", flush=True)
    print("Images trained:", len(records), flush=True)
    print("Optimizer steps:", completed, flush=True)
    print("Average loss:", sum(losses) / len(losses), flush=True)
    print("Duration (minutes):", round(elapsed / 60, 2), flush=True)
    print("Checkpoint:", OUTPUT, flush=True)

    del unet, optimizer, cached_data
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
