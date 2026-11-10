
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import save_file

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model_state_dict


PROJECT = Path.home() / "Desktop/bharat-ai"
MODEL = (
    Path.home() / ".cache/huggingface/hub"
    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    / "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
MANIFEST = PROJECT / "dataset/metadata/temple_training_manifest.jsonl"

OUTPUT = PROJECT / "storage/checkpoints/bharat_architecture_lora_v01"
DEVICE = torch.device("mps")

RESOLUTION = 128
LEARNING_RATE = 1e-5
MAX_STEPS = 32
SAVE_EVERY = 2
SEED = 42


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
        "experiment": "Indian Architecture LoRA v0.1",
        "base_model": "Stable Diffusion 1.5",
        "resolution": RESOLUTION,
        "completed_steps": step,
        "last_loss": loss_value,
        "device": "mps",
        "precision": "float32",
        "learning_rate": LEARNING_RATE
    }

    (OUTPUT / "training_info.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8"
    )

    print("Checkpoint saved:", step, flush=True)


def main():
    torch.manual_seed(SEED)

    print("=== BHARAT AI 2-STEP STABILITY TEST ===", flush=True)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 records, found {len(rows)}")

    # Load frozen encoders on CPU.
    print("Loading VAE and text encoder...", flush=True)

    vae = AutoencoderKL.from_pretrained(
        str(MODEL),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    vae.eval()
    vae.requires_grad_(False)

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

    cached = []

    print("Encoding images...", flush=True)

    with torch.no_grad():
        for i, row in enumerate(rows, start=1):
            path = Path(row["image_path"])
            if not path.is_absolute():
                path = PROJECT / path

            if not path.is_file():
                raise FileNotFoundError(path)

            image = Image.open(path).convert("RGB")
            image = ImageOps.fit(
                image,
                (RESOLUTION, RESOLUTION),
                method=Image.Resampling.LANCZOS
            )

            pixels = np.asarray(image).copy()
            tensor = (
                torch.from_numpy(pixels)
                .permute(2, 0, 1)
                .float() / 127.5 - 1.0
            ).unsqueeze(0)

            latent = vae.encode(tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

            tokens = tokenizer(
                row["caption"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            embedding = text_encoder(tokens.input_ids)[0]

            cached.append((latent.cpu(), embedding.cpu()))

            print(f"Encoded {i}/32", flush=True)

            del image, pixels, tensor, latent, tokens, embedding

    del vae, text_encoder, tokenizer
    gc.collect()

    print("Loading UNet on MPS float32...", flush=True)

    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL),
        subfolder="unet",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    unet.requires_grad_(False)

    config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )

    unet.add_adapter(config)
    unet.to(DEVICE)

    trainable = [p for p in unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        eps=1e-6
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

    unet.train()
    completed = 0

    for index in range(MAX_STEPS):
        latents, embeddings = cached[index]

        latents = latents.to(DEVICE, dtype=torch.float32)
        embeddings = embeddings.to(DEVICE, dtype=torch.float32)

        noise = torch.randn_like(latents)

        timestep = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (1,),
            device=DEVICE
        ).long()

        alpha = alphas[timestep].view(1, 1, 1, 1)

        noisy = (
            alpha.sqrt() * latents
            + (1 - alpha).sqrt() * noise
        )

        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latents
        else:
            raise ValueError("Unsupported prediction type")

        optimizer.zero_grad(set_to_none=True)

        prediction = unet(
            noisy,
            timestep,
            embeddings
        ).sample

        loss = F.mse_loss(prediction, target)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss before backward")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0
        )

        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Non-finite gradient detected")

        for p in trainable:
            if p.grad is not None and not torch.isfinite(p.grad).all().item():
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Invalid LoRA gradient detected")

        optimizer.step()

        for p in trainable:
            if not torch.isfinite(p).all().item():
                raise RuntimeError("Invalid LoRA parameter after optimizer step")

        completed += 1

        value = float(loss.detach().cpu())
        print(
            f"Step {completed}/{MAX_STEPS} | "
            f"Loss {value:.6f} | "
            f"Gradient norm {float(grad_norm):.6f}",
            flush=True
        )

        optimizer.zero_grad(set_to_none=True)

        del latents, embeddings, noise, noisy, prediction, loss, target
        gc.collect()
        torch.mps.empty_cache()

        if completed % SAVE_EVERY == 0:
            save_checkpoint(unet, completed, value)

    print("=== 2-STEP TEST PASSED ===", flush=True)
    print("Checkpoint:", OUTPUT, flush=True)

    del unet, optimizer, cached
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
