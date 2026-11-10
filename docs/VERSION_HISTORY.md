# Bharat AI — Version History

## v0.1 — Architecture LoRA
Status: Development experiment

- Base model: Stable Diffusion 1.5
- Method: LoRA fine-tuning
- Training candidates: 32 temple images
- Training steps: 32
- Resolution: 128x128
- LoRA rank: 4
- LoRA alpha: 4
- Learning rate: 0.00001
- Hardware: MacBook Pro M1 / MPS
- Checkpoint: storage/checkpoints/bharat_architecture_lora_v01
- Adapter successfully loaded into backend.
- Image generation tested.
- Visual improvement over base model not yet proven.
- Public release: Not approved.
- Dataset license review: Pending.

## v0.2 — All Indian Architecture
Status: Planned — not yet trained

Target dataset: 240 images

| Category | Target |
|---|---:|
| Temples | 60 |
| Forts & Palaces | 50 |
| Stepwells | 25 |
| Rock-cut Caves | 25 |
| Havelis & Houses | 40 |
| Historic Buildings | 40 |
| Total | 240 |

Planned:
- Diverse Indian architecture dataset
- English and Hindi captions
- Dataset quality checks
- Improved LoRA training
- Base model comparison
- Frontend model version page

## Future roadmap

| Version | Goal | Status |
|---|---|---|
| v0.3 | Improved Bharat AI | Future |
| v0.4 | Advanced generation features | Future |
| v1.0 | Public platform release | Future |
| v2.0 | Foundation model from scratch | Long-term vision |

Future versions are proposals, not completed releases.
