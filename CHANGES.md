# Changes from upstream MedMNIST/experiments

Goal: reproduce the MedMNIST v2 PyTorch baselines (ResNet-50 (224) for 2D, ResNet-50 + 3D for 3D) on current PyTorch, without changing the training recipe.

Applies to both `MedMNIST2D/train_and_eval_pytorch.py` and `MedMNIST3D/train_and_eval_pytorch.py`.

## Training recipe: unchanged

Adam, lr 0.001, 100 epochs, lr × 0.1 after epochs 50 and 75, model selected by best validation AUC and then evaluated on test. Batch size 128 for 2D (script default) and 32 for 3D (script default, and the value the paper states for 3D). The 2D 224 runs upscale the 28×28 files with `PIL.Image.NEAREST`, as in the original.

## Added

| Change | Why |
|---|---|
| `--seed` (default 0): seeds `random`, `numpy`, `torch` and CUDA at the start of `main` | The original sets no seed; the results CSV reports a seed per run. |
| Checkpoint after every epoch (`checkpoint.pth`: model, best model, optimizer, scheduler, best val AUC/epoch, RNG states), written atomically; on start, resume from it if present. Deleted once the run finishes. | Kaggle sessions stop at 12 hours; large datasets need more than one session. |
| Output folder is `output_root/<data_flag>/<run>` instead of `output_root/<data_flag>/<timestamp>` | A restarted run must find its own checkpoint; also stops two runs started in the same second from writing to one folder. |
| `epoch_log.csv`: per-epoch lr, train/val/test loss, AUC, ACC, and epoch time | Readable training curves for the report (the same values the original only wrote to TensorBoard). |
| `results.json`: dataset, model, run, seed, best epoch, final train/val/test AUC and ACC | Aggregation into the results CSV. |
| `--num_workers` (default 0, the original behaviour) for the DataLoaders | Speed only; shuffling still uses the main process RNG, so training is unchanged. |

## Fixed

| Change | Why |
|---|---|
| 3D script: `best_model.pth` now saves `best_model.state_dict()` (was `model.state_dict()`, the last epoch) | The file was named best but held the last epoch. Reported metrics were already computed from the best model, so results are unaffected. |

## Not changed (checked on torch 2.5.1 / torchvision 0.20.1)

- `resnet50(pretrained=False)` and `transforms.Resize(..., interpolation=PIL.Image.NEAREST)` still work and only emit deprecation warnings, so they were left as is.
- Install ACSConv with `pip install git+https://github.com/M3DV/ACSConv.git` (the README's `git://` URL no longer works on GitHub).
- The scripts set `CUDA_VISIBLE_DEVICES` from `--gpu_ids` and then use `cuda:<id>`, which fails for `--gpu_ids 1`. Not changed in code: the Kaggle runner sets `CUDA_VISIBLE_DEVICES` per job and always passes `--gpu_ids 0`.
