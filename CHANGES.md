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
| `--amp` (off by default): fp16 autocast for forward passes, `GradScaler` for the backward pass; loss computed in fp32; scaler state in the checkpoint | Compute budget. Kaggle smoke test (T4, fp32): one ChestMNIST epoch at 224 took 1,304 s, i.e. ~36 h per run and ~181 GPU-h for 5 runs, against ~120 GPU-h of Kaggle quota for the whole project. AMP uses the T4's tensor cores. Epochs, optimizer, lr, schedule and batch size are unchanged. **Deviation from the authors' fp32 code; stated in the report.** |
| `--skip_epoch_eval` (off by default): per epoch, evaluate only the val set; train and test are evaluated once at the end on the best model | The original evaluates train, val and test every epoch, but only val is used (model selection by best val AUC). Skipping the other two saves ~30% of the time and does not change training, model selection, or the final metrics. `epoch_log.csv` then has train loss (from training) and val loss/AUC/ACC per epoch; the per-epoch train/test AUC/ACC columns are `nan`. |

## Fixed

| Change | Why |
|---|---|
| Device is `cuda:0` after the script sets `CUDA_VISIBLE_DEVICES=<gpu_ids[0]>` (was `cuda:<gpu_ids[0]>`) | With `--gpu_ids 1` the original asked for `cuda:1` while only one GPU was visible, so the second GPU could not be used. Passing `--gpu_ids 0` with an outer `CUDA_VISIBLE_DEVICES=1` does not work either: the script overwrites it with `0`, and the Kaggle smoke test put both jobs on GPU 0 (one ran out of memory). |
| 3D script: `best_model.pth` now saves `best_model.state_dict()` (was `model.state_dict()`, the last epoch) | The file was named best but held the last epoch. Reported metrics were already computed from the best model, so results are unaffected. |

## Not changed (checked on torch 2.5.1 / torchvision 0.20.1)

- `resnet50(pretrained=False)` and `transforms.Resize(..., interpolation=PIL.Image.NEAREST)` still work and only emit deprecation warnings, so they were left as is.
- Install ACSConv with `pip install git+https://github.com/M3DV/ACSConv.git` (the README's `git://` URL no longer works on GitHub).
