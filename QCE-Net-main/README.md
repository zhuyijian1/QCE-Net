# QCE-Net

Official implementation for the QCE-Net action quality assessment model.

## Contents

The repository contains the model, dataset loaders, training entry point, and
evaluation entry point. Feature files and annotations are not included.
Please obtain the corresponding public datasets and prepare the extracted
features in the format expected by `datasets.py`.

## Install

```bash
pip install -r requirements.txt
```

## Run

Training:

```bash
python main.py --dataset FS1000 \
  --video-path ./dataset/FS1000/output_feature_fs1000_new \
  --audio-path ./dataset/FS1000/ast_feature_fs1000_new \
  --flow-path ./dataset/FS1000/i3d_avg_clip8_5s_fs1000 \
  --train-label-path ./dataset/FS1000/train_fs1000_new.txt \
  --test-label-path ./dataset/FS1000/val_fs1000_new.txt
```

Evaluation:

```bash
python main.py --test --ckpt ./ckpt/action_net_best.pkl
```

The default configuration targets FS1000. Other supported dataset loaders are
`FisV` and `RG`; their feature and annotation paths can be supplied through
the command-line arguments.

## Notes

This release excludes datasets, checkpoints, logs, and generated cache files.
The implementation is provided for research use. Please cite the associated
paper when using this code.
