# QCE-Net: Multi-Modal Action Quality Assessment

## Description

QCE-Net is a deep learning model for **action quality assessment (AQA)**. Given a video and its extracted visual, audio, and optical-flow features, the model predicts a continuous quality score. The implementation also supports inference when one or more modalities are unavailable.

## Dataset Information

The code provides loaders for the following datasets:

- `FS1000`: sample-level `.npy` feature files for visual, audio, and flow modalities, together with train/validation label files. https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating
- `FisV`: dataset-level `.npy` dictionaries containing RGB, audio, and flow features, together with score annotations. https://github.com/chmxu/MS_LSTM
- `RG`: dataset-level `.npy` dictionaries containing RGB, audio, and flow features, together with score annotations. https://github.com/qinghuannn/ACTION-NET

## Code Information

- `main.py`: main entry point for training and evaluation.
- `options.py`: command-line configuration and model settings.
- `datasets.py`: dataset loading, score normalization, temporal cropping, and padding.
- `train.py`: training loop and training correlation calculation.
- `test.py`: evaluation loop, MSE calculation, and Spearman correlation.
- `models/model.py`: QCE-Net architecture.
- `models/loss.py`: regression, hard triplet, and reconstruction losses.
- `models/transformer.py`: transformer encoder and decoder modules.
- `models/triplet_loss.py`: hard triplet loss implementation.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

The default configuration uses the `FS1000` loader:

```bash
python main.py --dataset FS1000
```

Example with explicit paths:

```bash
python main.py --dataset FS1000 ^
  --video-path ./dataset/FS1000/output_feature_fs1000_new ^
  --audio-path ./dataset/FS1000/ast_feature_fs1000_new ^
  --flow-path ./dataset/FS1000/i3d_avg_clip8_5s_fs1000 ^
  --train-label-path ./dataset/FS1000/train_fs1000_new.txt ^
  --test-label-path ./dataset/FS1000/val_fs1000_new.txt
```

On macOS/Linux, replace the `^` line-continuation characters with `\`, or place the command on one line.

### Evaluation

```bash
python main.py --test --ckpt ./ckpt/action_net_best.pkl
```

Evaluation reports the loss and Spearman rank correlation for the full `V+A+F` setting and for supported missing-modality combinations.

The main configurable arguments include `--dataset`, feature paths, annotation paths, `--batch`, `--epoch`, `--lr`, `--device`, `--num-workers`, and `--ckpt`.

## Requirements

- Python 3.10+
- Torch 2.4.1+
- NumPy
- SciPy
- tensorboardX

The exact package list is provided in `requirements.txt`. A CUDA-capable PyTorch installation is recommended for practical training, while CPU execution is supported for testing and small-scale verification.

## Methodology

RGB, optical-flow, and audio streams are encoded by modality-specific backbones and a shared temporal encoder. QCC calibrates pretrained features into a quality-oriented space, EFC reconstructs missing modalities through energy-guided iterative refinement, and the consensus-guided quality reasoning module predicts the final AQA score.

## License and Citation

This code is provided for academic and research use. If you use QCE-Net or this implementation in your work, please cite the associated paper.
