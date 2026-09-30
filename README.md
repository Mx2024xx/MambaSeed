# MambaSeed: A Lightweight Mamba-Based Hybrid Vision Network for Tomato Leaf Disease Recognition

MambaSeed is a lightweight hybrid vision network for crop leaf disease
recognition. It introduces a Local Continuous Branch Injection (LCBI)
mechanism to preserve fine-grained lesion edges and textures while maintaining
efficient global dependency modeling.

> **Repository status:** This repository contains a selected research snapshot:
> the model implementation, tomato training and evaluation scripts, a model
> complexity analysis script, a tomato dataset archive, and three trained
> checkpoints. Manuscript and publication details will be updated when finalized.

## Overview

Fine-grained disease cues such as lesion boundaries, texture variations, and
small pathological regions are essential for crop leaf recognition. Although
visual state-space models efficiently capture long-range dependencies, their
sequential feature propagation can gradually weaken these high-frequency local
cues, particularly in compact networks designed for edge deployment.

MambaSeed addresses this limitation through **Local Continuous Branch
Injection (LCBI)**. Instead of adding local information at only one layer, LCBI
repeatedly injects spatial responses throughout the feature hierarchy. This
keeps local structures involved as representations become deeper and more
semantic, while retaining the efficient global modeling path.

The network follows a five-stage hierarchical architecture. Its
**LCBI-MambaBlock** combines a MambaVisionMixer response with a lightweight
depthwise-separable convolution branch, allowing global context and local
texture information to be propagated together. At the final stage,
**LCBI-AttnBlock** combines multi-head self-attention with a parallel local
convolution pathway to balance high-level semantics and structural detail.

The design is intended to provide a compact balance among recognition
accuracy, computational efficiency, high-frequency information retention, and
robustness under illumination changes, motion blur, and spatial occlusion. The
manuscript evaluates this design on Tomato, Corn, and Potato leaf disease
datasets through classification, ablation, frequency-domain, efficiency, and
controlled-perturbation studies.

## Model Architecture

![Architecture of MambaSeed](assets/model_architecture.png)

**Figure 1. Architecture of MambaSeed.**

MambaSeed follows a five-stage hierarchical design. The LCBI-Mamba block
combines state-space global modeling with a depthwise-separable local
convolution branch. The final stage uses an LCBI-Attention block to integrate
global semantic information and local visual details.

## Repository Contents

- `model/mambaseed.py`: MambaSeed model implementation.
- `training/train_MambaSeed_224_tomato.py`: tomato training script.
- `testing/test_MambaSeed_224_tomato.py`: tomato evaluation script.
- `analysis/analyze_MambaSeed_224.py`: parameter-count and computational
  complexity analysis script.
- `dataset/tomato_dataset.zip`: tomato dataset archive with `train`,
  `validation`, and `test` splits.
- `model_weights/best_model_1.pth` through `best_model_3.pth`: three trained
  checkpoints.

## Running the Scripts

The scripts retain paths from the original research environment. Update the
dataset, model, output, and logging paths to match your local setup before
running them. The analysis script also refers to an external `MV1` directory.

The dataset archive is larger than GitHub's regular 100 MiB file limit. To
distribute it through GitHub, use Git LFS or attach it to a GitHub Release; see
[GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Citation

The final bibliographic record and DOI will be added after publication.

## Licenses and Third-Party Notices

The root `LICENSE` applies to the repository description and architecture
figure only. It does not grant rights to the model source code, dataset, or
trained checkpoints.

`model/mambaseed.py` is modified from NVIDIA's MambaVision project. The full
NVIDIA Source Code License-NC is included in
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/NVIDIA-MambaVision-Source-Code-License-NC.txt).
It limits use of the upstream work and derivative works to non-commercial
research or evaluation. The model code must retain its copyright and
attribution notices.

The tomato image archive is attributed and licensed separately; see
[`DATASET_NOTICE.md`](DATASET_NOTICE.md) for its PlantVillage source, citation,
and CC BY-SA 3.0 terms. The root `LICENSE` does not assign a separate blanket
license to the trained checkpoints.
