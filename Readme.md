# Dizi-DDSP: Chinese Dizi Neural Synthesizer with Physical Constraints

Dizi-DDSP is a high-fidelity neural audio synthesizer tailored for the Chinese Dizi (bamboo flute). Built upon the Differentiable Digital Signal Processing (DDSP) framework, this project combines deep learning architectures with physical acoustic constraints to model both the rich harmonic structures and dynamic breath airflows characteristic of traditional flute performances.

## 🚀 Key Features

- **Automated Score Generation**: Rule-based MIDI engine restricting sequences to the physical pitch boundaries and expressive velocities of a G-major Qudi.
- **High-Precision Synthesis Engine**: Separate additive harmonic synthesizer (60 harmonics) and filtered noise generator (65 bands) with energy compensation.
- **Robust Feature Extraction**: Accurate F0 tracking via PyIN and normalized RMS loudness extraction optimized for CPU stability.
- **Multi-Scale Spectral Loss**: Advanced loss function operating across multiple STFT window sizes for reliable timbre reconstruction.

## 📁 Repository Structure

* `model.py` - Core DDSP neural network architecture incorporating GRU and LayerNorm blocks.
* `train.py` - SOTA CPU-optimized training pipeline with Multi-Scale Spectral Loss.
* `dataset.py` - PyTorch custom dataset loader featuring dynamic range loudness scaling.
* `generate_scores.py` - Procedural MIDI generation factory for synthetic score creation.
* `batch_render.py` - Automated high-fidelity audio rendering using FluidSynth and SoundFont.
* `extract_features.py` - DSP pipeline for extracting pitch (F0) and envelope tracking.
* `prepare_data_split.py` - Reproducible deterministic dataset partitioning engine (8:1:1 split).
* `generate_dizi.py` - Production-ready inference script for high-quality audio generation.
* `probe_sf2.py` - SoundFont preset and bank inspection utility.

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/darkalpaca00-jpg/Dizi-DDSP-Case-Study.git](https://github.com/darkalpaca00-jpg/Dizi-DDSP-Case-Study.git)
cd Dizi-DDSP-Case-Study
pip install -r requirements.txt

2. Pipeline Execution
Run the data generation and feature extraction pipeline sequentially:

Bash
python generate_scores.py
python batch_render.py
python extract_features.py
python prepare_data_split.py

3. Training & Inference
Train the DDSP synthesizer and generate high-fidelity flute phrases:

Bash
python train.py
python generate_dizi.py
