---
title: minGRU and minLSTM Explorer
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# minGRU & minLSTM Explorer

An interactive demonstration of the paper **"Were RNNs All We Needed?"** (Feng et al., 2024, [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)).

### Features
1. **Mathematical Equivalence**: Verify the bit-level equivalence between sequential recurrent stepping and $O(\log T)$ parallel prefix scan.
2. **Speed & Memory Benchmarks**: Explore empirical training runtime and memory scaling up to 4096 sequence length.
3. **Training Loss Curves**: Inspect validation loss curves on TinyShakespeare comparing minGRU, minLSTM, and Transformer baselines.
4. **Selective Copy Playground**: Test token memorization and selective extraction in synthetic sequences.
5. **Gate Activation Heatmap**: Visualize input-dependent gate dynamics separating signal tokens from noise.
6. **Autoregressive Text Generation**: Interactive character-level generation demonstrating recurrent inference efficiency.
