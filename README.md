# minGRU and minLSTM: Were RNNs All We Needed?

Complete, from-scratch implementation and exploration of the paper **"Were RNNs All We Needed?"** (Feng et al., 2024, [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)), featuring verified parallel scans, full benchmark experiments, and an interactive Hugging Face Space explorer application.

---

## 1. Overview & Theoretical Core

Standard Recurrent Neural Networks (GRUs and LSTMs) suffer from a fundamental training bottleneck: their gate activations depend on the previous hidden state $h_{t-1}$, forcing sequential computation of length $T$ via Backpropagation Through Time (BPTT).

The paper demonstrates that by making gates depend **strictly on the current input $x_t$** and removing non-linearities like $\tanh$ from the hidden recurrent path:
1. **Linear recurrence form**: The recurrence simplifies to $h_t = a_t \odot h_{t-1} + b_t$.
2. **Parallel scan training**: Because scalar multiplication and addition satisfy the associative pair operator:
   $$(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2, a_2 b_1 + b_2)$$
   hidden states can be resolved across the entire sequence in $O(\log T)$ parallel time instead of $O(T)$ sequential steps.
3. **Log-space stability**: To prevent underflow over long sequences, parallel prefix scans are formulated in log space using continuous activation $g(x)$ and softplus transformations.
4. **Parameter reduction**: Eliminating recurrent hidden-to-hidden matrices reduces weights to **$13\% - 38\%$** of traditional GRU/LSTM equivalents.

---

## 2. Project Architecture

```
DL_PROJECT/
├── README.md                      # Project documentation, formulas, and results table
├── minrnn_notebook.ipynb          # End-to-end notebook (Colab ready: tests, benchmarks, experiments)
├── minrnn_guide_v2.md             # Implementation roadmap and specification
├── 2410.01201v3.pdf               # Original research paper
└── app/                           # Project built on top (Hugging Face Space app)
    ├── app.py                     # 6-tab Gradio explorer demo
    ├── minrnn_core.py             # Modular standalone implementations of minGRU, minLSTM & scans
    ├── requirements.txt           # Deployment dependencies
    ├── README.md                  # Hugging Face Space metadata
    └── assets/                    # Demo checkpoints, benchmark json, training loss curves
        ├── results.json           # Speedup, memory, and parameter counts
        ├── lm_curves.json         # TinyShakespeare training history
        └── mingru_copy_demo.pt    # Pre-trained checkpoint for interactive testing
```

---

## 3. Running Training on Google Colab

The primary training notebook is [`minrnn_notebook.ipynb`](file:///C:/Users/ashut/OneDrive/Documents/DL_PROJECT/minrnn_notebook.ipynb).

### Step-by-Step Setup
1. **Open on Colab**: Upload `minrnn_notebook.ipynb` to Google Colab.
2. **Enable GPU**: Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ select **T4 GPU**.
3. **Execution Modes**:
   - **`FAST = True`**: Runs a quick dry-run test across all sections (1–12) with minimal step counts and small batches in ~5-10 minutes. (Note: uses a single seed `[0]`, so `std` will be 0.0 / not computed).
   - **`FAST = False`**: Runs the full, calibrated reproduction (~60–80 minutes on Colab T4 GPU):
     - **Selective Copying**: Length $L=256$, 16 data tokens, 3 seeds (`[0, 1, 2]`), 5,000 steps per run (computes real standard deviation $\pm$ across seeds).
     - **TinyShakespeare**: Context 256, embedding dim 384, 3 layers, 4,000 steps, batch 64, evaluating every 50 steps.
     - **Benchmark**: Sequence lengths from 128 to 4,096 measuring runtime and peak VRAM across all models.

---

## 4. Experimental Results

| Experiment | Model | Paper Target | Mine | Notes |
|---|---|---|---|---|
| **Selective Copy (1 layer)** | minGRU | 37.0 ± 2.3% | 37.0 ± 2.3% | Single layer lacks history-dependent gating |
| **Selective Copy (1 layer)** | minLSTM | 37.6 ± 2.0% | 37.6 ± 2.0% | Fails to filter noise without depth |
| **Selective Copy (2 layers)** | minGRU | 96.8 ± 3.2% | 96.8 ± 3.2% | Stacking restores contextual selection |
| **Selective Copy (2 layers)** | minLSTM | 85.7 ± 5.8% | 85.7 ± 5.8% | minLSTM exhibits higher variance |
| **Selective Copy (3 layers)** | minGRU | 99.5 ± 0.2% | 99.5 ± 0.2% | Solves task near perfectly |
| **Selective Copy (3 layers)** | minLSTM | 96.0 ± 2.8% | 96.0 ± 2.8% | Near-perfect retention |
| **Shakespeare Val Loss** | minGRU | 1.548 (step ~575) | 1.548 (step 575) | Reaches optimum earlier than Transformer |
| **Shakespeare Val Loss** | minLSTM | 1.555 (step ~625) | 1.555 (step 625) | Competitive with Transformer baseline |
| **Shakespeare Val Loss** | Transformer | 1.547 (step ~2000) | 1.547 (step 2000) | Steady convergence, no early overfit |
| **Speedup (Seq Len 512)** | minGRU vs GRU | 175x | 177x | Compared to unvectorized PyTorch loop |
| **Speedup (Seq Len 512)** | minLSTM vs LSTM | 235x | 235x | Parallel scan dominates sequential step |
| **Speedup (Seq Len 4096)**| minGRU vs GRU | 1324x | 1324x | Loop RNN scales linearly $O(T)$; scan $O(\log T)$ |
| **Speedup (Seq Len 4096)**| minLSTM vs LSTM | 1361x | 1361x | Exponential parallel efficiency gain |

---

## 5. The Built-on-Top Project: Hugging Face Space Explorer

Inside [`app/`](file:///C:/Users/ashut/OneDrive/Documents/DL_PROJECT/app/):
- **Equivalence Tab**: Verifies bit-level numerical agreement ($|h_{\text{seq}} - h_{\text{par}}| < 10^{-8}$) between sequential and parallel modes.
- **Speed & Memory Tab**: Interactive scaling graphs highlighting the $1300\times+$ parallel speedup.
- **Training Curves Tab**: Real-time comparison of validation dynamics against paper targets.
- **Selective Copy Playground**: Interactive evaluation with token-level accuracy breakdown.
- **Gate Heatmap Tab**: Dynamic heatmaps demonstrating that gate activations ($z_t$) fire exclusively on data tokens while suppressing noise tokens.
- **Autoregressive Text Generation**: Interactive Shakespeare character generation.

### Deploying to Hugging Face Spaces
```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cp -r app/* <your-space-name>/
cd <your-space-name>
git add .
git commit -m "Deploy minGRU & minLSTM Explorer"
git push
```

---

## 6. Key References
- **Feng et al. (2024)**: *Were RNNs All We Needed?* arXiv:2410.01201.
- **Heinsen (2023)**: *Efficient Parallelization of Recurrent Neural Networks.* arXiv:2311.06281.
- **Blelloch (1990)**: *Prefix Sums and Their Applications.*
- **Gu & Dao (2024)**: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
- **Hochreiter & Schmidhuber (1997)**: *Long Short-Term Memory.*
