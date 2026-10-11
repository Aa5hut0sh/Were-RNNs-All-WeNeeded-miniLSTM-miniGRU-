
import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gradio as gr

from minrnn_core import MinGRU, MinLSTM, build_model

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# The 65 characters of TinyShakespeare in sorted order. Used only if assets/vocab.json is missing.
DEFAULT_CHARS = "\n !$&',-.3:;?" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz"

PAPER_HARDWARE = ("P100, T4 and Quadro 5000 GPUs with 16 GB (efficiency benchmark on a T4); "
                  "Mamba and xLSTM baselines were run on A100 80 GB GPUs")

_CACHE = {}


# ---------------------------------------------------------------- small helpers
def load_json(name, default=None):
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results():
    return load_json("results.json", {})


def load_checkpoint(model, filename, what):
    """Load weights strictly. A mismatch raises a visible error instead of silently using random weights."""
    path = os.path.join(ASSETS, filename)
    if not os.path.exists(path):
        raise gr.Error(f"Missing {filename} in app/assets/. Export it from the notebook first.")
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    except RuntimeError as e:
        raise gr.Error(f"{what}: checkpoint does not match the model definition. {str(e)[:400]}")
    model.eval()
    return model


# ---------------------------------------------------------------- tab 1: paper vs ours
def load_system_meta():
    ours = load_results().get("system_meta", {}).get("reproduction", {})
    return (
        "### Setup comparison\n\n"
        "| Specification | Original paper (Feng et al., 2024) | Our reproduction |\n"
        "|---|---|---|\n"
        f"| Hardware | {PAPER_HARDWARE} | {ours.get('hardware', 'Colab GPU')} |\n"
        f"| GRU and LSTM baselines | plain PyTorch loops | {ours.get('baseline', 'plain PyTorch loops and cuDNN')} |\n"
        f"| Selective Copy | L=4096, 16 data tokens, 400k steps | {ours.get('copy_scale', 'see notebook')} |\n"
        f"| Language model | 3 layers, d=384, 5k steps, batch 64 | {ours.get('lm_scale', 'see notebook')} |\n"
    )


def load_comparison_table():
    rows = load_results().get("table", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["No results.json found in app/assets/"]})


# ---------------------------------------------------------------- tab 2: benchmark
def plot_benchmarks():
    df = pd.DataFrame(load_results().get("bench", []))
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in df["model"].unique():
        sub = df[df["model"] == m].dropna(subset=["ms"])
        if sub.empty:
            continue
        axes[0].plot(sub["L"], sub["ms"], marker="o", label=m)
        if "mem_gb" in sub and sub["mem_gb"].notna().any():
            axes[2].plot(sub["L"], sub["mem_gb"], marker="o", label=m)
    axes[0].set(xlabel="Sequence length", ylabel="Step time (ms)", title="Training step time (log scale)")
    axes[0].set_yscale("log")
    for slow, fast in [("plain_gru", "mingru"), ("plain_lstm", "minlstm")]:
        s = df[df["model"] == slow].set_index("L")["ms"]
        f = df[df["model"] == fast].set_index("L")["ms"]
        sp = (s / f).dropna()
        if not sp.empty:
            axes[1].plot(sp.index, sp.values, marker="s", label=f"{slow} / {fast}")
    axes[1].set(xlabel="Sequence length", ylabel="Speedup (x)", title="Speedup over the loop-based RNN")
    axes[2].set(xlabel="Sequence length", ylabel="Peak memory (GB)", title="Peak GPU memory")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- tab 3: LM curves
def plot_lm_curves():
    curves = load_json("lm_curves.json")
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for name, hist in curves.items():
        ax.plot([p["step"] for p in hist], [p["val_loss"] for p in hist], label=name)
    ax.axhline(1.548, color="r", linestyle=":", alpha=0.7, label="paper minGRU best (1.548)")
    ax.set(xlabel="Step", ylabel="Validation loss", title="TinyShakespeare validation loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- tab 4: Selective Copy demo + gates
def copy_cfg():
    cfg = load_json("copy_demo_config.json", {}) or {}
    return dict(L=cfg.get("L", 64), n_data=cfg.get("n_data", 8), layers=cfg.get("layers", 3),
                d=cfg.get("d", 64), V=cfg.get("V", 16))


def get_copy_model():
    if "copy" not in _CACHE:
        c = copy_cfg()
        model = build_model("mingru", c["V"], c["d"], c["layers"], task="copy", max_len=c["L"])
        _CACHE["copy"] = load_checkpoint(model, "mingru_copy_demo.pt", "Selective Copy demo")
    return _CACHE["copy"]


def make_copy_batch(B, L, n, V, gen):
    """Same construction as the notebook: noise = 0, marker = 1, data tokens = 2..V-1."""
    x = torch.zeros(B, L, dtype=torch.long)
    pos = torch.rand(B, L - n, generator=gen).argsort(1)[:, :n].sort(1).values
    vals = torch.randint(2, V, (B, n), generator=gen)
    x.scatter_(1, pos, vals)
    x[:, L - n:] = 1
    return x, pos, vals


def run_selective_copy(seed):
    c = copy_cfg()
    model = get_copy_model()
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        x, _, vals = make_copy_batch(1, c["L"], c["n_data"], c["V"], gen)
        pred = model(x)[:, c["L"] - c["n_data"]:].argmax(-1)
        xb, _, vb = make_copy_batch(200, c["L"], c["n_data"], c["V"], torch.Generator().manual_seed(12345))
        pb = model(xb)[:, c["L"] - c["n_data"]:].argmax(-1)
    ok = (pred[0] == vals[0])
    marks = "  ".join("✓" if v else "✗" for v in ok.tolist())
    tok_acc = (pb == vb).float().mean().item() * 100
    exact = (pb == vb).all(1).float().mean().item() * 100
    return (
        f"Demo model: minGRU, {c['layers']} layers, trained on length {c['L']} with {c['n_data']} data tokens\n\n"
        f"Target:     {vals[0].tolist()}\n"
        f"Predicted:  {pred[0].tolist()}\n"
        f"Match:      {marks}\n"
        f"This example: {int(ok.sum())}/{c['n_data']} tokens correct\n\n"
        f"Over 200 fixed random sequences: token accuracy {tok_acc:.1f}%, exact match {exact:.1f}%"
    )


def plot_gate_heatmap(seed, layer):
    c = copy_cfg()
    model = get_copy_model()
    L, n = c["L"], c["n_data"]
    layer_idx = int(min(max(int(layer), 1), c["layers"])) - 1
    gen = torch.Generator().manual_seed(int(seed))
    x_ids, pos, _ = make_copy_batch(1, L, n, c["V"], gen)

    with torch.no_grad():
        x = model.emb(x_ids)
        gates = None
        for i, blk in enumerate(model.blocks):
            if i == layer_idx:
                _, gates = blk(x, return_gates=True)
                break
            x = blk(x)
    z = gates["z"][0].cpu().numpy()                      # (L, hidden)

    types = np.zeros(L)                                  # 0 noise, 1 output marker, 2 data
    types[pos[0].numpy()] = 2
    types[L - n:] = 1
    zmean = z.mean(1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.6, 3, 1.6]})
    ax0, ax1, ax2 = axes
    ax0.imshow(types[None, :], aspect="auto", cmap=ListedColormap(["#f1efe8", "#b4b2a9", "#185fa5"]), vmin=0, vmax=2)
    ax0.set_yticks([])
    ax0.set_title("Input tokens: white = noise, grey = output marker, blue = data")
    im = ax1.imshow(z.T, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    ax1.set(ylabel="Hidden unit", title=f"minGRU update gate z, layer {layer_idx + 1}")
    ax2.plot(np.arange(L), zmean, color="#534ab7")
    ax2.scatter(pos[0].numpy(), zmean[pos[0].numpy()], color="#185fa5", s=22, zorder=3, label="data token")
    ax2.set(xlabel="Time step", ylabel="Mean z")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=list(axes), pad=0.01, fraction=0.04, label="gate value z")
    return fig


# ---------------------------------------------------------------- tab 5: equivalence
def run_equivalence(model_name, T, d, use_log):
    torch.manual_seed(42)
    cls = MinGRU if model_name == "minGRU" else MinLSTM
    m = cls(int(d), int(d), use_log=bool(use_log)).double()
    x = torch.randn(1, int(T), int(d), dtype=torch.float64)
    with torch.no_grad():
        h_seq = m(x, parallel=False)
        h_par = m(x, parallel=True)
    diff = (h_seq - h_par).abs().max().item()
    status = "PASS" if diff < 1e-8 else "FAIL"
    return (f"Model: {model_name} | T = {int(T)} | d = {int(d)} | log-space variant: {bool(use_log)}\n"
            f"Max absolute difference between sequential and parallel output: {diff:.2e}\n"
            f"Status: {status} (float64, tolerance 1e-8)")


# ---------------------------------------------------------------- tab 6: text generation
def get_lm():
    if "lm" not in _CACHE:
        vocab = load_json("vocab.json", None)
        chars = "".join(vocab) if isinstance(vocab, (list, str)) else DEFAULT_CHARS
        cfg = load_json("lm_config.json", {}) or {}
        d, layers = cfg.get("d", 384), cfg.get("layers", 3)
        model = build_model("mingru", len(chars), d, layers, task="lm", max_len=512)
        _CACHE["lm"] = (load_checkpoint(model, "mingru_lm_demo.pt", "Shakespeare model"), chars)
    return _CACHE["lm"]


def generate_text(prompt, length, temperature):
    model, chars = get_lm()
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = [stoi[ch] for ch in prompt if ch in stoi] or [stoi.get("\n", 0)]
    cur = torch.tensor([ids], dtype=torch.long)
    out = [chars[i] for i in ids]
    temp = max(float(temperature), 0.1)
    with torch.no_grad():
        for _ in range(int(length)):
            logits = model(cur[:, -256:])[:, -1, :] / temp
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            out.append(chars[nxt.item()])
            cur = torch.cat([cur, nxt], dim=1)
    return "".join(out)


# ---------------------------------------------------------------- interface
with gr.Blocks(title="minGRU and minLSTM explorer") as demo:
    gr.Markdown("# Were RNNs All We Needed? minGRU and minLSTM")
    gr.Markdown("Explorer for a small-scale reproduction of Feng et al. (2024). "
                "All plots read results exported from the notebook.")

    with gr.Tab("1. Paper vs ours"):
        gr.Markdown(load_system_meta())
        gr.Dataframe(value=load_comparison_table())

    with gr.Tab("2. Speed and memory"):
        gr.Markdown("Time and memory for one training step (forward, loss, backward) against sequence length. "
                    "Our model width differs from the paper's (the paper does not state its width), "
                    "so compare the shapes of the curves rather than absolute numbers.")
        gr.Plot(value=plot_benchmarks())

    with gr.Tab("3. Training curves"):
        gr.Markdown("Validation loss on TinyShakespeare. The dotted line is the paper's best minGRU loss.")
        gr.Plot(value=plot_lm_curves())

    with gr.Tab("4. Selective Copy and gates"):
        cc = copy_cfg()
        gr.Markdown(f"Demo model: minGRU, {cc['layers']} layers, length {cc['L']}, {cc['n_data']} data tokens. "
                    "Layer 1's gate sees only the current token, so it is constant for each token type; "
                    "later layers also see history. Compare the gate on noise tokens with the gate on data tokens.")
        with gr.Row():
            seed_in = gr.Number(value=0, precision=0, label="Seed (changes the example)")
            layer_in = gr.Slider(1, cc["layers"], value=min(2, cc["layers"]), step=1, label="Layer")
        gh_btn = gr.Button("Show gate heatmap", variant="primary")
        gh_plot = gr.Plot()
        gh_btn.click(plot_gate_heatmap, [seed_in, layer_in], gh_plot)
        sc_btn = gr.Button("Run Selective Copy example")
        sc_out = gr.Textbox(label="Result", lines=9)
        sc_btn.click(run_selective_copy, [seed_in], sc_out)

    with gr.Tab("5. Equivalence"):
        gr.Markdown("Runs the same weights in sequential mode and in parallel-scan mode and compares the outputs "
                    "(float64). They agree to rounding error, about 1e-14, not bit for bit.")
        with gr.Row():
            model_opt = gr.Radio(["minGRU", "minLSTM"], value="minGRU", label="Model")
            log_opt = gr.Checkbox(value=True, label="Log-space variant")
        with gr.Row():
            t_slide = gr.Slider(8, 512, value=64, step=8, label="Sequence length T")
            d_slide = gr.Slider(4, 128, value=16, step=4, label="Hidden size d")
        eq_btn = gr.Button("Compare outputs")
        eq_out = gr.Textbox(label="Result", lines=4)
        eq_btn.click(run_equivalence, [model_opt, t_slide, d_slide, log_opt], eq_out)

    with gr.Tab("6. Text generation"):
        gr.Markdown("Character-level minGRU trained on TinyShakespeare. Expect Shakespeare-like words and layout, "
                    "not coherent sentences.")
        prompt_in = gr.Textbox(value="ROMEO:\n", label="Prompt", lines=2)
        with gr.Row():
            gen_len = gr.Slider(20, 400, value=200, step=10, label="Characters to generate")
            temp_slide = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
        gen_btn = gr.Button("Generate")
        gen_out = gr.Textbox(label="Output", lines=12)
        gen_btn.click(generate_text, [prompt_in, gen_len, temp_slide], gen_out)

if __name__ == "__main__":
    demo.launch()      # in Colab use demo.launch(share=True)