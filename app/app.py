
import os
import json

import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from minrnn_core import MinGRU, MinLSTM, build_model

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


DEFAULT_CHARS = "\n !$&',-.3:;?" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz"

PAPER_HARDWARE = ("P100, T4 and Quadro 5000 GPUs with 16 GB (efficiency benchmark on a T4); "
                  "Mamba and xLSTM baselines were run on A100 80 GB GPUs")

_CACHE = {}


# small helpers
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


def image_path(name):
    p = os.path.join(ASSETS, name)
    return p if os.path.exists(p) else None


# tab 1: paper vs ours
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


# tab 2: benchmark
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


#  tab 3: LM curves
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


#  tab 4: equivalence walkthrough
MAX_ROWS = 16     # time steps shown in the tables (the error uses all of them)
MAX_COLS = 6      # input features shown


def f4(v):
    return f"{v:.4f}"


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scan_rounds(a, b):
    """Doubling scan for one hidden unit, recording (A, B) after every round.
    a, b: 1-D tensors of length T. Pair rule: (a1,b1) o (a2,b2) = (a1*a2, a2*b1 + b2)."""
    A, B = a.clone(), b.clone()
    snaps = [(A.clone(), B.clone(), 0)]
    d = 1
    while d < len(a):
        A2, B2 = A.clone(), B.clone()
        A2[d:] = A[:-d] * A[d:]
        B2[d:] = A[d:] * B[:-d] + B[d:]
        A, B = A2, B2
        snaps.append((A.clone(), B.clone(), d))
        d *= 2
    return snaps


def run_equivalence(model_name, T, d, use_log, unit, seed):
    T, d, seed = int(T), int(d), int(seed)
    unit = min(max(int(unit), 1), d) - 1
    is_lstm = model_name == "minLSTM"
    torch.manual_seed(seed)
    m = (MinLSTM if is_lstm else MinGRU)(d, d, use_log=bool(use_log)).double()
    x = torch.randn(1, T, d, dtype=torch.float64)
    with torch.no_grad():
        h_seq, gates = m(x, parallel=False, return_gates=True)      # expected: the loop
        h_par = m(x, parallel=True)                                   # got: the scan
    diff_all = (h_seq - h_par).abs().max().item()
    status = "PASS" if diff_all < 1e-8 else "FAIL"

    if is_lstm:
        a, b = gates["f"][0, :, unit], gates["i"][0, :, unit] * gates["cand"][0, :, unit]
    else:
        a, b = 1.0 - gates["z"][0, :, unit], gates["z"][0, :, unit] * gates["cand"][0, :, unit]
    snaps = scan_rounds(a, b)
    rounds = len(snaps) - 1
    hs, hp = h_seq[0, :, unit], h_par[0, :, unit]
    R = min(T, MAX_ROWS)

    summary = (f"### Result: {status}\n"
               f"- Model **{model_name}**, T = {T}, d = {d}, log-space variant: **{bool(use_log)}**, seed {seed}\n"
               f"- Max absolute difference between sequential and parallel over all {T} x {d} values: **{diff_all:.2e}**\n"
               f"- Sequential needs **{T} steps in a row**; the scan needs **{rounds} rounds** (ceil(log2 {T}))\n"
               f"- Differences near 1e-15 are floating-point rounding (float64); tolerance 1e-8")

    cols = min(d, MAX_COLS)
    inp = ("**1. Input** (random T x d matrix, first rows and columns)\n\n"
           + md_table(["t"] + [f"x{k + 1}" for k in range(cols)],
                      [[t + 1] + [f4(x[0, t, k].item()) for k in range(cols)] for t in range(R)]))

    if is_lstm:
        head = ["t", "f' (forget)", "i' (input)", "candidate", "a_t = f'", "b_t = i' * cand"]
        rows = [[t + 1, f4(gates["f"][0, t, unit].item()), f4(gates["i"][0, t, unit].item()),
                 f4(gates["cand"][0, t, unit].item()), f4(a[t].item()), f4(b[t].item())] for t in range(R)]
    else:
        head = ["t", "z (update gate)", "candidate", "a_t = 1 - z", "b_t = z * cand"]
        rows = [[t + 1, f4(gates["z"][0, t, unit].item()), f4(gates["cand"][0, t, unit].item()),
                 f4(a[t].item()), f4(b[t].item())] for t in range(R)]
    gt = (f"**2. What the layer computes from the input** (hidden unit {unit + 1}). "
          "No column uses the previous hidden state, so all rows are computed at once.\n\n" + md_table(head, rows))

    prev = 0.0
    rows = []
    for t in range(R):
        rows.append([t + 1, f4(prev), f"{f4(a[t].item())} x {f4(prev)} + {f4(b[t].item())}", f"**{f4(hs[t].item())}**"])
        prev = hs[t].item()
    sq = ("**3. Sequential way (the expected output).** One step after another: h_t = a_t * h_(t-1) + b_t, with h_0 = 0\n\n"
          + md_table(["t", "h_(t-1)", "a_t * h_(t-1) + b_t", "h_t"], rows))

    head = ["t", "start"] + [f"round {k} (join t-{snaps[k][2]})" for k in range(1, rounds + 1)]
    rows = [[t + 1] + [f"B {f4(Bk[t].item())} / A {f4(Ak[t].item())}" for Ak, Bk, _ in snaps] for t in range(R)]
    sc = ("**4. Parallel way (scan).** Each cell is a pair (A, B). In every round each row merges with the row d steps "
          "above it, all rows at once. After the last round, B is h_t.\n\n" + md_table(head, rows))
    if use_log:
        sc += ("\n\nIn log-space mode the real parallel code works with logarithms. The table above runs the same scan "
               "on the plain a_t and b_t (the same function). The 'got (parallel)' column below comes from the real log-space scan.")

    final_B = snaps[-1][1]
    head = ["t", "expected (sequential)", "got (parallel)", "got (rounds above)", "error"]
    rows = [[t + 1, f4(hs[t].item()), f4(hp[t].item()), f4(final_B[t].item()), f"{abs(hs[t].item() - hp[t].item()):.1e}"]
            for t in range(R)]
    cmp_ = "**5. Expected vs got, and the error**\n\n" + md_table(head, rows)
    if T > R:
        cmp_ += f"\n\nTables show the first {R} of {T} time steps; the summary error covers all of them."
    return summary, inp, gt, sq, sc, cmp_


# tab 6: text generation
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
        gr.Image(value=image_path("benchmark.png"), label="Paper's result (Feng et al., 2024, efficiency figure)",
                 interactive=False)
        gr.Plot(value=plot_benchmarks(), label="Our result (live plot from results.json)")

    with gr.Tab("3. Training curves"):
        gr.Markdown("Validation loss on TinyShakespeare. First the paper's figure, then ours. "
                    "The dotted line in our plot is the paper's best minGRU loss.")
        gr.Image(value=image_path("Shakespeare.png"), label="Paper's result (Feng et al., 2024, Shakespeare training curves)",
                 interactive=False)
        gr.Plot(value=plot_lm_curves(), label="Our result (live plot from lm_curves.json)")

    with gr.Tab("4. Equivalence"):
        gr.Markdown("Pick a model, sequence length and hidden size. The same weights and the same input are run "
                    "sequentially (expected) and with the parallel scan (got), and the two are compared step by step.")
        with gr.Row():
            model_opt = gr.Radio(["minGRU", "minLSTM"], value="minGRU", label="Model")
            log_opt = gr.Checkbox(value=False, label="Log-space variant")
        with gr.Row():
            t_in = gr.Slider(2, 512, value=8, step=1, label="Sequence length T")
            d_in = gr.Slider(1, 64, value=3, step=1, label="Hidden size d")
        with gr.Row():
            unit_in = gr.Slider(1, 64, value=1, step=1, label="Hidden unit to inspect")
            seed_in = gr.Number(value=0, precision=0, label="Seed")
        eq_btn = gr.Button("Run comparison", variant="primary")
        eq_sum = gr.Markdown()
        eq_inp = gr.Markdown()
        eq_gate = gr.Markdown()
        eq_seq = gr.Markdown()
        eq_scan = gr.Markdown()
        eq_cmp = gr.Markdown()
        eq_btn.click(run_equivalence, [model_opt, t_in, d_in, log_opt, unit_in, seed_in],
                     [eq_sum, eq_inp, eq_gate, eq_seq, eq_scan, eq_cmp])

    with gr.Tab("5. Text generation"):
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
    demo.launch()      