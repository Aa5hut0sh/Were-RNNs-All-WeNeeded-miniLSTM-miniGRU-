import torch
import torch.nn as nn
import torch.nn.functional as F

def scan_loop(a, b, h0=None):
    """Reference sequential recurrence: h_t = a_t * h_(t-1) + b_t."""
    B, T, D = a.shape
    h = []
    h_prev = h0 if h0 is not None else torch.zeros(B, D, dtype=a.dtype, device=a.device)
    for t in range(T):
        h_prev = a[:, t] * h_prev + b[:, t]
        h.append(h_prev)
    return torch.stack(h, dim=1)

def scan_tree(a, b, h0=None):
    """Doubling parallel prefix scan via associative pair rule."""
    B, T, D = a.shape
    a_cur, b_cur = a, b
    if h0 is not None:
        b_cur = torch.cat([(b[:, :1] + a[:, :1] * h0.unsqueeze(1)), b[:, 1:]], dim=1)
    offset = 1
    while offset < T:
        b_next = torch.cat([b_cur[:, :offset], a_cur[:, offset:] * b_cur[:, :-offset] + b_cur[:, offset:]], dim=1)
        a_next = torch.cat([a_cur[:, :offset], a_cur[:, offset:] * a_cur[:, :-offset]], dim=1)
        a_cur, b_cur = a_next, b_next
        offset *= 2
    return b_cur

def scan_log(log_a, log_b, log_h0=None):
    """Numerically stable log-space scan based on Heinsen (2023)."""
    if log_h0 is None:
        a_star = torch.cumsum(log_a, dim=1)
        log_h = a_star + torch.logcumsumexp(log_b - a_star, dim=1)
        return torch.exp(log_h)
    else:
        log_values = torch.cat([log_h0.unsqueeze(1), log_b], dim=1)
        a_star = F.pad(torch.cumsum(log_a, dim=1), (0, 0, 1, 0))
        log_h = a_star + torch.logcumsumexp(log_values - a_star, dim=1)
        return torch.exp(log_h)[:, 1:]

def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRU(nn.Module):
    def __init__(self, d_in, d_hidden, use_log=True):
        super().__init__()
        self.use_log = use_log
        self.linear_z = nn.Linear(d_in, d_hidden)
        self.linear_h = nn.Linear(d_in, d_hidden)

    def forward(self, x, h0=None, parallel=True, return_gates=False):
        if self.use_log:
            k = self.linear_z(x)
            cand_raw = self.linear_h(x)
            if parallel:
                log_coeffs = -F.softplus(k)
                log_z = -F.softplus(-k)
                log_b = log_z + log_g(cand_raw)
                log_h0 = h0.log() if h0 is not None else None
                h = scan_log(log_coeffs, log_b, log_h0)
            else:
                z = torch.sigmoid(k)
                h = scan_loop(1.0 - z, z * g(cand_raw), h0)
            return (h, {"z": torch.sigmoid(k), "cand": g(cand_raw)}) if return_gates else h
        else:
            z = torch.sigmoid(self.linear_z(x))
            cand = self.linear_h(x)
            a, b = 1.0 - z, z * cand
            h = scan_tree(a, b, h0) if parallel else scan_loop(a, b, h0)
            return (h, {"z": z, "cand": cand}) if return_gates else h

class MinLSTM(nn.Module):
    def __init__(self, d_in, d_hidden, use_log=True):
        super().__init__()
        self.use_log = use_log
        self.linear_f = nn.Linear(d_in, d_hidden)
        self.linear_i = nn.Linear(d_in, d_hidden)
        self.linear_h = nn.Linear(d_in, d_hidden)

    def forward(self, x, h0=None, parallel=True, return_gates=False):
        if self.use_log:
            p_f = self.linear_f(x)
            p_i = self.linear_i(x)
            cand_raw = self.linear_h(x)
            if parallel:
                diff = F.softplus(-p_f) - F.softplus(-p_i)
                log_f = -F.softplus(diff)
                log_i = -F.softplus(-diff)
                log_b = log_i + log_g(cand_raw)
                log_h0 = h0.log() if h0 is not None else None
                h = scan_log(log_f, log_b, log_h0)
            else:
                f, i = torch.sigmoid(p_f), torch.sigmoid(p_i)
                f_prime, i_prime = f / (f + i), i / (f + i)
                h = scan_loop(f_prime, i_prime * g(cand_raw), h0)
            if return_gates:
                f, i = torch.sigmoid(p_f), torch.sigmoid(p_i)
                return h, {"f": f / (f + i), "i": i / (f + i), "cand": g(cand_raw)}
            return h
        else:
            f, i = torch.sigmoid(self.linear_f(x)), torch.sigmoid(self.linear_i(x))
            f_prime, i_prime = f / (f + i), i / (f + i)
            cand = self.linear_h(x)
            h = scan_tree(f_prime, i_prime * cand, h0) if parallel else scan_loop(f_prime, i_prime * cand, h0)
            return (h, {"f": f_prime, "i": i_prime, "cand": cand}) if return_gates else h

class MLP(nn.Module):
    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class MinRNNBlock(nn.Module):
    def __init__(self, d, kind="gru", expansion=2, conv=False, mlp=False, dropout=0.0, use_log=True):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel_size=4, groups=d) if conv else None
        rnn_cls = MinGRU if kind == "gru" else MinLSTM
        self.rnn = rnn_cls(d, expansion * d, use_log=use_log)
        self.proj = nn.Linear(expansion * d, d)
        self.drop = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d) if mlp else None
        self.mlp = MLP(d, dropout) if mlp else None

    def forward(self, x, return_gates=False):
        h = self.norm(x)
        if self.conv is not None:
            h = self.conv(F.pad(h.transpose(1, 2), (3, 0))).transpose(1, 2)
        gates = None
        if return_gates:
            y, gates = self.rnn(h, return_gates=True)
        else:
            y = self.rnn(h)
        x = x + self.drop(self.proj(y))
        if self.mlp is not None:
            x = x + self.mlp(self.mlp_norm(x))
        return (x, gates) if return_gates else x

class PlainRNNBlock(nn.Module):
    def __init__(self, d, kind="gru", dropout=0.0, mlp=False):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.kind = kind
        self.cell = nn.GRUCell(d, d) if kind == "gru" else nn.LSTMCell(d, d)
        self.drop = nn.Dropout(dropout)
        self.mlp_norm, self.mlp = (nn.LayerNorm(d), MLP(d, dropout)) if mlp else (None, None)

    def forward(self, x):
        h = self.norm(x)
        B, T, D = h.shape
        states = []
        if self.kind == "gru":
            s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            for t in range(T):
                s = self.cell(h[:, t], s)
                states.append(s)
        else:
            hx = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            cx = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            for t in range(T):
                hx, cx = self.cell(h[:, t], (hx, cx))
                states.append(hx)
        y = torch.stack(states, dim=1)
        x = x + self.drop(y)
        return x + self.mlp(self.mlp_norm(x)) if self.mlp is not None else x

class CudnnBlock(nn.Module):
    def __init__(self, d, kind="gru", dropout=0.0, mlp=False):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mlp_norm, self.mlp = (nn.LayerNorm(d), MLP(d, dropout)) if mlp else (None, None)

    def forward(self, x):
        y, _ = self.rnn(self.norm(x))
        x = x + self.drop(y)
        return x + self.mlp(self.mlp_norm(x)) if self.mlp is not None else x

class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads=6, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.n1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp, self.drop = nn.LayerNorm(d), MLP(d, dropout), nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        return x + self.mlp(self.n2(x))

class SeqModel(nn.Module):
    def __init__(self, vocab, d, n_layers, make_block, max_len=None, pos_emb=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d) if pos_emb else None
        self.blocks = nn.ModuleList([make_block() for _ in range(n_layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

def build_model(name, vocab, d, n_layers, task="copy", dropout=0.0, max_len=None):
    lm = (task == "lm")
    pos = False
    if name in ("mingru", "minlstm"):
        mk = lambda: MinRNNBlock(d, kind=name[3:], expansion=2 if lm else 6, conv=lm, mlp=lm, dropout=dropout)
    elif name in ("gru", "lstm"):
        mk = lambda: CudnnBlock(d, kind=name, dropout=dropout, mlp=lm)
    elif name in ("plain_gru", "plain_lstm"):
        mk = lambda: PlainRNNBlock(d, kind=name[6:], dropout=dropout, mlp=lm)
    elif name == "transformer":
        mk = lambda: TransformerBlock(d, n_heads=6 if d % 6 == 0 else 4, dropout=dropout)
        pos = True
    else:
        raise ValueError(name)
    return SeqModel(vocab, d, n_layers, mk, max_len=max_len, pos_emb=pos)
