# baseball_aging_curve_model.py
# Requires: pip install pybaseball pandas numpy torch scikit-learn tqdm

import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pybaseball import batting_stats
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 10        # max seasons we keep per player
FEAT_NAMES = ["k_percent", "bb_percent", "max_ev", "gb_fb", "xwoba", "age"]
N_FEATS = len(FEAT_NAMES)

###############################################################################
# 1. Load & preprocess data
###############################################################################

def load_statcast_2015_2024():
    frames = []
    for year in range(2015, 2025):
        df = batting_stats(year, qual=100)         # 100 PA cutoff
        df["season"] = year
        frames.append(df)
       
    full = pd.concat(frames, ignore_index=True)
    # rename for consistency
    '''
    full = full.rename(columns={
        "K%":"k_percent", "BB%":"bb_percent",
        "maxEV":"max_ev", "GB/FB":"gb_fb",
        "xwOBA":"xwoba"
    })
    full["age"] = full["Age"].astype(float)
    return full[["playerid","season"] + FEAT_NAMES]
    '''

    full = full.rename(columns={
    "K%":"k_percent", "BB%":"bb_percent",
    "maxEV":"max_ev", "GB/FB":"gb_fb",
    "xwOBA":"xwoba"
    })

    # ------------ NEW lines ------------
    pid_col = "playerid" if "playerid" in full.columns else (
            "IDfg"     if "IDfg"     in full.columns else
            "playerID" if "playerID" in full.columns else None)

    if pid_col is None:
        raise KeyError("Couldn’t find a player-ID column in batting_stats output. "
                    "Check pybaseball version.")

    full = full.rename(columns={pid_col: "playerid"})
    # -----------------------------------

    full["age"] = full["Age"].astype(float)
    return full[["playerid", "season"] + FEAT_NAMES]
    

raw = load_statcast_2015_2024().dropna()

# z-score each feature across the entire dataset
scaler = StandardScaler()
raw[FEAT_NAMES] = scaler.fit_transform(raw[FEAT_NAMES])

###############################################################################
# 2. Build sequences  (one per player)
###############################################################################

def build_sequences(df):
    seqs, labels = [], []
    for pid, grp in df.groupby("playerid"):
        grp = grp.sort_values("season")
        feats = grp[FEAT_NAMES].values
        targets = grp["xwoba"].shift(-1).values        # next-season label
        for i in range(len(grp)-1):                    # last season has no label
            seq = feats[:i+1]                          # seasons 0..i inclusive
            seqs.append(seq)
            labels.append(targets[i])
    return seqs, labels

seqs, ys = build_sequences(raw)

###############################################################################
# 3. Torch Dataset
###############################################################################

class PlayerDataset(Dataset):
    def __init__(self, seqs, ys, max_len=SEQ_LEN):
        self.seqs, self.ys = seqs, ys
        self.max_len = max_len
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        pad = self.max_len - len(seq)
        if pad > 0:
            seq = np.vstack([np.zeros((pad,N_FEATS)), seq])
            mask = np.hstack([np.zeros(pad), np.ones(len(self.seqs[idx]))])
        else:
            seq = seq[-self.max_len:]
            mask = np.ones(self.max_len)
        return torch.tensor(seq, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.bool), \
               torch.tensor(self.ys[idx], dtype=torch.float32)

train_ds = PlayerDataset(seqs, ys)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

###############################################################################
# 4. Transformer model
###############################################################################

class AgingTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(N_FEATS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, d_model))
        self.mean_head = nn.Linear(d_model, 1)
        self.logvar_head = nn.Linear(d_model, 1)
    def forward(self, x, mask):
        """
        x: (B, L, F)   mask: (B, L)  with 1 for real, 0 for pad
        """
        h = self.input_proj(x) + self.pos_embed        # add positional embedding
        h = h.transpose(0,1)                           # L,B,d_model for transformer
        key_padding = ~mask                            # True for pads
        enc = self.encoder(h, src_key_padding_mask=key_padding).transpose(0,1)
        last_valid = mask.sum(1)-1                     # index of last real season
        rep = enc[torch.arange(enc.size(0)), last_valid]  # (B, d_model)
        mu  = self.mean_head(rep).squeeze(1)
        logvar = self.logvar_head(rep).squeeze(1).clamp(-4, 3)  # avoid extremes
        return mu, logvar

model = AgingTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

###############################################################################
# 5. Training loop  (negative log-likelihood of Gaussian)
###############################################################################

def nll_gaussian(mu, logvar, y):
    var = logvar.exp()
    return 0.5*(var.log() + (y-mu)**2 / var)

for epoch in range(10):
    model.train()
    total = 0
    for x, mask, y in tqdm(train_loader, desc=f"epoch {epoch}"):
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        mu, logvar = model(x, mask)
        loss = nll_gaussian(mu, logvar, y).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * len(y)
    print(f"epoch {epoch} | NLL {total/len(train_ds):.4f}")

torch.save({"model": model.state_dict(),
            "scaler": scaler}, "aging_curve_transformer.pt")
print("Model saved to aging_curve_transformer.pt")

def sanity(df):
    n_players = df["playerid"].nunique()
    seasons_per = df.groupby("playerid")["season"].nunique()
    print(f"Unique players: {n_players}")
    print(f"Seasons per player — min: {seasons_per.min()}, "
          f"median: {seasons_per.median()}, max: {seasons_per.max()}")
    print(f"Season range in table: {df['season'].min()}–{df['season'].max()}")

# Call it
sanity(raw)
