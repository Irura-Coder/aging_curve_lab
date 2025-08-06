############################ FastAPI backend ############################
# Requires: pip install fastapi uvicorn torch pandas scikit-learn pybaseball
# Run with: uvicorn api:app --reload  (from backend/ folder)
########################################################################

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import torch, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from pybaseball import batting_stats
import os

###############################################################################
# 1. Load model + scaler (produced earlier by baseball_aging_curve_model.py)
###############################################################################

MODEL_PATH = os.getenv("MODEL_PATH", "aging_curve_transformer.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train it first via baseball_aging_curve_model.py")

# At the top of your file, after imports
print("=== DEBUGGING IMPORTS ===")
print(f"torch version: {torch.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

try:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    print("Model loaded successfully")
    scaler: StandardScaler = ckpt["scaler"]
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Mirror the architecture quickly (must match training script)
SEQ_LEN = 10
N_FEATS = 6  # k, bb, maxev, gbfb, xwoba, age


def build_model():
    import torch.nn as nn

    class AgingTransformer(nn.Module):
        def __init__(self, d_model: int = 64, nhead: int = 8, num_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(N_FEATS, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
            self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, d_model))
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor, mask: torch.Tensor):
            h = self.input_proj(x) + self.pos_embed
            h = h.transpose(0, 1)
            enc = self.encoder(h, src_key_padding_mask=~mask).transpose(0, 1)
            last_valid = mask.sum(1) - 1
            rep = enc[torch.arange(enc.size(0)), last_valid]
            mu = self.mean_head(rep).squeeze(1).detach().cpu().numpy()
            sd = (
                self.logvar_head(rep).squeeze(1).exp().sqrt().detach().cpu().numpy()
            )
            return mu, sd

    m = AgingTransformer()
    m.load_state_dict(ckpt["model"], strict=False)
    m.eval()
    return m


MODEL = build_model()

###############################################################################
# 2. Pre‑load Statcast data for 2015‑current to avoid repeated network calls
###############################################################################


def pull_statcast_cached() -> pd.DataFrame:
    cache = "players_2015_2024.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    
    frames = []
    for yr in range(2015, 2025):
        df = batting_stats(yr, qual=100)  # 100 PA cutoff
        df["season"] = yr
        frames.append(df)
    
    full = pd.concat(frames, ignore_index=True)
    full = full.rename(
        columns={
            "K%": "k_percent",
            "BB%": "bb_percent",
            "maxEV": "max_ev",
            "GB/FB": "gb_fb",
            "xwOBA": "xwoba",
        }
    )
    
    # Check what ID columns are available
    print("Available columns:", full.columns.tolist())
    
    # Prefer MLBAM ID if available, otherwise use FanGraphs
    if "playerid" in full.columns:  # MLBAM ID
        pid_col = "playerid"
        print("Using MLBAM playerid")
    elif "IDfg" in full.columns:  # FanGraphs ID
        pid_col = "IDfg"
        full = full.rename(columns={"IDfg": "playerid"})
        print("Using FanGraphs IDfg")
    else:
        # Fallback to any available ID column
        pid_col = next(c for c in ["player_id", "playerID"] if c in full.columns)
        full = full.rename(columns={pid_col: "playerid"})
        print(f"Using fallback ID: {pid_col}")
    
    full["age"] = full["Age"].astype(float)
    feat_cols = ["k_percent", "bb_percent", "max_ev", "gb_fb", "xwoba", "age"]
    full = full[["playerid", "Name", "Team", "season"] + feat_cols]
    full.to_parquet(cache)
    return full


RAW_DF = pull_statcast_cached()
# Ensure playerid is integer so == pid filters match
RAW_DF["playerid"] = RAW_DF["playerid"].astype(float)
print("Sample player-ids in RAW_DF:", RAW_DF["playerid"].unique()[:20])

# scale features once
FEAT_COLS = ["k_percent", "bb_percent", "max_ev", "gb_fb", "xwoba", "age"]
RAW_DF[FEAT_COLS] = scaler.transform(RAW_DF[FEAT_COLS])

###############################################################################
# 3. FastAPI app & schemas
###############################################################################

app = FastAPI(title="MLB Aging Curve API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerStub(BaseModel):
    id: int
    name: str
    team: str


class ForecastPoint(BaseModel):
    season: int
    mean: float
    sd: float


class ForecastResponse(BaseModel):
    curve: List[ForecastPoint]

###############################################################################
# 4. Helper for sequence → tensor and mask
###############################################################################


def pad_sequence(feats: np.ndarray):
    pad = SEQ_LEN - len(feats)
    if pad > 0:
        feats = np.vstack([np.zeros((pad, N_FEATS)), feats])
        mask = np.hstack(
            [np.zeros(pad, dtype=bool), np.ones(len(feats) - pad, dtype=bool)]
        )
    else:
        feats = feats[-SEQ_LEN:]
        mask = np.ones(SEQ_LEN, dtype=bool)
    return (
        torch.tensor(feats, dtype=torch.float32).unsqueeze(0),
        torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
    )

###############################################################################
# 5. Endpoints
###############################################################################


@app.get("/players/health")
def health():
    """Simple health check"""
    return {"status": "ok", "players_loaded": int(RAW_DF.playerid.nunique())}


@app.get("/players/search", response_model=List[PlayerStub])
async def player_search(q: str):
    q = q.strip()
    if len(q) < 2:
        return []
    mask = RAW_DF["Name"].str.contains(q, case=False, na=False)
    subset = RAW_DF.loc[mask, ["playerid", "Name", "Team"]].drop_duplicates("playerid")
    return [
        {"id": int(row.playerid), "name": row.Name, "team": row.Team}
        for row in subset.itertuples(index=False)
    ]


@app.get("/players/{pid}/forecast", response_model=ForecastResponse)
@app.post("/players/{pid}/forecast", response_model=ForecastResponse)
async def forecast(pid: int, overrides: Optional[Dict[str, Any]] = None):
    try:
        print(f"=== FORECAST DEBUG for player {pid} ===")
        
        df = RAW_DF[RAW_DF.playerid == pid].sort_values("season")
        print(f"Found {len(df)} rows for player {pid}")
        
        if df.empty:
            raise HTTPException(404, "Player not found")

        feats = df[FEAT_COLS].values
        print(f"Feature shape: {feats.shape}")
        print(f"Feature columns: {FEAT_COLS}")
        
        mask_input = ~np.isnan(feats).any(axis=1)
        feats = feats[mask_input]
        seasons = df.season.values[mask_input]
        
        print(f"After masking: {len(feats)} valid rows")
        print(f"Seasons: {seasons}")
        
        if len(feats) == 0:
            raise HTTPException(400, "No valid data for player")

        if overrides:
            last = feats[-1].copy()
            idx_map = {"k": 0, "bb": 1, "maxev": 2, "gbfb": 3}
            for k, v in overrides.get("overrides", {}).items():
                if k in idx_map:
                    last[idx_map[k]] += v
            feats[-1] = last

        x, m = pad_sequence(feats)
        print(f"Model input shape: {x.shape}")
        
        mu, sd = MODEL(x, m)
        next_season = int(seasons[-1] + 1)

        curve = [
            {
                "season": int(s),
                "mean": float(f[4] * scaler.scale_[4] + scaler.mean_[4]),
                "sd": 0.0,
            }
            for s, f in zip(seasons, feats)
        ]
        curve.append({"season": next_season, "mean": float(mu[0]), "sd": float(sd[0])})
        return {"curve": curve}
        
    except Exception as e:
        print(f"Error in forecast for player {pid}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.get("/players/{pid}/explain")
async def explain(pid: int):
    return {
        "text": "Model considers recent strike‑out trend and power metrics. Lower K% and stable Max EV keep projection above league average, but widening uncertainty reflects player’s age."
    }

@app.get("/debug/player/{name}")
async def debug_player(name: str):
    """Debug endpoint to see what IDs exist for a player"""
    mask = RAW_DF["Name"].str.contains(name, case=False, na=False)
    player_data = RAW_DF.loc[mask].head(5)
    
    result = []
    for _, row in player_data.iterrows():
        result.append({
            "name": row["Name"],
            "playerid": row["playerid"],
            "season": row["season"],
            "team": row["Team"]
        })
    return {"players": result}

@app.get("/debug/numpy")
async def debug_numpy():
    """Test if numpy is working"""
    try:
        import numpy as np
        return {
            "numpy_version": np.__version__,
            "test_array": np.array([1, 2, 3]).tolist(),
            "numpy_available": True
        }
    except Exception as e:
        return {"numpy_available": False, "error": str(e)}

@app.get("/debug/model")
async def debug_model():
    """Test if model loads correctly"""
    try:
        # Test model forward pass with dummy data
        dummy_x = torch.randn(1, 10, 6)  # batch_size=1, seq_len=10, features=6
        dummy_mask = torch.ones(1, 10, dtype=torch.bool)
        
        with torch.no_grad():
            mu, sd = MODEL(dummy_x, dummy_mask)
        
        return {
            "model_loaded": True,
            "dummy_output_mu": float(mu[0]),
            "dummy_output_sd": float(sd[0])
        }
    except Exception as e:
        return {"model_loaded": False, "error": str(e)}

###############################################################################
# If run directly: uvicorn api:app --reload
###############################################################################
