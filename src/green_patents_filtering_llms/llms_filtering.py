'''
@File    :   llms_filtering.py
@Time    :   11/2025
@Author  :   nikifori
@Version :   -
'''
import os
import re
import time
import json
import requests
import pathlib
import pandas as pd
from collections import Counter
from tqdm import tqdm  # progress bar

# --------------------- PATHS (fixed) ---------------------
# INPUT_CSV  = "/home/nikifori/Desktop/thesis/repo/data/test/ten-rows-template.csv"
INPUT_CSV  = "/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-continue.csv"
OUTPUT_CSV = "/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-llm-filtered.csv"
CSV_DELIM = ","  # explicitly comma

# --------------------- CHECKPOINTING ---------------------
CHECKPOINT_INTERVAL = 1000 # rows
def write_checkpoint(records: list[dict], path: str) -> None:
    """Write to a temp file then atomically replace the target."""
    df = pd.DataFrame.from_records(records)
    tmp = f"{path}.part"
    df.to_csv(tmp, index=False, sep=CSV_DELIM)
    os.replace(tmp, path)

# --------------------- MODELS (16GB-friendly) -----------------
MODELS = [
    "llama3.1:8b-instruct-q8_0",
    "qwen2.5:7b-instruct",
    "mistral:instruct",
]

# --------------------- OLLAMA API (/api/generate) -------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

TEMPERATURE = 0.1
TOP_P = 0.1
REPEAT_PENALTY = 1.05
NUM_CTX = 3072
NUM_PREDICT = 128
KEEP_ALIVE = -1  # keep the model loaded between requests

# --------------------- PROMPT (concise + rules) ----------------
SYSTEM_PROMPT = (
    "You are a scientific patent classifier.\n\n"
    "Task\n"
    "Decide if the patent is a “Green Patent”: a technical invention that mitigates climate change "
    "or enables clean/efficient energy systems. Return strict JSON ONLY:\n"
    "{\"label\": \"Green Patent\" | \"Not Green Patent\", \"reason\": \"one short sentence\"}\n\n"
    "Primary positive signals (classification anchors)\n"
    "1) CPC “Y” tags for climate/smart grids:\n"
    "   • Y02 — Climate change mitigation technologies (energy efficiency, low-carbon transport, renewables).\n"
    "   • Y04S — Smart grids (grid operation, smart metering, integration of EVs/renewables).\n"
    "2) IPC/CPC classes aligned with renewables & efficiency:\n"
    "   • F03D (wind), H02S (PV/solar), C10L (fuels/biofuels), H05B (efficient lighting/heating circuits),\n"
    "     F25B (heat pumps/refrigeration), B60L (EV propulsion/charging), B60K 6/20 (hybrid vehicles),\n"
    "     B09B (waste/recycling), C02F (water/wastewater treatment), B01D 53/62 (CO2 capture),\n"
    "     F01N (exhaust/emissions after-treatment), H02J (power supply/distribution, grid integration).\n"
    "3) Green keywords when backed by concrete technical means: solar/photovoltaic, wind turbine, renewable energy,\n"
    "   energy efficiency/energy-saving, efficient lighting, low-power electronics, EV/charging/BESS, grid-tied inverter,\n"
    "   smart grid/metering/demand response, PEM electrolyzer/SOEC, hydrogen storage, carbon capture/CCS, waste-to-energy,\n"
    "   wastewater treatment, PM2.5 sensors/IoT air monitoring, smart HVAC/BEMS, occupancy-aware lighting, ORC heat recovery,\n"
    "   adaptive traffic lights, AI-based route optimization, city digital twin/urban simulation.\n\n"
    "Decision policy\n"
    "• Label “Green Patent” if the invention’s primary technical contribution directly reduces emissions, improves energy efficiency,\n"
    "  enables renewable generation/storage, decarbonized transport, smart grids, waste/water treatment, or emissions control.\n"
    "• If mentions are generic buzzwords without a technical mechanism (device/process/control), or unrelated domains, label “Not Green Patent”.\n"
    "• When ambiguous, rely on concrete mechanisms, devices, processes, or control schemes that achieve the green effect.\n\n"
    "Output\n"
    "Strict JSON only with the exact schema. No extra text.\n"
)

# --------------------- Data helpers (pandas) -------------------
def load_data(path: str) -> list[dict]:
    """Read CSV via pandas and return list of dicts (records)."""
    df = pd.read_csv(path, sep=CSV_DELIM)
    return df.to_dict(orient="records")

def write_data(records: list[dict], path: str) -> None:
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False, sep=CSV_DELIM)

# --------------------- LLM helpers ------------------------------
def build_prompt(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return f"{SYSTEM_PROMPT}Title: {title}\nAbstract: {abstract}\n"

def call_ollama(session: requests.Session, model: str, prompt: str, retries: int = 3, timeout: int = 120) -> str:
    """POST /api/generate with format='json' (structured output) and keep_alive to keep model loaded."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,  # keep model resident between calls
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"Ollama call failed for model {model}: {last_err}")

def parse_json_strict_or_loose(s: str) -> dict:
    """Always return a dict with at least {label, reason}."""
    s = (s or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "label" in obj:
            return obj
    except Exception:
        pass
    if "{" in s and "}" in s:
        candidate = s[s.find("{"):s.rfind("}") + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "label" in obj:
                return obj
        except Exception:
            pass
    return {"label": "Not Green Patent", "reason": "Unparseable model output"}

# --- robust label parsing (fixes prior bug) + contradiction guard ---
GREEN_RE = re.compile(r"^\s*(green(?:\s*patent)?)\s*$", re.I)
NOTGREEN_RE = re.compile(r"^\s*(?:not\s+green(?:\s*patent)?|non[-\s]*green)\s*$", re.I)
NEG = re.compile(r"\b(no|not|without|unrelated|does\s+not|lack|lacks|absence)\b", re.I)

def normalize_label(label: str) -> str:
    t = (label or "").strip()
    if NOTGREEN_RE.match(t): return "Not Green Patent"
    if GREEN_RE.match(t):    return "Green Patent"
    tl = t.lower()
    if tl in {"not green", "not green patent", "no", "n"}: return "Not Green Patent"
    if tl in {"green", "green patent", "yes", "y"}: return "Green Patent"
    return "Not Green Patent"  # conservative default

# --------------------- Ensemble (unchanged logic) ----------------
def compute_final_label(rec: dict, models: list[str]) -> str:
    labels = []
    for model in models:
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", model)
        try:
            d = json.loads(rec.get(tag, "") or "{}")
            lbl = d.get("label")
            if lbl in ("Green Patent", "Not Green Patent"):
                labels.append(lbl)
        except Exception:
            pass

    valid_labels = [l for l in labels if l in ("Green Patent", "Not Green Patent")]
    if len(valid_labels) == 3:
        counts = Counter(valid_labels)
        return max(counts, key=counts.get)
    elif len(valid_labels) >= 1:
        counts = Counter(valid_labels)
        if len(counts) == 1:
            return valid_labels[0]
        elif counts.most_common(1)[0][1] > len(valid_labels) / 2:
            return counts.most_common(1)[0][0]
        else:
            return "Not Green Patent"  # conservative fallback on tie
    else:
        return "Not Green Patent"      # nothing parsed --> conservative

# --------------------- Main (model-wise sweeps + Session) -------
def main():
    in_path = pathlib.Path(INPUT_CSV)
    assert in_path.exists(), f"Input not found: {in_path}"

    records = load_data(str(in_path))

    # single HTTP session for connection reuse (keep-alive / pooling) :contentReference[oaicite:4]{index=4}
    session = requests.Session()

    # process one model at a time and checkpoint every N rows
    for model in MODELS:
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", model)
        need = [i for i, r in enumerate(records) if not r.get(tag)]
        if not need:
            continue

        processed_since_save = 0
        for i in tqdm(need, desc=f"Model sweep: {model}", unit="row"):
            rec = records[i]
            title = rec.get("Title", "") or rec.get("title", "")
            abstract = rec.get("Abstract", "") or rec.get("abstract", "")
            prompt = build_prompt(title, abstract)

            try:
                resp_text = call_ollama(session, model, prompt)
                parsed = parse_json_strict_or_loose(resp_text)
                label = normalize_label(parsed.get("label"))
                reason = (parsed.get("reason") or "").strip()
                rec[tag] = json.dumps({"label": label, "reason": reason}, ensure_ascii=False)
            except Exception as e:
                rec[tag] = json.dumps({"label": None, "reason": f"Error: {e}"}, ensure_ascii=False)

            processed_since_save += 1
            if processed_since_save >= CHECKPOINT_INTERVAL:
                write_checkpoint(records, OUTPUT_CSV)
                processed_since_save = 0

        # checkpoint at end of this model even if < interval
        write_checkpoint(records, OUTPUT_CSV)

    # finalize ensemble per row (same logic as before) + checkpointing
    processed_since_save = 0
    for rec in tqdm(records, desc="Finalizing ensemble", unit="row"):
        rec["Ensemble_Final_Label"] = compute_final_label(rec, MODELS)
        processed_since_save += 1
        if processed_since_save >= CHECKPOINT_INTERVAL:
            write_checkpoint(records, OUTPUT_CSV)
            processed_since_save = 0

    # final write
    write_data(records, OUTPUT_CSV)
    print(f"✔ Done. Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()