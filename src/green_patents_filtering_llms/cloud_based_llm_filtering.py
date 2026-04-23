'''
@File    :   cloud_based_llm_filtering.py
@Time    :   04/2026
@Author  :   nikifori
@Version :   -
'''
"""
OpenAI Batch patent classifier.

Key assumptions
---------------
- "Lens ID" is the unique patent key
- Existing classified patents in OUTPUT_CSV should be skipped on PREPARE,
  except optionally rows marked as RETRY_NEEDED when RETRY_FAILED_EXISTING=True
- Output is appended/updated in OUTPUT_CSV after COLLECT_ALL

How to use
----------
1) Set ACTION = "PREPARE" and run
2) Set ACTION = "SUBMIT_ALL" and run
3) Set ACTION = "STATUS_ALL" and run whenever you want
4) When all batches are completed, set ACTION = "COLLECT_ALL" and run

Requirements
------------
pip install openai pandas

Environment
-----------
export OPENAI_API_KEY="sk-..."
"""
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

# ============================================================
# USER SETTINGS
# ============================================================

ACTION = "COLLECT_ALL"
# Valid values:
# "PREPARE", "SUBMIT_ALL", "STATUS_ALL", "COLLECT_ALL"

MODEL = "gpt-5.4-nano"

# -1 means all patents from INPUT_CSV
MAX_PATENTS = 50000

# Max requests per batch file
# OpenAI allows up to 50,000 requests per batch
RECORDS_PER_BATCH = 1000

INPUT_CSV = "/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017_initial.csv"
OUTPUT_CSV = "/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-openai-batch-filtered-5_4_nano_minimal.csv"

WORK_DIR = "/home/nikifori/Desktop/thesis/repo/data/openai_batch_runs_green_patents_5_4_nano_minimal"

CSV_DELIM = ","

# Increase this if you still see too many incompletes.
MAX_OUTPUT_TOKENS = 400

# If True, PREPARE will also retry rows already present in OUTPUT_CSV
# whose FINAL_LABEL_COL == RETRY_LABEL.
# If False, PREPARE will only process truly new patents.
RETRY_FAILED_EXISTING = False

# If True, SUBMIT_ALL submits one batch at a time and waits until it reaches
# a terminal state before submitting the next one. This helps avoid the
# organization enqueued-token limit for batch processing.
SERIALIZE_BATCH_SUBMISSION = True

# Polling interval (seconds) while waiting for a submitted batch to finish.
BATCH_POLL_SECONDS = 30

# ============================================================
# OUTPUT COLUMNS / LABELS
# ============================================================

RAW_RESULT_COL = "openai_result_json"
FINAL_LABEL_COL = "Final_Label"
PATENT_KEY_COLUMN = "Lens ID"

PARSE_STATUS_COL = "openai_parse_status"
RESPONSE_STATUS_COL = "openai_response_status"
INCOMPLETE_REASON_COL = "openai_incomplete_reason"
RAW_RESPONSE_TEXT_COL = "openai_raw_response_text"

RETRY_LABEL = "RETRY_NEEDED"

PARSE_STATUS_OK = "ok"
PARSE_STATUS_EMPTY = "empty_output"
PARSE_STATUS_UNPARSEABLE = "unparseable_output"
PARSE_STATUS_BATCH_ERROR = "batch_error"
PARSE_STATUS_MISSING = "missing_result"

# ============================================================
# OPENAI CLIENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it in your environment before running this script."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# PROMPT + SCHEMA
# ============================================================

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

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["Green Patent", "Not Green Patent"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}

# ============================================================
# HELPERS
# ============================================================

GREEN_RE = re.compile(r"^\s*(green(?:\s*patent)?)\s*$", re.I)
NOTGREEN_RE = re.compile(r"^\s*(?:not\s+green(?:\s*patent)?|non[-\s]*green)\s*$", re.I)


def normalize_label(label: str | None) -> str:
    t = (label or "").strip()

    if NOTGREEN_RE.match(t):
        return "Not Green Patent"
    if GREEN_RE.match(t):
        return "Green Patent"

    tl = t.lower()
    if tl in {"not green", "not green patent", "no", "n"}:
        return "Not Green Patent"
    if tl in {"green", "green patent", "yes", "y"}:
        return "Green Patent"

    return RETRY_LABEL


def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def patent_key_from_record(rec: dict) -> str:
    if PATENT_KEY_COLUMN not in rec:
        raise KeyError(f"Required key column '{PATENT_KEY_COLUMN}' not found in record.")
    return normalize_text(rec.get(PATENT_KEY_COLUMN, ""))


def workdir() -> Path:
    p = Path(WORK_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def meta_path() -> Path:
    return workdir() / "meta.json"


def batch_input_path(batch_index: int) -> Path:
    return workdir() / f"batch_input_{batch_index:04d}.jsonl"


def batch_output_path(batch_index: int) -> Path:
    return workdir() / f"batch_output_{batch_index:04d}.jsonl"


def batch_error_path(batch_index: int) -> Path:
    return workdir() / f"batch_error_{batch_index:04d}.jsonl"


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_input(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return f"Title: {title}\nAbstract: {abstract}"


def build_response_body(title: str, abstract: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "store": False,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {
            "effort": "low"
        },
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_input(title, abstract),
                    }
                ],
            },
        ],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "green_patent_classifier",
                "schema": OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    }


def extract_output_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = body.get("output", [])
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts).strip()


def parse_prediction(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()

    if not raw_text:
        return {
            "label": RETRY_LABEL,
            "reason": "Empty model output",
            "parse_status": PARSE_STATUS_EMPTY,
        }

    try:
        obj = json.loads(raw_text)
    except Exception:
        if "{" in raw_text and "}" in raw_text:
            candidate = raw_text[raw_text.find("{"): raw_text.rfind("}") + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                return {
                    "label": RETRY_LABEL,
                    "reason": "Unparseable model output",
                    "parse_status": PARSE_STATUS_UNPARSEABLE,
                }
        else:
            return {
                "label": RETRY_LABEL,
                "reason": "Unparseable model output",
                "parse_status": PARSE_STATUS_UNPARSEABLE,
            }

    label = normalize_label(obj.get("label"))
    reason = str(obj.get("reason") or "").strip()
    if not reason:
        reason = "No reason provided"

    if label not in {"Green Patent", "Not Green Patent"}:
        return {
            "label": RETRY_LABEL,
            "reason": f"Unexpected label in model output: {obj.get('label')!r}",
            "parse_status": PARSE_STATUS_UNPARSEABLE,
        }

    return {
        "label": label,
        "reason": reason,
        "parse_status": PARSE_STATUS_OK,
    }


def validate_required_columns(df: pd.DataFrame, csv_name: str) -> None:
    required = [PATENT_KEY_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {csv_name}: {missing}")


def load_existing_output_df(output_csv_path: str) -> pd.DataFrame | None:
    output_path = Path(output_csv_path)
    if not output_path.exists():
        return None

    out_df = pd.read_csv(output_csv_path, sep=CSV_DELIM)
    if out_df.empty:
        return out_df

    validate_required_columns(out_df, output_csv_path)
    return out_df


def get_existing_output_index(output_csv_path: str) -> tuple[pd.DataFrame | None, dict[str, dict[str, Any]]]:
    out_df = load_existing_output_df(output_csv_path)
    if out_df is None or out_df.empty:
        return out_df, {}

    index: dict[str, dict[str, Any]] = {}
    for _, row in out_df.iterrows():
        rec = row.to_dict()
        key = patent_key_from_record(rec)
        if key:
            index[key] = rec
    return out_df, index


def should_retry_existing_record(rec: dict[str, Any]) -> bool:
    label = normalize_text(rec.get(FINAL_LABEL_COL, ""))
    return label == RETRY_LABEL


def select_records_for_prepare(input_df: pd.DataFrame) -> tuple[list[dict], int, int]:
    _, existing_index = get_existing_output_index(OUTPUT_CSV)

    records_all = input_df.to_dict(orient="records")
    records_to_process: list[dict] = []
    skipped_existing_non_retry = 0
    selected_existing_retries = 0

    for rec in records_all:
        key = patent_key_from_record(rec)
        existing = existing_index.get(key)

        if existing is None:
            records_to_process.append(rec)
            continue

        if RETRY_FAILED_EXISTING and should_retry_existing_record(existing):
            records_to_process.append(rec)
            selected_existing_retries += 1
        else:
            skipped_existing_non_retry += 1

    return records_to_process, skipped_existing_non_retry, selected_existing_retries


def is_terminal_batch_status(status: str | None) -> bool:
    return status in {"completed", "failed", "expired", "cancelled", "canceled"}


def wait_for_batch_terminal_status(batch_id: str, batch_index: int) -> dict[str, Any]:
    while True:
        job = client.batches.retrieve(batch_id)
        status = getattr(job, "status", None)

        print(f"Batch {batch_index}: polled status={status}")

        if is_terminal_batch_status(status):
            return {
                "status": status,
                "output_file_id": getattr(job, "output_file_id", None),
                "error_file_id": getattr(job, "error_file_id", None),
            }

        time.sleep(BATCH_POLL_SECONDS)


# ============================================================
# ACTIONS
# ============================================================

def prepare_batches() -> None:
    input_df = pd.read_csv(INPUT_CSV, sep=CSV_DELIM)
    validate_required_columns(input_df, INPUT_CSV)

    total_original = len(input_df)

    if MAX_PATENTS != -1:
        input_df = input_df.iloc[:MAX_PATENTS].copy()

    records_to_process, skipped_existing_non_retry, selected_existing_retries = select_records_for_prepare(input_df)

    total = len(records_to_process)
    if total == 0:
        print("No patents selected for processing.")
        print(f"Original records considered:         {len(input_df)}")
        print(f"Skipped existing non-retry records:  {skipped_existing_non_retry}")
        print(f"Selected existing retry records:     {selected_existing_retries}")
        return

    batch_count = math.ceil(total / RECORDS_PER_BATCH)
    batches = []

    for batch_index in range(batch_count):
        start = batch_index * RECORDS_PER_BATCH
        end = min(start + RECORDS_PER_BATCH, total)
        subset = records_to_process[start:end]

        input_path = batch_input_path(batch_index)

        with input_path.open("w", encoding="utf-8") as f:
            for local_idx, rec in enumerate(subset):
                prepare_idx = start + local_idx
                title = rec.get("Title", "") or rec.get("title", "")
                abstract = rec.get("Abstract", "") or rec.get("abstract", "")

                line = {
                    "custom_id": f"patent-{prepare_idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": build_response_body(title=title, abstract=abstract),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        batches.append(
            {
                "batch_index": batch_index,
                "start_index": start,
                "end_index_exclusive": end,
                "request_count": end - start,
                "input_jsonl": str(input_path),
                "input_file_id": None,
                "batch_id": None,
                "status": "prepared",
                "output_file_id": None,
                "error_file_id": None,
                "output_jsonl": str(batch_output_path(batch_index)),
                "error_jsonl": str(batch_error_path(batch_index)),
            }
        )

    meta = {
        "model": MODEL,
        "input_csv": INPUT_CSV,
        "output_csv": OUTPUT_CSV,
        "max_patents": MAX_PATENTS,
        "records_per_batch": RECORDS_PER_BATCH,
        "patent_key_column": PATENT_KEY_COLUMN,
        "retry_failed_existing": RETRY_FAILED_EXISTING,
        "serialize_batch_submission": SERIALIZE_BATCH_SUBMISSION,
        "batch_poll_seconds": BATCH_POLL_SECONDS,
        "total_original_records_in_input_csv": total_original,
        "total_records_considered_after_max_patents": len(input_df),
        "skipped_existing_non_retry": skipped_existing_non_retry,
        "selected_existing_retry_records": selected_existing_retries,
        "total_prepared_records": total,
        "batch_count": batch_count,
        "batches": batches,
        "prepared_records": [
            {
                "patent_key": patent_key_from_record(rec),
                "record": rec,
            }
            for rec in records_to_process
        ],
    }
    save_json(meta, meta_path())

    print("Prepared batch input files.")
    print(f"Original records in input CSV:       {total_original}")
    print(f"Records considered:                  {len(input_df)}")
    print(f"Skipped existing non-retry records:  {skipped_existing_non_retry}")
    print(f"Selected existing retry records:     {selected_existing_retries}")
    print(f"Prepared records total:              {total}")
    print(f"Batch count:                         {batch_count}")
    print(f"Meta file:                           {meta_path()}")


def submit_all_batches() -> None:
    meta = load_json(meta_path())
    if not meta:
        raise RuntimeError("Meta file not found. Run PREPARE first.")

    for batch in meta["batches"]:
        if batch.get("batch_id"):
            print(f"Skipping batch {batch['batch_index']} (already submitted).")
            continue

        with open(batch["input_jsonl"], "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")

        job = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "task": "green-patent-classification",
                "model": MODEL,
                "batch_index": str(batch["batch_index"]),
                "retry_failed_existing": str(RETRY_FAILED_EXISTING),
            },
        )

        batch["input_file_id"] = uploaded.id
        batch["batch_id"] = job.id
        batch["status"] = job.status

        print(
            f"Submitted batch {batch['batch_index']}: "
            f"batch_id={job.id}, requests={batch['request_count']}, status={job.status}"
        )

        save_json(meta, meta_path())

        if SERIALIZE_BATCH_SUBMISSION:
            print(
                f"Waiting for batch {batch['batch_index']} to reach a terminal state "
                f"before submitting the next batch..."
            )
            terminal_info = wait_for_batch_terminal_status(
                batch_id=job.id,
                batch_index=batch["batch_index"],
            )
            batch["status"] = terminal_info["status"]
            batch["output_file_id"] = terminal_info["output_file_id"]
            batch["error_file_id"] = terminal_info["error_file_id"]

            print(
                f"Batch {batch['batch_index']} finished with status={batch['status']}"
            )
            save_json(meta, meta_path())

    print("All possible batches submitted.")


def status_all_batches() -> None:
    meta = load_json(meta_path())
    if not meta:
        raise RuntimeError("Meta file not found. Run PREPARE first.")

    completed = 0
    total = len(meta["batches"])

    for batch in meta["batches"]:
        batch_id = batch.get("batch_id")
        if not batch_id:
            print(f"Batch {batch['batch_index']}: not submitted yet")
            continue

        job = client.batches.retrieve(batch_id)
        batch["status"] = job.status
        batch["output_file_id"] = getattr(job, "output_file_id", None)
        batch["error_file_id"] = getattr(job, "error_file_id", None)

        print(
            f"Batch {batch['batch_index']}: "
            f"status={job.status}, requests={batch['request_count']}"
        )

        if getattr(job, "errors", None):
            print("Errors:")
            try:
                for err in job.errors.data:
                    print(
                        f"  code={getattr(err, 'code', None)} "
                        f"line={getattr(err, 'line', None)} "
                        f"param={getattr(err, 'param', None)} "
                        f"message={getattr(err, 'message', None)}"
                    )
            except Exception:
                print(job.errors)

        if getattr(job, "error_file_id", None):
            print(f"error_file_id: {job.error_file_id}")

        if job.status == "completed":
            completed += 1

    save_json(meta, meta_path())
    print(f"Completed batches: {completed}/{total}")


def collect_all_batches() -> None:
    meta = load_json(meta_path())
    if not meta:
        raise RuntimeError("Meta file not found. Run PREPARE first.")

    prepared_records = meta.get("prepared_records", [])
    if not prepared_records:
        raise RuntimeError("No prepared_records found in meta.json. Run PREPARE again.")

    predictions_by_prepare_index: dict[int, dict[str, Any]] = {}

    for batch in meta["batches"]:
        batch_id = batch.get("batch_id")
        if not batch_id:
            raise RuntimeError(f"Batch {batch['batch_index']} has not been submitted.")

        job = client.batches.retrieve(batch_id)
        batch["status"] = job.status
        batch["output_file_id"] = getattr(job, "output_file_id", None)
        batch["error_file_id"] = getattr(job, "error_file_id", None)

        if job.status != "completed":
            raise RuntimeError(
                f"Batch {batch['batch_index']} is not completed yet. Current status: {job.status}"
            )

        if not batch["output_file_id"]:
            raise RuntimeError(
                f"Batch {batch['batch_index']} completed but has no output_file_id."
            )

        output_bytes = client.files.content(batch["output_file_id"]).read()
        Path(batch["output_jsonl"]).write_bytes(output_bytes)

        if batch["error_file_id"]:
            error_bytes = client.files.content(batch["error_file_id"]).read()
            Path(batch["error_jsonl"]).write_bytes(error_bytes)

        with open(batch["output_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                custom_id = row.get("custom_id", "")
                m = re.match(r"patent-(\d+)$", custom_id)
                if not m:
                    continue
                prepare_idx = int(m.group(1))

                response = row.get("response") or {}
                body = response.get("body") or {}
                response_status = body.get("status")
                incomplete_details = body.get("incomplete_details") or {}
                incomplete_reason = incomplete_details.get("reason")

                error = row.get("error")
                if error:
                    predictions_by_prepare_index[prepare_idx] = {
                        "label": RETRY_LABEL,
                        "reason": f"Batch error: {error.get('message', 'Unknown error')}",
                        "parse_status": PARSE_STATUS_BATCH_ERROR,
                        "response_status": response_status,
                        "incomplete_reason": incomplete_reason,
                        "raw_text": "",
                    }
                    continue

                raw_text = extract_output_text(body)
                pred = parse_prediction(raw_text)
                pred["response_status"] = response_status
                pred["incomplete_reason"] = incomplete_reason
                pred["raw_text"] = raw_text
                predictions_by_prepare_index[prepare_idx] = pred

    existing_output_df, _ = get_existing_output_index(OUTPUT_CSV)

    merged_rows_by_key: dict[str, dict[str, Any]] = {}

    if existing_output_df is not None and not existing_output_df.empty:
        for _, row in existing_output_df.iterrows():
            rec = row.to_dict()
            key = patent_key_from_record(rec)
            if key:
                merged_rows_by_key[key] = rec

    for prepare_idx, item in enumerate(prepared_records):
        rec = dict(item["record"])
        key = item["patent_key"]

        pred = predictions_by_prepare_index.get(
            prepare_idx,
            {
                "label": RETRY_LABEL,
                "reason": "Missing batch result",
                "parse_status": PARSE_STATUS_MISSING,
                "response_status": None,
                "incomplete_reason": None,
                "raw_text": "",
            },
        )

        rec[RAW_RESULT_COL] = json.dumps(
            {
                "label": pred["label"],
                "reason": pred["reason"],
                "parse_status": pred["parse_status"],
                "response_status": pred.get("response_status"),
                "incomplete_reason": pred.get("incomplete_reason"),
            },
            ensure_ascii=False,
        )
        rec[FINAL_LABEL_COL] = pred["label"]
        rec[PARSE_STATUS_COL] = pred["parse_status"]
        rec[RESPONSE_STATUS_COL] = pred.get("response_status")
        rec[INCOMPLETE_REASON_COL] = pred.get("incomplete_reason")
        rec[RAW_RESPONSE_TEXT_COL] = pred.get("raw_text", "")

        merged_rows_by_key[key] = rec

    final_rows = list(merged_rows_by_key.values())
    final_df = pd.DataFrame.from_records(final_rows)
    final_df.to_csv(OUTPUT_CSV, index=False, sep=CSV_DELIM)

    save_json(meta, meta_path())

    print("All batch results collected successfully.")
    print(f"Updated output CSV: {OUTPUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if ACTION == "PREPARE":
        prepare_batches()
    elif ACTION == "SUBMIT_ALL":
        submit_all_batches()
    elif ACTION == "STATUS_ALL":
        status_all_batches()
    elif ACTION == "COLLECT_ALL":
        collect_all_batches()
    else:
        raise RuntimeError(
            "Invalid ACTION. Use one of: PREPARE, SUBMIT_ALL, STATUS_ALL, COLLECT_ALL"
        )


if __name__ == "__main__":
    main()