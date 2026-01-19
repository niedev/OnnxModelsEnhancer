from typing import Dict, List, Optional
import huggingface_hub
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import math
import random
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


# Two-char language code -> desired weight (percent as fraction)
languages_distribution: Dict[str, float] = {
    # Latin-script (47%)
    "en": 0.18,
    "es": 0.06,
    "fr": 0.05,
    "de": 0.05,
    "it": 0.04,
    "pt": 0.03,
    "nl": 0.02,
    "pl": 0.02,
    "cs": 0.01,
    "sv": 0.01,

    # Asian (15%)
    "zh": 0.07,
    "ja": 0.05,
    "ko": 0.03,

    # Greek + Cyrillic (10%)
    "ru": 0.05,
    "uk": 0.02,
    "bg": 0.015,
    "el": 0.015,

    # Indic (10%)
    "hi": 0.03,
    "bn": 0.02,
    "ta": 0.015,
    "te": 0.015,
    "mr": 0.01,
    "ml": 0.01,

    # Arabic-script (7%)
    "ar": 0.04,
    "fa": 0.02,
    "ur": 0.01,

    # SEA (6%)
    "vi": 0.02,
    "th": 0.015,
    "id": 0.015,
    "km": 0.01,

    # African (5%)
    "ha": 0.02,
    "am": 0.01,
    "so": 0.01,
    "yo": 0.01,
}

# Map 2-char codes to FLORES+ iso_639_3 codes (as used in the dataset column iso_639_3)
# FLORES+ exposes: text, iso_639_3, iso_15924, etc.
iso6393_map: Dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "pt": "por",
    "it": "ita",
    "nl": "nld",
    "pl": "pol",
    "cs": "ces",
    "sv": "swe",

    "zh": "cmn",
    "ja": "jpn",
    "ko": "kor",

    "ru": "rus",
    "uk": "ukr",
    "bg": "bul",
    "el": "ell",
    "sr": "srp",

    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "ml": "mal",

    "ar": "arb",
    "fa": "pes",
    "ur": "urd",
    "ps": "pus",

    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "km": "khm",

    "sw": "swa",
    "ha": "hau",
    "am": "amh",
    "so": "som",
    "yo": "yor",
}



def take_to_dataset(streaming_ds, n: int) -> Dataset:
    return Dataset.from_list(list(streaming_ds.take(n)))


def getDatasetData(
    num_samples: int,
    split: str = "dev",
    seed: int = 42,
    hf_token: Optional[str] = None,
    log = False
) -> List[dict[str, str]]:
    """
    Build calibration samples from FLORES+ (openlanguagedata/flores_plus) using the
    user-specified language distribution.

    Output format: [{"lang": "<2-char-code>", "text": "<text>"}, ...]

    Notes:
      - FLORES+ is a *gated* dataset on Hugging Face. You must:
          1) accept its terms on the dataset page, and
          2) be logged in / provide a token (hf_token) if needed.

    """

    if num_samples <= 0:
        return []

    # Normalize weights (your list sums to ~1.01)
    total_w = sum(languages_distribution.values())
    if total_w <= 0:
        raise ValueError("Weights sum to 0; check distribution.")
    languages_distribution_norm = {k: v / total_w for k, v in languages_distribution.items()}

    # Compute integer target counts using largest-remainder method
    exact = {k: languages_distribution_norm[k] * num_samples for k in languages_distribution_norm}
    targets = {k: int(math.floor(exact[k])) for k in exact}
    remainder = num_samples - sum(targets.values())
    # Distribute leftover to biggest fractional parts
    frac_sorted = sorted(exact.keys(), key=lambda k: (exact[k] - targets[k]), reverse=True)
    for k in frac_sorted[: max(0, remainder)]:
        targets[k] += 1

    # Prepare result container
    collected: Dict[str, List[str]] = {k: [] for k in targets}
    needed = dict(targets)

    # Load FLORES+ (streaming so we don't keep the whole dataset in RAM)
    ds = load_dataset(
        "openlanguagedata/flores_plus",
        split=split,
        streaming=True,
        token=hf_token,
    )

    rng = random.Random(seed)

    # Stream once through the split, grabbing until all quotas are met
    remaining_total = sum(needed.values())
    for row in ds:
        if remaining_total <= 0:
            break

        row_iso = row.get("iso_639_3")
        text = row.get("text")

        if not row_iso or not text:
            continue

        # Find which 2-char language this row could serve
        # (small list, so O(L) scan is fine)
        for two in needed.keys():
            if needed[two] <= 0:
                continue
            if row_iso == iso6393_map[two]:
                collected[two].append(text)
                needed[two] -= 1
                remaining_total -= 1
                break

    # Sanity: ensure we collected everything
    missing = {k: v for k, v in needed.items() if v > 0}
    if missing:
        # You may hit this if the split doesn't contain enough rows for some language code
        # or the iso_639_3 mapping differs. Raise with actionable info.
        raise RuntimeError(
            "Not enough FLORES+ samples collected for some languages. Missing: "
            + ", ".join(f"{k}={v}" for k, v in missing.items())
            + f". Try another split (devtest), or verify iso_639_3 codes in FLORES+."
        )

    # Build final array
    out: List[dict[str, str]] = []
    for two, texts in collected.items():
        for t in texts:
            out.append({"lang": two, "text": t})

    #eventually print the data
    if(log):
        for o in out:
            print(o)
            print("")

    # Shuffle final output so GPTQ sees mixed languages/lengths
    rng.shuffle(out)

    return out


'''def getCalibrationData():
    huggingface_hub.login("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")

    flores = DatasetDict({
        "it": take_to_dataset(load_dataset("openlanguagedata/flores_plus", "ita_Latn", split="dev", streaming=True), 100),
        "en": take_to_dataset(load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="dev", streaming=True), 100),
    })
    for item in flores["it"]:
        print("")
        print(item["text"])'''