"""ASR smoke test used to choose the Whisper size: WER per dialect for each size on CPU.

    uv run --extra speech python -m eval.smoke [sizes...]    # writes eval/smoke_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from eval.smoke_set import SMOKE_SET, reference_text
from eval.synth import render_to
from eval.wer import ZERO, count_errors
from speech.textnorm import normalize_orthography

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio" / "smoke"

# A bilingual vocabulary prompt: area names and booking words in both scripts. It biases
# decoding toward the domain; it contains no utterance from the smoke set.
DOMAIN_PROMPT = ("Viewing booking in Dubai Marina, Al Barsha, JLT, Downtown, Al Reem Island, Khalifa City. "
                 "حجز معاينة شقة في دبي مارينا والبرشاء وجزيرة الريم، غرفتين، درهم، الساعة.")


def main(sizes: list[str]) -> dict:
    from faster_whisper import WhisperModel

    files = {it["id"]: render_to(AUDIO / f"{it['id']}.wav", it["segments"], it["voice"]) for it in SMOKE_SET}
    prev = ROOT / "smoke_results.json"
    out: dict = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    out.setdefault("models", {})
    out["n_utterances"] = len(SMOKE_SET)
    for variant in sizes:
        size, _, opt = variant.partition(":")
        prompt = DOMAIN_PROMPT if opt == "prompt" else None
        model = WhisperModel(size, device="cpu", compute_type="int8")
        model.transcribe(str(files["en-01"]), beam_size=1)  # warm up
        per_dialect: dict[str, object] = {}
        rows = []
        secs = []
        for it in SMOKE_SET:
            t0 = time.perf_counter()
            # No language hint: the caller may use either language, so detection is part of the test.
            segs, info = model.transcribe(str(files[it["id"]]), beam_size=1, vad_filter=False,
                                          initial_prompt=prompt)
            hyp = " ".join(s.text for s in segs)
            secs.append(time.perf_counter() - t0)
            ref_n, hyp_n = normalize_orthography(reference_text(it)), normalize_orthography(hyp)
            c = count_errors(ref_n, hyp_n)
            per_dialect[it["dialect"]] = per_dialect.get(it["dialect"], ZERO) + c
            rows.append({"id": it["id"], "dialect": it["dialect"], "detected": info.language,
                         "ref": reference_text(it), "hyp": hyp.strip(), "wer": round(c.wer, 3)})
        secs.sort()
        out["models"][variant] = {
            "wer_by_dialect": {d: round(c.wer, 3) for d, c in per_dialect.items()},
            "median_seconds_per_utterance": round(secs[len(secs) // 2], 2),
            "rows": rows,
        }
        print(size, out["models"][variant]["wer_by_dialect"], out["models"][variant]["median_seconds_per_utterance"])
    (ROOT / "smoke_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def rescore() -> dict:
    """Add WER after number normalization (speech/normalize.py) to every stored row.

    Raw WER counts "4" against "four" as an error. The normalized score does not,
    which is the difference "The Hidden Cost of Digits" is about.
    """
    from datetime import date

    from speech.normalize import normalize

    today = date(2026, 10, 1)
    path = ROOT / "smoke_results.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    for m in out["models"].values():
        per: dict = {}
        for r in m["rows"]:
            c = count_errors(normalize(r["ref"], today).text, normalize(r["hyp"], today).text)
            r["wer_normalized"] = round(c.wer, 3)
            per[r["dialect"]] = per.get(r["dialect"], ZERO) + c
        m["wer_normalized_by_dialect"] = {d: round(c.wer, 3) for d, c in per.items()}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["rescore"]:
        for name, m in rescore()["models"].items():
            print(name, m["wer_by_dialect"], m["wer_normalized_by_dialect"], m["median_seconds_per_utterance"])
    else:
        main(sys.argv[1:] or ["tiny", "base", "small"])
