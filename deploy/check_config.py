"""CI gate 4: the deployed configuration matches the manifest.

    python deploy/check_config.py http://127.0.0.1:8000

Three checks, standard library only:
1. `kubectl diff -k deploy/k8s` reports no drift between the manifests and the live objects.
2. The live Deployment runs the image and replica count the manifest asks for.
3. The running pod reports, on /config, the same values as the live ConfigMap. This catches a pod
   that started before a ConfigMap edit and never restarted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NS = "parley"
KEYS = {"PARLEY_NLU": "nlu", "PARLEY_LLM_MODEL": "llm_model", "PARLEY_LLM_URL": "llm_url",
        "PARLEY_ASR_MODEL": "asr_model", "PARLEY_TTS": "tts"}


def kubectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True)


def manifest_value(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"')
    raise KeyError(key)


def main(base: str) -> int:
    failures = []

    diff = kubectl("diff", "-k", str(ROOT / "k8s"))
    if diff.returncode != 0:
        failures.append("kubectl diff found drift:\n" + (diff.stdout or diff.stderr))

    manifest = (ROOT / "k8s" / "parley.yaml").read_text(encoding="utf-8")
    dep = json.loads(kubectl("-n", NS, "get", "deploy", "parley", "-o", "json").stdout)
    image = dep["spec"]["template"]["spec"]["containers"][0]["image"]
    if image != manifest_value(manifest, "image"):
        failures.append(f"image {image} != manifest {manifest_value(manifest, 'image')}")
    if dep["spec"]["replicas"] != int(manifest_value(manifest, "replicas")):
        failures.append(f"replicas {dep['spec']['replicas']} != manifest")
    if dep["status"].get("readyReplicas") != dep["spec"]["replicas"]:
        failures.append(f"ready replicas {dep['status'].get('readyReplicas')} != {dep['spec']['replicas']}")

    cm = json.loads(kubectl("-n", NS, "get", "configmap", "parley-config", "-o", "json").stdout)["data"]
    with urllib.request.urlopen(base + "/config", timeout=10) as r:
        live = json.loads(r.read())
    for env_key, cfg_key in KEYS.items():
        if cm.get(env_key) != live.get(cfg_key):
            failures.append(f"{env_key}: ConfigMap {cm.get(env_key)!r} but pod reports {live.get(cfg_key)!r}")

    if failures:
        print("FAIL\n" + "\n".join(failures))
        return 1
    print(f"OK: no drift, image {image}, config hash {live['config_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
