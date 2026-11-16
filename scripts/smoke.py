"""Run a self-contained CPU pipeline smoke test in temporary storage."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
                            capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise RuntimeError(f"{args}:\n{result.stdout}\n{result.stderr}")
    print("PASS", " ".join(map(str, args[:3])), flush=True)


def main():
    import numpy as np
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="acoustic-smoke-") as temp:
        tmp = Path(temp)
        cfg = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())
        cfg["paths"] = {key: str(tmp / key) for key in cfg["paths"]}
        config = tmp / "config.yaml"
        config.write_text(yaml.safe_dump(cfg))
        run("scripts/prepare_data.py", "--config", config, "--n-samples", 2, "--out", tmp / "samples")
        run("-m", "src.training.train", "--config", config, "--device", "cpu")
        checkpoint = tmp / "checkpoints/best.pt"
        sample = tmp / "sample.wav"
        t = np.arange(44100) / 44100
        sf.write(sample, (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 44100)
        run("-m", "src.inference.offline", "--checkpoint", checkpoint, "--input", sample, "--out", tmp / "result.json")
        run("-m", "src.evaluation.analyze_errors", "--checkpoint", checkpoint, "--out", tmp / "errors", "--n", 1)
        result = json.loads((tmp / "result.json").read_text())
        assert result[0]["n_frames"] == 96
        assert list((tmp / "errors").glob("*.png"))
    print("PASS acoustic generate/train/infer/analyze; temporary artifacts removed")


if __name__ == "__main__":
    main()
