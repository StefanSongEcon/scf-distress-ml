"""Build the static risk calculator (``docs/index.html``) from the predictive model file.

The page embeds the model coefficients, 2025 feature means and percentile table, and
re-implements the feature transforms of ``deploy.build_features`` in JavaScript; scoring runs
entirely in the visitor's browser. Served by GitHub Pages from ``docs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .predict_deploy import MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("web") / "calculator.html"
OUT = ROOT / "docs/index.html"


def build(model_path: Path = MODEL_PATH, out: Path = OUT) -> Path:
    m = json.loads(Path(model_path).read_text())
    perf = next(r for r in m["cv_performance"] if r["outcome"] == "LATE" and r["inputs"] == "deployed_8_inputs")
    slim = {"version": m["version"], "inputs": list(m["inputs"]), "feature_means_2025": m["feature_means_2025"],
            "outcomes": {k: {kk: v[kk] for kk in ("features", "coefficients", "intercept", "rate_2025", "percentiles_2025")}
                         for k, v in m["outcomes"].items()},
            "performance": {"auc": perf["roc_auc"], "top10": perf["recall_top10"]}}
    html = TEMPLATE.read_text()
    html = html.replace("__MODEL_JSON__", json.dumps(slim, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__MODEL_VERSION__", m["version"])
    out.write_text(html)
    (out.parent / ".nojekyll").write_text("")
    return out


if __name__ == "__main__":
    print(build())
