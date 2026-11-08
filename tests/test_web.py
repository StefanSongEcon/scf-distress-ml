import json
import re

import pytest

from scf_distress import predict_deploy as PD
from scf_distress import webapp

pytestmark = pytest.mark.skipif(not PD.MODEL_PATH.exists(), reason="predictive model not built")


def test_page_embeds_current_model(tmp_path):
    out = webapp.build(out=tmp_path / "index.html")
    html = out.read_text()
    blob = re.search(r'<script id="model-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    embedded = json.loads(blob.replace("<\\/", "</"))
    model = json.loads(PD.MODEL_PATH.read_text())
    assert embedded["inputs"] == list(model["inputs"])
    for oc, spec in model["outcomes"].items():
        e = embedded["outcomes"][oc]
        assert e["coefficients"] == spec["coefficients"]
        assert e["intercept"] == spec["intercept"]
        assert e["percentiles_2025"] == spec["percentiles_2025"]
    assert "__MODEL_JSON__" not in html and "__MODEL_VERSION__" not in html


def test_javascript_implements_every_model_feature():
    js = webapp.TEMPLATE.read_text()
    model = json.loads(PD.MODEL_PATH.read_text())
    for spec in model["outcomes"].values():
        for feat in spec["features"]:
            assert re.search(rf"\bf\.{feat}\b", js), f"feature {feat} not computed in the page"


def test_predictive_inputs_and_signs():
    model = json.loads(PD.MODEL_PATH.read_text())
    assert list(model["inputs"]) == PD.PRED_INPUTS
    c = model["outcomes"]["any_late_payment_2y"]["coefficients"]
    assert c["liquid_k"] < 0 and c["debt_k"] > 0 and c["recently_late"] > 0 and c["home_value_k"] < 0
    assert "homeowner" not in c  # dropped: crisis-era effect, and keeps the score monotone in home value
