# SynthHeart web app

A nurse-facing cardiovascular risk check plus the real-vs-synthetic evidence and the trust scorecard, built as
self-contained static pages. The synthetic-trained XGBoost model runs **in the browser**, so nothing typed into the
form is sent anywhere, and the pages can be hosted anywhere static (e.g. Gizmos).

This is separate from the Streamlit skeleton in `../dashboard/`, which needs a Python server.

## Three designs, one engine

| Page | Design | Best for |
|---|---|---|
| `dist/console.html` | **A · Clinical Console**: EHR-style sidebar workspace, grouped clinical sections | desk / workstation |
| `dist/care.html` | **B · Care Dashboard**: gradient hero, sliders, ring gauge | demos, laptops |
| `dist/bedside.html` | **C · Bedside Assessment**: 4-step wizard, large touch targets | tablets, bedside |
| `dist/index.html` | **main app = Design B** (copy of care.html) | |
| `dist/designs.html` | design picker | |

Every design has the same tabs: risk check (estimate, what moved it, what-if scenarios, worklist kept in memory
only), real vs synthetic evidence, trust scorecard, and method & limits.

| File | Role |
|---|---|
| `predictor.js` | runs the exported XGBoost trees + per-input explanations |
| `app.js` | shared logic: form, validation (same range rules as cleaning), result, charts, scorecard, animations |
| `base.css` | shared component styles, driven by each design's colour tokens |
| `panels.html` | shared evidence / scorecard / method panels |
| `designs/*.html` | layout, theme and motion of each design |
| `build_webapp.py` | embeds model + results + shared parts into each design → `dist/` |
| `verify_predictor.py` | checks the browser maths against XGBoost on the whole test set |

## Build

**Just open `dist/index.html`** in a browser; it is committed and self-contained.

To rebuild, you need the model and the results JSONs. The model is committed at `models/risk_model_synthetic.json`: a copy of
notebook 03's output from the AI Factory, trained only on synthetic patients, so it is safe to share.
It uses `results/03b_scorecard_full.json` when present, otherwise `03_scorecard_full.json`.

```bash
uv run python implementations/synthheart/webapp/build_webapp.py            # -> webapp/dist/*.html
uv run python implementations/synthheart/webapp/verify_predictor.py        # needs Node; must print PASS
```

For development without the real model, pass any compatible model with `--model path.json --test-model`; the pages
then show a red "TEST MODEL" banner. Never deploy a test-model build.

## Link options

Add to the URL after `#`, joined with `&`: `compare`, `scorecard`, `about` open a tab; `example=1`..`3` loads an
example patient (`add` also puts it on the worklist); `light` / `dark` force a theme; `static` turns animations off.
Example: `console.html#example=3&light`.
