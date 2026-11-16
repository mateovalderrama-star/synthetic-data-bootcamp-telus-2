# TabDDPM vs NVIDIA NeMo Safe Synthesizer

Both generators are trained on the same patients and judged by the same scorecard. This file has the run steps,
the experiment design, and the method comparison for the slide deck. Measured numbers land in
`results/comparison/comparison.md` once the runs finish.

NeMo Safe Synthesizer (NSS) here means the open-source Python library (`nemo-safe-synthesizer` on PyPI, Apache-2.0),
which is the same engine as the NeMo microservice, without the Kubernetes deployment.

## Run steps (AI Factory, H200)

All commands from `implementations/synthheart` unless noted.

```bash
# 0 · Get the code and make sure the SynthHeart environment and data exist (skip what you already have)
git pull origin main
(cd ../.. && uv sync --dev --group tabular-data)
uv run python -m ipykernel install --user --name synthheart --display-name "SynthHeart (.venv)"
ls data/cardio/cardio.csv || uv run jupyter nbconvert --to notebook --execute 01_data_prep_and_baseline.ipynb \
      --output 01_data_prep_and_baseline.executed.ipynb --ExecutePreprocessor.kernel_name=synthheart
mkdir -p logs

# 1 · One-time: build the NSS environment (.venv-nss) and download its base model (~10-20 min)
bash setup_nss.sh

# 2 · Smoke run: full training, only 2,000 rows generated, then scored. Proves the whole chain works.
.venv-nss/bin/python 05_nss_generate.py --name nss_smoke --set generation.num_records=2000 2>&1 | tee logs/05_nss_smoke.log
uv run python 06_compare_generators.py --modes tabddpm nss_smoke

# 3 · The real thing, in the background (survives a closed browser tab)
nohup bash run_nss_comparison.sh > logs/nss_comparison.log 2>&1 &
tail -f logs/nss_comparison.log        # Ctrl+C stops watching, not the run

# 4 · Save the results
git add results/ && git commit -m "TabDDPM vs NeMo Safe Synthesizer results" && git push origin main
```

Step 2 also re-runs TabDDPM once (run name `tabddpm`, ~10 min) so both generators are timed on the same node by the
same GPU monitor. The bootcamp's `full` results are not touched.

Run times for NSS are not known yet. NVIDIA quotes about 15 minutes on an A100 for its default tutorial run and about
an hour with differential privacy, both generating far fewer rows than we ask for. The smoke run gives the first real
number: check `stage_minutes` in `results/02_generation_summary_nss_smoke.json` before launching step 3.

### If something fails

| Symptom | Try |
|---|---|
| `setup_nss.sh`: torch cannot see the GPU | `CUDA=130 bash setup_nss.sh` (needs driver 580.65.06+) |
| `setup_nss.sh`: Hugging Face download fails | The AI Factory blocks huggingface.co. Download `HuggingFaceTB/SmolLM3-3B` elsewhere, copy it into `~/.cache/huggingface`, then `export HF_HUB_OFFLINE=1` |
| "kernels package not installed" or a Kernels Hub error | add `--set training.attn_implementation=sdpa` |
| "Generation stopped prematurely ..." | the model is under-trained: add `--set training.num_input_records_to_sample=82270`, or `--set generation.use_structured_generation=true` |
| Out of memory in generation | another job is on the GPU. vLLM reserves about 80% of the card. Check `nvidia-smi` |
| "Unable to automatically determine a noise multiplier" (DP arm) | raise epsilon: `--set privacy.epsilon=12.0` |
| An arm failed half-way | fix it, launch `run_nss_comparison.sh` again. Finished arms are skipped |
| Re-run one arm from scratch | delete `results/02_generation_summary_<name>.json` and `results/03b_scorecard_<name>.json` |

The full NSS log of each arm is `logs/05_nss_<arm>.log`. A failed scorecard notebook is saved with its error in
`logs/03b_<name>.executed.ipynb`.

## What gets saved

| Path | What | Committed |
|---|---|---|
| `results/comparison/comparison.md`, `.json` | every metric side by side | yes |
| `results/comparison/*.png` | headline metrics, runtime, distributions, correlation gaps, risk curves, privacy vs utility | yes |
| `results/comparison/figures/<run>/` | every figure from that run's scorecard notebook (ROC, MIA, importance ...) | yes |
| `results/comparison/reports/03b_<run>.html` | the executed scorecard notebook, self-contained | yes |
| `results/comparison/gpu_<run>.json` | GPU memory, utilisation, energy, per-stage timing | yes |
| `results/comparison/nss_native/` | NSS's own HTML report, metrics JSON and resolved config | yes |
| `results/02_generation_summary_<run>.json`, `03b_scorecard_<run>.json` | per-run details | yes |
| `runs/<run>/`, `logs/` | models, adapters, synthetic CSVs, raw logs | no (git-ignored) |

## Experiment design

| Run | Generator | Settings | Question it answers |
|---|---|---|---|
| `tabddpm` | TabDDPM | bootcamp `full` config (20k iterations, 2,000 diffusion steps) | the reference, re-timed |
| `nss_default` | NSS | as shipped: SmolLM3-3B + LoRA, 25,000 training records | how does the product do out of the box? |
| `nss_dp8` | NSS | differential privacy, ε = 8 | what does a formal privacy guarantee cost in quality and time? |
| `nss_3pass` | NSS | ~3 passes over the train split (default is ~0.6 of one pass) | is the default under-trained for this table? |

Held equal for every run:

- **Training data**: the 41,135-row train split. The holdout and test splits are never shown to a generator.
- **Output size**: 41,135 synthetic rows scored (1×), plus a 2× table for the "does more data help" test.
- **Post-processing**: the same rounding and the same medical-range filter (`synthheart_common.py`).
- **Scorecard**: `03b_scorecard_size_matched.ipynb`, unchanged, same seed, same locked real test set.
- **Hardware and timing**: same H200, same GPU monitor (`gpu_monitor.py`).

Three things are not equal, on purpose or by necessity:

1. **NSS holds out 5% of its input (max 2,000 rows)** for its own privacy tests, so it trains on 39,135 rows. The
   scorecard treats only those rows as "seen", so the privacy numbers stay exact. To give NSS all rows instead, add
   `--set data.holdout=0` (its own report then loses the membership-inference score).
2. **NSS's PII replacement is switched off.** This table has no names, addresses or free text. Left on, its column
   classifier calls an NVIDIA-hosted LLM endpoint. Usage telemetry is also switched off. No patient data leaves
   the AI Factory; the only outbound traffic is downloading the package and the base model.
3. **One generation call for NSS.** It generates 82,270 rows once; the 1× table is the first 41,135 of them.
   TabDDPM sampled 1× and 2× separately.

## How each one generates data

| | TabDDPM | NeMo Safe Synthesizer |
|---|---|---|
| Model family | Diffusion model, trained from scratch | Pre-trained language model, fine-tuned |
| Network | MLP denoiser, 6 hidden layers (512 → 1024 ×4 → 512), a few million weights | SmolLM3-3B (3 billion weights, frozen) plus a LoRA adapter (rank 32). TinyLlama-1.1B and Mistral-7B also supported |
| How a row is represented | Numbers are scaled; categories are one-hot | The row is written out as JSON text and split into tokens |
| Training objective | Learn to remove noise: Gaussian noise on numeric columns, multinomial noise on categorical columns | Predict the next token of the row text |
| How a row is generated | Start from pure noise, denoise in 2,000 steps. All columns of a row emerge together | Write the row one token at a time, left to right (vLLM, temperature 0.9), then check it parses and fits the schema |
| What can go wrong in output | Values off the real grid (height 167.6) and out of range: needs rounding and a rule filter | Malformed or invalid records: NSS drops them itself. Optional structured generation forces the format |
| Prior knowledge | None. Everything is learned from the table | Whatever the language model already knows about the world and about text |
| Column types | Numeric and categorical | Numeric, categorical, free text, dates (time series is experimental) |
| Setup per dataset | Column-type metadata files, column order matters | None: point it at a CSV |
| Privacy mechanism | None built in. Privacy is measured afterwards | Optional DP-SGD (Opacus): a formal (ε, δ) guarantee. Optional PII detection and replacement before training |
| Evaluation | We built the scorecard (midst-toolkit metrics + our own) | Built-in HTML report: Synthetic Quality Score and Data Privacy Score, each out of 10 |
| Hardware | Trained in 6.7 min on the H200. Small enough for a modest GPU | NVIDIA lists an A100 (80 GB class) as the minimum. Linux only |
| Smallest dataset | No hard limit | 1,000 rows (10,000 recommended with DP) |
| Multi-table | Yes, via ClavaDDPM in the same toolkit | Single table, with grouped records |
| Licence and support | Research code (Vector's midst-toolkit) | Apache-2.0, an NVIDIA product with docs, releases and a microservice version |

Sources: [NSS docs](https://nvidia-nemo.github.io/Safe-Synthesizer/),
[NSS configuration reference](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/configuration/),
[NSS evaluation](https://nvidia-nemo.github.io/Safe-Synthesizer/product-overview/evaluation/),
[NSS on PyPI](https://pypi.org/project/nemo-safe-synthesizer/), TabDDPM paper (Kotelnikov et al., ICML 2023), and
notebook 02 for the TabDDPM settings.

## Strengths and weaknesses, before the numbers

Expectations from how the methods work. The runs confirm or correct them.

**TabDDPM**
- For: built for exactly this kind of table (a dozen numeric and categorical columns). Tiny model, minutes to train,
  seconds to sample. Already within 0.003 AUC of real data and passing 8/8 targets.
- Against: no formal privacy guarantee. Cannot handle free text or detect PII. Needs hand-written metadata and a
  post-filter. Research code, so we own maintenance and the evaluation.

**NeMo Safe Synthesizer**
- For: differential privacy as a switch. PII replacement and a privacy report included. Handles free text and mixed
  tables. No per-dataset configuration. Vendor-supported.
- Against: a 3-billion-weight model to produce 12 small columns. Expect it to be much slower and to use far more GPU.
  Language models write numbers as text, so fine numeric detail and cross-column correlations are the likely weak
  spot. Output varies with sampling temperature.

**The trade-offs the deck should show**
1. **Quality vs cost**: TSTR AUC and fidelity against training time, generation time, GPU memory and energy.
2. **Measured privacy vs guaranteed privacy**: both get the same empirical attacks. Only NSS with DP can state a
   bound, and `nss_dp8` shows what that bound costs.
3. **Fit vs breadth**: TabDDPM fits this table. NSS covers data TabDDPM cannot touch (see the caveat below).
4. **Build vs buy**: who owns the evaluation, the upgrades and the support.

## Caveats to say out loud

- **This dataset favours TabDDPM.** It is small, clean, and all numeric or categorical. NSS's main selling points
  (free text, PII, mixed types) are not exercised at all. A fair headline is "on a simple clinical table", not
  "in general".
- **One run per arm.** No error bars. Small gaps are ties until repeated.
- **NSS's own scores and our scorecard measure different things.** Its quality score uses its own metrics and its
  privacy score uses a 2,000-row holdout. Report them as "NVIDIA's own report says", next to ours, not mixed in.
- **GPU memory for NSS is a reservation.** vLLM takes about 80% of the card regardless of need.

## Worth running next, if H200 time allows

1. **Repeat each arm 2-3 times** for error bars. For NSS: `05_nss_generate.py --name nss_default_r2`. Cheapest way
   to make every claim in the deck defensible.
2. **Privacy budget sweep**: `nss_dp4`, `nss_dp16` next to `nss_dp8`, for a privacy-utility curve with three points.
   ```bash
   .venv-nss/bin/python 05_nss_generate.py --name nss_dp4 --set privacy.dp_enabled=true --set privacy.epsilon=4.0 \
       --set training.batch_size=16 --set generation.use_structured_generation=true
   ```
3. **Small-clinic test**: train both on 2,000 patients. A pre-trained language model may hold up better with little
   data than a diffusion model trained from scratch. This needs a small change to how notebook 02 picks its train
   file, so it is not wired up yet.
4. **A table with free text or PII** (for example clinical notes plus demographics). TabDDPM cannot run on it, so
   this is a capability demo for NSS, not a head-to-head.
5. **A larger base model**: `--set training.pretrained_model=mistralai/Mistral-7B-Instruct-v0.3`. The model is gated
   on Hugging Face, so it needs an access token.
