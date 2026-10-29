// SynthHeart risk predictor: runs the exported XGBoost model in the browser.
//
// `model` comes from build_webapp.py, a compact copy of XGBoost's JSON model:
//   { features: [...], base_margin, trees: [{ f, t, l, r, m }, ...] }
// per tree, node arrays: f = split feature index, t = threshold (or leaf value at a leaf),
// l / r = child ids (-1 at a leaf), m = node mean value (for the explanation).
// XGBoost sends a row left when x < threshold, comparing in float32, so we do the same.
(function (root) {
  "use strict";

  function prepare(model) {
    for (const tree of model.trees) tree.t = tree.t.map(Math.fround);
    return model;
  }

  // Returns the risk (0-1) and, per feature, how much it pushed the score up or down.
  // The explanation is XGBoost's "approximate contributions" (Saabas): along the path a row takes,
  // each split's feature gets the change in node mean. Contributions + bias add up to the score exactly.
  function predict(model, row) {
    const x = model.features.map((name) => Math.fround(row[name]));
    const contributions = new Array(x.length).fill(0);
    let margin = model.base_margin;
    let bias = model.base_margin;
    for (const tree of model.trees) {
      let node = 0;
      bias += tree.m[0];
      while (tree.l[node] !== -1) {
        const child = x[tree.f[node]] < tree.t[node] ? tree.l[node] : tree.r[node];
        contributions[tree.f[node]] += tree.m[child] - tree.m[node];
        node = child;
      }
      margin += tree.t[node];
    }
    const byFeature = {};
    model.features.forEach((name, i) => { byFeature[name] = contributions[i]; });
    return { risk: 1 / (1 + Math.exp(-margin)), margin, bias, contributions: byFeature };
  }

  // Same derived feature as synthheart_common.add_features: BMI is computed, never typed in.
  function bmi(weightKg, heightCm) {
    return Math.round((weightKg / (heightCm / 100) ** 2) * 10) / 10;
  }

  const api = { prepare, predict, bmi };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SynthHeartPredictor = api;
})(typeof window !== "undefined" ? window : globalThis);
