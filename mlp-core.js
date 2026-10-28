(function (root) {
  'use strict';
  const Edges = typeof module !== 'undefined' && module.exports ? require('./core.js') : root.RgbEdges;
  const INPUTS = 7, HIDDEN = 12, MODEL_SCHEMA = 'rgb-edge-mlp/v1';
  const FEATURES = ['r/255', 'g/255', 'b/255', 'mean_r/255', 'mean_g/255', 'mean_b/255', 'score/195075'];
  const K = Math.sqrt(2 / Math.PI), A = 0.044715;
  const random = (seed = 42) => {
    let state = seed >>> 0;
    return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; };
  };
  function shuffle(values, rng) {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1)); [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  }
  function gelu(x) { return .5 * x * (1 + Math.tanh(K * (x + A * x ** 3))); }
  function geluPrime(x) {
    const t = Math.tanh(K * (x + A * x ** 3));
    return .5 * (1 + t) + .5 * x * (1 - t * t) * K * (1 + 3 * A * x * x);
  }
  function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
  function vector(image, result, i) {
    const d = image.data, m = result.means;
    return [d[4 * i] / 255, d[4 * i + 1] / 255, d[4 * i + 2] / 255,
      m[3 * i] / 255, m[3 * i + 1] / 255, m[3 * i + 2] / 255, result.scores[i] / Edges.MAX_SCORE];
  }
  function fingerprint(image) {
    let h = 2166136261;
    for (const v of image.data) { h ^= v; h = Math.imul(h, 16777619); }
    return `${image.width}x${image.height}:${h >>> 0}`;
  }
  function annotation(value) {
    if (value?.schema !== 'rgb-neighbor-edge/v1' || value.labels?.provenance !== 'manual_user_annotation') throw new Error('A 페이지의 학습 JSON 저장으로 받은 파일을 선택해 주세요.');
    const { width, height, png_data_url: png } = value.image || {}, mask = value.labels.mask;
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width * height > 1048576 ||
      typeof png !== 'string' || !/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(png) || !Array.isArray(mask) ||
      mask.length !== width * height || !mask.every(v => v === 0 || v === 1 || v === 2)) throw new Error('JSON의 이미지나 정답 정보가 올바르지 않습니다.');
    const encoding = value.labels.encoding;
    if (!encoding || encoding.unknown !== 0 || encoding.edge !== 1 || encoding.non_edge !== 2) throw new Error('A의 정답 코드 형식과 다릅니다.');
    return { width, height, png, mask, name: String(value.image.name || '내 이미지').slice(0, 160) };
  }
  function dataset(image, labels, name = 'image') {
    const result = Edges.calculate(image);
    if (!labels || labels.length !== image.width * image.height) throw new Error('이미지와 정답의 크기가 일치하지 않습니다.');
    let positive = 0, negative = 0;
    for (let i = 0; i < labels.length; i++) {
      if (![0, 1, 2].includes(labels[i])) throw new Error('정답 코드는 미지정 0 / 윤곽 1 / 비윤곽 2여야 합니다.');
      if (!result.valid[i] || !result.counts[i]) continue;
      if (labels[i] === 1) positive++;
      if (labels[i] === 2) negative++;
    }
    return { image, labels: Uint8Array.from(labels), result, name, key: fingerprint(image), positive, negative };
  }
  // Reservoir sampling bounds memory without treating unlabelled pixels as negatives.
  function balancedRows(datasets, perClass = 2000, seed = 42) {
    const rng = random(seed), buckets = [[], []], seen = [0, 0];
    for (const d of datasets) for (let i = 0; i < d.labels.length; i++) {
      if (!d.result.valid[i] || !d.result.counts[i] || !d.labels[i]) continue;
      const y = d.labels[i] === 1 ? 1 : 0, n = ++seen[y];
      const position = n <= perClass ? n - 1 : Math.floor(rng() * n);
      if (position < perClass) buckets[y][position] = { x: vector(d.image, d.result, i), y };
    }
    const count = Math.min(buckets[0].length, buckets[1].length);
    if (count < 2) throw new Error('학습용으로 윤곽과 비윤곽 정답을 각각 2개 이상 표시해 주세요.');
    return shuffle([...shuffle(buckets[0], rng).slice(0, count), ...shuffle(buckets[1], rng).slice(0, count)], rng);
  }
  function checkSplit(training, evaluation) {
    if (!training.length) throw new Error('학습에 사용할 자료를 선택해 주세요.');
    const keys = new Set(training.map(d => d.key));
    if (evaluation.some(d => keys.has(d.key))) throw new Error('같은 원본 이미지가 학습과 평가에 동시에 들어 있습니다. 용도를 하나로 정해 주세요.');
  }
  function scaler(rows) {
    const mean = Array(INPUTS).fill(0), std = Array(INPUTS).fill(0);
    for (const r of rows) for (let j = 0; j < INPUTS; j++) mean[j] += r.x[j] / rows.length;
    for (const r of rows) for (let j = 0; j < INPUTS; j++) std[j] += (r.x[j] - mean[j]) ** 2 / rows.length;
    for (let j = 0; j < INPUTS; j++) std[j] = Math.max(Math.sqrt(std[j]), .001);
    return { mean, std };
  }
  function normalized(model, x) {
    return x.map((v, j) => Math.max(-8, Math.min(8, (v - model.scaler.mean[j]) / model.scaler.std[j])));
  }
  function createModel(rows, seed = 42) {
    const rng = random(seed), weight = n => Array.from({ length: n }, () => (rng() * 2 - 1) * .4);
    return { schema: MODEL_SCHEMA, features: [...FEATURES], inputs: INPUTS, hidden: HIDDEN,
      activation: 'gelu_tanh_approximation', scaler: scaler(rows), w1: weight(INPUTS * HIDDEN),
      b1: Array(HIDDEN).fill(0), w2: weight(HIDDEN), b2: [0] };
  }
  function forward(model, input) {
    const x = normalized(model, input), z = [], a = [];
    let logit = model.b2[0];
    for (let h = 0; h < HIDDEN; h++) {
      let v = model.b1[h];
      for (let j = 0; j < INPUTS; j++) v += model.w1[h * INPUTS + j] * x[j];
      z.push(v); a.push(gelu(v)); logit += model.w2[h] * a[h];
    }
    return { x, z, a, logit, p: sigmoid(logit) };
  }
  function loss(model, row) {
    const z = forward(model, row.x).logit;
    return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - row.y * z;
  }
  function gradients(model, row) {
    const f = forward(model, row.x), delta = f.p - row.y;
    const g = { w1: Array(INPUTS * HIDDEN).fill(0), b1: Array(HIDDEN).fill(0), w2: Array(HIDDEN).fill(0), b2: [delta] };
    for (let h = 0; h < HIDDEN; h++) {
      g.w2[h] = delta * f.a[h];
      const hiddenDelta = delta * model.w2[h] * geluPrime(f.z[h]);
      g.b1[h] = hiddenDelta;
      for (let j = 0; j < INPUTS; j++) g.w1[h * INPUTS + j] = hiddenDelta * f.x[j];
    }
    return g;
  }
  function optimizer(model) {
    const moment = {}, variance = {};
    for (const key of ['w1', 'b1', 'w2', 'b2']) { moment[key] = model[key].map(() => 0); variance[key] = model[key].map(() => 0); }
    return { moment, variance, step: 0 };
  }
  function trainBatch(model, rows, opt, rate = .01) {
    const keys = ['w1', 'b1', 'w2', 'b2'], sum = {};
    for (const key of keys) sum[key] = model[key].map(() => 0);
    for (const row of rows) {
      const g = gradients(model, row);
      for (const key of keys) for (let i = 0; i < g[key].length; i++) sum[key][i] += g[key][i] / rows.length;
    }
    opt.step++;
    for (const key of keys) for (let i = 0; i < sum[key].length; i++) {
      const g = Math.max(-5, Math.min(5, sum[key][i]));
      opt.moment[key][i] = .9 * opt.moment[key][i] + .1 * g;
      opt.variance[key][i] = .999 * opt.variance[key][i] + .001 * g * g;
      model[key][i] -= rate * (opt.moment[key][i] / (1 - .9 ** opt.step)) /
        (Math.sqrt(opt.variance[key][i] / (1 - .999 ** opt.step)) + 1e-8);
      if (!Number.isFinite(model[key][i])) throw new Error('학습값이 유효한 범위를 벗어났습니다. 학습률을 낮춰 다시 시작해 주세요.');
    }
  }
  function evaluate(model, rows, cutoff = .5) {
    let tp = 0, fp = 0, fn = 0, tn = 0, totalLoss = 0;
    for (const r of rows) {
      const yes = forward(model, r.x).p >= cutoff; totalLoss += loss(model, r);
      if (r.y) yes ? tp++ : fn++; else yes ? fp++ : tn++;
    }
    return metricCounts(tp, fp, fn, tn, rows.length ? totalLoss / rows.length : null);
  }
  function metricCounts(tp, fp, fn, tn, averageLoss = null) {
    return { tp, fp, fn, tn, evaluated: tp + fp + fn + tn, loss: averageLoss,
      precision: tp + fp ? tp / (tp + fp) : null, recall: tp + fn ? tp / (tp + fn) : null,
      f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : null };
  }
  function loadModel(value) {
    const m = JSON.parse(JSON.stringify(value));
    if (m.schema !== MODEL_SCHEMA || m.inputs !== INPUTS || m.hidden !== HIDDEN ||
        m.activation !== 'gelu_tanh_approximation' || JSON.stringify(m.features) !== JSON.stringify(FEATURES)) {
      throw new Error('이 예제에서 저장한 7 → 12 → 1 MLP 모델 JSON을 선택해 주세요.');
    }
    for (const [key, length] of Object.entries({ w1: INPUTS * HIDDEN, b1: HIDDEN, w2: HIDDEN, b2: 1 })) {
      if (!Array.isArray(m[key]) || m[key].length !== length || !m[key].every(v => Number.isFinite(v) && Math.abs(v) < 10000)) throw new Error('모델의 가중치 형식이 올바르지 않습니다.');
    }
    if (!m.scaler || !['mean', 'std'].every(k => Array.isArray(m.scaler[k]) && m.scaler[k].length === INPUTS &&
      m.scaler[k].every(v => Number.isFinite(v) && (k === 'std' ? v >= .001 && v <= 100 : Math.abs(v) <= 100)))) throw new Error('모델의 입력 변환 정보가 올바르지 않습니다.');
    return m;
  }
  // Geometric object masks, rather than a score threshold, define synthetic ground truth.
  function demoImage(variant = 0) {
    const width = 96, height = 64, data = new Uint8ClampedArray(width * height * 4), object = new Uint8Array(width * height);
    const rng = random(100 + variant);
    const background = [40 + variant * 3, 55 + variant * 2, 50 + variant * 2];
    const rectangle = [190 + variant * 3, 135 + variant * 2, 88 + variant * 2], circle = [95 + variant * 2, 167 + variant * 2, 194 + variant * 2];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const rect = x >= 10 + variant && x < 39 + variant && y >= 10 && y < 51 - variant;
      const round = (x - 70 + variant) ** 2 + (y - 32) ** 2 < (17 + variant) ** 2;
      object[i] = rect ? 1 : round ? 2 : 0;
      const color = rect ? rectangle : round ? circle : background;
      const texture = rect && y % 9 < 2 ? -15 : 0, noise = (rng() - .5) * 4;
      for (let c = 0; c < 3; c++) data[4 * i + c] = Math.round(color[c] + texture + noise);
      data[4 * i + 3] = 255;
    }
    const labels = new Uint8Array(width * height).fill(2);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = y * width + x;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < width && ny < height && object[ny * width + nx] !== object[i]) labels[i] = 1;
      }
    }
    const d = dataset({ width, height, data }, labels, `도형 연습 ${variant + 1}`);
    d.provenance = 'synthetic_geometry';
    return d;
  }
  const api = { INPUTS, HIDDEN, FEATURES, MODEL_SCHEMA, random, shuffle, gelu, geluPrime, sigmoid, vector,
    fingerprint, annotation, dataset, balancedRows, checkSplit, createModel, forward, loss, gradients, optimizer,
    trainBatch, evaluate, metricCounts, loadModel, demoImage };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.EdgeMLP = Object.freeze(api);
})(typeof globalThis !== 'undefined' ? globalThis : this);
