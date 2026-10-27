(function (root) {
  'use strict';

  const MAX_SCORE = 3 * 255 * 255;
  const FORMULA = '(R-mean_R)^2 + (G-mean_G)^2 + (B-mean_B)^2';

  function calculate(image) {
    const { width, height, data } = image;
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
        width < 1 || height < 1 || !data || data.length !== width * height * 4) {
      throw new TypeError('A nonempty RGBA image with matching dimensions is required.');
    }
    const scores = new Float64Array(width * height);
    const means = new Float64Array(width * height * 3);
    const counts = new Uint8Array(width * height);
    const valid = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (data[4 * i + 3] === 0) continue;
        valid[i] = 1;
        let red = 0, green = 0, blue = 0, count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const j = 4 * (ny * width + nx);
            if (data[j + 3] === 0) continue;
            red += data[j]; green += data[j + 1]; blue += data[j + 2]; count++;
          }
        }
        counts[i] = count;
        if (count === 0) continue; // No valid neighbor: zero score; not a candidate/sample.
        const average = [red / count, green / count, blue / count];
        let score = 0;
        for (let channel = 0; channel < 3; channel++) {
          means[3 * i + channel] = average[channel];
          const difference = data[4 * i + channel] - average[channel];
          score += difference * difference;
        }
        scores[i] = score;
      }
    }
    return { width, height, scores, means, counts, valid };
  }

  function threshold(result, value) {
    if (!Number.isFinite(value) || value < 0 || value > MAX_SCORE) {
      throw new RangeError(`Threshold must be between 0 and ${MAX_SCORE}.`);
    }
    return Uint8Array.from(result.scores, (score, i) =>
      result.valid[i] && result.counts[i] > 0 && score > value ? 1 : 0);
  }

  function metrics(result, prediction, labels) {
    if (prediction.length !== result.scores.length || labels.length !== prediction.length) {
      throw new RangeError('Prediction, labels and image dimensions must match.');
    }
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < labels.length; i++) {
      if (!result.valid[i] || !result.counts[i] || labels[i] === 0) continue;
      if (labels[i] !== 1 && labels[i] !== 2) throw new RangeError('Unknown label code.');
      if (labels[i] === 1) prediction[i] ? tp++ : fn++;
      else prediction[i] ? fp++ : tn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : null;
    const recall = tp + fn ? tp / (tp + fn) : null;
    const f1 = 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : null;
    return { tp, fp, fn, tn, precision, recall, f1, evaluated: tp + fp + fn + tn };
  }

  function samples(image, result, labels) {
    if (labels.length !== result.scores.length || image.data.length !== labels.length * 4) {
      throw new RangeError('Labels and image dimensions must match.');
    }
    const rows = [];
    for (let i = 0; i < labels.length; i++) {
      if (!result.valid[i] || !result.counts[i] || labels[i] === 0) continue;
      if (labels[i] !== 1 && labels[i] !== 2) throw new RangeError('Unknown label code.');
      rows.push({
        x: i % result.width, y: Math.floor(i / result.width),
        rgb: Array.from(image.data.slice(4 * i, 4 * i + 3)),
        neighbor_mean: Array.from(result.means.slice(3 * i, 3 * i + 3)),
        neighbor_count: result.counts[i], score: result.scores[i],
        label: labels[i] === 1 ? 1 : 0,
      });
    }
    return rows;
  }

  const api = { MAX_SCORE, FORMULA, calculate, threshold, metrics, samples };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RgbEdges = Object.freeze(api);
})(typeof globalThis !== 'undefined' ? globalThis : this);
