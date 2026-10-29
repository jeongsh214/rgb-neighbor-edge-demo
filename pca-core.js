/* Centered PCA. No external dependencies; usable in a browser and Node. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PCALab = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  function validateRows(rows) {
    if (!Array.isArray(rows) || rows.length < 2 || !rows[0]?.length) throw new Error('같은 크기의 자료가 2개 이상 필요합니다.');
    const d = rows[0].length;
    if (rows.some(row => row.length !== d || Array.from(row).some(v => !Number.isFinite(v)))) throw new Error('자료의 크기와 숫자를 확인해 주세요.');
    return d;
  }
  function symmetricEigen(matrix) {
    const n = matrix.length;
    if (!n || matrix.some(r => r.length !== n || r.some(v => !Number.isFinite(v)))) throw new Error('정사각 실수 행렬이 필요합니다.');
    const a = matrix.map(r => Array.from(r));
    const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
    const scale = Math.max(...a.map(r => Math.max(...r.map(Math.abs))));
    const tolerance = scale * 1e-12;
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
      if (Math.abs(a[i][j] - a[j][i]) > tolerance) throw new Error('대칭 행렬이 필요합니다.');
    }
    let converged = scale === 0;
    for (let sweep = 0; sweep < 100 && !converged; sweep++) {
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) <= tolerance) continue;
        const tau = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
        const c = 1 / Math.hypot(1, t), s = t * c;
        a[p][p] -= t * apq; a[q][q] += t * apq; a[p][q] = a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          if (k !== p && k !== q) {
            const x = a[k][p], y = a[k][q];
            a[k][p] = a[p][k] = c * x - s * y;
            a[k][q] = a[q][k] = s * x + c * y;
          }
          const x = v[k][p], y = v[k][q];
          v[k][p] = c * x - s * y; v[k][q] = s * x + c * y;
        }
      }
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off = Math.max(off, Math.abs(a[p][q]));
      converged = off <= tolerance;
    }
    if (!converged) throw new Error('계산이 수렴하지 않았습니다. 다른 자료로 다시 시도해 주세요.');
    return Array.from({ length: n }, (_, i) => ({ value: a[i][i], vector: v.map(r => r[i]) })).sort((a, b) => b.value - a.value);
  }
  function fit(rows) {
    const d = validateRows(rows), n = rows.length;
    const mean = Array.from({ length: d }, (_, j) => rows.reduce((s, row) => s + row[j], 0) / n);
    const x = rows.map(row => Array.from(row, (a, j) => a - mean[j]));
    const totalEnergy = x.reduce((s, row) => s + dot(row, row), 0);
    const dual = d > n, m = dual ? n : d;
    const gram = Array.from({ length: m }, () => Array(m).fill(0));
    for (let i = 0; i < m; i++) for (let j = i; j < m; j++) {
      const value = dual ? dot(x[i], x[j]) : x.reduce((s, row) => s + row[i] * row[j], 0);
      gram[i][j] = gram[j][i] = value;
    }
    const eigen = symmetricEigen(gram), components = [], eigenvalues = [];
    const threshold = Math.max(0, eigen[0].value) * 1e-10;
    for (const pair of eigen) {
      if (pair.value <= threshold || components.length >= Math.min(n - 1, d)) continue;
      const direction = dual
        ? Array.from({ length: d }, (_, j) => x.reduce((s, row, i) => s + row[j] * pair.vector[i], 0) / Math.sqrt(pair.value))
        : pair.vector.slice();
      const norm = Math.sqrt(dot(direction, direction));
      for (let j = 0; j < d; j++) direction[j] /= norm;
      let pivot = 0;
      for (let j = 1; j < d; j++) if (Math.abs(direction[j]) > Math.abs(direction[pivot])) pivot = j;
      if (direction[pivot] < 0) for (let j = 0; j < d; j++) direction[j] *= -1;
      components.push(direction); eigenvalues.push(x.reduce((s, row) => s + dot(row, direction) ** 2, 0));
    }
    return { mean, components, eigenvalues, totalEnergy, sampleCount: n, featureCount: d, method: dual ? 'XXt' : 'XtX' };
  }
  function count(model, k) {
    if (!Number.isInteger(k) || k < 0 || k > model.components.length) throw new Error('주요 방향의 개수를 확인해 주세요.');
  }
  function transform(model, row, k = model.components.length) {
    count(model, k);
    if (row.length !== model.featureCount || Array.from(row).some(v => !Number.isFinite(v))) throw new Error('이미지 크기와 숫자를 확인해 주세요.');
    const centered = Array.from(row, (v, j) => v - model.mean[j]);
    return model.components.slice(0, k).map(u => dot(centered, u));
  }
  function reconstruct(model, scores) {
    count(model, scores.length);
    if (scores.some(v => !Number.isFinite(v))) throw new Error('사영 좌표를 확인해 주세요.');
    return model.mean.map((v, j) => v + scores.reduce((s, t, i) => s + t * model.components[i][j], 0));
  }
  function retained(model, k) {
    count(model, k);
    return model.totalEnergy > 0 ? Math.min(1, model.eigenvalues.slice(0, k).reduce((s, v) => s + v, 0) / model.totalEnergy) : null;
  }
  function rmse(a, b) { return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length); }
  return { fit, transform, reconstruct, retained, rmse, symmetricEigen, dot };
});
