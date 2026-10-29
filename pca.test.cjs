const test = require('node:test');
const assert = require('node:assert/strict');
const P = require('./pca-core.js');
const near = (a, b, eps = 1e-7) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const points = [[-2, -1], [-1, -2], [1, 2], [2, 1]];

test('report example has eigenvalues 18/2 and projected coordinates ±3/sqrt(2)', () => {
  const m = P.fit(points);
  near(m.mean[0], 0); near(m.mean[1], 0);
  near(m.eigenvalues[0], 18); near(m.eigenvalues[1], 2);
  near(m.components[0][0], 1 / Math.SQRT2); near(m.components[0][1], 1 / Math.SQRT2);
  near(P.transform(m, points[0], 1)[0], -3 / Math.SQRT2);
  near(P.transform(m, points[1], 1)[0], -3 / Math.SQRT2);
  near(P.retained(m, 1), .9);
  near(P.reconstruct(m, P.transform(m, points[0], 1))[0], -1.5);
});
test('translation changes mean but not principal directions or variation', () => {
  const m = P.fit(points.map(p => [p[0] + 30, p[1] - 20]));
  near(m.mean[0], 30); near(m.mean[1], -20);
  near(m.eigenvalues[0], 18); near(m.components[0][0], 1 / Math.SQRT2);
  const rec = P.reconstruct(m, P.transform(m, [28, -21])); near(rec[0], 28); near(rec[1], -21);
});
test('zero components reconstruct only the mean; all components reconstruct centered training rows', () => {
  const m = P.fit(points);
  assert.deepEqual(P.reconstruct(m, []), m.mean); near(P.retained(m, 0), 0);
  for (const p of points) { const r = P.reconstruct(m, P.transform(m, p)); near(r[0], p[0]); near(r[1], p[1]); }
});
test('dual eigensystem satisfies X^T X u = lambda u and orthonormality in original feature space', () => {
  const rows = Array.from({ length: 7 }, (_, i) => Array.from({ length: 19 }, (_, j) => Math.sin((i + 1) * (j + 1)) + i / 7));
  const m = P.fit(rows), x = rows.map(row => row.map((v, j) => v - m.mean[j]));
  assert.equal(m.method, 'XXt'); assert.ok(m.components.length <= rows.length - 1);
  m.components.forEach((u, i) => {
    m.components.forEach((v, j) => near(P.dot(u, v), +(i === j), 1e-6));
    const scores = x.map(row => P.dot(row, u));
    u.forEach((v, j) => near(x.reduce((s, row, t) => s + row[j] * scores[t], 0), m.eigenvalues[i] * v, 1e-6));
  });
  for (const row of rows) near(P.rmse(row, P.reconstruct(m, P.transform(m, row))), 0, 1e-6);
});
test('reconstruction squared error equals discarded eigenvalue sum and never increases with k', () => {
  const rows = Array.from({ length: 9 }, (_, i) => Array.from({ length: 20 }, (_, j) => Math.cos(i * j + i / 3)));
  const m = P.fit(rows); let previous = Infinity;
  for (let k = 0; k <= m.components.length; k++) {
    const error = rows.reduce((s, row) => s + P.rmse(row, P.reconstruct(m, P.transform(m, row, k))) ** 2 * row.length, 0);
    near(error, m.totalEnergy * (1 - P.retained(m, k)), 1e-6); assert.ok(error <= previous + 1e-6); previous = error;
  }
});
test('identical inputs have no artificial component and an undefined variation ratio', () => {
  const m = P.fit([[4, 4, 4], [4, 4, 4], [4, 4, 4]]);
  assert.equal(m.components.length, 0); assert.equal(P.retained(m, 0), null);
  assert.deepEqual(P.reconstruct(m, []), [4, 4, 4]);
});
test('new images use fitted mean/directions without changing the model', () => {
  const m = P.fit([[0, 0, 3], [2, 0, 3]]), before = JSON.stringify(m);
  const rec = P.reconstruct(m, P.transform(m, [1, 5, 3]));
  near(rec[0], 1); near(rec[1], 0); near(rec[2], 3); assert.equal(JSON.stringify(m), before);
});
test('tiny rescaled data retain their principal directions', () => {
  const m = P.fit(points.map(row => row.map(x => x * 1e-8)));
  assert.equal(m.components.length, 2); near(P.retained(m, 1), .9);
});
test('invalid samples, projection counts and nonfinite values are rejected', () => {
  for (const rows of [[], [[1]], [[1], [1, 2]], [[1], [NaN]], [[1], [Infinity]]]) assert.throws(() => P.fit(rows));
  const m = P.fit(points);
  for (const k of [-1, 1.5, 3, NaN]) assert.throws(() => P.transform(m, points[0], k));
  assert.throws(() => P.transform(m, [1])); assert.throws(() => P.reconstruct(m, [NaN]));
});
