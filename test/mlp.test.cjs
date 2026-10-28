const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../mlp-core.js');
const E = require('../core.js');
const rgba = values => ({ width: values.length, height: 1, data: Uint8ClampedArray.from(values.flatMap(v => [v, v, v, 255])) });

test('features use the same RGB, valid-neighbor means and score as A', () => {
  const im = rgba([0, 100, 200]), r = E.calculate(im), x = M.vector(im, r, 1);
  assert.deepEqual(x.slice(0, 6), Array(6).fill(100 / 255)); assert.equal(x[6], 0);
  const bright = rgba([0, 255]), b = E.calculate(bright);
  assert.equal(M.vector(bright, b, 1)[6], 1);
});
test('A annotation imports use the original mask and ignore untrusted sample features', () => {
  const value = { schema: 'rgb-neighbor-edge/v1', image: { width: 3, height: 1, name: 'A fixture', png_data_url: 'data:image/png;base64,aGVsbG8=' },
    labels: { provenance: 'manual_user_annotation', encoding: { unknown: 0, edge: 1, non_edge: 2 }, mask: [1, 0, 2] },
    samples: [{ label: 1, score: 999999999 }] };
  assert.deepEqual(M.annotation(value).mask, [1, 0, 2]);
  for (const mutation of [d => { d.schema = 'other'; }, d => { d.image.png_data_url = 'https://example.com/image.png'; },
    d => { d.labels.mask[1] = 3; }, d => { d.image.width = 4; }, d => { d.labels.encoding.unknown = 2; }, d => { d.labels.provenance = 'threshold'; }]) {
    const bad = JSON.parse(JSON.stringify(value)); mutation(bad); assert.throws(() => M.annotation(bad));
  }
});
test('unknown/transparent/isolated pixels are not converted into training negatives', () => {
  const image = rgba([0, 50, 100, 150, 200, 255]); image.data[23] = 0;
  const d = M.dataset(image, [1, 1, 2, 2, 0, 1]);
  assert.equal(d.positive, 2); assert.equal(d.negative, 2);
  assert.equal(M.balancedRows([d]).length, 4);
  assert.throws(() => M.balancedRows([M.dataset(rgba([255]), [1])]), /각각 2개/);
  assert.throws(() => M.dataset(image, [1]), /크기/);
  assert.throws(() => M.dataset(image, [1, 1, 2, 2, 3, 0]), /정답 코드/);
});
test('sampling is balanced, deterministic, bounded and does not change annotations', () => {
  const d = M.demoImage(0), before = [...d.labels], a = M.balancedRows([d], 100), b = M.balancedRows([d], 100);
  assert.equal(a.length, 200); assert.equal(a.filter(r => r.y).length, 100);
  assert.deepEqual(a, b); assert.deepEqual([...d.labels], before);
});
test('the same pixels under a different filename cannot be both training and evaluation', () => {
  const a = M.demoImage(0), b = M.dataset(a.image, a.labels, 'renamed.png');
  assert.throws(() => M.checkSplit([a], [b]), /동시에/);
  assert.doesNotThrow(() => M.checkSplit([a], [M.demoImage(3)]));
  assert.throws(() => M.checkSplit([], []), /선택/);
});
test('synthetic geometric labels distinguish texture from object boundaries', () => {
  const d = M.demoImage(0);
  assert.equal(d.labels[32 * 96 + 10], 1);
  assert.equal(d.labels[18 * 96 + 25], 2); // Rectangle stripe is not the object boundary.
  assert.equal(d.provenance, 'synthetic_geometry');
});
test('GELU derivative matches finite differences including negative inputs', () => {
  for (const x of [-4, -1.5, -.1, 0, .5, 2, 4]) {
    const delta = 1e-5, numerical = (M.gelu(x + delta) - M.gelu(x - delta)) / (2 * delta);
    assert.ok(Math.abs(numerical - M.geluPrime(x)) < 1e-7);
  }
});
test('backpropagation matches numerical loss derivatives for every weight and bias', () => {
  const rows = [{ x: [.2, .7, .1, .5, .4, .9, .1], y: 1 }, { x: [.8, .1, .3, .4, .7, .6, .8], y: 0 }];
  const model = M.createModel(rows), row = rows[0], analytic = M.gradients(model, row), epsilon = 1e-5;
  for (const key of ['w1', 'b1', 'w2', 'b2']) for (let i = 0; i < model[key].length; i++) {
    const original = model[key][i]; model[key][i] = original + epsilon; const plus = M.loss(model, row);
    model[key][i] = original - epsilon; const minus = M.loss(model, row); model[key][i] = original;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - analytic[key][i]) < 1e-7, `${key}[${i}]`);
  }
});
test('the nonlinear network actually learns XOR and changes its weights', () => {
  const rows = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].map(([a, b, y]) => ({ x: [a, b, 0, 0, 0, 0, 0], y }));
  const model = M.createModel(rows), before = M.evaluate(model, rows).loss, original = [...model.w1], opt = M.optimizer(model);
  for (let i = 0; i < 500; i++) M.trainBatch(model, rows, opt);
  const after = M.evaluate(model, rows);
  assert.ok(after.loss < before / 10); assert.equal(after.f1, 1); assert.notDeepEqual(model.w1, original);
});
test('model roundtrip preserves predictions and rejects malformed/nonfinite models', () => {
  const rows = M.balancedRows([M.demoImage(0)], 10), model = M.createModel(rows), x = rows[0].x;
  assert.equal(M.forward(model, x).p, M.forward(M.loadModel(JSON.parse(JSON.stringify(model))), x).p);
  for (const mutation of [m => { m.inputs = 8; }, m => { m.w1.pop(); }, m => { m.w2[0] = Infinity; }, m => { m.scaler.std[0] = 0; }, m => { m.features.reverse(); }]) {
    const broken = JSON.parse(JSON.stringify(model)); mutation(broken); assert.throws(() => M.loadModel(broken));
  }
});
test('undefined classification metrics are null and saturated logits have finite loss', () => {
  assert.equal(M.metricCounts(0, 0, 0, 0).f1, null);
  assert.equal(M.metricCounts(0, 0, 1, 1).precision, null);
  const rows = [{ x: Array(7).fill(0), y: 1 }, { x: Array(7).fill(1), y: 0 }], model = M.createModel(rows);
  for (const value of [-1000, 1000]) { model.b2[0] = value; assert.ok(Number.isFinite(M.loss(model, rows[0]))); }
});
