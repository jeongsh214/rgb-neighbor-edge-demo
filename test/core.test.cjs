const test = require('node:test');
const assert = require('node:assert/strict');
const { calculate, threshold, metrics, samples, MAX_SCORE } = require('../core.js');

function image(width, height, pixels) {
  return { width, height, data: Uint8ClampedArray.from(pixels.flatMap(p => typeof p === 'number' ? [p, p, p, 255] : p)) };
}

test('uniform RGB has zero scores, including valid black pixels', () => {
  const r = calculate(image(3, 3, Array(9).fill(0)));
  assert.ok(r.scores.every(x => x === 0));
  assert.equal(r.valid[4], 1); assert.equal(r.counts[4], 8);
  assert.equal(r.counts[0], 3);
  assert.ok(threshold(r, 0).every(x => x === 0));
});
test('center is excluded and all eight black neighbors count', () => {
  const r = calculate(image(3, 3, [0, 0, 0, 0, 255, 0, 0, 0, 0]));
  assert.equal(r.scores[4], MAX_SCORE); assert.equal(r.counts[4], 8);
  assert.deepEqual(Array.from(r.means.slice(12, 15)), [0, 0, 0]);
});
test('corners divide by actual neighbor count, not eight', () => {
  const r = calculate(image(2, 2, [0, 30, 60, 90]));
  assert.equal(r.counts[0], 3); assert.equal(r.means[0], 60);
  assert.equal(r.scores[0], 3 * 60 ** 2);
});
test('bright and dark sides both receive positive scores', () => {
  const r = calculate(image(2, 1, [0, 255]));
  assert.deepEqual(Array.from(r.scores), [MAX_SCORE, MAX_SCORE]);
});
test('channel means remain separate', () => {
  const r = calculate(image(3, 1, [[0, 30, 60, 255], [100, 60, 0, 255], [200, 90, 0, 255]]));
  assert.deepEqual(Array.from(r.means.slice(3, 6)), [100, 60, 30]);
  assert.equal(r.scores[1], 900);
});
test('mean cancellation produces zero despite different individual neighbors', () => {
  const r = calculate(image(3, 3, [0, 0, 0, 0, 100, 200, 200, 200, 200]));
  assert.equal(r.counts[4], 8); assert.equal(r.scores[4], 0);
});
test('transparent neighbors and centers are excluded; isolated center is not a candidate', () => {
  const r = calculate(image(2, 1, [[80, 80, 80, 255], [255, 255, 255, 0]]));
  assert.deepEqual(Array.from(r.counts), [0, 0]);
  assert.deepEqual(Array.from(r.valid), [1, 0]);
  assert.ok(r.scores.every(Number.isFinite));
  assert.deepEqual(Array.from(threshold(r, 0)), [0, 0]);
});
test('single pixel and fully transparent image have no samples', () => {
  for (const im of [image(1, 1, [255]), image(1, 1, [[255, 255, 255, 0]])]) {
    const r = calculate(im);
    assert.deepEqual(samples(im, r, Uint8Array.of(1)), []);
  }
});
test('threshold uses strict greater-than and validates numeric range', () => {
  const r = calculate(image(2, 1, [0, 10]));
  assert.deepEqual(Array.from(threshold(r, 300)), [0, 0]);
  assert.deepEqual(Array.from(threshold(r, 299)), [1, 1]);
  for (const v of [-1, NaN, Infinity, MAX_SCORE + 1]) assert.throws(() => threshold(r, v), RangeError);
});
test('evaluation ignores unknown labels and counts both classes', () => {
  const r = calculate(image(5, 1, [0, 0, 0, 0, 0]));
  const m = metrics(r, Uint8Array.of(1, 1, 0, 0, 1), Uint8Array.of(1, 2, 1, 2, 0));
  assert.deepEqual(m, { tp: 1, fp: 1, fn: 1, tn: 1, precision: .5, recall: .5, f1: .5, evaluated: 4 });
});
test('undefined metrics are null, not misleading perfect scores', () => {
  const r = calculate(image(2, 1, [0, 0]));
  assert.equal(metrics(r, threshold(r, 0), new Uint8Array(2)).precision, null);
  assert.equal(metrics(r, threshold(r, 0), new Uint8Array(2)).f1, null);
});
test('export samples preserve manual labels and exclude unspecified pixels', () => {
  const im = image(3, 1, [0, 100, 200]), r = calculate(im);
  const labels = Uint8Array.of(1, 0, 2);
  const before = Array.from(labels); threshold(r, 0);
  const rows = samples(im, r, labels);
  assert.deepEqual(Array.from(labels), before);
  assert.deepEqual(rows.map(row => [row.x, row.y, row.label]), [[0, 0, 1], [2, 0, 0]]);
  assert.deepEqual(rows[0].rgb, [0, 0, 0]);
  assert.equal(rows[0].neighbor_count, 1);
});
test('invalid dimensions and mismatched arrays fail explicitly', () => {
  assert.throws(() => calculate({ width: 2, height: 2, data: [] }), TypeError);
  assert.throws(() => calculate({ width: 0, height: 1, data: [] }), TypeError);
  const r = calculate(image(2, 1, [0, 0]));
  assert.throws(() => metrics(r, Uint8Array.of(0), Uint8Array.of(1)), RangeError);
});
