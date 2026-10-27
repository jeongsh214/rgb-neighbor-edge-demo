/* global RgbEdges */
'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const source = $('source'), heatmap = $('heatmap'), predictionCanvas = $('prediction');
  const original = document.createElement('canvas');
  let image, result, prediction, labels, sourceName, originalSize, cutoff = 1200;
  let mode = 'inspect', brush = 2, drawing = false, lastPoint = null, paintFrame = null;
  const number = (x) => Number(x.toFixed(3)).toLocaleString('ko-KR');
  const percent = (x) => x === null ? '—' : `${(x * 100).toFixed(1)}%`;

  function notice(text) { $('notice').textContent = text; }

  function setImage(canvas, name, size = { width: canvas.width, height: canvas.height }) {
    original.width = canvas.width; original.height = canvas.height;
    original.getContext('2d').drawImage(canvas, 0, 0);
    image = original.getContext('2d').getImageData(0, 0, original.width, original.height);
    sourceName = name; originalSize = size;
    for (const c of [source, heatmap, predictionCanvas]) {
      c.width = image.width; c.height = image.height;
      c.style.aspectRatio = `${image.width} / ${image.height}`;
    }
    labels = new Uint8Array(image.width * image.height);
    result = RgbEdges.calculate(image);
    $('image-info').textContent = `${name} · ${image.width} × ${image.height} 픽셀`;
    $('export-status').textContent = '';
    renderHeatmap(); renderPrediction(); renderLabels();
    inspect(Math.floor(image.width / 2), Math.floor(image.height / 2));
  }

  function demo() {
    const c = document.createElement('canvas'); c.width = 320; c.height = 200;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#303c38'; ctx.fillRect(0, 0, 320, 200);
    ctx.fillStyle = '#d69b6b'; ctx.fillRect(40, 40, 105, 120);
    ctx.fillStyle = '#79b6d3'; ctx.beginPath(); ctx.arc(235, 100, 51, 0, 2 * Math.PI); ctx.fill();
    // Deliberate surface texture and a bright speck: candidates need not be object contours.
    ctx.fillStyle = '#c08659';
    for (let y = 55; y < 151; y += 13) ctx.fillRect(53, y, 78, 3);
    ctx.fillStyle = 'white'; ctx.fillRect(180, 30, 2, 2);
    setImage(c, '도형과 무늬 예제');
    notice('도형 경계와 표면 무늬, 작은 밝은 점을 함께 비교해 보세요. 이미지는 브라우저 밖으로 전송되지 않습니다.');
  }

  function cancellation() {
    const c = document.createElement('canvas'); c.width = 3; c.height = 3;
    const data = c.getContext('2d').createImageData(3, 3);
    const values = [0, 0, 0, 0, 100, 200, 200, 200, 200];
    values.forEach((v, i) => data.data.set([v, v, v, 255], i * 4));
    c.getContext('2d').putImageData(data, 0, 0);
    setImage(c, '평균 상쇄 3×3 예제'); inspect(1, 1);
    notice('중앙은 RGB(100,100,100), 이웃은 검정 4개와 RGB(200,200,200) 4개입니다. 이웃 평균이 중심과 같아서 중앙 점수는 0입니다.');
  }

  function renderHeatmap() {
    const ctx = heatmap.getContext('2d');
    const pixels = ctx.createImageData(image.width, image.height);
    let max = 0;
    for (const score of result.scores) max = Math.max(max, score);
    for (let i = 0; i < result.scores.length; i++) {
      const t = max ? Math.sqrt(result.scores[i] / max) : 0;
      pixels.data.set([Math.round(22 + t * 233), Math.round(36 + t * 207), Math.round(32 + t * 121), result.valid[i] ? 255 : 0], i * 4);
    }
    ctx.putImageData(pixels, 0, 0);
  }

  function renderPrediction() {
    prediction = RgbEdges.threshold(result, cutoff);
    const ctx = predictionCanvas.getContext('2d');
    const pixels = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < prediction.length; i++) {
      const value = prediction[i] * 255;
      pixels.data.set([value, value, value, result.valid[i] ? 255 : 0], i * 4);
    }
    ctx.putImageData(pixels, 0, 0); updateEvaluation();
  }

  function renderLabels() {
    const ctx = source.getContext('2d'); ctx.clearRect(0, 0, source.width, source.height);
    ctx.drawImage(original, 0, 0);
    const overlay = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === 1) overlay.data.set([45, 127, 255, 180], 4 * i);
      if (labels[i] === 2) overlay.data.set([255, 203, 45, 180], 4 * i);
    }
    const layer = document.createElement('canvas'); layer.width = image.width; layer.height = image.height;
    layer.getContext('2d').putImageData(overlay, 0, 0); ctx.drawImage(layer, 0, 0);
    updateEvaluation();
  }

  function updateEvaluation() {
    if (!result || !prediction || !labels) return;
    const m = RgbEdges.metrics(result, prediction, labels);
    const edges = labels.reduce((n, v) => n + (v === 1), 0);
    const nonedges = labels.reduce((n, v) => n + (v === 2), 0);
    $('evaluation').innerHTML = `<div class="stats"><div>정밀도<strong>${percent(m.precision)}</strong></div><div>재현율<strong>${percent(m.recall)}</strong></div><div>F1<strong>${percent(m.f1)}</strong></div></div><p class="evaluation-counts">표시한 윤곽 ${edges}개 · 비윤곽 ${nonedges}개<br>평가한 픽셀 ${m.evaluated}개 · 맞춘 윤곽 ${m.tp} · 잘못 선택 ${m.fp} · 놓친 윤곽 ${m.fn}</p>`;
  }

  function inspect(x, y) {
    const i = y * image.width + x;
    const rgb = Array.from(image.data.slice(i * 4, i * 4 + 3));
    const mean = Array.from(result.means.slice(i * 3, i * 3 + 3));
    const parts = rgb.map((v, c) => (v - mean[c]) ** 2);
    $('pixel-detail').textContent = !result.valid[i] ? `(${x}, ${y}) · 완전 투명한 픽셀: 제외` :
      result.counts[i] === 0 ? `(${x}, ${y}) · 유효한 이웃이 없어 후보·평가·학습 샘플에서 제외` :
      `좌표 (${x}, ${y}) · 유효한 이웃 ${result.counts[i]}개\nRGB       [${rgb.join(', ')}]\n주변 평균 [${mean.map(number).join(', ')}]\n차이 제곱 [${parts.map(number).join(', ')}]\n점수 ${number(result.scores[i])} · ${prediction[i] ? '후보' : '후보 아님'}`;
  }

  function point(event) {
    const rect = source.getBoundingClientRect();
    return [Math.max(0, Math.min(image.width - 1, Math.floor((event.clientX - rect.left) * image.width / rect.width))),
      Math.max(0, Math.min(image.height - 1, Math.floor((event.clientY - rect.top) * image.height / rect.height)))];
  }

  function paintAt(x, y) {
    const code = mode === 'edge' ? 1 : mode === 'nonedge' ? 2 : 0;
    for (let dy = -brush; dy <= brush; dy++) for (let dx = -brush; dx <= brush; dx++) {
      if (dx * dx + dy * dy > brush * brush) continue;
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= image.width || ny >= image.height) continue;
      const i = ny * image.width + nx;
      if (result.valid[i] && result.counts[i] > 0) labels[i] = code;
    }
  }

  function move(event) {
    if (!image) return;
    const [x, y] = point(event); inspect(x, y);
    if (!drawing || mode === 'inspect') return;
    const start = lastPoint || [x, y];
    const steps = Math.max(1, Math.ceil(Math.hypot(x - start[0], y - start[1])));
    for (let t = 0; t <= steps; t++) paintAt(Math.round(start[0] + (x - start[0]) * t / steps), Math.round(start[1] + (y - start[1]) * t / steps));
    lastPoint = [x, y];
    if (!paintFrame) paintFrame = requestAnimationFrame(() => { paintFrame = null; renderLabels(); });
  }

  function download(blob, name) {
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  function exportPng(canvas, filename) { canvas.toBlob((blob) => { if (blob) download(blob, filename); }, 'image/png'); }

  $('upload').addEventListener('change', async (event) => {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 30 * 1024 * 1024) { notice('30MB 이하의 이미지를 선택해 주세요.'); event.target.value = ''; return; }
    try {
      const bitmap = await createImageBitmap(file);
      const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
      const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(bitmap.width * scale)); c.height = Math.max(1, Math.round(bitmap.height * scale));
      c.getContext('2d').drawImage(bitmap, 0, 0, c.width, c.height);
      const size = { width: bitmap.width, height: bitmap.height }; bitmap.close();
      setImage(c, file.name, size);
      notice(scale < 1 ? `원본 ${size.width}×${size.height}를 ${c.width}×${c.height}로 축소했습니다. 좌표·정답·내보내기는 축소된 이미지를 기준으로 합니다.` : '선택한 이미지는 브라우저 안에서만 처리되며 서버로 전송되지 않습니다.');
    } catch { notice('이미지를 읽을 수 없습니다. PNG, JPEG, WebP 또는 BMP 파일을 선택해 주세요.'); }
    event.target.value = '';
  });
  $('demo').addEventListener('click', demo); $('cancellation').addEventListener('click', cancellation);
  function setThreshold(value) {
    const n = Number(value); if (!Number.isFinite(n)) return;
    cutoff = Math.max(0, Math.min(RgbEdges.MAX_SCORE, Math.round(n)));
    $('threshold').value = String(cutoff); $('threshold-number').value = String(cutoff);
    $('threshold-readout').textContent = String(cutoff); renderPrediction();
  }
  $('threshold').addEventListener('input', (e) => setThreshold(e.target.value));
  $('threshold-number').addEventListener('change', (e) => setThreshold(e.target.value));
  document.querySelectorAll('input[name=mode]').forEach((radio) => radio.addEventListener('change', () => { mode = radio.value; }));
  $('brush').addEventListener('input', (e) => { brush = Number(e.target.value); $('brush-readout').textContent = brush; });
  $('clear-labels').addEventListener('click', () => { labels.fill(0); renderLabels(); $('export-status').textContent = ''; });
  source.addEventListener('pointerdown', (event) => { if (event.button !== 0) return; drawing = true; lastPoint = null; source.setPointerCapture(event.pointerId); move(event); });
  source.addEventListener('pointermove', move);
  source.addEventListener('pointerup', () => { drawing = false; lastPoint = null; });
  source.addEventListener('pointercancel', () => { drawing = false; lastPoint = null; });
  $('export-mask').addEventListener('click', () => exportPng(predictionCanvas, 'edge-candidates.png'));
  $('export-labels').addEventListener('click', () => {
    const c = document.createElement('canvas'); c.width = image.width; c.height = image.height;
    const ctx = c.getContext('2d'), pixels = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i] === 1 ? 255 : labels[i] === 2 ? 0 : 128;
      pixels.data.set([v, v, v, result.valid[i] && result.counts[i] ? 255 : 0], i * 4);
    }
    ctx.putImageData(pixels, 0, 0); exportPng(c, 'reviewed-labels.png');
  });
  $('export-json').addEventListener('click', () => {
    const rows = RgbEdges.samples(image, result, labels);
    if (!rows.length) { $('export-status').textContent = '먼저 원본 위에 윤곽 또는 비윤곽 정답을 표시해 주세요.'; return; }
    const dataset = {
      schema: 'rgb-neighbor-edge/v1',
      image: { name: sourceName, width: image.width, height: image.height, original_width: originalSize.width, original_height: originalSize.height, png_data_url: original.toDataURL('image/png') },
      config: { formula: RgbEdges.FORMULA, neighborhood: '3x3 excluding center', boundary: 'exclude outside and divide by actual neighbor count', transparency: 'exclude alpha=0; include alpha>0 and black RGB', threshold: cutoff, threshold_role: 'display only; never generates labels' },
      labels: { provenance: 'manual_user_annotation', encoding: { unknown: 0, edge: 1, non_edge: 2 }, mask: Array.from(labels) },
      samples: rows,
    };
    download(new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' }), 'reviewed-edge-dataset.json');
    $('export-status').textContent = `${rows.length}개의 직접 표시한 샘플을 저장했습니다.`;
  });
  demo();
})();
