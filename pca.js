/* UI for the PCA learning page. User images never leave the browser. */
(() => {
  'use strict';
  const P = PCALab, $ = id => document.getElementById(id), size = 32;
  let samples = [], model, selected = 0, query = null, queryName = '', busy = false;
  const f = (v, digits = 3) => (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits);
  function canvasFor(values) {
    const c = document.createElement('canvas'); c.width = c.height = size; paint(c, values); return c;
  }
  function paint(c, values, signed = false) {
    const ctx = c.getContext('2d'), data = ctx.createImageData(size, size);
    const scale = signed ? Math.max(...Array.from(values, Math.abs)) || 1 : 1;
    values.forEach((v, i) => {
      const brightness = signed ? 128 + 120 * v / scale : v;
      data.data[i * 4] = data.data[i * 4 + 1] = data.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(brightness)));
      data.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(data, 0, 0);
  }
  function demoFace(index) {
    const a = (index % 4 - 1.5) / 1.5, b = (Math.floor(index / 4) - 1), c = Math.sin(index * 1.7);
    const values = [];
    const gauss = (x, y, cx, cy, sx, sy) => Math.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2);
    for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
      const x = (i - 15.5) / 15.5, y = (j - 15.5) / 15.5;
      const width = .62 + .035 * b;
      const face = Math.max(0, Math.min(1, (1 - (x / width) ** 2 - ((y + .03) / .82) ** 2) * 16));
      let value = 40 + 7 * y + face * (158 + 23 * a * x + 9 * b);
      if (face > 0) {
        value -= face * 115 * gauss(x, y, -.25, -.16, .12 + .01 * b, .07 + .015 * c);
        value -= face * 115 * gauss(x, y, .25, -.16, .12 + .01 * b, .07 + .015 * c);
        value += face * 24 * gauss(x, y, -.27, -.18, .028, .025);
        value += face * 24 * gauss(x, y, .23, -.18, .028, .025);
        value -= face * 35 * gauss(x, y, .04, .08, .07, .17);
        const mouthY = .35 + .8 * c * x * x;
        value -= face * 80 * Math.exp(-((x / .24) ** 6) - ((y - mouthY) / .026) ** 2);
        value -= face * 110 * Math.max(0, Math.min(1, (-.48 + .15 * x + .035 * b - y) * 16));
      }
      values.push(Math.max(0, Math.min(255, Math.round(value))));
    }
    return values;
  }
  function gallery() {
    const host = $('face-gallery'); host.replaceChildren();
    samples.forEach((sample, i) => {
      const button = document.createElement('button'); button.className = 'face-choice'; button.type = 'button';
      button.setAttribute('aria-pressed', String(i === selected && !query));
      button.setAttribute('aria-label', `${sample.name} 선택`);
      const label = document.createElement('span'); label.textContent = sample.name;
      button.append(canvasFor(sample.values), label);
      button.addEventListener('click', () => { selected = i; query = null; queryName = ''; gallery(); render(); $('query-status').textContent = '분석에 사용한 이미지입니다. 새 이미지는 아래 버튼으로 따로 적용할 수 있습니다.'; });
      host.append(button);
    });
  }
  function analyze(nextSamples, source) {
    const nextModel = P.fit(nextSamples.map(s => s.values));
    samples = nextSamples; model = nextModel; selected = 0; query = null; queryName = '';
    gallery();
    $('sample-count').textContent = `${samples.length}장 · 32×32 흑백`;
    $('matrix-shape').textContent = `이미지 한 장은 1024개 밝기 값 → X는 ${samples.length}행 × 1024열. 각 열의 평균을 뺍니다.`;
    $('face-status').textContent = `${source}에서 주요 방향 ${model.components.length}개를 계산했습니다.`;
    $('component-count').max = String(model.components.length);
    $('component-count').value = String(Math.min(2, model.components.length));
    $('rank-label').textContent = `0 ~ ${model.components.length}개 선택`;
    $('query-status').textContent = '새 이미지는 기존 평균과 주요 방향에만 사영합니다. 분석 자료에 추가하지 않습니다.';
    paint($('mean-face'), model.mean);
    const host = $('eigen-gallery'); host.replaceChildren();
    model.components.slice(0, 4).forEach((u, i) => {
      const figure = document.createElement('figure'), c = document.createElement('canvas'), label = document.createElement('figcaption');
      c.width = c.height = size; c.setAttribute('aria-label', `주요 방향 ${i + 1}`); paint(c, u, true);
      label.textContent = `방향 ${i + 1} · ${(100 * model.eigenvalues[i] / model.totalEnergy).toFixed(1)}%`;
      figure.append(c, label); host.append(figure);
    });
    if (!model.components.length) host.textContent = '모든 이미지가 같아서 평균 이외의 주요 방향이 없습니다.';
    render();
  }
  function render() {
    if (!model) return;
    const k = Number($('component-count').value), row = query || samples[selected].values;
    const scores = P.transform(model, row, k), reconstruction = P.reconstruct(model, scores), ratio = P.retained(model, k);
    $('component-readout').textContent = `${k}개`;
    $('dimension-stat').textContent = `1024 → ${k}`;
    $('retained-stat').textContent = ratio === null ? '변동 없음' : `${(100 * ratio).toFixed(1)}%`;
    $('error-stat').textContent = f(P.rmse(row, reconstruction), 2);
    $('original-label').textContent = query ? `새 이미지 · ${queryName}` : samples[selected].name;
    $('reconstructed-label').textContent = k === 0 ? '평균 이미지만으로 복원' : `주요 방향 ${k}개로 복원`;
    $('reconstruction-note').textContent = k === 0
      ? '좌표를 하나도 남기지 않으면 어느 이미지를 선택해도 평균 이미지만 남습니다.'
      : query
        ? '새 이미지는 이미 구한 방향에 사영합니다. 모든 방향을 남겨도 분석 자료에 없는 차이는 복원되지 않을 수 있습니다.'
        : `이 이미지를 ${k}개 좌표로 표현했습니다. 방향을 더 남길수록 원래 픽셀 값과의 차이는 줄어듭니다.`;
    $('selected-scores').textContent = `t = [${scores.map(x => f(x, 2)).join(', ')}]${k === 0 ? ' · 평균만 사용' : ''}`;
    paint($('original-face'), row); paint($('reconstructed-face'), reconstruction);
    paint($('difference-face'), row.map((v, i) => Math.abs(v - reconstruction[i])));
    drawSpectrum(k);
  }
  function drawSpectrum(k) {
    const canvas = $('spectrum-chart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h); ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    const n = model.eigenvalues.length, left = 28, width = w - 56, base = h - 30;
    ctx.strokeStyle = '#d9e2d2'; ctx.beginPath(); ctx.moveTo(left, base); ctx.lineTo(w - left, base); ctx.stroke();
    if (!n) { ctx.fillStyle = '#6e806b'; ctx.fillText('자료의 변동이 없습니다.', w / 2, h / 2); $('spectrum-note').textContent = '모든 이미지가 같아서 방향별 변동 비율을 계산하지 않습니다.'; return; }
    const maxRatio = model.eigenvalues[0] / model.totalEnergy;
    model.eigenvalues.forEach((lambda, i) => {
      const ratio = lambda / model.totalEnergy, column = width / n;
      const bh = ratio / maxRatio * (h - 68), x = left + (i + .5) * column;
      ctx.fillStyle = i < k ? '#367c57' : '#dce5d3'; ctx.fillRect(x - column * .31, base - bh, column * .62, bh);
      ctx.fillStyle = '#4c6652'; ctx.fillText(`${(100 * ratio).toFixed(1)}%`, x, base - bh - 9); ctx.fillText(String(i + 1), x, base + 20);
    });
    $('spectrum-note').textContent = `큰 고윳값 순서입니다. 각 막대는 그 방향의 고윳값 ÷ 전체 고윳값의 합입니다. 초록색 ${k}개 방향을 현재 복원에 사용합니다.`;
  }
  async function readImage(file) {
    if (!file || file.size > 10 * 1024 * 1024 || !/^image\/(png|jpeg|webp|bmp|x-ms-bmp)$/.test(file.type)) throw new Error('10MB 이하의 PNG·JPEG·WebP·BMP 이미지를 선택해 주세요.');
    const url = URL.createObjectURL(file), image = new Image();
    try {
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('이미지를 읽지 못했습니다. 다른 파일을 선택해 주세요.')); image.src = url; });
      const c = document.createElement('canvas'); c.width = c.height = size;
      const ctx = c.getContext('2d', { willReadFrequently: true }), side = Math.min(image.naturalWidth, image.naturalHeight);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, size, size);
      ctx.drawImage(image, (image.naturalWidth - side) / 2, (image.naturalHeight - side) / 2, side, side, 0, 0, size, size);
      const data = ctx.getImageData(0, 0, size, size).data;
      return Array.from({ length: size * size }, (_, i) => .299 * data[4 * i] + .587 * data[4 * i + 1] + .114 * data[4 * i + 2]);
    } finally { URL.revokeObjectURL(url); }
  }
  function lock(value) {
    busy = value; $('demo-faces').disabled = value; $('face-upload').disabled = value; $('query-upload').disabled = value;
    $('face-gallery').setAttribute('aria-busy', String(value));
  }
  $('face-upload').addEventListener('change', async event => {
    if (busy) return;
    const files = Array.from(event.target.files); event.target.value = '';
    if (!files.length) return;
    if (files.length < 2 || files.length > 24) { $('face-status').textContent = '동시에 분석할 이미지를 2~24장 선택해 주세요. 기존 자료는 유지됩니다.'; return; }
    lock(true); $('face-status').textContent = '이미지를 읽고 주요 방향을 계산합니다…';
    try {
      const next = [];
      for (const file of files) next.push({ name: file.name, values: await readImage(file) });
      analyze(next, '내 이미지');
    } catch (error) { $('face-status').textContent = `${error.message} 기존 자료는 유지됩니다.`; }
    finally { lock(false); }
  });
  $('query-upload').addEventListener('change', async event => {
    if (busy) return;
    const file = event.target.files[0]; event.target.value = ''; if (!file) return;
    lock(true);
    try { query = await readImage(file); queryName = file.name; gallery(); render(); $('query-status').textContent = '분석에 쓰지 않은 새 이미지입니다. 기존 평균과 주요 방향을 그대로 사용합니다.'; }
    catch (error) { $('query-status').textContent = error.message; }
    finally { lock(false); }
  });
  const loadDemo = () => analyze(Array.from({ length: 12 }, (_, i) => ({ name: `얼굴 ${String(i + 1).padStart(2, '0')}`, values: demoFace(i) })), '예제 얼굴 12장');
  $('demo-faces').addEventListener('click', loadDemo);
  $('component-count').addEventListener('input', render);
  function download(blob, filename) {
    const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 20000);
  }
  $('save-reconstruction').addEventListener('click', () => $('reconstructed-face').toBlob(blob => { if (blob) download(blob, 'pca-reconstruction.png'); }, 'image/png'));
  const csvCell = (value, isName) => { const s = String(value); return `"${isName && /^[=+@\-\t\r]/.test(s) ? "'" : ''}${s.replaceAll('"', '""')}"`; };
  $('save-coordinates').addEventListener('click', () => {
    const k = Number($('component-count').value);
    const rows = [['image', ...Array.from({ length: k }, (_, i) => `PC${i + 1}`), 'RMSE']];
    samples.forEach(sample => { const scores = P.transform(model, sample.values, k); rows.push([sample.name, ...scores.map(v => f(v, 6)), f(P.rmse(sample.values, P.reconstruct(model, scores)), 6)]); });
    download(new Blob(['\ufeff', rows.map(row => row.map((v, i) => csvCell(v, i === 0)).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' }), 'pca-coordinates.csv');
  });
  const points = [[-2, -1], [-1, -2], [1, 2], [2, 1]];
  function pointExperiment() {
    const degrees = Number($('projection-angle').value), rad = degrees * Math.PI / 180, u = [Math.cos(rad), Math.sin(rad)];
    const t = points.map(p => P.dot(p, u)), energy = t.reduce((s, v) => s + v * v, 0);
    $('angle-readout').textContent = `${degrees}°`;
    $('projected-energy').textContent = f(energy);
    $('energy-explanation').textContent = `45°에서 최대 18, 135°에서 최소 2입니다. 현재 방향에서 전체 변동 20 중 ${(energy / 20 * 100).toFixed(1)}%를 담습니다.`;
    $('point-values').replaceChildren(...points.map((p, i) => {
      const tr = document.createElement('tr');
      [`(${p.join(', ')})`, f(t[i]), `(${f(t[i] * u[0], 2)}, ${f(t[i] * u[1], 2)})`].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); }); return tr;
    }));
    const c = $('point-chart'), ctx = c.getContext('2d'), w = c.width, h = c.height, unit = 56, cx = w / 2, cy = h / 2;
    const xy = p => [cx + p[0] * unit, cy - p[1] * unit];
    const line = (a, b, color, width = 1, dash = []) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(...xy(a)); ctx.lineTo(...xy(b)); ctx.stroke(); ctx.setLineDash([]); };
    const circle = (p, fill, stroke, radius = 6) => { ctx.beginPath(); ctx.arc(...xy(p), radius, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); } };
    ctx.clearRect(0, 0, w, h); ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    for (let i = -4; i <= 4; i++) { line([i, -3], [i, 3], '#edf0e6'); if (Math.abs(i) <= 3) line([-4.7, i], [4.7, i], '#edf0e6'); }
    line([-4.8, 0], [4.8, 0], '#bfcabb'); line([0, -3.2], [0, 3.2], '#bfcabb');
    ctx.fillStyle = '#70816b'; ctx.fillText('x', w - 27, cy - 9); ctx.fillText('y', cx + 13, 24); ctx.fillText('0', cx - 12, cy + 17);
    line(u.map(v => -4 * v), u.map(v => 4 * v), '#267451', 2.4);
    points.forEach((p, i) => { const projected = u.map(v => v * t[i]); line(p, projected, '#a8b7a2', 1.5, [4, 4]); circle(projected, '#fff', '#267451', 6); circle(p, '#d29a55', '#a5753e', 5); const pixel = xy(p); ctx.fillStyle = '#7d633e'; ctx.fillText(`(${p.join(',')})`, pixel[0] + (p[0] > 0 ? 24 : -24), pixel[1] + (i % 2 ? 20 : -13)); });
  }
  $('projection-angle').addEventListener('input', pointExperiment);
  $('best-direction').addEventListener('click', () => { $('projection-angle').value = '45'; pointExperiment(); });
  try { loadDemo(); pointExperiment(); } catch (error) { $('face-status').textContent = error.message; }
})();
