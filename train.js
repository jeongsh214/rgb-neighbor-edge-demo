/* global RgbEdges, EdgeMLP */
'use strict';
(() => {
  const $ = id => document.getElementById(id), M = EdgeMLP;
  const tick = () => new Promise(resolve => setTimeout(resolve, 0));
  const pct = n => n === null ? '—' : `${(100 * n).toFixed(1)}%`;
  const fmt = n => Number.isFinite(n) ? String(Number(n.toPrecision(4))) : '—';
  let datasets = [], model = null, busy = false, loading = false, stop = false;
  let preview = null, probabilities = null, cutoff = .5, predictionTicket = 0, evaluationTicket = 0;
  let trainedKeys = new Set(), history = [], initialWeight = null;

  function status(id, message, error = false) { $(id).textContent = message; $(id).classList.toggle('error', error); }
  function controls() {
    for (const id of ['dataset-upload', 'load-demo-data', 'clear-data', 'start-training', 'epochs', 'learning-rate', 'model-upload']) $(id).disabled = busy || loading;
    document.querySelectorAll('#datasets select,#datasets button').forEach(e => { e.disabled = busy || loading; });
    $('stop-training').disabled = !busy;
    $('save-model').disabled = !model || busy;
    $('save-prediction').disabled = !probabilities || busy;
    for (const id of ['dataset-upload', 'model-upload']) document.querySelector(`label[for="${id}"]`).setAttribute('aria-disabled', String(busy || loading));
  }
  function metricHTML(m) {
    return `<div>정밀도<strong>${pct(m.precision)}</strong></div><div>재현율<strong>${pct(m.recall)}</strong></div><div>F1<strong>${pct(m.f1)}</strong></div>`;
  }
  function emptyMetrics() { $('heldout-metrics').innerHTML = metricHTML({ precision: null, recall: null, f1: null }); }
  function resetModel() {
    model = null; trainedKeys = new Set(); probabilities = null; history = [];
    predictionTicket++; evaluationTicket++; emptyMetrics();
    $('initial-loss').textContent = $('current-loss').textContent = '—'; $('completed-epochs').textContent = '0';
    $('training-progress').value = 0; $('evaluation-count').textContent = '';
    $('weight-change').textContent = '학습하면 실제로 바뀐 가중치를 여기에 표시합니다.';
    status('evaluation-status', '학습이 끝나면 평가용 정답 전체와 비교합니다.');
    status('preview-metrics', ''); status('prediction-status', ''); status('model-status', ''); drawChart(); controls();
  }
  function changedData() {
    resetModel(); renderDatasets();
    status('training-status', '자료가 준비되었습니다. 용도를 확인하고 학습 시작을 눌러 주세요.');
  }
  function renderDatasets() {
    $('datasets').replaceChildren(); $('preview-select').replaceChildren();
    datasets.forEach((d, i) => {
      const tr = document.createElement('tr'), name = document.createElement('td'), counts = document.createElement('td');
      name.textContent = d.name; counts.textContent = `${d.positive.toLocaleString()} / ${d.negative.toLocaleString()}`;
      const roleCell = document.createElement('td'), select = document.createElement('select');
      select.setAttribute('aria-label', `${d.name} 용도`);
      for (const [value, label] of [['train', '학습'], ['eval', '평가'], ['unused', '미사용']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = label; select.append(option);
      }
      select.value = d.role;
      select.addEventListener('change', () => { d.role = select.value; changedData(); }); roleCell.append(select);
      const removeCell = document.createElement('td'), remove = document.createElement('button');
      remove.textContent = '제거'; remove.setAttribute('aria-label', `${d.name} 제거`);
      remove.addEventListener('click', () => { datasets.splice(i, 1); changedData(); }); removeCell.append(remove);
      tr.append(name, counts, roleCell, removeCell); $('datasets').append(tr);
      const option = document.createElement('option'); option.value = String(i); option.textContent = `${d.name} · ${d.role === 'train' ? '학습' : d.role === 'eval' ? '평가' : '미사용'}`; $('preview-select').append(option);
    });
    $('dataset-count').textContent = `${datasets.length}장`;
    const synthetic = datasets.filter(d => d.provenance === 'synthetic_geometry').length;
    $('data-source').textContent = synthetic ? `도형 ${synthetic}장의 정답은 도형의 실제 영역 경계로 만든 연습용 정답입니다. 자동 점수 임계값을 사용하지 않습니다.` : '내 자료의 정답은 A에서 사람이 직접 표시한 내용입니다. 미지정은 학습과 평가에서 제외합니다.';
    const preferred = datasets.findIndex(d => d.role === 'eval');
    const index = preferred >= 0 ? preferred : 0;
    $('preview-select').value = String(index); setPreview(datasets[index] || null); controls();
  }
  function loadDemo() {
    datasets = [0, 1, 2, 3].map(v => Object.assign(M.demoImage(v), { role: v === 3 ? 'eval' : 'train' }));
    changedData(); status('data-status', '도형 3장은 학습용, 다른 1장은 평가용입니다. 학습 시작을 눌러 보세요.');
  }
  function drawChart() {
    const c = $('loss-chart'), ctx = c.getContext('2d'), width = c.width, height = c.height;
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#f5f8f1'; ctx.fillRect(0, 0, width, height);
    ctx.font = '12px system-ui'; ctx.fillStyle = '#5b7262'; ctx.fillText('손실 ↓', 12, 18); ctx.fillText('학습 반복 →', width - 90, height - 8);
    if (!history.length) return;
    const max = Math.max(.01, ...history.map(p => p.loss)) * 1.1, lastEpoch = Math.max(1, history.at(-1).epoch);
    ctx.strokeStyle = '#2b7556'; ctx.lineWidth = 2.5; ctx.beginPath();
    history.forEach((p, i) => { const x = 40 + p.epoch / lastEpoch * (width - 60), y = height - 25 - p.loss / max * (height - 50); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.fillText(fmt(history[0].loss), 8, 38); ctx.fillText(fmt(history.at(-1).loss), width - 75, 22);
  }
  function drawDataset(d) {
    const ids = ['mlp-original', 'mlp-labels', 'mlp-prediction'];
    for (const id of ids) { $(id).width = d ? d.image.width : 96; $(id).height = d ? d.image.height : 64; }
    if (!d) return;
    $('mlp-original').getContext('2d').putImageData(new ImageData(Uint8ClampedArray.from(d.image.data), d.image.width, d.image.height), 0, 0);
    const ctx = $('mlp-labels').getContext('2d'), pixels = ctx.createImageData(d.image.width, d.image.height);
    for (let i = 0; i < d.labels.length; i++) {
      const value = d.labels[i] === 1 ? 255 : d.labels[i] === 2 ? 0 : 128;
      pixels.data.set([value, value, value, d.result.valid[i] && d.result.counts[i] ? 255 : 0], 4 * i);
    }
    ctx.putImageData(pixels, 0, 0);
  }
  function setPreview(d) {
    preview = d; probabilities = null; predictionTicket++; drawDataset(d);
    $('preview-name').textContent = d ? d.name : '자료 없음';
    status('preview-metrics', d && !model ? '모델을 학습하거나 저장한 모델을 불러오면 예측을 표시합니다.' : '');
    if (d && model && !busy) predictPreview(); else status('prediction-status', '');
    controls();
  }
  async function predictPreview() {
    if (!model || !preview) return;
    const ticket = ++predictionTicket, snapshot = preview, trained = model;
    status('prediction-status', '이미지의 각 픽셀을 예측 중입니다.');
    probabilities = null; controls();
    const values = new Float32Array(snapshot.labels.length);
    for (let i = 0; i < values.length; i++) {
      if (snapshot.result.valid[i] && snapshot.result.counts[i]) values[i] = M.forward(trained, M.vector(snapshot.image, snapshot.result, i)).p;
      if (i % 8192 === 0) { await tick(); if (ticket !== predictionTicket) return; }
    }
    if (ticket !== predictionTicket) return;
    probabilities = values; renderPrediction(); status('prediction-status', '예측 완료'); controls();
  }
  function renderPrediction() {
    if (!preview || !probabilities) return;
    const ctx = $('mlp-prediction').getContext('2d'), pixels = ctx.createImageData(preview.image.width, preview.image.height);
    const predicted = new Uint8Array(probabilities.length);
    for (let i = 0; i < probabilities.length; i++) {
      const valid = preview.result.valid[i] && preview.result.counts[i];
      const yes = valid && probabilities[i] >= cutoff; predicted[i] = yes ? 1 : 0;
      const value = yes ? 255 : 0; pixels.data.set([value, value, value, valid ? 255 : 0], 4 * i);
    }
    ctx.putImageData(pixels, 0, 0);
    const m = RgbEdges.metrics(preview.result, predicted, preview.labels);
    const role = trainedKeys.has(preview.key) ? '학습에 사용한 이미지의 확인값' : '학습에 사용하지 않은 이미지';
    status('preview-metrics', m.evaluated ? `${role} · 정답 ${m.evaluated.toLocaleString()}개 · 정밀도 ${pct(m.precision)} · 재현율 ${pct(m.recall)} · F1 ${pct(m.f1)}` : '정답이 없는 새 이미지입니다. 예측만 표시하며 성능 지표는 계산하지 않습니다.');
  }
  async function evaluateHeldout() {
    const ticket = ++evaluationTicket;
    const evaluation = datasets.filter(d => d.role === 'eval');
    emptyMetrics();
    if (!model) return;
    if (!evaluation.length) { status('evaluation-status', '평가용 이미지가 없습니다. 학습 이미지와 다른 원본의 정답 자료를 추가하고 용도를 평가로 선택해 주세요.'); return; }
    if (evaluation.some(d => trainedKeys.has(d.key))) { status('evaluation-status', '모델 학습에 사용한 원본이 평가용에 포함되어 평가하지 않습니다.', true); return; }
    status('evaluation-status', '평가용 이미지의 직접 표시된 정답 전체와 비교 중입니다.');
    let tp = 0, fp = 0, fn = 0, tn = 0;
    const snapshot = model, threshold = cutoff;
    for (const d of evaluation) for (let i = 0; i < d.labels.length; i++) {
      if (d.result.valid[i] && d.result.counts[i] && d.labels[i]) {
        const yes = M.forward(snapshot, M.vector(d.image, d.result, i)).p >= threshold;
        if (d.labels[i] === 1) yes ? tp++ : fn++; else yes ? fp++ : tn++;
      }
      if (i % 8192 === 0) { await tick(); if (ticket !== evaluationTicket) return; }
    }
    if (ticket !== evaluationTicket) return;
    const m = M.metricCounts(tp, fp, fn, tn);
    $('heldout-metrics').innerHTML = metricHTML(m); $('evaluation-count').textContent = `${evaluation.length}장 · 정답 ${m.evaluated.toLocaleString()}개`;
    status('evaluation-status', m.evaluated ? `맞춘 윤곽 ${tp.toLocaleString()} · 잘못 선택 ${fp.toLocaleString()} · 놓친 윤곽 ${fn.toLocaleString()} · 미지정은 제외했습니다.` : '평가용 이미지에 유효한 정답이 없습니다.');
  }
  async function train(event) {
    event.preventDefault(); if (busy || loading) return;
    const training = datasets.filter(d => d.role === 'train'), evaluation = datasets.filter(d => d.role === 'eval');
    let rows;
    try { M.checkSplit(training, evaluation); rows = M.balancedRows(training); }
    catch (error) { status('training-status', error.message, true); return; }
    const epochs = Number($('epochs').value), rate = Number($('learning-rate').value);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 500 || ![.003, .01, .03].includes(rate)) return;
    resetModel(); busy = true; stop = false; model = M.createModel(rows);
    trainedKeys = new Set(training.map(d => d.key)); initialWeight = model.w1[0]; controls();
    const opt = M.optimizer(model), rng = M.random(2026), initial = M.evaluate(model, rows).loss;
    $('initial-loss').textContent = fmt(initial); $('current-loss').textContent = fmt(initial);
    $('training-progress').max = epochs; history = [{ epoch: 0, loss: initial }]; drawChart();
    let completed = 0, successful = false;
    try {
      for (let epoch = 1; epoch <= epochs; epoch++) {
        M.shuffle(rows, rng);
        for (let i = 0; i < rows.length; i += 64) {
          if (stop) break;
          M.trainBatch(model, rows.slice(i, i + 64), opt, rate);
          if ((i / 64) % 8 === 0) await tick();
        }
        if (stop) break;
        completed = epoch;
        if (epoch === 1 || epoch % 5 === 0 || epoch === epochs) {
          const l = M.evaluate(model, rows).loss; history.push({ epoch, loss: l });
          $('current-loss').textContent = fmt(l); drawChart();
        }
        $('completed-epochs').textContent = String(epoch); $('training-progress').value = epoch;
        $('weight-change').textContent = `실제 첫 번째 가중치 W₁[1,1]: ${fmt(initialWeight)} → ${fmt(model.w1[0])}`;
        status('training-status', `${rows.length.toLocaleString()}개 샘플 · ${epoch} / ${epochs}회 학습 중`);
        await tick();
      }
      const finalLoss = M.evaluate(model, rows).loss; $('current-loss').textContent = fmt(finalLoss);
      history.push({ epoch: completed, loss: finalLoss }); drawChart();
      model.training = { provenance: [...new Set(training.map(d => d.provenance))], image_keys: [...trainedKeys], epochs_completed: completed, completed_full_schedule: !stop, optimizer: 'Adam', learning_rate: rate, balanced_samples: rows.length, initial_loss: initial, final_loss: finalLoss };
      status('training-status', `${stop ? '중단했습니다. 현재까지 조정한 모델을 사용할 수 있습니다.' : '학습 완료.'} 손실 ${fmt(initial)} → ${fmt(finalLoss)} · ${rows.length.toLocaleString()}개 샘플`);
      $('weight-change').textContent = `실제 첫 번째 가중치 W₁[1,1]: ${fmt(initialWeight)} → ${fmt(model.w1[0])}`;
      successful = true;
    } catch (error) { model = null; status('training-status', error.message, true); }
    finally { busy = false; controls(); }
    if (successful) { await evaluateHeldout(); await predictPreview(); }
  }
  async function readDataset(file) {
    if (file.size > 64 * 1024 * 1024) throw new Error('학습 JSON은 파일당 64MB 이하로 선택해 주세요.');
    const value = JSON.parse(await file.text());
    const { width, height, png, mask, name } = M.annotation(value);
    const bytes = Uint8Array.from(atob(png.split(',')[1]), c => c.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    try {
      if (bitmap.width !== width || bitmap.height !== height) throw new Error('PNG와 JSON의 이미지 크기가 다릅니다.');
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      const d = M.dataset(canvas.getContext('2d').getImageData(0, 0, width, height), mask, name);
      // Recompute every feature from original pixels, never trust imported samples/thresholds.
      d.provenance = 'manual_user_annotation'; d.role = 'train'; return d;
    } finally { bitmap.close(); }
  }
  function download(blob, filename) {
    const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  $('train-form').addEventListener('submit', train);
  $('stop-training').addEventListener('click', () => { stop = true; status('training-status', '현재까지의 모델을 남기고 학습을 중단합니다.'); });
  $('load-demo-data').addEventListener('click', loadDemo);
  $('clear-data').addEventListener('click', () => { datasets = []; changedData(); status('data-status', 'A에서 정답을 표시한 학습 JSON을 추가해 주세요.'); });
  $('dataset-upload').addEventListener('change', async event => {
    if (busy || loading) return; const files = [...event.target.files]; if (!files.length) return;
    loading = true; controls(); const messages = [];
    for (const file of files) {
      try {
        const d = await readDataset(file), index = datasets.findIndex(old => old.key === d.key);
        if (index >= 0) { d.role = datasets[index].role; datasets[index] = d; messages.push(`${d.name}: 같은 원본의 정답을 교체했습니다.`); }
        else if (datasets.length < 20) { datasets.push(d); messages.push(`${d.name}: 추가했습니다.`); }
        else throw new Error('한 번에 최대 20장까지 사용할 수 있습니다.');
      } catch (error) { messages.push(`${file.name}: ${error.message}`); }
    }
    loading = false; event.target.value = ''; changedData(); status('data-status', messages.join(' '));
  });
  $('preview-select').addEventListener('change', e => setPreview(datasets[Number(e.target.value)] || null));
  $('prediction-cutoff').addEventListener('input', e => {
    cutoff = Number(e.target.value); $('cutoff-readout').textContent = cutoff.toFixed(2); renderPrediction();
    if (model && !busy) evaluateHeldout();
  });
  $('prediction-upload').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 30 * 1024 * 1024) throw new Error('30MB 이하의 이미지를 선택해 주세요.');
      const bitmap = await createImageBitmap(file), scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement('canvas'); canvas.width = Math.max(1, Math.round(bitmap.width * scale)); canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close();
      const d = M.dataset(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height), new Uint8Array(canvas.width * canvas.height), `${file.name} · ${canvas.width}×${canvas.height}`);
      $('preview-select').selectedIndex = -1; setPreview(d);
    } catch (error) { status('prediction-status', error.message, true); }
    event.target.value = '';
  });
  $('save-model').addEventListener('click', () => {
    if (!model) return; download(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' }), 'trained-edge-mlp.json');
    status('model-status', '가중치와 입력 변환 정보가 담긴 모델을 저장했습니다.');
  });
  $('model-upload').addEventListener('change', async event => {
    if (busy || loading) return; const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error('이 예제에서 저장한 1MB 이하의 모델 JSON을 선택해 주세요.');
      const loaded = M.loadModel(JSON.parse(await file.text())); resetModel(); model = loaded;
      trainedKeys = new Set(Array.isArray(loaded.training?.image_keys) ? loaded.training.image_keys : []);
      if (typeof loaded.training?.initial_loss === 'number') $('initial-loss').textContent = fmt(loaded.training.initial_loss);
      if (typeof loaded.training?.final_loss === 'number') $('current-loss').textContent = fmt(loaded.training.final_loss);
      $('completed-epochs').textContent = String(loaded.training?.epochs_completed || 0);
      status('training-status', '저장한 모델을 불러왔습니다. 현재 자료로 다시 학습하면 새 모델로 교체됩니다.');
      status('model-status', '모델 불러오기 완료'); controls(); await evaluateHeldout(); await predictPreview();
    } catch (error) { status('model-status', `모델을 읽을 수 없습니다. ${error.message}`, true); }
    event.target.value = '';
  });
  $('save-prediction').addEventListener('click', () => { if (probabilities) $('mlp-prediction').toBlob(blob => { if (blob) download(blob, 'mlp-edge-prediction.png'); }, 'image/png'); });
  loadDemo();
})();
