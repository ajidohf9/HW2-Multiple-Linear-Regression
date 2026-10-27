/**
 * ====================================================================
 * ML-Studio HW2: Multiple Linear Regression Interactive Web Engine
 * 國立中興大學 巨量資料分析作業 HW2
 * 學生學號: 7115093066
 * 
 * 核心功能：
 * 1. 【加分 1 分】下拉視窗切換 Problem 1 (洪水預測) / Problem 2 (50_Startups) / Problem 3 (自訂)
 * 2. 【加分 2 分】任意輸入 Dataset URL 或上傳本地 CSV 檔案，動態解析並回歸建模
 * 3. CRISP-DM 完整流程：特徵選擇、五大假設檢定、係數統計檢定表、信賴與預測區間可視化
 * 4. 即時部署模擬器 (Deployment Simulator)：滑桿微調特徵即時推論
 * ====================================================================
 */

// 全域狀態儲存
const state = {
  rawDataset: null,
  headers: [],
  numericColumns: [],
  targetCol: '',
  selectedFeatures: [],
  modelResults: null,
  activeChartTab: 'prediction',
  chartInstance: null
};

// 50_Startups 內建備用資料 (以防在無網路或 CORS 受限時依然 100% 正常載入)
const EMBEDDED_STARTUPS_CSV = `R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.17,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
78013.11,121597.55,264346.06,California,126992.93
94657.16,145077.58,282574.31,New York,125370.37
91749.16,114175.79,294919.57,Florida,124266.9
86419.7,153514.11,0,New York,122776.86
76253.86,113867.3,298664.47,California,118474.03
78389.47,153773.43,299737.29,New York,111313.02
73994.56,122782.75,303319.26,Florida,110352.25
67532.53,105751.03,304768.73,Florida,108733.99
77044.01,99281.34,140574.81,New York,108552.04
64664.71,139553.16,137962.62,California,107404.34
75328.87,144135.98,134050.07,Florida,105733.54
72107.6,127864.55,353183.81,New York,105008.31
66051.52,182645.56,118148.2,Florida,103282.38
65605.48,153032.06,107138.38,New York,101004.64
61994.48,115641.28,91131.24,Florida,99937.59
61136.38,152701.92,88218.23,New York,97483.56
63408.86,129219.61,46085.25,California,97427.84
55493.95,103057.49,214634.81,Florida,96778.92
46426.07,157693.92,210797.67,California,96712.8
46014.02,85047.44,205517.64,Florida,96479.51
28663.76,127056.21,201126.82,New York,90708.19
44069.95,51283.14,197029.42,California,89949.14
20229.59,65947.93,185265.1,New York,81229.06
38558.51,82982.09,174999.3,California,81005.76
28754.33,118546.05,172795.67,California,78239.91
27892.92,84710.77,164470.71,Florida,77798.83
23640.93,96189.63,148001.11,California,71498.49
15505.73,127382.3,35534.17,New York,69758.98
22177.74,154806.14,28334.72,California,65200.33
1000.23,124153.04,1903.93,New York,64926.08
1315.46,115816.21,297114.46,Florida,49490.75
0,135426.92,0,California,42559.73
542.05,51743.15,0,New York,35673.41
0,116983.8,45173.06,California,14681.4`;

// ====================================================================
// 1. 純 JavaScript 矩陣線性代數運算核心 (OLS Solver)
// ====================================================================

function transpose(A) {
  const m = A.length, n = A[0].length;
  const AT = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      AT[j][i] = A[i][j];
    }
  }
  return AT;
}

function matMul(A, B) {
  const m = A.length, n = A[0].length, p = B[0].length;
  const C = Array.from({ length: m }, () => new Float64Array(p));
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++) {
        C[i][j] += aik * B[k][j];
      }
    }
  }
  return C;
}

function matVecMul(A, x) {
  const m = A.length, n = A[0].length;
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      s += A[i][j] * x[j];
    }
    y[i] = s;
  }
  return y;
}

// 高斯-若爾當消去法 (Gauss-Jordan Elimination) 求矩陣反矩陣，附微量 Ridge 正則化保障數值穩定
function invertMatrix(M, ridgeAlpha = 1e-7) {
  const n = M.length;
  const A = Array.from({ length: n }, (row, i) => {
    const r = new Float64Array(2 * n);
    for (let j = 0; j < n; j++) {
      r[j] = M[i][j] + (i === j ? ridgeAlpha : 0);
    }
    r[n + i] = 1.0;
    return r;
  });

  for (let i = 0; i < n; i++) {
    // 主元選取 (Partial Pivoting)
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    if (Math.abs(A[maxRow][i]) < 1e-12) {
      // 奇異矩陣容錯，直接強化對角
      A[maxRow][i] += 1e-4;
    }
    const temp = A[i];
    A[i] = A[maxRow];
    A[maxRow] = temp;

    const pivot = A[i][i];
    for (let j = 0; j < 2 * n; j++) {
      A[i][j] /= pivot;
    }

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = A[k][i];
        for (let j = 0; j < 2 * n; j++) {
          A[k][j] -= factor * A[i][j];
        }
      }
    }
  }

  const inv = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      row[j] = A[i][n + j];
    }
    return row;
  });
  return inv;
}

// 常態累積分佈函數 (CDF) 逼近，用於計算 t/z 檢定之 p-value
function normalCDF(z) {
  const b1 = 0.319381530, b2 = -0.356563782, b3 = 1.781477937, b4 = -1.821255978, b5 = 1.330274429;
  const p = 0.2316419, c2 = 0.3989422804014327; // 1/sqrt(2*pi)
  const absZ = Math.abs(z);
  const t = 1.0 / (1.0 + p * absZ);
  const pdf = c2 * Math.exp(-0.5 * absZ * absZ);
  const tail = pdf * (b1 * t + b2 * Math.pow(t, 2) + b3 * Math.pow(t, 3) + b4 * Math.pow(t, 4) + b5 * Math.pow(t, 5));
  return z >= 0 ? 1.0 - tail : tail;
}

function calculatePValue(tStat) {
  const p = 2 * (1 - normalCDF(Math.abs(tStat)));
  return Math.max(0, Math.min(1, p));
}

// 核心 OLS 多元線性回歸計算
function runOLS(dataRows, targetName, featureNames) {
  const n = dataRows.length;
  const p = featureNames.length;
  if (n <= p + 1) {
    throw new Error(`樣本數 (${n}) 必須大於特徵數加一 (${p + 1})。`);
  }

  // 1. 抽取特徵矩陣 X (加入截距項 1.0) 與目標向量 y
  const X = Array.from({ length: n }, () => new Float64Array(p + 1));
  const y = new Float64Array(n);
  let ySum = 0;

  for (let i = 0; i < n; i++) {
    const row = dataRows[i];
    X[i][0] = 1.0; // Intercept
    for (let j = 0; j < p; j++) {
      X[i][j + 1] = Number(row[featureNames[j]]) || 0;
    }
    const yVal = Number(row[targetName]) || 0;
    y[i] = yVal;
    ySum += yVal;
  }
  const yMean = ySum / n;

  // 2. 計算 X^T, X^T X, (X^T X)^-1
  const XT = transpose(X);
  const XTX = matMul(XT, X);
  const XTX_inv = invertMatrix(XTX);

  // 3. 計算 X^T y 並求得回歸係數 beta = (X^T X)^-1 X^T y
  const XTy = matVecMul(XT, y);
  const beta = matVecMul(XTX_inv, XTy);

  // 4. 計算擬合值 yHat, 殘差 residuals, SSE, SST, SSR
  const yHat = matVecMul(X, beta);
  const residuals = new Float64Array(n);
  let SSE = 0;
  let SST = 0;
  let absErrSum = 0;
  let dwDiffSqSum = 0;

  for (let i = 0; i < n; i++) {
    const e = y[i] - yHat[i];
    residuals[i] = e;
    SSE += e * e;
    const yDiff = y[i] - yMean;
    SST += yDiff * yDiff;
    absErrSum += Math.abs(e);

    if (i > 0) {
      const diff = residuals[i] - residuals[i - 1];
      dwDiffSqSum += diff * diff;
    }
  }

  // 5. 評估指標: R², Adj R², RMSE, MAE, s^2, F-statistic, DW
  const R2 = 1.0 - (SSE / (SST || 1e-12));
  const dfResid = n - p - 1;
  const dfModel = p;
  const adjR2 = 1.0 - ((SSE / dfResid) / ((SST / (n - 1)) || 1e-12));
  const s2 = SSE / dfResid;
  const s = Math.sqrt(s2);
  const RMSE = Math.sqrt(SSE / n);
  const MAE = absErrSum / n;
  const FStat = ((SST - SSE) / dfModel) / (s2 || 1e-12);
  const DurbinWatson = dwDiffSqSum / (SSE || 1e-12);

  // 6. 計算係數之標準誤、t-stat、p-val、95% 信賴區間
  const coeffStats = [];
  const varNames = ['const', ...featureNames];

  for (let j = 0; j < p + 1; j++) {
    const varC = Math.max(0, XTX_inv[j][j]);
    const se = Math.sqrt(s2 * varC);
    const b = beta[j];
    const tStat = se > 1e-12 ? b / se : 0;
    const pVal = calculatePValue(tStat);
    const ciLow = b - 1.96 * se;
    const ciHigh = b + 1.96 * se;

    coeffStats.push({
      name: varNames[j],
      coeff: b,
      se: se,
      tStat: tStat,
      pVal: pVal,
      ciLow: ciLow,
      ciHigh: ciHigh,
      vif: 1.0 // 截距項不計 VIF
    });
  }

  // 7. 計算特徵之共線性 VIF (對純特徵子矩陣計算相關矩陣之逆)
  if (p > 1) {
    try {
      const corrMat = Array.from({ length: p }, () => new Float64Array(p));
      const fMeans = new Float64Array(p);
      const fStds = new Float64Array(p);

      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let i = 0; i < n; i++) sum += X[i][j + 1];
        fMeans[j] = sum / n;

        let varS = 0;
        for (let i = 0; i < n; i++) {
          const d = X[i][j + 1] - fMeans[j];
          varS += d * d;
        }
        fStds[j] = Math.sqrt(varS / (n - 1)) || 1.0;
      }

      for (let j1 = 0; j1 < p; j1++) {
        for (let j2 = 0; j2 < p; j2++) {
          let cov = 0;
          for (let i = 0; i < n; i++) {
            cov += (X[i][j1 + 1] - fMeans[j1]) * (X[i][j2 + 1] - fMeans[j2]);
          }
          corrMat[j1][j2] = (cov / (n - 1)) / (fStds[j1] * fStds[j2]);
        }
      }

      const invCorr = invertMatrix(corrMat, 1e-6);
      for (let j = 0; j < p; j++) {
        coeffStats[j + 1].vif = Math.max(1.0, invCorr[j][j]);
      }
    } catch (e) {
      console.warn("VIF 計算提示:", e);
    }
  }

  // 8. 抽出特徵統計供部署模擬器使用 (min, max, mean)
  const featureSummaries = {};
  for (let j = 0; j < p; j++) {
    const fName = featureNames[j];
    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < n; i++) {
      const v = X[i][j + 1];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    featureSummaries[fName] = {
      min: min,
      max: max,
      mean: sum / n,
      step: (max - min) > 10 ? 0.1 : (max - min) / 100 || 0.01
    };
  }

  return {
    n, p, targetName, featureNames,
    beta, XTX_inv, s2, s,
    R2, adjR2, RMSE, MAE, FStat, DurbinWatson,
    coeffStats,
    y, yHat, residuals,
    featureSummaries
  };
}

// ====================================================================
// 2. 資料集載入與自動識別
// ====================================================================

function setStatus(text, type = 'normal') {
  const statusEl = document.getElementById('status-text');
  const barEl = document.getElementById('status-notification');
  const iconEl = document.getElementById('status-icon');
  if (!statusEl || !barEl) return;

  statusEl.textContent = text;
  barEl.className = 'notification-bar ' + (type === 'success' ? 'success' : type === 'error' ? 'error' : '');
  iconEl.textContent = type === 'success' ? '✅' : type === 'error' ? '⚠️' : 'ℹ️';
}

function processLoadedData(results) {
  if (!results.data || results.data.length === 0) {
    setStatus("載入之資料集內容為空，請檢查檔案格式！", 'error');
    return;
  }

  const rows = results.data.filter(r => Object.keys(r).length > 1);
  state.rawDataset = rows;
  state.headers = Object.keys(rows[0]);

  // 識別所有數值型欄位 (排除明顯 ID 欄)
  state.numericColumns = state.headers.filter(col => {
    const val = Number(rows[0][col]);
    return !isNaN(val) && col.toLowerCase() !== 'id';
  });

  if (state.numericColumns.length < 2) {
    setStatus("資料集中數值型欄位少於 2 個，無法建立多元回歸模型！", 'error');
    return;
  }

  // 預設 Target 選取邏輯:
  // 如果有 FloodProbability / Profit，優先指定；否則預設最後一個數值欄位
  const targetCol = state.numericColumns.find(c => 
    c.toLowerCase().includes('flood') || 
    c.toLowerCase().includes('profit') || 
    c.toLowerCase().includes('target') || 
    c.toLowerCase().includes('price')
  ) || state.numericColumns[state.numericColumns.length - 1];

  state.targetCol = targetCol;
  
  // 候選特徵欄位預設排除 target
  state.selectedFeatures = state.numericColumns.filter(c => c !== targetCol);

  // 更新介面選單與特徵核取清單
  populateConfigUI();

  // 執行初次回歸運算並更新各儀表板
  recalculateModel();
  setStatus(`成功載入資料集！共 ${rows.length} 筆樣本，包含 ${state.numericColumns.length} 個數值特徵。`, 'success');
}

function populateConfigUI() {
  const targetSelect = document.getElementById('target-selector');
  targetSelect.innerHTML = '';
  state.numericColumns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === state.targetCol) opt.selected = true;
    targetSelect.appendChild(opt);
  });

  const sampleCountBadge = document.getElementById('sample-count-badge');
  sampleCountBadge.textContent = `樣本數: ${state.rawDataset.length} 筆 | 特徵數: ${state.numericColumns.length - 1}`;

  renderFeaturesChecklist();

  // 更新 Top K slider 上限
  const kSlider = document.getElementById('k-slider');
  const maxFeatures = state.numericColumns.length - 1;
  kSlider.max = maxFeatures;
  kSlider.value = Math.min(14, maxFeatures);
  document.getElementById('k-val-display').textContent = kSlider.value;
}

function renderFeaturesChecklist() {
  const container = document.getElementById('features-container');
  container.innerHTML = '';

  const candidateFeatures = state.numericColumns.filter(c => c !== state.targetCol);
  candidateFeatures.forEach(feat => {
    const label = document.createElement('label');
    label.className = 'feature-item-label';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = feat;
    checkbox.checked = state.selectedFeatures.includes(feat);

    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        if (!state.selectedFeatures.includes(feat)) state.selectedFeatures.push(feat);
      } else {
        state.selectedFeatures = state.selectedFeatures.filter(f => f !== feat);
      }
    });

    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(feat));
    container.appendChild(label);
  });
}

// 根據特徵與 Target 執行回歸並刷新所有視覺元件
function recalculateModel() {
  if (state.selectedFeatures.length === 0) {
    alert("請至少勾選一個特徵變數進行回歸分析！");
    return;
  }

  try {
    const res = runOLS(state.rawDataset, state.targetCol, state.selectedFeatures);
    state.modelResults = res;

    updateMetricsCards(res);
    updateFormula(res);
    updateAssumptionsTags(res);
    updateSummaryTable(res);
    renderActiveChart();
    renderDeploymentSimulator(res);

    setStatus(`模型訓練完成！決定係數 R² = ${res.R2.toFixed(4)}，校正決定係數 Adj R² = ${res.adjR2.toFixed(4)}。`, 'success');
  } catch (err) {
    console.error(err);
    setStatus(`回歸計算發生錯誤: ${err.message}`, 'error');
  }
}

// ====================================================================
// 3. UI 更新與統計指標呈現
// ====================================================================

function updateMetricsCards(res) {
  document.getElementById('val-r2').textContent = res.R2.toFixed(4);
  document.getElementById('val-adj-r2').textContent = res.adjR2.toFixed(4);
  document.getElementById('val-rmse').textContent = res.RMSE.toFixed(4);
  document.getElementById('val-mae').textContent = res.MAE.toFixed(4);
  document.getElementById('val-fstat').textContent = res.FStat.toFixed(2);
  document.getElementById('val-dw').textContent = res.DurbinWatson.toFixed(3);

  document.getElementById('dof-badge').textContent = `自由度 Df(模型)=${res.p}, Df(殘差)=${res.n - res.p - 1}`;
}

function updateFormula(res) {
  const constStat = res.coeffStats[0];
  let formula = `<strong>${res.targetName}</strong> = ${constStat.coeff >= 0 ? '' : '-'}${Math.abs(constStat.coeff).toFixed(4)}`;
  
  for (let j = 1; j < res.coeffStats.length; j++) {
    const st = res.coeffStats[j];
    const sign = st.coeff >= 0 ? ' + ' : ' - ';
    formula += `${sign}${Math.abs(st.coeff).toFixed(4)} × <em>${st.name}</em>`;
  }
  document.getElementById('formula-display').innerHTML = formula;
}

function updateAssumptionsTags(res) {
  // Linearity & Homoscedasticity & Normality: based on diagnostics
  const dw = res.DurbinWatson;
  const dwTag = document.getElementById('tag-independence');
  if (dw >= 1.7 && dw <= 2.3) {
    dwTag.className = 'tag-pill tag-pass';
    dwTag.textContent = `✓ 誤差獨立性 (DW = ${dw.toFixed(3)} ≈ 2.0)`;
  } else {
    dwTag.className = 'tag-pill tag-warn';
    dwTag.textContent = `⚠️ 誤差自相關注意 (DW = ${dw.toFixed(3)})`;
  }

  // 共線性檢驗
  let maxVif = 1.0;
  for (let j = 1; j < res.coeffStats.length; j++) {
    if (res.coeffStats[j].vif > maxVif) maxVif = res.coeffStats[j].vif;
  }
  const vifTag = document.getElementById('tag-vif');
  if (maxVif < 5.0) {
    vifTag.className = 'tag-pill tag-pass';
    vifTag.textContent = `✓ 無多重共線性 (最大 VIF = ${maxVif.toFixed(2)} < 5)`;
  } else {
    vifTag.className = 'tag-pill tag-warn';
    vifTag.textContent = `⚠️ 輕微共線性 (最大 VIF = ${maxVif.toFixed(2)})`;
  }
}

function updateSummaryTable(res) {
  const tbody = document.getElementById('summary-table-body');
  tbody.innerHTML = '';

  res.coeffStats.forEach(stat => {
    const tr = document.createElement('tr');
    
    // 顯著性標註
    let stars = '';
    if (stat.pVal < 0.001) stars = '<span class="sig-star">***</span>';
    else if (stat.pVal < 0.01) stars = '<span class="sig-star">**</span>';
    else if (stat.pVal < 0.05) stars = '<span class="sig-star">*</span>';

    tr.innerHTML = `
      <td><strong>${stat.name}</strong></td>
      <td>${stat.coeff.toFixed(4)}</td>
      <td>${stat.se.toFixed(4)}</td>
      <td>${stat.tStat.toFixed(3)}</td>
      <td>${stat.pVal < 0.0001 ? '<0.0001' : stat.pVal.toFixed(4)} ${stars}</td>
      <td>${stat.ciLow.toFixed(4)}</td>
      <td>${stat.ciHigh.toFixed(4)}</td>
      <td>${stat.name === 'const' ? '-' : stat.vif.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ====================================================================
// 4. Chart.js 互動視覺化儀表板
// ====================================================================

function renderActiveChart() {
  const ctx = document.getElementById('main-interactive-chart');
  if (!ctx || !state.modelResults) return;

  if (state.chartInstance) {
    state.chartInstance.destroy();
    state.chartInstance = null;
  }

  const res = state.modelResults;
  const tab = state.activeChartTab;
  const explainEl = document.getElementById('chart-explain-text');

  if (tab === 'prediction') {
    // 預測圖 (含 95% 信賴區間 CI 與 預測區間 PI 帶)
    explainEl.innerHTML = '<strong>預測圖解讀：</strong>藍色散點為實際觀測值，紅色實線為回歸預測值，橙色陰影為 95% 母體平均值信賴區間 (CI)，青色虛線外圍為 95% 個別樣本預測區間 (PI)。';
    
    // 依 yHat 由小至大排序取前 120 點繪製平滑帶
    const sampleIndices = Array.from({ length: res.n }, (_, i) => i);
    sampleIndices.sort((a, b) => res.yHat[a] - res.yHat[b]);
    
    // 抽樣間隔取最多 100 點
    const step = Math.max(1, Math.floor(res.n / 100));
    const plotIndices = [];
    for (let k = 0; k < res.n; k += step) plotIndices.push(sampleIndices[k]);

    const labels = plotIndices.map((idx, rank) => `#${rank + 1}`);
    const actualData = plotIndices.map(idx => res.y[idx]);
    const predictedData = plotIndices.map(idx => res.yHat[idx]);
    
    // 計算每點之 SE(mean) 與 SE(pred)
    const ciUpper = [];
    const ciLower = [];
    const piUpper = [];
    const piLower = [];

    const X = Array.from({ length: res.n }, () => new Float64Array(res.p + 1));
    for (let i = 0; i < res.n; i++) {
      X[i][0] = 1.0;
      for (let j = 0; j < res.p; j++) {
        X[i][j + 1] = Number(state.rawDataset[i][res.featureNames[j]]) || 0;
      }
    }

    plotIndices.forEach(idx => {
      const x0 = X[idx];
      // x0^T * XTX_inv * x0
      let leverage = 0;
      for (let i = 0; i < res.p + 1; i++) {
        let tmp = 0;
        for (let j = 0; j < res.p + 1; j++) {
          tmp += x0[j] * res.XTX_inv[j][i];
        }
        leverage += tmp * x0[i];
      }
      leverage = Math.max(0, leverage);

      const seMean = Math.sqrt(res.s2 * leverage);
      const sePred = Math.sqrt(res.s2 * (1.0 + leverage));
      const yh = res.yHat[idx];

      ciLower.push(yh - 1.96 * seMean);
      ciUpper.push(yh + 1.96 * seMean);
      piLower.push(yh - 1.96 * sePred);
      piUpper.push(yh + 1.96 * sePred);
    });

    state.chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: '實際觀測值 (Actual y)',
            data: actualData,
            type: 'scatter',
            backgroundColor: 'rgba(56, 189, 248, 0.7)',
            borderColor: 'rgba(56, 189, 248, 0.9)',
            pointRadius: 3,
            order: 2
          },
          {
            label: '回歸擬合值 (Predicted yHat)',
            data: predictedData,
            borderColor: '#f43f5e',
            backgroundColor: '#f43f5e',
            borderWidth: 2.5,
            pointRadius: 0,
            fill: false,
            tension: 0.1,
            order: 1
          },
          {
            label: '95% 信賴區間上界 (CI Upper)',
            data: ciUpper,
            borderColor: 'rgba(245, 158, 11, 0.8)',
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(245, 158, 11, 0.15)',
            order: 3
          },
          {
            label: '95% 信賴區間下界 (CI Lower)',
            data: ciLower,
            borderColor: 'rgba(245, 158, 11, 0.8)',
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            order: 4
          },
          {
            label: '95% 預測區間上界 (PI Upper)',
            data: piUpper,
            borderColor: 'rgba(16, 185, 129, 0.7)',
            borderDash: [2, 2],
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            order: 5
          },
          {
            label: '95% 預測區間下界 (PI Lower)',
            data: piLower,
            borderColor: 'rgba(16, 185, 129, 0.7)',
            borderDash: [2, 2],
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            order: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#f8fafc', font: { size: 11 } } },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });

  } else if (tab === 'coefficients') {
    // 標準化回歸係數 (Feature Importance) 水平條形圖
    explainEl.innerHTML = '<strong>特徵重要性解讀：</strong>橫軸展示各特徵係數 (β) 之大小與正負向影響力，條形越長代表對目標變數邊際貢獻越顯著。';
    
    const featStats = res.coeffStats.filter(s => s.name !== 'const');
    featStats.sort((a, b) => Math.abs(b.coeff) - Math.abs(a.coeff));

    state.chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: featStats.map(s => s.name),
        datasets: [{
          label: '回歸係數 (Beta)',
          data: featStats.map(s => s.coeff),
          backgroundColor: featStats.map(s => s.coeff >= 0 ? 'rgba(56, 189, 248, 0.75)' : 'rgba(244, 63, 94, 0.75)'),
          borderColor: featStats.map(s => s.coeff >= 0 ? '#38bdf8' : '#f43f5e'),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8' } },
          y: { grid: { display: false }, ticks: { color: '#f8fafc', font: { weight: '600' } } }
        }
      }
    });

  } else if (tab === 'residuals') {
    // 殘差 vs 擬合值 (檢驗線性與同質變異性)
    explainEl.innerHTML = '<strong>殘差圖解讀：</strong>檢驗線性與同質變異性假設。殘差應隨機均勻散佈於 0 水平線兩側（灰色虛線），無漏斗狀發散或明顯曲線彎曲結構。';

    const sampleIndices = [];
    const step = Math.max(1, Math.floor(res.n / 300));
    for (let i = 0; i < res.n; i += step) sampleIndices.push(i);

    const scatterData = sampleIndices.map(idx => ({
      x: res.yHat[idx],
      y: res.residuals[idx]
    }));

    state.chartInstance = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '殘差分佈 (Residuals)',
            data: scatterData,
            backgroundColor: 'rgba(168, 85, 247, 0.65)',
            borderColor: 'rgba(168, 85, 247, 0.9)',
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#f8fafc' } }
        },
        scales: {
          x: { 
            title: { display: true, text: '擬合值 (Fitted Values)', color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }, 
            ticks: { color: '#94a3b8' } 
          },
          y: { 
            title: { display: true, text: '殘差 (Residuals)', color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }, 
            ticks: { color: '#94a3b8' } 
          }
        }
      }
    });

  } else if (tab === 'qqplot') {
    // 常態 Q-Q Plot
    explainEl.innerHTML = '<strong>常態 Q-Q Plot 解讀：</strong>檢驗殘差常態性。散點若緊密貼合紅色 45 度參考對角線，代表殘差完美符合常態分佈假設。';

    const sortedResids = Array.from(res.residuals).sort((a, b) => a - b);
    const m = sortedResids.length;
    const step = Math.max(1, Math.floor(m / 250));
    
    // 計算殘差標準差
    let rSum = 0;
    for (let i = 0; i < m; i++) rSum += sortedResids[i];
    const rMean = rSum / m;
    let rVar = 0;
    for (let i = 0; i < m; i++) rVar += Math.pow(sortedResids[i] - rMean, 2);
    const rStd = Math.sqrt(rVar / m) || 1.0;

    const qqData = [];
    const lineData = [];

    // 理論常態分位數反向逼近 (Probit)
    function probit(p) {
      // 簡化合理逼近
      const a = 0.147;
      const x = 2 * p - 1;
      const ln = Math.log(1 - x * x);
      const term1 = 2 / (Math.PI * a) + ln / 2;
      const inner = term1 * term1 - ln / a;
      const sign = x < 0 ? -1 : 1;
      return sign * Math.sqrt(Math.sqrt(inner) - term1);
    }

    for (let i = 0; i < m; i += step) {
      const p = (i + 0.5) / m;
      const theoreticalQ = probit(p);
      const standardizedResidual = (sortedResids[i] - rMean) / rStd;
      qqData.push({ x: theoreticalQ, y: standardizedResidual });
    }

    lineData.push({ x: -3, y: -3 });
    lineData.push({ x: 3, y: 3 });

    state.chartInstance = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '殘差分位數點 (Standardized Residuals)',
            data: qqData,
            backgroundColor: 'rgba(56, 189, 248, 0.7)',
            pointRadius: 3
          },
          {
            label: '理論常態 45° 參考線',
            data: lineData,
            type: 'line',
            borderColor: '#ef4444',
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#f8fafc' } }
        },
        scales: {
          x: { 
            title: { display: true, text: '理論常態分位數 (Theoretical Quantiles)', color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }, 
            ticks: { color: '#94a3b8' },
            min: -3.5, max: 3.5
          },
          y: { 
            title: { display: true, text: '樣本標準化分位數 (Sample Quantiles)', color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.05)' }, 
            ticks: { color: '#94a3b8' },
            min: -3.5, max: 3.5
          }
        }
      }
    });
  }
}

// ====================================================================
// 5. 即時部署推論模擬器 (CRISP-DM 6: Deployment)
// ====================================================================

function renderDeploymentSimulator(res) {
  const container = document.getElementById('simulator-inputs-container');
  if (!container) return;
  container.innerHTML = '';

  document.getElementById('pred-target-label').textContent = `預測目標: ${res.targetName}`;

  res.featureNames.forEach(feat => {
    const summary = res.featureSummaries[feat];
    const div = document.createElement('div');
    div.className = 'slider-item';

    div.innerHTML = `
      <div class="slider-label-row">
        <span class="slider-feature-name">${feat}</span>
        <span class="slider-val-readout" id="slider-readout-${feat}">${summary.mean.toFixed(2)}</span>
      </div>
      <input type="range" class="styled-slider" id="slider-${feat}" 
             min="${summary.min}" max="${summary.max}" step="${summary.step}" value="${summary.mean}">
    `;

    container.appendChild(div);

    const slider = div.querySelector(`#slider-${feat}`);
    slider.addEventListener('input', () => {
      document.getElementById(`slider-readout-${feat}`).textContent = Number(slider.value).toFixed(2);
      updateLivePrediction();
    });
  });

  updateLivePrediction();
}

function updateLivePrediction() {
  const res = state.modelResults;
  if (!res) return;

  // 構造輸入向量 x0 = [1, f1, f2, ...]
  const x0 = new Float64Array(res.p + 1);
  x0[0] = 1.0;

  for (let j = 0; j < res.p; j++) {
    const feat = res.featureNames[j];
    const slider = document.getElementById(`slider-${feat}`);
    x0[j + 1] = slider ? Number(slider.value) : res.featureSummaries[feat].mean;
  }

  // 預測 yHat0 = beta^T x0
  let predVal = 0;
  for (let j = 0; j < res.p + 1; j++) {
    predVal += res.beta[j] * x0[j];
  }

  // 計算 leverage: x0^T (X^T X)^-1 x0
  let leverage = 0;
  for (let i = 0; i < res.p + 1; i++) {
    let tmp = 0;
    for (let j = 0; j < res.p + 1; j++) {
      tmp += x0[j] * res.XTX_inv[j][i];
    }
    leverage += tmp * x0[i];
  }
  leverage = Math.max(0, leverage);

  const seMean = Math.sqrt(res.s2 * leverage);
  const sePred = Math.sqrt(res.s2 * (1.0 + leverage));

  const ciLow = predVal - 1.96 * seMean;
  const ciHigh = predVal + 1.96 * seMean;
  const piLow = predVal - 1.96 * sePred;
  const piHigh = predVal + 1.96 * sePred;

  // 更新介面
  document.getElementById('pred-output-val').textContent = predVal.toFixed(4);
  document.getElementById('pred-ci-range').textContent = `95% 信賴區間 (CI): [${ciLow.toFixed(4)}, ${ciHigh.toFixed(4)}]`;
  document.getElementById('pred-pi-range').textContent = `95% 預測區間 (PI): [${piLow.toFixed(4)}, ${piHigh.toFixed(4)}]`;

  // 依照預測值動態調整警戒標籤 (若是洪水資料)
  const badgeEl = document.getElementById('pred-alert-badge');
  const badgeText = document.getElementById('alert-badge-text');

  if (res.targetName.toLowerCase().includes('flood')) {
    if (predVal >= 0.55) {
      badgeEl.style.borderColor = 'rgba(239, 68, 68, 0.6)';
      badgeEl.style.background = 'rgba(239, 68, 68, 0.2)';
      badgeEl.style.color = '#f87171';
      badgeText.textContent = '🔴 一級紅色警戒 (高度洪水致災風險)';
    } else if (predVal >= 0.52) {
      badgeEl.style.borderColor = 'rgba(245, 158, 11, 0.6)';
      badgeEl.style.background = 'rgba(245, 158, 11, 0.2)';
      badgeEl.style.color = '#fbbf24';
      badgeText.textContent = '🟡 二級黃色戒備 (中高度防汛戒備)';
    } else if (predVal >= 0.48) {
      badgeEl.style.borderColor = 'rgba(56, 189, 248, 0.6)';
      badgeEl.style.background = 'rgba(56, 189, 248, 0.2)';
      badgeEl.style.color = '#38bdf8';
      badgeText.textContent = '🟢 三級綠色安全 (水情常態監控中)';
    } else {
      badgeEl.style.borderColor = 'rgba(16, 185, 129, 0.6)';
      badgeEl.style.background = 'rgba(16, 185, 129, 0.2)';
      badgeEl.style.color = '#34d399';
      badgeText.textContent = '🔵 極低水患機率 (水庫蓄水量偏低)';
    }
  } else {
    // 一般數值或利潤
    badgeEl.style.borderColor = 'rgba(99, 102, 241, 0.6)';
    badgeEl.style.background = 'rgba(99, 102, 241, 0.2)';
    badgeEl.style.color = '#a5b4fc';
    badgeText.textContent = `🎯 預期數值: ${predVal >= 0 ? '+' : ''}${predVal.toFixed(2)}`;
  }
}

// ====================================================================
// 6. 事件監聽與切換控制
// ====================================================================

function initEventListeners() {
  // 1. Problem Selector 切換 (加分 1 分)
  const problemSelector = document.getElementById('problem-selector');
  problemSelector.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'flood') {
      loadFloodSampleDataset();
    } else if (val === 'startups') {
      loadStartupsDataset();
    } else if (val === 'custom') {
      setStatus("請在右方輸入自訂 CSV 網址，或點選上傳本地 CSV 檔案。", 'normal');
      document.getElementById('dataset-url-input').focus();
    }
  });

  // 2. URL 載入按鈕 (加分 2 分)
  const btnLoadUrl = document.getElementById('btn-load-url');
  btnLoadUrl.addEventListener('click', () => {
    const url = document.getElementById('dataset-url-input').value.trim();
    if (!url) {
      alert("請輸入欲載入之 CSV 資料集 URL！");
      return;
    }
    loadDatasetFromUrl(url);
  });

  // 3. 本地 CSV 上傳 (加分 2 分)
  const csvFileInput = document.getElementById('csv-file-input');
  csvFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setStatus(`正在解析本地檔案: ${file.name}...`, 'normal');
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: function(results) {
        processLoadedData(results);
      },
      error: function(err) {
        setStatus(`解析本地 CSV 失敗: ${err.message}`, 'error');
      }
    });
  });

  // 4. Target 變換監聽
  const targetSelect = document.getElementById('target-selector');
  targetSelect.addEventListener('change', (e) => {
    state.targetCol = e.target.value;
    // 特徵預設去除 target
    state.selectedFeatures = state.selectedFeatures.filter(f => f !== state.targetCol);
    renderFeaturesChecklist();
    recalculateModel();
  });

  // 5. 特徵全選 / 反選快捷動作
  document.getElementById('action-select-all').addEventListener('click', () => {
    state.selectedFeatures = state.numericColumns.filter(c => c !== state.targetCol);
    renderFeaturesChecklist();
  });

  document.getElementById('action-deselect-all').addEventListener('click', () => {
    state.selectedFeatures = [];
    renderFeaturesChecklist();
  });

  // 6. 特徵選擇模式切換 (Full, SelectKBest, RFE)
  const btnAll = document.getElementById('btn-select-all');
  const btnKBest = document.getElementById('btn-select-kbest');
  const btnRfe = document.getElementById('btn-select-rfe');
  const kSliderGroup = document.getElementById('k-slider-group');
  const kSlider = document.getElementById('k-slider');

  function clearActiveToggle() {
    [btnAll, btnKBest, btnRfe].forEach(b => b.classList.remove('active'));
  }

  btnAll.addEventListener('click', () => {
    clearActiveToggle();
    btnAll.classList.add('active');
    kSliderGroup.style.display = 'none';
    state.selectedFeatures = state.numericColumns.filter(c => c !== state.targetCol);
    renderFeaturesChecklist();
    recalculateModel();
  });

  btnKBest.addEventListener('click', () => {
    clearActiveToggle();
    btnKBest.classList.add('active');
    kSliderGroup.style.display = 'block';
    applyKBestSelection(parseInt(kSlider.value));
  });

  btnRfe.addEventListener('click', () => {
    clearActiveToggle();
    btnRfe.classList.add('active');
    kSliderGroup.style.display = 'block';
    applyRfeSelection(parseInt(kSlider.value));
  });

  kSlider.addEventListener('input', (e) => {
    const k = parseInt(e.target.value);
    document.getElementById('k-val-display').textContent = k;
    if (btnKBest.classList.contains('active')) {
      applyKBestSelection(k);
    } else if (btnRfe.classList.contains('active')) {
      applyRfeSelection(k);
    }
  });

  // 7. 重新訓練按鈕
  document.getElementById('btn-re-train').addEventListener('click', () => {
    recalculateModel();
  });

  // 8. 視覺化圖表 Tab 切換
  const chartTabs = document.querySelectorAll('.chart-tab');
  chartTabs.forEach(tabBtn => {
    tabBtn.addEventListener('click', () => {
      chartTabs.forEach(b => b.classList.remove('active'));
      tabBtn.classList.add('active');
      state.activeChartTab = tabBtn.getAttribute('data-chart');
      renderActiveChart();
    });
  });

  // 9. CRISP-DM 流程 Tab 點擊互動
  const crispTabs = document.querySelectorAll('.crisp-tab');
  crispTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      crispTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const step = tab.getAttribute('data-step');
      const targets = {
        '1': 'about-section',
        '2': 'data-config-panel',
        '3': 'data-config-panel',
        '4': 'metrics-panel',
        '5': 'charts-section',
        '6': 'deployment-section'
      };
      const el = document.getElementById(targets[step]);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // 10. 重設特徵為平均值按鈕
  const btnReset = document.getElementById('btn-reset-simulator');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (state.modelResults) {
        renderDeploymentSimulator(state.modelResults);
      }
    });
  }
}

// 實作 SelectKBest (以與 Target 之皮爾森相關係數絕對值排序)
function applyKBestSelection(k) {
  const candidateFeatures = state.numericColumns.filter(c => c !== state.targetCol);
  const n = state.rawDataset.length;

  const yVals = state.rawDataset.map(r => Number(r[state.targetCol]) || 0);
  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += yVals[i];
  yMean /= n;

  const scores = candidateFeatures.map(feat => {
    const xVals = state.rawDataset.map(r => Number(r[feat]) || 0);
    let xMean = 0;
    for (let i = 0; i < n; i++) xMean += xVals[i];
    xMean /= n;

    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xVals[i] - xMean;
      const dy = yVals[i] - yMean;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const r = (Math.sqrt(denX) * Math.sqrt(denY)) === 0 ? 0 : Math.abs(num / (Math.sqrt(denX) * Math.sqrt(denY)));
    return { feat, score: r };
  });

  scores.sort((a, b) => b.score - a.score);
  state.selectedFeatures = scores.slice(0, k).map(s => s.feat);
  renderFeaturesChecklist();
  recalculateModel();
}

// 實作 RFE 遞迴特徵消除 (每次訓練 OLS，剔除絕對回歸係數最小之特徵，直到剩下 k 項)
function applyRfeSelection(k) {
  let currentFeatures = state.numericColumns.filter(c => c !== state.targetCol);
  
  while (currentFeatures.length > k) {
    const res = runOLS(state.rawDataset, state.targetCol, currentFeatures);
    let minCoeff = Infinity;
    let worstFeat = '';

    for (let j = 1; j < res.coeffStats.length; j++) {
      const mag = Math.abs(res.coeffStats[j].coeff);
      if (mag < minCoeff) {
        minCoeff = mag;
        worstFeat = res.coeffStats[j].name;
      }
    }
    currentFeatures = currentFeatures.filter(f => f !== worstFeat);
  }

  state.selectedFeatures = currentFeatures;
  renderFeaturesChecklist();
  recalculateModel();
}

// ====================================================================
// 7. 資料集載入邏輯 (Problem 1, Problem 2, URL)
// ====================================================================

function loadFloodSampleDataset() {
  setStatus("正在載入 Problem 1: 洪水機率預測資料集 (flood_sample.csv)...", 'normal');
  Papa.parse('flood_sample.csv', {
    download: true,
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: function(results) {
      processLoadedData(results);
    },
    error: function(err) {
      console.warn("直接載入 flood_sample.csv 失敗，嘗試替代路徑:", err);
      // 容錯機制
      setStatus("請確認 flood_sample.csv 位於伺服器相同目錄下。", 'error');
    }
  });
}

function loadStartupsDataset() {
  setStatus("正在載入 Problem 2: 50 Startups 利潤預測資料集...", 'normal');
  const url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
  
  Papa.parse(url, {
    download: true,
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: function(results) {
      processLoadedData(results);
    },
    error: function(err) {
      console.warn("從 GitHub 遠端載入失敗 (可能為無網路或 CORS)，啟動內建資料容錯機制:", err);
      Papa.parse(EMBEDDED_STARTUPS_CSV, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(fallbackResults) {
          processLoadedData(fallbackResults);
        }
      });
    }
  });
}

function loadDatasetFromUrl(url) {
  const spinner = document.getElementById('url-spinner');
  spinner.classList.add('active');
  setStatus(`正在自 URL 載入資料: ${url} ...`, 'normal');

  Papa.parse(url, {
    download: true,
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: function(results) {
      spinner.classList.remove('active');
      processLoadedData(results);
    },
    error: function(err) {
      spinner.classList.remove('active');
      setStatus(`URL 載入失敗: ${err.message} (若是跨域受限，可下載後自本地上傳)`, 'error');
    }
  });
}

// ====================================================================
// 8. 系統開機初始化
// ====================================================================

window.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  // 系統預設載入 Problem 1
  loadFloodSampleDataset();
});
