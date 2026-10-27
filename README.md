# HW2 多元線性回歸分析系統 (Multiple Linear Regression)
## 遵循 CRISP-DM 流程之互動式分析平台與預測系統

- **學生學號**：7115093066
- **課程名稱**：國立中興大學 巨量資料分析與機器學習
- **資料集來源**：
  - **Problem 1 (主要問題)**：[Kaggle Playground Series (Season 4, Episode 5) - Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5)（20 項環境工程數值特徵）
  - **Problem 2 (講義問題)**：[50 Startups Dataset](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)（3 項研發、管理、行銷特徵預測利潤）
  - **Problem 3 (自訂載入)**：支援任意公開 CSV 資料集 URL 或本地檔案動態解析建模

---

### 一、作業繳交架構與加分亮點 (對應 HW2 新版作業說明)

本專案完全遵循**新版作業說明規範**（主要繳交：GitHub Repository 與 Web 靜態發布平台），並完整實作各項**加分項目**：

```text
7115093066_hw2.zip
├── index.html                      # 【主要繳交】Web 互動式分析平台主頁 (支援 GitHub Pages)
├── style.css                       # 【Web 樣式】深色玻璃擬態 (Dark Glassmorphism) 響應式佈局
├── app.js                          # 【Web 核心】純前端 OLS 矩陣引擎、動態資料載入、區間計算與即時推論
├── flood_sample.csv                # Web 端預設輕量資料集 (1,000 筆，20 項特徵)
├── README.md                       # 本專案說明文件
├── 7115093066_hw2_report.pdf       # 【補充審查】完整分析報告書 (CRISP-DM、五大假設、95% CI/PI)
├── 7115093066_gpt_record.pdf       # 【補充審查】ChatGPT 對話紀錄存檔 (符合官方匯出規格)
├── 7115093066_hw2.ipynb            # 【補充審查】Jupyter Notebook 主程式 (已完整執行並留存圖表)
├── 7115093066_hw2.py               # 主程式 Python 腳本版本
└── images/                         # 高解析度分析視覺化圖檔
    ├── fig1_target_dist.png        # 目標變數分佈與常態擬合
    ├── fig2_correlation_matrix.png # 20項特徵相關性熱圖
    ├── fig3_feature_importance.png # 標準化回歸係數對比
    ├── fig4_prediction_intervals.png# 預測圖疊加 95% 信賴區間 (CI) 與預測區間 (PI)
    ├── fig5_regression_diagnostics.png# 五大回歸假設殘差診斷圖 (LOWESS, Q-Q, Scale-Location)
    └── fig6_model_comparison.png   # 多元回歸、特徵選擇與 Kaggle SOTA 模型對比
```

#### 🌟 核心加分功能說明 (滿分 +3 分)：
1. **【加分 1 分】下拉視窗切換不同 Problem**：
   - 介面頂端提供下拉選單，可無縫切換 **Problem 1 (洪水機率預測)** 與 **Problem 2 (50 Startups 利潤預測)**。切換後系統立即自動解析欄位、重新訓練 OLS 模型並即時更新儀表板與推論滑桿。
2. **【加分 2 分】任意輸入 Dataset URL 或上傳本地 CSV**：
   - 使用者可於輸入框貼上任何公開 CSV 檔案之 URL（例如各類開源回歸資料集），或直接從本機拖曳上傳 CSV。前端透過 PapaParse 即時解析數值特徵，自動推薦 Target 目標欄位並建立多元回歸方程式。

---

### 二、GitHub Pages 部署指引 (主要繳交方式)

本 Web 平台採用純前端架構（Vanilla HTML5 / CSS3 / JavaScript + Chart.js + PapaParse），無需任何伺服器端後端支援，可直接部署於 GitHub Pages：

1. **建立 GitHub 儲存庫**：
   - 將專案根目錄下之 `index.html`、`style.css`、`app.js`、`flood_sample.csv`、`7115093066_hw2_report.pdf`、`7115093066_gpt_record.pdf` 推送至 GitHub 倉庫（例如 `https://github.com/YourUsername/HW2-Multiple-Linear-Regression`）。
2. **啟用 GitHub Pages**：
   - 進入儲存庫之 **Settings** -> **Pages**。
   - **Branch** 選擇 `main` 分支，資料夾選擇 `/ (root)`，點擊 **Save**。
   - 約 1 分鐘後即可取得公開網址（如：`https://YourUsername.github.io/HW2-Multiple-Linear-Regression/`）。
3. **本地預覽方式**：
   - 亦可直接在專案目錄下以 Python 啟動輕量伺服器：
     ```bash
     python -m http.server 8000
     ```
   - 於瀏覽器開啟 `http://localhost:8000/` 即可進行互動操作。

---

### 三、CRISP-DM 六大階段亮點摘要

1. **Business Understanding (商業理解)**：以 20 項環境工程與治理特徵預測連續型洪水發生機率（`FloodProbability`），提供防汛調度、工程預算效益評估與產險定價之白箱客觀基準。
2. **Data Understanding (資料理解)**：20 項特徵全為數值指標，0 缺失值，母體呈高度對稱之常態鐘形分佈。
3. **Data Preparation (資料準備)**：80/20 訓練/測試切分，防範資料外洩，採用 `StandardScaler` 標準化特徵，確保回歸係數具備可比性。
4. **Modeling (建模與特徵選擇)**：
   - 建立基準 OLS 模型（$R^2 = 0.8496$）。
   - 交叉驗證優化 Ridge (L2) 與 Lasso (L1) 正則化。
   - 特徵選擇實驗：`SelectKBest` (F-regression, 14特徵, $R^2=0.5519$) vs `RFE` (遞迴消除, 14特徵, $R^2=0.5615$)，論證了 20 項特徵彼此正交且具備均勻邊際貢獻之結構特性。
5. **Evaluation (模型評估與五大假設檢驗)**：
   - 嚴格檢驗線性、殘差常態性（Q-Q Plot）、同質變異性（LOWESS 殘差水平平直）、獨立性（Durbin-Watson = 2.0188）與多重共線性（平均 VIF = 1.0030）。
   - 繪製**疊加 95% 信賴區間 (CI) 與 95% 預測區間 (PI)** 之高說服力視覺化圖表。
6. **Deployment (模型部署與即時推論)**：
   - 於 Web 介面封裝「即時推論模擬器」，使用者拖動各特徵數值滑桿，系統即時推算點估計值、95% CI、95% PI 及四級防汛警戒標籤。

---

### 四、AI 協助研究規範證明

- **NotebookLM 研究摘要**：收錄於報告書第八章及 Web 平台底部，深入研析 Kaggle 頂尖解法與線性模型白箱可解釋性優勢（全文 328 字，符合 100 字以上規範）。
- **ChatGPT 輔助紀錄**：收錄於報告書第七章並附錄完整 PDF 存檔 (`7115093066_gpt_record.pdf`)，依據參考範例之對話格式精確輸出。
