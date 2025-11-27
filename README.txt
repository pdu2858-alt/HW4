DEMO網頁：https://wws3xp4oatvwy92jvgegjc.streamlit.app/

專題摘要 (Abstract)
隨著環境保護意識的提升，精確的垃圾分類已成為資源回收的關鍵環節，然而傳統的人工辨識方式既耗時又容易產生誤判。本專案旨在開發一套基於深度學習（Deep Learning）的自動化垃圾影像分類系統，以輔助解決日常回收分類難題。

在技術實作上，本研究使用 Python 語言結合 TensorFlow 深度學習框架，採用遷移學習（Transfer Learning）技術，以預訓練的 MobileNetV2 模型為核心進行微調（Fine-tuning）。模型針對包含玻璃、紙張、紙板、塑膠、金屬及一般垃圾等六大類別的 Garbage Classification Dataset 進行訓練，以達到輕量化與高準確度的平衡。

為提升系統的實用性與易用性，本專案進一步整合 Streamlit 框架開發互動式網頁應用程式（Web App）。使用者僅需上傳物品圖片，系統即可即時回傳預測類別與信心指數（Confidence Score）。本實作不僅驗證了卷積神經網路（CNN）在物體識別上的優越效能，亦展示了 AI 模型落地應用於智慧生活場景的潛力與完整解決方案。
