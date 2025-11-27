import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. 設定頁面配置
st.set_page_config(
    page_title="垃圾分類識別系統",
    page_icon="♻️",
    layout="centered"
)

# 2. 標題與說明
st.title("♻️ 深度學習垃圾分類 Demo")
st.markdown("""
這是一個使用 **MobileNetV2** 與 **Transfer Learning** 訓練的影像辨識系統。
請上傳一張垃圾照片（如：寶特瓶、玻璃罐、紙箱），系統將會自動判斷類別。
""")

# 3. 載入模型 (使用快取，避免每次操作都重新載入)
@st.cache_resource
def load_model():
    # 請確保你的模型檔名與這裡一致，如果你的檔名不同，請修改這裡
    model_path = 'models/garbage_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"找不到模型檔案，請確認 {model_path} 是否存在。錯誤訊息: {e}")
        return None

with st.spinner('正在載入 AI 模型...'):
    model = load_model()

# 4. 定義類別名稱 (這是 Kaggle Garbage Classification 的標準順序)
# 順序必須與 train.py 訓練時的 class_indices 一致 (通常是按字母順序)
CLASS_NAMES = ['紙板 (Cardboard)', '玻璃 (Glass)', '金屬 (Metal)', '紙張 (Paper)', '塑膠 (Plastic)', '一般垃圾 (Trash)']

# 5. 圖片預處理函數
def process_image(image_data):
    """
    將圖片處理成模型看得懂的格式：
    1. 調整大小至 (224, 224)
    2. 轉換為陣列
    3. 歸一化 (除以 255，將數值壓在 0~1 之間，對應 train.py 的 rescale=1./255)
    """
    size = (224, 224)
    # 使用 LANCZOS 演算法進行高品質縮放
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # 歸一化 (必須與 train.py 一致)
    normalized_image_array = img_array.astype(np.float32) / 255.0
    
    # 增加一個維度 (Batch Size)，變成 (1, 224, 224, 3)
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# 6. 使用者介面 - 檔案上傳
uploaded_file = st.file_uploader("請選擇一張圖片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 顯示上傳的圖片
    image = Image.open(uploaded_file)
    st.image(image, caption='您上傳的圖片', use_column_width=True)
    
    # 進行預測
    st.write("🔍 AI 正在分析中...")
    
    # 預處理圖片
    data = process_image(image)
    
    # 模型推論
    prediction = model.predict(data)
    
    # 取得最高機率的類別
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    # 顯示結果
    st.markdown("---")
    if confidence > 0.6: # 信心度門檻
        st.success(f"識別結果：**{predicted_class_name}**")
        st.info(f"信心指數：**{confidence * 100:.2f}%**")
    else:
        st.warning(f"識別結果可能是：**{predicted_class_name}** (但我不確定 🤔)")
        st.caption(f"信心指數：{confidence * 100:.2f}%")
        
    # 顯示詳細機率長條圖
    st.markdown("### 詳細預測數據")
    st.bar_chart(dict(zip(CLASS_NAMES, prediction[0])))

elif model is None:
    st.warning("⚠️ 模型載入失敗，無法進行預測，請檢查 models 資料夾。")