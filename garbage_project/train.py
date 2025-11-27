import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# 設定參數
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # 因為用遷移學習，5-10輪通常就夠了
DATA_DIR = 'garbage_project/dataset' # 確保你的圖片資料夾名稱是這個

# 1. 資料增強與載入
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 80% 訓練, 20% 驗證
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. 建立模型 (使用 MobileNetV2 遷移學習)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # 凍結預訓練層

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(6, activation='softmax')(x) # 6類垃圾

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. 訓練模型
print("開始訓練模型...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 4. 儲存模型
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/garbage_model.h5')
print("模型已儲存至 models/garbage_model.h5")

# 輸出類別索引，供 app.py 使用
print("類別對應:", train_generator.class_indices)