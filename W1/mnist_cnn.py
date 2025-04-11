import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# 載入 MNIST 資料集
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

# reshape 為 CNN 所需格式，並標準化
xtrain = xtrain.reshape(-1, 28, 28, 1).astype('float32') / 255
xtest = xtest.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot 編碼
ytrain = tf.keras.utils.to_categorical(ytrain, 10)
ytest = tf.keras.utils.to_categorical(ytest, 10)

# 建立 CNN 模型（加入 Dropout）
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Dropout 加在 dense 層後面
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping 回調函數
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 訓練模型
history = model.fit(
    xtrain, ytrain,
    epochs=20,
    batch_size=128,
    validation_data=(xtest, ytest),
    callbacks=[early_stop]
)

# 測試資料準確率
test_loss, test_acc = model.evaluate(xtest, ytest)
print('Test accuracy:', test_acc)

# 視覺化準確率與損失變化
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 顯示隨機預測結果
for i in range(5):
    idx = random.randint(0, xtest.shape[0] - 1)
    img = xtest[idx].reshape(28,28)
    label = np.argmax(ytest[idx])
    pred = np.argmax(model.predict(xtest[idx].reshape(1,28,28,1), verbose=0))
    plt.imshow(img, cmap='gray')
    plt.title(f'True: {label}, Pred: {pred}')
    plt.show()
