import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.datasets import mnist

# 載入並預處理資料 
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# 攤平成 784 維向量並標準化
xtrain = xtrain.reshape(-1, 28*28).astype('float32') / 255
xtest = xtest.reshape(-1, 28*28).astype('float32') / 255

# 取部分資料加速訓練
xtrain_small, _, ytrain_small, _ = train_test_split(
    xtrain, ytrain, train_size=5000, stratify=ytrain, random_state=42
)
xtest_small, _, ytest_small, _ = train_test_split(
    xtest, ytest, train_size=1000, stratify=ytest, random_state=42
)

# GridSearchCV 調參 
param_grid = {
    'C': [1, 5, 10],
    'gamma': [0.01, 0.05, 0.1],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    svm.SVC(),
    param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("🔍 Running Grid Search... (this may take a while)")
grid.fit(xtrain_small, ytrain_small)

print("\n✅ Best Parameters:", grid.best_params_)
clf = grid.best_estimator_

# 模型預測與準確率 
ypred = clf.predict(xtest_small)
print("Accuracy:", accuracy_score(ytest_small, ypred))
print("Classification Report:\n", classification_report(ytest_small, ypred))

# 預測視覺化（5張隨機） 
for i in range(5):
    img = xtest_small[i].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.title(f'True: {ytest_small[i]}, Pred: {ypred[i]}')
    plt.axis('off')
    plt.show()

# 混淆矩陣視覺化 =====
cm = confusion_matrix(ytest_small, ypred)
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification Report 可視化（熱度圖）=====
report = classification_report(ytest_small, ypred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
sns.heatmap(df_report.iloc[:-1, :3], annot=True, cmap="YlGnBu")
plt.title("Classification Report (Precision / Recall / F1-score)")
plt.show()
