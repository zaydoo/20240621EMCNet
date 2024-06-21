from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm


logits = np.load("logits_array.npy")
print(logits.shape)
hsi_gt_b = np.load("hsi_gt_b.npy")
print(hsi_gt_b.shape)
label = np.zeros((logits.shape[0],))
for i in range(hsi_gt_b.shape[0]):
    for j in range(hsi_gt_b.shape[1]):
        label[i*716 + j] = hsi_gt_b[i][j]
print(logits)
print(label)
print(logits.shape)
print(label.shape)
# 创建一个 SVM 模型
clf = svm.SVC(kernel='rbf', gamma=0.7, C=1)
# 训练模型
clf.fit(logits, label)

plt.scatter(logits[:, 0], logits[:, 1], c=label, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 生成网格点来绘制决策边界
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# 将决策边界画出来
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 在图上标注支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')

plt.title('SVM 二分类训练结果')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

# 使用测试数据进行预测
y_pred = clf.predict(logits)

# 计算准确度
accuracy = accuracy_score(label, y_pred)
print(f'准确度：{accuracy:.2f}')

kc1 = cohen_kappa_score(label, y_pred)
print(f'kappa：{kc1:.2f}')

# 计算精确度
precision = precision_score(label, y_pred)
print(f'精确度：{precision:.2f}')

# 计算召回率
recall = recall_score(label, y_pred)
print(f'召回率：{recall:.2f}')

# 计算 F1 分数
f1 = f1_score(label, y_pred)
print(f'F1 分数：{f1:.2f}')

# 打印混淆矩阵
conf_matrix = confusion_matrix(label, y_pred)
print('混淆矩阵：')
print(conf_matrix)
