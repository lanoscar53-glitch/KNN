# -*- coding: utf-8 -*-
"""
KNN 猫狗分类（纯 NumPy 手写实现，不依赖 sklearn）
运行前：pip install numpy matplotlib
"""
import os
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*align should be passed.*")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置区 ==================
DATA_DIR = r"C:\Users\stary\Documents\kimi\workspace\data\cifar-10-batches-py"  # 本地数据路径
CAT, DOG = 3, 5                                # CIFAR-10 中 cat=3, dog=5
CLASSES = ["cat", "dog"]
K_LIST = [1, 3, 5, 7, 9, 11, 15, 21]           # 参与比较的 K 值
N_TRAIN, N_TEST = 3000, 50                     # 抽样数量（最大 10000 / 2000）

# ================== 1. 数据加载 ==================
def load_batch(path):
    """读取 CIFAR-10 的 pickle 批次文件。"""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], np.array(batch[b"labels"])

def to_gray(X):
    """3072 维 (R+G+B) 转灰度并归一化到 [0,1]。"""
    r, g, b = X[:, :1024], X[:, 1024:2048], X[:, 2048:]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32) / 255.0

def load_cat_dog(root):
    """加载本地数据，筛出猫狗两类，划分训练/测试集并抽样。"""
    X_tr, y_tr = [], []
    for i in range(1, 6):                                # 训练集：5 个批次合并
        data, labels = load_batch(os.path.join(root, f"data_batch_{i}"))
        X_tr.append(data)
        y_tr.append(labels)
    X_tr, y_tr = np.vstack(X_tr), np.concatenate(y_tr)
    X_te, y_te = load_batch(os.path.join(root, "test_batch"))  # 测试集

    def filter_cat_dog(X, y, n):
        mask = (y == CAT) | (y == DOG)                   # 只保留猫和狗
        X, y = X[mask], (y[mask] == DOG).astype(int)     # 猫->0, 狗->1
        idx = np.random.choice(len(X), min(n, len(X)), replace=False)
        return to_gray(X[idx]), y[idx], X[idx]           # 保留原图用于可视化

    X_train, y_train, _ = filter_cat_dog(X_tr, y_tr, N_TRAIN)
    X_test, y_test, X_test_raw = filter_cat_dog(X_te, y_te, N_TEST)
    return X_train, y_train, X_test, y_test, X_test_raw

# ================== 2. 纯 NumPy 实现 KNN ==================
def knn_predict(X_train, y_train, X_test, k):
    """
    KNN 分类（手写实现）：
    1) 计算每个测试样本到全部训练样本的欧氏距离
       ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b （矩阵化，避免双重循环）
    2) 取距离最近的 k 个邻居，多数投票决定类别
    """
    train_sq = np.sum(X_train ** 2, axis=1)              # ||a||^2, 形状 (N_train,)
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)  # ||b||^2, 形状 (N_test, 1)
    dist2 = test_sq + train_sq - 2 * X_test @ X_train.T  # 距离平方矩阵 (N_test, N_train)

    knn_idx = np.argsort(dist2, axis=1)[:, :k]           # 每行取最近 k 个邻居的下标
    knn_labels = y_train[knn_idx]                        # 邻居的类别 (N_test, k)
    # 多数投票：类别为 0/1，均值 >= 0.5 判为狗(1)，否则猫(0)；k 为偶数平局时判为猫
    return (knn_labels.mean(axis=1) >= 0.5).astype(int)

def accuracy(y_true, y_pred):
    """准确率 = 预测正确的比例。"""
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred):
    """手算二分类的 precision / recall / F1 并打印。"""
    print(f"{'':>10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for c, name in enumerate(CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))       # 真正例
        fp = np.sum((y_pred == c) & (y_true != c))       # 假正例
        fn = np.sum((y_pred != c) & (y_true == c))       # 假反例
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{name:>10}{p:>10.2f}{r:>10.2f}{f1:>10.2f}{np.sum(y_true == c):>10d}")

# ================== 3. 选 K（错误率比较） ==================
def select_best_k(X_train, y_train, X_test, y_test):
    """遍历 K_LIST，记录各 K 的错误率，返回 (最佳K, 错误率列表)。"""
    error_rates = []
    for k in K_LIST:
        err = 1 - accuracy(y_test, knn_predict(X_train, y_train, X_test, k))
        error_rates.append(err)
        print(f"K={k:>2d}  测试准确率={1 - err:.4f}  错误率={err:.4f}")
    return K_LIST[int(np.argmin(error_rates))], error_rates

# ================== 4. 可视化 ==================
def plot_error_curve(error_rates, best_k):
    """错误率-K 曲线。"""
    plt.figure(figsize=(8, 5))
    plt.plot(K_LIST, error_rates, "o-", color="steelblue")
    plt.axvline(best_k, ls="--", color="red", alpha=0.6, label=f"最佳 K={best_k}")
    plt.xlabel("K 值（近邻个数）")
    plt.ylabel("错误率")
    plt.title("KNN 错误率随 K 值变化曲线")
    plt.xticks(K_LIST)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_error_curve.png", dpi=150)
    plt.show()

def plot_predictions(X_train, y_train, X_test, X_test_raw, y_test, best_k, n=12):
    """随机抽 n 张测试图展示识别结果（绿=正确，红=错误）。"""
    idx = np.random.choice(len(X_test), n, replace=False)
    rows = int(np.ceil(n / 4))
    plt.figure(figsize=(12, 3 * rows))
    for i, j in enumerate(idx):
        img = X_test_raw[j].reshape(3, 32, 32).transpose(1, 2, 0)  # 还原彩色图
        pred = knn_predict(X_train, y_train, X_test[j:j + 1], best_k)[0]
        plt.subplot(rows, 4, i + 1)
        plt.imshow(img)
        plt.title(f"预测: {CLASSES[pred]}\n真实: {CLASSES[y_test[j]]}",
                  color="green" if pred == y_test[j] else "red", fontsize=11)
        plt.axis("off")
    plt.suptitle("测试集图片识别结果（绿=正确，红=错误）", fontsize=14)
    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150)
    plt.show()

# ================== 主流程 ==================
def main():
    np.random.seed(42)
    X_train, y_train, X_test, y_test, X_test_raw = load_cat_dog(DATA_DIR)
    print(f"训练集: {X_train.shape[0]} 张   测试集: {X_test.shape[0]} 张")

    best_k, error_rates = select_best_k(X_train, y_train, X_test, y_test)
    plot_error_curve(error_rates, best_k)

    print(f"\n使用最佳 K={best_k} 进行最终评估...")
    y_pred = knn_predict(X_train, y_train, X_test, best_k)
    print(f"最终测试准确率: {accuracy(y_test, y_pred):.4f}")
    classification_report(y_test, y_pred)

    plot_predictions(X_train, y_train, X_test, X_test_raw, y_test, best_k)
    print("完成！已生成 knn_error_curve.png 和 test_predictions.png")

if __name__ == "__main__":
    main()
