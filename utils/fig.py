import itertools

import numpy as np
from matplotlib import pyplot as plt


def draw_confusion_matrix(confusion, class_names):
    """
    绘制混淆矩阵
    :param confusion: 混淆矩阵
    :param class_names: 类别名
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion.max() / 2.0
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(
            j,
            i,
            f"{confusion[i, j]:.0f}",
            horizontalalignment="center",
            color="white" if confusion[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return plt

def draw_normalized_confusion_matrix(confusion, class_names):
    """
    绘制归一化混淆矩阵
    :param confusion: 混淆矩阵
    :param class_names: 类别名
    """
    confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_normalized, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion.max() / 2.0
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(
            j,
            i,
            f"{confusion[i, j]:.0f}\n{confusion_normalized[i, j]:.2f}",
            horizontalalignment="center",
            color="white" if confusion[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return plt

if __name__ == "__main__":
    confusion = np.random.randint(0, 100, (20, 20))
    class_names = [f"class_{i}" for i in range(20)]
    draw_confusion_matrix(confusion, class_names).show()
    draw_normalized_confusion_matrix(confusion, class_names).show()