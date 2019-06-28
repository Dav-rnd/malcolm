import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os

sns.set_style("whitegrid")


def plot_roc_curve(fpr, tpr, roc_auc, n_classes, folder_path):
    # Plot all ROC curves, in addition to the averaging micro and macro ROC curves
    plt.figure(figsize=(7, 6))

    if n_classes == 2:
        plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
    else:
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = sns.color_palette('hls', n_classes)
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right', prop={'size': 12})
    filepath = 'roc.png'
    if folder_path is not None:
        filepath = os.path.join(folder_path, filepath)
    plt.savefig(filepath, bbox_inches='tight')
    logging.info('ROC plot saved in {}'.format(filepath))


def plot_pr_curve(precision, recall, ap, n_classes, folder_path=None):
    plt.figure(figsize=(7, 6))
    lines = []
    labels = []
    colors = sns.color_palette('hls', n_classes)

    # Iso F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('Iso-f1 curves')

    if n_classes == 2:
        l, = plt.plot(recall, precision, color='darkorange', lw=2)
        lines.append(l)
        labels.append('PR curve (area = %0.2f)' % ap)
    else:
        # Precision-recall
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('Micro-average precision (area = {0:0.2f})'.format(ap["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'.format(i, ap[i]))

    if n_classes > 2:
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall')
    plt.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.48, -0.55 if n_classes > 2 else -0.25),
               frameon=False, prop={'size': 12})
    filepath = 'pr.png'
    if folder_path is not None:
        filepath = os.path.join(folder_path, filepath)
    plt.savefig(filepath, bbox_inches='tight')
    logging.info('Precision-recall plot saved in {}'.format(filepath))
