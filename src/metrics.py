import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from plots import plot_pr_curve, plot_roc_curve
from scipy import interp
import logging


def log_metrics(ref, pred_probas, n_classes, digits=3, plot=False, folder_path=None, threshold=None):
    """
    ref: (n_samples) true class labels
    pred_probas: (n_samples, n_classes) if n_classes > 2, otherwise (n_samples). Classification probability for each class
    """
    binary = n_classes == 2

    one_hot_ref = label_binarize(ref, classes=list(range(n_classes)))

    if binary:
        if len(pred_probas.shape) > 1:
            if pred_probas.shape[1] == 1:
                pred_probas = np.ravel(pred_probas)
            else:
                pred_probas = pred_probas[:, 1]
        # Traing or validation set
        if not threshold:
            precision, recall, f1s, threshold = benchmark_best_f1s(pred_probas, one_hot_ref, max_iter=100, max_depth=4)
            pred_labels = np.where(pred_probas >= threshold, 1, 0)
        # Testing set, applying validation threshold
        else:
            pred_labels = np.where(pred_probas >= threshold, 1, 0)
            precision, recall, f1s = precision_metrics(one_hot_ref, pred_labels)

        pred_labels_max = np.where(pred_probas >= 0.5, 1, 0)
        precision_max, recall_max, f1s_max = precision_metrics(one_hot_ref, pred_labels_max)
    else:
        pred_labels_max = pred_probas.argmax(axis=1)

    micro_roc_auc, macro_roc_auc = compute_roc_auc(one_hot_ref, pred_probas, n_classes, folder_path=folder_path, plot=plot)
    _, _, ap = compute_pr_auc(one_hot_ref, pred_probas, n_classes, folder_path=folder_path, plot=plot)

    output = ['']

    if binary:
        # We apply the threshold corresponding to the best F1 score on the validation set
        output.extend([
                   '================================',
                   '= Best threshold-based metrics =',
                   '================================',
                   'Threshold: {1:.{0}f}'.format(digits, threshold),
                   'Confusion matrix:',
                   str(metrics.confusion_matrix(ref, pred_labels)),
                   'Accuracy: {1:.{0}f}'.format(digits, metrics.accuracy_score(ref, pred_labels)),
                   'Prec    : {1:.{0}f}'.format(digits, precision),
                   'Recall  : {1:.{0}f}'.format(digits, recall),
                   'f1-score: {1:.{0}f}'.format(digits, f1s)])

    # Predicted labels are simply the class with the highest probability (hence t=0.5)
    output.extend(['===============================',
                   '=== Threshold-based metrics ===',
                   '===============================',
                   'Threshold: {1:.{0}f}'.format(digits, 0.5),
                   'Confusion matrix:',
                   str(metrics.confusion_matrix(ref, pred_labels_max)),
                   'Accuracy: {1:.{0}f}'.format(digits, metrics.accuracy_score(ref, pred_labels_max))])
    if binary:
        # Precision, recall and F1 available for binary classification only
        output.extend([
                   'Prec    : {1:.{0}f}'.format(digits, precision),
                   'Recall  : {1:.{0}f}'.format(digits, recall),
                   'f1-score: {1:.{0}f}'.format(digits, f1s)])

    # Area under the curves and log-loss, no threshold involved
    output.extend(['===============================',
                   '= Non-threshold-based metrics =',
                   '===============================',
                   'ROC AUC    : {1:.{0}f} (micro), {1:.{0}f} (macro)'.format(digits, micro_roc_auc, digits, macro_roc_auc),
                   'PR AUC (AP): {1:.{0}f}'.format(digits, ap),
                   'log-loss   : {1:.{0}f}'.format(digits, np.nan if len(set(ref)) == 1 else metrics.log_loss(ref, pred_probas)),
                   '========================='])

    logging.info('\n'.join(output))
    return threshold


def compute_roc_auc(y_true, pred_probas, n_classes, folder_path=None, plot=True):
    """
    Compute the ROC curve and ROC AUC for each class
    Micro-auc: the final AUC is weighted per class proportion
    Macro-auc: the final AUC gives each class the same weight
    y_true: (m_samples, n_classes)
    pred_probas: (m_samples, n_classes)
    """
    fpr, tpr, roc_auc = {}, {}, {}

    if n_classes == 2:
        fpr[0], tpr[0], _ = metrics.roc_curve(y_true, pred_probas)
        roc_auc[0] = metrics.auc(fpr[0], tpr[0])
        roc_auc["micro"], roc_auc["macro"] = roc_auc[0], roc_auc[0]
    else:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], pred_probas[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # --- Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), pred_probas.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # --- Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    if plot:
        plot_roc_curve(fpr, tpr, roc_auc, n_classes, folder_path)

    return roc_auc["micro"], roc_auc["macro"]


def compute_pr_auc(y_true, pred_probas, n_classes, folder_path=None, plot=True):
    """
    Compute the PR curve and PR AUC (average precision) for each class
    The average precision is computed with micro-averaging, thus weighting the AUC per class proportion
    y_true: (m_samples, n_classes)
    pred_probas: (m_samples, n_classes)
    """
    if n_classes == 2:
        precision, recall, _ = metrics.precision_recall_curve(y_true, pred_probas)
        ap = metrics.average_precision_score(y_true, pred_probas)
    else:
        precision, recall, ap = {}, {}, {}
        for i in range(n_classes):
            precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i], pred_probas[:, i])
            ap[i] = metrics.average_precision_score(y_true[:, i], pred_probas[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(), pred_probas.ravel())
        ap["micro"] = metrics.average_precision_score(y_true, pred_probas, average="micro")

    if plot:
        plot_pr_curve(precision, recall, ap, n_classes, folder_path)

    if n_classes != 2:
        precision, recall, ap = precision['micro'], recall['micro'], ap['micro']

    return precision, recall, ap


def precision_metrics(y_true, y_predicted):
    p = metrics.precision_score(y_true, y_predicted)
    r = metrics.recall_score(y_true, y_predicted)
    f1 = metrics.f1_score(y_true, y_predicted)
    return p, r, f1


def benchmark_best_f1s(probas, labels, max_iter=100, max_depth=1, depth=0, min_t=None, max_t=None, res=None):
    """
    Recursively find the best F1 score based on a set of probabilities and true labels
    probas: probability of an observation to be positive (n_samples)
    labels: observation labelled as 1 are positive (n_samples)
    """
    if depth == 0:
        res = []
        min_t, max_t = min(probas), max(probas)
    thresholds = np.linspace(min_t, max_t, int(np.ceil(max_iter/2.0)))
    if len(thresholds) == 0:
        thresholds = [0.5 * (min_t + max_t)]

    for t in thresholds:  # Log likelihood is < 0
        Y_predicted = np.where(probas >= t, 1, 0)
        p, r, f1 = precision_metrics(labels, Y_predicted)
        res.append((t, p, r, f1))

    res = np.array(res)
    best_res = res[res[:, 3].argmax()]

    if len(thresholds) == 1:
        return best_res[1] * 100, best_res[2] * 100, best_res[3], thresholds[0]

    step = (max_t - min_t) / max_iter * 3
    return benchmark_best_f1s(probas, labels, max_iter=max_iter/2, max_depth=max_depth, depth=depth+1,
                              min_t=best_res[0] - step, max_t=best_res[0] + step, res=res.tolist())


if __name__ == '__main__':
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    n_classes = 3  # 2

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    if n_classes == 2:
        X = X[y != 2]
        y = y[y != 2]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    log_metrics(y_test, y_score, n_classes=n_classes, plot=True)
