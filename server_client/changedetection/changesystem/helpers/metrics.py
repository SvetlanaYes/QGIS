import torch
import numpy as np
import pandas as pd


# from sklearn.metrics import confusion_matrix


class Metrics:

    def __init__(self, metric_names=["IoU", "Precision", "Recall", "F1score"], eps=1e-7):

        self.metrics = {key: 0 for key in metric_names}
        self.c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        self.eps = eps
        self.metric_mapper = {"IoU": self._compute_IoU,
                              "Precision": self._compute_Precision,
                              "Recall": self._compute_Recall,
                              "F1score": self._compute_F1score}

    def _update_metrics_with_batch(self, logits, targets):

        labels = targets.detach().cpu().numpy().flatten()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().flatten()

        label_values = np.unique(labels)

        if (label_values == 0).all():
            tn, fp, fn, tp = (labels == preds).sum(), 0, 0, 0
        elif (label_values == 1).all():
            tn, fp, fn, tp = 0, 0, (labels != preds).sum(), (labels == preds).sum()
        else:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        self.c_matrix['tn'] += tn
        self.c_matrix['fp'] += fp
        self.c_matrix['fn'] += fn
        self.c_matrix['tp'] += tp

    def _compute_IoU(self):
        self.metrics["IoU"] = self.c_matrix['tp'] / (self.c_matrix['tp'] +
                                                     self.c_matrix['fp'] +
                                                     self.c_matrix['fn'] +
                                                     self.eps)

    def _compute_Precision(self):
        self.metrics["Precision"] = self.c_matrix['tp'] / (self.c_matrix['tp'] +
                                                           self.c_matrix['fp'] +
                                                           self.eps)

    def _compute_Recall(self):
        self.metrics["Recall"] = self.c_matrix['tp'] / (self.c_matrix['tp'] +
                                                        self.c_matrix['fn'] +
                                                        self.eps)

    def _compute_F1score(self):
        P = self.metrics["Precision"]
        R = self.metrics["Recall"]
        self.metrics["F1score"] = 2 * P * R / (R + P + self.eps)

    def _compute_all_metrics(self):

        for key in self.metrics.keys():
            self.metric_mapper[key]()


def make_metrics_dataframe(model_names, method_names, metric_names):
    col_names = [(x, y) for x in method_names for y in metric_names]
    row_count = len(model_names)
    col_count = len(method_names)
    data = np.zeros((row_count, col_count * len(metric_names)))
    data_frame = pd.DataFrame(data, columns=col_names, index=model_names)
    data_frame.columns = pd.MultiIndex.from_tuples(data_frame.columns)
    return data_frame
