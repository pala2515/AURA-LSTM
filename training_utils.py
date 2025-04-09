import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class MetricsCallback(callbacks.Callback):
    def __init__(self, validation_data, use_binary_metrics=True):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.x_val, self.y_val_onehot = validation_data
        # Determine if it's binary based on the shape of y_val_onehot or explicitly passed
        self.is_binary = use_binary_metrics or (self.y_val_onehot.shape[1] == 2 or self.y_val_onehot.shape[1] == 1)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_prob = self.model.predict(self.x_val, verbose=0)

        if self.y_val_onehot.shape[1] > 1 : # One-hot encoded labels
             y_true = np.argmax(self.y_val_onehot, axis=1)
        else: # Single column binary labels (0 or 1) - Should ideally be one-hot encoded
             y_true = self.y_val_onehot.flatten().astype(int)


        if y_pred_prob.shape[1] > 1: # Softmax/Sigmoid output for multi-class/binary
            y_pred = np.argmax(y_pred_prob, axis=1)
            # For AUC with multi-class OVR/OVO, we need probabilities per class
            # For binary case represented as 2 columns (softmax), use prob of positive class
            y_pred_prob_for_auc = y_pred_prob[:, 1] if self.is_binary and y_pred_prob.shape[1] == 2 else y_pred_prob
            average_type = 'weighted' # Use weighted for multi-class, can use 'binary' if truly binary
            if self.is_binary: average_type = 'binary'

        elif y_pred_prob.shape[1] == 1: # Sigmoid output for binary
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            y_pred_prob_for_auc = y_pred_prob.flatten()
            average_type = 'binary'
        else:
             print("\nWarning: Unexpected prediction shape in MetricsCallback.")
             return logs # Return original logs if predictions are unusable


        try:
            precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)

            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1'] = f1
            print(f" - val_precision: {precision:.4f} - val_recall: {recall:.4f} - val_f1: {f1:.4f}", end="")

            # AUC Calculation
            if self.is_binary:
                roc_auc = roc_auc_score(y_true, y_pred_prob_for_auc)
                logs['val_roc_auc'] = roc_auc
                print(f" - val_roc_auc: {roc_auc:.4f}", end="")
            elif y_pred_prob.shape[1] > 1 : # Multi-class AUC
                 try:
                     # Use one-hot true labels for multi-class AUC calculation
                     roc_auc = roc_auc_score(self.y_val_onehot, y_pred_prob_for_auc, multi_class='ovr', average='weighted')
                     logs['val_roc_auc'] = roc_auc
                     print(f" - val_roc_auc: {roc_auc:.4f}", end="")
                 except ValueError as e_auc:
                      print(f" - Could not calculate ROC AUC: {e_auc}", end="")
                      logs['val_roc_auc'] = 0.0
            else:
                 logs['val_roc_auc'] = 0.0 # Should not happen if logic above is correct


        except Exception as e:
            print(f"\nError calculating metrics: {e}")
            logs['val_precision'] = 0.0
            logs['val_recall'] = 0.0
            logs['val_f1'] = 0.0
            logs['val_roc_auc'] = 0.0

        print()
        # Keras uses the returned logs dictionary to update history and print
        return logs # Although modifying logs in-place works, returning it is cleaner
