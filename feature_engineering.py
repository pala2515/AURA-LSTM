import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def remove_zero_columns(data):
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    if data.shape[1] == 0:
        return data # Return empty array if no columns
    non_zero_cols = ~np.all(data == 0, axis=0)
    if not np.any(non_zero_cols):
         print("Warning: All columns are zero. Returning original data.")
         return data
    return data[:, non_zero_cols]

def scale_features(train_features, test_features):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    return train_scaled, test_scaled, scaler

def one_hot_encode_labels(train_labels, test_labels):
    train_labels_arr = np.array(train_labels).reshape(-1, 1)
    test_labels_arr = np.array(test_labels).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories='auto')

    # Fit on combined unique labels to handle cases where test might have labels not in train
    # This is less common after stratified split, but safer
    all_unique_labels = np.unique(np.concatenate((train_labels_arr, test_labels_arr))).reshape(-1, 1)
    enc.fit(all_unique_labels)

    train_labels_onehot = enc.transform(train_labels_arr)
    test_labels_onehot = enc.transform(test_labels_arr)

    return train_labels_onehot, test_labels_onehot, enc
