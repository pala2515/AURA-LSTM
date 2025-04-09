def main(args):
    setup_gpu()

    # --- Datasets Configuration ---
    # Define your datasets here. Add more dictionaries to the list as needed.
    # Ensure the CSV files are in the correct 'path' relative to where you run train.py
    # and specify the correct 'smiles_col' and 'label_col'.
    datasets_info = [
        {'path': args.clintox_path, 'smiles_col': 'smiles', 'label_col': 'CT_TOX'},
        # Add other datasets like this:
        # {'path': 'data/tox21_nr_ar.csv', 'smiles_col': 'smiles', 'label_col': 'NR-AR'},
        # {'path': 'data/sider_hepatobiliary.csv', 'smiles_col': 'smiles', 'label_col': 'Hepatobiliary disorders'},
    ]

    # --- Parameters ---
    # Data Params
    N_AUGMENTS = args.n_augments
    TEST_SIZE = args.test_size
    RANDOM_STATE = args.random_state

    # GNN Params
    GNN_HIDDEN_DIM1 = args.gnn_dim1
    GNN_HIDDEN_DIM2 = args.gnn_dim2
    GNN_HIDDEN_DIM3 = args.gnn_dim3
    GAT_HEADS = args.gat_heads
    DROPOUT_RATE = args.dropout_rate

    # LSTM Params
    LSTM_UNITS_1 = args.lstm_units1
    LSTM_UNITS_2 = args.lstm_units2
    LSTM_DROPOUT = args.lstm_dropout

    # Training Params
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # --- 1. Load and Preprocess Data ---
    print("="*20 + " 1. Loading and Preprocessing Data " + "="*20)
    train_smiles, test_smiles, train_labels, test_labels = load_and_preprocess_data(
        datasets_info=datasets_info,
        n_augments=N_AUGMENTS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --- 2. Prepare Graph Data ---
    print("\n" + "="*20 + " 2. Preparing Graph Data " + "="*20)
    train_data_raw = []
    filtered_train_labels = []
    print("Processing training molecules...")
    for i, (smiles, label) in enumerate(zip(train_smiles, train_labels)):
        features, adj_matrix = molecule_to_graph(smiles)
        if features is not None and adj_matrix is not None and features.shape[0] > 0:
            train_data_raw.append((features, adj_matrix))
            filtered_train_labels.append(label)
        # else:
            # print(f"Skipping invalid graph for training SMILES idx {i}") # Optional debug print
        if (i+1) % 1000 == 0: print(f"  Processed {i+1}/{len(train_smiles)} training molecules")

    test_data_raw = []
    filtered_test_labels = []
    print("Processing test molecules...")
    for i, (smiles, label) in enumerate(zip(test_smiles, test_labels)):
        features, adj_matrix = molecule_to_graph(smiles)
        if features is not None and adj_matrix is not None and features.shape[0] > 0:
            test_data_raw.append((features, adj_matrix))
            filtered_test_labels.append(label)
        # else:
            # print(f"Skipping invalid graph for testing SMILES idx {i}") # Optional debug print
        if (i+1) % 1000 == 0: print(f"  Processed {i+1}/{len(test_smiles)} test molecules")


    if not train_data_raw or not test_data_raw:
        raise ValueError("No valid graph data could be generated for training or testing after filtering.")

    max_atoms = max([features.shape[0] for features, _ in train_data_raw + test_data_raw])
    input_feature_dim = train_data_raw[0][0].shape[1]
    print(f"Max atoms for padding: {max_atoms}")
    print(f"Input feature dimension: {input_feature_dim}")

    print("Padding sequences...")
    train_node_features = pad_sequences([features for features, _ in train_data_raw], maxlen=max_atoms, padding="post", dtype='float32', value=0.0)
    train_adj_matrices = np.array([np.pad(adj, ((0, max_atoms - adj.shape[0]), (0, max_atoms - adj.shape[1])), 'constant', constant_values=0.0) for _, adj in train_data_raw], dtype=np.float32)

    test_node_features = pad_sequences([features for features, _ in test_data_raw], maxlen=max_atoms, padding="post", dtype='float32', value=0.0)
    test_adj_matrices = np.array([np.pad(adj, ((0, max_atoms - adj.shape[0]), (0, max_atoms - adj.shape[1])), 'constant', constant_values=0.0) for _, adj in test_data_raw], dtype=np.float32)

    filtered_train_labels = np.array(filtered_train_labels)
    filtered_test_labels = np.array(filtered_test_labels)

    # --- 3. Extract Features using GNN Models ---
    print("\n" + "="*20 + " 3. Extracting Features with GNNs " + "="*20)

    gat_model = GATModel(
        input_dim=input_feature_dim,
        hidden_dim1=GNN_HIDDEN_DIM1, hidden_dim2=GNN_HIDDEN_DIM2, hidden_dim3=GNN_HIDDEN_DIM3,
        dropout_rate=DROPOUT_RATE, num_heads=GAT_HEADS
    )
    gcn_model = EnhancedGCNModel(
        input_dim=input_feature_dim,
        hidden_dim1=GNN_HIDDEN_DIM1, hidden_dim2=GNN_HIDDEN_DIM2, hidden_dim3=GNN_HIDDEN_DIM3,
        dropout_rate=DROPOUT_RATE
    )
    gin_model = EnhancedGINModel(
        input_dim=input_feature_dim,
        hidden_dim1=GNN_HIDDEN_DIM1, hidden_dim2=GNN_HIDDEN_DIM2, hidden_dim3=GNN_HIDDEN_DIM3,
        dropout_rate=DROPOUT_RATE
    )

    # Build models with dummy input
    dummy_nodes = np.random.rand(1, max_atoms, input_feature_dim).astype(np.float32)
    dummy_adj = np.random.rand(1, max_atoms, max_atoms).astype(np.float32)
    print("Building GNN models...")
    _ = gat_model([dummy_nodes, dummy_adj], training=False)
    _ = gcn_model([dummy_nodes, dummy_adj], training=False)
    _ = gin_model([dummy_nodes, dummy_adj], training=False)
    print("GNN models built.")


    print("Predicting with GAT...")
    train_gat_features = gat_model.predict([train_node_features, train_adj_matrices], batch_size=BATCH_SIZE)
    test_gat_features = gat_model.predict([test_node_features, test_adj_matrices], batch_size=BATCH_SIZE)

    print("Predicting with GCN...")
    train_gcn_features = gcn_model.predict([train_node_features, train_adj_matrices], batch_size=BATCH_SIZE)
    test_gcn_features = gcn_model.predict([test_node_features, test_adj_matrices], batch_size=BATCH_SIZE)

    print("Predicting with GIN...")
    train_gin_features = gin_model.predict([train_node_features, train_adj_matrices], batch_size=BATCH_SIZE)
    test_gin_features = gin_model.predict([test_node_features, test_adj_matrices], batch_size=BATCH_SIZE)

    combined_train_features = np.concatenate((train_gat_features, train_gcn_features, train_gin_features), axis=1)
    combined_test_features = np.concatenate((test_gat_features, test_gcn_features, test_gin_features), axis=1)

    print(f"Combined GNN features shape (Train): {combined_train_features.shape}")

    print("Removing zero columns...")
    combined_data = np.concatenate((combined_train_features, combined_test_features), axis=0)
    combined_data_cleaned = remove_zero_columns(combined_data)
    train_features_gnn = combined_data_cleaned[:combined_train_features.shape[0], :]
    test_features_gnn = combined_data_cleaned[combined_train_features.shape[0]:, :]
    print(f"GNN features shape after cleaning (Train): {train_features_gnn.shape}")

    # --- 4. Feature Engineering (Scaling only) ---
    print("\n" + "="*20 + " 4. Feature Engineering " + "="*20)

    print("Scaling features (StandardScaler)...")
    train_features_scaled, test_features_scaled, _ = scale_features(train_features_gnn, test_features_gnn)

    print("One-hot encoding labels...")
    train_labels_onehot, test_labels_onehot, label_encoder = one_hot_encode_labels(filtered_train_labels, filtered_test_labels)
    num_classes = train_labels_onehot.shape[1]
    print(f"Number of classes detected: {num_classes}")
    print(f"Label encoder categories: {label_encoder.categories_}")

    # Final features for LSTM are the scaled GNN features
    train_features_final = train_features_scaled
    test_features_final = test_features_scaled

    # --- 5. LSTM Model and Training ---
    print("\n" + "="*20 + " 5. LSTM Model Training " + "="*20)

    timesteps = 1
    train_features_lstm = train_features_final.reshape(train_features_final.shape[0], timesteps, train_features_final.shape[1])
    test_features_lstm = test_features_final.reshape(test_features_final.shape[0], timesteps, test_features_final.shape[1])
    print(f"LSTM input shape: (batch_size, {timesteps}, {train_features_final.shape[1]})")

    # Define LSTM Model
    model = keras.Sequential(name="toxicity_lstm_classifier")
    model.add(layers.Input(shape=(timesteps, train_features_lstm.shape[2])))
    model.add(layers.Bidirectional(layers.LSTM(LSTM_UNITS_1, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001))))
    model.add(layers.Bidirectional(layers.LSTM(LSTM_UNITS_2))) # Last LSTM layer should not return sequences if followed by Dense
    model.add(layers.Dropout(LSTM_DROPOUT))
    # Use sigmoid for binary (num_classes=2 treated as binary) or softmax for multi-class
    output_activation = 'sigmoid' if num_classes <= 2 else 'softmax'
    # If output_activation is sigmoid, ensure num_classes is 1 if you use BinaryCrossentropy
    output_units = 1 if output_activation == 'sigmoid' and num_classes == 2 else num_classes # Adjust units based on activation/loss
    model.add(layers.Dense(output_units, activation=output_activation))


    # Compile Model
    loss_function = keras.losses.BinaryCrossentropy() if output_units == 1 else keras.losses.CategoricalCrossentropy()
    # Adjust labels if using sigmoid output with BinaryCrossentropy
    train_labels_for_loss = train_labels_onehot[:, 1].reshape(-1, 1) if output_units == 1 else train_labels_onehot
    test_labels_for_loss = test_labels_onehot[:, 1].reshape(-1, 1) if output_units == 1 else test_labels_onehot


    model.compile(
        loss=loss_function,
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[keras.metrics.AUC(name="AUC")] # Add other metrics if needed e.g., 'accuracy'
    )

    model.summary()

    # Setup Callback
    # Pass the correct label format to the callback for metric calculation
    metrics_callback = MetricsCallback(validation_data=(test_features_lstm, test_labels_onehot), use_binary_metrics=(output_units==1))


    # Train Model
    print("\nStarting training...")
    history = model.fit(
        train_features_lstm,
        train_labels_for_loss, # Use labels adjusted for loss function
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_data=(test_features_lstm, test_labels_for_loss), # Use labels adjusted for loss function
        callbacks=[metrics_callback]
    )

    # --- 6. Evaluation ---
    print("\n" + "="*20 + " 6. Final Evaluation " + "="*20)
    loss, auc = model.evaluate(test_features_lstm, test_labels_for_loss, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Retrieve final metrics from history (more reliable than calling callback again)
    print("\nFinal validation metrics from training history (last epoch):")
    for metric in ['val_loss', 'val_AUC', 'val_precision', 'val_recall', 'val_f1', 'val_roc_auc']:
         if metric in history.history:
              print(f"  {metric}: {history.history[metric][-1]:.4f}")
         # Note: val_roc_auc might be named differently depending on the exact callback implementation used during fit

    print("\nTraining finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN+LSTM model for toxicity prediction.")

    # --- Command Line Arguments ---
    # Data Args
    parser.add_argument('--clintox_path', type=str, default='data/clintox.csv', help='Path to the clintox dataset CSV file.')
    # Add arguments for other dataset paths if you want them configurable
    # parser.add_argument('--tox21_path', type=str, default='data/tox21.csv', help='Path to the Tox21 dataset.')
    parser.add_argument('--n_augments', type=int, default=5, help='Number of augmentations per SMILES.')
    parser.add_argument('--test_size', type=float, default=0.3, help='Fraction of data for the test set.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for splits.') # Changed default

    # GNN Args
    parser.add_argument('--gnn_dim1', type=int, default=64, help='Hidden dimension 1 for GNNs.')
    parser.add_argument('--gnn_dim2', type=int, default=32, help='Hidden dimension 2 for GNNs.')
    parser.add_argument('--gnn_dim3', type=int, default=16, help='Hidden dimension 3 for GNNs.')
    parser.add_argument('--gat_heads', type=int, default=8, help='Number of attention heads for GAT.') # Reduced default
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for GNNs.')

    # LSTM Args
    parser.add_argument('--lstm_units1', type=int, default=128, help='Units in the first LSTM layer.')
    parser.add_argument('--lstm_units2', type=int, default=64, help='Units in the second LSTM layer.')
    parser.add_argument('--lstm_dropout', type=float, default=0.3, help='Dropout rate after LSTM layers.')

    # Training Args
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.') # Reduced default
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')

    args = parser.parse_args()
    main(args)
