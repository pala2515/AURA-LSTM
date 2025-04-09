class EnhancedGCNModel(keras.Model):
    # Default dropout_rate updated to match table spec (0.2)
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(EnhancedGCNModel, self).__init__(name="EnhancedGCNModel_2Block")

        # First GCN block (Hidden Dim 1: 64)
        self.gcn1_block1 = GCNConv(hidden_dim1, activation='relu')
        self.gcn2_block1 = GCNConv(hidden_dim1, activation='relu')
        self.gcn3_block1 = GCNConv(hidden_dim1, activation='relu')
        self.dropout1 = Dropout(dropout_rate) # Use the passed dropout_rate (0.2)

        # Second GCN block (Hidden Dim 2: 32)
        self.gcn1_block2 = GCNConv(hidden_dim2, activation='relu')
        self.gcn2_block2 = GCNConv(hidden_dim2, activation='relu')
        self.gcn3_block2 = GCNConv(hidden_dim2, activation='relu')
        self.dropout2 = Dropout(dropout_rate) # Use the passed dropout_rate (0.2)

        # Removed third block

        self.flatten = GlobalSumPool()
        self.concat = Concatenate()

    def call(self, inputs, training=None):
        x, a = inputs

        # First GCN block
        x1 = self.gcn1_block1([x, a])
        x1 = self.gcn2_block1([x1, a])
        x1 = self.gcn3_block1([x1, a])
        x1 = self.dropout1(x1, training=training)
        x1_pool = self.flatten(x1)

        # Second GCN block
        x2 = self.gcn1_block2([x, a]) # Use original x as input
        x2 = self.gcn2_block2([x2, a])
        x2 = self.gcn3_block2([x2, a])
        x2 = self.dropout2(x2, training=training)
        x2_pool = self.flatten(x2)

        # Removed third block processing

        # Concatenate outputs from the two blocks
        out = self.concat([x1_pool, x2_pool])
        return out
