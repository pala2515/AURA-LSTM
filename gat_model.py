import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GATConv, GlobalSumPool
from tensorflow.keras.layers import Dropout, Concatenate

class GATModel(keras.Model):
    # Default dropout_rate and num_heads updated to match table specs
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.3, num_heads=8):
        super(GATModel, self).__init__(name="GATModel_2Block")

        # First GAT block (Hidden Dim 1: 64)
        self.gat1_block1 = GATConv(hidden_dim1, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.gat2_block1 = GATConv(hidden_dim1, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.gat3_block1 = GATConv(hidden_dim1, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.dropout1 = Dropout(dropout_rate) # Use the passed dropout_rate (0.3)

        # Second GAT block (Hidden Dim 2: 32)
        self.gat1_block2 = GATConv(hidden_dim2, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.gat2_block2 = GATConv(hidden_dim2, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.gat3_block2 = GATConv(hidden_dim2, activation='relu', attn_heads=num_heads, concat_heads=True)
        self.dropout2 = Dropout(dropout_rate) # Use the passed dropout_rate (0.3)

        # Removed third block

        self.flatten = GlobalSumPool()
        self.concat = Concatenate()

    def call(self, inputs, training=None):
        x, a = inputs

        # First GAT block
        x1 = self.gat1_block1([x, a])
        x1 = self.gat2_block1([x1, a])
        x1 = self.gat3_block1([x1, a])
        x1 = self.dropout1(x1, training=training)
        x1_pool = self.flatten(x1)

        # Second GAT block
        x2 = self.gat1_block2([x, a]) # Use original x as input
        x2 = self.gat2_block2([x2, a])
        x2 = self.gat3_block2([x2, a])
        x2 = self.dropout2(x2, training=training)
        x2_pool = self.flatten(x2)

        # Removed third block processing

        # Concatenate outputs from the two blocks
        out = self.concat([x1_pool, x2_pool])
        return out

    # build method remains optional
