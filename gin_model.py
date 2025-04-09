# GINConv layer remains the same structurally
class GINConv(Layer):
    def __init__(self, units, epsilon=0.0, activation=None, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        self.units = units
        self.epsilon = epsilon
        self.activation = keras.activations.get(activation)
        self.mlp = None

    def build(self, input_shape):
         node_shape = input_shape[0]
         feature_dim = node_shape[-1]
         self.mlp = keras.Sequential([
            layers.Dense(self.units, activation=self.activation, input_shape=(feature_dim,)),
            layers.Dense(self.units)
         ], name=f"{self.name}_mlp")
         super(GINConv, self).build(input_shape)

    def call(self, inputs):
        x, a = inputs
        aggregated = tf.matmul(a, x)
        epsilon_tensor = tf.constant(self.epsilon, dtype=x.dtype)
        x_updated = (1.0 + epsilon_tensor) * x + aggregated
        x_transformed = self.mlp(x_updated)
        return x_transformed

    def get_config(self):
        config = super(GINConv, self).get_config()
        config.update({
            'units': self.units,
            'epsilon': self.epsilon,
            'activation': keras.activations.serialize(self.activation),
        })
        return config

class EnhancedGINModel(keras.Model):
    # Default dropout_rate updated to match table spec (0.3)
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.3):
        super(EnhancedGINModel, self).__init__(name="EnhancedGINModel_2Block")

        # First GIN block (Hidden Dim 1: 64)
        self.gin1_block1 = GINConv(hidden_dim1, activation='relu')
        self.gin2_block1 = GINConv(hidden_dim1, activation='relu')
        self.gin3_block1 = GINConv(hidden_dim1, activation='relu')
        self.dropout1 = Dropout(dropout_rate) # Use the passed dropout_rate (0.3)

        # Second GIN block (Hidden Dim 2: 32)
        self.gin1_block2 = GINConv(hidden_dim2, activation='relu')
        self.gin2_block2 = GINConv(hidden_dim2, activation='relu')
        self.gin3_block2 = GINConv(hidden_dim2, activation='relu')
        self.dropout2 = Dropout(dropout_rate) # Use the passed dropout_rate (0.3)

        # Removed third block

        self.flatten = GlobalSumPool()
        self.concat = Concatenate()

    def call(self, inputs, training=False):
        x, a = inputs

        # First GIN block
        x1 = self.gin1_block1([x, a])
        x1 = self.gin2_block1([x1, a])
        x1 = self.gin3_block1([x1, a])
        x1 = self.dropout1(x1, training=training)
        x1_pool = self.flatten(x1)

        # Second GIN block
        x2 = self.gin1_block2([x, a]) # Use original x
        x2 = self.gin2_block2([x2, a])
        x2 = self.gin3_block2([x2, a])
        x2 = self.dropout2(x2, training=training)
        x2_pool = self.flatten(x2)

        # Removed third block processing

        # Concatenate outputs from the two blocks
        out = self.concat([x1_pool, x2_pool])
        return out
