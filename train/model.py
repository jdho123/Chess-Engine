# pylint: disable=no-member
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class PerspectiveMerge(tf.keras.layers.Layer):
    def call(self, inputs):
        w_acc, b_acc, turn = inputs
        turn = tf.cast(tf.reshape(turn, (-1, 1)), tf.bool)
        first = tf.where(turn, w_acc, b_acc)
        second = tf.where(turn, b_acc, w_acc)

        return tf.concat((first, second), axis=1)


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class BucketedDense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        num_buckets: int = 16,
        min_material: int = 0,
        max_material: int = 78,
        activation: str = None,
        **kwargs
    ):
        super(BucketedDense, self).__init__(**kwargs)
        self.units = units
        self.num_buckets = num_buckets
        self.min_material = min_material
        self.max_material = max_material

        self.kernel = None
        self.bias = None

        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_shape = input_shape[0]
        input_dim = feature_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.num_buckets, input_dim, self.units),
            initializer="glorot_uniform",
            name="kernel",
            trainable=True,
        )

        self.bias = self.add_weight(
            shape=(self.num_buckets, self.units),
            initializer="zeros",
            name="bias",
            trainable=True,
        )

        super(BucketedDense, self).build(input_shape)

    def call(self, inputs):
        features, material_count = inputs

        num_normal_buckets = self.num_buckets - 1

        mat_float = tf.cast(material_count, tf.float32)
        normalized = (mat_float - self.min_material) / (
            self.max_material - self.min_material
        )
        scaled = normalized * float(num_normal_buckets)

        standard_idx = tf.cast(tf.floor(scaled), tf.int32)
        standard_idx = tf.clip_by_value(standard_idx, 0, num_normal_buckets - 1)
        bucket_idx = tf.where(
            mat_float > self.max_material, num_normal_buckets, standard_idx
        )
        bucket_idx = tf.reshape(bucket_idx, (-1,))

        W_batch = tf.gather(self.kernal, bucket_idx)
        b_batch = tf.gather(self.bias, bucket_idx)

        output = tf.einsum("bi,bij->bj", features, W_batch) + b_batch

        if self.activation is not None:
            output = self.activation(output)

        return output


def build_model(
    input_dim: int, accumulator_dim: int, hidden_dim: int, bucket_params: dict
) -> tf.keras.Model:
    white = tf.keras.Input(shape=(input_dim,), name="white")
    black = tf.keras.Input(shape=(input_dim,), name="black")
    material_value = tf.keras.Input(shape=(1,), name="material_value")
    turn = tf.keras.Input(shape=(1,), name="turn")

    shared_weights = tf.keras.layers.Dense(accumulator_dim, name="shared_weights")

    acc_white = shared_weights(white)
    acc_black = shared_weights(black)

    merged = PerspectiveMerge()([acc_white, acc_black, turn])
    crelu1 = tf.keras.layers.ReLU(max_value=1.0)(merged)

    hidden = tf.keras.layers.Dense(hidden_dim)(crelu1)
    crelu2 = tf.keras.layers.ReLU(max_value=1.0)(hidden)

    bucketed = BucketedDense(hidden_dim, **bucket_params)(crelu2)
    crelu3 = tf.keras.layers.ReLU(max_value=1.0)(bucketed)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(crelu3)

    model = tf.keras.Model(inputs=(white, black, material_value, turn), outputs=output)

    return model
