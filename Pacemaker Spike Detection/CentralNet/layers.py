import tensorflow as tf
from tensorflow.keras.layers import Layer
import math

# Implementation of Fusion knot of CentralNet
class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer(minval=0.01, maxval=1)
        w_reg = tf.keras.regularizers.L1()
        self.w = self.add_weight(shape=(len(input_shape),1), dtype="float32", 
                                initializer=w_init, trainable=True,
                                regularizer=w_reg)

    # Perform weighted Summation
    def call(self, inputs):
        wsum = None
        i = 0
        for _input in inputs:
            if wsum is None:
                wsum = self.w[i] * _input
            else:
                wsum += self.w[i] * _input
            i += 1
        return wsum

# Apply layer with the same structure to different input
def ApplyUniformLayer(layer, inputs, *args, **kwargs):
    outputs = list()
    for _input in inputs:
        outputs.append(layer(*args, **kwargs)(_input))
    return outputs
