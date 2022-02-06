import tensorflow as tf
from tensorflow.keras.layers import Layer
import math
class GaussLayer(Layer):
    def __init__(self, order=6, **kwargs):
        super(GaussLayer, self).__init__(**kwargs)
        self.order = order
    
    def build(self, input_shape):
        self.t_ax = tf.expand_dims(tf.linspace(0.0, 2*math.pi, 500), axis = 0)
        self.t_ax = tf.tile(self.t_ax, [self.order,1])
    
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super().get_config().copy()
        config.update({
            'order': self.order,
        })
        return config
        
    def call(self, inputs):
        def _gauss(paras):
            a = tf.expand_dims(paras[0:6], axis = 1)
            theta = tf.expand_dims(paras[6:12], axis = 1)
            c = tf.expand_dims(paras[12:18], axis = 1)
            gauss_fcn = a * tf.exp(-tf.square( (self.t_ax-theta) / (c + 1e-6) )*0.5 )
            return tf.reduce_sum(gauss_fcn, axis = 0)
        return tf.map_fn(_gauss, inputs)