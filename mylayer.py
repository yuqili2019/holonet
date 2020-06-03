import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):

  def build(self, input_shape):

    self.bias = self.add_weight(shape=(1,1,1), initializer='he_normal', dtype=tf.float32, name='x')  #input_shape[1:4]

  def call(self, inputs):
    return tf.abs(inputs) + tf.tile(self.bias, (65,68,68))
