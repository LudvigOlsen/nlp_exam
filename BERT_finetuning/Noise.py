import tensorflow as tf

from utils import iqr, median, blend
from tensorflow.keras.backend import in_train_phase
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations

class Noise(tf.keras.layers.Layer):
  """
  check also:
    https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/noise.py
  """

  def __init__(self,
               amount=0.025,
               distribution="normal",
               relative=True,
               activation=None,
               **kwargs):
    super(Noise, self).__init__(**kwargs)

    self.distribution = distribution
    self.relative = relative
    self.amount = amount
    self.activation = activations.get(activation)

    if distribution == "normal":
      self.generator = self.normal_generator
    elif distribution == "robust_normal":
      self.generator = self.robust_normal_generator
    elif distribution == "uniform":
      self.generator = self.uniform_generator
    else:
      raise KeyError("Could not recognize 'distribution'.")

  # def build(self, input_shape):
    
  def call(self, inputs, **kwargs):
    """
    kwargs used are 'training' and 'amount'
    """

    is_training = kwargs.get('training', False)
    amount = kwargs.get('amount', None)
    if amount is None:
      amount = self.amount

    def noised():
      noise = self.generator(inputs, self.relative)
      if self.relative:
        noise = tf.multiply(inputs, noise)
        return inputs + tf.multiply(noise, amount)
      else:
        # Blend the two
        return blend(inputs, noise, amount)

    outputs = in_train_phase(noised, inputs, training=is_training)

    # Apply activation function
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return outputs

  def get_config(self):
    # From parent
    config = super(Noise, self).get_config()

    # From current
    config.update({
      "distribution": self.distribution,
      "relative": self.relative,
      "amount": self.amount,
      'activation': activations.serialize(self.activation)
    })

    return config

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  def normal_generator(self, x, relative):

    if relative:
      return tf.random.truncated_normal(
        array_ops.shape(x),
        mean=1,
        stddev=0.5,
        dtype=tf.dtypes.float32,
        seed=None,
        name="normal_noise_generator"
      )
    else:
      return tf.random.truncated_normal(
        array_ops.shape(x),
        mean=tf.math.reduce_mean(x),
        stddev=tf.math.reduce_std(x),
        dtype=tf.dtypes.float32,
        seed=None,
        name="normal_noise_generator"
      )

  def uniform_generator(self, x, relative):
    if relative:
      return tf.random.uniform(
        array_ops.shape(x),
        minval=0,
        maxval=2,
        dtype=tf.dtypes.float32,
        seed=None,
        name="uniform_noise_generator"
      )
    else:
      return tf.random.uniform(
        array_ops.shape(x),
        minval=tf.math.reduce_min(x),
        maxval=tf.math.reduce_max(x),
        dtype=tf.dtypes.float32,
        seed=None,
        name="uniform_noise_generator"
      )

  def robust_normal_generator(self, x, relative): 
    if relative:
      raise KeyError("robust_gaussian is not meaningful when 'relative' is True.")
    else:
      return tf.random.truncated_normal( # TODO Should be truncated by min and max, not 2x std
        array_ops.shape(x),
        mean=median(x),
        stddev=iqr(x),
        dtype=tf.dtypes.float32,
        seed=None,
        name="robust_normal_noise_generator"
      )
