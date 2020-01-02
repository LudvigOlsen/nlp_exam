import tensorflow as tf
from ScalingFactor import ScalingFactor
from utils import dropconnect
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.ops import standard_ops, nn, gen_math_ops, math_ops, sparse_ops
from tensorflow.python.eager import context

# from tensorflow.python.keras._impl.keras import initializers

from utils import dropconnect


class ScaledLinear(tf.keras.layers.Layer):

  def __init__(self, units,
               activation=None,
               use_bias=True,
               scale=True,
               scf_min=0.2,
               scf_max=2.0,
               dropconnect_prob=0.05,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None, 
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(ScaledLinear, self).__init__(
      activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

    # Save params
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.dropconnect_prob = dropconnect_prob
    self.scale = scale
    self.scf_min = scf_min
    self.scf_max = scf_max
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.kwargs = kwargs
    
    # Initialize scaling factor
    self.scaler = ScalingFactor(scf_min=self.scf_min,
                               scf_max=self.scf_max,
                               name="scaling_factor")

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `ScaledLinear` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `ScaledLinear` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})

    self.kernel = self.add_weight(
      shape=(last_dim, self.units),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      dtype=self.dtype,
      trainable=True,
      name="kernel")

    if self.use_bias:
      self.bias = self.add_weight(
        shape=(self.units,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        name="bias")
    else:
      self.bias = None

    # This sets self.built = True
    super().build(input_shape)

    # self.before = tf.Variable(100, tf.int16)
    # self.after = tf.Variable(200, tf.int16)

  def call(self, inputs, **kwargs):

    is_training = kwargs.get('training', False)


    if self.dropconnect_prob > 0.0:
      def dropconnected():
        return dropconnect(self.kernel, self.dropconnect_prob)

      # Apply dropconnect if in training
      # Fails when overwriting kernel, hence the "DC"
      self.kernelDC = K.in_train_phase(dropconnected, 
                                      self.kernel, 
                                      training=is_training)
    else:
      self.kernelDC = self.kernel

      
    # Apply kernel to inputs
    # Note: This part came from Dense()
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.kernelDC, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernelDC)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.kernelDC)

    # Add bias
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)

    # Apply scaling factor
    if self.scale:
      outputs = self.scaler(outputs)
    
    # Apply activation function
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
        'The innermost dimension of input_shape must be defined, but saw: %s'
        % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    # From parent
    config = super(ScaledLinear, self).get_config()
    # From current
    config.update({
      'units': self.units,
      'use_bias': self.use_bias,
      'scale': self.scale,
      'scf_min': self.scf_min,
      'scf_max': self.scf_max,
      'dropconnect_prob': self.dropconnect_prob,
      'activation': activations.serialize(self.activation),
      'kernel_initializer': initializers.serialize(self.kernel_initializer),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
      'kernel_constraint': constraints.serialize(self.kernel_constraint),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config
