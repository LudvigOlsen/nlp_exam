import tensorflow as tf
from ScalingFactor import ScalingFactor
from utils import dropconnect
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.ops import standard_ops, nn, gen_math_ops, math_ops, sparse_ops
from tensorflow.python.eager import context
from ScaledLinear import ScaledLinear


class DensePipe(tf.keras.layers.Layer):
    
    def __init__(self, units,
                 num_layers,
                 extract_every_n_layers=3, # Saves every n layer and return them concatenated
                 activation=None,
                 use_bias=True,
                 scale=True,
                 scf_min=0.2,
                 scf_max=2.0,
                 dropconnect_prob=0.00,
                 dropout_prob=0.05,
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

        super(DensePipe, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        # Save params
        self.units = units
        self.num_layers = num_layers
        self.extract_every_n_layers = extract_every_n_layers
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.dropconnect_prob = dropconnect_prob
        self.dropout_prob = dropout_prob
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

        # Create Dense layers
        self.dense_layers = [ScaledLinear(
            units=self.units,
            activation=self.activation,
            scf_min=self.scf_min,
            scf_max=self.scf_max,
            scale=self.scale,
            dropconnect_prob=self.dropconnect_prob,
            kernel_initializer=self.kernel_initializer, 
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name='dense') for layer in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.dropout_prob)

    def call(self, inputs, **kwargs):
        
        is_training = kwargs.get('training', False)
        
        outputs = inputs
        every_n_computed_outputs = []
        
        for l in range(self.num_layers):
            outputs = self.dense_layers[l](outputs, training=is_training)
            outputs = self.dropout(outputs, training=is_training)
            if l % self.extract_every_n_layers == 0 and l != self.num_layers:
                every_n_computed_outputs.append(outputs)
            if l == self.num_layers:
                every_n_computed_outputs.append(outputs)

        outputs = tf.keras.layers.concatenate(
          every_n_computed_outputs,
          axis=-1)
        
        return outputs

        


