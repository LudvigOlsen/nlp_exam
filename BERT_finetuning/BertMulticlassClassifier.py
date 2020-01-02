# coding=utf-8
# Copyright 2020, Ludvig Renbo Olsen
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from transformers import TFBertPreTrainedModel, TFBertMainLayer, TFBertForSequenceClassification

from Noise import Noise
from ScaledLinear import ScaledLinear
from DensePipe import DensePipe


class CustomBertForSequenceClassification(TFBertPreTrainedModel):
  r"""
  This is a customization of TFBertForSequenceClassification by Hugging Face.

  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
      **logits**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, config.num_labels)``
          Classification (or regression if config.num_labels==1) scores (before SoftMax).
      **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
          list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
          of shape ``(batch_size, sequence_length, hidden_size)``:
          Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      **attentions**: (`optional`, returned when ``config.output_attentions=True``)
          list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

  Examples::

      import tensorflow as tf
      from transformers import BertTokenizer, CustomBertForSequenceClassification

      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased')
      input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
      outputs = model(input_ids)
      logits = outputs[0]

  """

  def __init__(self, config, *inputs, **kwargs):
    super(CustomBertForSequenceClassification, self).__init__(config, *inputs, **kwargs)
    self.num_labels = config.num_labels
    self.noise_amount = config.noise_amount
    self.noise_distribution = config.noise_distribution
    self.add_dense = config.add_dense
    self.add_dense_2 = config.add_dense_2

    self.bert = TFBertMainLayer(config, name='bert')
    self.dropout = tf.keras.layers.Dropout(config.dense_dropout_prob)

    self.noise_layer = Noise(amount=self.noise_amount, 
                             activation="relu",
                             distribution=self.noise_distribution, 
                             relative=True)

    if self.add_dense:
      
      self.dense_layer_1 = ScaledLinear(
        config.dense_units,
        activation="relu",
        scf_min=config.scf_min,
        scf_max=config.scf_max,
        scale=config.scale_dense,
        dropconnect_prob=config.dropconnect_prob,
        kernel_initializer=get_initializer(config.initializer_range),
        name='dense')

    if self.add_dense_2:

      self.dense_pipe = DensePipe( 
        units=20,
        num_layers=10,
        extract_every_n_layers=3, # Saves every n layer and return them concatenated
        activation="relu",
        use_bias=True,
        scale=True,
        scf_min=0.2,
        scf_max=2.0,
        dropconnect_prob=0.00,
        dropout_prob=0.1,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros')

    self.classifier = ScaledLinear(config.num_labels,
                                   scf_min=config.scf_min,
                                   scf_max=config.scf_max,
                                   scale=config.scale_logits,
                                   dropconnect_prob=0.0, # config.dropconnect_prob,
                                   kernel_initializer=get_initializer(config.initializer_range),
                                   name='classifier')

                                   
  def call(self, inputs, **kwargs):
    outputs = self.bert(inputs, **kwargs)
    is_training = kwargs.get('training', False)
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output, training=is_training)
    if self.noise_amount > 0.0:
      noised_output = self.noise_layer(pooled_output, training=is_training)
    else: 
      noised_output = pooled_output
    if self.add_dense:
      densed_output = self.dense_layer_1(noised_output, training=is_training)
      densed_output = self.dropout(densed_output, training=is_training)
      if self.add_dense_2:
        dense_piped_output = self.dense_pipe(densed_output, training=is_training)
        densed_output = tf.keras.layers.concatenate(
          [densed_output, dense_piped_output],
          axis=-1)
    else:
      densed_output = noised_output
    logits = self.classifier(densed_output, training=is_training)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

    return outputs  # logits, (hidden_states), (attentions)


# https://github.com/huggingface/transformers/blob/master/transformers/modeling_tf_utils.py
def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.
  Args:
    initializer_range: float, initializer range for stddev.
  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
