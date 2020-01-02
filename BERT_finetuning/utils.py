import tensorflow as tf
from transformers import BertConfig
import tensorflow_probability as tfp


def update_bert_config(config, d):
  config = config.to_dict()
  config.update(d)
  return BertConfig.from_dict(config)


def dropconnect(w, p):
  return tf.nn.dropout(w, rate=p) * p


def add_noise(w, distribution, amount, relative=True):
  pass


def iqr(x):
  q1 = tfp.stats.percentile(x, 25., interpolation='linear')
  q3 = tfp.stats.percentile(x, 75., interpolation='linear')
  return q3 - q1


def median(x):
  return tfp.stats.percentile(x, 50.)


def blend(x1, x2, amount):
  x1 = tf.dtypes.cast(x1, tf.float32)
  x2 = tf.dtypes.cast(x2, tf.float32)
  x1_weighted = tf.multiply(x1, (1 - amount))
  x2_weighted = tf.multiply(x2, amount)
  return x1_weighted + x2_weighted
