import tensorflow as tf


class ScalingFactor(tf.keras.layers.Layer):

  def __init__(self, scf_min=0.2, scf_max=2.0, name=None, trainable=True, **kwargs):
    super(ScalingFactor, self).__init__(name=name, trainable=trainable, **kwargs)

    self.__scf_min = scf_min
    self.__scf_max = scf_max
    self.__trainable = trainable
    self.__name = name

    # Initialize scaling factor
    self.s = self.add_weight(
      name=self.__name,
      shape=(1,),
      dtype="float32",
      initializer=tf.ones_initializer(),
      trainable=self.__trainable)

    # Truncate to ensure a useful value
    self.s = tf.clip_by_value(self.s,
                              clip_value_min=self.__scf_min,
                              clip_value_max=self.__scf_max,
                              name=None if self.__name is None else self.__name + "_truncated")

  def call(self, inputs):
    return inputs * self.s

  def get_config(self):
    config = super(ScalingFactor, self).get_config()

    config.update({
      'scf_min': self.__scf_min,
      'scf_max': self.__scf_max,
    })
    
    return config
