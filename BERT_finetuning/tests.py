from BertMulticlassClassifier import CustomBertForSequenceClassification
from utils import update_bert_config

RUN_SPECIAL_LAYER_TEST = False
RUN_BERT_TEST = True

if (RUN_BERT_TEST):
  import tensorflow as tf
  from transformers import BertTokenizer, TFBertModel, BertConfig

  # Tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
  input_ids = tf.constant(tokenizer.encode("men trekanten havde ikke noget tøj på"))[None, :]  # Batch size 1
  print(input_ids)

  # Set config
  config = BertConfig.from_pretrained('bert-base-multilingual-cased')
  updates = {"scale_logits": True,
             "num_labels": 3,
             "scf_min": 0.3,
             "scf_max": 2.0,
            #  "apply_dropconnect": True,
             "dropconnect_prob": 0.8,
             "noise_distribution": "normal",
             "noise_amount": 0.025}
  config = update_bert_config(
    config,
    updates
  )

  # Instantiate model
  model = CustomBertForSequenceClassification(config)

  # Apply model
  outputs = model(input_ids)
  last_hidden_states = outputs[0]
  print(last_hidden_states)

if (RUN_SPECIAL_LAYER_TEST):
  import tensorflow as tf
  from ScaledLinear import ScaledLinear
  from ScalingFactor import ScalingFactor

  x = tf.ones((2, 2))

  print("Normal Dense Layer:")
  normal_dense = tf.keras.layers.Dense(4)
  z = normal_dense(x)
  print(z)
  print()

  print("Scaled Dense Layer:")
  scaled_linear_layer = ScaledLinear(4)
  y = scaled_linear_layer(x)
  print(y)
  print(scaled_linear_layer.weights)
  print(scaled_linear_layer.variables)
  config = scaled_linear_layer.get_config()
  print(config)
  print("From config:")
  new_scaled_linear_layer = ScaledLinear.from_config(config)
  print(new_scaled_linear_layer.weights)

  print()

  # NOTE: Current implementation of clip_by_value hides this variable
  # in eager execution - should be updated soon!
  print("Scaling Factor")
  scaler = ScalingFactor()
  print(scaler.weights)
  config = scaler.get_config()
  print(config)
  print("From config:")
  new_scaler = ScalingFactor.from_config(config)
  print(new_scaler.weights)
