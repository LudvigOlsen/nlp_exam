import tensorflow as tf
import tensorflow_datasets
import os
import time
import argparse

from transformers import *

from BertMulticlassClassifier import CustomBertForSequenceClassification
from utils import update_bert_config
from convert_examples_to_features import convert_examples_to_features

"""
Set the following
"""

COMPUTER = "docker"

PRETRAINED_NAME = 'bert-base-multilingual-cased'
if COMPUTER == "x":
    PROJECT_PATH = ""
elif COMPUTER == "docker":
    PROJECT_PATH = ""

parser = argparse.ArgumentParser(description='Finetune BERT model')
parser.add_argument('--fold', type=int, nargs=1,
                    help='fold indice')

args = parser.parse_args()
fold = args.fold[0]
print("Fold: ",args.fold[0])


# LOAD_CKPT_SAVE_MODEL = True
# CKPT_TO_LOAD = "upsampled_model_1575978661.1679058/ckpt_2"

MODEL_NAME = "upsampled_model_fold_" + str(fold) + "_" + str(time.time())
print("model: ", MODEL_NAME)
DATA_PATH = PROJECT_PATH + 'data/preprocessed/'
IS_UPSAMPLED = True
prefix = "upsampled_" if IS_UPSAMPLED else ""
TRAIN_PATH = DATA_PATH + prefix + 'train_fold_{}.tfrecord'.format(fold)
TEST_PATH = DATA_PATH + prefix + 'test_fold_{}.tfrecord'.format(fold)
CHECKPOINT_DIR_PATH = './training_checkpoints/' + MODEL_NAME
MODEL_SAVE_PATH = 'saved_model/' + MODEL_NAME + "/"
USE_GLUE_DATA = False
LABELS = [0,1,2]  # "Control":0, "Depression":1, "Schizophrenia":2
CLASS_WEIGHTS = {0:0.33, 1:0.33, 2:0.33} if IS_UPSAMPLED else {0:0.13, 1:0.55, 2:0.32} # approximately inverse frequency weighting
MAX_SENTENCE_LENGTH = 212 # Max length in dataset for splits
BATCH_SIZE = 15
SHUFFLE_SIZE = 70
EPOCHS = 4
STEPS_PER_EPOCH = None
min_fold = 1477 if IS_UPSAMPLED else 797
VALIDATION_STEPS = (min_fold-BATCH_SIZE)//BATCH_SIZE # Smallest fold is 797 examples
LEARNING_RATE = 3e-5 # -5 
USE_MULTIPROCESSING = True

# Set configs
model_config = BertConfig.from_pretrained(PRETRAINED_NAME)
updates = {"scale_logits": False,
            "num_labels": 3,
            "scf_min": 0.3,
            "scf_max": 2.0,
            "dropconnect_prob": 0.0,# 0.05,
            "noise_distribution": "normal",
            "noise_amount": 0.0, #075,
            "add_dense": True,
            "add_dense_2": False,
            "dense_units": 512,
            "scale_dense": True,
            "dense_dropout_prob": 0.25 }
model_config = update_bert_config(model_config, updates)
print(model_config)

"""

"""

# Name of the checkpoint files
checkpoint_prefix = os.path.join(CHECKPOINT_DIR_PATH, "ckpt_{epoch}")

# Callbacks for tensorboard and checkpoints
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)
]


# Set distribution strategy
strategy = tf.distribute.MirroredStrategy()

# Load tokenizer from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_NAME)

# Load datasets

if USE_GLUE_DATA:
    # GLUE DATA:
    data = tensorflow_datasets.load('glue/mrpc')
    # tf dataset has idx:tf.int32, label:tf.int64 , sentence1:tf.string , sentence2:tf.string

    # Prepare dataset for GLUE as a tf.data.Dataset instance
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    valid_dataset = valid_dataset.batch(64)

else:
    # Create a description of the features.
    feature_description = {
        'idx': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'sentence': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    train_dataset = tf.data.TFRecordDataset([TRAIN_PATH]).map(_parse_function)
    valid_dataset = tf.data.TFRecordDataset([TEST_PATH]).map(_parse_function)
    train_dataset = convert_examples_to_features(
        examples=train_dataset, 
        tokenizer=tokenizer, 
        max_length=MAX_SENTENCE_LENGTH, 
        label_list=LABELS,
        task='multiclassClassification')
    valid_dataset = convert_examples_to_features(
        examples=valid_dataset, 
        tokenizer=tokenizer,
        max_length=MAX_SENTENCE_LENGTH, 
        label_list=LABELS,
        task='multiclassClassification')
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE) #.repeat(2)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)


    # if strategy is not None:
    #     train_dataset = strategy.experimental_distribute_dataset(train_dataset)

with strategy.scope():
    
    # Load model
    model = CustomBertForSequenceClassification(model_config)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset, validation_steps=VALIDATION_STEPS, 
                    callbacks=callbacks, class_weight=CLASS_WEIGHTS, use_multiprocessing=USE_MULTIPROCESSING)

# model.save(MODEL_SAVE_PATH, save_format='tf')
