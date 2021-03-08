# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

# Modified to train Iris model with early stopping - CL

import os
import numpy as np
import json
import pandas as pd
# Set random seed
np.random.seed(0)
import tensorflow_hub as hub

import tensorflow as tf
import argparse
from os import listdir


import datetime
from IPython.display import Image
from sklearn.model_selection import train_test_split
#from matplotlib.pyplot import imread


NUM_IMAGES = 200
IMG_SIZE=224
BATCH_SIZE=32
NUM_EPOCHS=100

def process_image(image_path,img_size=IMG_SIZE):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.convert_image_dtype(image,tf.float32)
  image=tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image

def get_image_label(image_path,label):
  image = process_image(image_path)
  return image,label

def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  if test_data:
    print("Creating Test Data Batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch
  elif valid_data:
    print("Creating Validation Data Bataches...")
    data = tf.data.Dataset.from_tensor_slices(tf.constant(X),tf.constant(y))
    data_batch = data.map(get_image_label).batch(batch_size)
    return data_batch
  else:
    print("Creating Training Data Batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data=data.shuffle(buffer_size=len(X))
    data_batch=data.map(get_image_label).batch(batch_size)
    return data_batch

def create_model(train_data,val_data,input_shape,output_shape,model_url):
  print("Building model with : "+ model_url)
  print(input_shape)
  model=tf.keras.Sequential([
                             hub.KerasLayer(model_url),
                             tf.keras.layers.Dense(units=output_shape, activation="softmax")
                             ])
  
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )
    
  model.build(input_shape)

  return model


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def _load_data(file_path, channel):
    # Take the set of files and read them all into a single pandas dataframe
    csvfiles = list_files(file_path, "csv")
    print("CSV Files count:")
    print(csvfiles)
    input_files = [ os.path.join(file_path, file) for file in csvfiles]
    print("Input Files count:" + str(len(input_files)))
    #input_files = [ os.path.join(file_path, file) for file in os.listdir(file_path) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(file_path, channel))
        
    raw_data = [ pd.read_csv(file, engine="python") for file in input_files ]
    df = pd.concat(raw_data)  
    
    print("DF Count:")
    print(df.count())
    filenames = [file_path+"/"+fname+".jpg" for fname in df["ID"]]

    features = df["ID"].values
    label = df["Class"].values
    print(filenames)
    return  filenames,label


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    train_data, train_labels = _load_data(args.train,'train')
    eval_data, eval_labels = _load_data(args.test,'test')
    unique_class = np.unique(train_labels)
    train_boolean_labels=[label==unique_class for label in train_labels]
    eval_boolean_labels=[label==unique_class for label in eval_labels]
    print(train_boolean_labels)
    IMG_SIZE=224
    OUTPUT_SHAPE=len(unique_class)
    INPUT_SHAPE=[None,IMG_SIZE,IMG_SIZE,3]
    NUM_IMAGES = 200
    BATCH_SIZE=32
    NUM_EPOCHS=100
    MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
    train_data = create_data_batches(train_data,train_boolean_labels)
    eval_data = create_data_batches(eval_data,eval_boolean_labels)
    print(train_data)
    print(eval_data)
    classifier = create_model(train_data, eval_data,INPUT_SHAPE,OUTPUT_SHAPE,MODEL_URL)
    print("Done")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)
    classifier.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=eval_data,
            validation_freq=1,
            callbacks=[early_stopping])
    print("Predictions")
    predictions=classifier.predict(eval_data,verbose=1)
    print(predictions)
    if args.current_host == args.hosts[0]:
        #save model to an S3 directory with version number '00000001'
        classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
