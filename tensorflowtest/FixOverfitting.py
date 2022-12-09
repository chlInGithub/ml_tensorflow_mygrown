import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras import regularizers

print(tf.__version__)

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile


logdir = pathlib.Path(tempfile.mktemp())/"tensorboard_logs"
print(logdir)
# 删除目录
shutil.rmtree(logdir, ignore_errors=True)

# 文件2G，太大了
#gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

FEATURES = 28
print([float(),] * (FEATURES+1))

ds = np.random.randn(10, 29)
print(ds[0])
print()
ds = tf.constant(ds)
print(ds)


def pack_row(row):
  label = row[0]
  print(label)
  features = tf.stack(row[1:], 1)
  return features, label

packed_ds = ds.batch(2).map(pack_row).unbatch()
print(packed_ds)

for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)