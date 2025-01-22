"""

https://www.tensorflow.org/api_docs/python/tf/keras/metrics/KLDivergence

"""

import tensorflow as tf
from tensorflow import keras

m = keras.metrics.KLDivergence()
m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
m.result()