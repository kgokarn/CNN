#!/usr/bin/env python3

import keras
import numpy as np
import cnn_common_big

weights_file = 'cnn_1train_default.h5'

(x_train, y_train, x_test, y_test) = cnn_common_big.load_data()
model = cnn_common_big.make_model(x_train.shape[1:], y_train.shape[1])

keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3)
#model.load_weights(weights_file)
model.fit(x_train, y_train, batch_size=64, epochs=100,
          callbacks=[cnn_common_big.WeightScale()])

score = model.evaluate(x_test, y_test)
print('model.evaluate(x_test, y_test) =', score)

print('saving model weights to', weights_file)
model.save_weights(weights_file)

