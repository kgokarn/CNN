#!/usr/bin/env python3

import keras
from keras.models import Model
import numpy as np
import cnn_common_big

weights_file = 'cnn_1train_default.h5'
exports_file = 'cnn_1train_weights_aftest.npz'

(x_train, y_train, x_test, y_test) = cnn_common_big.load_data()
model = cnn_common_big.make_model(x_train.shape[1:], y_train.shape[1])

print('loading model weights from:', weights_file)
model.load_weights(weights_file)

score = model.evaluate(x_test, y_test)
print('model.evaluate(x_test, y_test) =', score)

# save values of all layers of test[0] for debugging
t = np.array([x_test[0]])
m0 = Model(inputs=model.input, outputs=model.get_layer(index=0).output)
m1 = Model(inputs=model.input, outputs=model.get_layer(index=1).output)
m2 = Model(inputs=model.input, outputs=model.get_layer(index=2).output)
m3 = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
m4 = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
m5 = Model(inputs=model.input, outputs=model.get_layer(index=5).output)
m6 = Model(inputs=model.input, outputs=model.get_layer(index=6).output)
m7 = Model(inputs=model.input, outputs=model.get_layer(index=7).output)
m8 = Model(inputs=model.input, outputs=model.get_layer(index=9).output)
m9 = Model(inputs=model.input, outputs=model.get_layer(index=10).output)
m10 = Model(inputs=model.input, outputs=model.get_layer(index=11).output)
m11 = Model(inputs=model.input, outputs=model.get_layer(index=12).output)
m12 = Model(inputs=model.input, outputs=model.get_layer(index=13).output)

print('exporting data to:', exports_file)
np.savez(exports_file,
         l0weights=model.get_layer(index=0).get_weights()[0][::, ::, 0],
         l3weights=model.get_layer(index=3).get_weights()[0],
         l3biases=model.get_layer(index=3).get_weights()[1],
         l5weights=model.get_layer(index=5).get_weights()[0],
         l5biases=model.get_layer(index=5).get_weights()[1],
         l8weights=model.get_layer(index=9).get_weights()[0],
         l8biases=model.get_layer(index=9).get_weights()[1],
         l11weights=model.get_layer(index=12).get_weights()[0],
         l11biases=model.get_layer(index=12).get_weights()[1],
         l12weights=model.get_layer(index=13).get_weights()[0],
         l12biases=model.get_layer(index=13).get_weights()[1],
         x_test=x_test[:10000, ::, ::, 0],
         y_test=y_test[:10000],
         test0l0=m0.predict(t),
         test0l1=m1.predict(t),
         test0l2=m2.predict(t),
         test0l3=m3.predict(t),
         test0l4=m4.predict(t),
         test0l5=m5.predict(t),
         test0l6=m6.predict(t),
         test0l7=m7.predict(t),
         test0l8=m8.predict(t),
         test0l9=m9.predict(t),
         test0l10=m10.predict(t),
         test0l11=m11.predict(t),
         test0l12=m12.predict(t),
         test0l13=model.predict(t))
