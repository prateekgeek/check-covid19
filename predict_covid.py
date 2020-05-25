# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam


INIT_LR = 1e-3
EPOCHS = 25
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# load model
model = load_model('covid19.model.old')
#model.compile(loss="binary_crossentropy", optimizer=opt,
#	metrics=["accuracy"])
# summarize model.
#model.summary()
# load dataset

img_width, img_height = 224, 224
img = cv2.imread('NORMAL.png')
img = np.array(img) / 255.0
img = cv2.resize(img, (img_width, img_height))
img = np.reshape(img, [1, img_width, img_height, 3])
#dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# evaluate the model
#score = model.evaluate(X, Y, verbose=0)
predIdxs = model.predict(img)
predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs)
