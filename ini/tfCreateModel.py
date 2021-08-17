# To add a new cell, type '# %%'
# %%
# functions for display

import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_image(i, predictionsArray, trueLabels, img):
    trueLabel, img = trueLabels[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predictedLabel = np.argmax(predictionsArray)
    if predictedLabel == trueLabel:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classNames[predictedLabel],
                                         100*np.max(predictionsArray),
                                         classNames[trueLabel]),
               color=color)


def plot_value_array(i, predictionsArray, trueLabels):
    trueLabel = trueLabels[i]
    plt.grid(False)
    plt.xticks(range(outputShape))
    plt.yticks([])
    thisplot = plt.bar(range(outputShape), predictionsArray, color="#777777")
    plt.ylim([0, 1])
    predictedLabel = np.argmax(predictionsArray)

    thisplot[predictedLabel].set_color('red')
    thisplot[trueLabel].set_color('blue')


# %%%
# TensorFlow and tf.keras

# Helper libraries

print(tf.__version__)


# %%

dataImages = None
dataLabels = None

datasetFileName = "dataset.npz"

with np.load(datasetFileName) as data:
    dataImages = data['images']
    dataLabels = data['labels']

classNames = ['Clean', 'Contaminated']


# %%
print(dataImages.shape)
dataImages.dtype

print(dataLabels.shape)
dataLabels.dtype


# %%
# Split the data up in train and test sets
trainImages, testImages, trainLabels, testLabels = train_test_split(
    dataImages, dataLabels, test_size=0.33, random_state=42)

del dataImages
del dataLabels

trainImages.dtype
print(trainImages.shape)

print(trainLabels.shape)
trainLabels.dtype

trainImages = trainImages / 255.0
testImages = testImages / 255.0

inputShape = trainImages[0].shape
outputShape = len(classNames)


# %%
maxIterations = 10
testAccList = []
thresholdAcc = 0.97
lastTestAcc = 0.0

model = None
testLoss = 0.0
testAcc = 0.0
modelDir = 'tf_model'

for iter in range(maxIterations):

    model = tf.keras.Sequential([
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Flatten(input_shape=inputShape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(outputShape)
    ])
    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(trainImages, trainLabels, epochs=10)

    testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=2)

    testAccList.append(testAcc)

    # print('\nTest accuracy:', testAcc)

    exportPath = ""

    if testAcc > thresholdAcc:
        # SavedModel format
        version = f"4_{(testAcc*100):.0f}"

        # for SavedModel format
        exportPath = os.path.join(modelDir, version)
        # save the model
        model.save(exportPath, save_format="tf")
        print('\nexport path = {}'.format(exportPath))

        # HDF5 format
        exportPath = os.path.join(modelDir, f"{version}.h5")
        # Save the entire model to a HDF5 file.
        # The '.h5' extension indicates that the model should be saved to HDF5.
        model.save(exportPath)
        print("saved: ", exportPath)

        thresholdAcc = testAcc

# %%
testAccList

#%%
