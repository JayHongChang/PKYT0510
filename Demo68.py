import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f"train shape={train_images.shape}, test shape={test_images.shape}")
print(f"train label length={len(train_labels)}, test length={len(test_labels)}")
print(np.unique(train_labels, return_counts=True))


def plotImage(index):
    plt.title("the image marked as %d" % train_labels[index])
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

def plotTestImage(index):
    plt.title("the image marked as %d"%test_labels[index])
    plt.imshow(test_images[index])
    plt.show()

plotTestImage(5000)