from keras import utils

orig = 4
orig2 = 9
NUM_DIGIT = 20
print(f"orig:{orig}, converted={utils.to_categorical(orig, NUM_DIGIT)}")
print(f"orig:{orig2}, converted={utils.to_categorical(orig2, NUM_DIGIT)}")