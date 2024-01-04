import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


centers = [[-5, 2], [-2, 2], [1, 2], [5, -2]]

X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

print(f"{X_train}")

model = Sequential([
  Dense(25, activation='relu'),
  Dense(15, activation='relu'),
  Dense(4, activation='softmax')
])


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.legacy.Adam(0.001))

model.fit(
  X_train, y_train, epochs=20
)

p_nonpreferred = model.predict(X_train)

print(p_nonpreferred)
print(f"the largest value is {np.max(p_nonpreferred)}, and the smallest is {np.min(p_nonpreferred)}")


preferred_model = Sequential([
  Dense(25, activation='relu'),
  Dense(15, activation='relu'),
  Dense(4, activation='linear')
])

preferred_model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=tf.keras.optimizers.legacy.Adam(0.001),
)

preferred_model.fit(
  X_train, y_train, epochs=40
)

p_preffered=preferred_model.predict(X_train)
print(f"two example preferred vectors {p_preffered[:2]}")
print(f"largest value {np.max(p_preffered)} and the smallest is {np.min(p_preffered)}")

sm_preffered = tf.nn.softmax(p_preffered).numpy()
print(f"two example output vectors {sm_preffered[:2]}")
print(f"the largest value is {np.max(sm_preffered)} and the smallest is {np.min(sm_preffered)}")


for i in range(5):
  print( f"{p_preffered[i]}, category: {np.argmax(p_preffered[i])}")