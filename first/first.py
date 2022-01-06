print("STEP1")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("STEP2")
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

print("STEP3")
import tensorflow as tf

try:
    print("STEP4")
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.\
    print("INVALID DEVICE")
    pass

print("STEP5")
print("TensorFlow version:", tf.__version__)

print("STEP6")
mnist = tf.keras.datasets.mnist

print("STEP7")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("STEP8")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

print("STEP9")
predictions = model(x_train[:1]).numpy()
print(predictions)

print("STEP10")
res=tf.nn.softmax(predictions).numpy()
print(res)

print("STEP11: LOSS")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
theloss=loss_fn(y_train[:1], predictions).numpy()
print(theloss)

print("STEP12: COMPILE")
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("STEP13: FIT")
model.fit(x_train, y_train, epochs=5)

print("STEP14: EVALUATE")
evaluation=model.evaluate(x_test,  y_test, verbose=2)
print(evaluation)


print("STEP15: PROBABILITY")
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
theproba=probability_model(x_test[:5])
print(theproba)

