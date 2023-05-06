import tensorflow as tf

'''
Készíts egy metódust ami a mnist adatbázisból betölti a train és test adatokat. (tf.keras.datasets.mnist.load_data())
Majd a tanitó, és tesztelő adatokat normalizálja, és vissza is tér velük.


Egy példa a kimenetre: train_images, train_labels, test_images, test_labels
függvény neve: mnist_digit_data
'''

def mnist_digit_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255
    train_labels = train_labels / 255
    train_labels = train_labels.astype("float32") / 255
    test_labels = test_labels.astype("float32") / 255
    return train_images, train_labels, test_images, test_labels

'''
Készíts egy neurális hálót, ami képes felismerni a kézírásos számokat.
A háló kimenete legyen 10 elemű, és a softmax aktivációs függvényt használja.
Hálon belül tetszőleges számú réteg lehet.


Egy példa a kimenetre: model,
return type: keras.engine.sequential.Sequential
függvény neve: mnist_model
'''

def mnist_model() -> tf.keras.Sequential:
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
  ])

  model = tf.keras.Sequential([model,
                       tf.keras.layers.Softmax()])
  return model

'''
Készíts egy metódust, ami a bemeneti hálot compile-olja.
Optimizer: Adam
Loss: SparseCategoricalCrossentropy(from_logits=False)

Egy példa a bemenetre: model
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_compile
'''

def model_compile(model: tf.keras.Sequential) -> tf.keras.Sequential:
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy']
  )
  return model

'''
Készíts egy metódust, ami a bemeneti hálót feltanítja.

Egy példa a bemenetre: model,epochs, train_images, train_labels
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_fit
'''

def model_fit(model: tf.keras.Sequential, epochs: int, train_images, train_labels) -> tf.keras.Sequential:
  model.fit(train_images, train_labels, epochs=epochs)
  return model

'''
Készíts egy metódust, ami a bemeneti hálót kiértékeli a teszt adatokon.

Egy példa a bemenetre: model, test_images, test_labels
Egy példa a kimenetre: test_loss, test_acc
return type: float, float
függvény neve: model_evaluate
'''

def model_evaluate(model: tf.keras.Sequential, test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
  return test_loss, test_acc

'''
train_images, train_labels, test_images, test_labels = mnist_digit_data()
model = mnist_model()
model = model_compile(model)
model = model_compile(model)
model = model_fit(model, 10, train_images, train_labels)

test_loss, test_acc = model_evaluate(model, test_images, test_labels)

print("loss:", test_loss)
print("acc:", test_acc)
'''