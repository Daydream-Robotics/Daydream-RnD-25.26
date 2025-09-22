import tensorflow as tf
print("Tensorflow ver:", tf.__version__)
from keras import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import losses, optimizers, metrics
import datetime
#timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = tf.summary.create_file_writer(log_dir + "/train")
test_writer  = tf.summary.create_file_writer(log_dir + "/test")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32).cache().prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).cache().prefetch(tf.data.AUTOTUNE)

class MyModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(64, 3, padding="same", activation='relu')
        self.pool = MaxPooling2D()
        self.flatten = Flatten()
        self.dropout = Dropout(0.3)
        self.d1 = Dense(128, activation='relu')
        self.out = Dense(10)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.d1(x)
        return self.out(x)
    
model = MyModel()
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam()

train_loss = metrics.Mean(name="train_loss")
train_acc  = metrics.SparseCategoricalAccuracy(name="train_acc")
test_loss  = metrics.Mean(name="test_loss")
test_acc   = metrics.SparseCategoricalAccuracy(name="test_acc")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_acc.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_fn(labels, predictions)
    test_loss.update_state(t_loss)
    test_acc.update_state(labels, predictions)


EPOCHS = 12
for epoch in range(EPOCHS):
    #reset metrics
    train_loss.reset_state(); train_acc.reset_state()
    test_loss.reset_state(); test_acc.reset_state()
    
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={train_loss.result():.4f} train_acc={train_acc.result()*100:.2f}% | "
        f"test_loss={test_loss.result():.4f}  test_acc={test_acc.result()*100:.2f}%"
    )

    with train_writer.as_default():
        tf.summary.scalar("loss", train_loss.result(), step=epoch)
    tf.summary.scalar("accuracy", train_acc.result(), step=epoch)

    with test_writer.as_default():
        tf.summary.scalar("loss", test_loss.result(), step=epoch)
        tf.summary.scalar("accuracy", test_acc.result(), step=epoch)