import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("MNIST_data/",one_hot=True)
mnist.train.images.shape
mnist.test.images.shape
tf.__version__
mnist.train.labels[0:4]
img_shape= (28,28)
img_size_flat = 784
num_classes = 10
def getClass(c):
    for i ,j in enumerate(c):
        if j==1.0:
            return i


def plot_images(images,cls_true,cls_pred=None):
    assert len(images) == len(cls_true)==9
    fig,axes  = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(getClass(cls_true[i]))
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

images = mnist.test.images[0:9]
cls_true = mnist.test.labels[0:9]
plot_images(images,cls_true)

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
def optimize(num_iterations):
    for i in range(num_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: batch_x,
                           y_true: batch_y}
        session.run(optimizer, feed_dict=feed_dict_train)

feed_dict_test = {x: mnist.test.images,
                  y_true: mnist.test.labels,
                  y_true_cls: np.argmax(mnist.test.labels, axis=1)}


def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true =np.argmax(mnist.test.labels, axis=1)
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def plot_example_errors():
    correct,cls_pred=session.run([correct_prediction,y_pred_cls],feed_dict=feed_dict_test)
    incorrect =(correct==False)
    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = np.argmax(mnist.test.labels[incorrect], axis=1)
    plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_weights():
    w =session.run(weights)
    w_max = np.max(w)
    w_min = np.min(w)
    fig,axes =plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):

        if i < 10:
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()



print_accuracy()
#Accuracy on test-set: 9.8%
plot_example_errors()

optimize(num_iterations=1)
print_accuracy()
#Accuracy on test-set: 30.5%
plot_example_errors()
plot_weights()

optimize(num_iterations=9)
print_accuracy()
#Accuracy on test-set: 69.1%
plot_example_errors()
plot_weights()

optimize(num_iterations=990)
print_accuracy()
#Accuracy on test-set: 91.5%
plot_example_errors()
plot_weights()
print_confusion_matrix()
