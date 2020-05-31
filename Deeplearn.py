import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()


#########################################################

# Read dataset
def read_dataset():
    df = pd.read_csv("sonar.csv")
    print("Maravellous infosystems : Dataset loaded successfully")
    print("Marvellous infosystems: Number of columns", len(df.columns))

    # Feature of dataset
    X = df[df.columns[0:60]].values

    # Label of dataset
    y = df[df.columns[60]]

    # Encode the dependent variable
    encoder = LabelEncoder()

    # Encode character labels into integer 1 or 0(One hot encode)
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    print("Marvellous Infosystems: X.shape", X.shape)

    return (X, Y)


################################################################################

# Define the encoder function to set M => 1, R => 0
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


################################################################################

# Model for training
def multilayer_perceptron(x, weight, biases):
    # Hidden layer with RELU activations
    # First layer performs matrix multiplication with weights
    layer_1 = tf.add(tf.matmul(x, weight['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with sigmoid activations
    # Second layer performs matrix multiplication of layer1 with weights
    layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activations
    # Third layer performs matrix multiplication of layer2 with weights
    layer_3 = tf.add(tf.matmul(layer_2, weight['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with RELU activations
    # Fourth layer performs matrix multiplication of layer3 with weights
    layer_4 = tf.matmul(layer_3, weight['h4']) + biases['b4']
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activations
    out_layer = tf.matmul(layer_4, weight['out']) + biases['out']
    return out_layer


####################################################################################

def main():
    # Read dataset
    X, Y = read_dataset()

    # Shuffle the dataset to mix up the rows
    X, Y = shuffle(X, Y, random_state=1)

    # Convert the dataset into train and test datasets
    # FOr testing we use 10% and for training we use 90%
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.30, random_state=415)

    # Inspect the shape of the train and test datasets
    print("Marvellous Infosystems: train_x.shape", train_x.shape)
    print("Marvellous Infosystems: train_y.shape", train_y.shape)
    print("Marvellous Infosystems: test_x.shape", test_x.shape)
    print("Marvellous Infosystems: test_y.shape", test_y.shape)

    # Define the parameters which are required for rensors ie hyperparameters

    # Change in a variable in each iteration
    learning_rate = 0.3

    # Total number of iterations to minimize the error
    training_epochs = 1000
    cost_history = np.empty(shape=[1], dtype=float)

    # Number of features <=> number of columns
    n_dim = X.shape[1]
    print("Number of columns are n_dim", n_dim)

    # As we have two classes as R and M
    n_class = 2

    # Path which contains model fines
    model_path = "Marvellous"

    # Define the number o#f hidden layers an the
    # number of neurons for each layer

    # Number of hidden layer are 4
    # NNeurons in each layer are 60
    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    # Placeholder to store inputs
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, [None, n_dim])

    # Placeholder to store outputs
    y_ = tf.compat.v1.placeholder(tf.float32, [None, n_class])

    # Model parameters
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # define the weights and biases for each layer
    # Create variable which contains random values
    weights = {
        'h1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class])),
    }

    # Create varible which contains random variables
    biases = {
        'b1': tf.Variable(tf.random.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_class]))
    }

    # Initialization of variables
    init = tf.compat.v1.initialize_all_variables()

    saver = tf.compat.v1.train.Saver()

    # Call to model function for training
    y = multilayer_perceptron(x, weights, biases)

    # Define the cost of function to calculate loss
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # Function to reduce loss
    training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Calculate the cost and accuracy for each epoch
    for epoch in range(training_epochs):
        sess = tf.compat.v1.Session()
        sess.run(init)

        sess.run(training_step,feed_dict = {x:train_x,y_:train_y})
        accuracy_history = []
        cost = sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
        cost_history = np.append(cost_history,cost)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(y,feed_dict={x:test_x})
        mse = tf.reduce_mean(tf.square(pred_y-test_y))
        mse_ = sess.run(mse)
        accuracy = (sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))
        accuracy_history.append(accuracy)
        print('epoch: ',epoch,'-','cost:',cost,"- MSE: ",mse_,"- Train Accuracy: ",accuracy)





    # Model gets saved in the file
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s", save_path)

    # Display graph for accuracy history
    plt.plot(accuracy_history)
    plt.title("Marvellous Infosystems: Accuracy History")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # Print the mean square error
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.square(pred_y - test_y))
    print("Test Accuract:", (sess.run(y, feed_dict={x: test_x, y_: test_y})))

    # Print the final mean square error
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    print("Mean Square Error: %4f" % sess.run(mse))


if __name__ == "__main__":
    main()
