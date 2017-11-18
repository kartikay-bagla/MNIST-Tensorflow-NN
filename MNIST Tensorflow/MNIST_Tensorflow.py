# A neural network which trains and recognizes handwritten digits.
# Made by Kartkay Bagla, Class XII - D, Amity International 
# School, Noida. Using Python 3, and the tensorflow module.

#IMPORTS
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#GLOBAL VARS
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

no_nodes_hl1 = 500 #hidden layer 1 nodes
no_nodes_hl2 = 500 #hidden layer 2 nodes
no_nodes_hl3 = 500 #hidden layer 3 nodes
n_classes = 10     #output layer nodes
batch_size = 100   #no of images to be trained on at a time

x = tf.placeholder('float', [None, 784]) 
    #TF uses placeholders to store variable types and initializes them when the
    #session is run. X is a placeholder for an array of 784 values of the type
    #float.
    #None calls an error if anything else occupies the variable space.
    
y= tf.placeholder('float')

#FUNCTIONS
def neural_network_model(data):

    #weights n biases defined here
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,no_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([no_nodes_hl1]))}
                     
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([no_nodes_hl1,no_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([no_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([no_nodes_hl3,no_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([no_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([no_nodes_hl3,n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    #flow starts here
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        #matmul multiplies data(pixels) with weights and add adds the biases
        #matrix used to ensure that each pixel gets multiplied with specific weight only

    l1= tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
        #cost is a measure of how wrong we are and we want to minimize this

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    no_epochs = 10 # no of epochs i.e. cycles of feed forward and back propagation

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #initializes all vars

        for epoch in range(no_epochs):
            epoch_loss = 0 # loss in epoch

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) 
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c

            print ('Epoch', epoch + 1, 'completed out of', no_epochs, '\nloss:', epoch_loss)
        
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1)) 
                #tells us no of correct predictions

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print ('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#MAIN
train_neural_network(x)       

