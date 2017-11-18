Simple Neural Network to classify handwritten digits

Pictures of handwritten digits imported from the MNIST database in 28x28 pixel format

Steps of Working:
1) We define no of nodes for each layers. I.e. 3 hidden layers, 1 input layer and 1 output layer.
   Each hidden layer has 500 nodes, the input layer has 784 nodes(for each pixel) and the output
   layer has 10 nodes (one hot encoding for 0-9)

2) We resolve the 28x28 matrix into a [1,784] where each pixel contains a float which tells
   its grayscale value.

3) We then create the neural network model by defining the connections b/w all layers n
   specifying weights and biases(initilaized randomly).

4) We then define the flow by matrix multiplying data i.e. [1x784] with h1 weights [784x500] and
   add the biases for each node after which we pass it through ReLU for converting it into values
   b/w 0 to 1.

5) This flow is continued with the second and third layer till we get the output matrix as a [1x10]
   matrix which is the probability of each digit acc to neural network.

6) This prediction is stored into a variable which is used to calculate the error/cost. The cost
   is to be minimized which can only be done via modification of weights for which we use an
   optimizer called AdamOptimizer.

7) We run the session with a for loop with no of epochs(cycles) and then we calculate the epoch
   the epoch loss for each batch(55000 images in batches of 100(defined in Vars)). The loss is
   used to optimize the weights and biases which decreas the loss value and hence the accuracy
   improves making the network efficient.

Requires Tensorflow in Python 3.