import numpy as np
import sys

class NN:
    def __init__(self, sizes, fname_train, fname_test):
        """
        Initialiser for the neural network, sets 2 layers of 16 with their weights, 
        Each weight and neuron is initialised at random
        """
        
        self.layer_no = len(sizes)
        self.layer_sizes = sizes
        
        # initialising weights and biases in normal distribution, biases for layer 1 - layer_no, weights for 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.fname_test = fname_test
        self.fname_train = fname_train
        
    def readTrainData(self):
        """
        reading the data for training the nn
        """
        sys.stdout.write("Reading data...\r")
        # need method to only read some of these values, files are huge
        raw_train_data = np.loadtxt(self.fname_train, delimiter=",")
        
        self.train_data = []
        for i in range(raw_train_data.shape[0]):
            self.train_data.append(self.makeTuple(raw_train_data[i]))
            
        sys.stdout.write("Reading data complete.\n")
    
    def readTestData(self):
        """
        reading in the data for testing the nn
        """
        sys.stdout.write("Reading data...\r")
        raw_test_data = np.loadtxt(self.fname_test, delimiter=",")
        
        self.test_data = []
        for i in range(raw_test_data.shape[0]):
            self.test_data.append(self.makeTuple(raw_test_data[i]))
            
        return self.test_data
        sys.stdout.write("Reading data complete.\n")
        
    def buildTestData(self):
        """
        a test image to save time in the early debugging/building phase
        """
        self.buildtest = np.random.rand(100, 785)
        
        self.train_data = []
        for i in range(self.buildtest.shape[0]):
            self.train_data.append(self.makeTuple(self.buildtest[i]))
        
    
    def makeOpt(self, value):
        """
        produces the optimal output array for the given vvalue
        """
        array = np.zeros(10)
        array[int(value)] = 1.0
        
        return array
    
    def makeTuple(self, data):
        """
        takes data as an array where first value is number shown, rest are image grayscale
        returns a tuple (x, y) where x is img array, y is optimal activation
        """
        
        x = data[1:] / 255 # NEED TO NORMALISE THE IMAGE VALUES TO BETWEEN 0 AND 1
        y = self.makeOpt(data[0])
        
        return (x, y)
        
    
    def feedForward(self, a):
        """
        runs input a through the nn
        """
        
        for b, w in zip(self.biases, self.weights):
            
            a = np.dot(w, a)
            for i in range(a.shape[0]):
                a[i] += b[i]
                
            a = self.sigmoid(a)
            
        return a
    
    def stochasticGradDescent(self, epochs, batch_size, eta):
        """
        takes epoch number of random samples of training data and alters weights and biases to minimise cost function
        """
        n = len(self.train_data)
        # loop over epochs
        for i in range(epochs):
            # randomise samples and then make list of mini batches
            np.random.shuffle(self.train_data) #shuffles in place
            
            batches = [self.train_data[j : j + batch_size] for j in range(0, n, batch_size)]
            
            for batch in batches:
                self.updateNetwork(batch, eta)
            
            score = self.evaluate(self.test_data)
            score = (score / len(self.test_data)) * 100
            sys.stdout.write("After {} epochs: {}%\n".format(i, score))
                
    def updateNetwork(self, batch, eta):
        """
        updates weights and biases based on back propogation after feeding batch through the nn
        """
        grad_bs = [np.zeros(b.shape) for b in self.biases]
        grad_ws = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backProp(x, y) # find change in gradient
            
            for i in range(len(grad_bs)):
                for j in range(len(grad_bs[i])):
                    grad_bs[i][j] += delta_grad_b[i][j]
                    
            #grad_bs = [gb + dgb for gb, dgb in zip(grad_bs, delta_grad_b)] # update gradient
            grad_ws = [gw + dgw for gw, dgw in zip(grad_ws, delta_grad_w)]
            """
            for i in range(len(grad_ws)):
                for j in range(len(grad_ws[i])):
                    for k in range(len(grad_ws[i][j])):
                        grad_ws[i][j][k] += delta_grad_w[i][j][k]
            """
        # update weights and biases
        # new weight is current weight - average gradient for batch * speed factor, eta
        self.weights = [w - (eta / len(batch)) * gw for w, gw in zip(self.weights, grad_ws)]
        """
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= (eta / len(batch)) * grad_ws[i][j][k]
        """
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= (eta / len(batch)) * grad_bs[i][j]
                
        #self.biases = [b - (eta / len(batch)) * gb for b, gb in zip(self.biases, grad_bs)]

        
    def backProp(self, x, y):
        """
        back propogation algorithm
        """
        
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        current_layer_activation = x
        activations = [x] # list of activations layer by layer
        zs = []
        
        # feeding forward to get layers activations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_layer_activation)# + b # remember z is just unsigmoided activation
            for i in range(z.shape[0]): # error because final bias has second dimension
                z[i] += b[i]
            zs.append(z)
            current_layer_activation = self.sigmoid(z)
            activations.append(current_layer_activation)

        # going backwards
        
        delta = self.cost_der(activations[-1], y) * self.sigmoid_prime(zs[-1]) #  delta is the error in the layer, size 10

        # last set of ws and bs
        grad_b[-1] = delta 
        #grad_w[-1] = np.dot(delta, activations[-2].T) 

        for j in range(len(delta)): # this is a component wise calculation of weights
            for k in range(len(activations[-2].T)):
                grad_w[-1][j, k] = delta[j] * activations[-2].T[k]

        for l in range(2, self.layer_no):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            
            grad_b[-l] = delta
            #grad_w[-l] = np.dot(delta, activations[-l-1].T)

            for j in range(len(delta)):
                for k in range(len(activations[-l-1].T)):
                    grad_w[-l][j, k] = delta[j] * activations[-l-1].T[k]
        
        return (grad_b, grad_w)
    
    def cost_der(self, output_activation, y):
        """
        partial derivative of cost fnct wrt a
        both output and y are vectors
        """
        return (output_activation - y)
    
    def sigmoid(self, z):
        """
        returns x as a value between 0 and 1 using sigmoid function
        """
        return 1 / (1 + np.exp(-z))
        
    def sigmoid_prime(self, z):
        """
        first derivative of sigmoid wrt z
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def evaluate(self, test_data):
        """
        evaluates the effectiveness of the NN by evaluating number of correct 
        predictions in a set of test data
        """

        results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in test_data]
        
        return sum([(x == y) for x, y in results])
    