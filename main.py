import NNMk2 as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys


if(__name__=="__main__"):
    if(len(sys.argv) < 2):
        sys.stdout.write("Must specify 'train', 'test' or 'show'.")
        exit()
    else:
        training_set = "mnist_train_reduced10.csv"
        testing_set = "mnist_test_reduced10.csv"
        fname = "SLP"
        if(sys.argv[1] == "train"):
            """
            train the network using the mnist dataset for 30 epochs
            """
            layers = [784, 100, 10]
            network = nn.NN(layers, training_set, testing_set) 
            
            #network.buildTestData()
            network.readTrainData()
            network.readTestData()
        
            network.stochasticGradDescent(epochs=30, batch_size=10, eta=0.1, lmbda=5.0)
            
            output = open("slp.pkl", "wb")
            pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)
            output.close()
            exit()
            
        elif(sys.argv[1] == "test"):
            """
            run some test data through the network and display the success rate
            """
            exit()
            
        elif(sys.argv[1] == "show"):
            """
            select n random images from the set and show them, with the guess they give
            """
            if(len(sys.argv) < 3):
                number_to_show = 1
            else:
                number_to_show = int(sys.argv[2])
                
            with open("slp.pkl", "rb") as input1:
                network = pickle.load(input1)
            
            raw_test_data = np.loadtxt(testing_set, delimiter=",")
            
            random_ints = np.random.randint(low=0, high=raw_test_data.shape[0], size=number_to_show)
            
            to_show = raw_test_data[random_ints]
            
            
            for i in range(to_show.shape[0]):
                image, answer = network.makeTuple(to_show[i])
                answer = np.argmax(answer)
                guess = np.argmax(network.feedForward(image))
                
                image = np.reshape(image, (28, 28))
                plt.figure(figsize=(5, 5))
                plt.imshow(image, cmap="gray")
                titlestring = "Guess: {}, Answer: {}".format(guess, answer)
                plt.title(titlestring)
                plt.show()
            exit()
            
        elif(sys.argv[1] == "random"):
            """
            run a random set of pixels through the NN and show confidence
            """
            if(len(sys.argv) < 3):
                number_to_show = 1
            else:
                number_to_show = int(sys.argv[2])
                
            with open("slp.pkl", "rb") as input1:
                network = pickle.load(input1)
                
            for i in range(number_to_show):
                random_image = np.random.randint(low=0, high=255, size=784)
                
                final_layer = network.feedForward(random_image)
                guess = np.argmax(final_layer)
                certainty = final_layer.max()
                
                random_image = np.reshape(random_image, (28, 28))
                plt.figure(figsize=(5, 5))
                plt.imshow(random_image, cmap="gray")
                titlestring="Guess: {}, Certainty: {:.2f}".format(guess, certainty)
                plt.title(titlestring)
                plt.show()