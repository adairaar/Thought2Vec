import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm, trange
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

import oneHotStudent as ohs
from ml_tools import softmax, initialize

class Thought2Vec:
    """
    Python implementation of Word2Vec.
    # Arguments
        method : `str`
            choose method for word2vec (options: 'cbow', 'skipgram')
            [default: 'cbow']
        window_size: `integer`
            size of window [default: 1]
        n_hidden: `integer`
            size of hidden layer [default: 2]
        n_epochs: `integer`
            number of epochs [default: 1]
        learning_rate: `float` [default: 0.1]
        corpus: `str`
            corpus text
    """
    def __init__(self, method='cbow', window_size=1, n_hidden=2, n_epochs=1, 
                 corpus=pd.DataFrame(), learning_rate=0.1, early_stop = 1):
        self.window = window_size
        self.N = n_hidden
        self.n_epochs = n_epochs
        self.corpus = corpus
        self.eta = learning_rate
        self.early_stop = early_stop
        if method == 'cbow':
            self.method = self.cbow
        elif method == 'skipgram':
            self.method = self.skipgram
        else:
            raise ValueError("Method not recognized. Aborting.")
    
    def onehotvecs(self, df):
        r = df.shape[0]
        stu_dict = {}
        for i in range(r):
            stu = ohs.oneHotStudent(df[i:i+1],pre=0)
            stu_dict[i] = stu.df_onehot
        return stu_dict
    
    def cbow(self, context, label, W1, W2, loss):
        """
        Implementation of Continuous-Bag-of-Words Word2Vec model
        :param context: all the context words (these represent the inputs)
        :param label: the center word (this represents the label)
        :param W1: weights from the input to the hidden layer
        :param W2: weights from the hidden to the output layer
        :param loss: float that represents the current value of the loss function
        :return: updated weights and loss
        """
        # context is 'x' from tokenizer, it is a c x V matrix
        # label is 'y' from tokenizer, it is a 1 x V matrix
        
        label = pd.DataFrame(label).T
        x = np.matrix(np.mean(context, axis=0))
        
        # x is a 1 x V matrix
        # W1 is a VxN matrix
        # h is a N x 1 matrix
        h = np.matmul(W1.T, x.T)
        
        # u is a V x 1 matrix
        u = np.matmul(W2.T, h)
        
        # W2 is an N x V matrix
        # y_pred is a V x 1 matrix
        y_pred = softmax(u)
        # e is a V x 1 matrix
        e = -label.T + y_pred
        e = e.to_numpy()
        # h is N x 1 and e is V x 1 so dW2 is N x V
        dW2 = np.outer(h, e)
        # x.T is a V x 1 matrix, W2e is a Nx1 so dW1 this is V x N
        dW1 = np.outer(x.T, np.matmul(W2, e))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        # label is a 1xV matrix so label.T is a Vx1 matrix
        label = label.to_numpy()
        loss += -float(np.sum([label.T == 1])) + np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss

    def skipgram(self, context, x, W1, W2, loss):
        """
        Implementation of Skip-Gram Word2Vec model
        :param context: all the context words (these represent the labels)
        :param x: the center word (this represents the input)
        :param W1: weights from the input to the hidden layer
        :param W2: weights from the hidden to the output layer
        :param loss: float that represents the current value of the loss function
        :return: updated weights and loss
        """
        # context is "x" from tokenizer, it is a c x V matrix
        # "x" is "y" from tokenizer, it is a 1 x V matrix
        # W1 has dimension V x N (N= number of features, V = vocab size)
        # x has dimension V x 1
        h = np.matmul(W1.T, x.T)
        # h has dimension N x 1
        # W2 has dimension N x V
        # u has dimension V x 1
        u = np.dot(W2.T, h)
        # y_pred has dimension V x 1
        y_pred = softmax(u)

        # context is a c by V matrix
        # e is a V x c matrix
        e = np.outer(y_pred,np.array([1]*context.shape[0]))-context.T

        # np.sum(e, axis=1) is a V x 1 vectors
        # h is an N x 1 Vector
        # dW2 is a N x V matrix
        dW2 = np.outer(h, np.sum(e, axis=1))
        # x is a V x 1 matrix
        # np.dot(W2, np.sum(e,axis=1)) is a product (N x V) (Vx 1) is Nx1
        # dW1 is an V x N matrix
        dW1 = np.outer(x.T, np.dot(W2, np.sum(e, axis=1)))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        loss += - np.sum([[label.T == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss
    
    def trainTargetDF(self, df):
        '''
        Arrange each question to become the training vector and other vectors to become target vectors. When
        selecting a question after question 1 as trainging question, the other vectors will create a cylce.
        For example, if question 28 is chosen, then the order of target question vectors will become 
        29, 30, 1, 2, 3, ..., 26, 27.
        Question should be an integer between 1 and 30, inclusive.
        '''
        for question in range(1, 31):
            df_train = df.loc[question - 1]
            df_target = pd.concat([df.loc[question:30], df.loc[:question-2]])
            yield np.array(df_train).astype(np.float64), np.array(df_target).astype(np.float64)
            

    def predict(self, x, W1, W2):
        """Predict output from input data and weights
        :param x: input data
        :param W1: weights from input to hidden layer
        :param W2: weights from hidden layer to output layer
        :return: output of neural network
        """
        h = np.mean([np.matmul(W1.T, xx) for xx in x], axis=0)
        u = np.dot(W2.T, h)
        return softmax(u)

    def run(self):
        """
        Main method of the Word2Vec class.
        :return: the final values of the weights W1, W2 and a history of the value of the loss function vs. epoch
        """
        if len(self.corpus) == 0:
            raise ValueError('You need to specify a corpus of text.')

            
        print("Creating one-hot student answer vectors")
        cores=mp.cpu_count()
        stu_dict = {}
        df_split = np.array_split(self.corpus, cores, axis=0)

        # create the multiprocessing pool
        pool = Pool(cores)

        # process the DataFrame by mapping function to each df across the pool
        df_out = np.vstack(pool.map(self.onehotvecs, df_split))

        # close down the pool and join
        pool.close()
        pool.join()
        pool.clear()

        print("Creating student answer dictionary")
        row_count = 0
        for i in range(0,cores):
            for j in range(0,len(df_out[i][0])):
                stu_dict[row_count] = df_out[i][0][j]
                row_count += 1
                
        # initialize weight matrices
        print("Initializing weights")
        V = stu_dict[0].shape[1]
        W1, W2 = initialize(V, self.N)

        loss_vs_epoch = []
        loss_low = np.inf
        
        print("Begining training")
        for e in trange((self.n_epochs), desc='Epochs'):
            loss = 0.0
            rand_student_order = np.random.choice(self.corpus.shape[0], 
                                                  self.corpus.shape[0], 
                                                  replace=False)
            # shuffle data without replacement
            for i in tqdm(rand_student_order, desc='Students', leave=False):
                for center, context in self.trainTargetDF(stu_dict[i]):
                    W1, W2, loss = self.method(context, center, W1, W2, loss)
            loss_vs_epoch.append(loss)
            
            if loss < loss_low:
                loss_low = loss
                W1_best, W2_best = W1, W2
                
            # Early stopping and returning best result
            if loss > loss_vs_epoch[max(0,e-self.early_stop)]:
                print("Training complete. Loss now increasing.")
                return W1_best, W2_best, loss_vs_epoch

        print("Training complete.")
        return W1, W2, loss_vs_epoch