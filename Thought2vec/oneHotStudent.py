import pandas as pd
import numpy as np

class oneHotStudent():
    '''
    Turn each student into a one-hot vector encoding of their FCI answers, creating a format to run
    through thought2vec.
    
    The class is called on each row of FCI test data, incidating if the pre or posttest is used.
    '''
    def __init__(self, df_student, pre):
        self.student = df_student.reset_index()
        self.pre = pre
        self.df_onehot = self.oneHotQ(self.student,self.pre)
        
    @staticmethod
    def oneHotQ(student,pre=True):
        '''
        Turn each student's answers to the FCI into a 30 x 150 matrix of one-hot encoded answers to each
        FCI question. Answering 'A' to the first question gives a vector [1,0,0,...,0]; answering 'B' to 
        that same question gives [0,1,0,0,...,0]; answering E to the last question gives [0,0,0,...,1].

        This preprocesses each student's answer into a form to run through Thought2vec
        '''
        if pre == True:
            col_list = ['001', '002', '003', '004', '005', '006', '007', '008',
           '009', '010', '011', '012', '013', '014', '015', '016', '017', '018',
           '019', '020', '021', '022', '023', '024', '025', '026', '027', '028',
           '029', '030']
        else:
            col_list = ['101', '102', '103', '104', '105', '106', '107', '108',
           '109', '110', '111', '112', '113', '114', '115', '116', '117', '118',
           '119', '120', '121', '122', '123', '124', '125', '126', '127', '128',
           '129', '130']
        
        zero_matrix = np.zeros((30,150))
        df_onehot = pd.DataFrame(zero_matrix, columns=range(1,151))

        for row in range(0,student.shape[0]):
            for i, col in enumerate(col_list):
                #zeros = [0]*150
                val = int(student.iloc[row,i+2])
                #zeros[val-1+i*5] = 1
                #df_onehot.loc[row*30+i] = zeros
                df_onehot.iloc[row*30+i, val-1+i*5] = 1

        return df_onehot
    
    
    def trainTargetDF(self):
        '''
        Arrange each question to become the training vector and other vectors to become target vectors. When
        selecting a question after question 1 as trainging question, the other vectors will create a cylce.
        For example, if question 28 is chosen, then the order of target question vectors will become 
        29, 30, 1, 2, 3, ..., 26, 27.
        Question should be an integer between 1 and 30, inclusive.
        '''
        for question in range(1, 31):
            df_train = self.df_onehot.loc[question - 1]
            df_target = pd.concat([self.df_onehot.loc[question:30], self.df_onehot.loc[:question-2]])
            for i in range(0,29):
                yield np.array(df_train).astype(np.float64), np.array(df_target).astype(np.float64)[i]