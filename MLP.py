#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
import sys
try:
	os.chdir(os.path.join(os.getcwd(), '.ipynb_checkpoints'))
	print(os.getcwd())
except:
	pass


class NN(object):
    
    def __init__(self,hidden_dims=(1024,2048),n_hidden=2,mode='train',datapath=None,model_path=None):
        h1 = hidden_dims[0]
        h2 = hidden_dims[1]
        out = 10
        features = 784
        total_param_h1 = features + h1 * features + h1 * 1
        total_param_h2 = total_param_h1 + h2 * total_param_h1 + h2 * 1
        total_param_nn = total_param_h2 + out * total_param_h2

    def verif_param_nn(self, total_param_nn):
        if not(total_param_nn < 1000000 && total_param_nn > 500000):
            sys.exit('ERROR! Number of parameter in NN: ',total_param_nn)
    
    #%%
    def initialize_weights(self,n_hidden,dims):

    #%%
    def forward(self,input,labels,..):

    #%%
    def activation(self,input):

    #%%
    def loss(self,prediction,..):

    #%%
    def softmax(self,input,..):

    #%%
    def backward(self,cache,labels,...):

    #%%
    def update(self,grads,..):

    #%%
    def train(self):

    #%%
    def test(self):


