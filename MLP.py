#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '.ipynb_checkpoints'))
	print(os.getcwd())
except:
	pass

#%%
class NN(object):
    
    def __init__(self,hidden_dims=(1024,2048),n_hidden=2,mode='train',datapath=None,model_path=None):
        .
        .
        .


#%%
def initialize_weights(self,n_hidden,dims):
    .
    .
    


#%%
def forward(self,input,labels,..):
    .
    .


#%%
def activation(self,input):
    .
    .


#%%
def loss(self,prediction,..):
    .
    .


#%%
def softmax(self,input,..):
    .
    .


#%%
def backward(self,cache,labels,...):
    .
    .


#%%
def update(self,grads,..):
    .
    .


#%%
def train(self):
    .
    .


#%%
def test(self):
    .
    .


