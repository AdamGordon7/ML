import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

class convert:
  def rgb_to_grayscale(rgb_matrix):
        grayscale_matrix=np.zeros((28,28),dtype=int)

        for row in range(28):       
            for col in range(28):
                curr=rgb_matrix[row][col]
                r=curr[0]
                g=curr[1]
                b=curr[2]
                
                if r==0.000 and g==0.000 and b==0.000:
                  grayscale_matrix[row][col]=0
                else:
                  grayscale_matrix[row][col]=1

        return  grayscale_matrix 

#propogates a single datapoint through the network
def run(model, datapoint):
  #convert point to grayscale
  #reshape 
  #return argmax of the propogated point
  datapoint=convert.rgb_to_grayscale(datapoint)
  datapoint=datapoint.reshape(1,1,28,28)
  datapoint=torch.tensor(datapoint,dtype=torch.float)
  output=torch.argmax(model(datapoint)).numpy()

  return output


def main():
    #create model
    model=Net()

    model_weights_path='model.pth'
    #load the saved model and evaluate it 
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    #read in input and reshape
    inputs=np.loadtxt(sys.stdin).reshape(-1,28,28,3)


    # input_file="/Users/adamgordon/Downloads/school/third/ML/Project 2/assignment data/inputs.txt"
    # labels_file="/Users/adamgordon/Downloads/school/third/ML/Project 2/assignment data/labels.txt"
    # training_data=np.loadtxt(input_file,delimiter=" ")
    # training_labels=np.loadtxt(labels_file)

    # len_training_data=len(training_data)

    # training_data=training_data.reshape(len_training_data,28,28,3)

    # datapoint=training_data[0]
    # with torch.no_grad():
    #     output=run(model,datapoint)
    #     print(output)


    with torch.no_grad():
        for datapoint in inputs:
            output=run(model,datapoint)
            sys.stdout.write(str(output))

main()