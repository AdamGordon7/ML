import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST


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


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print("EXAMPLE: ",example_data.shape)

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

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      #save current state to file for reload later 
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs+1 ):
  train(epoch)
  test()



input_file="/Users/adamgordon/Downloads/school/third/ML/Project 2/assignment data/inputs.txt"
labels_file="/Users/adamgordon/Downloads/school/third/ML/Project 2/assignment data/labels.txt"

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




training_data=np.loadtxt(input_file,delimiter=" ")
training_labels=np.loadtxt(labels_file)

len_training_data=len(training_data)

training_data=training_data.reshape(len_training_data,28,28,3)

# training_data_grayscale=np.zeros((len_training_data,28,28))
# for i in range(len_training_data):
#     training_data_grayscale[i]=convert.rgb_to_grayscale(training_data[i])

# final_traing_data=training_data_grayscale


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

predictions=[]
with torch.no_grad():
  for i in training_data:
    prediction=run(network,i)
    predictions.append(prediction)

errors=0
for i in range(2000):
  if(predictions[i]!=training_labels[i]):
    errors+=1

print("MY TEST: ", 100-(errors/2000)*100)










