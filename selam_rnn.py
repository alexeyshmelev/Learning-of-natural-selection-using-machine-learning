import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader

num_classes = 2
input_size = 20
hidden_size = 20 
batch_size = 1   
sequence_length = 20 
num_layers = 1  # one-layer rnn

def FormData(train_number, class_number):

  data = []
  labels = []

  for i in range(train_number):
    for j in range(0, class_number):
      file_line = 'selection_' + str(j) + '_frequency_' + str(i) + '.txt'
      file = open(file_line, 'r')
      for line in file:
        array = line.split('\t')
        data += [float(array[2])]
      file.close()
  
  for i in range(class_number):
    labels.append(i)
  
  labels = Variable(torch.LongTensor(labels)) #сразу выводим labels в нужном формате

  return data, labels
      


class SampleDataset(Dataset):
  def __init__(self, secuence):
    self.samples=secuence
 
  def __len__(self):
      return len(self.samples)
 
  def __getitem__(self,idx):
      return (self.samples[idx])


class Model(nn.Module):

  def __init__(self, num_layers, hidden_size):
    super(Model, self).__init__() #вызываем конструктор класса nn.Module через метод super(), здесь показан пример для версии Python 2.0
    self.num_layers = num_layers #количество скрытых слоёв RNN
    self.hidden_size = hidden_size #сколько скрытых состояний
    self.rnn = nn.RNN(input_size=20, hidden_size=20, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(40, num_classes)
    self.relu=nn.ReLU()

  def forward(self, x): # Initialize hidden and cell states (num_layers * num_directions, batch, hidden_size)
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), 40))
    h_0 = h_0.view(2, -1, 20)
    #print("h_0:",h_0)

    x_m = x.view(batch_size, -1, sequence_length) # h_0: (num_layers * num_directions, batch, hidden_size)
    #print("x_m:",x_m)
    out, _ = self.rnn(x_m, h_0)
    out = self.relu(out)
    out = self.fc(out.view(-1, 40))
    #print("out:", out.view(-1, 40))
    return self.relu(out)

# Initialize model, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
model = Model(num_layers, hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(model)

data, labels = FormData(1, 2)
print(data, labels)
dataset = DataLoader(data, batch_size=40, shuffle=False, num_workers=2)

# Train the model
for i, batch in enumerate(dataset):
  inputs = batch.clone().detach().requires_grad_(True).float()
  inputs = inputs.view(1, 2, -1)
  print(inputs)
  for epoch in range(100):
    outputs = model(inputs)
    #print(outputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
   
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))


print("Learning finished!")
print(model(inputs))
