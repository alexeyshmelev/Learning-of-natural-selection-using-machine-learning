import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import settings

#на даный момент сеть не предназначена для работы с участками естественного отбора, идущими НЕ через 1 (т.е. допустим 0.00, 0.02, 0.04 ... - не подходит)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(sys.path)
#print(device)

num_classes = settings.num_classes
input_size = settings.input_size
hidden_size = settings.hidden_size
batch_size = settings.batch_size  
sequence_length = settings.sequence_length
num_layers = settings.num_layers
train_number = settings.train_number
dim = settings.dim
line_format = '%.' + str(dim) + 'f'

def FormData(train_number, class_number):

  data = []

  for i in range(train_number):
    for j in range(0, class_number):
      if j == 0:
        path = '../selam/simulation/locus_0.0'
      else:
        locus_number = line_format % (j/num_classes)
        path = '../selam/simulation/locus_' + locus_number
      file_list = sorted(os.listdir(path))
      exact_file = path + '/' + file_list[i]
      file = open(exact_file, 'r')
      for line in file:
        array = line.split('\t')
        data += [float(array[2])]
      data.append(j)
      file.close()

  return data


class SampleDataset(TensorDataset):
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
    self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
    self.linear = nn.Linear(2*hidden_size, num_classes)

  def forward(self, x):
    x_m = x.view(-1, sequence_length, input_size) # (batch_size, sequence_length, input_size)
    out, _ = self.rnn(x_m)
    out = self.linear(out.view(-1, 2*hidden_size))
    return out

# Initialize model, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
model = Model(num_layers, hidden_size)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
print(model)

data = FormData(train_number, num_classes)

if settings.boot_from_file:

  model = Model(num_layers, hidden_size) #определяем класс с помощью уже натренированной сети
  model.to(device)
  model.load_state_dict(torch.load('rnn.pth'))
  model.eval()

else:

  for epoch in range(10):
    dataset = DataLoader(data, batch_size=settings.data_loader_batch_size, shuffle=False)
    for i, batch in enumerate(dataset): #тренируем сеть
      labels = batch[-1::].type(dtype=torch.long).to(device)
      inputs = batch[:-1:].clone().detach().requires_grad_(True).float().view((1, num_classes)).to(device)
      times = 0
      while True:
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print('Epoch: {}, Time: {}, loss: {:.6}'.format(epoch, times + 1, loss.item()))
        times += 1
        if loss < 0.01 and times > 6:
          break

  print("Learning finished!")

torch.save(model.state_dict(), 'rnn.pth')

m = nn.Softmax(dim=1)
test_data = []
path = '../selam/test_data_locus_0.9.txt'
file = open(path, 'r')
for line in file:
  array = line.split('\t')
  test_data += [float(array[2])]
file.close()
test_data = torch.tensor(test_data).to(device)
test_outputs = model(test_data)
test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
max = np.argmax(test_outputs)
print('Natural selection was in locus {:.2} with the probability of {}'.format(max/10, test_outputs[max]))
