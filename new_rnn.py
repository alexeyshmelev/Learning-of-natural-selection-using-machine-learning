import sys
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import settings

num_classes = settings.num_classes
input_size = settings.input_size
hidden_size = settings.hidden_size
batch_size = settings.batch_size  
sequence_length = settings.sequence_length
num_layers = settings.num_layers

print(sys.path)

def FormData(train_number, class_number):

  data = []

  for i in range(train_number):
    for j in range(0, class_number):
      file_line = 'sample_data/simulation/locus0.' + str(j) + '/selection_in_0.' + str(j) + '_var_' + str(i) + '.txt'
      file = open(file_line, 'r')
      for line in file:
        array = line.split('\t')
        data += [float(array[2])]
      file.close()
  
  #for i in range(class_number):
    #labels.append(i)
  
  #labels = Variable(torch.LongTensor(labels)) #сразу выводим labels в нужном формате
  #print("LABELS: ",labels)

  return data#, labels
      


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

  def forward(self, x): # Initialize hidden and cell states (num_layers * num_directions, batch, hidden_size)

    #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), 100))
    #h_0 = torch.zeros(1, 2, 10)
    #h_0 = h_0.view(-1, 2, 10)

    #print("h_0:",h_0)

    x_m = x.view(-1, sequence_length, 10) # h_0: (num_layers * num_directions, batch, hidden_size)
    #print("x_m:",x_m)
    out, _ = self.rnn(x_m)
    #print("OUTPUT: ", _);
    #print("OUT",out)
    #out = self.relu(out)
    out = self.linear(out.view(-1, 20))
    #print("out:", out.view(-1, 40))
    #return self.relu(out)
    return out

# Initialize model, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
model = Model(num_layers, hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(model)

#data, labels = FormData(10, 10)
data = FormData(10, 10)
#print(data, labels)
dataset = DataLoader(data, batch_size=10, shuffle=False, num_workers=2)

if settings.boot_from_file:
  model = Model(num_layers, hidden_size)
  model.load_state_dict(torch.load('sample_data/rnn.pth'))
  model.eval()
else:
# Train the model
  for i, batch in enumerate(dataset):
    inputs = batch.clone().detach().requires_grad_(True).float()
    inputs = inputs.view((1, 10))
    #print(inputs)
    labels = torch.zeros(1, dtype=torch.long)
    index = i % 10
    labels[0] = index
    for epoch in range(100):
      outputs = model(inputs)
      #outputs = outputs.view(10, 1)
      #print("OUTPUT: ", outputs.size())
      #print("LABELS: ", labels)
      optimizer.zero_grad()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
   
      print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))

torch.save(model.state_dict(), 'sample_data/rnn.pth')

print("Learning finished!")
m = nn.Softmax(dim=1)
test_data = torch.FloatTensor([0.9315, 0.966, 0.934, 0.9775, 0.9385, 0.951, 0.924, 0.9535, 0.932, 0.8845])
#test_data = torch.FloatTensor([0.9595, 0.909, 0.8625, 0.895, 0.8445, 0.9405, 0.9345, 0.8915, 0.9375, 0.886])
test_outputs = model(test_data)
print(m(test_outputs))
