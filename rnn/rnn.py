import os
import sys
import time
import torch
import settings #our custom file
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from multiprocessing import Process
from torch.utils.data import TensorDataset, DataLoader

#на даный момент сеть не предназначена для работы с участками естественного отбора, идущими НЕ через 1 (т.е. допустим 0.00, 0.02, 0.04 ... - не подходит)

num_classes = settings.num_classes
input_size = settings.input_size
hidden_size = settings.hidden_size
batch_size = settings.batch_size  
sequence_length = settings.sequence_length
num_layers = settings.num_layers
train_number = settings.train_number
dim = settings.dim
line_format = '%.' + str(dim) + 'f'

def SLS(): ##########################################################################################################################################################################################################

  def FormData(train_number, class_number):

    data = []
    temp_data = []

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
          temp_data += [float(array[2])]
        temp_data.append(j)
        data.append(temp_data)
        temp_data = []
        file.close()

    return data


  class SampleDataset(TensorDataset):
    def __init__(self, secuence):
      self.samples=torch.tensor(secuence)
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        return (self.samples[index])


  class SLS(nn.Module):

    def __init__(self, sls_input_size, sls_hidden_size, sls_num_layers, sls_sequence_length, sls_num_classes):
      super(SLS, self).__init__() #вызываем конструктор класса nn.Module через метод super(), здесь показан пример для версии Python 2.0

      self.sls_input_size = sls_input_size
      self.sls_hidden_size = sls_hidden_size
      self.sls_num_layers = sls_num_layers
      self.sls_sequence_length = sls_sequence_length
      self.sls_num_classes = sls_num_classes

      self.rnn = nn.LSTM(input_size = self.sls_input_size, hidden_size = self.sls_hidden_size, batch_first=True, bidirectional=True)
      self.linear = nn.Linear(2*self.sls_input_size, self.sls_input_size)

    def forward(self, input):

      input = input.view(-1, self.sls_sequence_length, self.sls_input_size) # (batch_size, sequence_length, input_size)
      output, _ = self.rnn(input)
      output = output.view(self.sls_sequence_length, 2*self.sls_input_size)
      out = self.linear(output)
      out = out.view(-1, self.sls_num_classes)
      
      return out

  data = FormData(train_number, num_classes)

  if settings.boot_from_file:

    model_sls = SLS(num_layers, hidden_size) #определяем класс с помощью уже натренированной сети
    model_sls.cuda(0)
    model_sls.load_state_dict(torch.load('rnn.pth'))
    model_sls.eval()

  else:

    # Initialize model_sls, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_sls = SLS(input_size, hidden_size, num_layers, sequence_length, num_classes)
    model_sls.cuda(0)
    criterion_sls = torch.nn.CrossEntropyLoss()
    optimizer_sls = torch.optim.Adam(model_sls.parameters(), lr=0.006)
    print(model_sls)
    x_axis = []
    y_axis = []

    start_time = time.time()

    for epoch in range(20):
      print('Epoch_sls: {}, time spent: {} sec'.format(epoch, time.time() - start_time), flush=True)
      start_time = time.time()
      x_axis.append(len(x_axis))
      temp_loss = []
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        labels = batch[0][-1::].type(dtype=torch.long).cuda(0)
        inputs = batch[0][:-1:].clone().detach().requires_grad_(True).float().view(1, num_classes).cuda(0)
        times = 0
        while True:
          outputs = model_sls(inputs)
          optimizer_sls.zero_grad()
          loss = criterion_sls(outputs, labels)
          temp_loss.append(loss.item())
          loss.backward()
          optimizer_sls.step()
          times += 1
          if times > 0:
            break
      y_axis.append(sum(temp_loss)/(2*num_classes))

    plt.clf()
    plt.plot(x_axis, y_axis, '.-g', label='Mean loss', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig('sls_loss.png')

    print("Learning finished!")

  torch.save(model_sls.state_dict(), 'rnn.pth')

  m = nn.Softmax(dim=1)
  test_data = []
  path = '../selam/test_data_locus_0.9_generation_20.txt'
  file = open(path, 'r')
  for line in file:
    array = line.split('\t')
    test_data += [float(array[2])]
  file.close()
  test_data = torch.tensor(test_data).cuda(0)
  test_outputs = model_sls(test_data)
  test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
  max = np.argmax(test_outputs)
  print('Natural selection was in locus {:.2} with the probability of {}'.format(max/10, test_outputs[max]))

def GEN(): ######################################################################################################################################################################################################################

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
        #print(int(exact_file.split('_')[7]), flush=True)
        data.append(int(exact_file.split('_')[7]))
        file.close()

    return data


  class SampleDataset(TensorDataset):
    def __init__(self, secuence):
      self.samples=secuence
  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return (self.samples[idx])

  class GEN(nn.Module):

    def __init__(self, gen_num_classes, gen_hidden_size):
      super(GEN, self).__init__() 
      self.gen_num_classes = gen_num_classes
      self.gen_hidden_size = gen_hidden_size
      self.rnn = nn.RNN(gen_num_classes, gen_hidden_size, batch_first=True, bidirectional=True)
      self.linear = nn.Linear(2*gen_hidden_size, batch_size)

    def forward(self, x):
      x = x.view(-1, sequence_length, input_size) # (batch_size, sequence_length, input_size)
      out, _ = self.rnn(x)
      out = self.linear(out.view(-1, 2*self.gen_hidden_size))
      #print('GEN {}'.format(out))
      return out


  data = FormData(train_number, num_classes)

  if settings.boot_from_file:
    a = 1
    #model_sls = SLS(num_layers, hidden_size) #определяем класс с помощью уже натренированной сети
    #model_sls.to(device)
    #model_sls.load_state_dict(torch.load('rnn.pth'))
    #model_sls.eval()

  else:

    # Initialize model_gen, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_gen = GEN(num_classes, hidden_size)
    model_gen.cuda(1)
    criterion_gen = torch.nn.MSELoss() 
    optimizer_gen = torch.optim.SGD(model_gen.parameters(), lr=0.006)
    print(model_gen)
    x_axis = []
    y_axis = []

    for epoch in range(1):
      dataset = DataLoader(data, batch_size=settings.data_loader_batch_size, shuffle=False)
      for i, batch in enumerate(dataset): #тренируем сеть
        labels = batch[-1::].type(dtype=torch.float).view(1, 1).cuda(1)
        #print(labels.size(), flush=True)
        inputs = batch[:-1:].clone().detach().requires_grad_(True).float().view(1, num_classes).cuda(1)
        #print(inputs.size(), flush=True)
        times = 0
        while True:
          outputs = model_gen(inputs)
          optimizer_gen.zero_grad()
          loss = criterion_gen(outputs, labels)
          #x_axis.append(len(x_axis))
          #y_axis.append(loss.item())
          loss.backward()
          optimizer_gen.step()
          #print('Epoch_gen: {}, Time: {}, loss: {:.6}'.format(epoch, times + 1, loss.item()), flush=True)
          times += 1
          if times > 6:
            break

    #plt.clf()
    #plt.plot(x_axis, y_axis, 'go', label='Loss', alpha=0.5)
    #plt.legend(loc='best')
    #plt.savefig('gen_loss.png')

    print("Learning finished!")
  test_data = []
  path = '../selam/test_data_locus_0.9_generation_20.txt'
  file = open(path, 'r')
  for line in file:
    array = line.split('\t')
    test_data += [float(array[2])]
  file.close()
  test_data = torch.tensor(test_data).cuda(1)
  test_outputs = model_gen(test_data)
  print(test_outputs)

p1 = Process(target=SLS)
p1.start()
#p2 = Process(target=GEN)
#p2.start()
p1.join()
#p2.join()
