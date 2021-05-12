import os
import sys
import time
import torch
import math
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

    def __init__(self, gen_num_classes):
      super(SLS, self).__init__() 
      encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes, nhead=1, dim_feedforward=10000, dropout=0) # dim_feedforward=2048 --- for generation, 
      self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
      self.decoder = nn.Linear(gen_num_classes, gen_num_classes)

    def forward(self, x):
      output = self.transformer_encoder(x)
      output = self.decoder(output).view(1, -1)
      return output

  data = FormData(train_number, num_classes)

  if settings.boot_from_file:

    model_sls = SLS(num_classes) #определяем класс с помощью уже натренированной сети
    model_sls.cuda(0)
    model_sls.load_state_dict(torch.load('transformer_loc.pth'))
    model_sls.eval()

  else:

    # Initialize model_sls, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_sls = SLS(num_classes)
    model_sls.cuda(0) #переместили модель на графический процессор прежде, чем объявлять torch.optim для неё, это правильно
    criterion_sls = torch.nn.CrossEntropyLoss()
    optimizer_sls = torch.optim.Adam(model_sls.parameters(), lr=0.001)
    print(model_sls)
    x_axis = []
    y_axis = []

    start_time = time.time()

    for epoch in range(100):
      print('Epoch_sls: {}, time spent: {} sec'.format(epoch, time.time() - start_time), flush=True)
      start_time = time.time()
      x_axis.append(len(x_axis))
      temp_loss = []
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        labels = batch[0][-1::].type(dtype=torch.long).cuda(0)
        inputs = batch[0][:-1:].clone().detach().requires_grad_(True).float().view(1, 1, num_classes).cuda(0)
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
      y_axis.append(sum(temp_loss)/9000)

    plt.clf()
    plt.plot(x_axis, y_axis, '.-g', label='Mean loss', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig('sls_loss.png')

    print("Learning finished!")

  torch.save(model_sls.state_dict(), 'transformer_loc.pth')

  m = nn.Softmax(dim=1)
  test_data = []
  path = '../selam/test_data_locus_0_9.txt'
  file = open(path, 'r')
  for line in file:
    array = line.split('\t')
    test_data += [float(array[2])]
  file.close()
  test_data = torch.tensor(test_data).cuda(0)
  test_outputs = model_sls(test_data.view(1, 1, num_classes))
  test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
  max = np.argmax(test_outputs)
  print('Natural selection was in locus {:.2} with the probability of {}'.format(max/100, test_outputs[max]))

def GEN(): ######################################################################################################################################################################################################################

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
          temp_data += [float(array[2]), float(array[3]), float(array[4]), float(array[5]), float(array[6])]
        temp_data.append(float(exact_file.split('_')[5])*100)
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

  class GEN(nn.Module):

    def __init__(self, gen_num_classes):
      super(GEN, self).__init__() 
      encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes*5, nhead=5, dim_feedforward=10000, dropout=0) # dim_feedforward=2048 --- for generation, 
      self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
      self.decoder = nn.Linear(gen_num_classes*5, 1)

    def forward(self, x):
      output = self.transformer_encoder(x)
      output = self.decoder(output).view(1, 1)
      return output


  data = FormData(train_number, num_classes)

  if settings.boot_from_file:
    model_gen = GEN(num_classes) #определяем класс с помощью уже натренированной сети
    model_gen.cuda(0)
    model_gen.load_state_dict(torch.load('transformer_add.pth'))
    model_gen.eval()

  else:

    # Initialize model_gen, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_gen = GEN(num_classes)
    model_gen.cuda(0)
    criterion_gen = torch.nn.MSELoss() 
    optimizer_gen = torch.optim.SGD(model_gen.parameters(), lr=0.001)
    print(model_gen)
    x_axis = []
    y_axis = []

    for epoch in range(2):
      print(epoch)
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        labels = batch[0][-1::].type(dtype=torch.float).view(1, 1).cuda(0)
        print(labels, flush=True)
        inputs = batch[0][:-1:].clone().detach().requires_grad_(True).float().view(1, 1, num_classes*5).cuda(0)
        #print(inputs, flush=True)
        times = 0
        while True:
          optimizer_gen.zero_grad()
          outputs = model_gen(inputs)
          loss = criterion_gen(outputs, labels)
          print(loss, flush=True)
          #x_axis.append(len(x_axis))
          #y_axis.append(loss.item())
          loss.backward()
          optimizer_gen.step()
          #print('Epoch_gen: {}, Time: {}, loss: {:.6}'.format(epoch, times + 1, loss.item()), flush=True)
          times += 1
          if times > 2:
            break

    torch.save(model_gen.state_dict(), 'transformer_add.pth')

    #plt.clf()
    #plt.plot(x_axis, y_axis, 'go', label='Loss', alpha=0.5)
    #plt.legend(loc='best')
    #plt.savefig('gen_loss.png')

    print("Learning finished!")

  test_data = []
  path = '../selam/test_data_gen_20.txt'
  file = open(path, 'r')
  for line in file:
    array = line.split('\t')
    test_data += [float(array[2]), float(array[3]), float(array[4]), float(array[5]), float(array[6])]
  file.close()
  test_data = torch.tensor(test_data).float().view(1, 1, num_classes*5).cuda(0)
  test_outputs = model_gen(test_data)
  print('TEST: ', test_outputs/100)

def INT():

  def FormData(train_number):

    data = []
    temp_data = []

    for i in range(train_number):
      path = '../selam/next_gen_simulation_sm'
      file_list = sorted(os.listdir(path))
      exact_file = path + '/' + file_list[i]
      file = open(exact_file, 'r')
      for line in file:
        array = line.split('\t')
        temp = [array[2], array[3], array[4], array[5], array[6]]
        temp = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp]
        temp[0] *= 10
        temp[1] *= 1000
        temp[2] *= 1000
        temp[3] *= 10000000
        temp[4] *= 1000000
        temp_data += temp
      temp_data.append(float(exact_file.split('_')[3].split('/')[1]))
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

  class PositionalEncoding(nn.Module):
    def __init__(self, dropout=0):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)

      pe = torch.zeros(100, 5)
      #print(pe)
      position = torch.arange(0, 100, dtype=torch.float).unsqueeze(1)
      #print(position)
      div_term = torch.exp(torch.arange(0, 5, 1).float() * (-math.log(10000.0) / 5))
      #print(div_term)
      pe[:, :] = torch.sin(position * div_term)
      #pe[:, 1::2] = torch.cos(position * div_term)
      #print(pe)
      pe = pe.unsqueeze(0).transpose(0, 1).cuda(0)
      self.register_buffer('pe', pe)
      print(self.pe.shape)

    def forward(self, x):
      x = x + self.pe[:x.size(0)]
      #print(x)
      return self.dropout(x)

  class INT(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=100000, dropout=0):
      super(INT, self).__init__() 
      self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
      self.linear1 = nn.Linear(d_model, dim_feedforward)
      self.dropout = nn.Dropout(dropout)
      self.linear2 = nn.Linear(dim_feedforward, d_model)
      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)
      self.activation = nn.ReLU()
      self.decoder_1 = nn.Linear(5, 20000)
      self.decoder_2 = nn.Linear(20000, 10000)
      self.decoder_3 = nn.Linear(10000, 5000)
      self.decoder_4 = nn.Linear(5000, 1000)
      self.decoder_5 = nn.Linear(10000, 1)
      #self.decoder_6 = nn.Linear(10, 1)
      self.tg = nn.Tanh()
      self.sm = nn.Softmax(dim=1)

    def forward(self, src, weights):
      src2, attn = self.self_attn(src, src, src)
      if weights == True:
        print('ATTENTION WEIGHTS', attn)
      src = src + self.dropout1(src2)
      src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
      src = src + self.dropout2(src2)
      output = self.decoder_1(src.view(1000, 5))
      output = self.decoder_2(output.view(1000, 20000))
      output = self.decoder_3(output.view(1000, 10000))
      output = self.decoder_4(output.view(1000, 5000))
      output = self.decoder_5(output.view(100, 10000)).view(1, 100)
      #output = self.tg(output)
      #output = self.decoder_6(output.view(1, 100)).view(1, 1)
      return output


  data = FormData(7000) #(train_number, num_classes)
  positional_encoding = PositionalEncoding()

  if settings.boot_from_file:
    model_int = INT() #определяем класс с помощью уже натренированной сети
    model_int.cuda(0)
    model_int.load_state_dict(torch.load('transformer_inf.pth'))
    model_int.eval()

  else:

    # Initialize model_int, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_int = INT(5, 5)
    model_int.cuda(0)
    optimizer_int = torch.optim.Adam(model_int.parameters(), lr=0.0001)
    criterion_int = torch.nn.MSELoss()
    print(model_int)
    x_axis = []
    y_axis = []
    needed_labels = [(i+1)*100 for i in range(9)]
    file_list = sorted(os.listdir('../selam/next_gen_simulation_test'))
    error = -1

    for epoch in range(200):
      temp_y_axis = []
      print(epoch)
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        label = batch[0][-1::]
        for j in needed_labels:
          if j == label.item():
            #labels = batch[0][-1::].type(dtype=torch.long).cuda(0)
            item = label[0] / 100
            labels = [item for i in range(100)]
            labels = torch.tensor(labels)
            labels = labels.type(dtype=torch.float).view(1, 100).cuda(0)
            #print(labels, flush=True)
            inputs = batch[0][:-1:].clone().detach().float().view(1000, 1, 5).cuda(0)
            #print('INPUTS', inputs, flush=True)
            times = 0
            while True:
              optimizer_int.zero_grad()
              outputs = model_int(inputs, False)
              loss = criterion_int(outputs, labels)
              #print(loss, flush=True)
              temp_y_axis.append(loss.item())
              loss.backward()
              #torch.nn.utils.clip_grad_norm_(model_int.parameters(), 0.5)
              optimizer_int.step()
              #print('Epoch_gen: {}, Time: {}, loss: {:.6}'.format(epoch, times + 1, loss.item()), flush=True)
              times += 1
              if times > 0:
                break

      x_axis.append(epoch)
      y_axis.append(np.sum(temp_y_axis) / 7000)
      plt.clf()
      plt.plot(x_axis, y_axis, 'go', label='Loss', alpha=0.5)
      plt.legend(loc='best')
      plt.savefig('int_loss.png')

      data_list_difference = []
      temp_data = []
      for step in range(len(os.listdir('../selam/next_gen_simulation_test'))):
        generation = float(file_list[step].split('_')[0])
        test_data = []
        exact_file = '../selam/next_gen_simulation_test' + '/' + file_list[step]
        file = open(exact_file, 'r')
        for line in file:
            array = line.split('\t')
            temp_data += [array[2], array[3], array[4], array[5], array[6]]
            temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
            temp_data[0] *= 10
            temp_data[1] *= 1000
            temp_data[2] *= 1000
            temp_data[3] *= 10000000
            temp_data[4] *= 1000000
            test_data += temp_data
            temp_data = []
        file.close()
        test_data = torch.tensor(test_data).float().view(1000, 1, 5).cuda(0)
        test_outputs = model_int(test_data, False)
        test_outputs = test_outputs.cpu().detach().numpy().flatten()
        test_outputs = np.sum(test_outputs)
        data_list_difference.append(abs(test_outputs-generation))

      plt.clf()
      plt.hist(data_list_difference)
      plt.axvline(x=np.mean(data_list_difference))
      plt.savefig('difference_' + str(epoch) + '.png')

      if error == -1:
        error = np.sum(data_list_difference)
      else:
        if np.sum(data_list_difference) < error:
          print('New minimum - ' + str(epoch), flush=True)
          torch.save(model_int.state_dict(), 'transformer_inf.pth')
          error = np.sum(data_list_difference)

    print("Learning finished!")

p1 = Process(target=INT)
p1.start()
#p2 = Process(target=GEN)
#p2.start()
p1.join()
#p2.join()
