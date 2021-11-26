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

def SLS(): #пытались определять позицию под отбором с помощью трансформера, но точность была очень маленькая, поэтому данная функция пока что не используется, сейчас применяется старая архитектура с использованием RNN 

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
      encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes, nhead=1, dim_feedforward=10000, dropout=0)
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

  gen_int = [(i+1)*100 for i in range(10)]

  adm_start = 0.0002
  adm_end = 0.01
  l = np.log(adm_end) - np.log(adm_start)
  adm_int = [np.exp(l*(i+1)/10 + np.log(adm_start)) for i in range(10)]

  frc_start = 0.0005
  frc_end = 0.02
  l = np.log(frc_end) - np.log(frc_start)
  frc_int = [np.exp(l*(i+1)/10 + np.log(frc_start)) for i in range(10)]

  def FormData(train_number):

    data = []
    temp_data = []

    for i in range(train_number):
      path = '../selam/next_gen_simulation_final'
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
      temp_data.append(float(exact_file.split('_')[4]))
      temp_data.append(float(exact_file.split('_')[5]))
      data.append(temp_data)
      temp_data = []
      file.close()

    return data


  def FormTarget(gen, adm, frc):

    gen_array = np.full(10, 0)
    adm_array = np.full(10, 0)
    frc_array = np.full(10, 0)

    adm_num = 0
    frc_num = 0

    for k, elem in enumerate(gen_int):
      if gen == elem:
        gen_num = k

    for k, elem in enumerate(adm_int):
      if adm > elem:
        adm_num = k + 1

    for k, elem in enumerate(frc_int):
      if frc > elem:
        frc_num = k + 1

    gen_array[gen_num] = 1
    adm_array[adm_num] = 1
    frc_array[frc_num] = 1
    array = torch.tensor(np.concatenate([gen_array, adm_array, frc_array])).view(1, 30).type(dtype=torch.float).cuda(0)
    # array = torch.argmax(array).view(1).type(dtype=torch.float).cuda(0)

    return array



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

    def __init__(self):
      super(INT, self).__init__()

      # bottom
      self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (1000, 5)
      self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (1000, 5)
      self.mp1 = nn.MaxPool2d(kernel_size=(5, 1), padding=0) # (200, 5)

      self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (200, 5)
      self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (200, 5)
      self.mp2 = nn.MaxPool2d(kernel_size=(5, 1), padding=0) # (40, 5)

      self.conv5 = nn.Conv2d(64, 128, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (40, 5)
      self.conv6 = nn.Conv2d(128, 128, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (40, 5)
      self.mp3 = nn.MaxPool2d(kernel_size=(5, 1), padding=0) # (8, 5)

      # bottleneck
      self.conv7 = nn.Conv2d(128, 256, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (8, 5)
      self.conv8 = nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (8, 5)

      # top
      self.up = nn.Upsample(scale_factor=(5, 1), mode='nearest')

      self.conv9 = nn.Conv2d(384, 128, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (40, 5)
      self.conv10 = nn.Conv2d(128, 128, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (40, 5)

      self.conv11 = nn.Conv2d(192, 64, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (200, 5)
      self.conv12 = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (200, 5)

      self.conv13 = nn.Conv2d(96, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (1000, 5)
      self.conv14 = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (1000, 5)

      # final
      self.conv15 = nn.Conv2d(32, 1, kernel_size=(5, 1), stride=1, padding=(2, 0)) # (1000, 5)

      self.dropout = nn.Dropout()
      self.fc1 = nn.Linear(5000, 2500)
      self.fc2 = nn.Linear(2500, 1250)
      self.fc3 = nn.Linear(1250, 625)
      self.fc4 = nn.Linear(625, 30)
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
      self.sm = nn.Softmax(dim=1)

    def forward(self, src):
      # bottom
      c1 = self.relu(self.conv1(src))
      c1 = self.relu(self.conv2(c1))
      p1 = self.mp1(c1)
      c2 = self.relu(self.conv3(p1))
      c2 = self.relu(self.conv4(c2))
      p2 = self.mp2(c2)
      c3 = self.relu(self.conv5(p2))
      c3 = self.relu(self.conv6(c3))
      p3 = self.mp3(c3)

      # bottleneck
      bn = self.relu(self.conv7(p3))
      bn = self.relu(self.conv8(bn))

      # top
      us = self.up(bn)
      concat = torch.cat((us, c3), 1)
      u1 = self.relu(self.conv9(concat))
      u1 = self.relu(self.conv10(u1))

      us = self.up(u1)
      concat = torch.cat((us, c2), 1)
      u2 = self.relu(self.conv11(concat))
      u2 = self.relu(self.conv12(u2))

      us = self.up(u2)
      concat = torch.cat((us, c1), 1)
      u3 = self.relu(self.conv13(concat))
      u3 = self.relu(self.conv14(u3))

      # final
      final = self.sigmoid(self.conv15(u3))

      output = self.dropout(final)

      output = self.fc1(output.view(1, 5000))
      output = self.fc2(output)
      output = self.fc3(output)
      output = self.fc4(output)
      
      return self.sm(output.view(3, 10)).view(1, 30)


  data = FormData(11128) #(train_number, num_classes)
  positional_encoding = PositionalEncoding()

  if settings.boot_from_file:
    model_int = INT() #определяем класс с помощью уже натренированной сети
    model_int.cuda(0)
    model_int.load_state_dict(torch.load('transformer_inf.pth'))
    model_int.eval()

  else:

    # Initialize model_int, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_int = INT()
    model_int.cuda(0)
    optimizer_int = torch.optim.Adam(model_int.parameters(), lr=0.001)
    criterion_int = torch.nn.BCELoss()
    print(model_int)
    x_axis = []
    y_axis = []
    # needed_labels = [(i+1)*100 for i in range(9)]
    # file_list = sorted(os.listdir('../selam/next_gen_simulation_test'))
    # error = -1

    for epoch in range(100):
      temp_y_axis = []
      print(epoch)
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        gen = batch[0][-3:-2:]
        adm = batch[0][-2:-1:]
        frc = batch[0][-1::]
        labels = FormTarget(gen.item(), adm.item(), frc.item())
        #print(labels, flush=True)
        # inputs = batch[0][:-3:].clone().detach().float().view(1000, 1, 5).cuda(0) #for transformer
        inputs = batch[0][:-3:].clone().detach().float().view(1, 1, 1000, 5).cuda(0) #for transformer
        #print('INPUTS', inputs, flush=True)
        times = 0
        while True:
          optimizer_int.zero_grad()
          outputs = model_int(inputs)
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
      y_axis.append(np.sum(temp_y_axis) / 11128)
      print(np.sum(temp_y_axis) / 11128, flush=True)
      plt.clf()
      plt.plot(x_axis, y_axis, '.-g', label='Loss', alpha=0.5)
      plt.legend(loc='best')
      plt.savefig('int_loss.png')
      if epoch % 10 == 0:
        torch.save(model_int.state_dict(), 'transformer_inf_epoch_' + str(epoch) + '.pth')
        torch.save(optimizer_int.state_dict(), 'transformer_inf_criterion_epoch_' + str(epoch) + '.pth')

      # data_list_difference = []
      # temp_data = []
      # for step in range(len(os.listdir('../selam/next_gen_simulation_test'))):
      #   generation = float(file_list[step].split('_')[0])
      #   test_data = []
      #   exact_file = '../selam/next_gen_simulation_test' + '/' + file_list[step]
      #   file = open(exact_file, 'r')
      #   for line in file:
      #       array = line.split('\t')
      #       temp_data += [array[2], array[3], array[4], array[5], array[6]]
      #       temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
      #       temp_data[0] *= 10
      #       temp_data[1] *= 1000
      #       temp_data[2] *= 1000
      #       temp_data[3] *= 10000000
      #       temp_data[4] *= 1000000
      #       test_data += temp_data
      #       temp_data = []
      #   file.close()
      #   test_data = torch.tensor(test_data).float().view(1000, 1, 5).cuda(0)
      #   test_outputs = model_int(test_data, False)
      #   test_outputs = test_outputs.cpu().detach().numpy().flatten()
      #   test_outputs = np.sum(test_outputs)
      #   data_list_difference.append(abs(test_outputs-generation))

      # plt.clf()
      # plt.hist(data_list_difference)
      # plt.axvline(x=np.mean(data_list_difference))
      # plt.savefig('difference_' + str(epoch) + '.png')

      # if error == -1:
      #   error = np.sum(data_list_difference)
      # else:
      #   if np.sum(data_list_difference) < error:
      #     print('New minimum - ' + str(epoch), flush=True)
      #     torch.save(model_int.state_dict(), 'transformer_inf.pth')
      #     torch.save(optimizer_int.state_dict(), 'transformer_inf_criterion.pth')
      #     error = np.sum(data_list_difference)

    print("Learning finished!")

p1 = Process(target=INT)
p1.start()
#p2 = Process(target=GEN)
#p2.start()
p1.join()
#p2.join()