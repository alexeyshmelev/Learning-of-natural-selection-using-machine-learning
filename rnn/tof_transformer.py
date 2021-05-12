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

num_classes = settings.num_classes
input_size = settings.input_size
hidden_size = settings.hidden_size
batch_size = settings.batch_size  
sequence_length = settings.sequence_length
num_layers = settings.num_layers
train_number = settings.train_number
dim = settings.dim
line_format = '%.' + str(dim) + 'f'

def TOF():

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

  class TOF(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=60000, dropout=0):
      super(TOF, self).__init__() 
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
      self.decoder_5 = nn.Linear(1000, 1)
      self.decoder_6 = nn.Linear(1000, 2)
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
      output = self.decoder_5(output.view(1000, 1000)).view(1, 1000)
      #output = self.tg(output)
      output = self.decoder_6(output.view(1, 1000)).view(1, 2)
      return output


  data = FormData(7000) #(train_number, num_classes)

  if settings.boot_from_file:
    model_tof = TOF() #определяем класс с помощью уже натренированной сети
    model_tof.cuda(0)
    model_tof.load_state_dict(torch.load('transformer_tof.pth'))
    model_tof.eval()

  else:

    # Initialize model_tof, set loss and optimizer function, CrossEntropyLoss = LogSoftmax + NLLLoss
    model_tof = TOF(5, 5)
    model_tof.cuda(0)
    optimizer_tof = torch.optim.Adam(model_tof.parameters(), lr=0.0001)
    criterion_tof = torch.nn.CrossEntropyLoss()
    print(model_tof)
    x_axis = []
    y_axis = []
    total_list_accuracy = []
    #needed_labels = [(i+1)*100 for i in range(9)]
    file_list = sorted(os.listdir('../selam/next_gen_simulation_test'))
    accuracy = -1
    m = nn.Softmax(dim=1)

    for epoch in range(200):
      temp_y_axis = []
      print(epoch)
      dataset = DataLoader(data, batch_size=settings.batch_size, shuffle=True, collate_fn=SampleDataset)
      for i, batch in enumerate(dataset): #тренируем сеть
        label = batch[0][-1::]
        if label.item() <= 500:
          labels = [0]
          labels = torch.tensor(labels)
          labels = labels.type(dtype=torch.long).cuda(0)
          inputs = batch[0][:-1:].clone().detach().float().view(1000, 1, 5).cuda(0)
          times = 0
        if label.item() > 500:
          labels = [1]
          labels = torch.tensor(labels)
          labels = labels.type(dtype=torch.long).cuda(0)
          inputs = batch[0][:-1:].clone().detach().float().view(1000, 1, 5).cuda(0)
          times = 0
        while True:
          optimizer_tof.zero_grad()
          outputs = model_tof(inputs, False)
          loss = criterion_tof(outputs, labels)
          # print(loss, flush=True)
          temp_y_axis.append(loss.item())
          loss.backward()
          #torch.nn.utils.clip_grad_norm_(model_tof.parameters(), 0.5)
          optimizer_tof.step()
          #print('Epoch_gen: {}, Time: {}, loss: {:.6}'.format(epoch, times + 1, loss.item()), flush=True)
          times += 1
          if times > 0:
            break

      x_axis.append(epoch)
      y_axis.append(np.sum(temp_y_axis) / 7000)
      plt.clf()
      plt.plot(x_axis, y_axis, '.-g', label='Loss', alpha=0.5)
      plt.legend(loc='best')
      plt.savefig('tof_loss.png')

      temp_list_accuracy = []
      temp_data = []
      for step in range(len(os.listdir('../selam/next_gen_simulation_test'))):
        test_data = []
        generation = float(file_list[step].split('_')[0])
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
        test_outputs = m(model_tof(test_data, False))
        test_outputs = test_outputs.cpu().detach().numpy().flatten()
        test_outputs = np.argmax(test_outputs)
        if test_outputs == 0 and generation <= 500:
          temp_list_accuracy.append(1)
        if test_outputs == 1 and generation > 500:
          temp_list_accuracy.append(1)

      total_list_accuracy.append(np.sum(temp_list_accuracy))

      plt.clf()
      plt.plot(x_axis, total_list_accuracy, '.-r', label='Accuracy', alpha=0.5)
      plt.legend(loc='best')
      plt.savefig('accuracy_' + str(epoch) + '.png')

      if accuracy == -1:
        accuracy = np.sum(temp_list_accuracy)
      else:
        if np.sum(temp_list_accuracy) >= accuracy:
          print('New accuracy in epoch ' + str(epoch), flush=True)
          torch.save(model_tof.state_dict(), 'transformer_tof.pth')
          accuracy = np.sum(temp_list_accuracy)

    print("Learning finished!")

  # m = nn.Softmax(dim=1)
  # test_data = []
  # temp_data = []
  # path = '../selam/next_gen_simulation/100_2.txt'
  # file = open(path, 'r')
  # for line in file:
  #   array = line.split('\t')
  #   temp_data += [array[2], array[3], array[4], array[5], array[6]]
  #   temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
  #   temp_data[0] *= 10
  #   temp_data[1] *= 1000
  #   temp_data[2] *= 1000
  #   temp_data[3] *= 1000000
  #   temp_data[4] *= 10000000
  #   test_data += temp_data
  #   temp_data = []
  # file.close()
  # test_data = torch.tensor(test_data).float().view(100, 1, 5).cuda(0)
  # test_outputs = model_tof(test_data, True)
  # print('TEST: ', test_outputs)

p1 = Process(target=TOF)
p1.start()
p1.join()
