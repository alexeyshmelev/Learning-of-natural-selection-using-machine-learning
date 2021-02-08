import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(1, 1, 1, batch_first=True)

  def forward(self, input):
    output, hidden_state = self.lstm(input)
    #print('DATA:', self.lstm.weight_hh_l0[0])
    return output

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(10, 1, 1, batch_first=True)
    self.linear = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
    self.softmax = nn.Softmax(dim=0)

  def forward(self, input):
    input = input.view(1, 10)
    initial_input = input.clone()
    for i in range(10):
      if i == 0:
        input = self.linear[i](input)
      else:
        input = torch.cat((input, self.linear[i](initial_input)), 0)
    input = input.view(-1, 10, 10)
    output, hidden_state = self.lstm(input)
    output = output.view(1, 10)
    return output

class RNN():
  def __init__(self):
    super(RNN, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()
    print(self.encoder)
    print(self.decoder)

    self.loss = nn.CrossEntropyLoss()
    self.encoder_optimizer = optim.Adam(self.encoder.parameters())
    self.decoder_optimizer = optim.Adam(self.decoder.parameters())


  def train(self, input, target, test):

    # Encoder
    output_encoder = self.encoder.forward(input)

    # Decoder
    output_decoder = self.decoder.forward(output_encoder)

    if test:
      print(np.argmax(torch.nn.functional.softmax(output_decoder).detach().numpy().flatten()))

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    loss = self.loss(output_decoder, target)
    print(loss)
    
    loss.backward()

    self.encoder_optimizer.step()
    self.decoder_optimizer.step()
    

    return 0

def main():
  rnn = RNN()

  input = [[10, 1, 1, 1, 1, 1, 1, 1, 1, 1], [10, 1, 1, 1, 1, 1, 1, 1, 1, 2], [10, 1, 1, 1, 1, 1, 1, 1, 1, 3], [10, 1, 1, 1, 1, 1, 1, 1, 1, 4], [10, 1, 1, 1, 1, 1, 1, 1, 1, 5], [10, 1, 1, 1, 1, 1, 1, 1, 1, 6], [10, 1, 1, 1, 1, 1, 1, 1, 1, 7], [10, 1, 1, 1, 1, 1, 1, 1, 1, 8], [10, 1, 1, 1, 1, 1, 1, 1, 1, 9], [10, 1, 1, 1, 1, 1, 1, 1, 1, 10]]
  labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  
  labels = torch.tensor(labels).type(dtype=torch.long)

  losses = []
  for epoch in range(100):
    input_copy = input[epoch % 10]
    labels_copy = [labels[epoch % 10]]
    labels_copy = torch.tensor(labels_copy).type(dtype=torch.long)
    input_copy = torch.tensor(input_copy).float().requires_grad_(True).view(1, 10, 1)
    rnn.train(input_copy, labels_copy, False)
  
  rnn.train(input_copy, labels_copy, True)

main()
