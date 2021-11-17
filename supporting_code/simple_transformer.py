import torch.nn as nn
import torch.optim as optim
import torch
import math

class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    #self.prep = nn.Linear(6, 1)
    encoder_layers = nn.TransformerEncoderLayer(d_model=6, nhead=6, dim_feedforward=6000, dropout=0)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
    self.decoder = nn.Linear(6, 1)
    self.m = nn.Softmax(dim=1)

  def forward(self, input):
    #output = self.prep(input)
    output = self.transformer_encoder(input.view(10, 1, 6))
    output = self.decoder(output.view(10, 6)).view(1, 10)
    return output

class PositionalEncoding(nn.Module):
  def __init__(self, dropout=0):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(10, 6)
    #print(pe)
    position = torch.arange(0, 10, dtype=torch.float).unsqueeze(1)
    #print(position)
    div_term = torch.exp(torch.arange(0, 6, 1).float() * (-math.log(10000.0) / 6))
    #print(div_term)
    pe[:, :] = torch.sin(position * div_term)
    #pe[:, 1::2] = torch.cos(position * div_term)
    #print(pe)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)
    #print(self.pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    #print(x)
    return self.dropout(x)

def main():

  positional_encoding = PositionalEncoding()
  input = [[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [5, 5, 5, 5, 5, 5]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 6, 6, 6, 6, 6]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 8, 8, 8, 8]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [9, 9, 9, 9, 9, 9]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10]]]
  labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

  model = Attention()
  print(model)
  #model.train()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(10000):
    input_copy = input[epoch % 10]
    #input_copy = [[i / 10 for i in j] for j in input_copy]
    #print(input_copy)
    labels_copy = labels[epoch % 10]
    #input_copy = input[2]
    #labels_copy = labels[2]
    labels_copy = torch.tensor(labels_copy).type(dtype=torch.long)
    #input_copy = [[i] for i in input_copy]
    input_copy = torch.tensor(input_copy).type(dtype=torch.float)
    #print(input_copy.view(10, 1, 6))
    #print(labels_copy)
    output = model(positional_encoding(input_copy.view(10, 1, 6)))
    #print('ENCODER:', model.transformer_encoder.layers[0].self_attn.out_proj.weight.shape)
    #print(output)
    optimizer.zero_grad()
    loss = criterion(output, labels_copy)
    print('LOSS', loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    #print('----------PARAMETERS----------', list(model.parameters()))

  m = nn.Softmax(dim=1)
  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 6, 6, 6, 6, 6]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 8, 8, 8, 8]]]).type(dtype=torch.float)
  output = model(positional_encoding(input_copy.view(10, 1, 6)))
  print(m(output))

main()
