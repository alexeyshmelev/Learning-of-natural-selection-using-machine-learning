import torch.nn as nn
import torch.optim as optim
import torch

class Attention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward=10000, dropout=0):
    super(Attention, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.activation = nn.ReLU()

    self.decoder = nn.Linear(60, 10)
    self.m = nn.Softmax(dim=1)

  def forward(self, src):
    src2, attn = self.self_attn(src, src, src)
    #print(attn)
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    output = self.decoder(src).view(1, 10)
    return output


def main():

  input = [[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [5, 5, 5, 5, 5, 5]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 6, 6, 6, 6, 6]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 8, 8, 8, 8]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [9, 9, 9, 9, 9, 9]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10]]]
  labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

  model = Attention(60, 10)
  print(model)
  #model.train()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(3000):
    input_copy = input[epoch % 10]
    labels_copy = labels[epoch % 10]
    #input_copy = input[2]
    #labels_copy = labels[2]
    labels_copy = torch.tensor(labels_copy).type(dtype=torch.long)
    #input_copy = [[i] for i in input_copy]
    input_copy = torch.tensor(input_copy).type(dtype=torch.float)
    #print(input_copy.shape)
    #print(labels_copy)
    output = model(input_copy.view(1, 1, 60))
    #print('ENCODER:', model.transformer_encoder.layers[0].self_attn.out_proj.weight.shape)
    #print(output)
    optimizer.zero_grad()
    loss = criterion(output, labels_copy)
    print('LOSS', loss)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    #print('----------PARAMETERS----------', list(model.parameters()))

  m = nn.Softmax(dim=1)
  input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]]).type(dtype=torch.float)
  output = model(input_copy.view(1, 1, 60))
  print(m(output))

main()
