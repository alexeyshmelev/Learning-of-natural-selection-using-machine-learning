import torch.nn as nn
import torch.optim as optim
import torch

class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    self.linears_1 = nn.ModuleList([nn.Linear(6, 6) for i in range(10)])
    self.linears_2 = nn.ModuleList([nn.Linear(6, 1) for i in range(10)])
    self.softmax = nn.Softmax(dim=1)
    self.hidden = nn.Linear(60, 120)
    self.out = nn.Linear(120, 10)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  def forward(self, input):
    for i in range(10):
      if i == 0:
        inputs_1 = self.tanh(self.linears_1[i](input[i].view(1, 6)))
      else:
        inputs_1 = torch.cat((inputs_1, self.tanh(self.linears_1[i](input[i].view(1, 6)))), 0)
    for i in range(10):
      if i == 0:
        inputs_2 = self.linears_2[i](inputs_1[i].view(1, 6))
      else:
        inputs_2 = torch.cat((inputs_2, self.linears_2[i](inputs_1[i].view(1, 6))), 1)
    #print(inputs_2)
    att_weights = self.softmax(inputs_2).view(10, 1)
    print(att_weights)
    multiplication = torch.mul(att_weights, input).view(1, -1)
    #print(multiplication)
    #sum = torch.sum(multiplication, 0).view(1, 6)
    #print(sum)
    output = self.hidden(multiplication)
    output = self.sigmoid(self.out(output))
    #print(output)
    return output


def main():

  input = [[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [5, 5, 5, 5, 5, 5]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 6, 6, 6, 6, 6]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 8, 8, 8, 8]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [9, 9, 9, 9, 9, 9]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10]]]
  labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

  model = Attention()
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
    input_copy = torch.tensor(input_copy).type(dtype=torch.float).view(10, 6)
    #print(input_copy)
    #print(labels_copy)
    output = model(input_copy)
    #print(output)
    optimizer.zero_grad()
    loss = criterion(output, labels_copy)
    print('LOSS', loss)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    optimizer.step()
    #print('----------PARAMETERS----------', list(model.parameters()))

  #input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]]).type(dtype=torch.float).view(10, 1)

main()
