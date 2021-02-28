import torch.nn as nn
import torch.optim as optim
import torch

class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    self.key = nn.Linear(6, 6)
    self.query = nn.Linear(6, 6)
    self.value = nn.Linear(6, 6)
    self.out = nn.Linear(6, 1)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, input):
    key = self.key(input)
    #print('KEY:', self.key.weight)
    print('KEY: ', key)
    query = self.query(input)
    print('QUERY: ', query)
    value = self.value(input)
    print('VALUE: ', value)
    for i in range(10):
      print('NEW QUERY: ', query[i].view(1, 1, -1))
      print('TRANSPOSED KEY: ', torch.transpose(key, 0, 1))
      att_score = torch.bmm(query[i].view(1, 1, -1), torch.transpose(key, 0, 1).view(1, 6, 10))
      print('ATTENTION SCORES', att_score)
      att_score = self.softmax(att_score)
      print('ATTENTION SCORES', att_score)
      multiplication = torch.mul(att_score.view(1, 10, 1), value)
      print('ATENTION DATA:', att_score.view(1, 10, 1))
      print('VALUE DATA:', value)
      print('MULTIPLICATION:', multiplication)
      if i == 0:
        output = torch.sum(multiplication, 1)
      else:
        output = torch.cat((output, torch.sum(multiplication, 1)), 0)
      #print('FINAL OUTPUT:', output)
    #print(output)
    #print(torch.sum(output, 1))
    output = torch.sum(output, 1).view(1, -1)
    #output = self.out(output).view(1, 10)
    print('FINAL OUTPUT:', output)
    #print('OUT:', self.out.weight)
    return output


def main():

  input = [[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [5, 5, 5, 5, 5, 5]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 6, 6, 6, 6, 6]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [8, 8, 8, 8, 8, 8]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [9, 9, 9, 9, 9, 9]], [[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10]]]
  labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

  model = Attention()
  print(model)
  optimizer = optim.SGD(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(1):
    input_copy = input[epoch % 10]
    labels_copy = labels[epoch % 10]
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
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
    optimizer.step()
    #print('----------PARAMETERS----------', list(model.parameters()))

  #input_copy = torch.tensor([[[10, 10, 10, 10, 10, 10], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1,], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]]).type(dtype=torch.float).view(10, 1)

main()
