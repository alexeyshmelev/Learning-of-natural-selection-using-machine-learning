# Lab 12 RNN
import torch
import torch.nn as nn
from torch.autograd import Variable

#x_data = [5.6, 4.5, 7.8, 5.7, 3.4, 4.9, 6.2, 4.8, 5.1, 5.5, 19.2, 22.7, 17.7, 16.5, 25.1, 23.8, 19.9, 18.1, 23.3, 17.9]
#y_data = [1, 0]
#x_data = [[[i * 10 for i in x_data]]]

x_data = []
y_data = []

for i in [0, 1]:
  needed_line = ''
  file_line = 'frequency' + str(i) + '.txt'
  file = open(file_line, 'r')

  for line in file:
    array = line.split('\t')
    x_data += [float(array[2])]

  file.close()

  file_line = 'example' + str(i) + '_selection.txt'
  file = open(file_line, 'r')
  line = file.read()
  file.close()
  array = line.split('\t')
  y_data += [float(array[3])*10]

y_data = [0, 1]


# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.tensor(x_data, requires_grad=True))
inputs = inputs.view(1, 2, -1)
print(inputs)
labels = Variable(torch.LongTensor(y_data))
print(labels)




num_classes = 2
input_size = 20
hidden_size = 20 
batch_size = 1   
sequence_length = 20 
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self, num_layers, hidden_size):
        super(Model, self).__init__() #вызываем конструктор класса nn.Module через метод super(), здесь показан пример для версии Python 2.0
        self.num_layers = num_layers #количество скрытых слоёв RNN
        self.hidden_size = hidden_size #сколько скрытых состояний
        self.rnn = nn.RNN(input_size=20,
                          hidden_size=20, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(40, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), 40))
        h_0 = h_0.view(2, -1, 20)
        #print("h_0:",h_0)

        x_m = x.view(batch_size, -1, sequence_length)
        #print("x_m:",x_m)
        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(x_m, h_0)
        #print("out:", out.view(-1, 40))
        return self.fc(out.view(-1, 40))


# Instantiate RNN model
model = Model(num_layers, hidden_size)
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = model(inputs)
    #print(outputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
   
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))


print("Learning finished!")
print(model(inputs))
