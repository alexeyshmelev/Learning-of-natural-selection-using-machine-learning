import numpy as np
import random

train_data={
'5.6 4.5 7.8 5.7 3.4 4.9 6.2 4.8 5.1 5.5': 'october',
'19.2 22.7 17.7 16.5 25.1 23.8 19.9 18.1 23.3 17.9': 'july',
'5.8 4.9 3.3 5.5 5.0 7.8 6.2 6.1 4.4 4.9': 'october',
'25.4 26.2 19.2 18.5 16.3 18.1 21.9 20.1 22.7 19.5' : 'july',
}

test_data = {
    '7.2 8.4 5.3 5.6 4.8 4.9 5.9 4.4 5.1 5.8' : 'october',
    #'-1000 -1000 -1000 -1000 -1000 -1000 -1000 -1000 -1000 -1000' : 'july',
}

class RNN:
  
  def __init__(self, input_size, output_size, hidden_size=32):
    # конструктор класса, создаём матрицы коэффициентов
    self.Whh = np.random.randn()
    self.Wxh = np.random.randn()
    self.Why = np.random.randn(output_size, 1)

    # задаём смещения
    self.bh = 0
    self.by = 0

  def forward(self, inputs): # пишем фазу прямого распространения
    
    h = 0 # массив из ноликов

    self.last_inputs = inputs # сохраняем исходный данные для BPTT
    self.last_hs = { 0: h } # заносим исходные данные в словарь

    # пошла фаза прямого распространения
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh * float(x) + self.Whh * h + self.bh)
      self.last_hs[i + 1] = h

    # считаем выход
    y = self.Why * h + self.by

    return y, h

  def backprop(self, d_y, learn_rate=2e-2): # функция для обратного распространения
    
    n = len(self.last_inputs)

    # считаем частные производные dL/dWhy и dL/dby.
    d_Why = d_y * self.last_hs[n]
    d_by = d_y

    # деаем dL/dWhh, dL/dWxh, и dL/dbh нулями
    d_Whh = 0
    d_Wxh = 0
    d_bh = 0

    # считаем dL/dh для последнего h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why * d_y

    # BPTT
    for t in reversed(range(n)):
      # An intermediate value: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp * self.last_hs[t]

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp * float(self.last_inputs[t])

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh * temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Update weights and biases using gradient descent.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by




def softmax(xs):
  # считаем Softmax функцию
  return np.exp(xs) / sum(np.exp(xs))



def createInputs(text):
 
  inputs = text.split(' ')
  return inputs


def processData(data, backprop=True):
  
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    
    inputs = createInputs(x)
    if y == 'october':
      target=0
    if y == 'july':
      target=1

    
    # фаза прямого распространения
    out, _ = RNN.forward(rnn, inputs)
    probs = softmax(out)

    

    if backprop:

      # Calculate loss / accuracy
      loss -= np.log(probs[target])
      num_correct += int(np.argmax(probs) == target)

      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      RNN.backprop(rnn, d_L_d_y)

  

    if not backprop:
      if probs[0] > probs[1]:
        print('This is {} with a {} probability'.format('october', max(probs)[0]))
      if probs[0] < probs[1]:
        print('This is {} with a {} probability'.format('july', max(probs)[0]))

  return loss / len(data), num_correct / len(data)


rnn = RNN(10, 2)

for epoch in range(1000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

processData(test_data, backprop=False)
