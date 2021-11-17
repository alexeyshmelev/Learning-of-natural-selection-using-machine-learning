import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #print('OUR INPUT: ',input.view(1, 1, 1))
        output, hidden = self.gru(input.view(1, 1, 1), hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
 


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.attn = nn.Linear(self.hidden_size + 1, 10) #self.hidden_size+1
        self.attn_combine = nn.Linear(self.hidden_size + 1, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print('DECODER INPUT1: ', input)
        #print('DECODER HIDDEN1: ', hidden)
        input = input.view(1, 1, -1)
        #print('DECODER INPUT2: ', input)
        #input = self.dropout(input)
        #print('DECODER INPUT3: ', input)
        #print('CAT: ',torch.cat((input[0], hidden[0]), 1))

        attn_weights = F.softmax(self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        #print('ATTENTION WEIGHTS: ', attn_weights)
        #print('ENCODER OUTPUTS: ', encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        #print('ATTENTION APPLIED: ', attn_applied)


        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        #output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        #output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, test):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = 10
    target_length = 10
    #print('input_length', input_length)
    #print('target_length', target_length)
    #print('input_tensor', input_tensor)
    #print('Encoder', encoder)

    encoder_outputs = torch.zeros(10, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #print('encoder_outputs', encoder_outputs.size())
    decoder_input = torch.tensor([[0] if x != target_tensor[0] else [1] for x in range(10)]).type(dtype=torch.float)
    #print(decoder_input)
    decoder_output = torch.tensor([[0]]).type(dtype=torch.float)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True

    if use_teacher_forcing and not test:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
        #for di in range(1):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_output[0], decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], decoder_input[di])
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item() / target_length

    if test:
      result = []
      for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_output[0], decoder_hidden, encoder_outputs)
        result += decoder_output
      print('Result: ', result)
      return 0
    

def main():

  input = [[10, 1, 1, 1, 1, 1, 1, 1, 1, 1], [10, 1, 1, 1, 1, 1, 1, 1, 1, 2], [10, 1, 1, 1, 1, 1, 1, 1, 1, 3], [10, 1, 1, 1, 1, 1, 1, 1, 1, 4], [10, 1, 1, 1, 1, 1, 1, 1, 1, 5], [10, 1, 1, 1, 1, 1, 1, 1, 1, 6], [10, 1, 1, 1, 1, 1, 1, 1, 1, 7], [10, 1, 1, 1, 1, 1, 1, 1, 1, 8], [10, 1, 1, 1, 1, 1, 1, 1, 1, 9], [10, 1, 1, 1, 1, 1, 1, 1, 1, 10]]
  labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

  encoder = EncoderRNN(1, 10)
  decoder = AttnDecoderRNN(10, 1, dropout_p=0.1)
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.006)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.006)
  criterion = nn.MSELoss()
  
  #labels = torch.tensor(labels).type(dtype=torch.long)

  losses = []
  for epoch in range(1000):
    input_copy = input[epoch % 10]
    labels_copy = labels[epoch % 10]
    labels_copy = torch.tensor(labels_copy).type(dtype=torch.long)
    #print(labels_copy)
    input_copy = torch.tensor(input_copy).type(dtype=torch.float).view(10, 1)
    loss = train(input_copy, labels_copy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, False)
    print('LOSS: ', loss)

  input_copy = torch.tensor([10, 1, 1, 1, 1, 1, 1, 1, 1, 1]).type(dtype=torch.float).view(10, 1)
  loss = train(input_copy, labels_copy, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, True)

main()
