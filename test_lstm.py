import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, 1, bidirectional=True, batch_first=True)
input = torch.randn(3, 5, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input)
# output, (hn, cn) = rnn(input, (h0, c0))
print(output.size())
print(hn.view(1, 2, 3, 20)[-1])
# print(hn[-2:])
# print(cn[-1])