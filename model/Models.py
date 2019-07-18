import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length + 1
        self.hidden_size = hidden_size

        # transformation of the input to hidden
        self.linear_in = nn.Linear(14, hidden_size)
        # recurrent cell
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        # from a hidden state to the vocab
        self.linear_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden=None):
        """
        Performs a forward pass
        """
        batch_size = inputs.shape[0]

        x = self.linear_in(inputs)
        outputs = []

        h = torch.zeros([batch_size, self.hidden_size], device=device)
        c = torch.zeros([batch_size, self.hidden_size], device=device)

        state = (h, c)

        hidden_states = []
        for i in range(self.max_length):
            state = self.rnn(x, state)
            h, _ = state
            outputs.append(self.linear_out(h))
            hidden_states.append(h)

        outputs = F.softmax(torch.stack(outputs, dim=1), dim=2)
        return outputs, torch.stack(hidden_states, dim=1).reshape(batch_size, -1)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length + 1
        self.hidden_size = hidden_size

        # transformation of the input to hidden
        self.linear_in = nn.Linear(14, hidden_size)
        # recurrent cell
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        # from a hidden state to the vocab
        self.linear_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden=None):
        """
        Performs a forward pass
        """
        batch_size = inputs.shape[0]

        x = self.linear_in(inputs)
        outputs = []

        h = torch.zeros([batch_size, self.hidden_size], device=device)

        hidden_states = []
        for i in range(self.max_length):
            h = self.rnn(x, h)
            outputs.append(self.linear_out(h))
            hidden_states.append(h)

        outputs = F.softmax(torch.stack(outputs, dim=1), dim=2)
        return outputs, torch.stack(hidden_states, dim=1).reshape(batch_size, -1)
