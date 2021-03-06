import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ILTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, batch, targets):
        batch_size, seq_length = targets.shape[0], targets.shape[1]

        batch = batch.to(device)
        targets = targets.to(device)

        predictions, hidden_states = self.model(batch)
        loss = self.criterion(
            predictions.reshape(-1, predictions.shape[2]), targets.reshape(-1)
        )

        # Calculate accuracy
        sequences = torch.argmax(predictions, dim=2)
        accuracy = torch.sum(sequences == targets).type(torch.float) / (
            batch_size * seq_length
        )

        return loss, torch.mean(accuracy), sequences, hidden_states
