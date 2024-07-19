import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

# Define input parameters
input_size = 10   # Input feature dimension
hidden_size = 20  # Hidden layer dimension
output_size = 5   # Output feature dimension
num_layers = 1    # Number of LSTM layers

# Create a model instance
model = BiRNN(input_size, hidden_size, output_size, num_layers)

# Print the model
print(model)

# Test the model with dummy data
x = torch.randn(32, 15, input_size)  # Batch size of 32, sequence length of 15
output = model(x)
print(output.shape)  # Expected output: [32, 15, output_size]