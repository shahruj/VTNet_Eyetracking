import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class ETClassifier(nn.Module):
    def __init__(self, gru_input_size, gru_hidden_size, cnn_input_channels, cnn_output_size, classification_hidden_size):
        super(ETClassifier, self).__init__()
        
        # GRU Sub-model
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, batch_first=True)
        
        # CNN Sub-model
        self.conv1 = nn.Conv2d(in_channels=cnn_input_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1_cnn = nn.Linear(32 * 7 * 7, cnn_output_size)  # Assuming input size to CNN is 28x28

        # Classification Network
        self.fc1 = nn.Linear(gru_hidden_size + cnn_output_size, classification_hidden_size)
        self.fc2 = nn.Linear(classification_hidden_size, 1) 
        
    def forward(self, x_gru, x_cnn):
        # GRU forward pass
        _, gru_hidden = self.gru(x_gru)
        gru_out = gru_hidden.squeeze(0)  # Removing the num_layers dimension
        
        # CNN forward pass
        x = F.relu(self.conv1(x_cnn))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)  # Flatten the output for the fully connected layer
        cnn_out = F.relu(self.fc1_cnn(x))
        
        # Combine GRU and CNN outputs
        combined_features = torch.cat((gru_out, cnn_out), dim=1)
        
        # Classification forward pass
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x
