import torch
import torch.nn as nn


# after GRU acc and gyro are concatenated together and feed to another GRU layer
class AG_DoorINet(nn.Module):
    def __init__(self, drop_out=0.2, model_name='AG-DoorINet'):
        super(AG_DoorINet, self).__init__()
        self.drop_out = drop_out
        self.gyro_only = False
        self.model_name = model_name
        self.gru1 = nn.GRU(input_size=3, hidden_size=64, bidirectional=True, batch_first=True, num_layers=3)
        self.gru2 = nn.GRU(input_size=3, hidden_size=64, bidirectional=True, batch_first=True, num_layers=3)
        self.gru3 = nn.GRU(input_size=256, hidden_size=256, bidirectional=True, batch_first=True, num_layers=2)
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
            nn.Linear(20*512, 5*512),
            nn.Dropout(p=self.drop_out),
            nn.Tanh(),
            nn.Linear(5*512,512),
            nn.Dropout(p=self.drop_out),
            nn.Tanh(),
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,4),
            nn.Tanh(),
            nn.Linear(4,1))

    def forward(self, x1, x2):
        x1, _ = self.gru1(x1.swapaxes(1,2))
        x2, _ = self.gru2(x2.swapaxes(1,2))
        x = torch.cat((x1,x2), axis=2)
        x, _ = self.gru3(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x


# after GRU acc and gyro are concatenated together and feed to another GRU layer
class G_DoorINet(nn.Module):
    def __init__(self, drop_out=0.2, model_name='G-DoorINet'):
        super(G_DoorINet, self).__init__()
        self.drop_out = drop_out
        self.gyro_only = True
        self.model_name = model_name
        self.short_name = short_name
        self.gru1 = nn.GRU(input_size=3, hidden_size=64, bidirectional=True, batch_first=True)
        self.gru3 = nn.GRU(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
            nn.Linear(10*512, 5*512),
            nn.Dropout(p=self.drop_out),
            nn.Tanh(),
            nn.Linear(5*512,512),
            nn.Tanh(),
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,4),
            nn.Tanh(),
            nn.Linear(4,1))

    def forward(self, x1):
        x1, _ = self.gru1(x1.swapaxes(1,2))
        x, _ = self.gru3(x1)
        x = self.flatten(x)
        x = self.lin(x)
        return x