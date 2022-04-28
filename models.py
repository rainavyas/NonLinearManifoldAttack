import torch.nn as nn

class FFN(nn.Module):
    '''
    Feed Forward Network for binary classification
    '''
    def __init__(self, num_hidden_layers=1, hidden_layer_size=10):
        super().__init__()
        inp_layer = nn.Sequential(nn.Linear(3, hidden_layer_size), nn.ReLU())
        out_layer = nn.Linear(hidden_layer_size, 1)

        hidden_layers = []
        for i in range(num_hidden_layers-1):
            hidden_layers.append(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
        
        self.model = nn.Sequential(inp_layer, *hidden_layers, out_layer)
    
    def forward(self, x):
        '''
        x: Tensor [batch x 3]
        '''
        return self.model(x).squeeze(dim=1)