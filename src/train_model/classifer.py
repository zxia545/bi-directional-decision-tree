from torch import nn 

class SampleClassifer(nn.Module):
    def __init__(self, hid=3, output_feature=4):
        super().__init__()
        self.softmax = nn.Softmax()
        self.layer = nn.Sequential(
            nn.Linear(2, hid, bias=True), nn.ReLU(),
            nn.Linear(hid, hid, bias=True), nn.ReLU(),
            nn.Linear(hid, output_feature, bias=True))

    def forward(self, input_id):
        return self.softmax(self.layer(input_id))
    
class DynamicClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=3, output_size=4):
        super(DynamicClassifier, self).__init__()
        
        # Define the layers
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        
        # Define softmax operation
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the second dimension to normalize
        
    def forward(self, x):
        # Pass the input tensor through each layer in the model
        x = self.layer(x)
        
        # Apply the softmax operation
        x = self.softmax(x)
        
        return x