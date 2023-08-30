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