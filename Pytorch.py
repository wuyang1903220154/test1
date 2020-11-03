class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.model(x)  
