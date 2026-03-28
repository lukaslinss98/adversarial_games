from torch import Tensor, nn


class QNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dims),
        )

    def forward(self, x) -> Tensor:
        return self.net(x)
