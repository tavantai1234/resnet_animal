import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.bl1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        self.bl2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512)
        )

        self.activation = nn.ReLU()
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fl = nn.Linear(in_features=512, out_features=10)
        # self.softmax = nn.Softmax


    def forward(self, x):
        x = self.bl1(x)
        x = self.bl2(x)  # Conv5 * 3
        x = self.activation(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fl(x)
        # x = self.softmax(x)
        return x


if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224)
    model = ResnetCustom()
    if torch.backends.mps.is_available():
        data = data.to("mps")
        model = model.to("mps")
    y = model(data)
    print(y)