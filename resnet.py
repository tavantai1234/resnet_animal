import torch.nn as nn
import torch

class ResnetCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.bl1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        self.bl21 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.bl22 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.bl31 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.bl32 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512)
        )
        self.bl41 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.bl42 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.bl51 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.bl52 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(512)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(1024)
        )
        self.skip5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(2048)
        )
        self.activation = nn.ReLU()
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fl = nn.Linear(in_features=2048, out_features=10)
        # self.softmax = nn.Softmax


    def forward(self, x):
        x = self.bl1(x) # conv1
        x = self.bl21(x) + self.skip2(x) # Conv2 * 1
        x = self.activation(x)
        x = self.bl22(x) + x # conv2 * 2
        x = self.activation(x)
        x = self.bl22(x) + x # conv2 * 3
        x = self.activation(x)
        x = self.bl31(x) + self.skip3(x) # Conv3 * 1
        x = self.activation(x)
        x = self.bl32(x) + x # conv3 * 2
        x = self.activation(x)
        x = self.bl32(x) + x  # conv3 * 3
        x = self.activation(x)
        x = self.bl32(x) + x  # conv3 * 4
        x = self.activation(x)
        x = self.bl41(x) + self.skip4(x)  # Conv4 * 1
        x = self.activation(x)
        x = self.bl42(x) + x # conv4 * 2
        x = self.activation(x)
        x = self.bl42(x) + x  # conv4 * 3
        x = self.activation(x)
        x = self.bl42(x) + x  # conv4 * 4
        x = self.activation(x)
        x = self.bl42(x) + x  # conv4 * 5
        x = self.activation(x)
        x = self.bl42(x) + x  # conv4 * 6
        x = self.activation(x)
        x = self.bl51(x) + self.skip5(x) # Conv5 * 1
        x = self.activation(x)
        x = self.bl52(x) + x # Conv5 * 2
        x = self.activation(x)
        x = self.bl52(x) + x  # Conv5 * 3
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