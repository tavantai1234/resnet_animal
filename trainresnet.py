from animal import Animaldataset, transformer
from resnet import ResnetCustom
from torch.utils.data import DataLoader
import torch
import logging



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # DataLoader
    train_data = Animaldataset(root="/kaggle/working/resnet_animal/animals", train=True, transform=transformer)
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    test_data = Animaldataset(root="/kaggle/working/resnet_animal/animals", train=False, transform=transformer)
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=True,
        num_workers=2
        ,drop_last=False
    )

    # Setup Hyperparameters
    epochs = 10
    model = ResnetCustom()
    cen_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # checking mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device: {}".format(device))

    # Training
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            if torch.backends.mps.is_available():
                model = model.to("mps")
                images, labels = images.to("mps"), labels.to("mps")

            predict = model(images)
            print(predict)
            print(labels.shape)
            print(type(predict))
            loss = cen_loss(predict, labels)
            print("Epoch {}/{}, Iteration {}/{}, Loss: {}".format(epoch + 1, epochs, i+1, len(train_loader), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



















