from animal import Animaldataset
from resnet import ResnetCustom
from torch.utils.data import DataLoader
import torch
import logging
from argparse import ArgumentParser
from torchvision.transforms import ToTensor, Compose, Resize
from tqdm.autonotebook import tqdm



def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--image_size", type=float, default=224, help="Image size for resizing input images")
    parser.add_argument("--root", type=str, default="/kaggle/working/resnet_animal/animals", help="URL of the dataset root directory")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = get_args()
    print(args.epochs)
    print(args.batch_size)
    transformer = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # DataLoader
    train_data = Animaldataset(root=args.root, train=True, transform=transformer)
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_data = Animaldataset(root=args.root, train=False, transform=transformer)
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=False
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
        progress_bar = tqdm(train_loader, colour="cyan")
        for i, (images, labels) in enumerate(progress_bar):
            if torch.backends.mps.is_available():
                model = model.to("mps")
                images, labels = images.to("mps"), labels.to("mps")
            elif torch.cuda.is_available():
                model = model.to("cuda")
                images, labels = images.to("cuda"), labels.to("cuda")
                
            predict = model(images)
            loss = cen_loss(predict, labels)
            progress_bar.set_description("Epoch {}/{}, Iteration {}/{}, Loss: {}".format(epoch + 1, epochs, i+1, len(train_loader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



















