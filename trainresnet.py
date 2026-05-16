from animal import Animaldataset
from resnet import ResnetCustom
from torch.utils.data import DataLoader
import torch
import logging
from argparse import ArgumentParser
from torchvision.transforms import ToTensor, Compose, Resize
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import os
import shutil


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size for training")
    parser.add_argument("--image_size", "-i", type=float, default=224, help="Image size for resizing input images")
    parser.add_argument("--root", "-r", type=str, default="/kaggle/working/resnet_animal/animals", help="URL of the dataset root directory")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--title", "-t",  type=str, default="trained_model", help="Title for the saved model")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    # call get_args() to parse command-line arguments
    args = get_args()

    # checking for GPU device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:        
        device = torch.device("cpu")

    logging.info("device: {}".format(device))

    # Transformer
    transformer = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])

    # checking file/folder
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.title):
        os.makedirs(args.title)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # DataLoader
    train_data = Animaldataset(root=args.root, train=True, transform=transformer)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_data = Animaldataset(root=args.root, train=False, transform=transformer)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    # Setup Hyperparameters
    epochs = args.epochs
    model = ResnetCustom()
    cen_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   
    # Define Writer for Tensorboard
    writer = SummaryWriter(args.logging)
    num_iter = len(train_loader)

    best_acc = 0.0
    # Training
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="cyan")
        for i, (images, labels) in enumerate(progress_bar):
            model = model.to(device)
            images, labels = images.to(device), labels.to(device)
                
            predict = model(images)
            loss = cen_loss(predict, labels)
            progress_bar.set_description("Epoch {}/{}, Iteration {}/{}, Loss: {}".format(epoch + 1, epochs, i+1, len(train_loader), loss.item()))
            writer.add_scalar("Loss/train", loss.item(), epoch * num_iter + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluation
        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                model = model.to(device)
                images, labels = images.to(device), labels.to(device)
                
                predict = model(images)
                index = torch.argmax(predict, dim=1)
                all_predictions.extend(index)
                all_labels.extend(labels)
        all_predictions = [i.item() for i in all_predictions]
        all_labels = [i.item() for i in all_labels]

        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        # torch.save(model.state_dict(), "{}/last.pt".format(args.title))
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, "{}/last.pt".format(args.title))

        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, "{}/best.pt".format(args.title))
            best_acc = accuracy




















