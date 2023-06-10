import torch
import torchvision.models as models
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_and_validate():
    resnet = models.resnet152(pretrained=True)

    num_classes = 14

    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.00175, momentum=0.9, weight_decay=0.01)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    train_dir = 'data/training'
    test_dir = 'data/test'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    validation_dataset = ImageFolder(test_dir, transform=val_transform)

    batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 20

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0

        train_dataloader_with_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        resnet.train()
        for images, labels in train_dataloader_with_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_dataloader_with_bar.set_postfix({'Loss': loss.item()})

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        resnet.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(validation_dataloader)
        val_accuracy = 100 * correct / total

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        model_folder = 'models'
        if val_loss < best_val_loss:
            os.makedirs(model_folder, exist_ok=True)

            model_path = os.path.join(model_folder, "best_model.pth")

            torch.save(resnet.state_dict(), model_path)

            best_val_loss = val_loss

    return resnet


def test_model(resnet):
    state_dict = torch.load("models/best_model.pth")

    resnet.load_state_dict(state_dict)

    unknown_dir = 'data/unknown'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unknown_dataset = ImageFolder(unknown_dir, transform=transform)
    unknown_dataloader = DataLoader(unknown_dataset, batch_size=32, shuffle=False)

    resnet.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in unknown_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy}%")


def main():
    trained_model = train_and_validate()
    test_model(trained_model)


if __name__ == "__main__":
    main()
