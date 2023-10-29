from torch.utils.data import DataLoader
from .dataset_preperation.dataset_preperation import DataSetPreperation
from tqdm import tqdm
import torchvision
import torch
from torch import nn

class Training:
    def __init__(self, dataset_path, validation_ratio, batch_size, num_of_epochs):
        self.dataset_path = dataset_path
        self.validation_ratio = validation_ratio
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.device = self.get_device()

    def get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_pretrained_model(self, num_of_classes = 90):
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(self.device)
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        pretrained_vit_transforms = pretrained_vit_weights.transforms()
        pretrained_vit.heads = nn.Linear(in_features=768, out_features=num_of_classes).to(self.device)
        return pretrained_vit, pretrained_vit_transforms

    def get_datasets(self, dataset_path, validation_ratio, transform):
        datset_preperation = DataSetPreperation(dataset_path=dataset_path, validation_ratio=validation_ratio,
                                                transform=transform)
        train_dataset, validation_dataset, class_names = datset_preperation.get_train_val_datasets()
        return train_dataset, validation_dataset, class_names

    def get_dataloaders(self, train_dataset, validation_dataset, batch_size, num_workers=None):
        train_dataloader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle= True, num_workers= num_workers)
        validation_dataloader = DataLoader(dataset= validation_dataset, batch_size= batch_size, num_workers=num_workers)
        return train_dataloader, validation_dataloader

    def train_step(self, model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc

    def test_step(self, model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device):

        # Put model in eval mode
        model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    def train(self, model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module,
              epochs: int,
              device: torch.device):

        # Create empty results dictionary
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []
                   }

        # Make sure model on target device
        model.to(device)

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model=model,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
            test_loss, test_acc = self.test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)

            # Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # Return the filled results at the end of the epochs
        return results

    def execute_training(self, number_of_classes, learning_rate, save_path):
        model, pretrained_vit_transforms = self.load_pretrained_model(num_of_classes=number_of_classes)
        train_dataset, validation_dataset, class_names = self.get_datasets(dataset_path=self.dataset_path,validation_ratio=self.validation_ratio, transform=pretrained_vit_transforms)
        train_dataloader, validation_dataloader = self.get_dataloaders(train_dataset=train_dataset, validation_dataset=validation_dataset,batch_size=self.batch_size,num_workers=2)
        optimizer = torch.optim.Adam(params=model.parameters(), lr= learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        pretrained_vit_results = self.train(model=model,
                                            train_dataloader=train_dataloader,
                                            test_dataloader=validation_dataloader,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=self.num_of_epochs,
                                            device=self.device)
        torch.save(model, save_path)











