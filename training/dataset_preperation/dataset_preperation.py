from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DataSetPreperation:
    def __init__(self, dataset_path, validation_ratio, transform):
        self.dataset_path = dataset_path
        self.validation_ratio = validation_ratio
        self.transform = transform

    def get_transforms(self, image_size):
        manual_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        return manual_transform

    def load_dataset(self, dataset_path, transform):
        dataset = ImageFolder(dataset_path, transform=transform)
        class_names = dataset.classes
        return dataset, class_names

    def split_dataset(self, dataset, validation_ratio):
        dataset_length = len(dataset)
        val_size = int(dataset_length*validation_ratio)
        train_size = dataset_length - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        return train_set, val_set


    def get_train_val_datasets(self):
        # transform = self.get_transforms(image_size=self.image_size)
        dataset, class_names = self.load_dataset(dataset_path=self.dataset_path, transform=self.transform)
        train_dataset, validation_datset= self.split_dataset(dataset=dataset, validation_ratio=self.validation_ratio)
        return train_dataset, validation_datset, class_names