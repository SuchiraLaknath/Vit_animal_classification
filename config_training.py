import os
dataset_path = 'Data/animals'
validation_ratio = 0.2
batch_size = 16
num_of_epochs = 5
model_save_path = 'training/model_save_path/model_01.pt'
num_of_classes = len(os.listdir(dataset_path))
learning_rate = 0.01