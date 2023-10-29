from training.training import Training
import config_training

def main():
    dataset_path = config_training.dataset_path
    validation_ratio = config_training.validation_ratio
    batch_size = config_training.batch_size
    num_of_epochs = config_training.num_of_epochs
    number_of_classes = config_training.num_of_classes
    learning_rate = config_training.learning_rate
    model_save_path = config_training.model_save_path


    training = Training(dataset_path=dataset_path,validation_ratio=validation_ratio, batch_size= batch_size, num_of_epochs= num_of_epochs)
    training.execute_training(number_of_classes=number_of_classes, learning_rate=learning_rate, save_path=model_save_path)

if __name__ == '__main__':
    main()