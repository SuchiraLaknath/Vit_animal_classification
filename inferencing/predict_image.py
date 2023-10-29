import torch
from torchvision import transforms
from PIL import Image

class PredictImage:
    def __init__(self, model_path, class_names_path) -> None:
        
        self.class_names = self.get_class_names(class_name_path=class_names_path)
        self.transforms = self.get_transforms()
        self.device = self.get_device()
        self.model = self.load_model(model_path=model_path)

    def get_class_names(self, class_name_path):
        file = open(class_name_path, "r")
        data = file.read()
        class_name_list = data.split("\n")
        file.close()
        return class_name_list



    def get_transforms(self):
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return image_transform
    
    def load_model(self, model_path):
        model = torch.load(model_path)
        model.to(self.device)
        return model

    
    def get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def predict_image(self, image_path):
        image = Image.open(image_path)
        self.model.eval()
        with torch.inference_mode():
            transformed_image = self.transforms(image).unsqueeze(dim=0)
            transformed_image = transformed_image.to(self.device)
            predictions = self.model(transformed_image)
        predictions_after_softmax = torch.softmax(predictions, dim=1)
        predicted_class_index = torch.argmax(predictions_after_softmax)
        predicted_class = self.class_names[predicted_class_index]
        return predicted_class





