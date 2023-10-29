from inferencing.predict_image import PredictImage
import config_inference

def main():
    image_path = config_inference.image_path
    class_names_file_path = config_inference.class_names_file_path
    model_path = config_inference.model_path
    predict_image = PredictImage(model_path=model_path,class_names_path=class_names_file_path)
    class_name = predict_image.predict_image(image_path=image_path)
    print(f"class name = {class_name}")

if __name__ == "__main__":
    main()