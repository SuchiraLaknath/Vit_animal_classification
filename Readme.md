This Repo contains a VIT based image classification training and inferencing code.
used ViT_B_16 model.
Original model is trained with imagenet1k_v1 dataset.
we are doing tranfer leaning on last layer.

tested in conda environment with python3.9 (in windows envoronment).
Install pytorch 1.12 version depending on your cuda version.
for setup conda environment i have attached a environment.yml
requirement.txt file will be attached to intall dependancies.

you can setup training parameters in config_training.py except model, loss_fn and optimizer.
and you can set predict parameters in config.inference.py .
after setting config_training.py you can execute run_training.py for training.
after setting config_inference.py you can execute run_predictions.py for predictions.