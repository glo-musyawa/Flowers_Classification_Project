# Flower Classification Project 🌸

This repository contains a deep learning project that classifies different types of flowers using either a pre-trained VGG16 or ALEXNET network. The model has been trained on 102 flower categories and achieves 93% accuracy using VGG16 and 89% accuracy using ALEXNET on unseen data.

## Project Overview

- **Train the Model:** The `train.py` script loads the dataset, trains the model using the VGG16 architecture, and saves the trained model as `flower_classifier_checkpoint.pth`.
- **Predict Flower Type:** The `predict.py` script uses the saved model checkpoint to predict the type of flower from an input image, displaying the top k predicted flower types along with their probabilities.


## Files in this Repository

- **`train.py`:** Script to train the flower classification model.
- **`predict.py`:** Script to load the model and make predictions on new images.
- **`cat_to_name.json`:** A JSON file mapping flower category numbers to their names.
- **`Flower_Image_Classifier_Tutorial_File.md`:** Step-by-step instructions on how to use the training and prediction scripts.
- **Sample images**: Includes image `Attempt_Mexican_Aster.jpg` for testing the prediction script.


## Dependencies

All necessary dependencies are automatically installed when running both the `train.py` and `predict.py` files. I've added several functionalities to make the scripts more flexible and dynamic:

- **Pretrained Model Selection**: The `train.py` script now supports two pretrained models, `vgg16` and `alexnet`, allowing the user to dynamically choose between them when training the model.
- **GPU Availability**: Both `train.py` and `predict.py` include functionality to check GPU availability and enable users to utilize it for faster training and prediction if available.
- **Custom Checkpoint Saving**: Users can specify the path where they wish to save their trained model's checkpoint, ensuring flexibility in managing saved models.
- **Epoch and Hidden Unit Customization**: Users can determine the number of epochs and hidden units they want to use during training, allowing for better control over the model architecture.
- **Prediction Customization**: The `predict.py` script allows users to select the number of top predictions (`topk`) to display, further refining their results.
- **Checkpoint Path**: During prediction, users can specify the path to the saved model checkpoint, enabling them to reload and use different models as needed.
  
Additionally, an image is included for testing the models' capabilities, enabling quick verification of the prediction accuracy.

## Training the Model

To train the flower classifier model using `train.py`, run the following command:

```bash
!python path/to/train.py
```

You can customize various parameters by providing additional arguments. Below is a list of the available options for training your model:

### Available Options:

- **`--gpu`**: Enables the use of GPU for training, if available. By default, the script will use the CPU if no GPU is detected or this flag is omitted.
  
- **`--arch {vgg16, alexnet}`**: Specifies the model architecture to be used. The script is dynamic and supports both `vgg16` and `alexnet` pretrained models. The default model is `vgg16` if no architecture is specified.
  
- **`--save_dir SAVE_DIR`**: Allows users to specify the directory where they would like to save the model’s checkpoint. If not provided, the checkpoint will be saved in the current directory by default.
  
- **`--learning_rate LEARNING_RATE`**: Sets the learning rate for the model training. The default learning rate is `0.0001`.
  
- **`--hidden_units HIDDEN_UNITS`**: Defines the number of hidden units in the classifier. The default value is `4096`.
  
- **`--epochs EPOCHS`**: Determines how many epochs the model will train for. The default number of epochs is `23`.

### Example Command:

```bash
!python train.py --gpu --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 20 --save_dir /path/to/save/checkpoint
```

This command trains a `vgg16` model with:
- GPU enabled (if available),
- A learning rate of `0.001`,
- 512 hidden units,
- 20 epochs,
- Saving the model checkpoint to the specified directory.



## Making Predictions

Once the model is trained, you can use the `predict.py` script to classify new flower images:

```bash
python predict.py --cat_json-path /path/to/cat_to_name.json --image-path /path/to/image.jpg
```
Below are the options available for customizing predictions:

### Available Options:

- **`--gpu`**: Enables the use of GPU for faster prediction calculations, if available. By default, the script will use the CPU if no GPU is detected or this flag is omitted.
  
- **`--image-path IMAGE_PATH`**: Specifies the path to the image you want to classify. This is a required argument.
  
- **`--cat_json-path CAT_JSON_PATH`**: Specifies the path to the `cat_to_name.json` file, which contains the mapping of category labels to actual flower names. This is a required argument.

- **`--checkpoint-path CHECKPOINT_PATH`**: The path to the saved checkpoint of the model. This is a required argument, allowing the script to load the trained model for prediction.

- **`--topk TOPK`**: Specifies how many top predictions to return for the image. The default is `5`.

### Example Command:

```bash
!python predict.py --gpu --image-path /path/to/image.jpg --cat_json-path /path/to/cat_to_name.json --checkpoint-path /path/to/checkpoint.pth --topk 5
```

This command runs the `predict.py` script with:
- GPU enabled (if available),
- The specified image file,
- The category-to-name mapping file,
- A saved model checkpoint,
- Returning the top 5 predictions for the image.


The script will load the trained model, preprocess the image, and print the top 5 flower category predictions with their probabilities.


## Example Output

For an image of a Mexican Aster:

```
The predicted result is:
mexican aster flower

Here are the top 5 predictions of the flower in the image:
***********************************
[1.0, 0.0, 0.0, 0.0, 0.0]
['mexican aster', 'californian poppy', 'english marigold', 'black-eyed susan', 'primula']
```


## License

This project is licensed under the MIT License. See the LICENSE file for more details.

