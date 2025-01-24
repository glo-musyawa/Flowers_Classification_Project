# Flower Classification Project 🌸

This repository contains a deep learning project that classifies different types of flowers using a pre-trained VGG16 network. The model has been trained on 102 flower categories and achieves 93% accuracy on unseen data.

## Project Overview

- **Train the Model:** The `train.py` script loads the dataset, trains the model using the VGG16 architecture, and saves the trained model as `flower_classifier_checkpoint.pth`.
- **Predict Flower Type:** The `predict.py` script uses the saved model checkpoint to predict the type of flower from an input image, displaying the top 5 predicted flower types along with their probabilities.


## Files in this Repository

- **`train.py`:** Script to train the flower classification model.
- **`predict.py`:** Script to load the model and make predictions on new images.
- **`cat_to_name.json`:** A JSON file mapping flower category numbers to their names.
- **`Flower_Image_Classifier_Tutorial_File.md`:** Step-by-step instructions on how to use the training and prediction scripts.
- **Sample images**: Includes images like `Attempt_Canterbury.jpg` and `Attempt_Mexican_Aster.jpg` for testing the prediction script.


## Dependencies

To run this project, you'll need:

- Python 3.x
- Torch
- Torchvision
- Matplotlib
- NumPy
- Pillow

You can install the necessary libraries using:
```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Training the Model

To train the model, simply run the following command in your terminal:

```bash
python path/to/train/folder/train.py
```

The script will download the dataset, preprocess it, and train the model using the VGG16 architecture. It will save the model checkpoint as `flower_classifier_checkpoint.pth`.

---

## Making Predictions

Once the model is trained, you can use the `predict.py` script to classify new flower images:

```bash
python predict.py --cat_json-path /path/to/cat_to_name.json --image-path /path/to/image.jpg
```

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

