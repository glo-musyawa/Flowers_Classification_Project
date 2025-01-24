Am so excited to share this project with you, welcome for the experience! Give it your best shot on images from the flower categories I trained it on. The model is trained on 23 epochs and achieves 93% accuracy on unseen data.

1. To run the train.py you just need to specify the path where the train.py file is located, then use this command:

!python path/to/train/folder/train.py

In train.py, the training, validation, and test datasets are loaded from a URL and extracted. They are preprocessed and the model is trained. The results will display the train, validation, and test accuracy, as well as the loss (cross-entropy loss). Finally, the model checkpoint will be saved as flower_classifier_checkpoint.pth.

Next, the predict.py file:

To run the predict.py file with no issues, specify the following in the command line argument:

image_path - Path to the image to predict
cat_json_path - Path to the cat_to_name json file

Command:

!python predict.py --cat_json-path /path/to/the cat_to_name.json --image-path /Path/to/the/image/to/predict

Since the model checkpoint was saved in the same folder, there will be no issues. Running predict.py will load the trained model, preprocess the image, and output the top 5 probabilities and flower names for the top 5 predictions. The top prediction flower name will also be displayed.

