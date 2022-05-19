import os
import numpy as np
import tensorflow as tf

def load_model(model_dir):
    """
    Given the path to the directory where the model has been saved,
    load the model into the script.
    """
    assert os.path.exists(model_dir), f"Model path does not exist: {model_dir}"
    return tf.keras.models.load_model(model_dir)

def prepare_data_for_cnn(jetimages, ceiling=255.):
    """
    Normalizes and fixes the input dimensions so that the image data
    is ready to be forwarded to the convolutional neural network.
    """
    # Stack the images so that they have (N_samples, 800) shape
    images = np.vstack(jetimages)
    return images / ceiling