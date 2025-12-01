import argparse
import cv2
import json
import os
import pickle
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, message=".*input_shape*")

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import matplotlib.patches as mpatches
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50, ResNet101, MobileNet, MobileNetV2, EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

plt.rcParams["font.family"] = "serif"


def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total_pixels = tf.cast(tf.size(y_true, out_type=tf.int32), tf.float32)
    accuracy = correct_predictions / total_pixels
    return accuracy


def precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def recall(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1_score = 2 * (prec * rec) / (prec + rec + tf.keras.backend.epsilon())
    return f1_score


# Intersection over Union (IoU)
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou


def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    epsilon = tf.keras.backend.epsilon()
    dice = (2 * intersection + epsilon) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)
    return dice


def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    possible_negatives = tf.reduce_sum(1 - y_true)
    specificity = true_negatives / (possible_negatives + tf.keras.backend.epsilon())
    return specificity


# Boundary Displacement Error (BDE)
def bde(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    # Compute the Sobel edge maps for y_true and y_pred
    edges_true = tf.image.sobel_edges(y_true)
    edges_pred = tf.image.sobel_edges(y_pred)

    # Calculate the magnitude of the Sobel gradients (for edge strength)
    magnitude_true = tf.sqrt(tf.square(edges_true[..., 0]) + tf.square(edges_true[..., 1]))
    magnitude_pred = tf.sqrt(tf.square(edges_pred[..., 0]) + tf.square(edges_pred[..., 1]))

    # Calculate the distance between the edges and average the displacement error
    displacement_error = tf.norm(magnitude_true - magnitude_pred, axis=-1)
    bde = tf.reduce_mean(displacement_error)
    return bde


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def plot_history(history, save_plot_path=None):
    try:
        history = history.history
    except AttributeError:
        pass

    plt.figure(figsize=(24, 12))

    # Loss
    plt.subplot(3, 3, 1)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(3, 3, 2)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Precision
    plt.subplot(3, 3, 3)
    plt.plot(history["precision"], label="Training Precision")
    plt.plot(history["val_precision"], label="Validation Precision")
    plt.title("Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend()

    # Recall
    plt.subplot(3, 3, 4)
    plt.plot(history["recall"], label="Training Recall")
    plt.plot(history["val_recall"], label="Validation Recall")
    plt.title("Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()

    # F1 Score
    plt.subplot(3, 3, 5)
    plt.plot(history["f1_score"], label="Training F1 Score")
    plt.plot(history["val_f1_score"], label="Validation F1 Score")
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()

    # Specificity
    plt.subplot(3, 3, 6)
    plt.plot(history["specificity"], label="Training Specificity")
    plt.plot(history["val_specificity"], label="Validation Specificity")
    plt.title("Specificity")
    plt.xlabel("Epochs")
    plt.ylabel("Specificity")
    plt.legend()

    # Dice Coefficient
    plt.subplot(3, 3, 7)
    plt.plot(history["dice_coefficient"], label="Training Dice Coefficient")
    plt.plot(history["val_dice_coefficient"], label="Validation Dice Coefficient")
    plt.title("Dice Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.legend()

    # IoU
    plt.subplot(3, 3, 8)
    plt.plot(history["iou"], label="Training IoU")
    plt.plot(history["val_iou"], label="Validation IoU")
    plt.title("Intersection over Union (IoU)")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()

    # Boundary Displacement Error (BDE)
    plt.subplot(3, 3, 9)
    plt.plot(history["bde"], label="Training BDE")
    plt.plot(history["val_bde"], label="Validation BDE")
    plt.title("Boundary Displacement Error (BDE)")
    plt.xlabel("Epochs")
    plt.ylabel("BDE")
    plt.legend()

    plt.tight_layout()
    if save_plot_path:
        plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
    plt.show()


class ModelManager:
    """
    Example usage:
    m = ModelManager()
    m.list_backbones()
    m.list_models()
    m.select_backbone('resnet50')
    m.select_model('unet')
    m.get_current_model_summary()
    m.train(epochs=50)
    m.save_model()
    m.load_model()
    """
    def __init__(self, batch_size=4, augment_data=True):
        # Set up directories
        self.images_dir = "segmentation/images/"
        self.masks_dir = "segmentation/masks/"
        self.test_images_dir = "segmentation/test/images/"
        self.test_masks_dir = "segmentation/test/masks/"
        self.models_dir = "models/"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Store training parameters
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.val_split = 0.2
        self.shuffle = True  # For shuffling training images

        # Define normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Define augmentations
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
        ])
        self.valid_transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
        ])

        # Load datasets once during initialization
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_datasets()

        # Lists to hold all backbones and models  https://keras.io/api/applications/
        self.backbones = ["resnet50", "resnet101", "mobilenet", "mobilenetv2", "efficientnetv2b0", "efficientnetv2b1", "efficientnetv2b2", "efficientnetv2b3", "efficientnetv2s", "efficientnetv2m"]
        self.models = ["unet", "unet_plus_plus", "unet_3plus", "segnet", "mask_rcnn", "mask_rcnn_custom", "linknet", "deep_lab_v3"]
        self.models_not_augmented = ["unet_not_augmented"] + [f"unet_{backbone}_not_augmented" for backbone in self.backbones]

        # Placeholder for selected backbone, model, and history
        self.current_backbone = None
        self.current_model_name = None
        self.current_model = None
        self.history = None

        # Define metrics to be used for all models
        self.metrics = [accuracy, precision, recall, f1_score, iou, dice_coefficient, specificity, bde]

        # Define loss function to be used
        self.loss = bce_dice_loss  # Alternate: "binary_crossentropy"

        # Define custom metrics for model loading
        self.custom_objects = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou,
            "dice_coefficient": dice_coefficient,
            "specificity": specificity,
            "bde": bde,
            "bce_dice_loss": bce_dice_loss,
        }

        # Define callbacks for training
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                start_from_epoch=10,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
 
    def list_backbones(self):
        """Returns a list of available backbones."""
        return self.backbones

    def list_models(self):
        """Returns a list of available models."""
        return self.models

    def select_backbone(self, backbone_type=None):
        """Selects a backbone by its name and reloads the model."""
        if backbone_type in self.backbones or backbone_type is None:
            self.current_backbone = backbone_type
            if self.current_model_name:
                self._reload_model()
        else:
            raise ValueError(f"Backbone '{backbone_type}' not found. Use 'list_backbones()'.")

    def select_model(self, model_name):
        """Selects a model by its name and reloads the model."""
        if model_name in (self.models + self.models_not_augmented):
            self.current_model_name = model_name
            self._reload_model()
        else:
            raise ValueError(f"Model '{model_name}' not found. Use 'list_models()'.")

    def get_current_model_summary(self):
        """Prints a summary of the currently selected model."""
        if self.current_model:
            self.current_model.summary()
        else:
            self._raise_no_model_selected()

    def _reload_model(self):
        """Reloads the current model with the selected encoder (backbone)."""
        if self.current_model_name is None:
            raise ValueError("No model selected.")

        tf.compat.v1.reset_default_graph()

        # Dynamically construct the method name based on the model name
        model_method_name = self.current_model_name.lower()
        if "not_augmented" in model_method_name:
            self.current_model = self.dummy_model(name=model_method_name)
            print(f"Reloaded model '{self.current_model_name}'")
            return

        # Check if the method exists in the class and re-create the model
        if hasattr(self, model_method_name):
            model_method = getattr(self, model_method_name)
            self.current_model = model_method()
            print(f"Reloaded model '{self.current_model_name}' with backbone '{self.current_backbone}'.")
        else:
            raise ValueError(f"Model method '{model_method_name}' not found in ModelManager.")

    def _raise_no_model_selected(self):
        raise ValueError("No model selected. Use 'select_model()' to choose a model.")

    def _load_and_augment(self, image_path, mask_path, is_train=True):
        """Load and preprocess images with augmentation for training."""
        image_path = image_path.numpy().decode("utf-8")
        mask_path = mask_path.numpy().decode("utf-8")

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_size = (512, 512)

        # Resize images using the provided img_size tuple
        img = cv2.resize(img, img_size)
        mask = cv2.resize(mask, img_size) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        # Apply augmentations based on whether it is training or validation
        transform = self.train_transform if is_train else self.valid_transform
        augmented = transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        return img, mask

    def _parse_image_mask(self, image_path, mask_path, is_train=True):
        """TensorFlow function to wrap the preprocessing for training and validation."""
        img, mask = tf.py_function(func=self._load_and_augment, inp=[image_path, mask_path, is_train], Tout=[tf.float32, tf.float32])
        img.set_shape([512, 512, 3])
        mask.set_shape([512, 512, 1])
        return img, mask

    def _simple_data_load(self, image_files, mask_files, img_size=(512, 512)):
        """Load images with no augmentation."""
        images = []
        masks = []
        for img_file, mask_file in zip(image_files, mask_files):
            img = cv2.imread(img_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            if mask is None:
                print(f"Warning: Could not read mask {mask_file}")
                continue
            img = cv2.resize(img, img_size) / 255.0
            mask = cv2.resize(mask, img_size) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)

    def _create_datasets(self):
        """Create TensorFlow Dataset with train-validation split."""
        # Get sorted lists of image and mask file paths
        image_files = sorted([os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)])
        mask_files = sorted([os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)])
        test_image_files = sorted([os.path.join(self.test_images_dir, f) for f in os.listdir(self.test_images_dir)])
        test_mask_files = sorted([os.path.join(self.test_masks_dir, f) for f in os.listdir(self.test_masks_dir)])

        if self.augment_data:
            # Split into training and validation sets
            train_images, val_images, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=self.val_split, random_state=42)
            print(f"Total images: {len(image_files)+len(test_image_files)}, Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_image_files)}")
    
            # Create training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
            if self.shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=len(train_images))
            train_dataset = train_dataset.map(lambda img, msk: self._parse_image_mask(img, msk, is_train=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_dataset = train_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
            # Create validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
            val_dataset = val_dataset.map(lambda img, msk: self._parse_image_mask(img, msk, is_train=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_dataset = val_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            # Create validation dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((test_image_files, test_mask_files))
            test_dataset = test_dataset.map(lambda img, msk: self._parse_image_mask(img, msk, is_train=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        else:
            X, Y = self._simple_data_load(image_files, mask_files)
            X_test, Y_test = self._simple_data_load(test_image_files, test_mask_files)
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.val_split, random_state=42)

            print(f"Total images: {len(image_files)+len(test_image_files)}, Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(test_image_files)}")
            print("Training without dataset augmentation")

            # Convert to tf.data.Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(self.batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(self.batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(self.batch_size)

        return train_dataset, val_dataset, test_dataset

    def train(self, epochs):
        """
        Trains the current model.

        Parameters:
            epochs (int): Number of epochs.
        """
        if self.current_model:
            self.current_model.compile(optimizer=Adam(learning_rate=1e-4), loss=self.loss, metrics=self.metrics)
            self.history = self.current_model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset, callbacks=self.callbacks, verbose=1)

            # Add new field to history for early stopping callback
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    stopped_epoch = callback.stopped_epoch
                    if stopped_epoch > 0:
                        self.history.history['stopped_epoch'] = stopped_epoch + 1  # Epoch is 1-based, callback is 0-based
                    else:
                        self.history.history['stopped_epoch'] = epochs
                    break
        else:
            self._raise_no_model_selected()

    def evaluate(self, image_idx=0, batch_idx=0, image_file=None, mask_file=None, use_test_images=False, save_mask_path=None, save_plot_path=None, show_plots=True):
        """
        Evaluates the current model on a validation, test, or adhoc image.

        Parameters:
            image_idx (int): Validation dataset image index.
            batch_idx (int): Batch index (typically 0-3).
            image_file (str): Alternate input image file.
            mask_file (str): Alternate input mask file.
            use_test_images (bool): Will use test images instead of validation images.
            save_mask_path (str): Path to save the predicted mask as a PNG file.
            save_plot_path (str): Path to save the plot as a PNG file.
        """

        # Reverse normalization: (normalized_value * std) + mean
        def denormalize_image(image, mean, std):
            image = image * std + mean
            image = np.clip(image, 0, 1)
            return image

        if self.current_model:
            if image_file and mask_file:
                images, masks = self._simple_data_load([image_file], [mask_file])
                image = (images[0] - self.mean) / self.std
                mask = masks[0]
            else:
                images, masks = [], []
                dataset = self.val_dataset
                if use_test_images:
                    dataset = self.test_dataset
                for img, mask in dataset:
                    images.append(img.numpy())
                    masks.append(mask.numpy())
                image = images[image_idx][batch_idx]
                mask = masks[image_idx][batch_idx]

            denormalized_image = denormalize_image(image, self.mean, self.std)

            start_time = time.time()
            prediction = self.current_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            inference_time = time.time() - start_time
            print(f"Inference Time: {inference_time:.4f} seconds")

            pred_mask = (prediction.squeeze() > 0.5).astype(np.uint8)
            true_mask = (mask.squeeze() > 0.5).astype(np.uint8)
            iou_score = iou(pred_mask, true_mask)
            iou_score = float(iou_score) if hasattr(iou_score, 'numpy') else iou_score
            print(f"IoU Score: {iou_score:.4f}")

            # Save the predicted mask as a 512x512 PNG image
            if save_mask_path:
                cv2.imwrite(save_mask_path, pred_mask * 255)
                print(f"Predicted mask saved to {save_mask_path}")

            if show_plots:
                plt.figure(figsize=(16, 4))

                plt.subplot(1, 4, 1)
                plt.title("Original Image")
                plt.imshow(denormalized_image)
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.title("Input Image (Normalized)")
                plt.imshow(np.clip(image, 0, 1))
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.title("Ground Truth Mask")
                plt.imshow(mask.squeeze(), cmap="gray")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.title("Predicted Mask")
                plt.imshow(prediction.squeeze(), cmap="gray")
                plt.axis("off")

                # plt.subplots_adjust(wspace=0.1)

                if save_plot_path:
                    plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
                plt.show()
            # return inference_time, iou_score
        else:
            self._raise_no_model_selected()

    def evaluate_model_performance(self, output_file, runs=5):
        """
        Evaluates model load, inference time, and iou on test images and logs them.

        Parameters:
            output_file (str): The output file to save times.
            runs (int): The number of runs to do.
        """
        results = {self.current_model.name: {"model_load_times": [], "model_inference_times": [], "test_iou": []}}

        self.load_model()
        for run in range(runs):
            load_time = self.load_model()
            print(self.current_model)
            print(self.current_model.name)
            results[self.current_model.name]["model_load_times"].append(load_time)

        self.evaluate(0, 0, use_test_images=True)
        total_images = len(os.listdir(self.test_images_dir))
        num_batches = (total_images + self.batch_size - 1) // self.batch_size
        for batch_index in range(num_batches):
            start_idx = batch_index * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_images)
            for image_index in range(start_idx, end_idx):
                inference_time, iou = self.evaluate(batch_index, image_index - start_idx, use_test_images=True, show_plots=False)
                results[self.current_model.name]["model_inference_times"].append(inference_time)
                results[self.current_model.name]["test_iou"].append(iou)

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[self.current_model.name] = results[self.current_model.name]
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def plot_metrics(self, save_plot_path=None):
        """Calls external function to plot history metrics."""
        if self.history:
            plot_history(self.history, save_plot_path)

    def plot_model_comparison(self, model=None, metric="val_iou", plot_best=False, compare_not_augmented=False, save_plot_path=None):
        """
        Plots the target metric scores of all models saved as pickle files.

        Parameters:
            model (str or list): Only compare using this model.
            metric (str): The training metric to compare against.
            plot_best (bool): If True, only plot the best variant of each model based on the metric.
            compare_not_augmented (bool): If True, compare the specified model with its not-augmented versions.
            save_plot_path (str): Path to save the plot as a PNG file.
        """

        def load_pickle(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)

        model_scores = []
        if isinstance(model, str):
            model = [model]

        # Case for loading from JSON file if metric is 'test_iou'
        if metric == "test_iou":
            with open("model_performance.json", "r") as file:
                data = json.load(file)

            for model_name, times in data.items():
                if model is None or any(m in model_name for m in model):
                    if "_not_augmented" in model_name and not compare_not_augmented:
                        continue
                    metric_values = times.get("test_iou", [])
                    average_values = np.mean(metric_values)
                    model_name += "_history.pkl" # to match the format of the history pickle files
                    if any(b in model_name for b in self.backbones):
                        name_split = model_name.rsplit("_", 4 if '_not_augmented' in model_name else 2)
                    else:
                        name_split = model_name.rsplit("_", 3 if '_not_augmented' in model_name else 1)
                    base_model_name = name_split[0]
                    model_scores.append((base_model_name, model_name, average_values))
        else:
            # Iterate through all pickle files in the models directory
            for pickle_file in os.listdir(self.models_dir):
                if pickle_file.endswith(".pkl") and (model is None or any(m in pickle_file for m in model)):
                    if "_not_augmented" in pickle_file and not compare_not_augmented:
                        continue
                    history = load_pickle(os.path.join(self.models_dir, pickle_file))
    
                    # Check for the target metric in the history and retrieve its values
                    if metric in history:
                        metric_values = history[metric]
                        best_metric = max(metric_values)
                        if any(b in pickle_file for b in self.backbones):
                            name_split = pickle_file.rsplit("_", 4 if '_not_augmented' in pickle_file else 2)
                        else:
                            name_split = pickle_file.rsplit("_", 3 if '_not_augmented' in pickle_file else 1)
                        base_model_name = name_split[0]
                        model_scores.append((base_model_name, pickle_file, best_metric))

        # If plot_best is True, filter to keep only the best variant of each model
        if plot_best:
            best_model_scores = {}
            for base_model_name, full_file_name, score in model_scores:
                if base_model_name not in best_model_scores or score > best_model_scores[base_model_name][1]:
                    best_model_scores[base_model_name] = (full_file_name, score)
            model_scores = [(base_name, data[0], data[1]) for base_name, data in best_model_scores.items()]

        if compare_not_augmented:
            if plot_best:
                raise ValueError("Set plot_best to false.")
            if len(model) != 1:
                raise ValueError("Specify exactly one model for comparison with _not_augmented versions.")
            model_scores = [ms for ms in model_scores if ms[0] == model[0] or f"{model[0]}_not_augmented" in ms[1]]
        
        # Sort models by their metric score
        model_scores.sort(key=lambda x: x[2])

        # Extract model names and their corresponding metric scores
        model_names = [model[1].replace("_history.pkl", "") for model in model_scores]
        iou_scores = [model[2] for model in model_scores]

        # Set dynamic plot height based on number of models
        num_models = len(model_names)
        fig_height = max(5, num_models * 0.18)

        plt.figure(figsize=(10, fig_height))
        bars = plt.barh(model_names, iou_scores, color=["lightcoral" if "_not_augmented" in name else "skyblue" for name in model_names])
        plt.xlabel(f"{metric} Score")
        plt.ylabel("Model")
        plt.title(f"Model {metric} Scores")

        # Set x-axis limit to add space for text labels
        max_score = max(iou_scores) if iou_scores else 1
        plt.xlim(0, max_score * 1.1)
        if compare_not_augmented:
            plt.xlim(0, max_score * 1.13)

        # Adjust y-axis limits to remove extra space at the top and bottom
        plt.ylim(-0.7, len(model_names) - 0.25)

        # Calculate padding based on x-axis range
        xlim = plt.xlim()
        x_range = xlim[1] - xlim[0]
        padding = x_range * 0.01

        for bar, score in zip(bars, iou_scores):
            plt.text(bar.get_width() + padding, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

        # Add legend for augmented and not_augmented
        if compare_not_augmented:
            augmented_patch = mpatches.Patch(color="skyblue", label="Augmented")
            not_augmented_patch = mpatches.Patch(color="lightcoral", label="Not Augmented")
            plt.legend(
                handles=[augmented_patch, not_augmented_patch], 
                loc="best",
                handlelength=1,
                handleheight=0.8,
                handletextpad=0.5
            )

        plt.tight_layout()
        if save_plot_path:
            if len(bars) > 30:
                plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=125)
            else:
                plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
        plt.show()

    def plot_model_times(self, model=None, load_times_only=False, inference_times_only=False, save_plot_path=None):
        """
        Plots the model load and inference times.

        Parameters:
            model (str or list): Only compare using this model.
            load_times_only (bool): Plot model load times.
            inference_times_only (bool): Plot model inference times.
            save_plot_path (str): Path to save the plot as a PNG file.
        """
        with open("model_performance.json", "r") as file:
            data = json.load(file)

        if isinstance(model, str):
            model = [model]

        average_load_times = []
        average_inference_times = []

        for model_name, times in data.items():
            if model is not None and not any(m in model_name for m in model):
                continue
            if "_not_augmented" in model_name:
                continue

            # Calculate average load and inference times
            load_times = times["model_load_times"]
            inference_times = times["model_inference_times"]

            if load_times:
                average_load_times.append((model_name, np.mean(load_times)))
            if inference_times:
                average_inference_times.append((model_name, np.mean(inference_times)))

        # Sort models by times in ascending order
        average_load_times.sort(key=lambda x: x[1], reverse=True)
        average_inference_times.sort(key=lambda x: x[1], reverse=True)

        # Extract model names and average times for plotting
        load_model_names = [model[0] for model in average_load_times]
        load_times = [model[1] for model in average_load_times]

        inference_model_names = [model[0] for model in average_inference_times]
        inference_times = [model[1] for model in average_inference_times]

        # Set dynamic plot height based on number of models
        num_load_models = len(load_model_names)
        load_fig_height = max(5, num_load_models * 0.18)

        num_inference_models = len(inference_model_names)
        inference_fig_height = max(5, num_inference_models * 0.18)

        # Plot model load times
        if not inference_times_only:
            plt.figure(figsize=(10, load_fig_height))
            load_bars = plt.barh(load_model_names, load_times, color="skyblue")
            plt.xlabel("Average Load Time (s)")
            plt.ylabel("Model")
            plt.title("Average Model Load Times")
            plt.xlim(0, max(load_times) * 1.1 if load_times else 1)
            plt.ylim(-0.7, len(load_model_names) - 0.25)
    
            # Calculate padding based on x-axis range
            xlim = plt.xlim()
            x_range = xlim[1] - xlim[0]
            padding = x_range * 0.01
    
            for bar, time in zip(load_bars, load_times):
                plt.text(bar.get_width() + padding, bar.get_y() + bar.get_height() / 2, f"{time:.3f}", va="center")
    
            plt.tight_layout()
            if save_plot_path:
                if len(load_bars) > 30:
                    plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=125)
                else:
                    plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
            plt.show()

        # Plot model inference times
        if not load_times_only:
            plt.figure(figsize=(10, inference_fig_height))
            inference_bars = plt.barh(inference_model_names, inference_times, color="skyblue")
            plt.xlabel("Average Inference Time (s)")
            plt.ylabel("Model")
            plt.title("Average Model Inference Times")
            plt.xlim(0, max(inference_times) * 1.1 if inference_times else 1)
            plt.ylim(-0.7, len(inference_model_names) - 0.25)
    
            # Calculate padding based on x-axis range
            xlim = plt.xlim()
            x_range = xlim[1] - xlim[0]
            padding = x_range * 0.01
    
            for bar, time in zip(inference_bars, inference_times):
                plt.text(bar.get_width() + padding, bar.get_y() + bar.get_height() / 2, f"{time:.3f}", va="center")
    
            plt.tight_layout()
            if save_plot_path:
                if len(inference_bars) > 30:
                    plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=125)
                else:
                    plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
            plt.show()

    def plot_model_size(self, model=None, save_plot_path=None):
        """
        Plots the model size.

        Parameters:
            model (str or list): Only compare using this model.
            save_plot_path (str): Path to save the plot as a PNG file.
        """
        model_sizes = []
        for file_name in os.listdir(self.models_dir):
            if file_name.endswith(".keras") and "not_augmented" not in file_name:
                model_path = os.path.join(self.models_dir, file_name)
                model_size = os.path.getsize(model_path) / (1024 * 1024)

                if model is None or any(m in file_name for m in (model if isinstance(model, list) else [model])):
                    if "_not_augmented" in file_name:
                        continue
                    model_sizes.append((file_name.replace(".keras", ""), model_size))

        model_sizes.sort(key=lambda x: x[1], reverse=True)
        model_names = [model[0] for model in model_sizes]
        model_sizes_mb = [model[1] for model in model_sizes]

        num_models = len(model_names)
        fig_height = max(5, num_models * 0.18)

        # Plot model sizes
        plt.figure(figsize=(10, fig_height))
        bars = plt.barh(model_names, model_sizes_mb, color="skyblue")
        plt.xlabel("Model Size (MB)")
        plt.ylabel("Model")
        plt.title("Model Sizes")
        plt.xlim(0, max(model_sizes_mb) * 1.2 if model_sizes_mb else 1)
        plt.ylim(-0.7, len(model_names) - 0.25)

        # Calculate padding based on x-axis range
        xlim = plt.xlim()
        x_range = xlim[1] - xlim[0]
        padding = x_range * 0.01

        for bar, size in zip(bars, model_sizes_mb):
            plt.text(bar.get_width() + padding, bar.get_y() + bar.get_height() / 2, f"{size:.2f} MB", va="center")

        plt.tight_layout()
        if save_plot_path:
            if len(bars) > 30:
                plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=125)
            else:
                plt.savefig(save_plot_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
        plt.show()

    def save_model_plot(self):
        plot_model(
            self.current_model,
            to_file=self.models_dir + f'model_plot_{self.current_model.name}.png',
            show_shapes=False,
            show_layer_names=True,
            rankdir='TB'
        )

    def save_model(self):
        """Saves the current model to a file."""
        if self.current_model:
            self.current_model.save(self.models_dir + f"{self.current_model.name}.keras")
            with open(self.models_dir + f"{self.current_model.name}_history.pkl", "wb") as file:
                try:
                    pickle.dump(self.history.history, file)
                except AttributeError:
                    pickle.dump(self.history, file)
        else:
            self._raise_no_model_selected()

    def load_model(self):
        """Loads a model from a file and sets it as the current model."""
        if self.current_model:
            model_file_name = self.models_dir + f"{self.current_model.name}.keras"
            history_file_name = self.models_dir + f"{self.current_model.name}_history.pkl"

            start_time = time.time()
            self.current_model = load_model(model_file_name, custom_objects=self.custom_objects)
            load_time = time.time() - start_time
            print(f"Model Load Time: {load_time:.4f} seconds")

            with open(history_file_name, "rb") as file:
                self.history = pickle.load(file)

            return load_time
        else:
            self._raise_no_model_selected()

    def encoder(self, input_size):
        inputs = Input(input_size)
        backbones = {
            "resnet50": (ResNet50, ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]),
            "resnet101": (ResNet101, ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]),
            "mobilenet": (MobileNet, ["conv_pw_1_relu", "conv_pw_3_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]),
            "mobilenetv2": (MobileNetV2, ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"]),
            "efficientnetv2b0": (EfficientNetV2B0, ["stem_activation", "block2b_add", "block3b_add", "block4c_add", "block6h_add"]),
            "efficientnetv2b1": (EfficientNetV2B1, ["stem_activation", "block2c_add", "block3c_add", "block4d_add", "block6i_add"]),
            "efficientnetv2b2": (EfficientNetV2B2, ["stem_activation", "block2c_add", "block3c_add", "block4d_add", "block6j_add"]),
            "efficientnetv2b3": (EfficientNetV2B3, ["stem_activation", "block2c_add", "block3c_add", "block4e_add", "block6k_add"]),
            "efficientnetv2s": (EfficientNetV2S, ["stem_activation", "block2c_add", "block3c_add", "block4f_add", "block6o_add"]),
            "efficientnetv2m": (EfficientNetV2M, ["stem_activation", "block2c_add", "block3c_add", "block4g_add", "block6r_add"]),
        }

        # Check if the current backbone is in the dictionary
        if self.current_backbone in backbones:
            backbone, layer_names = backbones[self.current_backbone]
        else:
            # If no valid backbone specified, return None (the default model encoder will be used)
            return None, inputs

        # Initialize the backbone model without the top classification layers
        encoder_model = backbone(include_top=False, weights="imagenet", input_tensor=inputs)

        # Freeze the layers to retain pre-trained weights initially
        # for layer in encoder_model.layers[:100]:
        #     layer.trainable = False

        # Extract feature layers from the encoder
        feature_maps = [encoder_model.get_layer(name).output for name in layer_names]

        return feature_maps, inputs

    def dummy_model(self, name):
        # This returns a dummy object with just the name attribute
        return SimpleNamespace(name=name)

    def unet(self, input_size=(512, 512, 3)):
        # Get encoder feature maps or default to U-Net encoder
        feature_maps, inputs = self.encoder(input_size)

        # If no backbone is provided, use the default U-Net encoder
        if not feature_maps:
            conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
            conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
            conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
            conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
            conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
            conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)

            feature_maps = [conv1, conv2, conv3, conv4, conv5]

        # Decoder
        conv1, conv2, conv3, conv4, conv5 = feature_maps

        up6 = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
        conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
        conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
        conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
        conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

        if self.current_backbone:
            up10 = UpSampling2D(size=(2, 2))(conv9)
            conv10 = Conv2D(64, 3, activation="relu", padding="same")(up10)
            conv10 = Conv2D(64, 3, activation="relu", padding="same")(conv10)

            output = Conv2D(1, 1, activation="sigmoid")(conv10)
            model_name = f"unet_{self.current_backbone}"
            if not self.augment_data:
                model_name = f"unet_{self.current_backbone}_not_augmented"
            model = Model(inputs=inputs, outputs=output, name=model_name)
        else:
            output = Conv2D(1, 1, activation="sigmoid")(conv9)
            model_name = "unet"
            if not self.augment_data:
                model_name = f"unet_not_augmented"
            model = Model(inputs=inputs, outputs=output, name=model_name)

        return model

    def unet_plus_plus(self, input_size=(512, 512, 3)):
        # Get encoder feature maps or default to U-Net++ encoder
        feature_maps, inputs = self.encoder(input_size)

        def conv_block(x, filters):
            x = Conv2D(filters, 3, activation="relu", padding="same")(x)
            x = Conv2D(filters, 3, activation="relu", padding="same")(x)
            return x

        # If no backbone is provided, use the default U-Net++ encoder
        if not feature_maps:
            conv1 = conv_block(inputs, 64)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = conv_block(pool1, 128)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = conv_block(pool2, 256)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = conv_block(pool3, 512)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = conv_block(pool4, 1024)

            feature_maps = [conv1, conv2, conv3, conv4, conv5]

        conv1, conv2, conv3, conv4, conv5 = feature_maps

        # Decoder with nested skip connections
        up6 = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = conv_block(merge6, 512)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = conv_block(merge7, 256)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = conv_block(merge8, 128)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = conv_block(merge9, 64)

        if self.current_backbone:
            up10 = UpSampling2D(size=(2, 2))(conv9)

            output = Conv2D(1, 1, activation="sigmoid")(up10)
            model = Model(inputs=inputs, outputs=output, name=f"unet_plus_plus_{self.current_backbone}")
        else:
            output = Conv2D(1, 1, activation="sigmoid")(conv9)
            model = Model(inputs=inputs, outputs=output, name="unet_plus_plus")

        return model

    def unet_3plus(self, input_size=(512, 512, 3)):
        """
        Use a lighter backbone, batch_size=1, or mixed_precision if memory usage is too high.
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        """

        # Force a default resnet50 backbone
        if not self.current_backbone:
            self.current_backbone = "resnet50"

        feature_maps, inputs = self.encoder(input_size)

        # Encoder layers
        conv1, conv2, conv3, conv4, conv5 = feature_maps

        # Full-scale skip connections with up-sampling
        conv1_upsampled = UpSampling2D(size=(2, 2))(conv1)  # 512x512
        conv2_upsampled = UpSampling2D(size=(4, 4))(conv2)  # 512x512
        conv3_upsampled = UpSampling2D(size=(8, 8))(conv3)  # 512x512
        conv4_upsampled = UpSampling2D(size=(16, 16))(conv4)  # 512x512
        conv5_upsampled = UpSampling2D(size=(32, 32))(conv5)  # 512x512

        # Concatenate all full-scale features
        full_scale_features = concatenate([conv1_upsampled, conv2_upsampled, conv3_upsampled, conv4_upsampled, conv5_upsampled], axis=3)

        # Apply convolution to aggregate features
        conv_final = Conv2D(256, 3, activation="relu", padding="same")(full_scale_features)
        conv_final = BatchNormalization()(conv_final)
        conv_final = Conv2D(256, 3, activation="relu", padding="same")(conv_final)
        conv_final = BatchNormalization()(conv_final)
        conv_final = Conv2D(1, 1, activation="sigmoid")(conv_final)

        model = Model(inputs=inputs, outputs=conv_final, name=f"unet_3plus_{self.current_backbone}")

        return model

    def segnet(self, input_size=(512, 512, 3)):
        feature_maps, inputs = self.encoder(input_size)

        if feature_maps:
            pool1, pool2, pool3 = feature_maps[1:4]
        else:
            # If no selected backbone, use default encoder layers
            inputs = Input(input_size)

            conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
            conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
            conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
            conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Decoder
        up1 = UpSampling2D(size=(2, 2))(pool3)
        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(up1)
        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)

        up2 = UpSampling2D(size=(2, 2))(conv4)
        conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(up2)
        conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv5)

        up3 = UpSampling2D(size=(2, 2))(conv5)
        conv6 = Conv2D(64, (3, 3), activation="relu", padding="same")(up3)
        conv6 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv6)

        if self.current_backbone:
            up4 = UpSampling2D(size=(2, 2))(conv6)

            outputs = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(up4)
            model = Model(inputs=[inputs], outputs=[outputs], name=f"segnet_{self.current_backbone}")
        else:
            outputs = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv6)
            model = Model(inputs=[inputs], outputs=[outputs], name="segnet")

        return model

    def mask_rcnn(self, input_size=(512, 512, 3)):
        # Force a default resnet50 backbone
        if not self.current_backbone:
            self.current_backbone = "resnet50"

        feature_maps, inputs = self.encoder(input_size)

        # Encoder layers
        conv1, conv2, conv3, conv4, _ = feature_maps

        # FPN to enhance multi-scale feature extraction
        fpn4 = Conv2D(256, (1, 1), padding="same")(conv4)
        fpn3 = Add()([Conv2D(256, (1, 1), padding="same")(conv3), UpSampling2D(size=(2, 2))(fpn4)])
        fpn2 = Add()([Conv2D(256, (1, 1), padding="same")(conv2), UpSampling2D(size=(2, 2))(fpn3)])
        fpn1 = Add()([Conv2D(256, (1, 1), padding="same")(conv1), UpSampling2D(size=(2, 2))(fpn2)])

        # Decoder for Segmentation Masks
        up1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(fpn4)
        up1 = concatenate([up1, fpn3])

        up2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(up1)
        up2 = concatenate([up2, fpn2])

        up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(up2)
        up3 = concatenate([up3, fpn1])

        # Final upsampling to reach (512, 512)
        up4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(up3)
        # up4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(up4)

        # Output layer for Mask
        output = Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(up4)

        # Create model
        model = Model(inputs=inputs, outputs=output, name=f"mask_rcnn_{self.current_backbone}")

        return model

    def mask_rcnn_custom(self, input_size=(512, 512, 3)):
        """
        Custom version using bilinear interpolation for better feature alignment
        """
        # Force a default resnet50 backbone
        if not self.current_backbone:
            self.current_backbone = "resnet50"

        feature_maps, inputs = self.encoder(input_size)

        # Encoder layers
        conv1, conv2, conv3, conv4, _ = feature_maps

        # Adjust the FPN to ensure correct shapes
        fpn4 = Conv2D(256, (1, 1), padding="same")(conv4)

        # Upsample fpn4 to match conv3
        up_fpn4 = UpSampling2D(size=(conv3.shape[1] // fpn4.shape[1], conv3.shape[2] // fpn4.shape[2]), interpolation="bilinear")(fpn4)
        fpn3 = Add()([Conv2D(256, (1, 1), padding="same")(conv3), up_fpn4])

        # Upsample fpn3 to match conv2
        up_fpn3 = UpSampling2D(size=(conv2.shape[1] // fpn3.shape[1], conv2.shape[2] // fpn3.shape[2]), interpolation="bilinear")(fpn3)
        fpn2 = Add()([Conv2D(256, (1, 1), padding="same")(conv2), up_fpn3])

        # Upsample fpn2 to match conv1
        up_fpn2 = UpSampling2D(size=(conv1.shape[1] // fpn2.shape[1], conv1.shape[2] // fpn2.shape[2]), interpolation="bilinear")(fpn2)
        fpn1 = Add()([Conv2D(256, (1, 1), padding="same")(conv1), up_fpn2])

        # Decoder for Segmentation Masks
        up1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(fpn4)
        up1 = concatenate([up1, fpn3])

        up2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(up1)

        # Upsample fpn2 to match up2
        up_fpn2_up2 = UpSampling2D(size=(up2.shape[1] // fpn2.shape[1], up2.shape[2] // fpn2.shape[2]), interpolation="bilinear")(fpn2)
        up2 = concatenate([up2, up_fpn2_up2])

        up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(up2)

        # Upsample fpn1 to match up3
        up_fpn1_up3 = UpSampling2D(size=(up3.shape[1] // fpn1.shape[1], up3.shape[2] // fpn1.shape[2]), interpolation="bilinear")(fpn1)
        up3 = concatenate([up3, up_fpn1_up3])

        # Final upsampling to reach (512, 512) - Ensure only upsampling to the target size
        up4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(up3)

        # Only upsample once to reach the target size (512, 512)
        up4 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding="same", activation="relu")(up4)

        # Output layer for Mask
        output = Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(up4)

        # Create model
        model = Model(inputs=inputs, outputs=output, name=f"mask_rcnn_custom_{self.current_backbone}")

        return model

    def linknet(self, input_size=(512, 512, 3)):
        # Force a default resnet50 backbone
        if not self.current_backbone:
            self.current_backbone = "resnet50"

        feature_maps, inputs = self.encoder(input_size)

        # Encoder layers
        conv1, conv2, conv3, conv4, conv5 = feature_maps

        # Decoder
        def decoder_block(input_tensor, skip_tensor, num_filters):
            x = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(input_tensor)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            skip_tensor = Conv2D(num_filters, kernel_size=1, padding="same")(skip_tensor)

            x = Add()([x, skip_tensor])

            x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        # Decode the ResNet features
        up6 = decoder_block(conv5, conv4, 512)  # 32x32
        up7 = decoder_block(up6, conv3, 256)  # 64x64
        up8 = decoder_block(up7, conv2, 128)  # 128x128
        up9 = decoder_block(up8, conv1, 64)  # 256x256

        # Final upsampling to match the input size
        up10 = UpSampling2D(size=(2, 2))(up9)  # 512x512
        final_conv = Conv2D(32, 3, padding="same", activation="relu")(up10)
        final_conv = Conv2D(32, 3, padding="same", activation="relu")(final_conv)

        outputs = Conv2D(1, 1, activation="sigmoid")(final_conv)

        model = Model(inputs=inputs, outputs=outputs, name=f"linknet_{self.current_backbone}")

        return model

    def deep_lab_v3(self, input_size=(512, 512, 3)):
        # Force a default resnet50 backbone
        if not self.current_backbone:
            self.current_backbone = "resnet50"

        feature_maps, inputs = self.encoder(input_size)

        # Encoder layers
        _, _, _, _, conv5 = feature_maps

        # ASPP (Atrous Spatial Pyramid Pooling)
        x = Conv2D(256, (1, 1), dilation_rate=(1, 1), padding="same")(conv5)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Add a series of dilated convolutions
        dilation_rates = [3, 6, 9]
        for rate in dilation_rates:
            x_dilated = Conv2D(256, (3, 3), dilation_rate=(rate, rate), padding="same")(conv5)
            x_dilated = BatchNormalization()(x_dilated)
            x_dilated = Activation("relu")(x_dilated)
            x = tf.keras.layers.Concatenate()([x, x_dilated])

        # ASPP feature map needs to be upsampled to match the original input shape
        x = Conv2D(256, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Gradual upsampling to match input resolution
        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Apply a final convolutional layer to predict the segmentation mask
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
        x = Conv2D(1, (1, 1), padding="same")(x)

        # Apply a sigmoid activation for binary classification
        outputs = Activation("sigmoid")(x)

        # Create model
        model = Model(inputs, outputs, name=f"deep_lab_v3_{self.current_backbone}")

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, help='Name of the backbone to use')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='Batch size (reduce to save memory) [default: %(default)s]')
    parser.add_argument('--epochs', type=int, default=50, required=False, help='Training epochs (will stop early based on metrics [default: %(default)s]')
    parser.add_argument('--testing_mode', action='store_true', help='Enable testing mode to measure load, inference times, and iou score')
    parser.add_argument('--output_file', type=str, default="model_performance.json", help='Output file to log load and inference times')
    parser.add_argument('--augment_data', action='store_false', help='Should input images be augmented')
    args = parser.parse_args()

    m = ModelManager(args.batch_size, args.augment_data)
    if args.backbone:
        m.select_backbone(args.backbone)
    m.select_model(args.model)
    if args.testing_mode:
        m.evaluate_model_performance(args.output_file)
    else:
        m.train(args.epochs)
        m.save_model()
