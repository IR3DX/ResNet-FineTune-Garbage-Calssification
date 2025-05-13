# Garbage Classification with ResNet-18

## Project Overview

This project fine-tunes a **ResNet-18** neural network to classify images of different types of garbage into six categories: **Metal**, **Glass**, **Trash**, **Plastic**, **Paper**, and **Cardboard**. The model achieves a high accuracy of approximately **95%** on the test set and includes a **Gradio** interface for testing on unseen data

### Key Points:
- **Dataset**: Garbage classification dataset containing images labeled by type.
- **Model**: ResNet-18, fine-tuned on the garbage classification dataset.
- **Accuracy**: ~95% on the test set.
- **Challenge**: The model may misclassify objects due to irrelevant background features (e.g., a metal item on a brown table being misclassified as cardboard).
- **Future Work**: The introduction of a **separation model** to isolate the object from the background could improve the accuracy further.

## Requirements

To run the project, you'll need the following libraries:

- `torch`
- `torchvision`
- `kagglehub`
- `gradio`
- `sklearn`
- `PIL`

## Challenges & Future Work
### 1. Background Interference
One of the key challenges encountered during training was that the model sometimes misclassified objects because of the background, as seen in the case of metal objects placed on a brown table being classified as cardboard. The model picked up on the background color rather than the object itself.

###Possible Solution:
A potential improvement could be to use image segmentation techniques, like a separation model, to isolate the object from the background. This could significantly improve classification accuracy.

### 2. Fine-tuning the Model
In the current setup, ResNet-18 was fine-tuned with a low learning rate (0.0001) that was slowly adjusted to fit from a learning rate of (0.001), which worked well. However, hyperparameter tuning could be explored to further optimize model performance.

### 3. Data Augmentation
Although the model achieves a good accuracy, additional data augmentation (e.g., random cropping, flipping, and color jitter) could be applied to improve the robustness of the model and reduce overfitting.

## Conclusion
This project demonstrates how to use a ResNet-18 model for garbage classification with a high accuracy of around 95% on a test set. However, background interference remains an issue, and future work can focus on improving this aspect through segmentation techniques.
