# Retnify: Retinal Disease Classification Using Deep Learning
# Group members
- Idil
- Ej
- Wu
- Gaurav

## Overview

Retnify is a deep learning project focused on classifying retinal diseases using OCT (Optical Coherence Tomography) images. The project leverages state-of-the-art convolutional neural networks (CNNs) such as SimpleCNN, VGG16, ResNet-18, and ResNet-50 with Squeeze-and-Excitation (SE) blocks to achieve high accuracy in identifying four retinal conditions:

- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

The project demonstrates the application of transfer learning, data augmentation, and advanced visualization techniques like Grad-CAM to ensure robust and interpretable results.

---

## Dataset

The project uses the **OCT2017** dataset, which contains retinal OCT images categorized into four classes. The dataset is split into training, validation, and test sets:

- **Training Set**: Imbalanced with varying class distributions.
- **Validation Set**: Created using an 80/20 split from the training set.
- **Test Set**: Balanced with equal samples across all classes.

### Dataset Statistics

- **Training Set**: Significant class imbalance observed.
- **Test Set**: Perfectly balanced with 250 samples per class.

---

## Methodology

### Preprocessing

1. **Image Resizing**: All images resized to 224x224 pixels to match input requirements of CNN architectures.
2. **Data Augmentation**: Applied to the training set to improve generalization:
   - Horizontal flipping
   - Random rotations
   - Normalization

### Models

1. **SimpleCNN**: A custom CNN architecture with three convolutional layers and two fully connected layers.
2. **VGG16**: A pre-trained model fine-tuned for the OCT dataset.
3. **ResNet-18**: A residual network fine-tuned end-to-end.
4. **ResNet-50 with SE Blocks**: Enhanced with Squeeze-and-Excitation blocks for better feature recalibration.

### Training

- **Loss Function**: CrossEntropyLoss with class weights to handle imbalance.
- **Optimizer**: Adam optimizer with learning rates tuned for each model.
- **Metrics**: Accuracy, F1-score, and loss tracked for both training and validation phases.

---

## Results

### Key Metrics

- **SimpleCNN**: Achieved ~92.15% accuracy but struggled with class imbalance.
- **VGG16**: Reached 97.08% validation accuracy but showed signs of overfitting.
- **ResNet-18**: Balanced performance with ~97% validation accuracy.
- **ResNet-50 with SE Blocks**: Best performance with ~97.21% validation accuracy and minimal overfitting.

### Visualizations

1. **Class Distribution**: Bar graphs showing imbalanced training set and balanced test set.
2. **Training Curves**: Loss and accuracy plots for all models.
3. **Confusion Matrices**: Highlighted minimal misclassifications for ResNet-50.
4. **Grad-CAM**: Visualized model focus areas to ensure interpretability.

---

## Conclusion

### Key Insights

- **Best Model**: ResNet-50 with SE Blocks demonstrated the highest accuracy and generalization.
- **Challenges**: Addressing class imbalance was critical for improving model performance.
- **Future Work**: Explore additional attention mechanisms and larger datasets for further improvements.

### Final Submission

The ResNet-50 model will be submitted as the final model due to its superior performance and robustness.

---

## How to Run

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- PyTorch
- torchvision
- pandas
- matplotlib
- seaborn
- tqdm
- PIL (Pillow)
- scikit-learn
- Grad-CAM


### Steps to Run the Project

1. Clone the repository to your local machine.
2. Install the required Python libraries found in the first block in the notebook:

```python
# %pip install torch torchvision
# %pip install pandas
# %pip install matplotlib
# %pip install seaborn
# %pip install pillow
# %pip install tqdm
# %pip install scikit-learn
# %pip install grad-cam
```
3. Ensure the OCT2017 dataset has been downloaded and extracted into the expected directory structure referenced in the notebook.
4. Follow the cells in the notebook to preprocess the dataset, train the models, and evaluate the results.

## Notes

- The dataset used is the OCT2017 dataset. Ensure the dataset is downloaded and placed in the appropriate directory as specified in the notebook.
- The notebook includes preprocessing steps such as resizing, normalization, and data augmentation.
- Models implemented include SimpleCNN, VGG16, and ResNet-50 with SE Blocks.
- Results include metrics like accuracy, F1-score, and visualizations such as confusion matrices and Grad-CAM outputs.

---

## Acknowledgments

- **Dataset**: OCT2017 dataset.
- **Libraries**: PyTorch, torchvision, Grad-CAM.
- **Contributors**: Special thanks to the team for their efforts in developing and fine-tuning the models.

---

## References

1. [OCT2017 Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
3. [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
