# Applied Machine Learning Project

This project focuses on building a wildlife image classifier using MobileNetV2 and TensorFlow. The model is trained to classify images into different wildlife categories.

## Project Structure

- `build_model.py`: Script to build, train, evaluate, and save the model.
- `wildlife_dataset/`: Directory containing the training, validation, and test datasets.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas

You can install the required packages using the following command:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

## Dataset

The dataset should be organized into the following structure:

```
wildlife_dataset/
    train/
        class1/
        class2/
        ...
    val/
        class1/
        class2/
        ...
    test/
        class1/
        class2/
        ...
```

## Running the Project

1. Ensure the dataset is in the correct structure as mentioned above.
2. Run the `build_model.py` script to train and evaluate the model:

```bash
python build_model.py
```

## Output

- The script will print the classification report and display the confusion matrix.
- The trained model will be saved as `wildlife_classifier.h5`.
- The best model during training will be saved as `best_wildlife_classifier.h5`.
- Training and validation accuracy plots will be displayed.
