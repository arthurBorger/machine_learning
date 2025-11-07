import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_model_evaluation(model, x_test, y_test, history, exercise_title):
    """
    Plot confusion matrix and training/validation loss for model evaluation.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model to evaluate
    x_test : numpy.ndarray
        Test data features
    y_test : numpy.ndarray
        Test data labels
    history : keras.callbacks.History
        Training history object from model.fit()
    exercise_title : str
        Title for the plots (e.g., "Exercise 3: Alpha Dropout")
    """
    # Generate predictions and confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred)

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Confusion Matrix ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    # --- Plot 2: Training and Validation Loss ---
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Training and Validation Loss - {exercise_title}')
    axes[1].legend()
    axes[1].grid(True)

    # Adjust spacing
    plt.tight_layout()
    plt.show()