import pandas as pd

# Create a DataFrame to display results
def display_results(cv_means, cv_stds, val_scores):
    '''
    Example usage:
    df = display_results(cv_means, cv_stds, val_scores)
    print(df)
    '''
    df = pd.DataFrame({
        "CV Mean Accuracy": cv_means,
        "CV Std": cv_stds,
        "Validation Accuracy": val_scores
    })
 
    return df