# Modeling Approach

We plan to use a 90/10 train test split with k-fold validation.  Missing variables will be handled with imputation.  The free text field of chief complaint will be fed through an LLM to extract a numerical feature vector.  Numerical variables will be standardized.  Variables will be removed until the maximum correlation is less than .95.

## Models

For the sake of interpretability we will use the following models:

1. Linear regression on all features without interaction terms
    1. Without regularization
    2. With ridge regression
    3. With lasso regression
2. Linear regression with all second order terms
    1. Without regularization
    2. With ridge regression
    3. With lasso regression
3. A decision tree model 

# Evaluation

Models will be evaluated based on RMSE and interpretability.