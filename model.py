from scipy.stats import mode
import pandas as pd
import numpy as np

def feature_df(filename): 
    data = pd.read_csv(filename)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Group by user ID
    grouped = data.groupby('id')

    # Initialize a list to store results
    results = []

    # Iterate over each user group
    for user_id, group in grouped:
        # Calculate Deposit/Withdraw & Buy/Sell Ratio
        num_deposit_withdraw = group[group['trade_type'].isin(['deposit', 'withdraw'])].shape[0]
        num_buy_sell = group[group['trade_type'].isin(['buy', 'sell'])].shape[0]
        deposit_withdraw_buy_sell_ratio = num_buy_sell / num_deposit_withdraw if num_buy_sell != 0 else 1

        # Calculate Time Between Buy & Sell Trades Mode & Variance
        buy_sell_trades = group[group['trade_type'].isin(['buy', 'sell'])]
        time_diffs = buy_sell_trades['timestamp'].diff().dt.total_seconds().dropna().iloc[::2]
        time_diff_mode = mode(time_diffs)[0] if not time_diffs.empty else 0
        time_diff_variance = time_diffs.var() if not time_diffs.empty else 0
        
        # Calculate Buy Volumes Variance
        buy_sell_volumes_variance = buy_sell_trades['amount'].diff().dropna().iloc[::2].var() if not buy_sell_trades.empty else 0
        
        # Calculate Profit/Loss (Average Percentage)
        profits = []
        for i in range(0, len(buy_sell_trades) - 1, 2):
            if buy_sell_trades.iloc[i]['trade_type'] == 'buy' and buy_sell_trades.iloc[i + 1]['trade_type'] == 'sell':
                buy_amount = buy_sell_trades.iloc[i]['amount']
                sell_amount = buy_sell_trades.iloc[i + 1]['amount']
                profit = (sell_amount - buy_amount) / buy_amount * 100
                profits.append(profit)
        average_profit_loss_percentage = sum(profits) / len(profits) if profits else 0
        
        # Append results
        if user_id < int(1000 * 0.85):
            results.append({
                'deposit_withdraw_buy_sell_ratio': deposit_withdraw_buy_sell_ratio,
                'time_diff_mode': time_diff_mode,
                'time_diff_variance': 0 if np.isnan(time_diff_variance) else time_diff_variance,
                'bsvvariance': 0 if np.isnan(buy_sell_volumes_variance) else time_diff_variance,
                'average_profit_loss_percentage': average_profit_loss_percentage,
                'is_fraud': 0
            })
        else:
            results.append({
                'deposit_withdraw_buy_sell_ratio': deposit_withdraw_buy_sell_ratio,
                'time_diff_mode': time_diff_mode,
                'time_diff_variance': 0 if np.isnan(time_diff_variance) else time_diff_variance,
                'bsvvariance':  0 if np.isnan(buy_sell_volumes_variance) else time_diff_variance,
                'average_profit_loss_percentage': average_profit_loss_percentage,
                'is_fraud': 1
            })
    return results    

def new_data_df(filename): 
    data = pd.read_csv(filename)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Group by user ID
    grouped = data.groupby('id')

    # Initialize a list to store results
    results = []

    # Iterate over each user group
    for user_id, group in grouped:
        # Calculate Deposit/Withdraw & Buy/Sell Ratio
        num_deposit_withdraw = group[group['trade_type'].isin(['deposit', 'withdraw'])].shape[0]
        num_buy_sell = group[group['trade_type'].isin(['buy', 'sell'])].shape[0]
        deposit_withdraw_buy_sell_ratio = num_buy_sell / num_deposit_withdraw if num_buy_sell != 0 else 1

        # Calculate Time Between Buy & Sell Trades Mode & Variance
        buy_sell_trades = group[group['trade_type'].isin(['buy', 'sell'])]
        time_diffs = buy_sell_trades['timestamp'].diff().dt.total_seconds().dropna().iloc[::2]
        time_diff_mode = mode(time_diffs)[0] if not time_diffs.empty else 0
        time_diff_variance = time_diffs.var() if not time_diffs.empty else 0
        
        # Calculate Buy Volumes Variance
        buy_sell_volumes_variance = buy_sell_trades['amount'].diff().dropna().iloc[::2].var() if not buy_sell_trades.empty else 0
        
        # Calculate Profit/Loss (Average Percentage)
        profits = []
        for i in range(0, len(buy_sell_trades) - 1, 2):
            if buy_sell_trades.iloc[i]['trade_type'] == 'buy' and buy_sell_trades.iloc[i + 1]['trade_type'] == 'sell':
                buy_amount = buy_sell_trades.iloc[i]['amount']
                sell_amount = buy_sell_trades.iloc[i + 1]['amount']
                profit = (sell_amount - buy_amount) / buy_amount * 100
                profits.append(profit)
        average_profit_loss_percentage = sum(profits) / len(profits) if profits else 0
        
        # Append results
        if user_id < int(1000 * 0.85):
            results.append({
                'deposit_withdraw_buy_sell_ratio': deposit_withdraw_buy_sell_ratio,
                'time_diff_mode': time_diff_mode,
                'time_diff_variance': 0 if np.isnan(time_diff_variance) else time_diff_variance,
                'bsvvariance': 0 if np.isnan(buy_sell_volumes_variance) else time_diff_variance,
                'average_profit_loss_percentage': average_profit_loss_percentage,
            })
    return results    

# Convert results to DataFrame
features_df = pd.DataFrame(feature_df("mock_trade_data.csv"))

# Display the features DataFrame
# print(features_df)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score

## PLEASE REPLACE THIS DATA WITH YOUR DATA YA
# Set features and target variable
features = ['deposit_withdraw_buy_sell_ratio', 'time_diff_mode', 'time_diff_variance', 'bsvvariance', 'average_profit_loss_percentage']

X = features_df[features].values  # Use square brackets to extract columns by their names

y = features_df['is_fraud'].values  # Target variable - is_fraudulent

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instatiate StandardScaler and fit the training set
ss = StandardScaler()
ss.fit(X_train)

X_train_sc = ss.transform(X_train)
X_test_sc = ss.transform(X_test)

# EXAMPLE ALGORITHM - replace with suitable algorithm
# Initialize the logistic regression model
logreg_model = LogisticRegression(random_state=42)

# Fit the Logistic Regression model
logreg_model.fit(X_train_sc, y_train)

# train score
# print(logreg_model.score(X_train_sc, y_train))

# test score
# print(logreg_model.score(X_test_sc, y_test))

# Predict on the test set
y_pred = logreg_model.predict(X_test_sc)

y_confidence = logreg_model.predict_proba(X_test_sc)

print(logreg_model.predict_proba(ss.transform(pd.DataFrame(new_data_df("test_data.csv")))))
# input()
# prediction result
# print(y_pred)

# check cross validation score
# print(cross_val_score(logreg_model, X_train_sc, y_train, cv = 5).mean())

#Interpretation - if the train score is higher than the test score, suggesting an overfitting. 
#The train, test, and cross-validation scores are very close, indicating good generalization and minimal overfitting or underfitting.

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, logreg_model.predict_proba(X_test)[:, 1])

# print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 Score: {f1:.2f}')
# print(f'ROC AUC: {roc_auc:.2f}')

# rf_model = RandomForestClassifier(random_state=42)

# # Fit the Logistic Regression model
# rf_model.fit(X_train_sc, y_train)

# # train score
# print(rf_model.score(X_train_sc, y_train))

# # test score
# print(rf_model.score(X_test_sc, y_test))

# # Predict on the test set
# y_pred = rf_model.predict(X_test_sc)

# y_confidence = rf_model.predict_proba(X_test_sc)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, logreg_model.predict_proba(X_test)[:, 1])

# print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 Score: {f1:.2f}')
# print(f'ROC AUC: {roc_auc:.2f}')


# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming you have your model predictions
# def plot_confusion_matrix(y_true, y_pred, model_name="logreg_model"):
#     # Calculate confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Create labels for the plot
#     labels = ['Not Fraudulent', 'Fraudulent']
    
#     # Create a figure
#     plt.figure(figsize=(8, 6))
    
#     # Plot confusion matrix as a heatmap
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=labels,
#                 yticklabels=labels)
    
#     plt.title(f'Confusion Matrix - {model_name}')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
    
#     # Add text annotations explaining each quadrant
#     plt.text(-0.4, -0.4, 
#              f"""
#              True Negatives (TN): {cm[0][0]}
#              False Positives (FP): {cm[0][1]}
#              False Negatives (FN): {cm[1][0]}
#              True Positives (TP): {cm[1][1]}
#              """,
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
#     plt.show()
    
#     # Calculate and print metrics
#     tn, fp, fn, tp = cm.ravel()
    
#     # False Positive Rate (FPR) = FP / (FP + TN)
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
#     # Precision = TP / (TP + FP)
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
#     # Recall = TP / (TP + FN)
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
#     print(f"\nMetrics for {model_name}:")
#     print(f"False Positive Rate: {fpr:.2%}")
#     print(f"Precision: {precision:.2%}")
#     print(f"Recall: {recall:.2%}")
#     print("\nInterpretation:")
#     print(f"- Out of all non-fraudulent cases, {fpr:.2%} were incorrectly flagged as fraudulent")
#     print(f"- Out of all cases flagged as fraudulent, {precision:.2%} were actually fraudulent")
#     print(f"- Out of all actual fraudulent cases, {recall:.2%} were correctly identified")

# # Use the function with your model
# if len(X) > 10:  # If we have enough samples for train-test split
#     for name, model_dict in results.items():
#         model = model_dict['model']
#         y_pred = model.predict(X_test_scaled)
#         print(f"\n{'-'*50}")
#         plot_confusion_matrix(y_test, y_pred, name)
# else:
#     # For small dataset case
#     y_pred = model.predict(X_scaled)
#     plot_confusion_matrix(y, y_pred, "Single Model")
