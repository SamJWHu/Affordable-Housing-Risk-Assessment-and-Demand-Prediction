# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:17:36 2024

@author: SamJWHu
"""

# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Define Affordable Housing Options
housing_options = [
    'Modular and Prefabricated Housing',
    'Inclusionary Zoning Policies',
    'Affordable Housing Bonds',
    'Public-Private Partnerships (PPPs)',
    'Rent-to-Own Schemes',
    'Community Land Trusts (CLTs)',
    'Co-Living Spaces',
    'Slum Upgrading Programs',
    'Microfinance for Housing'
]

# Step 2: Generate Synthetic Dataset with Rating Agency Perspective
np.random.seed(42)
samples_per_option = 55
data = []

for option in housing_options:
    for _ in range(samples_per_option):
        # Simulate features based on option characteristics
        if option == 'Modular and Prefabricated Housing':
            cost_per_unit = np.random.normal(50000, 5000)
            construction_time = np.random.normal(6, 1)
            environmental_impact = np.random.uniform(0.8, 0.95)
            policy_support = np.random.uniform(0.7, 0.95)
            social_acceptance = np.random.uniform(0.7, 0.9)
            economic_conditions = np.random.uniform(0.7, 0.95)
            market_risk = np.random.uniform(0.1, 0.3)
            investor_return = np.random.uniform(0.08, 0.12)
            scalability = np.random.uniform(0.85, 0.98)
            demand_satisfaction = np.random.uniform(0.8, 0.95)
            # Risk assessment features
            default_risk = np.random.uniform(0.01, 0.03)
            regulatory_compliance = np.random.uniform(0.9, 1.0)
            liquidity_risk = np.random.uniform(0.05, 0.1)
        elif option == 'Inclusionary Zoning Policies':
            cost_per_unit = np.random.normal(60000, 7000)
            construction_time = np.random.normal(9, 2)
            environmental_impact = np.random.uniform(0.6, 0.8)
            policy_support = np.random.uniform(0.8, 0.98)
            social_acceptance = np.random.uniform(0.6, 0.8)
            economic_conditions = np.random.uniform(0.7, 0.9)
            market_risk = np.random.uniform(0.2, 0.4)
            investor_return = np.random.uniform(0.06, 0.1)
            scalability = np.random.uniform(0.75, 0.9)
            demand_satisfaction = np.random.uniform(0.7, 0.85)
            default_risk = np.random.uniform(0.02, 0.05)
            regulatory_compliance = np.random.uniform(0.85, 0.95)
            liquidity_risk = np.random.uniform(0.1, 0.15)
        elif option == 'Affordable Housing Bonds':
            cost_per_unit = np.random.normal(55000, 6000)
            construction_time = np.random.normal(8, 1.5)
            environmental_impact = np.random.uniform(0.7, 0.85)
            policy_support = np.random.uniform(0.85, 0.98)
            social_acceptance = np.random.uniform(0.7, 0.85)
            economic_conditions = np.random.uniform(0.8, 0.95)
            market_risk = np.random.uniform(0.15, 0.35)
            investor_return = np.random.uniform(0.07, 0.11)
            scalability = np.random.uniform(0.8, 0.95)
            demand_satisfaction = np.random.uniform(0.75, 0.9)
            default_risk = np.random.uniform(0.015, 0.04)
            regulatory_compliance = np.random.uniform(0.88, 0.97)
            liquidity_risk = np.random.uniform(0.08, 0.12)
        elif option == 'Public-Private Partnerships (PPPs)':
            cost_per_unit = np.random.normal(53000, 6000)
            construction_time = np.random.normal(7, 1.5)
            environmental_impact = np.random.uniform(0.75, 0.9)
            policy_support = np.random.uniform(0.85, 0.99)
            social_acceptance = np.random.uniform(0.75, 0.9)
            economic_conditions = np.random.uniform(0.8, 0.95)
            market_risk = np.random.uniform(0.1, 0.3)
            investor_return = np.random.uniform(0.08, 0.12)
            scalability = np.random.uniform(0.85, 0.98)
            demand_satisfaction = np.random.uniform(0.8, 0.95)
            default_risk = np.random.uniform(0.01, 0.03)
            regulatory_compliance = np.random.uniform(0.9, 0.99)
            liquidity_risk = np.random.uniform(0.05, 0.1)
        elif option == 'Rent-to-Own Schemes':
            cost_per_unit = np.random.normal(62000, 7000)
            construction_time = np.random.normal(9, 2)
            environmental_impact = np.random.uniform(0.65, 0.8)
            policy_support = np.random.uniform(0.7, 0.9)
            social_acceptance = np.random.uniform(0.7, 0.85)
            economic_conditions = np.random.uniform(0.75, 0.9)
            market_risk = np.random.uniform(0.2, 0.4)
            investor_return = np.random.uniform(0.07, 0.1)
            scalability = np.random.uniform(0.75, 0.9)
            demand_satisfaction = np.random.uniform(0.75, 0.9)
            default_risk = np.random.uniform(0.025, 0.05)
            regulatory_compliance = np.random.uniform(0.8, 0.9)
            liquidity_risk = np.random.uniform(0.1, 0.15)
        elif option == 'Community Land Trusts (CLTs)':
            cost_per_unit = np.random.normal(48000, 5000)
            construction_time = np.random.normal(10, 2)
            environmental_impact = np.random.uniform(0.8, 0.95)
            policy_support = np.random.uniform(0.6, 0.85)
            social_acceptance = np.random.uniform(0.8, 0.95)
            economic_conditions = np.random.uniform(0.7, 0.85)
            market_risk = np.random.uniform(0.05, 0.2)
            investor_return = np.random.uniform(0.04, 0.08)
            scalability = np.random.uniform(0.6, 0.8)
            demand_satisfaction = np.random.uniform(0.65, 0.8)
            default_risk = np.random.uniform(0.02, 0.045)
            regulatory_compliance = np.random.uniform(0.75, 0.9)
            liquidity_risk = np.random.uniform(0.12, 0.18)
        elif option == 'Co-Living Spaces':
            cost_per_unit = np.random.normal(70000, 8000)
            construction_time = np.random.normal(8, 1.5)
            environmental_impact = np.random.uniform(0.6, 0.75)
            policy_support = np.random.uniform(0.6, 0.8)
            social_acceptance = np.random.uniform(0.7, 0.85)
            economic_conditions = np.random.uniform(0.7, 0.9)
            market_risk = np.random.uniform(0.2, 0.4)
            investor_return = np.random.uniform(0.06, 0.09)
            scalability = np.random.uniform(0.7, 0.85)
            demand_satisfaction = np.random.uniform(0.6, 0.8)
            default_risk = np.random.uniform(0.03, 0.06)
            regulatory_compliance = np.random.uniform(0.7, 0.85)
            liquidity_risk = np.random.uniform(0.15, 0.2)
        elif option == 'Slum Upgrading Programs':
            cost_per_unit = np.random.normal(40000, 5000)
            construction_time = np.random.normal(12, 2)
            environmental_impact = np.random.uniform(0.5, 0.7)
            policy_support = np.random.uniform(0.5, 0.75)
            social_acceptance = np.random.uniform(0.6, 0.8)
            economic_conditions = np.random.uniform(0.5, 0.7)
            market_risk = np.random.uniform(0.3, 0.5)
            investor_return = np.random.uniform(0.03, 0.06)
            scalability = np.random.uniform(0.65, 0.8)
            demand_satisfaction = np.random.uniform(0.6, 0.8)
            default_risk = np.random.uniform(0.035, 0.065)
            regulatory_compliance = np.random.uniform(0.6, 0.8)
            liquidity_risk = np.random.uniform(0.18, 0.25)
        elif option == 'Microfinance for Housing':
            cost_per_unit = np.random.normal(45000, 5000)
            construction_time = np.random.normal(11, 2)
            environmental_impact = np.random.uniform(0.6, 0.75)
            policy_support = np.random.uniform(0.6, 0.85)
            social_acceptance = np.random.uniform(0.7, 0.85)
            economic_conditions = np.random.uniform(0.6, 0.8)
            market_risk = np.random.uniform(0.25, 0.45)
            investor_return = np.random.uniform(0.04, 0.08)
            scalability = np.random.uniform(0.7, 0.85)
            demand_satisfaction = np.random.uniform(0.65, 0.8)
            default_risk = np.random.uniform(0.03, 0.055)
            regulatory_compliance = np.random.uniform(0.75, 0.9)
            liquidity_risk = np.random.uniform(0.12, 0.18)
        
        # Calculate Overall Risk Score (simplified for demonstration)
        overall_risk_score = (
            default_risk * 0.5 +
            market_risk * 0.3 +
            liquidity_risk * 0.2
        )
        
        # Assign Rating Grade based on Overall Risk Score
        if overall_risk_score <= 0.05:
            rating_grade = 'AAA'
        elif overall_risk_score <= 0.07:
            rating_grade = 'AA'
        elif overall_risk_score <= 0.09:
            rating_grade = 'A'
        elif overall_risk_score <= 0.12:
            rating_grade = 'BBB'
        elif overall_risk_score <= 0.15:
            rating_grade = 'BB'
        elif overall_risk_score <= 0.18:
            rating_grade = 'B'
        else:
            rating_grade = 'CCC'
        
        data.append({
            'Option': option,
            'Cost_Per_Unit': cost_per_unit,
            'Construction_Time': construction_time,
            'Environmental_Impact': environmental_impact,
            'Policy_Support': policy_support,
            'Social_Acceptance': social_acceptance,
            'Economic_Conditions': economic_conditions,
            'Market_Risk': market_risk,
            'Investor_Return': investor_return,
            'Scalability': scalability,
            'Demand_Satisfaction': demand_satisfaction,
            'Default_Risk': default_risk,
            'Regulatory_Compliance': regulatory_compliance,
            'Liquidity_Risk': liquidity_risk,
            'Overall_Risk_Score': overall_risk_score,
            'Rating_Grade': rating_grade
        })
    
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Data Validation
print("Statistical Summary of the Generated Data:")
print(df.describe())

# Step 4: Feature Engineering
# Encode 'Option' and 'Rating_Grade' using One-Hot Encoding and Label Encoding respectively
df_encoded = pd.get_dummies(df, columns=['Option'])

# Label Encode 'Rating_Grade'
label_encoder = LabelEncoder()
df_encoded['Rating_Grade_Label'] = label_encoder.fit_transform(df_encoded['Rating_Grade'])

numeric_features = [
    'Cost_Per_Unit', 'Construction_Time', 'Environmental_Impact',
    'Policy_Support', 'Social_Acceptance', 'Economic_Conditions',
    'Market_Risk', 'Investor_Return', 'Scalability',
    'Demand_Satisfaction', 'Default_Risk', 'Regulatory_Compliance',
    'Liquidity_Risk', 'Overall_Risk_Score'
]

scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Step 5: Data Splitting with Indices
X = df_encoded.drop(columns=['Demand_Satisfaction', 'Rating_Grade', 'Rating_Grade_Label'])
y_regression = df_encoded['Demand_Satisfaction'].values
y_classification = df_encoded['Rating_Grade_Label'].values

# Create an array of indices
indices = df_encoded.index.values

# First split: train+val and test sets
X_train_val, X_test, y_reg_train_val, y_reg_test, y_clf_train_val, y_clf_test, idx_train_val, idx_test = train_test_split(
    X, y_regression, y_classification, indices, test_size=0.2, random_state=42
)

# Second split: train and validation sets
X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val, idx_train, idx_val = train_test_split(
    X_train_val, y_reg_train_val, y_clf_train_val, idx_train_val, test_size=0.25, random_state=42
)

# Convert to NumPy arrays and ensure dtype is np.float32
X_train = X_train.values.astype(np.float32)
X_val = X_val.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)
y_reg_train = y_reg_train.astype(np.float32)
y_reg_val = y_reg_val.astype(np.float32)
y_reg_test = y_reg_test.astype(np.float32)
y_clf_train = y_clf_train.astype(np.int32)
y_clf_val = y_clf_val.astype(np.int32)
y_clf_test = y_clf_test.astype(np.int32)

# Step 6: Extract Constraint Variables
investor_return_index = list(X.columns).index('Investor_Return')
market_risk_index = list(X.columns).index('Market_Risk')
scalability_index = list(X.columns).index('Scalability')
default_risk_index = list(X.columns).index('Default_Risk')
liquidity_risk_index = list(X.columns).index('Liquidity_Risk')
regulatory_compliance_index = list(X.columns).index('Regulatory_Compliance')

investor_return_train = X_train[:, investor_return_index].astype(np.float32)
investor_return_val = X_val[:, investor_return_index].astype(np.float32)
investor_return_test = X_test[:, investor_return_index].astype(np.float32)

market_risk_train = X_train[:, market_risk_index].astype(np.float32)
market_risk_val = X_val[:, market_risk_index].astype(np.float32)
market_risk_test = X_test[:, market_risk_index].astype(np.float32)

scalability_train = X_train[:, scalability_index].astype(np.float32)
scalability_val = X_val[:, scalability_index].astype(np.float32)
scalability_test = X_test[:, scalability_index].astype(np.float32)

default_risk_train = X_train[:, default_risk_index].astype(np.float32)
default_risk_val = X_val[:, default_risk_index].astype(np.float32)
default_risk_test = X_test[:, default_risk_index].astype(np.float32)

liquidity_risk_train = X_train[:, liquidity_risk_index].astype(np.float32)
liquidity_risk_val = X_val[:, liquidity_risk_index].astype(np.float32)
liquidity_risk_test = X_test[:, liquidity_risk_index].astype(np.float32)

regulatory_compliance_train = X_train[:, regulatory_compliance_index].astype(np.float32)
regulatory_compliance_val = X_val[:, regulatory_compliance_index].astype(np.float32)
regulatory_compliance_test = X_test[:, regulatory_compliance_index].astype(np.float32)

# Step 7: Define the Neural Network for Regression and Classification
def create_regression_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_classification_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

regression_model = create_regression_model(X_train.shape[1])
classification_model = create_classification_model(X_train.shape[1], len(label_encoder.classes_))

# Step 8: Define Custom Loss Function with Constraints for Regression Model
D = 96000  # Daily demand

# Mean and std from standardization
investor_return_mean = scaler.mean_[numeric_features.index('Investor_Return')]
investor_return_std = scaler.scale_[numeric_features.index('Investor_Return')]
market_risk_mean = scaler.mean_[numeric_features.index('Market_Risk')]
market_risk_std = scaler.scale_[numeric_features.index('Market_Risk')]
scalability_mean = scaler.mean_[numeric_features.index('Scalability')]
scalability_std = scaler.scale_[numeric_features.index('Scalability')]

# Constraint thresholds
R_min = 0.07
R_max = 0.5
S_min = 0.7

# Normalize thresholds
R_min_norm = (R_min - investor_return_mean) / investor_return_std
R_max_norm = (R_max - market_risk_mean) / market_risk_std
S_min_norm = (S_min - scalability_mean) / scalability_std

def custom_loss(y_true, y_pred):
    # y_true: [investor_return, market_risk, scalability, demand_satisfaction]
    investor_return = y_true[:, 0]
    market_risk = y_true[:, 1]
    scalability = y_true[:, 2]
    actual_demand_satisfaction = y_true[:, 3]
    
    # Constraints
    # 1. Demand Constraint
    daily_contribution = y_pred * D
    total_contribution = K.sum(daily_contribution)
    demand_constraint = K.maximum(0.0, D - total_contribution) / D
    
    # 2. Investor Return Constraint
    financial_constraint = K.mean(K.maximum(0.0, R_min_norm - investor_return))
    
    # 3. Market Risk Constraint
    risk_constraint = K.mean(K.maximum(0.0, market_risk - R_max_norm))
    
    # 4. Scalability Constraint
    scalability_constraint = K.mean(K.maximum(0.0, S_min_norm - scalability))
    
    # Data Loss
    data_loss = tf.reduce_mean(tf.square(actual_demand_satisfaction - y_pred))
    
    # Total Loss
    total_loss = data_loss + demand_constraint + financial_constraint + risk_constraint + scalability_constraint
    return total_loss

# Step 9: Compile the Regression Model
# Prepare y_train_combined and y_val_combined
y_reg_train_combined = np.stack([investor_return_train, market_risk_train, scalability_train, y_reg_train], axis=1).astype(np.float32)
y_reg_val_combined = np.stack([investor_return_val, market_risk_val, scalability_val, y_reg_val], axis=1).astype(np.float32)
y_reg_test_combined = np.stack([investor_return_test, market_risk_test, scalability_test, y_reg_test], axis=1).astype(np.float32)

regression_model.compile(optimizer='adam', loss=custom_loss)

# Step 10: Train the Regression Model
reg_history = regression_model.fit(
    X_train, y_reg_train_combined,
    validation_data=(X_val, y_reg_val_combined),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Step 11: Evaluate the Regression Model
reg_test_loss = regression_model.evaluate(X_test, y_reg_test_combined, verbose=1)
print(f"Regression Test Loss: {reg_test_loss}")

# Step 12: Make Predictions with the Regression Model
y_reg_pred_test = regression_model.predict(X_test)

# Step 13: Train the Classification Model
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

clf_history = classification_model.fit(
    X_train, y_clf_train,
    validation_data=(X_val, y_clf_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Step 14: Evaluate the Classification Model
clf_test_loss, clf_test_accuracy = classification_model.evaluate(X_test, y_clf_test, verbose=1)
print(f"Classification Test Loss: {clf_test_loss}")
print(f"Classification Test Accuracy: {clf_test_accuracy}")

# Step 15: Make Predictions with the Classification Model
y_clf_pred_test_probs = classification_model.predict(X_test)
y_clf_pred_test = np.argmax(y_clf_pred_test_probs, axis=1)

# Step 16: Classification Report
print("\nClassification Report:")
print(classification_report(y_clf_test, y_clf_pred_test, target_names=label_encoder.classes_))

# Step 17: Calculate Total Daily Contribution and Individual Contributions
daily_contribution_test = y_reg_pred_test.flatten() * D  # Total contribution per sample

# Create test DataFrame
test_df = df_encoded.iloc[idx_test].copy()
test_df['Predicted_Demand_Satisfaction'] = y_reg_pred_test.flatten()
test_df['Actual_Demand_Satisfaction'] = y_reg_test
test_df['Daily_Contribution'] = daily_contribution_test
test_df['Predicted_Rating_Label'] = y_clf_pred_test
test_df['Predicted_Rating_Grade'] = label_encoder.inverse_transform(y_clf_pred_test)

# Map option names from one-hot encoding
option_columns = [col for col in df_encoded.columns if col.startswith('Option_')]
test_df['Option'] = test_df[option_columns].idxmax(axis=1).str.replace('Option_', '')

# Group by Option to get total contributions per option
option_contributions = test_df.groupby('Option')['Daily_Contribution'].sum().reset_index()

# Display individual contributions
print("\nIndividual Daily Contributions by Housing Option:")
print(option_contributions)

# Total contribution
total_contribution_test = daily_contribution_test.sum()
print(f"\nTotal Daily Contribution in Test Set: {total_contribution_test:.2f} units")
print(f"Demand Constraint Satisfied in Test Set: {total_contribution_test >= D}")

# Step 18: Check Constraints on Test Set
financial_constraint_test = np.mean(np.maximum(0.0, R_min_norm - investor_return_test))
print(f"Investor Return Constraint Satisfied in Test Set: {financial_constraint_test == 0.0}")

risk_constraint_test = np.mean(np.maximum(0.0, market_risk_test - R_max_norm))
print(f"Market Risk Constraint Satisfied in Test Set: {risk_constraint_test == 0.0}")

scalability_constraint_test = np.mean(np.maximum(0.0, S_min_norm - scalability_test))
print(f"Scalability Constraint Satisfied in Test Set: {scalability_constraint_test == 0.0}")

# Step 19: Model Evaluation Metrics for Regression
# De-normalize y_reg_test and y_reg_pred_test for evaluation
demand_satisfaction_mean = scaler.mean_[numeric_features.index('Demand_Satisfaction')]
demand_satisfaction_std = scaler.scale_[numeric_features.index('Demand_Satisfaction')]

y_reg_test_denorm = y_reg_test * demand_satisfaction_std + demand_satisfaction_mean
y_reg_pred_test_denorm = y_reg_pred_test.flatten() * demand_satisfaction_std + demand_satisfaction_mean

# Calculate evaluation metrics
mse = mean_squared_error(y_reg_test_denorm, y_reg_pred_test_denorm)
mae = mean_absolute_error(y_reg_test_denorm, y_reg_pred_test_denorm)
r2 = r2_score(y_reg_test_denorm, y_reg_pred_test_denorm)

print(f"\nRegression Mean Squared Error on Test Set: {mse:.4f}")
print(f"Regression Mean Absolute Error on Test Set: {mae:.4f}")
print(f"Regression R^2 Score on Test Set: {r2:.4f}")

# Step 20: Visualize Training Progress for Regression
plt.figure(figsize=(8, 6))
plt.plot(reg_history.history['loss'], label='Training Loss')
plt.plot(reg_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Regression Model Training and Validation Loss')
plt.show()

# Step 21: Visualize Prediction Results for Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test_denorm, y_reg_pred_test_denorm, alpha=0.6)
plt.xlabel('Actual Demand Satisfaction')
plt.ylabel('Predicted Demand Satisfaction')
plt.title('Regression Model: Actual vs Predicted Demand Satisfaction')
plt.plot([y_reg_test_denorm.min(), y_reg_test_denorm.max()], [y_reg_test_denorm.min(), y_reg_test_denorm.max()], 'k--', lw=2)
plt.show()

# Step 22: Plot Residuals for Regression
residuals = y_reg_test_denorm - y_reg_pred_test_denorm
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Regression Model Residuals Distribution')
plt.show()

# Step 23: Visualize Training Progress for Classification
plt.figure(figsize=(8, 6))
plt.plot(clf_history.history['loss'], label='Training Loss')
plt.plot(clf_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Classification Model Training and Validation Loss')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(clf_history.history['accuracy'], label='Training Accuracy')
plt.plot(clf_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Classification Model Training and Validation Accuracy')
plt.show()

# Step 24: Confusion Matrix for Classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_clf_test, y_clf_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Classification Model Confusion Matrix')
plt.show()
