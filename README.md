# Affordable-Housing-Risk-Assessment-and-Demand-Prediction
This project utilizes a Physics-Informed Neural Network (PINN) to evaluate and predict the demand satisfaction of various affordable housing options. By integrating a rating agency's perspective, the model assesses financial and operational risks, assigns rating grades, and forecasts daily contributions to meet housing demand, facilitating informed investment and policy decisions.

📊 Features
Synthetic Data Generation: Simulates a diverse dataset of affordable housing options with relevant financial, environmental, and social indicators.
Risk Assessment and Grading: Evaluates each housing option based on default risk, regulatory compliance, and liquidity risk, assigning rating grades from 'AAA' to 'CCC'.
Model Development: Constructs separate regression and classification neural network models, incorporating constraints to ensure reliable predictions.
Comprehensive Evaluation: Measures model performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), R² score, accuracy, and provides detailed classification reports.
Visualization: Generates plots for training progress, actual vs. predicted demand satisfaction, residual distributions, and confusion matrices to facilitate result interpretation.
Individual Contribution Analysis: Calculates and displays the daily contributions of each affordable housing option towards meeting the overall housing demand.
🚀 Getting Started
🛠 Installation
Clone the Repository

bash
Copy code
git clone https://github.com/samjwhu/affordable-housing-risk-assessment.git
cd affordable-housing-risk-assessment
Create a Virtual Environment (Optional but Recommended)

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install Dependencies

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, install the necessary libraries manually:

bash
Copy code
pip install pandas numpy tensorflow scikit-learn matplotlib
🎯 Usage
Run the Python Script

bash
Copy code
python main.py
Ensure that main.py contains the complete Python code provided in this project.

Review Outputs

Console Output: Displays statistical summaries, individual contributions, constraint satisfaction, and evaluation metrics.
Visualizations: Generates plots for training/validation loss, actual vs. predicted demand satisfaction, residual distributions, and classification confusion matrices.
📂 Project Structure
bash
Copy code
affordable-housing-risk-assessment/
│
├── data/
│   └── synthetic_housing_data.csv  # (Generated by the script)
│
├── notebooks/
│   └── analysis.ipynb             # (Optional: Jupyter notebooks for exploration)
│
├── main.py                         # (Main Python script)
├── README.md                       # (Project description)
├── requirements.txt                # (Python dependencies)
└── LICENSE                         # (Project license)
🔍 Model Evaluation and Insights
Regression Metrics: Evaluates the accuracy of demand satisfaction predictions using MSE, MAE, and R² score.
Classification Metrics: Assesses the performance of risk grading with accuracy scores and detailed classification reports.
Constraint Satisfaction: Ensures that the model adheres to predefined financial and scalability constraints.
Individual Contributions: Analyzes the daily impact of each housing option in meeting the overall demand, aiding in strategic decision-making.
📈 Visualization Examples
Training and Validation Loss


Actual vs. Predicted Demand Satisfaction


Residuals Distribution


Confusion Matrix

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

📜 License
This project is licensed under the GNU General Public License v2.0.

📧 Contact
For any questions or suggestions, please contact r02522318@gmail.com
