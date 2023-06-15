from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\kalha\random_forest_model.joblib'  # Specify the path to your model file
model = load(model_path)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    current_loan_amount = float(request.form['Current Loan Amount'])
    term = float(request.form['Term'])
    credit_score = float(request.form['Credit Score'])
    annual_income = float(request.form['Annual Income'])
    home_ownership = request.form['Home Ownership']
    purpose = request.form['Purpose']
    monthly_debt = float(request.form['Monthly Debt'])
    years_of_credit_history = float(request.form['Years of Credit History'])
    number_of_open_accounts = float(request.form['Number of Open Accounts'])
    number_of_credit_problems = float(request.form['Number of Credit Problems'])
    current_credit_balance = float(request.form['Current Credit Balance'])
    maximum_open_credit = float(request.form['Maximum Open Credit'])
    bankruptcies = float(request.form['Bankruptcies'])
    tax_liens = float(request.form['Tax Liens'])

    # Create a dataframe with the form data
    data = pd.DataFrame([[current_loan_amount, term, credit_score, annual_income, home_ownership, purpose,
                          monthly_debt, years_of_credit_history, number_of_open_accounts,
                          number_of_credit_problems, current_credit_balance, maximum_open_credit,
                          bankruptcies, tax_liens]],
                        columns=['Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
                                 'Home Ownership', 'Purpose', 'Monthly Debt', 'Years of Credit History',
                                 'Number of Open Accounts', 'Number of Credit Problems',
                                 'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies',
                                 'Tax Liens'])

    # Make the prediction
    prediction = model.predict(data)[0]

    # Render the prediction template with the result
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
