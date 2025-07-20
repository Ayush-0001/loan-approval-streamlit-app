import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# Generate synthetic dataset
def generate_data(n=200):
    np.random.seed(42)
    income = np.random.randint(20000, 150000, n)
    credit_score = np.random.randint(300, 850, n)
    age = np.random.randint(21, 65, n)
    loan_amount = np.random.randint(1000, 50000, n)
    employment = np.random.choice([0, 1], n)  # 0 = unemployed, 1 = employed
    approved = (
        (income > 50000) & 
        (credit_score > 600) & 
        (employment == 1) & 
        (loan_amount < 0.5 * income)
    ).astype(int)

    return pd.DataFrame({
        "Income": income,
        "CreditScore": credit_score,
        "Age": age,
        "LoanAmount": loan_amount,
        "Employed": employment,
        "Approved": approved
    })

# Train model
def train_model():
    df = generate_data()
    X = df.drop("Approved", axis=1)
    y = df["Approved"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# App interface
def main():
    st.title("💰 Loan Approval Predictor")
    st.write("Fill the form to check if your loan might be approved.")

    income = st.number_input("Annual Income", 10000, 200000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age = st.slider("Age", 21, 65, 30)
    loan_amount = st.number_input("Loan Amount", 500, 100000, step=500)
    employment_status = st.selectbox("Employment Status", ["Unemployed", "Employed"])
    employed = 1 if employment_status == "Employed" else 0

    model = train_model()

    if st.button("Predict"):
        user_data = pd.DataFrame([{
            "Income": income,
            "CreditScore": credit_score,
            "Age": age,
            "LoanAmount": loan_amount,
            "Employed": employed
        }])
        prediction = model.predict(user_data)[0]
        prob = model.predict_proba(user_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2%})")
        else:
            st.error(f"❌ Loan Not Approved (Confidence: {prob:.2%})")

        # Visual 1: Scatter Plot
        df = generate_data()
        fig = px.scatter(df, x="Income", y="LoanAmount", color=df["Approved"].astype(str),
                         title="Loan Approval Scatter (Green = Approved)", labels={"color": "Approved"})
        fig.add_scatter(x=[income], y=[loan_amount], mode="markers", name="You",
                        marker=dict(size=15, color="red"))
        st.plotly_chart(fig)

        # Visual 2: Feature Importance
        st.subheader("🧠 Feature Importance")
        feat_imp = pd.Series(model.feature_importances_, index=user_data.columns).sort_values()
        bar = px.bar(
            feat_imp,
            orientation='h',
            title="Which Features Matter Most?",
            labels={'value': 'Importance', 'index': 'Feature'},
            color=feat_imp.values,
            color_continuous_scale='Blues'
        )
        bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(bar, use_container_width=True)

if __name__ == "__main__":
    main()
