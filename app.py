import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="FintechSurvive AI", layout="centered")


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    data = {
        "startup_name": [
            "Kuda","Carbon","Cowrywise","PiggyVest","Flutterwave",
            "Paystack","Moniepoint","TeamApt","Paga","Interswitch",
            "Bankly","Umba","Thepeer","Mono","Okra",
            "Bloc","Sudo","Anchor","Lenco","Brass",
            "StartupA","StartupB","StartupC","StartupD","StartupE",
            "StartupF","StartupG","StartupH","StartupI","StartupJ",
            "StartupK","StartupL","StartupM","StartupN","StartupO",
            "StartupP","StartupQ","StartupR","StartupS","StartupT",
            "StartupU","StartupV","StartupW","StartupX","StartupY",
            "StartupZ","Startup1","Startup2","Startup3","Startup4",
            "Startup5","Startup6","Startup7","Startup8","Startup9",
            "Startup10","Startup11","Startup12","Startup13","Startup14",
        ],
        "funding_usd": [
            25000000,15000000,3000000,30000000,250000000,
            200000000,170000000,50000000,35000000,188000000,
            2000000,10000000,2200000,15000000,3000000,
            5000000,4500000,3000000,5500000,4000000,
            500000,800000,200000,100000,1500000,
            300000,50000,150000,400000,600000,
            1000000,200000,80000,250000,500000,
            100000,50000,300000,700000,400000,
            2000000,1500000,800000,300000,100000,
            50000,200000,150000,500000,1000000,
            3000000,2500000,1200000,400000,100000,
            50000,80000,200000,350000,600000,
        ],
        "funding_rounds": [
            4,3,2,3,5,
            4,4,3,4,5,
            2,3,2,3,2,
            2,2,2,2,2,
            1,1,1,1,1,
            1,1,1,1,1,
            2,1,1,1,1,
            1,1,1,1,1,
            2,2,1,1,1,
            1,1,1,1,2,
            2,2,2,1,1,
            1,1,1,1,1,
        ],
        "team_size": [
            350,200,80,150,900,
            800,1200,500,300,2000,
            50,80,20,40,30,
            60,45,35,55,40,
            5,8,3,2,10,
            4,2,3,6,7,
            12,3,2,4,5,
            2,1,4,8,5,
            15,12,7,4,2,
            1,3,2,6,10,
            20,18,9,5,2,
            1,2,4,6,8,
        ],
        "years_operating": [
            6,9,7,7,9,
            9,10,9,15,24,
            5,6,3,4,4,
            4,4,3,5,4,
            1,2,1,1,2,
            1,1,1,2,1,
            3,1,1,2,2,
            1,1,2,2,1,
            3,3,2,1,1,
            1,2,1,2,3,
            4,3,2,1,1,
            1,1,2,2,2,
        ],
        "has_revenue": [
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,0,1,1,
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            1,0,0,0,0,
            0,0,0,1,0,
            1,1,0,0,0,
            0,0,0,1,1,
            1,1,1,0,0,
            0,0,0,1,1,
        ],
        "pivot_count": [
            1,2,1,0,1,
            0,1,1,2,1,
            1,2,1,0,1,
            0,1,0,1,0,
            2,3,1,2,1,
            3,2,1,0,2,
            1,3,2,1,2,
            3,4,2,1,2,
            1,1,2,3,2,
            3,2,3,1,1,
            1,0,1,2,3,
            4,3,2,1,1,
        ],
        "market": [
            "Neobanking","Lending","Savings","Savings","Payments",
            "Payments","Agency Banking","Agency Banking","Payments","Payments",
            "Neobanking","Lending","Infrastructure","Infrastructure","Infrastructure",
            "Neobanking","Infrastructure","Infrastructure","Neobanking","Neobanking",
            "Lending","Payments","Savings","Lending","Payments",
            "Neobanking","Savings","Lending","Payments","Infrastructure",
            "Lending","Payments","Savings","Neobanking","Lending",
            "Payments","Infrastructure","Savings","Lending","Payments",
            "Neobanking","Lending","Payments","Savings","Infrastructure",
            "Payments","Lending","Neobanking","Savings","Payments",
            "Agency Banking","Neobanking","Lending","Payments","Savings",
            "Infrastructure","Lending","Payments","Neobanking","Savings",
        ],
        "has_cbn_license": [
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,0,0,0,
            1,0,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            1,0,0,0,0,
            0,0,0,1,0,
            1,0,0,0,0,
            0,0,0,0,1,
            1,1,1,0,0,
            0,0,0,1,1,
        ],
        "survived": [
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            1,0,0,0,0,
            0,0,0,1,0,
            1,1,0,0,0,
            0,0,0,1,1,
            1,1,1,0,0,
            0,0,0,1,1,
        ],
    }

    df = pd.DataFrame(data)

    le = LabelEncoder()
    df["market_enc"] = le.fit_transform(df["market"])

    features = ["funding_usd","funding_rounds","team_size","years_operating",
                "has_revenue","pivot_count","market_enc","has_cbn_license"]

    X = df[features]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_pred_prob),
        "report":   classification_report(y_test, y_pred, output_dict=True),
        "cm":       confusion_matrix(y_test, y_pred),
    }

    return model, df, le, features, metrics


model, df, le, features, metrics = load_and_train()

FEATURE_LABELS = ["Funding (USD)","Funding Rounds","Team Size","Years Operating",
                  "Has Revenue","Pivot Count","Market","CBN License"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("FintechSurvive AI")
st.caption("Nigerian Fintech Startup Survival Predictor - XGBoost + SHAP")
st.markdown("Predict whether a Nigerian fintech startup will **Survive or Fail** based on key growth signals.")
st.divider()

# ── Model Metrics ─────────────────────────────────────────────────────────────
st.subheader("Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
c2.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")
c3.metric("Precision", f"{metrics['report']['1']['precision']:.2f}")
c4.metric("Recall",    f"{metrics['report']['1']['recall']:.2f}")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.subheader("Startup Profile")

markets = sorted(df["market"].unique().tolist())

col1, col2 = st.columns(2)
with col1:
    funding      = st.number_input("Total Funding Raised (USD)", min_value=0, max_value=500000000, value=500000, step=50000)
    funding_rds  = st.slider("Number of Funding Rounds", 0, 10, 1)
    team_size    = st.slider("Team Size", 1, 2000, 10)
    market       = st.selectbox("Market Segment", markets)
with col2:
    years_op     = st.slider("Years Operating", 0, 25, 2)
    has_revenue  = st.selectbox("Generating Revenue?", ["Yes", "No"])
    pivot_count  = st.slider("Number of Pivots", 0, 5, 1)
    cbn_license  = st.selectbox("Has CBN License / Approval?", ["Yes", "No"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Survival", use_container_width=True):

    mkt_enc  = le.transform([market])[0]
    rev_enc  = 1 if has_revenue == "Yes" else 0
    cbn_enc  = 1 if cbn_license == "Yes" else 0

    X_input = pd.DataFrame([[funding, funding_rds, team_size, years_op,
                              rev_enc, pivot_count, mkt_enc, cbn_enc]],
                             columns=features)

    pred      = model.predict(X_input)[0]
    prob      = model.predict_proba(X_input)[0]
    surv_prob = prob[1] * 100
    fail_prob = prob[0] * 100

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.success("SURVIVE")
        verdict_color = "#2ecc71"
        verdict_text  = "SURVIVE"
    else:
        st.error("FAIL")
        verdict_color = "#e74c3c"
        verdict_text  = "FAIL"

    st.markdown(
        "<h1 style='text-align:center;font-size:4rem;color:" + verdict_color + "'>"
        + verdict_text + "</h1>",
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)
    p1.metric("Survival Probability", f"{surv_prob:.1f}%")
    p2.metric("Failure Probability",  f"{fail_prob:.1f}%")

    # Probability bar
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh([""], [surv_prob], color="#2ecc71", height=0.5)
    ax.barh([""], [fail_prob], left=[surv_prob], color="#e74c3c", height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Survive vs Fail Probability")
    ax.legend(["Survive", "Fail"], loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.subheader("Why this prediction? (SHAP Explanation)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    sv          = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": FEATURE_LABELS,
        "SHAP Value": sv,
        "Direction": ["Increases survival" if v > 0 else "Decreases survival" for v in sv]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#2ecc71" if v > 0 else "#e74c3c" for v in shap_df["SHAP Value"]]
    bars    = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, height=0.55)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on survival prediction)")
    ax.set_title("Factors Driving This Prediction")
    for bar, val in zip(bars, shap_df["SHAP Value"]):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(shap_df[["Feature","Direction"]].reset_index(drop=True),
                 hide_index=True, use_container_width=True)

st.divider()

# ── Market Insights ───────────────────────────────────────────────────────────
st.subheader("Market Insights")
tab1, tab2, tab3 = st.tabs(["Survival by Market", "Funding vs Survival", "Feature Importance"])

with tab1:
    surv_market = df.groupby("market")["survived"].mean().sort_values(ascending=True) * 100
    fig, ax     = plt.subplots(figsize=(7, 4))
    colors      = ["#2ecc71" if v >= 50 else "#e74c3c" for v in surv_market.values]
    bars        = ax.barh(surv_market.index, surv_market.values, color=colors, height=0.55)
    ax.axvline(50, color="gray", linewidth=1, linestyle="--", label="50% line")
    ax.set_xlabel("Survival Rate (%)")
    ax.set_title("Survival Rate by Market Segment")
    ax.legend()
    for bar, val in zip(bars, surv_market.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, ax   = plt.subplots(figsize=(7, 4))
    survived  = df[df["survived"] == 1]
    failed    = df[df["survived"] == 0]
    ax.scatter(survived["funding_usd"]/1e6, survived["years_operating"],
               color="#2ecc71", alpha=0.7, s=60, label="Survived")
    ax.scatter(failed["funding_usd"]/1e6,   failed["years_operating"],
               color="#e74c3c", alpha=0.7, s=60, label="Failed")
    ax.set_xlabel("Total Funding (USD Millions)")
    ax.set_ylabel("Years Operating")
    ax.set_title("Funding vs Years Operating")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    importance = pd.Series(model.feature_importances_, index=FEATURE_LABELS).sort_values(ascending=True)
    fig, ax    = plt.subplots(figsize=(7, 4))
    max_val    = importance.max()
    colors     = ["#e74c3c" if v == max_val else "#3498db" for v in importance.values]
    bars       = ax.barh(importance.index, importance.values * 100, color=colors, height=0.55)
    ax.set_xlabel("Importance (%)")
    ax.set_title("What Predicts Startup Survival the Most?")
    for bar, val in zip(bars, importance.values):
        ax.text(val * 100 + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Confusion Matrix ──────────────────────────────────────────────────────────
with st.expander("Confusion Matrix and Technical Details"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion Matrix**")
        cm     = metrics["cm"]
        fig, ax = plt.subplots(figsize=(4, 3))
        im      = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Fail", "Survive"])
        ax.set_yticklabels(["Fail", "Survive"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Technical Details**")
        st.markdown(
            "- **Algorithm:** XGBoost Classifier\n"
            "- **Trees:** 200 estimators\n"
            "- **Max Depth:** 4\n"
            "- **Learning Rate:** 0.1\n"
            "- **Explainability:** SHAP TreeExplainer\n"
            "- **Samples:** " + str(len(df)) + " Nigerian fintech startups\n"
            "- **Features:** 8 (funding, team, market, license, revenue, pivots)\n"
            "- **Target:** Survived (1) or Failed (0)\n"
            "- **Split:** 80/20 stratified"
        )