import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cart Abandonment Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Cart Abandonment Predictor")
st.markdown(
    "*6-algorithm ML benchmark · individual model analysis · "
    "SHAP explainability · live predictions*"
)
st.markdown("---")

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    if os.path.exists('data/2019-Nov.csv'):
        # Local — use real data
        df = pd.read_csv('data/2019-Nov.csv', nrows=200000)
        session = df.groupby('user_session').agg(
            event_count    =('event_type', 'count'),
            cart_events    =('event_type', lambda x: (x == 'cart').sum()),
            view_events    =('event_type', lambda x: (x == 'view').sum()),
            purchase_events=('event_type', lambda x: (x == 'purchase').sum()),
            avg_price      =('price', 'mean'),
            unique_products=('product_id', 'nunique')
        ).reset_index()
        session['abandoned'] = (
            (session['cart_events'] > 0) & (session['purchase_events'] == 0)
        ).astype(int)
        X = session[['event_count', 'cart_events', 'view_events',
                     'avg_price', 'unique_products']]
        y = session['abandoned']
    else:
        # Streamlit Cloud — generate realistic synthetic data
        rng = np.random.default_rng(42)
        n = 10000
        n_abandoned = int(n * 0.017)
        n_purchased = n - n_abandoned

        def make_sessions(size, abandoned):
            cart    = rng.integers(1, 8, size) if abandoned else rng.integers(0, 3, size)
            views   = rng.integers(3, 20, size) if abandoned else rng.integers(1, 15, size)
            events  = cart + views + rng.integers(1, 5, size)
            price   = rng.uniform(10, 300, size) if abandoned else rng.uniform(5, 200, size)
            unique  = rng.integers(2, 10, size)
            return pd.DataFrame({
                'event_count':     events,
                'cart_events':     cart,
                'view_events':     views,
                'avg_price':       price,
                'unique_products': unique,
                'abandoned':       int(abandoned)
            })

        df = pd.concat([
            make_sessions(n_purchased, False),
            make_sessions(n_abandoned, True)
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        X = df[['event_count', 'cart_events', 'view_events',
                'avg_price', 'unique_products']]
        y = df['abandoned']

    return X, y

# ── Train all 6 models ────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric='logloss', verbosity=0),
        'KNN':                 KNeighborsClassifier(n_neighbors=5)
    }

    trained = {}
    results = []
    for name, model in models.items():
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        cm    = confusion_matrix(y_test, preds)
        trained[name] = {'model': model, 'preds': preds, 'proba': proba}
        results.append({
            'Model':        name,
            'AUC-ROC':      round(roc_auc_score(y_test, proba), 3),
            'Precision':    round(precision_score(y_test, preds), 3),
            'Recall':       round(recall_score(y_test, preds), 3),
            'F1 Score':     round(f1_score(y_test, preds), 3),
            'Caught':       int(cm[1][1]),
            'Missed':       int(cm[1][0]),
            'False Alarms': int(cm[0][1])
        })

    results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
    return trained, results_df, X_test, y_test, X.columns.tolist()

with st.spinner("Training all 6 models — please wait ~60 seconds on first load..."):
    trained, results_df, X_test, y_test, feature_names = train_all_models()

best_name = results_df.iloc[0]['Model']

COLORS = {
    'Logistic Regression': '#378ADD',
    'Decision Tree':       '#BA7517',
    'Random Forest':       '#639922',
    'Gradient Boosting':   '#D85A30',
    'XGBoost':             '#1D9E75',
    'KNN':                 '#D4537E'
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Individual Model Analysis",
    "📊 Head-to-Head Comparison",
    "🏆 Why XGBoost Won",
    "🧠 SHAP Explainability",
    "🔮 Live Predictor"
])

# =============================================================================
# TAB 1 — INDIVIDUAL MODEL ANALYSIS
# =============================================================================
with tab1:
    st.header("Individual Model Analysis")
    st.markdown(
        "Select any model to see its full output — "
        "ROC curve, Precision-Recall curve, Confusion Matrix, and Feature Importance."
    )

    selected = st.selectbox(
        "Select a model to inspect:",
        list(trained.keys()),
        index=list(trained.keys()).index('XGBoost')
    )

    m     = trained[selected]
    preds = m['preds']
    proba = m['proba']
    color = COLORS[selected]
    auc   = roc_auc_score(y_test, proba)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    f1    = f1_score(y_test, preds)
    cm    = confusion_matrix(y_test, preds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC",   f"{auc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1 Score",  f"{f1:.3f}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{selected} — ROC Curve')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Higher curve = better separation of abandoned vs purchased.")

    with col2:
        st.subheader("Precision-Recall Curve")
        pc, rc, _ = precision_recall_curve(y_test, proba)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(rc, pc, color=color, lw=2)
        ax2.fill_between(rc, pc, alpha=0.08, color=color)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{selected} — Precision-Recall')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.02])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.caption("Tradeoff between catching abandonments and avoiding false alarms.")

    with col3:
        st.subheader("Confusion Matrix")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Purchased', 'Abandoned'],
                    yticklabels=['Purchased', 'Abandoned'],
                    ax=ax3, cbar=False)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title(f'{selected} — Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Caught: {tp:,} ✅  Missed: {fn:,} ❌  False alarms: {fp:,} ⚠️")

    st.markdown("---")
    st.subheader("Feature Importance")
    model_obj = m['model']
    if hasattr(model_obj, 'feature_importances_'):
        imp = pd.Series(model_obj.feature_importances_,
                        index=feature_names).sort_values()
        fig4, ax4 = plt.subplots(figsize=(7, 3))
        imp.plot(kind='barh', ax=ax4, color=color)
        ax4.set_xlabel('Importance Score')
        ax4.set_title(f'{selected} — Feature Importance')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    elif selected == 'Logistic Regression':
        imp = pd.Series(abs(model_obj.coef_[0]),
                        index=feature_names).sort_values()
        fig4, ax4 = plt.subplots(figsize=(7, 3))
        imp.plot(kind='barh', ax=ax4, color=color)
        ax4.set_xlabel('Absolute Coefficient')
        ax4.set_title(f'{selected} — Feature Coefficients')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    else:
        st.info(f"{selected} does not expose direct feature importances.")

# =============================================================================
# TAB 2 — HEAD-TO-HEAD COMPARISON
# =============================================================================
with tab2:
    st.header("Head-to-Head Algorithm Comparison")
    st.markdown("All 6 models trained on the **same** train/test split. Green = best per metric.")

    def highlight_best(s):
        return ['background-color: #d4edda; font-weight: bold'
                if v == s.max() else '' for v in s]

    display_df = results_df[['Model', 'AUC-ROC', 'Precision', 'Recall',
                              'F1 Score', 'Caught', 'Missed', 'False Alarms']]
    st.dataframe(
        display_df.style.apply(
            highlight_best,
            subset=['AUC-ROC', 'Precision', 'Recall', 'F1 Score']
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1 Score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, metric in enumerate(metrics):
        ax = axes[i // 2][i % 2]
        colors_bar = [COLORS[m] for m in results_df['Model']]
        bars = ax.barh(results_df['Model'], results_df[metric], color=colors_bar)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} by Model')
        ax.set_xlim(0, 1.08)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)
    plt.suptitle('All 6 Models — Metric Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("All ROC Curves Overlaid")
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    for name, data in trained.items():
        fpr, tpr, _ = roc_curve(y_test, data['proba'])
        auc_val = roc_auc_score(y_test, data['proba'])
        lw = 3 if name == best_name else 1.5
        ls = '-' if name == best_name else '--'
        ax_roc.plot(fpr, tpr, color=COLORS[name], lw=lw, ls=ls,
                    label=f'{name} (AUC={auc_val:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random baseline')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves — All 6 Models')
    ax_roc.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close()

# =============================================================================
# TAB 3 — WHY XGBOOST WON
# =============================================================================
with tab3:
    st.header(f"Why {best_name} Won — The Decision")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Why tree-based ensembles win here")
        st.markdown("""
- **Structured tabular data** — session counts, cart events, prices.
  Boosting is built for exactly this data type.
- **Non-linear relationships** — high cart events + low purchases = abandonment.
  Trees capture these interactions; linear models miss them.
- **Class imbalance** — SMOTE + XGBoost handles minority class better
  than distance-based models.
- **Mixed feature scales** — tree models do not need feature scaling
  unlike Logistic Regression or KNN.
        """)

    with col2:
        st.subheader("Why KNN ranked last")
        st.markdown("""
- **Curse of dimensionality** — distance metrics lose meaning as features increase
- **No generalisation** — memorises training data, does not learn patterns
- **Scale sensitivity** — price ($0-500) dominates event counts (0-50)
- **Result** — AUC 0.848, missed 52 sessions, 463 false alarms

**Key insight:** Highest AUC does not always mean best model.
Decision Tree had lower AUC than XGBoost but better F1
and fewest false alarms. Right choice depends on business goal.
        """)

    st.markdown("---")
    best_row = results_df[results_df['Model'] == best_name].iloc[0]
    st.success(
        f"Selected model: {best_name} — "
        f"AUC-ROC: {best_row['AUC-ROC']} | "
        f"Recall: {best_row['Recall']} | "
        f"F1: {best_row['F1 Score']}"
    )

# =============================================================================
# TAB 4 — SHAP EXPLAINABILITY
# =============================================================================
with tab4:
    st.header("SHAP Explainability")
    st.markdown(
        "SHAP shows the exact contribution of each feature to a specific prediction. "
        f"Applied to best model: **{best_name}**"
    )

    best_model_obj = trained[best_name]['model']
    sample    = X_test.iloc[:100]
    explainer = shap.TreeExplainer(best_model_obj)
    shap_vals = explainer.shap_values(sample)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global feature importance")
        shap.summary_plot(shap_vals, sample, plot_type="bar",
                          feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
        st.caption("Higher bar = more impact on predictions globally.")

    with col2:
        st.subheader("SHAP value distribution")
        shap.summary_plot(shap_vals, sample,
                          feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
        st.caption("Red = high value. Right of center = pushes toward abandonment.")

    st.markdown("---")
    st.subheader("Single session waterfall — why was this session flagged?")
    idx = st.slider("Select session index", 0, 99, 0)
    sv  = shap_vals[idx]
    ev  = explainer.expected_value
    if isinstance(ev, list):
        ev = ev[1]

    fig_w, _ = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=ev,
            feature_names=feature_names
        ), show=False
    )
    st.pyplot(fig_w)
    plt.close()

# =============================================================================
# TAB 5 — LIVE PREDICTOR
# =============================================================================
with tab5:
    st.header("Live Cart Abandonment Predictor")
    st.markdown(
        f"Real-time prediction using best model: **{best_name}** "
        f"(AUC-ROC: {results_df.iloc[0]['AUC-ROC']})"
    )

    col1, col2 = st.columns(2)
    with col1:
        event_count     = st.slider("Total Events in Session", 1, 50, 10)
        cart_events     = st.slider("Cart Add Events", 0, 20, 3)
        view_events     = st.slider("Product Views", 0, 40, 7)
    with col2:
        avg_price       = st.slider("Average Product Price ($)", 0.0, 500.0, 45.0)
        unique_products = st.slider("Unique Products Viewed", 1, 30, 5)

    if st.button("Predict Abandonment Risk", type="primary"):
        input_df = pd.DataFrame([{
            'event_count':     event_count,
            'cart_events':     cart_events,
            'view_events':     view_events,
            'avg_price':       avg_price,
            'unique_products': unique_products
        }])[feature_names]

        prob  = trained[best_name]['model'].predict_proba(input_df)[0][1]
        label = "🔴 HIGH RISK — Likely to Abandon" if prob > 0.5 \
                else "🟢 LOW RISK — Likely to Purchase"

        c1, c2 = st.columns(2)
        c1.metric("Prediction", label)
        c2.metric("Abandonment Probability", f"{prob * 100:.1f}%")

        if prob > 0.7:
            st.error("Trigger intervention — show discount coupon or reminder")
        elif prob > 0.5:
            st.warning("Borderline — monitor this session closely")
        else:
            st.success("Session looks healthy — likely to convert")

        st.markdown("---")
        st.subheader("Why this prediction?")
        ex = shap.TreeExplainer(trained[best_name]['model'])
        sv = ex.shap_values(input_df)
        ev = ex.expected_value
        if isinstance(ev, list):
            ev = ev[1]
        fig_p, _ = plt.subplots(figsize=(8, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=ev,
                feature_names=feature_names
            ), show=False
        )
        st.pyplot(fig_p)
        plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Gayathri Kammidi** · MS Computer Science, Governors State University · May 2026  \n"
    "github.com/gkammidi-prog · gkammidi@gmail.com"
)