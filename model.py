import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
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

X = session[['event_count','cart_events','view_events','avg_price','unique_products']]
y = session['abandoned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sm = SMOTE(random_state=42)
X_train_r, y_train_r = sm.fit_resample(X_train, y_train)

# ── helper to print results ────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm    = confusion_matrix(y_test, preds)
    print(f"\n── {name} Results ──")
    print(f"AUC-ROC   : {roc_auc_score(y_test, proba):.3f}")
    print(f"Precision : {precision_score(y_test, preds):.3f}")
    print(f"Recall    : {recall_score(y_test, preds):.3f}")
    print(f"F1 Score  : {f1_score(y_test, preds):.3f}")
    print(f"Caught    : {cm[1][1]} | Missed: {cm[1][0]} | False alarms: {cm[0][1]}")
    return {
        'Model':     name,
        'AUC-ROC':   round(roc_auc_score(y_test, proba), 3),
        'Precision': round(precision_score(y_test, preds), 3),
        'Recall':    round(recall_score(y_test, preds), 3),
        'F1 Score':  round(f1_score(y_test, preds), 3),
        'Caught':    cm[1][1],
        'Missed':    cm[1][0],
        'FalseAlarm':cm[0][1]
    }

results = []

# ── Model 1: Logistic Regression ──────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_r, y_train_r)
results.append(evaluate("Logistic Regression", lr, X_test, y_test))

# ── Model 2: Decision Tree ─────────────────────────────
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_r, y_train_r)
results.append(evaluate("Decision Tree", dt, X_test, y_test))

# ── Model 3: Random Forest ─────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_r, y_train_r)
results.append(evaluate("Random Forest", rf, X_test, y_test))

# ── Model 4: Gradient Boosting ────────────────────────
print("\nTraining Gradient Boosting...")
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_r, y_train_r)
results.append(evaluate("Gradient Boosting", gb, X_test, y_test))

# ── Model 5: XGBoost ──────────────────────────────────
print("\nTraining XGBoost...")
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train_r, y_train_r)
results.append(evaluate("XGBoost", xgb, X_test, y_test))

# ── Model 6: KNN ──────────────────────────────────────
print("\nTraining KNN...")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_r, y_train_r)
results.append(evaluate("KNN", knn, X_test, y_test))

# ── Comparison table ───────────────────────────────────
print("\n\n── Comparison So Far ──")
print(f"{'Model':<22} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Caught':>7} {'Missed':>7} {'FalseAlarm':>11}")
print("-" * 80)
for r in results:
    print(f"{r['Model']:<22} {r['AUC-ROC']:>6} {r['Precision']:>6} {r['Recall']:>6} {r['F1 Score']:>6} {r['Caught']:>7} {r['Missed']:>7} {r['FalseAlarm']:>11}")