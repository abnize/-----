# Titanic 데이터 머신러닝 비교 코드 - 제출용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report

# ================================
# 1. 데이터 불러오기
# ================================
titanic = sns.load_dataset("titanic")
print("원본 데이터 크기:", titanic.shape)
print(titanic.head())

# ================================
# 2. 전처리
# ================================
# 중요한 피처만 선택
df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']] #기존 컬럼에서 일부만 가져옴

# 결측치 처리
df['age'].fillna(df['age'].median(), inplace=True) # fillna(값) : 결측치(NaN)를 지정 값으로 채움.
# 여기서는 해당 열의 평균으로 채워서, 이후 연산(비교/그룹바이)이 깔끔하게 돌아가게 함.
# inplace=True로 바로 df['age']에 반영. # age의 결측치를 age의 평균으로 채움
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# 범주형 → 숫자형 변환
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0
df['embarked'] = le.fit_transform(df['embarked'])

# 독립변수, 종속변수 분리
X = df.drop('survived', axis=1)
y = df['survived']

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (로지스틱회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 3. 간단한 시각화
# ================================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='survived')
plt.title("Survivors vs Deaths Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(data=df, x='pclass', y='survived')
plt.title("Survival rate by Pclass")
plt.show()

# ================================
# 4. 모델 학습 + GridSearch
# ================================
# (1) Logistic Regression
log_params = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
log_reg = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid=log_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
log_reg.fit(X_train_scaled, y_train)
print("Logistic Best Params:", log_reg.best_params_) #log_reg...~ 최적의 모델이라는 뜻~

# (2) Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("RF Best Params:", rf.best_params_) # rf.best...~ 최적의 모델이라는 뜻

# ================================
# 5. 성능 평가
# ================================
models = {
    "Logistic Regression": log_reg.best_estimator_,
    "Random Forest": rf.best_estimator_
}

for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n=== {name} 성능 ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

# ================================
# 6. ROC 곡선 비교
# ================================
plt.figure(figsize=(8,6))

for name, model in models.items():
    if name == "Logistic Regression":
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Logistic vs Random Forest")
plt.legend()
plt.show()
