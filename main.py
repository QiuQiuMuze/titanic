import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# =========================
# 1. 读取数据
# =========================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_passenger_id = test["PassengerId"]

# 为了统一做特征工程，先拼接
train["is_train"] = 1
test["is_train"] = 0
test["Survived"] = np.nan

full = pd.concat([train, test], axis=0, ignore_index=True)

# =========================
# 2. 特征工程
# =========================

# 2.1 提取称谓 Title
full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

# 合并少见称谓
full["Title"] = full["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare"
})

# 2.2 家庭规模
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# 2.3 Ticket group size（共享票号的人数）
ticket_counts = full["Ticket"].value_counts()
full["TicketGroupSize"] = full["Ticket"].map(ticket_counts)

# 2.4 Cabin 是否存在
full["HasCabin"] = full["Cabin"].notna().astype(int)

# 2.5 填充 Embarked
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# 2.6 填充 Fare
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# 2.7 分组填充 Age：按 Sex + Pclass + Title
full["Age"] = full.groupby(["Sex", "Pclass", "Title"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
full["Age"] = full["Age"].fillna(full["Age"].median())

# 2.8 连续特征分桶（Titanic 常见有效手段）
full["FareBin"] = pd.qcut(full["Fare"], 4, labels=False, duplicates="drop")
full["AgeBin"] = pd.cut(full["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)

# 补可能的空值
full["AgeBin"] = full["AgeBin"].fillna(full["AgeBin"].mode()[0]).astype(int)
full["FareBin"] = full["FareBin"].fillna(full["FareBin"].mode()[0]).astype(int)

# =========================
# 3. 选择特征
# =========================
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "SibSp",
    "Parch",
    "HasCabin",
    "TicketGroupSize",
    "AgeBin",
    "FareBin"
]

data = full[feature_cols + ["Survived", "is_train"]].copy()

train_processed = data[data["is_train"] == 1].drop(columns=["is_train"])
test_processed = data[data["is_train"] == 0].drop(columns=["is_train", "Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"].astype(int)
X_test = test_processed.copy()

# CatBoost 的类别特征列
cat_features = ["Sex", "Embarked", "Title"]

# =========================
# 4. 本地验证
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=200
)

valid_pred = model.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_pred)
print(f"\n验证集准确率: {valid_acc:.4f}")
print(f"最佳迭代数: {model.get_best_iteration()}")

# =========================
# 5. 用全量训练集重训最终模型
# =========================
best_iter = model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 800

final_model = CatBoostClassifier(
    iterations=best_iter,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0
)

final_model.fit(
    X,
    y,
    cat_features=cat_features
)

test_pred = final_model.predict(X_test).astype(int).reshape(-1)

# =========================
# 6. 生成提交文件
# =========================
submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Survived": test_pred
})

submission.to_csv("submission_catboost.csv", index=False)
print("完成！已生成 submission_catboost.csv")