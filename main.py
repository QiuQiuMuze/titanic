import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# 1. 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_passenger_id = test["PassengerId"]

# 合并数据方便统一处理
train["is_train"] = 1
test["is_train"] = 0
test["Survived"] = np.nan
full = pd.concat([train, test], axis=0, ignore_index=True)

# 2. 提取 Title
full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
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

# 3. 家庭特征
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# 4. 填缺失
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

full["Age"] = full.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
full["Age"] = full["Age"].fillna(full["Age"].median())

# 5. 选更稳的特征
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone"
]

data = full[feature_cols + ["Survived", "is_train"]].copy()
data = pd.get_dummies(data, columns=["Sex", "Embarked", "Title"], drop_first=False)

train_processed = data[data["is_train"] == 1].drop(columns=["is_train"])
test_processed = data[data["is_train"] == 0].drop(columns=["is_train", "Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"].astype(int)
X_test = test_processed

# 6. 多模型交叉验证
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
}

best_model_name = None
best_score = -1
best_model = None

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    print(f"{name} CV accuracy: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print(f"\n选择模型: {best_model_name}, CV平均准确率: {best_score:.4f}")

# 7. 用最佳模型训练全量数据
best_model.fit(X, y)
predictions = best_model.predict(X_test).astype(int)

# 8. 导出提交文件
submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Survived": predictions
})

submission.to_csv("submission_improved2.csv", index=False)
print("完成！已生成 submission_improved2.csv")