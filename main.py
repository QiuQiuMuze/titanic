import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# ======================
# 1. 读取数据
# ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

full = pd.concat([train, test], sort=False)

# ======================
# 2. 强特征工程（关键！！）
# ======================

# Title
full["Title"] = full["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
full["Title"] = full["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
)
full["Title"] = full["Title"].replace(["Mlle","Ms"], "Miss")
full["Title"] = full["Title"].replace("Mme", "Mrs")

# Family
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Ticket group
ticket_counts = full["Ticket"].value_counts()
full["TicketGroupSize"] = full["Ticket"].map(ticket_counts)

# Cabin处理
full["Cabin"] = full["Cabin"].fillna("U")
full["CabinLetter"] = full["Cabin"].str[0]

# Embarked
full["Embarked"].fillna(full["Embarked"].mode()[0], inplace=True)

# Fare
full["Fare"].fillna(full["Fare"].median(), inplace=True)

# Age（最重要）
full["Age"] = full.groupby(["Sex", "Pclass", "Title"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Age/Fare 分桶
full["AgeBin"] = pd.cut(full["Age"], bins=[0,12,18,35,60,100], labels=False)
full["FareBin"] = pd.qcut(full["Fare"], 4, labels=False, duplicates="drop")

# ======================
# 3. 编码
# ======================
cat_cols = ["Sex","Embarked","Title","CabinLetter"]

for col in cat_cols:
    full[col] = full[col].astype("category").cat.codes

# ======================
# 4. 拆分数据
# ======================
train_df = full[full["Survived"].notna()]
test_df = full[full["Survived"].isna()]

X = train_df.drop(["Survived","Name","Ticket","Cabin","PassengerId"], axis=1)
y = train_df["Survived"].astype(int)

X_test = test_df.drop(["Survived","Name","Ticket","Cabin","PassengerId"], axis=1)

# ======================
# 5. KFold训练（关键提升点）
# ======================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))
val_scores = []

for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50)]
    )

    preds = model.predict(X_val)
    val_scores.append(accuracy_score(y_val, preds))

    test_preds += model.predict(X_test) / 5

print("CV score:", np.mean(val_scores))

# ======================
# 6. 提交
# ======================
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (test_preds > 0.5).astype(int)
})

submission.to_csv("submission_lgb.csv", index=False)