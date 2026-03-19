import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 1. 读取数据
# =========================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 保存 test 的 PassengerId
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
# 例如 "Braund, Mr. Owen Harris" -> "Mr"
full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

# 合并一些少见称谓
title_map = {
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
}
full["Title"] = full["Title"].replace(title_map)

# 2.2 构造家庭规模特征
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

# 2.3 是否独自出行
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# 2.4 填充 Embarked
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# 2.5 填充 Fare
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# 2.6 用 Title + Pclass 分组填 Age
full["Age"] = full.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# 如果还有极少数没填上，再用全局中位数补
full["Age"] = full["Age"].fillna(full["Age"].median())

# 2.7 Cabin 缺失很多，但可以提取“有没有 Cabin 信息”
full["HasCabin"] = full["Cabin"].notna().astype(int)

# 2.8 Ticket 也可以做一个简单特征：票号前缀是否纯数字
full["TicketPrefix"] = full["Ticket"].astype(str).str.replace(r"[0-9\.\/]", "", regex=True).str.strip()
full["TicketPrefix"] = full["TicketPrefix"].replace("", "NONE")

# 只保留常见前缀，其余归 RareTicket
ticket_counts = full["TicketPrefix"].value_counts()
common_tickets = ticket_counts[ticket_counts >= 10].index
full["TicketPrefix"] = full["TicketPrefix"].apply(lambda x: x if x in common_tickets else "RareTicket")

# =========================
# 3. 选择特征列
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
    "TicketPrefix"
]

model_data = full[feature_cols + ["Survived", "is_train"]].copy()

# one-hot 编码类别变量
model_data = pd.get_dummies(
    model_data,
    columns=["Sex", "Embarked", "Title", "TicketPrefix"],
    drop_first=False
)

# 拆回 train/test
train_processed = model_data[model_data["is_train"] == 1].drop(columns=["is_train"])
test_processed = model_data[model_data["is_train"] == 0].drop(columns=["is_train", "Survived"])

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"].astype(int)
X_test = test_processed

# =========================
# 4. 本地验证
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 模型1：随机森林
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_valid)
rf_acc = accuracy_score(y_valid, rf_pred)

# 模型2：梯度提升树
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_valid)
gb_acc = accuracy_score(y_valid, gb_pred)

print(f"RandomForest 验证集准确率: {rf_acc:.4f}")
print(f"GradientBoosting 验证集准确率: {gb_acc:.4f}")

# =========================
# 5. 选更好的模型，在全量训练集上重训
# =========================
if gb_acc >= rf_acc:
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    best_model_name = "GradientBoosting"
else:
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    best_model_name = "RandomForest"

final_model.fit(X, y)
test_predictions = final_model.predict(X_test).astype(int)

print(f"最终选择模型: {best_model_name}")

# =========================
# 6. 生成提交文件
# =========================
submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Survived": test_predictions
})

submission.to_csv("submission_improved.csv", index=False)
print("完成！已生成 submission_improved.csv")