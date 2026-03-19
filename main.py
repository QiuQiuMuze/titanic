import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from catboost import CatBoostClassifier

# ======================
# 1. 读取数据
# ======================
print("1. 读取数据...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 记录 train 的长度，方便后续精准拆分
n_train = len(train)
full = pd.concat([train, test], sort=False).reset_index(drop=True)

# ======================
# 2. 特征工程
# ======================
print("2. 进行特征工程...")
# Title
full["Title"] = full["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
full["Title"] = full["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
)
full["Title"] = full["Title"].replace(["Mlle", "Ms"], "Miss")
full["Title"] = full["Title"].replace("Mme", "Mrs")

# Family
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Ticket group
ticket_counts = full["Ticket"].value_counts()
full["TicketGroupSize"] = full["Ticket"].map(ticket_counts)

# Embarked
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# Fare
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# Age
full["Age"] = full.groupby(["Sex", "Pclass", "Title"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# 编码
for col in ["Sex", "Embarked", "Title"]:
    full[col] = full[col].astype("category").cat.codes
full = full.fillna(-1)

# ======================
# 3. 数据拆分 (修复 0 samples 报错)
# ======================
print("3. 拆分数据集并进行标准化...")
# 使用行号精准切片，不依赖 Survived 列的 NaN
train_df = full.iloc[:n_train].copy()
test_df = full.iloc[n_train:].copy()

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title",
            "FamilySize", "IsAlone", "TicketGroupSize", "SibSp", "Parch"]

# 转为 numpy 数组，提升运算速度并防止部分报错
X = train_df[features].values
y = train_df["Survived"].astype(int).values
X_test = test_df[features].values

# ======================
# 4. 数据标准化 (修复逻辑回归不收敛报错)
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ======================
# 5. KFold 融合预测
# ======================
print("4. 开始模型训练与 KFold 融合预测...")
models = [
    LogisticRegression(max_iter=1000, random_state=42),
    RandomForestClassifier(n_estimators=200, random_state=42),
    GradientBoostingClassifier(random_state=42),
    CatBoostClassifier(verbose=0, random_state=42)
]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存放最终融合的测试集预测结果
final_test_preds = np.zeros(len(X_test))

for model_obj in models:
    model_name = model_obj.__class__.__name__
    print(f"   -> 正在训练 {model_name}...")

    # 用于存放当前模型在 K 折中的测试集预测结果
    fold_preds = np.zeros(len(X_test))

    for train_idx, val_idx in kf.split(X, y):
        # 针对逻辑回归使用标准化后的数据，树模型使用原数据
        if isinstance(model_obj, LogisticRegression):
            X_tr, y_tr = X_scaled[train_idx], y[train_idx]
            X_te = X_test_scaled
        else:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X_test

        # 必须克隆模型，确保每一折训练都是从零开始
        model = clone(model_obj)
        model.fit(X_tr, y_tr)

        # 累加测试集预测概率
        fold_preds += model.predict_proba(X_te)[:, 1] / kf.n_splits

    # 将当前模型的预测结果累加到最终结果中（取平均）
    final_test_preds += fold_preds / len(models)

# ======================
# 6. 生成提交文件
# ======================
print("5. 保存预测结果...")
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (final_test_preds > 0.5).astype(int)
})

submission.to_csv("submission_combine.csv", index=False)
print("✅ 运行完成！结果已保存至 submission_combine.csv")