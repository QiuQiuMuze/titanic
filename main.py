import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ===== 数据预处理 =====

# 性别转数字
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

# 填充年龄缺失值
train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)

# 选特征
features = ["Pclass", "Sex", "Age"]

X = train[features]
y = train["Survived"]
X_test = test[features]

# ===== 训练模型 =====
model = RandomForestClassifier()
model.fit(X, y)

# ===== 预测 =====
predictions = model.predict(X_test)

# ===== 生成提交文件 =====
output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

output.to_csv("submission.csv", index=False)

print("完成！生成 submission.csv")