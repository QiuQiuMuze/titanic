import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ======================
# 1. 极简数据读取
# ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False).reset_index(drop=True)

# ======================
# 2. 核心特征工程（精简版
# ======================
# 提取头衔
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
full['Title'] = full['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full['Title'] = full['Title'].replace('Mlle', 'Miss')
full['Title'] = full['Title'].replace('Ms', 'Miss')
full['Title'] = full['Title'].replace('Mme', 'Mrs')

# 填补缺失值
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['Fare'] = full['Fare'].fillna(full['Fare'].median())
# 年龄用 Sex, Pclass, Title 的中位数填补
full['Age'] = full.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

# 关键：提取姓氏（用于发现家庭组）
full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0])

# ======================
# 3.Woman-Child-Group
# ======================
# 我们只看妇女和儿童（Master）
full['IsWomanOrChild'] = ((full['Sex'] == 'female') | (full['Title'] == 'Master')).astype(int)

# 建立家庭生存参考
# 在训练集中，计算每个姓氏的生存情况
family_survival = full.iloc[:891].groupby('Surname')['Survived'].mean()
full['FamilySurv'] = full['Surname'].map(family_survival).fillna(0.5)

# 如果这个家庭里有人死了，那么这个组里的其他人也很危险
# 如果这个家庭里有人活了，那么这个组里的其他人也很安全
# 只给妇女和儿童应用这个逻辑
full['GroupHazard'] = 0.5
full.loc[(full['IsWomanOrChild'] == 1) & (full['FamilySurv'] == 0), 'GroupHazard'] = 0
full.loc[(full['IsWomanOrChild'] == 1) & (full['FamilySurv'] == 1), 'GroupHazard'] = 1

# ======================
# 4. 准备训练
# ======================
# 编码
full['Sex'] = full['Sex'].map({'male': 0, 'female': 1})
full['Embarked'] = full['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
full['Title'] = full['Title'].map(title_mapping)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 'GroupHazard']

X = full.iloc[:891][features]
y = train['Survived']
X_test = full.iloc[891:][features]

# ======================
# 5. 使用一个极其稳健的随机森林
# ======================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,        # 严格限制深度，防止它学坏
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X, y)

# ======================
# 6. 生成提交
# ======================
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission_final.csv", index=False)