import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from catboost import CatBoostClassifier

# ======================
# 1. 数据读取与初步合并
# ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
n_train = len(train)
full = pd.concat([train, test], sort=False).reset_index(drop=True)

# ======================
# 2. 核心特征工程：生存组挖掘
# ======================
# 提取姓氏
full['Surname'] = full['Name'].apply(lambda x: x.split(',')[0])

# 定义“妇女和儿童”组 (泰坦尼克号核心生存逻辑)
# 寻找同一姓氏或同一客票号的群体
full['WomanChild'] = ((full['Sex'] == 'female') | (full['Age'] <= 12)).astype(int)
family_groups = full.groupby('Surname')['Survived'].mean()
ticket_groups = full.groupby('Ticket')['Survived'].mean()

# 创建特征：如果同组人中有人遇难/生存，会极大影响该成员的预测
full['FamilySurv'] = full['Surname'].map(family_groups).fillna(0.5)
full['TicketSurv'] = full['Ticket'].map(ticket_groups).fillna(0.5)
full['GroupSurvival'] = (full['FamilySurv'] + full['TicketSurv']) / 2

# ======================
# 3. 基础特征提取
# ======================
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_dict = {
    "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty",
    "Don": "Royalty", "Sir": "Royalty", "Dr": "Officer", "Rev": "Officer",
    "Countess": "Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs",
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Royalty"
}
full['Title'] = full['Title'].map(title_dict)

full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
full['Embarked'] = full['Embarked'].fillna('S')
full['Fare'] = full['Fare'].fillna(full['Fare'].median())

# ======================
# 4. 预测填充缺失年龄 (比中位数better)
# ======================
age_features = ['Pclass', 'Sex', 'Title', 'SibSp', 'Parch', 'Fare']
age_df = full[['Age'] + age_features].copy()
age_df['Sex'] = LabelEncoder().fit_transform(age_df['Sex'])
age_df['Title'] = LabelEncoder().fit_transform(age_df['Title'].astype(str))

train_age = age_df[age_df['Age'].notnull()]
test_age = age_df[age_df['Age'].isnull()]

if not test_age.empty:
    rf_age = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_age.fit(train_age.drop('Age', axis=1), train_age['Age'])
    full.loc[full['Age'].isnull(), 'Age'] = rf_age.predict(test_age.drop('Age', axis=1))

# ======================
# 5. 编码与准备
# ======================
for col in ['Sex', 'Embarked', 'Title']:
    full[col] = LabelEncoder().fit_transform(full[col])

# 选择精选特征
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title',
            'FamilySize', 'GroupSurvival', 'IsAlone']

X = full.iloc[:n_train][features].values
y = train['Survived'].values
X_test = full.iloc[n_train:][features].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ======================
# 6. 模型调参与加权融合
# ======================
# 调整后的模型参数，注重抗过拟合
models = [
    ('LR', LogisticRegression(C=0.1, max_iter=1000), True),
    ('RF', RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=42), False),
    ('GB', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42), False),
    ('CB', CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, verbose=0, random_state=42), False)
]

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 增加到10折
final_preds = np.zeros(len(X_test))

for name, model_obj, use_scaled in models:
    print(f"训练 {name}...")
    fold_preds = np.zeros(len(X_test))

    for train_idx, val_idx in kf.split(X, y):
        xt, xv = (X_scaled[train_idx], X_scaled[val_idx]) if use_scaled else (X[train_idx], X[val_idx])
        yt, yv = y[train_idx], y[val_idx]

        m = clone(model_obj)
        m.fit(xt, yt)
        fold_preds += m.predict_proba(X_test_scaled if use_scaled else X_test)[:, 1] / kf.n_splits

    # 给树模型（RF, CB）稍高的权重，逻辑回归稍低
    weight = 0.3 if name in ['RF', 'CB'] else 0.2
    final_preds += fold_preds * weight

# ======================
# 7. 生成提交
# ======================
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (final_preds > 0.5).astype(int)
})
submission.to_csv("submission_survive.csv", index=False)
print("策略：生存组挖掘 + 随机森林补年龄 + 10折加权融合")