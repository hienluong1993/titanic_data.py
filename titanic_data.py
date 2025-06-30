import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="🚢 타이타닉 생존자 데이터 탐색 및 예측", layout="wide", page_icon="⚓")

@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    return train

train = load_data()

st.title("🚢 타이타닉 생존자 데이터 탐색 및 실시간 예측")
st.image("Stower_Titanic.jpg", use_column_width=True)

# ------------------------------
# 데이터 필터
st.markdown("### 🎛️ 데이터 필터")
sex_filter = st.multiselect("성별 선택", options=train['Sex'].unique(), default=train['Sex'].unique())
pclass_filter = st.multiselect("객실 등급 (Pclass)", options=sorted(train['Pclass'].unique()), default=sorted(train['Pclass'].unique()))
min_age = int(train['Age'].min(skipna=True))
max_age = int(train['Age'].max(skipna=True))
age_range = st.slider("나이 범위 선택", min_value=min_age, max_value=max_age, value=(min_age, max_age))

filtered_data = train[
    (train['Sex'].isin(sex_filter)) &
    (train['Pclass'].isin(pclass_filter)) &
    (train['Age'].notna()) &
    (train['Age'].between(age_range[0], age_range[1]))
]

st.markdown(f"✅ 현재 필터 기준 데이터: **{filtered_data.shape[0]} rows**")

if filtered_data.empty:
    st.warning("⚠️ 데이터가 없습니다. 필터를 변경하세요.")
    st.stop()

# ------------------------------
# 실시간 생존율
survival_rate = filtered_data['Survived'].mean() * 100
st.success(f"⚡ 현재 필터 조건 기준 생존 확률: **{survival_rate:.2f}%**")

# ------------------------------
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 데이터 보기", "📊 기초 통계", "📈 시각화", "🤖 생존 예측"])

with tab1:
    st.subheader("📄 필터링된 데이터")
    st.dataframe(filtered_data, height=450)

with tab2:
    st.subheader("📊 기술 통계")
    st.write(filtered_data.describe())

    st.subheader("🔢 생존자 분포")
    survived_counts = filtered_data["Survived"].value_counts().sort_index()
    st.bar_chart(survived_counts)

with tab3:
    st.subheader("🧭 Feature 별 생존과의 상관관계 분석")

    st.markdown("#### 1️⃣ 성별과 생존")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=filtered_data, ax=ax1)
    ax1.set_title("성별 vs 생존 여부")
    st.pyplot(fig1)
    st.info("💡 여성 생존률이 더 높음.")

    st.markdown("#### 2️⃣ 나이와 생존")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=filtered_data[filtered_data['Survived']==1]['Age'], label='Survived', fill=True, ax=ax2)
    sns.kdeplot(data=filtered_data[filtered_data['Survived']==0]['Age'], label='Not Survived', fill=True, ax=ax2)
    ax2.legend()
    ax2.set_title("나이 vs 생존 여부")
    st.pyplot(fig2)
    st.info("💡 어린 승객의 생존률이 더 높음.")

    st.markdown("#### 3️⃣ 좌석 등급과 생존")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=filtered_data, ax=ax3)
    ax3.set_title("좌석 등급 vs 생존 여부")
    st.pyplot(fig3)
    st.info("💡 1등급의 생존률이 가장 높음.")

    st.markdown("#### 4️⃣ 운임과 생존")
    fig4, ax4 = plt.subplots()
    sns.kdeplot(data=filtered_data[filtered_data['Survived']==1]['Fare'], label='Survived', fill=True, ax=ax4)
    sns.kdeplot(data=filtered_data[filtered_data['Survived']==0]['Fare'], label='Not Survived', fill=True, ax=ax4)
    ax4.legend()
    ax4.set_title("운임 vs 생존 여부")
    st.pyplot(fig4)
    st.info("💡 운임이 높을수록 생존률이 높음.")

    st.markdown("#### 5️⃣ 출발항과 생존")
    fig5, ax5 = plt.subplots()
    sns.countplot(x='Embarked', hue='Survived', data=filtered_data, ax=ax5)
    ax5.set_title("출발항 vs 생존 여부")
    st.pyplot(fig5)
    st.info("💡 Cherbourg(C)에서 탑승한 승객의 생존률이 가장 높음.")

# ------------------------------
# 🤖 ML 생존 예측
with tab4:
    st.subheader("🤖 실시간 생존 예측 (Logistic Regression)")

    # 사용자 입력
    col1, col2 = st.columns(2)
    with col1:
        sex_input = st.selectbox("성별", ["male", "female"])
        age_input = st.slider("나이", 0, 80, 30)
        pclass_input = st.selectbox("좌석 등급", [1, 2, 3])
    with col2:
        fare_input = st.slider("운임", 0, 600, 50)
        embarked_input = st.selectbox("출발항", ["C", "Q", "S"])

    # 모델 훈련
    @st.cache_data
    def train_model():
        df = train.dropna(subset=['Age', 'Embarked'])
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        df['Sex'] = le_sex.fit_transform(df['Sex'])
        df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
        X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
        y = df['Survived']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model, le_sex, le_embarked

    model, le_sex, le_embarked = train_model()

    # 예측
    input_df = pd.DataFrame({
        'Pclass': [pclass_input],
        'Sex': [le_sex.transform([sex_input])[0]],
        'Age': [age_input],
        'Fare': [fare_input],
        'Embarked': [le_embarked.transform([embarked_input])[0]]
    })

    if st.button("🚀 생존 여부 예측하기"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"🌟 예측 결과: **생존** (생존 확률: {probability*100:.2f}%)")
        else:
            st.error(f"💀 예측 결과: **사망** (생존 확률: {probability*100:.2f}%)")
        st.caption("이 예측은 Logistic Regression 모델을 기반으로 한 실험용이며 실제 상황과 다를 수 있습니다.")

# ------------------------------
# 다운로드 버튼
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button("📥 필터된 데이터 다운로드", data=csv, file_name='filtered_titanic_data.csv', mime='text/csv')
