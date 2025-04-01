import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 设置网页标题
st.set_page_config(page_title="Diagnostic model for prostate cancer", layout="centered")

# 加载模型
model_filename = "gas_sensor_model.pkl"
try:
    model = joblib.load(model_filename)
    st.sidebar.success("✅ 模型成功加载！")
except Exception as e:
    st.sidebar.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 定义特征列
feature_columns = ['Slope_1_2', 'Intercept_1_2', 'Slope_21_22', 'Intercept_21_22',
                   'Mean_2_22', 'Median_2_22', 'Variance_2_22',
                   'Mean_22_41', 'Median_22_41', 'Variance_22_41']

# 页面标题
st.title("Diagnostic model for prostate cancer")

# --- 上传单个样本文件进行预测 ---
st.subheader("Upload CSV file")
uploaded_file = st.file_uploader("Upload CSV, Excel or JSON file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    try:
        # 读取上传数据
        if uploaded_file.name.endswith(".csv"):
            try:
                data = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                data = pd.read_csv(uploaded_file, encoding="gbk")
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            data = pd.read_json(uploaded_file)

        st.subheader("Preview of data")
        st.dataframe(data.head())

        # 检查数据是否只有一行
        if data.shape[0] != 1:
            st.error("The uploaded file must contain only one row of feature values.")
        else:
            # 检查特征完整性
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                st.error(f"Missing features: {missing_features}")
            else:
                # 预测
                prediction = model.predict(data[feature_columns])[0]
                prediction_prob = model.predict_proba(data[feature_columns])[0]

                label_mapping = {0: "neg", 1: "pos"}
                predicted_label = label_mapping[prediction]

                # 显示预测结果
                st.subheader("Projected results")
                st.success(f"Prediction result: {predicted_label}")
                st.write(f"Probability distribution: {prediction_prob}")

                # 绘制饼状图
                fig, ax = plt.subplots()
                ax.pie(prediction_prob, labels=["Negative", "Positive"], autopct='%1.1f%%', startangle=90,
                       colors=["#4682b4", "#ffa500"])
                ax.axis('equal')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error while processing file: {e}")


# --- 方式 2：手动输入数据进行单个预测 ---
st.subheader("Manual input of features")

# 显示手动输入框
input_data = {}
for feature in feature_columns:
    input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

# 点击预测按钮
if st.button("positive/negative"):
    input_df = pd.DataFrame([input_data])
    try:
        # 模型预测
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0]

        label_mapping = {0: "neg", 1: "pos"}
        predicted_label = label_mapping[prediction]

        # 显示预测结果和概率
        st.success(f"Prediction result: {predicted_label}")
        st.write(f"Probability distribution: {prediction_prob}")

        # 绘制饼状图
        fig, ax = plt.subplots()
        ax.pie(prediction_prob, labels=["Negative", "Positive"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # 保证饼图是圆形
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
