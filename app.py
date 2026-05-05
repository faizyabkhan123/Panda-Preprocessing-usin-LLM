import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq

# -------- CONFIG --------
st.set_page_config(page_title="Panda + LLM Preprocessing Agent", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("API Key not found in Streamlit Secrets!")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------- UI STYLE --------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E86C1;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 25px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Panda + LLM Preprocessing Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Data Cleaning • Analysis • Explainability</div>', unsafe_allow_html=True)

# -------- SESSION --------
if "df" not in st.session_state:
    st.session_state.df = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------- LLM ENGINE --------
def run_llm(prompt, df):
    try:
        dtypes = df.dtypes.astype(str).to_dict()

        explain_keywords = ["what", "why", "explain", "difference", "meaning"]
        mode = "explain" if any(k in prompt.lower() for k in explain_keywords) else "code"

        if mode == "code":
            system_prompt = f"""
            You are a Python data analyst.

            DataFrame columns:
            {dtypes}

            Task: {prompt}

            Rules:
            - Output ONLY Python code
            - Use pandas or plotly
            - Use st.plotly_chart(fig)
            - DO NOT use fig.show()
            - DO NOT open new tabs
            - Last line must be: st.session_state.df = df
            """
        else:
            system_prompt = f"""
            You are a professional data scientist.

            Dataset structure:
            {dtypes}

            Question: {prompt}

            Explain clearly in simple, practical terms.
            Do NOT generate code.
            """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"ERROR: {str(e)}"

# -------- PREPROCESSING --------
def basic_preprocessing(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df.columns = df.columns.str.strip().str.lower()
    return df

def moderate_preprocessing(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    for col in df.select_dtypes(include=['number']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df

def generate_advanced_plan(df):
    plan = []
    if df.duplicated().sum() > 0:
        plan.append("Remove duplicate rows")
    if df.isnull().sum().sum() > 0:
        plan.append("Fill missing values (smart imputation)")
    if len(df.select_dtypes(include='object').columns) > 0:
        plan.append("Encode categorical variables")
    plan.append("Remove outliers using IQR")
    plan.append("Normalize numeric features")
    plan.append("Drop highly correlated features (>0.9)")
    return plan

def advanced_preprocessing(df):
    from sklearn.preprocessing import StandardScaler

    df = moderate_preprocessing(df)

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
    df.drop(columns=drop_cols, inplace=True)

    return df

# -------- SIDEBAR --------
with st.sidebar:
    st.title("📂 Data Panel")
    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(file)
        else:
            st.session_state.df = pd.read_excel(file)

# -------- MAIN --------
if st.session_state.df is not None:
    df = st.session_state.df

    # KPI
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", df.isnull().sum().sum())
    c4.metric("Duplicates", df.duplicated().sum())

    st.dataframe(df.head(10), width='stretch')

    # -------- AUTO SUMMARY --------
    if st.button("🔍 Auto Dataset Summary"):
        st.info(f"""
        Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        Numeric Columns: {list(df.select_dtypes(include='number').columns)}
        Categorical Columns: {list(df.select_dtypes(include=['object','string']).columns)}
        """)

    # -------- PREPROCESSING --------
    st.subheader("⚙️ Preprocessing Mode")
    mode = st.radio("Choose level:", ["Basic", "Moderate", "Advanced"])

    if st.button("Run Preprocessing"):
        if mode == "Basic":
            st.session_state.df = basic_preprocessing(df)
            st.success("Basic preprocessing applied")

        elif mode == "Moderate":
            st.session_state.df = moderate_preprocessing(df)
            st.success("Moderate preprocessing applied")

        elif mode == "Advanced":
            plan = generate_advanced_plan(df)

            st.warning("Planned Actions:")
            for p in plan:
                st.write("✔", p)

            if st.button("Confirm Execution"):
                st.session_state.df = advanced_preprocessing(df)
                st.success("Advanced preprocessing applied")

    # -------- VISUALIZATION --------
    st.subheader("📈 Visualization")
    x = st.selectbox("X-axis", df.columns)
    y = st.selectbox("Y-axis", df.columns)
    chart = st.selectbox("Chart", ["Scatter", "Bar", "Histogram"])

    if chart == "Scatter":
        fig = px.scatter(df, x=x, y=y)
    elif chart == "Bar":
        fig = px.bar(df, x=x, y=y)
    else:
        fig = px.histogram(df, x=x)

    st.plotly_chart(fig, width='stretch')

    # -------- CHAT --------
    st.subheader("💬 AI Data Analyst")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about data, preprocessing, or analysis..."):
        st.session_state.chat.append({"role": "user", "content": prompt})

        response = run_llm(prompt, df)

        if "ERROR" in response:
            st.session_state.chat.append({"role": "assistant", "content": response})

        elif "st.session_state.df" in response:
            try:
                exec_env = {"df": df.copy(), "pd": pd, "px": px, "st": st}
                exec(response, exec_env)

                if "df" in exec_env:
                    st.session_state.df = exec_env["df"]

                st.session_state.chat.append({"role": "assistant", "content": "Task Applied ✅"})

            except Exception as e:
                st.session_state.chat.append({"role": "assistant", "content": str(e)})

        else:
            st.session_state.chat.append({"role": "assistant", "content": response})

        st.rerun()

else:
    st.info("📂 Upload a dataset to begin analysis")