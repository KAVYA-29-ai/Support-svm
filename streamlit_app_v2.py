import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SVM Income Classifier", layout="wide")
st.title("🎯 SVM-Based Income Classification")
st.markdown("Predict income level using Support Vector Machine models")

page = st.sidebar.selectbox("Choose a section", ["Home", "Train Model", "Make Predictions", "Model Comparison"])

@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess data from local CSV"""
    try:
        df = pd.read_csv("/workspaces/Support-svm/UCI_Adult_Income_Dataset.csv")
    except:
        st.error("Failed to load dataset.")
        return None, None, None, None, None, None
    
    df = df.replace("?", np.nan).dropna()
    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    label_encoders = {}
    for col in cat_cols:
        if col != "income":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["income"].astype(str))
    
    X = df.drop("income", axis=1)
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, scaler, label_encoders, num_cols, df.drop("income", axis=1).columns.tolist()

@st.cache_resource
def train_both_models(X, y):
    """Train both models and cache them - runs only once"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to numpy if needed
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_np = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_test_np = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
    
    # Traditional SVM - quick training
    batch_model = SVC(kernel="rbf", probability=True, random_state=42, max_iter=500)
    batch_model.fit(X_train_np, y_train_np)
    
    # Streaming SVM - optimized
    stream_model = SVC(kernel="rbf", probability=True, random_state=42, max_iter=500)
    global_sv_X = None
    global_sv_y = None
    
    batch_size = 10000  # Larger batches = faster
    for i in range(len(X_train_np) // batch_size + 1):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train_np))
        if start >= end:
            break
        X_batch = X_train_np[start:end]
        y_batch = y_train_np[start:end]
        
        local_model = SVC(kernel="rbf", random_state=42, max_iter=500)
        local_model.fit(X_batch, y_batch)
        sv_idx = local_model.support_
        
        if global_sv_X is None:
            global_sv_X = X_batch[sv_idx]
            global_sv_y = y_batch[sv_idx]
        else:
            global_sv_X = np.vstack((global_sv_X, X_batch[sv_idx]))
            global_sv_y = np.hstack((global_sv_y, y_batch[sv_idx]))
        
        stream_model.fit(global_sv_X, global_sv_y)
    
    return batch_model, stream_model, X_test_np, y_test_np

if page == "Home":
    st.header("Welcome! 👋")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### SVM Income Classification
        - **Traditional SVM**: Full dataset training
        - **Streaming SVM**: Batch-wise training with support vector consolidation
        - Predicts: ≤$50K or >$50K
        """)
    
    with col2:
        st.success("""
        ✅ Model Training  
        ✅ Real-time Predictions  
        ✅ Performance Comparison  
        ✅ Confusion Matrix  
        ✅ Visualizations
        """)

elif page == "Train Model":
    st.header("🚀 Train SVM Models")
    
    if st.button("Load Data & Train Models"):
        data = load_and_prepare_data()
        if data[0] is not None:
            X, y, scaler, label_encoders, num_cols, features = data
            st.success(f"✅ Data loaded! Shape: {X.shape}")
            
            with st.spinner("Training models..."):
                batch_model, stream_model, X_test, y_test = train_both_models(X, y)
                
                batch_pred = batch_model.predict(X_test)
                stream_pred = stream_model.predict(X_test)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Traditional SVM")
                    st.metric("Accuracy", f"{accuracy_score(y_test, batch_pred):.4f}")
                    st.metric("Precision", f"{precision_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("Recall", f"{recall_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("F1 Score", f"{f1_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, batch_model.predict_proba(X_test), multi_class='ovr'):.4f}")
                
                with col2:
                    st.subheader("Streaming SVM")
                    st.metric("Accuracy", f"{accuracy_score(y_test, stream_pred):.4f}")
                    st.metric("Precision", f"{precision_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("Recall", f"{recall_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("F1 Score", f"{f1_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, stream_model.predict_proba(X_test), multi_class='ovr'):.4f}")
                
                st.success("✅ Training complete!")

elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    data = load_and_prepare_data()
    if data[0] is not None:
        X, y, scaler, label_encoders, num_cols, features = data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 100, 35)
            education_years = st.slider("Education Level (years)", 1, 16, 10)
            hours_per_week = st.slider("Hours per week", 1, 100, 40)
        
        with col2:
            capital_gain = st.slider("Capital Gain ($)", 0, 100000, 0)
            capital_loss = st.slider("Capital Loss ($)", 0, 5000, 0)
        
        with col3:
            try:
                workclass_opts = list(label_encoders.get("workclass", LabelEncoder()).classes_) if "workclass" in label_encoders else []
                workclass = st.selectbox("Workclass", workclass_opts if workclass_opts else ["Private"])
            except:
                workclass = "Private"
            
            try:
                occupation_opts = list(label_encoders.get("occupation", LabelEncoder()).classes_) if "occupation" in label_encoders else []
                occupation = st.selectbox("Occupation", occupation_opts if occupation_opts else ["Tech-support"])
            except:
                occupation = "Tech-support"
        
        if st.button("Predict Income 🎯"):
            with st.spinner("Loading models & predicting..."):
                batch_model, stream_model, X_test, y_test = train_both_models(X, y)
                
                try:
                    user_data = X.iloc[0:1].copy()
                    user_data.iloc[0] = X.mean()
                    
                    col_list = list(X.columns)
                    
                    for feat, val in [("age", age), ("education-num", education_years), ("hours-per-week", hours_per_week), ("capital-gain", capital_gain), ("capital-loss", capital_loss)]:
                        if feat in col_list:
                            idx = col_list.index(feat)
                            mean_v = float(X[feat].mean())
                            std_v = float(X[feat].std())
                            user_data.iloc[0, idx] = (float(val) - mean_v) / std_v if std_v > 1e-10 else (float(val) - mean_v)
                    
                    if "workclass" in col_list and "workclass" in label_encoders:
                        try:
                            idx = col_list.index("workclass")
                            user_data.iloc[0, idx] = label_encoders["workclass"].transform([workclass])[0]
                        except:
                            pass
                    
                    if "occupation" in col_list and "occupation" in label_encoders:
                        try:
                            idx = col_list.index("occupation")
                            user_data.iloc[0, idx] = label_encoders["occupation"].transform([occupation])[0]
                        except:
                            pass
                    
                    batch_pred = batch_model.predict(user_data)[0]
                    batch_prob = batch_model.predict_proba(user_data)[0]
                    
                    stream_pred = stream_model.predict(user_data)[0]
                    stream_prob = stream_model.predict_proba(user_data)[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Traditional SVM")
                        pred_text = "🟢 High (>$50K)" if batch_pred == 1 else "🔴 Low (≤$50K)"
                        st.write(f"**{pred_text}**")
                        st.progress(float(max(batch_prob)))
                        st.write(f"Confidence: {max(batch_prob):.2%}")
                    
                    with col2:
                        st.subheader("Streaming SVM")
                        pred_text = "🟢 High (>$50K)" if stream_pred == 1 else "🔴 Low (≤$50K)"
                        st.write(f"**{pred_text}**")
                        st.progress(float(max(stream_prob)))
                        st.write(f"Confidence: {max(stream_prob):.2%}")
                    
                    st.success("✅ Done!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "Model Comparison":
    st.header("📊 Model Comparison")
    
    data = load_and_prepare_data()
    if data[0] is not None:
        X, y, scaler, label_encoders, num_cols, features = data
        
        with st.spinner("Training models & generating comparison..."):
            batch_model, stream_model, X_test, y_test = train_both_models(X, y)
            
            batch_pred = batch_model.predict(X_test)
            stream_pred = stream_model.predict(X_test)
            
            batch_proba = batch_model.predict_proba(X_test)
            stream_proba = stream_model.predict_proba(X_test)
            
            metrics = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                "Traditional SVM": [
                    accuracy_score(y_test, batch_pred),
                    precision_score(y_test, batch_pred, average='weighted', zero_division=0),
                    recall_score(y_test, batch_pred, average='weighted', zero_division=0),
                    f1_score(y_test, batch_pred, average='weighted', zero_division=0),
                    roc_auc_score(y_test, batch_proba, multi_class='ovr')
                ],
                "Streaming SVM": [
                    accuracy_score(y_test, stream_pred),
                    precision_score(y_test, stream_pred, average='weighted', zero_division=0),
                    recall_score(y_test, stream_pred, average='weighted', zero_division=0),
                    f1_score(y_test, stream_pred, average='weighted', zero_division=0),
                    roc_auc_score(y_test, stream_proba, multi_class='ovr')
                ]
            }
            
            metrics_df = pd.DataFrame(metrics)
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.subheader("Chart")
            st.bar_chart(metrics_df.set_index("Metric"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Traditional SVM - Confusion Matrix")
                cm = confusion_matrix(y_test, batch_pred)
                st.write(pd.DataFrame(cm, columns=["Pred ≤50K", "Pred >50K"], index=["Actual ≤50K", "Actual >50K"]))
                st.text(classification_report(y_test, batch_pred, target_names=["≤50K", ">50K"]))
            
            with col2:
                st.subheader("Streaming SVM - Confusion Matrix")
                cm = confusion_matrix(y_test, stream_pred)
                st.write(pd.DataFrame(cm, columns=["Pred ≤50K", "Pred >50K"], index=["Actual ≤50K", "Actual >50K"]))
                st.text(classification_report(y_test, stream_pred, target_names=["≤50K", ">50K"]))
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                better_acc = "Traditional" if accuracy_score(y_test, batch_pred) > accuracy_score(y_test, stream_pred) else "Streaming"
                st.info(f"**Better Accuracy**: {better_acc} SVM")
            with col2:
                better_f1 = "Traditional" if f1_score(y_test, batch_pred, average='weighted', zero_division=0) > f1_score(y_test, stream_pred, average='weighted', zero_division=0) else "Streaming"
                st.info(f"**Better F1**: {better_f1} SVM")
            with col3:
                better_auc = "Traditional" if roc_auc_score(y_test, batch_proba, multi_class='ovr') > roc_auc_score(y_test, stream_proba, multi_class='ovr') else "Streaming"
                st.info(f"**Better ROC-AUC**: {better_auc} SVM")
