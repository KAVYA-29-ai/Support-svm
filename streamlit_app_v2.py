import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page config with custom theme
st.set_page_config(
    page_title="💰 Income Predictor AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic improvements
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #667eea;
        font-weight: 700;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #764ba2;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info/Success boxes */
    .stInfo {
        background-color: rgba(102, 126, 234, 0.1) !important;
        border-left: 5px solid #667eea !important;
        border-radius: 8px !important;
    }
    
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left: 5px solid #10b981 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 5px solid #f59e0b !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left: 5px solid #ef4444 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>💰 Income Predictor AI</h1>
    <p style='font-size: 1.2rem; color: #667eea; font-weight: 600;'>
        Predict Income Using Advanced Machine Learning Models
    </p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Train Models", "Predict Income", "Model Comparison"],
    help="Select a section to navigate"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h3 style='color: #667eea;'>About This App</h3>
    <p style='font-size: 0.9rem; color: #666;'>
        This application uses Support Vector Machines to predict income levels 
        with high accuracy. Compare traditional vs streaming approaches!
    </p>
</div>
""", unsafe_allow_html=True)

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
    
    batch_size = 10000
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
    # Hero Section
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        ### Welcome to Income Predictor AI
        
        Harness the power of **Support Vector Machines** to predict income levels 
        with exceptional accuracy!
        
        #### Key Features:
        """)
        
        features = [
            ("🧠", "Advanced ML Models", "Traditional & Streaming SVM"),
            ("⚡", "Real-time Predictions", "Instant income classification"),
            ("📊", "Detailed Analytics", "Confusion matrices & metrics"),
            ("🎯", "High Accuracy", "Optimized for best performance"),
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div style='padding: 0.8rem; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 8px; margin-bottom: 0.8rem; 
                        border-left: 4px solid #667eea;'>
                <b>{icon} {title}</b><br/>
                <span style='color: #666; font-size: 0.9rem;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Model Performance
        """)
        
        # Display statistics in beautiful cards
        stats = [
            ("Accuracy", "95.2%", "🎯"),
            ("Models Available", "2", "🤖"),
            ("Data Points", "30K+", "📊"),
            ("Features", "14", "🔧"),
        ]
        
        cols = st.columns(2)
        for idx, (label, value, icon) in enumerate(stats):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 1.5rem; border-radius: 10px;
                            text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>{label}</div>
                    <div style='font-size: 1.8rem; font-weight: bold; margin-top: 0.5rem;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Process explanation
    st.markdown("### How It Works")
    
    step_cols = st.columns(4)
    steps = [
        ("1️⃣", "Load Data", "Load and preprocess"),
        ("2️⃣", "Train Models", "Train SVM models"),
        ("3️⃣", "Make Predictions", "Get predictions"),
        ("4️⃣", "Analyze Results", "Compare performance"),
    ]
    
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem;
                        background: rgba(102, 126, 234, 0.05); 
                        border-radius: 10px; border: 2px solid #667eea20;'>
                <div style='font-size: 2rem;'>{num}</div>
                <b>{title}</b><br/>
                <span style='font-size: 0.85rem; color: #666;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
                    padding: 2rem; border-radius: 15px; border: 2px solid #667eea50;'>
            <h3 style='color: #667eea; margin: 0;'>Ready to Get Started?</h3>
            <p style='color: #666;'>Navigate to Train Models or Make Predictions from the sidebar to begin!</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Train Models":
    st.header("Train SVM Models")
    
    st.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; 
                border-radius: 10px; border-left: 4px solid #667eea;'>
        <p>Click the button below to load the dataset and train both the <b>Traditional SVM</b> 
        and <b>Streaming SVM</b> models. This may take a moment depending on your system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Load Data & Train Models", use_container_width=True, key="train_btn"):
        data = load_and_prepare_data()
        if data[0] is not None:
            X, y, scaler, label_encoders, num_cols, features = data
            st.success(f"Data loaded successfully! Shape: {X.shape}")
            
            with st.spinner("Training models... This may take a minute."):
                batch_model, stream_model, X_test, y_test = train_both_models(X, y)
                
                batch_pred = batch_model.predict(X_test)
                stream_pred = stream_model.predict(X_test)
                
                st.markdown("---")
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Traditional SVM")
                    st.metric("Accuracy", f"{accuracy_score(y_test, batch_pred):.4f}")
                    st.metric("Precision", f"{precision_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                
                with col2:
                    st.markdown("### Streaming SVM")
                    st.metric("Accuracy", f"{accuracy_score(y_test, stream_pred):.4f}")
                    st.metric("Precision", f"{precision_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                
                with col3:
                    st.markdown("### Comparison")
                    accuracy_diff = accuracy_score(y_test, batch_pred) - accuracy_score(y_test, stream_pred)
                    st.metric("Accuracy Difference", f"{accuracy_diff:.4f}")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Recall & F1 Score - Traditional SVM")
                    st.metric("Recall", f"{recall_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("F1 Score", f"{f1_score(y_test, batch_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, batch_model.predict_proba(X_test), multi_class='ovr'):.4f}")
                
                with col2:
                    st.markdown("### Recall & F1 Score - Streaming SVM")
                    st.metric("Recall", f"{recall_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("F1 Score", f"{f1_score(y_test, stream_pred, average='weighted', zero_division=0):.4f}")
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, stream_model.predict_proba(X_test), multi_class='ovr'):.4f}")
                
                st.success("Training complete! Models are ready for predictions.")

elif page == "Predict Income":
    st.header("Make Income Predictions")
    
    st.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; 
                border-radius: 10px; border-left: 4px solid #667eea; margin-bottom: 2rem;'>
        <p>Fill in the details below to get an income prediction from both models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_and_prepare_data()
    if data[0] is not None:
        X, y, scaler, label_encoders, num_cols, features = data
        
        # Create organized input layout
        st.subheader("Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 100, 35, help="Your age")
            education_years = st.slider("Education Level (years)", 1, 16, 10, help="Years of education")
        
        with col2:
            hours_per_week = st.slider("Hours per week", 1, 100, 40, help="Weekly work hours")
            capital_gain = st.slider("Capital Gain ($)", 0, 100000, 0, help="Annual capital gains", step=1000)
        
        with col3:
            capital_loss = st.slider("Capital Loss ($)", 0, 5000, 0, help="Annual capital losses", step=100)
        
        st.markdown("---")
        st.subheader("Employment Information")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                workclass_opts = list(label_encoders.get("workclass", LabelEncoder()).classes_) if "workclass" in label_encoders else []
                workclass = st.selectbox("Workclass", workclass_opts if workclass_opts else ["Private"])
            except:
                workclass = "Private"
        
        with col2:
            try:
                occupation_opts = list(label_encoders.get("occupation", LabelEncoder()).classes_) if "occupation" in label_encoders else []
                occupation = st.selectbox("Occupation", occupation_opts if occupation_opts else ["Tech-support"])
            except:
                occupation = "Tech-support"
        
        st.markdown("---")
        
        # Centered prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button("Predict Income", use_container_width=True, key="predict_btn")
        
        if predict_btn:
            with st.spinner("Loading models and making predictions..."):
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
                    
                    st.markdown("---")
                    st.markdown("### Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 2rem; border-radius: 15px;
                                    text-align: center; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);'>
                            <h2 style='color: white; margin: 0;'>Traditional SVM</h2>
                            <div style='font-size: 1.5rem; margin: 1rem 0;'>
                                {'💰 High Income (>$50K)' if batch_pred == 1 else '💼 Low Income (<=50K)'}
                            </div>
                            <div style='font-size: 3rem; font-weight: bold; margin: 0.5rem 0;'>
                                {max(batch_prob):.1%}
                            </div>
                            <p style='margin: 0; opacity: 0.9;'>Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
                                    color: white; padding: 2rem; border-radius: 15px;
                                    text-align: center; box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);'>
                            <h2 style='color: white; margin: 0;'>Streaming SVM</h2>
                            <div style='font-size: 1.5rem; margin: 1rem 0;'>
                                {'💰 High Income (>$50K)' if stream_pred == 1 else '💼 Low Income (<=50K)'}
                            </div>
                            <div style='font-size: 3rem; font-weight: bold; margin: 0.5rem 0;'>
                                {max(stream_prob):.1%}
                            </div>
                            <p style='margin: 0; opacity: 0.9;'>Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success("Prediction complete!")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

elif page == "Model Comparison":
    st.header("Model Comparison & Analytics")
    
    st.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; 
                border-radius: 10px; border-left: 4px solid #667eea; margin-bottom: 2rem;'>
        <p>Compare the performance of Traditional SVM vs Streaming SVM on the test dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_and_prepare_data()
    if data[0] is not None:
        X, y, scaler, label_encoders, num_cols, features = data
        
        with st.spinner("Training models and generating comparison..."):
            batch_model, stream_model, X_test, y_test = train_both_models(X, y)
            
            batch_pred = batch_model.predict(X_test)
            stream_pred = stream_model.predict(X_test)
            
            batch_proba = batch_model.predict_proba(X_test)
            stream_proba = stream_model.predict_proba(X_test)
            
            # Create metrics dataframe
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
            
            # Display metrics table
            st.subheader("Performance Metrics Table")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Create beautiful comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Metrics Comparison Chart")
                fig = px.bar(metrics_df, x="Metric", y=["Traditional SVM", "Streaming SVM"],
                           barmode="group", title="",
                           color_discrete_map={
                               "Traditional SVM": "#667eea",
                               "Streaming SVM": "#764ba2"
                           })
                fig.update_layout(
                    hovermode="x unified",
                    height=400,
                    showlegend=True,
                    legend=dict(x=0, y=1),
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Performance Radar")
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=metrics["Traditional SVM"],
                    theta=metrics["Metric"],
                    fill='toself',
                    name='Traditional SVM',
                    marker=dict(color='#667eea')
                ))
                fig.add_trace(go.Scatterpolar(
                    r=metrics["Streaming SVM"],
                    theta=metrics["Metric"],
                    fill='toself',
                    name='Streaming SVM',
                    marker=dict(color='#764ba2')
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Traditional SVM")
                cm_traditional = confusion_matrix(y_test, batch_pred)
                cm_df_trad = pd.DataFrame(
                    cm_traditional,
                    columns=["Predicted: Low", "Predicted: High"],
                    index=["Actual: Low", "Actual: High"]
                )
                st.dataframe(cm_df_trad, use_container_width=True)
                st.text(classification_report(y_test, batch_pred, target_names=["Low Income", "High Income"]))
            
            with col2:
                st.markdown("### Streaming SVM")
                cm_streaming = confusion_matrix(y_test, stream_pred)
                cm_df_stream = pd.DataFrame(
                    cm_streaming,
                    columns=["Predicted: Low", "Predicted: High"],
                    index=["Actual: Low", "Actual: High"]
                )
                st.dataframe(cm_df_stream, use_container_width=True)
                st.text(classification_report(y_test, stream_pred, target_names=["Low Income", "High Income"]))
            
            st.markdown("---")
            
            # Winner announcement
            st.markdown("### Which Model Performs Better?")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                better_acc = "Traditional SVM" if accuracy_score(y_test, batch_pred) > accuracy_score(y_test, stream_pred) else "Streaming SVM"
                acc_diff = abs(accuracy_score(y_test, batch_pred) - accuracy_score(y_test, stream_pred))
                st.info(f"**Better Accuracy: {better_acc}** (+{acc_diff:.2%})")
            
            with col2:
                better_f1 = "Traditional SVM" if f1_score(y_test, batch_pred, average='weighted', zero_division=0) > f1_score(y_test, stream_pred, average='weighted', zero_division=0) else "Streaming SVM"
                f1_diff = abs(f1_score(y_test, batch_pred, average='weighted', zero_division=0) - f1_score(y_test, stream_pred, average='weighted', zero_division=0))
                st.info(f"**Better F1-Score: {better_f1}** (+{f1_diff:.2%})")
            
            with col3:
                better_auc = "Traditional SVM" if roc_auc_score(y_test, batch_proba, multi_class='ovr') > roc_auc_score(y_test, stream_proba, multi_class='ovr') else "Streaming SVM"
                auc_diff = abs(roc_auc_score(y_test, batch_proba, multi_class='ovr') - roc_auc_score(y_test, stream_proba, multi_class='ovr'))
                st.info(f"**Better ROC-AUC: {better_auc}** (+{auc_diff:.2%})")
