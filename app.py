import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

st.set_page_config(
    page_title="Crop Yield ML Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Nunito:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --card:      #1a2235;
    --border:    #2a3a55;
    --cyan:      #00e5ff;
    --lime:      #b5ff2d;
    --coral:     #ff5e78;
    --purple:    #b06cff;
    --yellow:    #ffe135;
    --text:      #e8f0fe;
    --muted:     #8899bb;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Nunito', sans-serif;
}
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stToolbar"] { display: none; }
section[data-testid="stSidebar"] { background: var(--surface) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2235 40%, #0b0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,229,255,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(181,255,45,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 3px;
    line-height: 1;
    background: linear-gradient(90deg, var(--cyan), var(--lime));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    font-family: 'Nunito', sans-serif;
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: .5rem;
}

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
    flex-wrap: wrap !important;
}
[data-baseweb="tab"] {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: .82rem !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    letter-spacing: .5px !important;
    text-transform: uppercase !important;
    transition: all .2s ease !important;
}
[data-baseweb="tab"]:hover {
    color: var(--cyan) !important;
    background: rgba(0,229,255,.06) !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,229,255,.15), rgba(181,255,45,.1)) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(0,229,255,.35) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

/* ── Section headings ── */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 2px;
    color: var(--lime);
    margin: 1.5rem 0 .6rem;
}
.section-subtitle {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 1.5px;
    color: var(--cyan);
    margin: 1.2rem 0 .4rem;
}

/* ── Insight / annotation card (text that appears BELOW charts) ── */
.insight-card {
    background: linear-gradient(135deg, var(--card), #1e2d45);
    border-left: 4px solid var(--cyan);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 1rem 0 1.5rem;
    font-size: 1rem;
    color: var(--text);
    line-height: 1.7;
    font-family: 'Nunito', sans-serif;
}
.insight-card.lime   { border-color: var(--lime);   }
.insight-card.coral  { border-color: var(--coral);  }
.insight-card.purple { border-color: var(--purple); }
.insight-card.yellow { border-color: var(--yellow); }

/* ── Metric grid ── */
.metric-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    min-width: 130px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-box .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 1px;
}
.metric-box .split { font-size: .85rem; color: var(--muted); margin-top: .1rem; }

/* ── Data table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Matplotlib dark theme override ── */
.stPlotlyChart, .stImage { border-radius: 12px; }

/* ── Divider ── */
.fancy-hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--lime), transparent);
    margin: 1.5rem 0;
}

/* ── Tag chip ── */
.chip {
    display: inline-block;
    background: rgba(0,229,255,.12);
    border: 1px solid rgba(0,229,255,.3);
    color: var(--cyan);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}
.chip.lime   { background: rgba(181,255,45,.12); border-color: rgba(181,255,45,.3); color: var(--lime); }
.chip.coral  { background: rgba(255,94,120,.12); border-color: rgba(255,94,120,.3); color: var(--coral); }
.chip.purple { background: rgba(176,108,255,.12); border-color: rgba(176,108,255,.3); color: var(--purple); }

/* ── Upload notice ── */
.upload-notice {
    background: rgba(255,94,120,.08);
    border: 1px dashed var(--coral);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: var(--coral);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#1a2235",
    "axes.edgecolor":    "#2a3a55",
    "axes.labelcolor":   "#b0c4de",
    "axes.titlecolor":   "#e8f0fe",
    "xtick.color":       "#8899bb",
    "ytick.color":       "#8899bb",
    "text.color":        "#e8f0fe",
    "grid.color":        "#1e2d40",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#111827",
    "legend.edgecolor":  "#2a3a55",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
ACCENT_PALETTE = ["#00e5ff", "#b5ff2d", "#ff5e78", "#b06cff", "#ffe135", "#ff9f43"]
sns.set_palette(ACCENT_PALETTE)

st.markdown("""
<div class="hero">
  <p class="hero-title">🌾 Crop Yield ML Explorer</p>
  <p class="hero-sub">
    An end-to-end machine-learning walkthrough — from raw data to tuned models —
    for predicting agricultural crop yield.
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean():
    data = pd.read_csv("yield_df.csv")
    if "Unnamed: 0" in data.columns:
        data.drop(columns="Unnamed: 0", inplace=True)
    return data

@st.cache_data
def build_features(_data):
    data = _data.copy()
    # Preserve original categorical labels before encoding (for error analysis)
    original_labels = _data[["Area", "Item"]].copy()
    encoder = TargetEncoder(cols=["Area", "Item"])
    data[["Area", "Item"]] = encoder.fit_transform(data[["Area", "Item"]], data["hg/ha_yield"])
    X = data.drop(columns="hg/ha_yield")
    y = data["hg/ha_yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Align original labels to the test split using its index, then reset so it
    # stays in sync with the numpy arrays returned below
    test_labels = original_labels.loc[X_test.index].reset_index(drop=True)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return data, X_train_s, X_test_s, y_train, y_test, test_labels

@st.cache_resource
def train_all(X_train, X_test, y_train, y_test):
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["lr"] = lr

    # Polynomial (degree 2)
    poly2 = PolynomialFeatures(degree=2)
    Xp2   = poly2.fit_transform(X_train)
    poly_m = LinearRegression()
    poly_m.fit(Xp2, y_train)
    models["poly2"]   = poly_m
    models["poly2_t"] = poly2

    # Ridge (degree 2)
    ridge = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler_post_poly", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge.fit(X_train, y_train)
    models["ridge"] = ridge

    # Lasso (degree 2)
    lasso = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler_post_poly", StandardScaler()),
        ("lasso", Lasso(alpha=1.0)),
    ])
    lasso.fit(X_train, y_train)
    models["lasso"] = lasso

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=5)
    dt.fit(X_train, y_train)
    models["dt"] = dt

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["rf"] = rf

    # Ridge Tuned (degree 7, alpha 0.591)
    ridge_tuned = Pipeline([
        ("poly", PolynomialFeatures(degree=7)),
        ("scaler_post_poly", StandardScaler()),
        ("ridge", Ridge(alpha=0.591)),
    ])
    ridge_tuned.fit(X_train, y_train)
    models["ridge_tuned"] = ridge_tuned

    return models

def metrics_html(y_true, y_pred, label="Test", accent="#00e5ff"):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="label">R² — {label}</div>
        <div class="value" style="color:{accent}">{r2:.4f}</div>
      </div>
      <div class="metric-box">
        <div class="label">MAE — {label}</div>
        <div class="value" style="color:{accent}">{mae:,.0f}</div>
      </div>
      <div class="metric-box">
        <div class="label">MSE — {label}</div>
        <div class="value" style="color:{accent}">{mse:,.0f}</div>
      </div>
      <div class="metric-box">
        <div class="label">RMSE — {label}</div>
        <div class="value" style="color:{accent}">{rmse:,.0f}</div>
      </div>
    </div>"""

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs. Predicted"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#00e5ff", s=18, label="Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#ff5e78", lw=2, label="Perfect Fit")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_residuals(y_test, y_pred, title="Residual Plot"):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(y=0, color="#ff5e78", linestyle="--", lw=2)
    ax.scatter(y_pred, residuals, alpha=0.4, color="#b5ff2d", s=18)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Errors)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

@st.cache_data
def compute_learning_curve(_model, X_train, y_train, model_key: str):
    """Cache the expensive CV computation so it only runs once per model."""
    sizes, tr_scores, val_scores = learning_curve(
        _model, X_train, y_train,
        cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    return sizes, tr_scores, val_scores

def plot_learning_curve(model, X_train, y_train, title="Learning Curve", model_key: str = ""):
    sizes, tr_scores, val_scores = compute_learning_curve(model, X_train, y_train, model_key)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tr_scores.mean(axis=1),  label="Training Score",   color="#00e5ff", lw=2, marker="o")
    ax.plot(sizes, val_scores.mean(axis=1), label="Validation Score", color="#b5ff2d", lw=2, marker="s")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def fancy_hr():
    st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)

def insight(text, color=""):
    cls = f"insight-card {color}".strip()
    st.markdown(f'<div class="{cls}"> {text}</div>', unsafe_allow_html=True)

try:
    raw_data = load_and_clean()
    data_ready = True
except FileNotFoundError:
    data_ready = False

if not data_ready:
    st.markdown("""
    <div class="upload-notice">
        <strong>yield_df.csv</strong> not found in the working directory.<br>
      Place the CSV file alongside <code>app.py</code> and restart the app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

feat_data, X_train, X_test, y_train, y_test, test_labels = build_features(raw_data)

with st.spinner("Training all models — hang tight..."):
    models = train_all(X_train, X_test, y_train, y_test)


tabs = st.tabs([
    "Overview",
    "Data Cleaning",
    "EDA",
    "Linear Separability",
    "Feature Engineering",
    "Linear Regression",
    "Polynomial Regression",
    "Ridge Regularization",
    "Lasso Regularization",
    "Decision Tree",
    "Random Forest",
    "Hyperparameter Tuning",
    "Error Analysis",
])

with tabs[0]:
    st.markdown('<p class="section-title">Project Overview</p>', unsafe_allow_html=True)

    # Problem Statement
    st.markdown("""
    <div class="insight-card lime">
        <strong style="font-size:1.1rem; font-family:'Bebas Neue',sans-serif; letter-spacing:1.5px;">
             PROBLEM STATEMENT
        </strong><br><br>
        Farmers experience fluctuating agricultural productivity due to <strong>climate variability</strong>,
        <strong>rainfall inconsistency</strong>, <strong>soil degradation</strong>, and changing farming practices.
        Accurate crop yield prediction can help farmers, policymakers, and agricultural organizations
        improve planning and food security.
    </div>
    """, unsafe_allow_html=True)

    fancy_hr()

    # Description
    st.markdown('<p class="section-subtitle">Description</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        Predict crop yield using a combination of environmental and agricultural features drawn from
        historical records spanning <strong>1990 – 2013</strong>. The features used are:
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box" style="text-align:left; padding:1.2rem 1.4rem;">
            <div class="label">Numerical Features</div>
            <ul style="margin:.6rem 0 0; padding-left:1.2rem; color:var(--text); font-size:.95rem; line-height:2;">
                <li> Rainfall</li>
                <li> Pesticide use</li>
                <li> Average temperature</li>
                <li> Years: 1990 – 2013</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box" style="text-align:left; padding:1.2rem 1.4rem;">
            <div class="label">Categorical Features</div>
            <ul style="margin:.6rem 0 0; padding-left:1.2rem; color:var(--text); font-size:.95rem; line-height:2;">
                <li> Type of crop</li>
                <li> Location / Area</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box" style="text-align:left; padding:1.2rem 1.4rem;">
            <div class="label">Target Variable</div>
            <ul style="margin:.6rem 0 0; padding-left:1.2rem; color:var(--text); font-size:.95rem; line-height:2;">
                <li> Historical yield data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    fancy_hr()

    # Dataset
    st.markdown('<p class="section-subtitle">Dataset</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card purple">
        <strong style="font-family:'Bebas Neue',sans-serif; letter-spacing:1.2px; font-size:1.05rem;">
             yield_df — Kaggle
        </strong><br><br>
        The <code>yield_df</code> dataset is sourced from <strong>Kaggle</strong> and contains global
        crop yield records compiled from the FAO (Food and Agriculture Organization of the United Nations).
        It covers multiple crops across numerous countries, paired with climate and agricultural input data
        for each year.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Source", "Kaggle / FAO")
    col_b.metric("Time Span", "1990 – 2013")
    col_c.metric("Target", "hg/ha Yield")
    col_d.metric("Task Type", "Regression")

    fancy_hr()


with tabs[1]:
    st.markdown('<p class="section-title">Data Cleaning</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{raw_data.shape[0]:,}")
    col2.metric("Columns", raw_data.shape[1])
    col3.metric("Missing Values", int(raw_data.isnull().sum().sum()))

    fancy_hr()

    st.markdown('<p class="section-subtitle">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(raw_data.head(10), use_container_width=True)

    st.markdown('<p class="section-subtitle">Column Info</p>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column":   raw_data.columns,
        "Dtype":    raw_data.dtypes.values.astype(str),
        "Non-Null": raw_data.notnull().sum().values,
        "Nulls":    raw_data.isnull().sum().values,
    })
    st.dataframe(info_df, use_container_width=True)

    insight(
        "The <code>Unnamed: 0</code> column was dropped — it was a redundant index artifact "
        "from the original CSV export and adds no predictive value.",
        color="lime"
    )

with tabs[2]:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Value counts
    st.markdown('<p class="section-subtitle">Value Counts</p>', unsafe_allow_html=True)
    vc_col = st.selectbox("Choose a column", ["Item", "Area", "Year"])
    vc = raw_data[vc_col].value_counts().head(20)
    vc_df = vc.reset_index()
    vc_df.columns = [vc_col, "Count"]
    st.dataframe(vc_df, use_container_width=True, hide_index=True)

    fancy_hr()

    st.markdown('<p class="section-subtitle">Numeric Distributions</p>', unsafe_allow_html=True)
    numeric_cols = raw_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_per_row = 2
    pairs = [numeric_cols[i:i+cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]
    for pair in pairs:
        grid = st.columns(len(pair))
        for ax_col, col in zip(grid, pair):
            with ax_col:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.histplot(raw_data[col], kde=True, ax=ax, color="#00e5ff", alpha=0.6)
                ax.set_title(col, fontweight="bold")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    insight(
        "After reviewing column types and missing values (none found), distributions of numeric columns "
        "reveal clear left- or right-skewness — important context for choosing the right scaler later.",
        color="coral"
    )

    fancy_hr()

    st.markdown('<p class="section-subtitle">Correlation Heatmap</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        raw_data.corr(numeric_only=True),
        annot=True, fmt=".2f",
        cmap="coolwarm", ax=ax,
        linewidths=0.5, linecolor="#0b0f1a",
    )
    ax.set_title("Correlation Heatmap", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    insight(
        "No feature pairs show strong multicollinearity (high positive or negative correlation). "
        "This initially suggests a linear model could perform well on this dataset.",
        color="purple"
    )

with tabs[3]:
    st.markdown('<p class="section-title">Linear Separability</p>', unsafe_allow_html=True)
    st.info("Generating pairplot — this may take a few seconds...")

    pairplot_vars = ["hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    sample = raw_data.sample(min(800, len(raw_data)), random_state=42)
    fig = sns.pairplot(
        sample,
        hue="Item",
        vars=pairplot_vars,
        palette="viridis",
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 15},
    )
    fig.figure.set_facecolor("#111827")
    for ax in fig.axes.flatten():
        if ax:
            ax.set_facecolor("#1a2235")
    st.pyplot(fig.figure, use_container_width=True)
    plt.close(fig.figure)

    insight(
        "The pairplot reveals a <strong>non-linear relationship</strong> between features and the target "
        "variable <code>hg/ha_yield</code>. The scattered, curved point clouds across crop types suggest "
        "that a polynomial regression model may capture the underlying patterns more effectively than "
        "a simple linear model.",
        color="lime"
    )

with tabs[4]:
    st.markdown('<p class="section-title">Feature Engineering</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-subtitle">Encoded Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(feat_data.head(10), use_container_width=True)

    st.markdown('<p class="section-subtitle">Column Types After Encoding</p>', unsafe_allow_html=True)
    info_df2 = pd.DataFrame({
        "Column": feat_data.columns,
        "Dtype":  feat_data.dtypes.values.astype(str),
    })
    st.dataframe(info_df2, use_container_width=True)

    st.markdown(
        '<span class="chip">Area</span> <span class="chip lime">Item</span> '
        '<span class="chip coral">→ Target Encoded</span>',
        unsafe_allow_html=True
    )

    insight(
        "Both <code>Area</code> and <code>Item</code> have high cardinality (many unique values). "
        "<strong>Target Encoding</strong> was chosen over one-hot encoding for this reason",
        color="yellow"
    )

with tabs[5]:
    st.markdown('<p class="section-title">Linear Regression</p>', unsafe_allow_html=True)

    lr   = models["lr"]
    tr_p = lr.predict(X_train)
    te_p = lr.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_p, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_p, "Test",  "#00e5ff"), unsafe_allow_html=True)

    insight(
        "Similar R² scores for both training and testing (0.73) confirm the model is neither "
        "overfitting nor underfitting; a healthy generalisation. However, 0.73 leaves room for "
        "improvement.",
        color=""
    )

    fancy_hr()

    st.markdown('<p class="section-subtitle">Actual vs. Predicted</p>', unsafe_allow_html=True)
    plot_actual_vs_predicted(y_test, te_p)
    insight(
        "Predictions cluster well around the diagonal for lower yield values, but the model "
        "<strong>under-predicts for high actual values</strong> — the scatter fans out above the "
        "line at the right. This systematic bias hints at non-linearity.",
        color="coral"
    )

    fancy_hr()

    st.markdown('<p class="section-subtitle">Residual Plot</p>', unsafe_allow_html=True)
    plot_residuals(y_test, te_p)
    insight(
        "The residuals form an approximate <strong>U-shaped</strong> pattern around zero rather "
        "than a random cloud. This structured pattern is a diagnostic signal that the underlying "
        "relationship is <em>not</em> linear — polynomial or tree-based models are warranted.",
        color="purple"
    )

with tabs[6]:
    st.markdown('<p class="section-title">Polynomial Regression</p>', unsafe_allow_html=True)

    poly2   = models["poly2_t"]
    poly_m  = models["poly2"]
    tr_pp   = poly_m.predict(poly2.transform(X_train))
    te_pp   = poly_m.predict(poly2.transform(X_test))

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_pp, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_pp, "Test",  "#00e5ff"), unsafe_allow_html=True)

    insight(
        "Stepping up to degree-2 polynomial features boosts R² by almost <strong>+0.10</strong> — "
        "a meaningful gain that confirms the non-linear structure seen in the residuals.",
        color="lime"
    )

    fancy_hr()

    st.markdown('<p class="section-subtitle">Residual Plot</p>', unsafe_allow_html=True)
    plot_residuals(y_test, te_pp)
    insight(
        "With polynomial features the residuals are now scattered more <strong>randomly around zero</strong> "
        "— the U-shaped bias from linear regression is gone. The model fits the data structure much better.",
        color="coral"
    )

    fancy_hr()

    st.markdown('<p class="section-subtitle">Learning Curve</p>', unsafe_allow_html=True)
    plot_learning_curve(poly_m, poly2.transform(X_train), y_train, "Learning Curve — Polynomial (degree 2)", model_key="poly2")
    insight(
        "Training and validation scores converge as dataset size grows, indicating good "
        "generalisation without severe overfitting at degree 2.",
        color="purple"
    )

with tabs[7]:
    st.markdown('<p class="section-title">Ridge Regularization</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="chip">Degree 2</span> <span class="chip lime">L2 Penalty</span> <span class="chip coral">α = 1.0</span>',
        unsafe_allow_html=True
    )

    ridge  = models["ridge"]
    tr_r   = ridge.predict(X_train)
    te_r   = ridge.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_r, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_r, "Test",  "#00e5ff"), unsafe_allow_html=True)

    fancy_hr()
    st.markdown('<p class="section-subtitle">Learning Curve</p>', unsafe_allow_html=True)
    plot_learning_curve(ridge, X_train, y_train, "Learning Curve — Polynomial Ridge (degree 2)", model_key="ridge")
    insight(
        "Ridge regression adds an L2 penalty that shrinks large coefficients without eliminating them. "
        "Using degree-2 polynomial features with Ridge helps capture further non-linearity while "
        "keeping the model stable.",
        color="purple"
    )

with tabs[8]:
    st.markdown('<p class="section-title">Lasso Regularization</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="chip">Degree 2</span> <span class="chip lime">L1 Penalty</span> <span class="chip coral">α = 1.0</span>',
        unsafe_allow_html=True
    )

    lasso  = models["lasso"]
    tr_l   = lasso.predict(X_train)
    te_l   = lasso.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_l, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_l, "Test",  "#00e5ff"), unsafe_allow_html=True)

    insight(
        "Lasso applies an L1 penalty that can zero out entire feature coefficients, acting as "
        "automatic feature selection. At α = 1.0 some polynomial terms are eliminated, which can "
        "hurt performance if those terms carried genuine signal.",
        color="yellow"
    )

with tabs[9]:
    st.markdown('<p class="section-title">Decision Tree</p>', unsafe_allow_html=True)
    st.markdown('<span class="chip">max_depth = 5</span>', unsafe_allow_html=True)

    dt    = models["dt"]
    tr_dt = dt.predict(X_train)
    te_dt = dt.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_dt, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_dt, "Test",  "#00e5ff"), unsafe_allow_html=True)

    fancy_hr()

    st.markdown('<p class="section-subtitle">Feature Importances</p>', unsafe_allow_html=True)
    feat_names = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    fi = pd.Series(dt.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(fi.index, fi.values, color=ACCENT_PALETTE[:len(fi)])
    ax.set_xlabel("Importance")
    ax.set_title("Decision Tree — Feature Importances", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    #insight(
        #"A shallow decision tree (depth 5) is interpretable and fast but may miss complex "
        #"interactions.",
        #color="lime"
    #)

with tabs[10]:
    st.markdown('<p class="section-title">Random Forest</p>', unsafe_allow_html=True)
    st.markdown('<span class="chip">100 estimators</span> <span class="chip lime">random_state = 42</span>', unsafe_allow_html=True)

    rf    = models["rf"]
    tr_rf = rf.predict(X_train)
    te_rf = rf.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_rf, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_rf, "Test",  "#00e5ff"), unsafe_allow_html=True)

    fancy_hr()

    st.markdown('<p class="section-subtitle">Feature Importances</p>', unsafe_allow_html=True)
    fi_rf = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(fi_rf.index, fi_rf.values, color=ACCENT_PALETTE[:len(fi_rf)])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fancy_hr()
    st.markdown('<p class="section-subtitle">Learning Curve</p>', unsafe_allow_html=True)
    plot_learning_curve(rf, X_train, y_train, "Learning Curve — Random Forest", model_key="rf")
    insight(
        "The Random Forest ensembles many decision trees, reducing variance and achieving "
        "higher accuracy than a single tree. The learning curve shows the model is generalising "
        "well",
        color="coral"
    )

with tabs[11]:
    st.markdown('<p class="section-title">Hyperparameter Tuning</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card purple">
     <strong>Strategy:</strong> RandomizedSearchCV with 10 iterations and 5-fold cross-validation,
    searching over polynomial degree (1–9) and Ridge alpha (0.01–10.01).
    The best configuration found was <strong>degree = 7, α ≈ 0.591</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<span class="chip purple">degree = 7</span> <span class="chip lime">α = 0.591</span> <span class="chip coral">CV folds = 5</span>',
        unsafe_allow_html=True
    )

    ridge_t  = models["ridge_tuned"]
    tr_rt    = ridge_t.predict(X_train)
    te_rt    = ridge_t.predict(X_test)

    st.markdown('<p class="section-subtitle">Training Metrics — Tuned Ridge</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_train, tr_rt, "Train", "#b5ff2d"), unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Testing Metrics — Tuned Ridge</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test,  te_rt, "Test",  "#00e5ff"), unsafe_allow_html=True)

    fancy_hr()

    st.markdown('<p class="section-subtitle">Model Comparison</p>', unsafe_allow_html=True)
    comparison = {
        "Linear Regression":       r2_score(y_test, models["lr"].predict(X_test)),
        "Polynomial (deg 2)":      r2_score(y_test, models["poly2"].predict(models["poly2_t"].transform(X_test))),
        "Ridge (deg 2)":           r2_score(y_test, models["ridge"].predict(X_test)),
        "Lasso (deg 2)":           r2_score(y_test, models["lasso"].predict(X_test)),
        "Decision Tree":           r2_score(y_test, models["dt"].predict(X_test)),
        "Random Forest":           r2_score(y_test, models["rf"].predict(X_test)),
        "Ridge Tuned (deg 7)":     r2_score(y_test, te_rt),
    }
    comp_df = pd.DataFrame.from_dict(comparison, orient="index", columns=["Test R²"]).sort_values("Test R²")
    colors  = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(comp_df))]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(comp_df.index, comp_df["Test R²"], color=colors)
    ax.set_xlabel("Test R²")
    ax.set_title("Model Comparison — Test R²", fontweight="bold")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, comp_df["Test R²"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#e8f0fe")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    insight(
        "Hyperparameter tuning pushed the Ridge model to its best configuration. "
        "The bar chart above shows the final ranking of all models by test R² — "
        "a clear summary of every approach explored in this notebook.",
        color="lime"
    )

    fancy_hr()
    st.markdown('<p class="section-subtitle">Learning Curve — Tuned Ridge</p>', unsafe_allow_html=True)
    plot_learning_curve(ridge_t, X_train, y_train, "Learning Curve — Polynomial Ridge Tuned (degree 7)", model_key="ridge_tuned")
    insight(
        "The learning curve for the tuned model shows generalising well as the curves converge at a "
        "high value for the R² score",
        color="coral"
    )

with tabs[12]:
    st.markdown('<p class="section-title">Error Analysis</p>', unsafe_allow_html=True)

    model_options = {
        "Ridge Tuned (deg 7)": "ridge_tuned",
        "Random Forest":                    "rf",
        "Ridge (deg 2)":                    "ridge",
        "Polynomial (deg 2)":               "poly2",
        "Decision Tree":                    "dt",
        "Linear Regression":                "lr",
        "Lasso (deg 2)":                    "lasso",
    }
    chosen_label = st.selectbox(
        "Select model to analyse",
        list(model_options.keys()),
        index=0,
    )
    chosen_key = model_options[chosen_label]


    if chosen_key == "poly2":
        ea_preds = models["poly2"].predict(models["poly2_t"].transform(X_test))
    else:
        ea_preds = models[chosen_key].predict(X_test)

    y_test_arr   = np.array(y_test)
    abs_errors   = np.abs(y_test_arr - ea_preds)
    signed_errors = y_test_arr - ea_preds

    ea_df = test_labels.copy().reset_index(drop=True)
    ea_df["Actual"]       = y_test_arr
    ea_df["Predicted"]    = ea_preds
    ea_df["Abs_Error"]    = abs_errors
    ea_df["Signed_Error"] = signed_errors


    st.markdown('<p class="section-subtitle">Overall Error Metrics</p>', unsafe_allow_html=True)
    st.markdown(metrics_html(y_test, ea_preds, "Test", "#b06cff"), unsafe_allow_html=True)
    fancy_hr()

    top_n = st.slider("Number of top groups to display", min_value=5, max_value=30, value=15, step=5)

    st.markdown('<p class="section-subtitle">Error by Crop Item</p>', unsafe_allow_html=True)

    item_err = (
        ea_df.groupby("Item")
        .agg(
            MAE=("Abs_Error", "mean"),
            RMSE=("Abs_Error", lambda x: np.sqrt((x**2).mean())),
            Count=("Abs_Error", "count"),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )

    st.markdown("**Top items by MAE (highest error first)**")
    top_items = item_err.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3.5, top_n * 0.32)))
    colors_bar = ["#ff5e78" if i < 5 else "#b06cff" if i < 10 else "#00e5ff"
                  for i in range(len(top_items))]
    bars = ax.barh(top_items["Item"][::-1], top_items["MAE"][::-1], color=colors_bar[::-1])
    ax.set_xlabel("Mean Absolute Error (hg/ha)")
    ax.set_title(f"Top {top_n} Crop Items — MAE", fontweight="bold")
    for bar, val in zip(bars, top_items["MAE"][::-1]):
        ax.text(bar.get_width() + top_items["MAE"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8, color="#e8f0fe")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Item Error Summary Table**")
    st.dataframe(
        item_err.style.background_gradient(subset=["MAE", "RMSE"], cmap="Reds"),
        use_container_width=True, hide_index=True
    )

    insight(
        "Crops with the <strong>highest MAE</strong> are where the model struggles most — "
        "often items with high yield variance or sparse training samples.",
        color="coral"
    )
    fancy_hr()


    st.markdown('<p class="section-subtitle">Error by Area / Region</p>', unsafe_allow_html=True)

    area_err = (
        ea_df.groupby("Area")
        .agg(
            MAE=("Abs_Error", "mean"),
            RMSE=("Abs_Error", lambda x: np.sqrt((x**2).mean())),
            Count=("Abs_Error", "count"),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )

    st.markdown("**Top areas by MAE (highest error first)**")
    top_areas = area_err.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3.5, top_n * 0.32)))
    area_colors = ["#ff5e78" if i < 5 else "#b06cff" if i < 10 else "#00e5ff"
                   for i in range(len(top_areas))]
    bars = ax.barh(top_areas["Area"][::-1], top_areas["MAE"][::-1], color=area_colors[::-1])
    ax.set_xlabel("Mean Absolute Error (hg/ha)")
    ax.set_title(f"Top {top_n} Areas — MAE", fontweight="bold")
    for bar, val in zip(bars, top_areas["MAE"][::-1]):
        ax.text(bar.get_width() + top_areas["MAE"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8, color="#e8f0fe")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Area Error Summary Table**")
    st.dataframe(
        area_err.style.background_gradient(subset=["MAE", "RMSE"], cmap="Reds"),
        use_container_width=True, hide_index=True
    )

    #insight(
        #"Areas with high MAE often reflect regions where crop yield varies widely across "
        #"different crop types, or where training data is sparse. "
        #"Cross-reference with the Item breakdown above to identify crop–region combinations "
        #"that are driving the largest errors.",
        #color="purple"
    #)
    fancy_hr()


    st.markdown('<p class="section-subtitle">Worst Individual Predictions</p>', unsafe_allow_html=True)

    worst_n = st.slider("Show worst N predictions", min_value=10, max_value=100, value=20, step=10)
    worst_df = (
        ea_df[["Item", "Area", "Actual", "Predicted", "Abs_Error"]]
        .sort_values("Abs_Error", ascending=False)
        .head(worst_n)
        .reset_index(drop=True)
    )
    worst_df.index += 1
    worst_df.columns = ["Crop Item", "Area", "Actual (hg/ha)", "Predicted (hg/ha)",
                         "Abs Error"]
    st.dataframe(
        worst_df.style
            .background_gradient(subset=["Abs Error"], cmap="Reds")
            .format({"Actual (hg/ha)": "{:,.0f}", "Predicted (hg/ha)": "{:,.0f}",
                     "Abs Error": "{:,.0f}"}),
        use_container_width=True
    )

    #insight(
        #"The table above lists the individual rows where the model is farthest from the truth. "
        #"Use these as diagnostic cases — cross-reference the Crop Item and Area columns with "
        #"the grouped bar charts above to confirm whether the errors are systemic (entire crop/region "
        #"is hard to model) or isolated outliers.",
        #color="yellow"
    #)