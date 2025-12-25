import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Tunisie x Data Science – Dashboard CAN",
    page_icon="🇹🇳",
    layout="wide",
)

# CSS dark + effet stade
st.markdown(
    """
    <style>
    body {
        background-color: #050814;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3, h4, h5 {
        color: #F5F5F5;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-label, .metric-value {
        color: #F5F5F5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
@st.cache_data
def load_data(path="results.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# Filtre Tunisie
df_tunisie = df[(df["home_team"] == "Tunisia") | (df["away_team"] == "Tunisia")].copy()
df_tunisie["date"] = pd.to_datetime(df_tunisie["date"])
df_tunisie = df_tunisie[df_tunisie["date"] >= "2020-01-01"].sort_values("date")
df_tunisie.reset_index(drop=True, inplace=True)

# =========================
# 2. FEATURE ENGINEERING
# =========================

# Résultat textuel
def match_result(row):
    if row['home_team'] == 'Tunisia':
        if row['home_score'] > row['away_score']:
            return "Win"
        elif row['home_score'] < row['away_score']:
            return "Loss"
        else:
            return "Draw"
    else:
        if row['away_score'] > row['home_score']:
            return "Win"
        elif row['away_score'] < row['home_score']:
            return "Loss"
        else:
            return "Draw"

df_tunisie["result"] = df_tunisie.apply(match_result, axis=1)

# Adversaire
df_tunisie["opponent"] = df_tunisie.apply(
    lambda row: row["away_team"] if row["home_team"] == "Tunisia" else row["home_team"],
    axis=1
)

# Variables de score
df_tunisie["goals_scored"] = df_tunisie.apply(
    lambda row: row["home_score"] if row["home_team"] == "Tunisia" else row["away_score"],
    axis=1,
)
df_tunisie["goals_against"] = df_tunisie.apply(
    lambda row: row["away_score"] if row["home_team"] == "Tunisia" else row["home_score"],
    axis=1,
)
df_tunisie["goal_diff"] = df_tunisie["goals_scored"] - df_tunisie["goals_against"]

# Résultat encodé
label_map = {"Loss": 0, "Draw": 1, "Win": 2}
df_tunisie["result_encoded"] = df_tunisie["result"].map(label_map)
df_tunisie["win"] = (df_tunisie["result"] == "Win").astype(int)

# Forme récente (5 derniers matchs avant)
df_tunisie = df_tunisie.sort_values("date")
df_tunisie["form"] = df_tunisie["result"].eq("Win").shift(1).rolling(5).mean()

# Match à domicile
df_tunisie["is_home"] = (df_tunisie["home_team"] == "Tunisia").astype(int)

# Match officiel
df_tunisie["official"] = (df_tunisie["tournament"] != "Friendly").astype(int)

# Force de l’adversaire (simple proxy : buts marqués moyens à domicile)
team_strength = df.groupby("home_team")["home_score"].mean()
df_tunisie["opponent_strength"] = df_tunisie["opponent"].map(team_strength)

# Match en Afrique
# Tous les pays d'Afrique (CAF)
africa = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cameroon", "Central African Republic", "Chad",
    "Comoros", "Republic of the Congo", "Democratic Republic of the Congo",
    "Ivory Coast", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
    "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia",
    "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
    "Tunisia", "Uganda", "Zambia", "Zimbabwe"
]

def travel_level(country):
    if country == "Tunisia":
        return 0
    elif country in africa:
        return 1
    return 2

df_tunisie["in_africa"] = df_tunisie["country"].isin(africa).astype(int)
df_tunisie["travel"] = df_tunisie["country"].apply(travel_level)

# Rolling offensif/défensif
df_tunisie["attack_5"] = df_tunisie["goals_scored"].shift(1).rolling(5).mean()
df_tunisie["defense_5"] = df_tunisie["goals_against"].shift(1).rolling(5).mean()

# Head-to-head win rate
h2h = df_tunisie.groupby("opponent")["win"].mean()
df_tunisie["h2h_rate"] = df_tunisie["opponent"].map(h2h)

# Nettoyage : on enlève les premières lignes avec NaN
df_tunisie = df_tunisie.dropna().reset_index(drop=True)

# =========================
# 3. MODELE ML (XGBoost)
# =========================

features = [
    "is_home",
    "form",
    "official",
    "opponent_strength",
    "travel",
    "attack_5",
    "defense_5",
    "h2h_rate",
]

X = df_tunisie[features]
y = df_tunisie["result_encoded"]

train_size = int(len(df_tunisie) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    eval_metric="mlogloss",
    use_label_encoder=False,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("🇹🇳 Tunisie x Data Science")
page = st.sidebar.radio(
    "Navigation",
    ["🏟 Overview", "📊 Performance", "🤖 Modèle & Prédictions", "ℹ️ À propos"],
)

# =========================
# PAGE 1 : OVERVIEW
# =========================
if page == "🏟 Overview":
    st.title("🇹🇳 Tunisie – Dashboard Football Dark Mode")

    col1, col2, col3, col4 = st.columns(4)
    total_matches = len(df_tunisie)
    wins = (df_tunisie["result"] == "Win").sum()
    draws = (df_tunisie["result"] == "Draw").sum()
    losses = (df_tunisie["result"] == "Loss").sum()

    col1.metric("Matchs analysés (2020–2025)", total_matches)
    col2.metric("Victoires", wins)
    col3.metric("Nuls", draws)
    col4.metric("Défaites", losses)

    # Pie chart win/draw/loss
    counts = df_tunisie["result"].value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={"Win": "#2ecc71", "Draw": "#f1c40f", "Loss": "#e74c3c"},
        hole=0.45,
        title="Répartition des résultats (2020–2025)",
        template="plotly_dark",
    )
    fig_pie.update_layout(
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        """
        **Highlights :**
        - La Tunisie remporte une majorité de ses matchs depuis 2020.
        - Les défaites sont relativement rares, mais souvent liées aux matchs à l’extérieur ou aux gros adversaires.
        - Ce dashboard sert de base pour analyser la dynamique avant la CAN.
        """
    )

# =========================
# PAGE 2 : PERFORMANCE
# =========================
elif page == "📊 Performance":
    st.title("📊 Performance de la Tunisie – Vue Data")

    # Résultats par année
    df_tunisie["year"] = df_tunisie["date"].dt.year
    results_by_year = (
        df_tunisie.groupby(["year", "result"]).size().unstack(fill_value=0)
    )
    results_by_year = results_by_year[["Win", "Draw", "Loss"]]

    fig_year = px.bar(
        results_by_year,
        x=results_by_year.index,
        y=["Win", "Draw", "Loss"],
        title="Résultats par année",
        labels={"value": "Nombre de matchs", "year": "Année"},
        template="plotly_dark",
    )
    fig_year.update_layout(
        barmode="stack",
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5",
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # Différence de buts dans le temps
    fig_goals = px.line(
        df_tunisie,
        x="date",
        y="goal_diff",
        title="Évolution de la différence de buts",
        template="plotly_dark",
        markers=True,
    )
    fig_goals.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_goals.update_layout(
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5",
    )
    st.plotly_chart(fig_goals, use_container_width=True)

    # Domicile vs extérieur
    df_tunisie["Lieu"] = np.where(df_tunisie["is_home"] == 1, "Domicile", "Extérieur")
    perf_home = df_tunisie.groupby(["Lieu", "result"]).size().unstack(fill_value=0)

    fig_home = px.bar(
        perf_home,
        x=perf_home.index,
        y=["Win", "Draw", "Loss"],
        title="Résultats à domicile vs extérieur",
        template="plotly_dark",
    )
    fig_home.update_layout(
        barmode="stack",
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5",
    )
    st.plotly_chart(fig_home, use_container_width=True)

# =========================
# PAGE 3 : MODELE & PREDS
# =========================
elif page == "🤖 Modèle & Prédictions":
    st.title("🤖 Modèle prédictif – Tunisie x XGBoost")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy (test)", f"{acc*100:.1f} %")
    col2.metric("Macro F1 (test)", f"{macro_f1*100:.1f} %")

    st.subheader("Importance des variables")
    importance = model.feature_importances_
    feat_importance = pd.DataFrame(
        {"feature": features, "importance": importance}
    ).sort_values("importance", ascending=False)

    fig_imp = px.bar(
        feat_importance,
        x="importance",
        y="feature",
        orientation="h",
        template="plotly_dark",
        title="Facteurs qui influencent le plus le résultat",
    )
    fig_imp.update_layout(
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🔮 Simulation de prédiction sur un match réel (dans le set de test)")

    # On crée un DataFrame de test lisible
    compare_df = pd.DataFrame({
        "Date": df_tunisie["date"].iloc[train_size:],
        "Adversaire": df_tunisie["opponent"].iloc[train_size:],
        "Vrai": y_test.replace({0: "Loss", 1: "Draw", 2: "Win"}).values,
        "Prédit": (
            pd.Series(y_pred)
            .astype(int)
            .replace({0: "Loss", 1: "Draw", 2: "Win"})
            .values
        ),
        "Proba_Win": y_proba[:, 2],
    }).reset_index(drop=True)

    match_idx = st.slider(
        "Choisir un match du set de test",
        min_value=0,
        max_value=len(compare_df) - 1,
        value=0,
    )
    row = compare_df.iloc[match_idx]

    st.write(
        f"**Match du {row['Date'].date()} vs {row['Adversaire']}**"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Résultat réel", row["Vrai"])
    col_b.metric("Prédiction modèle", row["Prédit"])
    col_c.metric("Proba de victoire", f"{row['Proba_Win']*100:.1f} %")

    # Bar chart des proba complètes
    probs_row = y_proba[match_idx]
    prob_df = pd.DataFrame({
        "Resultat": ["Loss", "Draw", "Win"],
        "Proba": probs_row
    })
    fig_prob = px.bar(
        prob_df,
        x="Resultat",
        y="Proba",
        template="plotly_dark",
        title="Distribution des probabilités pour ce match",
    )
    fig_prob.update_layout(
        yaxis_tickformat=".0%",
        paper_bgcolor="#050814",
        plot_bgcolor="#050814",
        font_color="#F5F5F5",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # =========================
    # 🔮 PREDICTION MATCH FUTUR (SCENARIO CAN)
    # =========================

    st.markdown("---")
    st.subheader("🏆 Scénario CAN – Prédiction d’un match à venir")

    # === INTERFACE UTILISATEUR ===
    opponent_future = st.selectbox(
        "Équipe adverse (CAN)",
        sorted(df_tunisie["opponent"].unique()),
        key="future_opponent"
    )

    location_future = st.radio(
        "Lieu du match",
        ["Domicile", "Extérieur", "Terrain neutre"],
        horizontal=True
    )

    match_type_future = st.radio(
        "Type de match",
        ["Officiel (CAN)", "Amical"],
        horizontal=True
    )

    # === FEATURES BASEES SUR LES DERNIERS MATCHS CONNUS ===
    last_matches = df_tunisie.sort_values("date").tail(5)

    form_future = last_matches["win"].mean()
    attack_5_future = last_matches["goals_scored"].mean()
    defense_5_future = last_matches["goals_against"].mean()

    is_home_future = 1 if location_future == "Domicile" else 0
    official_future = 1 if "Officiel" in match_type_future else 0

    opponent_strength_future = team_strength.get(
        opponent_future,
        team_strength.mean()
    )

    h2h_rate_future = h2h.get(
        opponent_future,
        df_tunisie["win"].mean()
    )

    travel_future = travel_level(opponent_future)

    # === DATAFRAME POUR LE MODELE ===
    future_match = pd.DataFrame([{
        "is_home": is_home_future,
        "form": form_future,
        "official": official_future,
        "opponent_strength": opponent_strength_future,
        "travel": travel_future,
        "attack_5": attack_5_future,
        "defense_5": defense_5_future,
        "h2h_rate": h2h_rate_future
    }])

    # === PREDICTION ===
    if st.button("⚽ Lancer la prédiction CAN"):
        proba_future = model.predict_proba(future_match)[0]
        pred_future = np.argmax(proba_future)

        label_map_inv = {0: "Défaite", 1: "Nul", 2: "Victoire"}

        st.success(
            f"🔮 **Tunisie vs {opponent_future}** → "
            f"**{label_map_inv[pred_future]}**"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Défaite", f"{proba_future[0]*100:.1f} %")
        col2.metric("Nul", f"{proba_future[1]*100:.1f} %")
        col3.metric("Victoire", f"{proba_future[2]*100:.1f} %")

        # === GRAPHIQUE DES PROBABILITES ===
        prob_future_df = pd.DataFrame({
            "Résultat": ["Défaite", "Nul", "Victoire"],
            "Probabilité": proba_future
        })

        fig_future = px.bar(
            prob_future_df,
            x="Résultat",
            y="Probabilité",
            template="plotly_dark",
            title="Probabilités estimées – Match CAN"
        )
        fig_future.update_layout(
            yaxis_tickformat=".0%",
            paper_bgcolor="#050814",
            plot_bgcolor="#050814",
            font_color="#F5F5F5",
        )
        st.plotly_chart(fig_future, use_container_width=True)

# =========================
# PAGE 4 : À PROPOS
# =========================
else:
    st.title("ℹ️ À propos du projet")

    st.markdown(
        """
        **Projet :** Analyse & prédiction des performances de l’équipe nationale de Tunisie 🇹🇳  
        **Tech :** Python, Pandas, XGBoost, Plotly, Streamlit  
        
        **Pipeline :**
        1. Chargement des résultats de matchs internationaux (dataset Kaggle).
        2. Filtrage des matchs de la Tunisie (2020–2025).
        3. Feature engineering :
           - Forme récente (5 derniers matchs)
           - Domicile / extérieur
           - Match officiel vs amical
           - Force de l’adversaire (proxy data)
           - Rolling offensif / défensif (buts marqués / encaissés)
           - Historique face-à-face (head-to-head win rate)
        4. Entraînement d’un modèle XGBoost multi-classes (Win / Draw / Loss).
        5. Évaluation sur les derniers matchs + dashboard interactif dark mode “stade”.
        
        Ce dashboard peut servir :
        - à illustrer un post LinkedIn pendant la CAN,
        - à montrer tes compétences en **Data Science + ML + Storytelling**,
        - à être enrichi (classement FIFA, expected goals, composition, etc.).
        """
    )
