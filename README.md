# 🇹🇳 Tunisie x Data Science – Dashboard CAN

## Analyse & prédiction des performances de l’équipe nationale de Tunisie (2020–2025)

### Présentation du projet

Ce projet propose :

L’analyse statistique des matchs récents de l’équipe de Tunisie

La visualisation des performances selon différents facteurs : année, domicile/extérieur, adversaire…

Un modèle de Machine Learning (XGBoost) pour prédire les résultats futurs

Un dashboard interactif construit avec Streamlit + Plotly

🔧 Technos : Python · Pandas · Scikit-learn · XGBoost · Plotly · Streamlit
📅 Période étudiée : 2020 → 2025
🎯 Objectif spécial : Simuler des scénarios CAN 🇹🇳⚽

### Installation & exécution
- 1️⃣ Cloner le projet
git clone <repo_url>
cd tunisie_dashboard

- 2️⃣ Installer les dépendances
pip install -r requirements.txt

- 3️⃣ Lancer l’application
streamlit run app.py

### 📂 Dataset
Source	Description
Kaggle	Résultats internationaux de football
Filtrage	Seuls les matchs impliquant la Tunisie sont conservés
Pré-traitement	Gestion des équipes non vues → valeurs moyennes

### Certaines nations rares/absentes → prédictions moins précises

### 🧠 Pipeline Data & ML

Import & nettoyage des données

Filtrage des matchs de la Tunisie

Feature Engineering :

Forme récente (5 derniers matchs)

Home vs Away

Match officiel vs amical

Force de l’adversaire (buts moyens)

Rolling offensif / défensif

Historique face-à-face

Distance / localisation

Construction du dataset ML

### Modélisation : XGBoost multi-classes

### Évaluation :

Accuracy ✔️

Macro-F1 🔥 (équilibrée sur classes)

Intégration dans le Dashboard interactif

📊 Fonctionnalités du Dashboard
🏁 Overview

Nombre de matchs & type de résultats

Pie chart : Victoires / Nuls / Défaites

📈 Performance sportive

Résultats par année

Évolution de la différence de buts

Statistiques domicile vs extérieur

🤖 Modélisation & Prédictions

Scores du modèle + matrice de confusion

Explication des features importantes

Simulation d’un match du dataset test

Simulation de match futur (CAN Ready)

ℹ️ À propos

Documentation du pipeline analytique

🏆 Scénario CAN – Simulation

🛠 Sélection interactive :

Adversaire (liste déroulante)

Lieu (domicile / extérieur)

Affichage des probabilités :

🟢 Win / ⚪ Draw / 🔴 Loss

Explication des facteurs les plus influents

🔍 Limite : équipes non présentes → features moyennes

🔧 Améliorations futures

Intégration du classement FIFA 📉

xG / Stats d’occasions / tirs cadrés ⚽

Données sur les compositions (XI, joueurs)

Deep Learning (LSTM / Transformers) 🧬

Analyse individuelle par joueur