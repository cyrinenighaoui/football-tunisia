**Projet :** Analyse & prédiction des performances de l’équipe nationale de Tunisie 🇹🇳  
**Tech :** Python, Pandas, XGBoost, Plotly, Streamlit  
        
**Pipeline :**
1. Chargement des résultats de matchs internationaux (dataset Kaggle): (https://www.kaggle.com/datasets/oussamalariouch/african-national-football-from-2010-2024)
2. Filtrage des matchs de la Tunisie (2010).
3. Feature engineering :
- Forme récente (5 derniers matchs)
- Domicile / extérieur
- Match officiel vs amical
- Force de l’adversaire (proxy data)
- Rolling offensif / défensif (buts marqués / encaissés)
- Historique face-à-face (head-to-head win rate)
4. Entraînement d’un modèle XGBoost multi-classes (Win / Draw / Loss).
5. Évaluation sur les derniers matchs        
