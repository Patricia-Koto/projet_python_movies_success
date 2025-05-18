# Prédiction du Succès d’un Film

## I. Objectif
Ce projet propose une application web interactive développée avec **Streamlit** permettant de prédire si un film aura du succès commercial en se basant sur ses caractéristiques.

## II. Préparation des données et choix du modèle
- Les tests statistiques de corrélatio ont été faits avant de choisir les variables d'intérêt.
- La base des données a été néttoyée

## III. Fonctionnalités
-  Identifiants par défaut (lui seul peut ajouter pu supprimer les utilisateurs):
      utilisateur : admin
      mot de passe : admin123
- Prédiction du succès d’un film (Succès ou Échec)
- Affichage de la probabilité associée
- Tableau d’historique des prédictions exportable en PDF
- Dashboard avec :
  - Taux de succès par mois
  - Budget et revenu moyen par genre
  - Top 10 des films les plus rentables
- Interface d’administration :
  - Ajouter / Supprimer des utilisateurs
  - Historique global visible uniquement par l’admin

## IV. Technologies

- **Python**
- **Streamlit**
- **scikit-learn**
- **pandas / matplotlib / seaborn / plotly**
- **fpdf2** pour l’export PDF
- ** etc
## V. Fichiers inclus
- app.py – Script principal Streamlit
- modele_succes_film.pkl – Modèle pré-entraîné (Random Forest)
- tmdb_5000_movies.csv – Jeu de données films
- users.json – Fichier de comptes utilisateurs
- README.md – Ce fichier
- requirements.txt – Liste des packages Python
- stat_descriptive.ipynb – les analyses descriptives
- test_comparaison_modeles.ipynb – analyse faite pour choisir le bon modèle
- test_robustesse_model_film.ipynb – test pour voir si le modèle est robuste ou pas
- video_presentation.mp4 – vidéo de démonstration

## Autrice
Patricia KOTO NGBANGA
