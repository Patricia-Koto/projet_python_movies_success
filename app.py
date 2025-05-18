# === IMPORTS ===

#pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn joblib fpdf2 spicy
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from fpdf import FPDF
import base64
import json
import os
import hashlib
import plotly.express as px

# === AUTHENTIFICATION ===
USER_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_FILE):
        default_users = {"admin": hash_password("admin123")}
        with open(USER_FILE, "w") as f:
            json.dump(default_users, f)
        return default_users
    else:
        with open(USER_FILE, "r") as f:
            return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

USERS = load_users()

# === SESSION STATE INIT ===
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'history' not in st.session_state:
    st.session_state.history = []

# === EXPORT PDF HISTORIQUE ===
def export_history_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Historique des prédictions", ln=1, align='C')

    headers = ["Date", "Utilisateur", "Budget", "Durée", "Mois", "Genres", "Résultat"]
    for header in headers:
        pdf.cell(30, 10, header, border=1)
    pdf.ln()

    for entry in data:
        pdf.cell(30, 10, entry['date'], border=1)
        pdf.cell(30, 10, entry['utilisateur'], border=1)
        pdf.cell(30, 10, str(entry['budget']), border=1)
        pdf.cell(30, 10, str(entry['runtime']), border=1)
        pdf.cell(30, 10, str(entry['mois_sortie']), border=1)
        pdf.cell(30, 10, entry['genres'][:10], border=1)
        pdf.cell(30, 10, entry['résultat'], border=1)
        pdf.ln()

    filename = "historique_predictions.pdf"
    pdf.output(filename)

    with open(filename, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{filename}">📄 Télécharger l\'historique en PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

# === PAGE : Prédiction ===
def prediction_page():
    st.title("Prédiction du Succès d'un Film")
    st.write("Remplis les caractéristiques du film :")

    budget = st.number_input("Budget du film ($)", min_value=1000, step=1000)
    runtime = st.slider("Durée du film (minutes)", 30, 240)
    mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet',
                   'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    mois_choisi = st.selectbox("Mois de sortie", [""] + mois_labels)

    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
    genres_selection = st.multiselect("Genres", genres + ["Autre genre"])

    langues_map = {
        'Anglais': 'en', 'Français': 'fr', 'Espagnol': 'es',
        'Japonais': 'ja', 'Chinois': 'zh', 'Autre langue': 'autre'
    }
    langue_choisie = st.selectbox("Langue originale", [""] + list(langues_map.keys()))

    if st.button("Prédire"):
        if not (budget and runtime and mois_choisi and genres_selection and langue_choisie):
            st.warning("Veuillez remplir tous les champs.")
            return

        model = joblib.load("modele_succes_film.pkl")
        langue_code = langues_map[langue_choisie]
        mois_sortie = mois_labels.index(mois_choisi) + 1

        input_dict = {
            'budget': budget,
            'runtime': runtime
        }

        for g in genres:
            input_dict[g] = 1 if g in genres_selection else 0
        for l in ['en', 'fr', 'es', 'ja', 'zh']:
            input_dict[f'lang_{l}'] = 1 if l == langue_code else 0
        for m in range(1, 13):
            input_dict[f'mois_{m}'] = 1 if m == mois_sortie else 0

        input_df = pd.DataFrame([input_dict])
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        resultat = "Succès" if prediction == 1 else "Échec"
        st.subheader(f"Résultat : {resultat} ({proba*100:.2f}%)")

        st.session_state.history.append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "utilisateur": st.session_state.username,
            "budget": budget,
            "runtime": runtime,
            "mois_sortie": mois_sortie,
            "genres": ", ".join(genres_selection),
            "langue": langue_choisie,
            "résultat": resultat
        })

    if st.session_state.history:
        st.subheader("Historique des prédictions")
        st.dataframe(pd.DataFrame(st.session_state.history))
        export_history_to_pdf(st.session_state.history)

# === PAGE : Connexion ===
def login_page():
    st.title("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username in USERS and USERS[username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Identifiants incorrects.")

# === PAGE : Dashboard ===
def dashboard_page():
    st.title("Dashboard")
    try:
        df = pd.read_csv("tmdb_5000_movies.csv")
        df = df[(df['budget'] > 0) & (df['runtime'] > 0) & (df['revenue'] > 0)]
        df['rentabilite'] = df['revenue'] / df['budget']
        df['succes'] = (df['rentabilite'] > 1.5).astype(int)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['mois_sortie'] = df['release_date'].dt.month
        df['genre_principal'] = df['genres'].fillna('[]').apply(eval).apply(lambda x: x[0]['name'] if x else 'Unknown')
        df['langue'] = df['original_language'].fillna('unknown')

        st.subheader("Top 10 des meilleurs films (revenus)")
        top_df = df[['title', 'release_date', 'budget', 'revenue', 'homepage']].copy()
        top_df = top_df[top_df['revenue'] > 0]
        top_df = top_df.sort_values(by='revenue', ascending=False).head(10).reset_index(drop=True)
        top_df.index = top_df.index + 1
        top_df['release_date'] = pd.to_datetime(top_df['release_date'], errors='coerce').dt.date
        top_df['Lien'] = top_df['homepage'].fillna('').apply(lambda url: f'<a href="{url}" target="_blank">Voir</a>' if url else 'Non dispo')
        top_df = top_df.drop(columns='homepage')
        top_df.insert(0, 'Rang', top_df.index)
        top_df = top_df.rename(columns={
            'title': 'Titre du film',
            'release_date': 'Date de sortie',
            'budget': 'Budget ($)',
            'revenue': 'Revenu ($)'
        })
        styled_table = top_df.style.set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#cce5ff')]},
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},
        ]).to_html(escape=False, index=False)
        st.markdown(f"<div style='text-align:center'>{styled_table}</div>", unsafe_allow_html=True)
        #st.markdown(styled_table, unsafe_allow_html=True)

        st.subheader("Taux de succès selon le mois de sortie")
        mois_success = df.groupby('mois_sortie')['succes'].mean().reset_index()
        mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        mois_success['mois'] = mois_success['mois_sortie'].apply(lambda x: mois_labels[int(x)-1] if pd.notnull(x) and 1 <= int(x) <= 12 else 'Inconnu')
        import plotly.express as px
        fig1 = px.line(mois_success, x='mois', y='succes', markers=True,
                      labels={'mois': 'Mois', 'succes': 'Taux de succès'},
                      title="Taux de succès des films par mois",
                      height=400)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Succès des films par genre")
        genre_success = df[df['succes'] == 1]['genre_principal'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        genre_total = df['genre_principal'].value_counts()
        genre_success_percent = (genre_success / genre_total * 100).fillna(0)
        bars = ax2.bar(genre_success_percent.index, genre_success_percent.values, color='skyblue', edgecolor='black')
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Genres")
        ax2.set_ylabel("% de films à succès")
        ax2.set_title("Succès des films par genre (en %)")
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.subheader("Budget moyen par genre")
        budget_moyen = df.groupby('genre_principal')['budget'].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        palette = sns.color_palette("Blues", len(budget_moyen))
        sns.barplot(data=budget_moyen, x='genre_principal', y='budget', ax=ax3, palette=palette)
        ax3.set_title("Budget moyen par genre")
        ax3.set_ylabel("Budget moyen ($)")
        ax3.set_xlabel("Genre")
        ax3.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        st.subheader("Revenu moyen par genre")
        revenu_moyen = df.groupby('genre_principal')['revenue'].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        palette = sns.color_palette("Greens", len(revenu_moyen))
        sns.barplot(data=revenu_moyen, x='genre_principal', y='revenue', ax=ax4, palette=palette)
        ax4.set_title("Revenu moyen par genre")
        ax4.set_ylabel("Revenu moyen ($)")
        ax4.set_xlabel("Genre")
        ax4.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")



# === PAGE : À propos ===
def about_page():
    st.title("À propos")
    st.markdown("""
    Cette application prédit si un film aura du succès avant sa sortie et donne le pourcentage du succès ou de l'échec.
    - Elle utilise un modèle entraîné 
    - Elle affiche quelques statistiques clées via Dashboard
    - Seul l'admin peut ajouter un utilisateur en lui créant ses identifiants
                
    Cette application a été élaborée par : Patricia KOTO NGBANGA.
    """)

# === PAGE : Admin ===
def supprimer_utilisateur(nom):
    users = load_users()
    if nom in users:
        del users[nom]
        save_users(users)

def admin_page():
    st.title("Admin : Gestion des utilisateurs")

    st.subheader("Ajouter un utilisateur")
    new_user = st.text_input("Nom", key="new_user")
    new_pass = st.text_input("Mot de passe", type="password", key="new_pass")
    if st.button("Ajouter"):
        if not new_user or not new_pass:
            st.warning("Champs requis.")
        elif new_user in USERS:
            st.warning("Utilisateur déjà existant.")
        else:
            USERS[new_user] = hash_password(new_pass)
            save_users(USERS)
            st.success("Utilisateur ajouté.")

    st.subheader("Supprimer un utilisateur")
    choix = st.selectbox("Sélectionner", ["" ] + [u for u in USERS if u != "admin"])
    if st.button("Supprimer"):
        if choix:
            confirm = st.checkbox(f"Confirmer suppression de {choix}")
            if confirm:
                supprimer_utilisateur(choix)
                st.success(f"L'utilisateur '{choix}' a été supprimé.")
                st.rerun()

# === MAIN ===
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Pages", ["Prédiction", "Dashboard", "À propos"] + (["Admin"] if st.session_state.username == "admin" else []))
    st.sidebar.write(f"Connecté : {st.session_state.username}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    if page == "Prédiction":
        prediction_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "À propos":
        about_page()
    elif page == "Admin":
        admin_page()

if __name__ == "__main__":
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()
