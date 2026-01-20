#coding:utf8
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import urllib.request

# --- 0. TÉLÉCHARGEMENT DES DONNÉES (Pour Colab) ---
if not os.path.exists('data'):
    os.makedirs('data')
url = "https://raw.githubusercontent.com/MaximeForriez/Sorbonne-M1-Analyse-de-donnees/main/Seance-08/Exercice/src/data/Socioprofessionnelle-vs-sexe.csv"
try:
    urllib.request.urlretrieve(url, "./data/Socioprofessionnelle-vs-sexe.csv")
except:
    print("Erreur de téléchargement. Vérifiez si le fichier est déjà uploadé.")

# --- 1. FONCTIONS LOCALES ---

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    """
    Formate les données en tableau indexé par les catégories.
    """
    indexValeurs = {}
    for element in range(0, len(nom)):
        indexValeurs.update({element: nom[element]})
    # On crée le DF et on renome les index (lignes) avec les catégories
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def calcul_chi2_manuel(tableau_obs):
    """
    BONUS : Algorithme manuel du Chi2
    1. Calcule le tableau théorique (indépendance)
    2. Calcule la distance (Observed - Expected)^2 / Expected
    """
    # Totaux
    total_general = tableau_obs.values.sum()
    sommes_lignes = tableau_obs.sum(axis=1).values
    sommes_colonnes = tableau_obs.sum(axis=0).values

    chi2_total = 0
    ddl = (len(sommes_lignes) - 1) * (len(sommes_colonnes) - 1)

    print("\n--- Détail du calcul manuel (Bonus) ---")

    # Double boucle pour parcourir chaque case (i, j)
    # i = ligne, j = colonne
    rows, cols = tableau_obs.shape
    for i in range(rows):
        for j in range(cols):
            obs = tableau_obs.iloc[i, j]

            # Calcul de l'effectif théorique (Attendu)
            # Formule : (Total Ligne * Total Colonne) / Total Général
            attendu = (sommes_lignes[i] * sommes_colonnes[j]) / total_general

            # Contribution au Chi2 : (O - E)² / E
            contribution = ((obs - attendu)**2) / attendu
            chi2_total += contribution

    return chi2_total, ddl

# --- 2. MAIN PROGRAM ---

# Chargement
raw_data = ouvrirUnFichier("./data/Socioprofessionnelle-vs-sexe.csv")

# Nettoyage et Formatage
# On s'assure que les colonnes numériques sont bien interprétées
df_clean = pd.DataFrame(raw_data)

# Création du tableau de contingence propre (Index = Catégorie, Colonnes = Femmes/Hommes)
# Note : on utilise la fonction locale demandée
tableau_croise = tableauDeContingence(
    df_clean["Catégorie"],
    {"Femmes": df_clean["Femmes"], "Hommes": df_clean["Hommes"]}
)

print("--- Tableau de Contingence Observé ---")
print(tableau_croise)

# --- Calcul des marges ---
print("\n--- Calcul des marges ---")
totaux_colonnes = tableau_croise.sum(axis=0)
totaux_lignes = tableau_croise.sum(axis=1)
total_N = totaux_colonnes.sum()

print(f"Total Général (N) : {total_N}")
print("Totaux Lignes :")
print(totaux_lignes.to_dict())

# --- Test du Chi2 (SCIPY - Méthode Automatique) ---
print("\n" + "="*30)
print("TEST DU CHI2 (Scipy - Automatique)")
print("="*30)
khi2, p_value, ddl, expected = stats.chi2_contingency(tableau_croise)

print(f"Statistique Chi2 : {khi2:.4f}")
print(f"Degrés de liberté : {ddl}")
print(f"P-value          : {p_value:.4e}")

if p_value < 0.05:
    print("=> Rejet de H0 : Il y a une dépendance significative entre Sexe et CSP.")
else:
    print("=> H0 acceptée : Indépendance.")

# --- Intensité de liaison (Phi2) ---
print("\n--- Intensité de liaison ---")
# Phi2 = Chi2 / N
phi2 = khi2 / total_N
print(f"Phi² de Pearson : {phi2:.4f}")
# V de Cramer (pour info, souvent plus lisible car entre 0 et 1)
v_cramer = np.sqrt(phi2 / (min(tableau_croise.shape) - 1))
print(f"V de Cramer     : {v_cramer:.4f}")


# ==============================================================================
# BONUS : ALGORITHME MANUEL DU CHI2
# ==============================================================================
print("\n" + "="*50)
print("BONUS : ALGORITHME MANUEL DU CHI2")
print("="*50)

# Appel de la fonction locale créée plus haut
chi2_manu, ddl_manu = calcul_chi2_manuel(tableau_croise)

print(f"\nRésultat Calcul Manuel : {chi2_manu:.4f}")
print(f"Résultat Scipy         : {khi2:.4f}")

diff = abs(chi2_manu - khi2)
if diff < 0.01:
    print("✅ SUCCÈS : L'algorithme manuel trouve le même résultat que Scipy !")
else:
    print("❌ ÉCART : Il y a une différence, vérifiez la formule.")

# ==============================================================================
# BONUS : ANOVA & AFC
# ==============================================================================
print("\n" + "="*40)
print("BONUS")
print("="*40)

# --- BONUS 1 : ANOVA (Analyse de la Variance) ---
# On veut voir si la moyenne des "Pour" change significativement selon les échantillons ?
# (Dans notre cas simulé, ils viennent de la même loi, donc on ne devrait pas voir de différence,
# mais on teste le principe).
print("\n--- 1. ANOVA (Sur les échantillons) ---")
try:
    df_sondage = ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv")

    # On compare les 3 groupes : Pour, Contre, Sans opinion
    # Est-ce que les moyennes de ces 3 groupes sont différentes ?
    groupe_pour = df_sondage["Pour"]
    groupe_contre = df_sondage["Contre"]
    groupe_sans = df_sondage["Sans opinion"]

    stat_f, p_anova = stats.f_oneway(groupe_pour, groupe_contre, groupe_sans)

    print(f"Statistique F : {stat_f:.2f}")
    print(f"P-value : {p_anova:.2e}")
    if p_anova < 0.05:
        print("=> Différence significative entre les groupes (Logique, ce ne sont pas les mêmes proportions).")
    else:
        print("=> Pas de différence significative.")
except:
    print("Erreur données ANOVA")


# --- BONUS 2 : AFC (Analyse Factorielle des Correspondances) ---
# C'est complexe à coder de zéro. Voici une version simplifiée utilisant la SVD (Algèbre linéaire).
print("\n--- 2. AFC (Simplifiée) ---")

# 1. Tableau des fréquences (f_ij)
N = mon_tableau.sum().sum()
P = mon_tableau / N

# 2. Marges
r = P.sum(axis=1) # Marges lignes
c = P.sum(axis=0) # Marges colonnes

# 3. Matrice des résidus standardisés (S)
# Formule : (P_ij - r_i*c_j) / sqrt(r_i * c_j)
# On utilise numpy pour le calcul matriciel
R_mat = np.diag(1/np.sqrt(r))
C_mat = np.diag(1/np.sqrt(c))
# Matrice à décomposer : S = R^(-1/2) * (P - r*c.T) * C^(-1/2)
# Version simplifiée : Contribution au Chi2 signée
P_numpy = P.values
r_numpy = r.values.reshape(-1, 1)
c_numpy = c.values.reshape(1, -1)
Attendu = r_numpy @ c_numpy # Produit matriciel
Ecart = (P_numpy - Attendu) / np.sqrt(Attendu)

# 4. Décomposition en Valeurs Singulières (SVD)
U, s, Vt = np.linalg.svd(Ecart, full_matrices=False)

print("Valeurs propres (Inertie) :", np.round(s**2, 4))
print("Pourcentage d'inertie expliqué par l'axe 1 :", round(s[0]**2 / sum(s**2) * 100, 1), "%")

# Interprétation simple
print("\nInterprétation rapide de l'Axe 1 :")
# On regarde les coordonnées des colonnes (Sexe) sur l'axe 1
coords_col = Vt[0, :]
print(f"Coordonnées Axe 1 : Femmes={coords_col[0]:.2f}, Hommes={coords_col[1]:.2f}")
print("=> L'axe 1 oppose très nettement les Hommes et les Femmes.")
print("=> C'est le facteur principal qui structure les métiers.")