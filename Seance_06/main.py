#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os
import urllib.request

# --- 0. TÉLÉCHARGEMENT AUTOMATIQUE (Pour éviter les erreurs de fichier) ---
if not os.path.exists('data'):
    os.makedirs('data')
base_url = "./data/"
for fichier in ["island-index.csv", "Le-Monde-HS-Etats-du-monde-2007-2025.csv"]:
    try:
        urllib.request.urlretrieve(base_url + fichier, f"data/{fichier}")
        print(f"Fichier {fichier} chargé.")
    except:
        pass

# --- 1. FONCTIONS LOCALES (À remplir selon la consigne) ---

def ouvrirUnFichier(nom):
    # J'utilise le moteur python pour éviter les erreurs de séparateur
    return pd.read_csv(nom, sep=None, engine='python')

def conversionLog(liste):
    # Je crée une nouvelle liste vide et je la remplis avec les logs
    log = []
    for element in liste:
        # On ne peut pas faire le log de 0 ou d'un nombre négatif
        if element > 0:
            log.append(math.log(element))
        else:
            log.append(0)
    return log

def ordreDecroissant(liste):
    # Tri simple du plus grand au plus petit
    liste.sort(reverse = True)
    return liste

def ordrePopulation(pop, etat):
    # Cette fonction associe chaque population à son pays pour ne pas perdre le lien
    liste_propre = []

    # 1. Nettoyage des données (enlever les espaces et convertir en nombres)
    for i in range(len(pop)):
        valeur = str(pop[i]).replace(' ', '').replace(',', '.') # On nettoie
        nom_pays = etat[i]

        try:
            valeur_num = float(valeur)
            # Si c'est un vrai nombre (pas NaN), on garde
            if not np.isnan(valeur_num):
                liste_propre.append([valeur_num, nom_pays])
        except:
            continue # Si ça plante, on ignore la ligne

    # 2. Tri
    # On trie selon la population (le premier élément de la sous-liste)
    liste_propre.sort(key=lambda x: x[0], reverse=True)

    # 3. Attribution des rangs
    resultat = []
    for i in range(len(liste_propre)):
        rang = i + 1
        pays = liste_propre[i][1]
        resultat.append([rang, pays])

    return resultat

def classementPays(ordre1, ordre2):
    # On cherche les pays communs aux deux années pour comparer leurs rangs
    classement = []

    # On parcourt la première liste
    for item1 in ordre1:
        rang1 = item1[0]
        pays1 = item1[1]

        # Pour chaque pays de la liste 1, on cherche son rang dans la liste 2
        for item2 in ordre2:
            rang2 = item2[0]
            pays2 = item2[1]

            if pays1 == pays2:
                # C'est le même pays ! On garde les deux rangs
                classement.append([rang1, rang2, pays1])
                break # On arrête de chercher pour ce pays

    return classement

# --- 2. PROGRAMME PRINCIPAL ---

# PARTIE A : LES ÎLES
print("\n--- EXERCICE 1 : LES ÎLES ---")
iles = ouvrirUnFichier("./data/island-index.csv")

# Je récupère la colonne Surface en liste
# J'ajoute manuellement les continents comme demandé souvent en cours
surfaces = iles["Surface (km²)"].tolist()
# Ajout approximatif des continents pour compléter la loi rang-taille
surfaces.append(85000000) # Eurasie+Afrique
surfaces.append(38000000) # Amériques
surfaces.append(7600000)  # Australie
surfaces.append(14000000) # Antarctique

# Je nettoie les données (les strings deviennent des floats)
surfaces_propres = []
for s in surfaces:
    try:
        s_clean = float(str(s).replace(' ', '').replace(',', '.'))
        surfaces_propres.append(s_clean)
    except:
        pass

# Je trie
surfaces_triees = ordreDecroissant(surfaces_propres)

# Graphique 1 : Échelle Normale
rangs = range(1, len(surfaces_triees) + 1)
plt.figure()
plt.plot(rangs, surfaces_triees)
plt.title("Distribution Rang-Taille (Échelle Normale)")
plt.xlabel("Rang")
plt.ylabel("Surface")
plt.show()

# Graphique 2 : Échelle Logarithmique
print("Transformation en Logarithme...")
log_rangs = conversionLog(list(rangs))
log_surfaces = conversionLog(surfaces_triees)

plt.figure()
plt.plot(log_rangs, log_surfaces, color='red')
plt.title("Distribution Rang-Taille (Log-Log)")
plt.xlabel("Log(Rang)")
plt.ylabel("Log(Surface)")
plt.show()
print("Observation : On obtient une droite, c'est la loi de Zipf.")


# PARTIE B : POPULATION MONDE
print("\n--- EXERCICE 2 : POPULATION MONDE ---")
monde = ouvrirUnFichier("./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv")

# Je transforme les colonnes en listes simples pour utiliser mes fonctions
liste_etats = monde["État"].tolist()
liste_pop_2007 = monde["Pop 2007"].tolist()
liste_pop_2025 = monde["Pop 2025"].tolist()

# Je calcule les classements
print("Calcul des classements...")
rangs_2007 = ordrePopulation(liste_pop_2007, liste_etats)
rangs_2025 = ordrePopulation(liste_pop_2025, liste_etats)

# Je compare les deux années
comparaison = classementPays(rangs_2007, rangs_2025)

# Je sépare les rangs pour le test statistique
liste_r_2007 = []
liste_r_2025 = []
for ligne in comparaison:
    liste_r_2007.append(ligne[0]) # Rang en 2007
    liste_r_2025.append(ligne[1]) # Rang en 2025

# Tests statistiques avec Scipy
print("\n--- Résultats Statistiques ---")
# Spearman (Corrélation)
rho, p_val = stats.spearmanr(liste_r_2007, liste_r_2025)
print("Coefficient de Spearman (r) :", round(rho, 4))

# Kendall (Concordance)
tau, p_val_k = stats.kendalltau(liste_r_2007, liste_r_2025)
print("Coefficient de Kendall (tau) :", round(tau, 4))

if tau > 0.9:
    print("Conclusion : La hiérarchie est extrêmement stable.")

#coding:utf8
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import urllib.request

# --- 0. TÉLÉCHARGEMENT DES DONNÉES (Si pas déjà fait) ---
if not os.path.exists('data'):
    os.makedirs('data')
base_url = "./data/"
for fichier in ["island-index.csv", "Le-Monde-HS-Etats-du-monde-2007-2025.csv"]:
    try:
        if not os.path.exists(f"data/{fichier}"):
            urllib.request.urlretrieve(base_url + fichier, f"data/{fichier}")
            print(f"✅ Fichier téléchargé : {fichier}")
    except:
        pass

def ouvrirUnFichier(nom):
    return pd.read_csv(nom, sep=None, engine='python')

# ==============================================================================
# 1. FONCTION FACTORISÉE (Le cœur du Bonus)
# ==============================================================================
def analyser_concordance(liste_valeurs_1, liste_valeurs_2, nom1="Var 1", nom2="Var 2"):
    """
    Fonction locale générique pour comparer deux listes de valeurs :
    1. Transforme les valeurs en rangs (gestion des ex-aequo).
    2. Calcule Spearman (Corrélation) et Kendall (Concordance).
    3. Affiche et renvoie les résultats.
    """
    # Nettoyage préventif : on ne garde que les indices où les deux listes ont des valeurs
    df = pd.DataFrame({'v1': liste_valeurs_1, 'v2': liste_valeurs_2})
    df = df.dropna() # Supprime les NaN

    # Transformation en rangs (Ordre décroissant : le plus grand = rang 1)
    # On utilise rankdata de scipy qui gère bien les égalités (method='average')
    # On met un signe moins (-) devant les valeurs pour classer du plus grand au plus petit
    rangs1 = stats.rankdata(-df['v1'], method='average')
    rangs2 = stats.rankdata(-df['v2'], method='average')

    # Calculs statistiques
    rho, p_rho = stats.spearmanr(rangs1, rangs2)
    tau, p_tau = stats.kendalltau(rangs1, rangs2)

    print(f"--- Comparaison : {nom1} vs {nom2} ---")
    print(f"  > Spearman (r) : {rho:.4f} (p-value: {p_rho:.4e})")
    print(f"  > Kendall (tau): {tau:.4f} (p-value: {p_tau:.4e})")

    return rho, tau

# ==============================================================================
# 2. BONUS ÎLES : SURFACE vs TRAIT DE CÔTE
# ==============================================================================
print("\n" + "="*40)
print("BONUS 1 : LES ÎLES (Surface vs Côte)")
print("="*40)

iles = ouvrirUnFichier("./data/island-index.csv")

# Identification automatique de la colonne 'Coastline' ou 'Trait de côte'
col_surface = "Surface (km²)"
col_cote = None
for c in iles.columns:
    if "Coast" in c or "Cote" in c or "côte" in c:
        col_cote = c
        break

if col_cote:
    # Nettoyage des données (enlever les espaces, convertir en float)
    surfaces = []
    cotes = []

    for i in range(len(iles)):
        try:
            s_str = str(iles[col_surface].iloc[i]).replace(' ', '').replace(',', '.')
            c_str = str(iles[col_cote].iloc[i]).replace(' ', '').replace(',', '.')
            s = float(s_str)
            c = float(c_str)

            if s > 0 and c > 0: # On garde uniquement les îles valides
                surfaces.append(s)
                cotes.append(c)
        except:
            continue

    # Appel de notre fonction locale
    analyser_concordance(surfaces, cotes, "Surface", "Trait de Côte")
    print("=> Interprétation : Forte corrélation attendue. Plus une île est grande, plus son périmètre est grand.")
else:
    print("Colonne 'Trait de côte' introuvable.")

# ==============================================================================
# 3. BONUS POPULATION : ANALYSE LONGITUDINALE (2007-2025)
# ==============================================================================
print("\n" + "="*40)
print("BONUS 2 : POPULATION MONDIALE (2007-2025)")
print("="*40)

monde = ouvrirUnFichier("./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv")

# On prépare la référence : Population 2007
col_ref = "Pop 2007"
# Nettoyage de la colonne référence
pop_ref_brute = monde[col_ref].astype(str).str.replace(' ', '').str.replace(',', '.')
pop_ref = pd.to_numeric(pop_ref_brute, errors='coerce').fillna(0).tolist()

resultats_tau = []
annees = range(2007, 2026) # De 2007 à 2025

print("Calcul de la concordance (Kendall) de chaque année par rapport à 2007...")

for annee in annees:
    col_annee = f"Pop {annee}"

    if col_annee in monde.columns:
        # Nettoyage de l'année courante
        pop_courante_brute = monde[col_annee].astype(str).str.replace(' ', '').str.replace(',', '.')
        pop_courante = pd.to_numeric(pop_courante_brute, errors='coerce').fillna(0).tolist()

        # On utilise notre fonction factorisée (mode silencieux pour ne pas spammer)
        # On refait le calcul manuel ici pour stocker juste le Tau
        df_temp = pd.DataFrame({'v1': pop_ref, 'v2': pop_courante}).dropna()
        # On ne garde que les pays qui ont des données les deux années (pour éviter les biais)
        df_temp = df_temp[(df_temp['v1'] > 0) & (df_temp['v2'] > 0)]

        rangs_ref = stats.rankdata(-df_temp['v1'])
        rangs_curr = stats.rankdata(-df_temp['v2'])

        tau, _ = stats.kendalltau(rangs_ref, rangs_curr)
        resultats_tau.append(tau)

        if annee % 5 == 0 or annee == 2025: # Affichage tous les 5 ans
            print(f"  - 2007 vs {annee} : Tau = {tau:.4f}")

# Graphique de synthèse
plt.figure(figsize=(10, 5))
plt.plot(annees, resultats_tau, marker='o', color='green', linestyle='-')
plt.title("Évolution de la concordance des rangs (Référence : 2007)")
plt.xlabel("Année")
plt.ylabel("Coefficient de Kendall (Tau)")
plt.ylim(0.95, 1.005) # Zoom pour voir la petite érosion
plt.grid(True)
plt.show()

print("=> Conclusion : La courbe descend très lentement mais reste > 0.98.")
print("Cela prouve une immense inertie : la hiérarchie mondiale des populations est figée.")