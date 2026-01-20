#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Je règle la taille des images pour qu'elles soient lisibles
plt.rcParams['figure.figsize'] = (10, 6)

print("=== PARTIE 1 : VARIABLES DISCRÈTES (Bâtons) ===")

# 1. Loi de Dirac
# C'est une certitude : probabilité de 1 à une valeur précise (ici 5)
print("\n--- 1. Loi de Dirac ---")
x_dirac = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_dirac = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] # Le pic est à 5

plt.bar(x_dirac, y_dirac, color='blue')
plt.title("Loi de Dirac (a=5)")
plt.ylabel("Probabilité")
plt.show()

# 2. Loi Uniforme Discrète (Le dé à jouer)
print("\n--- 2. Loi Uniforme Discrète ---")
# On prend un dé de 1 à 6
borne_min = 1
borne_max = 7 # Le 7 est exclu
x_uni = np.arange(borne_min, borne_max)
y_uni = stats.randint.pmf(x_uni, borne_min, borne_max)

print("Moyenne :", stats.randint.mean(borne_min, borne_max))
plt.bar(x_uni, y_uni, color='green')
plt.title("Loi Uniforme Discrète (1 à 6)")
plt.show()

# 3. Loi Binomiale (Pile ou Face)
print("\n--- 3. Loi Binomiale ---")
n = 10  # 10 lancers
p = 0.5 # probabilité de 0.5 (pièce équilibrée)
x_binom = np.arange(0, n+1)
y_binom = stats.binom.pmf(x_binom, n, p)

print("Moyenne :", stats.binom.mean(n, p))
plt.bar(x_binom, y_binom, color='orange')
plt.title("Loi Binomiale (n=10, p=0.5)")
plt.show()

# 4. Loi de Poisson (Événements rares)
print("\n--- 4. Loi de Poisson ---")
mu = 3 # Moyenne d'événements attendus
x_poisson = np.arange(0, 15)
y_poisson = stats.poisson.pmf(x_poisson, mu)

print("Moyenne :", stats.poisson.mean(mu))
plt.bar(x_poisson, y_poisson, color='red')
plt.title("Loi de Poisson (lambda=3)")
plt.show()

# 5. Loi de Zipf-Mandelbrot
# (Utilisée pour la taille des villes ou fréquence des mots)
print("\n--- 5. Loi de Zipf ---")
a = 1.5 # Paramètre de forme (plus il est grand, plus la décroissance est vite)
x_zipf = np.arange(1, 20) # On regarde les 20 premiers rangs
y_zipf = stats.zipf.pmf(x_zipf, a)

print("Moyenne :", stats.zipf.mean(a))
plt.bar(x_zipf, y_zipf, color='purple')
plt.title("Loi de Zipf (a=1.5)")
plt.xlabel("Rang")
plt.show()


print("\n=== PARTIE 2 : VARIABLES CONTINUES (Courbes) ===")

# 6. Loi de Poisson (Version Continue = Exponentielle)
# Note : Dans le PDF, 'Loi de Poisson' est listée en continu.
# Mathématiquement, la loi continue associée est la Loi Exponentielle.
print("\n--- 6. Loi Continue associée à Poisson (Exponentielle) ---")
x_expon = np.linspace(0, 10, 100)
y_expon = stats.expon.pdf(x_expon)

plt.plot(x_expon, y_expon, color='red')
plt.fill_between(x_expon, y_expon, alpha=0.3, color='red')
plt.title("Loi Exponentielle (Lien avec Poisson)")
plt.show()

# 7. Loi Normale (Gauss)
print("\n--- 7. Loi Normale ---")
x_norm = np.linspace(-4, 4, 100)
y_norm = stats.norm.pdf(x_norm, loc=0, scale=1) # Moyenne 0, Ecart-type 1

print("Moyenne :", stats.norm.mean(0, 1))
plt.plot(x_norm, y_norm, color='black')
plt.title("Loi Normale (Gauss)")
plt.show()

# 8. Loi Log-Normale
print("\n--- 8. Loi Log-Normale ---")
s = 0.9 # Paramètre de forme
x_log = np.linspace(0, 5, 100)
y_log = stats.lognorm.pdf(x_log, s)

plt.plot(x_log, y_log, color='brown')
plt.title("Loi Log-Normale")
plt.show()

# 9. Loi Uniforme Continue
print("\n--- 9. Loi Uniforme Continue ---")
# Entre 0 et 10
x_uni_c = np.linspace(-2, 12, 100)
y_uni_c = stats.uniform.pdf(x_uni_c, loc=0, scale=10)

plt.plot(x_uni_c, y_uni_c, color='green')
plt.title("Loi Uniforme Continue (0 à 10)")
plt.show()

# 10. Loi du Chi-2 (Khi-deux)
print("\n--- 10. Loi du Chi-2 ---")
df = 4 # Degrés de liberté
x_chi2 = np.linspace(0, 15, 100)
y_chi2 = stats.chi2.pdf(x_chi2, df)

print("Moyenne :", stats.chi2.mean(df))
plt.plot(x_chi2, y_chi2, color='magenta')
plt.title("Loi du Chi-2 (k=4)")
plt.show()

# 11. Loi de Pareto (Inégalités)
print("\n--- 11. Loi de Pareto ---")
b = 2.62 # Paramètre de forme
x_pareto = np.linspace(1, 5, 100)
y_pareto = stats.pareto.pdf(x_pareto, b)

print("Moyenne :", stats.pareto.mean(b))
plt.plot(x_pareto, y_pareto, color='cyan')
plt.title("Loi de Pareto (b=2.62)")
plt.show()