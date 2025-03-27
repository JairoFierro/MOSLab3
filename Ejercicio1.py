# Ejercicio 1: Newton-Raphson en 2D para Polinomios Cúbicos

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Definir la función simbólica
x = sp.Symbol('x')
f = 3*x**3 - 10*x**2 - 56*x + 50

### 1.1: Implementar el algoritmo de Newton-Raphson para funciones unidimensionales.
def newton_raphson(f, x0, alpha=1, tol=1e-6, max_iter=100):
    df = sp.diff(f, x)
    ddf = sp.diff(df, x)
    f_lambda = sp.lambdify(x, df, 'numpy')
    ddf_lambda = sp.lambdify(x, ddf, 'numpy')
    
    x_n = x0
    for _ in range(max_iter):
        df_xn = f_lambda(x_n)
        ddf_xn = ddf_lambda(x_n)
        
        if abs(df_xn) < tol:
            break
        
        x_n = x_n - alpha * (df_xn / ddf_xn)
    
    return x_n

### 1.2: Calcular analíticamente la primera y segunda derivada de f(x).
df = sp.diff(f, x)
ddf = sp.diff(df, x)
print("Primera derivada:", df)
print("Segunda derivada:", ddf)

### 1.3: Experimentar con diferentes valores iniciales x0 en el intervalo [−6, 6].
initial_values = np.linspace(-6, 6, 5)
critical_points = [newton_raphson(f, x0) for x0 in initial_values]

### 1.4: Utilizar diferentes valores para el factor de convergencia α.
alpha_values = [0.2, 0.5, 0.8, 1.0]
results_alpha = {alpha: [newton_raphson(f, x0, alpha=alpha) for x0 in initial_values] for alpha in alpha_values}

### 1.5: Graficar la función junto con los puntos encontrados, destacando mínimos y máximos.
x_vals = np.linspace(-6, 6, 400)
f_lambda = sp.lambdify(x, f, 'numpy')
y_vals = f_lambda(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="f(x)")
plt.scatter(critical_points, f_lambda(np.array(critical_points)), color='red', label="Extremos locales")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Newton-Raphson para encontrar extremos locales")
plt.legend()
plt.grid()

plt.draw()  # Fuerza el renderizado de la gráfica
plt.pause(3)  # Espera un segundo para que la gráfica se muestre
plt.show()  # No bloquea la ejecución

### 1.6: Analizar el comportamiento de la convergencia para diferentes valores iniciales.
for alpha, points in results_alpha.items():
    print(f"Para alpha={alpha}, puntos críticos encontrados: {points}")

