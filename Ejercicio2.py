# Ejercicio 2: Análisis de Extremos Locales y Globales

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# 2.1: Calcular analíticamente la primera y segunda derivada de f(x).

x = sp.Symbol('x')
f = x**5 - 8*x**3 + 10*x + 6


primera_derivada = sp.diff(f, x)
segunda_derivada = sp.diff(primera_derivada, x)


f_lambdified = sp.lambdify(x, f, 'numpy')
df_lambdified = sp.lambdify(x, primera_derivada, 'numpy')
ddf_lambdified = sp.lambdify(x, segunda_derivada, 'numpy')

# 2.2 Aplicar el método de Newton-Raphson desde diferentes puntos iniciales para encontrar todos los posibles extremos.
def newton_raphson(func, dfunc, ddfunc, x0, tol=1e-6, max_iter=100):
    x_n = x0
    
    for _ in range(max_iter):
        f_x = dfunc(x_n)
        df_x = ddfunc(x_n)
        
        if abs(df_x) < 1e-8:  # Esta parte ayuda a evitar divisiones por cero
            break
        x_n = x_n - f_x / df_x
        
        if abs(f_x) < tol:
            return x_n
    return None  # No converge

# Explorar múltiples valores iniciales en el intervalo [-3, 3]
intervalo = np.linspace(-3, 3, 20)
raices = set()

for x0 in intervalo:
    raiz = newton_raphson(f_lambdified, df_lambdified, ddf_lambdified, x0)
    
    if raiz is not None:
        raices.add(round(raiz, 4))

# 2.3: Clasificar los puntos encontrados como mínimos o máximos locales.
extremos = sorted(raices)

maximos, minimos = [], []
for r in extremos:
    
    segunda_derivada = ddf_lambdified(r)
    if segunda_derivada > 0:
        minimos.append(r)
        
    elif segunda_derivada < 0:
        maximos.append(r)

# 2.4: Identificar entre todos los extremos el máximo global y el mínimo global.
y_minimos = [f_lambdified(m) for m in minimos]
y_maximos = [f_lambdified(m) for m in maximos]

min_global = minimos[np.argmin(y_minimos)]
max_global = maximos[np.argmax(y_maximos)]

# 2.5: Graficar la función con todos los extremos locales (en negro) y destacar el máximo global y el mínimo global (en rojo).
x_vals = np.linspace(-3, 3, 400)
y_vals = f_lambdified(x_vals)

plt.plot(x_vals, y_vals, label='f(x)', color='blue')
plt.scatter(minimos, y_minimos, color='black', label='Mínimos locales')
plt.scatter(maximos, y_maximos, color='black', label='Máximos locales')
plt.scatter(min_global, f_lambdified(min_global), color='red', label='Mínimo global', s=100)
plt.scatter(max_global, f_lambdified(max_global), color='red', label='Máximo global', s=100)
plt.legend()

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Extremos de f(x)')
plt.grid()
plt.show()

# 2.6: Analizar la convergencia del método para esta función.
def evaluar_convergencia():
    iteraciones = []
    
    for x0 in intervalo:
        x_n = x0
        contador = 0
        
        while contador < 100:
            f_x = df_lambdified(x_n)
            df_x = ddf_lambdified(x_n)
            
            if abs(df_x) < 1e-8:
                break
            x_n = x_n - f_x / df_x
            contador += 1
            
            if abs(f_x) < 1e-6:
                break
        iteraciones.append(contador)
        
    plt.plot(intervalo, iteraciones, marker='o')
    plt.xlabel('Valor inicial')
    plt.ylabel('Iteraciones hasta convergencia')
    plt.title('Convergencia del método Newton-Raphson')
    plt.grid()
    plt.show()

evaluar_convergencia()
