import numpy as np
import sympy as sp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# Definir las variables simbólicas 
x, y = sp.symbols('x y')


# Se define la función simbólicamente
f_sym = (x - 1)**2 + 100 * (y - x**2)**2


# Calcular el gradiente de la función de Rosenbrock en términos de x y y
# Definir el gradiente simbólicamente
grad_f_sym = [sp.diff(f_sym, var) for var in (x, y)]

# Convertir el gradiente simbólico en una función de Python con lambdify
grad_f_func = [sp.lambdify((x, y), g) for g in grad_f_sym]



# Calcular la matriz hessiana de la función de Rosenbrock en términos de x y y
hessian_f_sym = sp.hessian(f_sym, (x, y))

# Convertir la matriz hessiana simbólica en una función de Python con lambdify
hessian_f_func = sp.lambdify((x, y), hessian_f_sym)


# Función Newton-Raphson 
def newton_raphson_2d(f_grad, f_hessian, x0, tol=1e-6, max_iter=100):
    x_n = np.array(x0, dtype=float)
    history = [x_n.copy()]
    
    for i in range(max_iter):
        grad_eval = np.array([f_grad[0](*x_n), f_grad[1](*x_n)])
        hess_eval = np.array(f_hessian(*x_n), dtype=float)

        # Criterio de parada
        if np.linalg.norm(grad_eval) < tol:
            break

        # Actualización de Newton-Raphson
        delta = np.linalg.solve(hess_eval, grad_eval)
        x_n = x_n - delta
        history.append(x_n.copy())
    
    return x_n, np.array(history)

# Punto inicial para el método de Newton-Raphson
x0 = [0, 10]

# Llamar a la función de Newton-Raphson
sol, iteraciones = newton_raphson_2d(grad_f_func, hessian_f_func, x0)

print("Solución:", sol)



# -------------------------------------------------------
# Función numérica para evaluar f(x, y) a partir de f_sym
f_func = sp.lambdify((x, y), f_sym, modules='numpy')

# Crear una malla de puntos para X y Y
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_func(X, Y)

# Extraer puntos iterativos
iteraciones = np.array(iteraciones)
x_iter, y_iter = iteraciones[:, 0], iteraciones[:, 1]
z_iter = f_func(x_iter, y_iter)

# -------------------------------------------------------
# Crear figura y eje 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superficie
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

# Trayectoria de iteraciones
ax.plot(x_iter, y_iter, z_iter, color='black', marker='o', label='Iteraciones')

# Punto final en rojo
ax.scatter(x_iter[-1], y_iter[-1], z_iter[-1], color='red', s=60, label=f'Mínimo encontrado ({sol[0]:.0f}, {sol[1]:.0f})')

# Ejes y título
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Método Newton-Raphson aplicado en la función Rosenbrock')
ax.legend()
plt.show()



# -------------------------------------------------------

normas_grad = [
    np.linalg.norm([
        grad_f_func[0](x, y),
        grad_f_func[1](x, y)
    ]) for x, y in iteraciones
]

# Graficar la norma del gradiente en escala logarítmica
plt.figure(figsize=(8, 5))
plt.semilogy(normas_grad, marker='o', color='blue')
plt.xlabel('Iteración')
plt.ylabel('||∇f(x, y)||')
plt.title('Convergencia del gradiente en Newton-Raphson')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()