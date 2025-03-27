import numpy as np
import sympy as sp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir las variables simbólicas 
x, y, z = sp.symbols('x y z')


# Se define la función  f(x, y, z, w) = x^2 + y^2 + z^2 simbólicamente
f_sym = x**2 + y**2 + z**2 


# Calcular el gradiente de la función
grad_f = f_sym.diff(x), f_sym.diff(y), f_sym.diff(z)
grad_f = sp.Matrix(grad_f)

# Calcular la matriz hessiana de la función
hessian_f = sp.hessian(f_sym, (x, y, z))


# Convertir el gradiente simbólico en una función de Python con lambdify
grad_f_func = sp.lambdify((x, y, z), grad_f, 'numpy')
hessian_f_func = sp.lambdify((x, y, z), hessian_f, 'numpy')
f_func = sp.lambdify((x, y, z), f_sym, 'numpy')

# Función  para aplicar el metodo Newton-Raphson 
def newton_method(grad_func, hess_func, x0):
    eps=1e-5
    max_iter=100
    xk = x0.copy()
    history = [xk.copy()]
    
    for i in range(max_iter):
        grad = np.array(grad_func(*xk)).astype(float).flatten()
        hess = np.array(hess_func(*xk)).astype(float)
        
        #Calcular la inversa
        norm_grad = np.linalg.norm(grad)

        # Criterio de parada basado en la norma del gradiente
        if norm_grad < eps:
            break
        
        # Cálculo del vector para actualizar la solución actual
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("La Hessiana no invertible.")
            break
        
        # Actualizar la solución
        xk = xk - delta
        history.append(xk.copy())
    
    return xk, history

# Punto inicial para el método de Newton-Raphson
x0 = np.array([1, 2, 3])

# Invocar la función de Newton-Raphson
sol, hist = newton_method(grad_f_func, hessian_f_func, x0)

print("Mínimo encontrado en:", sol)

hist_array = np.array(hist)

x_vals = hist_array[:, 0]
y_vals = hist_array[:, 1]
z_vals = hist_array[:, 2]

# Crear gráfica 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_vals, y_vals, z_vals, 'bo--', label='Trayectoria')
ax.scatter(x_vals[0], y_vals[0], z_vals[0], c='green', s=100, label='Inicio')
ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], c='red', s=100, label='Mínimo')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Convergencia del método de Newton en 3D")
ax.legend()
plt.show()


# Convergencia bidimensional

hist_array = np.array(hist)


x_vals = hist_array[:, 0]
y_vals = hist_array[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='--', color='blue', label='Trayectoria')
plt.scatter(x_vals[0], y_vals[0], color='green', label='Inicio', s=100)
plt.scatter(x_vals[-1], y_vals[-1], color='red', label='Mínimo', s=100)

plt.title("Proyección de convergencia en el plano (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

