# Problema 4: Gradiente Descendente en Optimización

#PARTE 1
import numpy as np
import matplotlib.pyplot as plt

# 4.1: Calcular analíticamente el gradiente de la función de pérdida

def gradient(x, y):
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y + 1)
    return np.array([grad_x, grad_y])

# 4.2: Implementar el algoritmo de Gradiente Descendente
def gradient_descent(x0, y0, alpha, iterations):
    
    x, y = x0, y0
    trajectory = [(x, y)]
    
    for _ in range(iterations):
        
        grad = gradient(x, y)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        trajectory.append((x, y))
    
    return np.array(trajectory)

# 4.3: Experimentar con diferentes valores para el parámetro de paso α
alphas = [0.1, 0.5, 1.0]
initial_point = (-4, 4)
iterations = 20

trajectories = {alpha: gradient_descent(*initial_point, alpha, iterations) for alpha in alphas}

# 4.4: Graficar la trayectoria de los parámetros durante la optimización
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = (X - 2)**2 + (Y + 1)**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30, cmap='coolwarm')

for alpha, traj in trajectories.items():
    
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'α = {alpha}')
    
plt.scatter(2, -1, color='red', marker='x', s=100, label='Óptimo Global')
plt.legend()
plt.title("Trayectoria del Gradiente Descendente")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 4.5: Destacar el valor óptimo final y compararlo con la solución analítica
for alpha, traj in trajectories.items():
    
    final_x, final_y = traj[-1]
    print(f'Para α = {alpha}, valor final: ({final_x:.5f}, {final_y:.5f})')
    print(f'Error respecto al óptimo: {np.linalg.norm([final_x - 2, final_y + 1]):.5f}\n')

# 4.6: Analizar la sensibilidad del método al valor de α --> Analisis en el documento




#PARTE 2
import numpy as np
import matplotlib.pyplot as plt
import time

# 4.1. Calcular analíticamente el gradiente y la matriz Hessiana de la función propuesta.
# Definimos la función objetivo
def f(x, y):
    return (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2

# Calculamos el gradiente
def gradient(x, y):
    df_dx = 2 * (x - 2) * (y + 2)**2 + 2 * (x + 1)
    df_dy = 2 * (y - 1) + 2 * (x - 2)**2 * (y + 2)
    return np.array([df_dx, df_dy])

# Calculamos la matriz Hessiana
def hessian(x, y):
    d2f_dx2 = 2 * (y + 2)**2 + 2
    d2f_dy2 = 2 + 2 * (x - 2)**2
    d2f_dxdy = 4 * (x - 2) * (y + 2)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# 4.2. Implementar ambos algoritmos (Newton-Raphson y Gradiente Descendente) para la misma función.
# Método de Gradiente Descendente
def gradient_descent(x0, y0, alpha=0.01, tol=1e-6, max_iter=1000):
    x, y = x0, y0
    trajectory = [(x, y)]
    start_time = time.time()
    for i in range(max_iter):
        grad = gradient(x, y)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    exec_time = time.time() - start_time
    return np.array(trajectory), i+1, exec_time

# Método de Newton-Raphson
def newton_raphson(x0, y0, tol=1e-6, max_iter=1000):
    x, y = x0, y0
    trajectory = [(x, y)]
    start_time = time.time()
    for i in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)
        delta = np.linalg.solve(hess, -grad)
        x += delta[0]
        y += delta[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    exec_time = time.time() - start_time
    return np.array(trajectory), i+1, exec_time

# 4.3. Utilizar el mismo punto inicial (x0, y0) = (-2, -3) para ambos métodos.
x0, y0 = -2, -3

# 4.4. Experimentar con diferentes valores del parámetro de paso α y determinar el valor óptimo.
# Ejecutamos los métodos
traj_gd, iter_gd, time_gd = gradient_descent(x0, y0, alpha=0.005)
traj_nr, iter_nr, time_nr = newton_raphson(x0, y0)

# 4.5. Graficar en una misma figura las trayectorias de convergencia de ambos métodos.
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-4, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'r-o', markersize=3, label='Gradiente Descendente')
plt.plot(traj_nr[:, 0], traj_nr[:, 1], 'b-s', markersize=3, label='Newton-Raphson')
plt.scatter([x0], [y0], c='black', marker='x', s=100, label='Punto Inicial')
plt.legend()
plt.title('Comparación de Trayectorias de Convergencia')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 4.6. Realizar un análisis comparativo
print("\nAnálisis Comparativo:")
print(f"Número de iteraciones hasta la convergencia: GD = {iter_gd}, NR = {iter_nr}")
print(f"Tiempo de ejecución: GD = {time_gd:.6f}s, NR = {time_nr:.6f}s")
print(f"Precisión final del resultado (gradiente norm): GD = {np.linalg.norm(gradient(traj_gd[-1, 0], traj_gd[-1, 1])):.6e}, NR = {np.linalg.norm(gradient(traj_nr[-1, 0], traj_nr[-1, 1])):.6e}")

# 4.7. Concluir cuál método es más adecuado para esta función específica.
# En este caso, Newton-Raphson tiende a converger en menos iteraciones pero tiene un mayor costo computacional por iteración.
# Gradiente Descendente es más simple y robusto frente a cambios en el paso α, aunque puede requerir más iteraciones.

# 4.8. Presentar una tabla comparativa de ventajas y desventajas de cada método.
print("\nTabla Comparativa:")
print("Método | Iteraciones | Tiempo de Ejecución | Precisión Final")
print("-------------------------------------------------------------")
print(f"Gradiente Descendente | {iter_gd} | {time_gd:.6f}s | {np.linalg.norm(gradient(traj_gd[-1, 0], traj_gd[-1, 1])):.6e}")
print(f"Newton-Raphson | {iter_nr} | {time_nr:.6f}s | {np.linalg.norm(gradient(traj_nr[-1, 0], traj_nr[-1, 1])):.6e}")
