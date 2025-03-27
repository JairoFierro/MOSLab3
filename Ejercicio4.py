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