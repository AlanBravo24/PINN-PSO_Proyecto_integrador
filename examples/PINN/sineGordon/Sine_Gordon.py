import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from pyDOE import lhs

from swarm.optimizers.pso_adam import pso
from swarm.utils import multilayer_perceptron, decode
import matplotlib.animation as animation

# --- Configuración de Semillas ---
np.random.seed(1234)
tf.random.set_seed(1234)

# --- Definición del Problema y Solución Analítica ---

# Dominio espacial y temporal
x_min, x_max = -10.0, 10.0
t_min, t_max = 0.0, 0.5

# Velocidad del Kink (solitón)
c = 0.5
c2_inv_sqrt = 1.0 / np.sqrt(1.0 - c**2)

def analytical_solution(x, t):
    """ Solución analítica (Kink) para Sine-Gordon """
    return 4.0 * np.arctan(np.exp((x - c * t) * c2_inv_sqrt))

# --- Muestreo de Puntos de Entrenamiento ---

N_ic = 200  # Puntos de condición inicial (IC)
N_b = 200   # Puntos de condición de borde (BC)
N_f = 1000  # Puntos de colocación (residuales)

# Límites del dominio
lb = np.array([x_min, t_min])
ub = np.array([x_max, t_max])

# Puntos de Condición Inicial (IC) (t=0)
x_ic = np.linspace(x_min, x_max, N_ic).reshape(-1, 1)
t_ic = np.zeros_like(x_ic)
X_ic = np.hstack((x_ic, t_ic))
u_ic = analytical_solution(x_ic, t_ic)

# Puntos de Condición de Borde (BC) (x=-10 y x=10)
t_bc = np.linspace(t_min, t_max, N_b).reshape(-1, 1)
x_bc_left = np.full_like(t_bc, x_min)
x_bc_right = np.full_like(t_bc, x_max)

X_bc_left = np.hstack((x_bc_left, t_bc))
X_bc_right = np.hstack((x_bc_right, t_bc))

u_bc_left = analytical_solution(x_bc_left, t_bc)
u_bc_right = analytical_solution(x_bc_right, t_bc)

# Convertir a tensores
X_ic_tf = tf.convert_to_tensor(X_ic, dtype=tf.float32)
u_ic_tf = tf.convert_to_tensor(u_ic, dtype=tf.float32)
X_bc_left_tf = tf.convert_to_tensor(X_bc_left, dtype=tf.float32)
u_bc_left_tf = tf.convert_to_tensor(u_bc_left, dtype=tf.float32)
X_bc_right_tf = tf.convert_to_tensor(X_bc_right, dtype=tf.float32)
u_bc_right_tf = tf.convert_to_tensor(u_bc_right, dtype=tf.float32)


# Puntos de Colocación (Residuales)
X_f_data = lb + (ub - lb) * lhs(2, N_f)
x_f_tf = tf.convert_to_tensor(X_f_data[:, 0:1], dtype=tf.float32)
t_f_tf = tf.convert_to_tensor(X_f_data[:, 1:2], dtype=tf.float32)


# --- Definición del Modelo de Física (PINN) ---

def f_model(w, b, x, t):
    """ Calcula el residual de la PDE Sine-Gordon: f = u_tt - u_xx + sin(u) """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        
        u = multilayer_perceptron(w, b, tf.concat([x, t], 1))
        
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    
    u_xx = tape.gradient(u_x, x)
    u_tt = tape.gradient(u_t, t)
    
    del tape
    
    f_u = u_tt - u_xx + tf.sin(u)
    
    return f_u

@tf.function
def loss(w, b):
    """ Función de pérdida total (Residual + IC + BC) """
    
    f_u_pred = f_model(w, b, x_f_tf, t_f_tf)
    mse_f = tf.reduce_mean(tf.square(f_u_pred))
    
    u_ic_pred = multilayer_perceptron(w, b, X_ic_tf)
    mse_ic = tf.reduce_mean(tf.square(u_ic_tf - u_ic_pred))
    
    u_bc_left_pred = multilayer_perceptron(w, b, X_bc_left_tf)
    u_bc_right_pred = multilayer_perceptron(w, b, X_bc_right_tf)
    mse_b = tf.reduce_mean(tf.square(u_bc_left_tf - u_bc_left_pred)) + \
            tf.reduce_mean(tf.square(u_bc_right_tf - u_bc_right_pred))

    # Forzar el cumplimiento de los bordes (que definen el movimiento)
    return mse_f + 50.0 * mse_ic + 200.0 * mse_b

def loss_grad():
    def _loss(w, b):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(w)
            tape.watch(b)
            loss_value = loss(w, b)
        trainable_variables = w + b
        grads = tape.gradient(loss_value, trainable_variables)
        return loss_value, grads
    return _loss

def run_swarm(swarm, X):
    swarm_y = []
    for particle in swarm:
        w, b = decode(particle, layer_sizes)
        swarm_y.append(multilayer_perceptron(w, b, X))
    return swarm_y

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

# --- Entrenamiento del Ensamble PSO-PINN ---

# --- CAMBIO: AUMENTAR CAPACIDAD DE LA RED ---
layer_sizes = [2] + 5 * [30] + [1] # 15 -> 30 neuronas
pop_size = 20  # Tamaño del ensamble
n_iter = 5000  # Iteraciones de entrenamiento

# --- Mantener Hiperparámetros Exploratorios ---
opt = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    # Hiperparámetros (beta, c1, c2)
    0.99,   # beta (inercia)
    0.08,   # c1 (cognitivo)
    0.5,    # c2 (social)
    # ---
    initialization_method="xavier",
    verbose=True,
    gd_alpha=1e-3,          
)
# --- FIN DE CAMBIOS ---


print("Iniciando entrenamiento...")
start = time.time()
opt.train()
end = time.time()
print("Tiempo transcurrido: %2d:%2d:%2d" % format_time(end - start))


# --- Procesamiento de Resultados y Visualización ---

# Rejilla para la visualización final
uxn = 256
utn = 100
x_plot = np.linspace(x_min, x_max, uxn)
t_plot = np.linspace(t_min, t_max, utn)

X, T = np.meshgrid(x_plot, t_plot)
X_flat = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star_flat = analytical_solution(X_flat[:, 0], X_flat[:, 1])

swarm = opt.get_swarm()
preds = run_swarm(swarm, X_flat.astype(np.float32))
mean = tf.squeeze(tf.reduce_mean(preds, axis=0))
var = tf.squeeze(tf.math.reduce_std(preds, axis=0))

print("Pérdida Final (Last Loss): ", opt.loss_history[-1])

error_u = np.linalg.norm(u_star_flat - mean, 2) / np.linalg.norm(u_star_flat, 2)
print("Error L2: %e" % (error_u))

# --- Animación ---

time_steps = utn - 1
fps = 15

def snapshot(i):
    """ Genera un frame de la animación en el tiempo t_plot[i] """
    l_ind = i * uxn
    u_ind = (i + 1) * uxn
    
    plt.clf()
    for k in range(len(preds)):
        plt.plot(x_plot, preds[k][l_ind:u_ind], 'c-', linewidth=0.3, alpha=0.5)
    
    plt.plot(x_plot, u_star_flat[l_ind:u_ind], "b-", linewidth=3, label="Exacta")
    
    plt.plot(
        x_plot,
        mean[l_ind:u_ind],
        "r--",
        linewidth=3,
        label="PSO-PINN (Media)",
    )
    
    plt.fill_between(
        x_plot,
        mean[l_ind:u_ind] - 3 * var[l_ind:u_ind],
        mean[l_ind:u_ind] + 3 * var[l_ind:u_ind],
        color="gray",
        alpha=0.3,
        label="Incertidumbre (3 std)"
    )

    plt.title(f"Sine-Gordon Kink (t = {t_plot[i]:.2f})")
    plt.xlabel("$x$")
    plt.ylabel("$u(t,x)$")
    plt.xlim(x_min, x_max)
    plt.ylim(
        np.min(u_star_flat) - 0.5,
        np.max(u_star_flat) + 0.5
    )
    plt.grid()
    plt.legend(loc="upper left")


fig = plt.figure(figsize=(10, 6), dpi=150)
anim = animation.FuncAnimation(fig, snapshot, frames=time_steps, interval=50)

print("Guardando animación...")
anim.save("sine_gordon_demo.gif", fps=fps)
print("Animación guardada como 'sine_gordon_demo.gif'")