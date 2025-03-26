import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def initialize_weights_glorot(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_out, n_in))

def initialize_weights_he(n_in, n_out):
    return np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)

class NeuralNetwork:
    def __init__(self, layer_sizes, activations=None, dropout_rates=None):
        """
        Initialize the neural network with support for different activation functions and dropout.
        
        Parameters:
        -----------
        layer_sizes : list
            List containing the size of each layer
        activations : list, optional
            List containing the activation function for each layer (default: tanh for all)
        dropout_rates : list, optional
            List containing dropout rate for each layer (default: no dropout)
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Initialize activation functions
        if activations is None:
            self.activations = ['tanh'] * (self.num_layers - 1)
        else:
            self.activations = activations
        
        # Map activation functions to their implementations
        self.activation_funcs = {
            'tanh': tanh,
            'relu': relu,
            'sigmoid': sigmoid
        }
        
        self.activation_primes = {
            'tanh': tanh_prime,
            'relu': relu_prime,
            'sigmoid': sigmoid_prime
        }
        
        # Initialize dropout rates (0 means no dropout)
        if dropout_rates is None:
            self.dropout_rates = [0] * (self.num_layers - 1)
        else:
            self.dropout_rates = dropout_rates
            
        # Initialize weights with appropriate initializers based on activation function
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            if self.activations[i] == 'relu':
                # He initialization for ReLU
                self.weights.append(initialize_weights_he(layer_sizes[i], layer_sizes[i+1]))
            else:
                # Glorot/Xavier initialization for tanh/sigmoid
                self.weights.append(initialize_weights_glorot(layer_sizes[i], layer_sizes[i+1]))
        
        # Initialize biases, velocities, and batch normalization parameters
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]
        
        # For L2 regularization
        self.lambd = 0.0  # regularization parameter, 0 means no regularization
        
        # Track if we're in training mode (for dropout)
        self.training = True
    
    def toggle_training(self, training=True):
        """Toggle between training and evaluation mode (affects dropout)"""
        self.training = training
    
    def apply_dropout(self, layer_output, dropout_rate):
        """Apply dropout to layer output during training"""
        if not self.training or dropout_rate == 0:
            return layer_output, None
        
        # Generate dropout mask
        keep_prob = 1 - dropout_rate
        mask = np.random.binomial(1, keep_prob, size=layer_output.shape) / keep_prob
        
        # Apply mask
        return layer_output * mask, mask
    
    def feedforward(self, a, return_intermediates=False):
        """
        Perform forward propagation through the network.
        
        Parameters:
        -----------
        a : ndarray
            Input to the network
        return_intermediates : bool, optional
            If True, return intermediate values for backpropagation
            
        Returns:
        --------
        If return_intermediates is False, returns only the final output.
        If return_intermediates is True, returns (activations, zs, dropout_masks)
        """
        activations = [a]
        zs = []
        dropout_masks = []
        
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            
            # Apply activation function
            activation_func = self.activation_funcs[self.activations[i]]
            a = activation_func(z)
            
            # Apply dropout (except at the output layer)
            if i < len(self.weights) - 1 and self.dropout_rates[i] > 0:
                a, mask = self.apply_dropout(a, self.dropout_rates[i])
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
                
            activations.append(a)
        
        if return_intermediates:
            return activations, zs, dropout_masks
        else:
            return activations[-1]
    
    def backpropagation(self, x, y):
        """
        Perform backpropagation to compute gradients.
        
        Parameters:
        -----------
        x : ndarray
            Input to the network
        y : ndarray
            Target output
            
        Returns:
        --------
        tuple (grad_b, grad_w) containing gradients
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass
        activations, zs, dropout_masks = self.feedforward(x, return_intermediates=True)
        
        # Backward pass
        # Compute the error at the output layer
        delta = self.cost_derivative(activations[-1], y)
        
        # Apply the derivative of the activation function
        output_activation = self.activations[-1]
        delta = delta * self.activation_primes[output_activation](zs[-1])
        
        # Set the gradients for the output layer
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)
        
        # Add L2 regularization to the weights gradient (not biases)
        if self.lambd > 0:
            grad_w[-1] += (self.lambd / len(x)) * self.weights[-1]
        
        # Backpropagate the error
        for l in range(2, len(self.biases) + 1):
            # Get the activation derivative for this layer
            activation_func = self.activations[-l]
            delta = np.dot(self.weights[-l+1].T, delta)
            
            # Apply dropout mask from the forward pass
            if dropout_masks[-l] is not None:
                delta = delta * dropout_masks[-l]
            
            # Apply activation derivative
            delta = delta * self.activation_primes[activation_func](zs[-l])
            
            # Set gradients
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].T)
            
            # Add L2 regularization
            if self.lambd > 0:
                grad_w[-l] += (self.lambd / len(x)) * self.weights[-l]
        
        return (grad_b, grad_w)
    
    def gradient_descent(self, mini_batch, eta, mu=0.9):
        """
        Update network parameters using gradient descent with momentum.
        
        Parameters:
        -----------
        mini_batch : list
            List of (x, y) tuples for training
        eta : float
            Learning rate
        mu : float, optional
            Momentum coefficient (default: 0.9 or 0.0)
        """
        # Inicializar acumuladores de gradientes
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        #  Acumular gradientes para cada ejemplo en el mini-batch
        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(x, y)
            grad_b = [gb + db for gb, db in zip(grad_b, delta_b)]
            grad_w = [gw + dw for gw, dw in zip(grad_w, delta_w)]

        # Actualizar pesos y biases como el valor del momentum
        self.velocity_w = [
            mu * vw - eta * gw
            for vw, gw in zip(self.velocity_w, grad_w)
        ]
        self.velocity_b = [
            mu * vb - eta * gb
            for vb, gb in zip(self.velocity_b, grad_b)
        ]

        # Hacer actualizaciones
        self.weights = [w + vw for w, vw in zip(self.weights, self.velocity_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, self.velocity_b)]
    
    def cost_derivative(self, output_activations, y):
        """Derivative of the cost function (MSE)"""
        return output_activations - y
    
    def update_learning_rate(self, epoch, initial_lr, schedule_type="step_decay", decay=0.1, drop_every=1000):
        """
        Update learning rate based on schedule.
        
        Parameters:
        -----------
        epoch : int
            Current epoch number
        initial_lr : float
            Initial learning rate
        schedule_type : str, optional
            Type of learning rate schedule
        decay : float, optional
            Decay factor
        drop_every : int, optional
            Number of epochs between drops for step decay
            
        Returns:
        --------
        float : Updated learning rate
        """
        if schedule_type == "step_decay":
            return initial_lr * (decay ** np.floor(epoch / drop_every))
        elif schedule_type == "exp_decay":
            return initial_lr * np.exp(-decay * epoch)
        elif schedule_type == "inv_scaling":
            return initial_lr / (1 + decay * epoch)
        elif schedule_type == "cosine":
            # Cosine annealing
            max_epochs = 1000  # Default max epochs
            return initial_lr * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
        else:
            return initial_lr
    
    def train(self, training_data, epochs, mini_batch_size, learning_rate, 
            decay=0.1, schedule_type="step_decay", validation_data=None,
            early_stopping_patience=None, verbose=True):
        """
        Train the neural network with progress tracking.
        
        Parameters:
        -----------
        training_data : list
            List of (x, y) tuples for training
        epochs : int
            Number of epochs to train
        mini_batch_size : int
            Size of mini-batches
        learning_rate : float
            Initial learning rate
        decay : float, optional
            Learning rate decay factor
        schedule_type : str, optional
            Type of learning rate schedule
        validation_data : list, optional
            List of (x, y) tuples for validation
        early_stopping_patience : int, optional
            Number of epochs to wait for improvement before stopping
        verbose : bool, optional
            Whether to print progress
            
        Returns:
        --------
        dict : Training history containing loss and validation metrics
        """
        try:
            from tqdm.auto import tqdm
            use_tqdm = True
        except ImportError:
            print("tqdm not installed. Install with 'pip install tqdm' for progress bars.")
            use_tqdm = False
        
        n = len(training_data)
        loss_history = []
        val_loss_history = [] if validation_data else None
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        # Enable dropout during training
        self.toggle_training(True)
        
        # Create epoch iterator with tqdm if available
        epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if use_tqdm else range(epochs)
        
        for j in epoch_iter:
            # Shuffle training data
            np.random.shuffle(training_data)
            
            # Create mini-batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Create mini-batch iterator with tqdm if available and verbose
            if use_tqdm and verbose:
                batch_iter = tqdm(mini_batches, desc=f"Epoch {j+1}/{epochs}", 
                                leave=False, unit="batch")
            else:
                batch_iter = mini_batches
            
            # Train on mini-batches
            for mini_batch in batch_iter:
                self.gradient_descent(mini_batch, learning_rate)
            
            # Update learning rate
            new_lr = self.update_learning_rate(j, learning_rate, schedule_type, decay)
            
            # Calculate training loss
            self.toggle_training(False)  # Disable dropout for evaluation
            current_loss = np.mean(
                [np.mean((self.feedforward(x) - y) ** 2) for x, y in training_data]
            )
            loss_history.append(current_loss)
            
            # Calculate validation loss if validation data provided
            current_val_loss = None
            if validation_data:
                current_val_loss = np.mean(
                    [np.mean((self.feedforward(x) - y) ** 2) for x, y in validation_data]
                )
                val_loss_history.append(current_val_loss)
                
                # Early stopping check
                if early_stopping_patience:
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        # Save best model weights
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        if use_tqdm:
                            epoch_iter.set_postfix({
                                "early_stopping": True,
                                "best_val_loss": f"{best_val_loss:.6f}"
                            })
                        elif verbose:
                            print(f"Early stopping at epoch {j+1}")
                        break
            
            # Re-enable dropout for next training epoch
            self.toggle_training(True)
            
            # Update progress bar with metrics
            metrics = {
                "loss": f"{current_loss:.6f}",
                "lr": f"{new_lr:.6f}"
            }
            if current_val_loss is not None:
                metrics["val_loss"] = f"{current_val_loss:.6f}"
            
            if use_tqdm:
                epoch_iter.set_postfix(metrics)
            elif verbose and j % 10 == 0:
                val_str = f", Val Loss: {current_val_loss:.6f}" if current_val_loss is not None else ""
                print(f"Epoch {j+1}/{epochs}, Loss: {current_loss:.6f}{val_str}, LR: {new_lr:.6f}")
            
            # Update learning rate for next epoch
            learning_rate = new_lr
        
        # Restore best model if early stopping was used and best weights were saved
        if early_stopping_patience and best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"Restored best model with validation loss: {best_val_loss:.6f}")
        
        # Return results
        self.toggle_training(False)  # Ensure dropout is disabled after training
        return {
            "loss": loss_history,
            "val_loss": val_loss_history
        }


# Crear datos de entrenamiento
# Se generan 100 valores entre 0 y 2𝜋, y el arreglo se convierte en una columna
x_vals = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# Calcular el seno de los valores generados
y_vals = np.sin(x_vals)

# Convierte los vectores de entrenamiento a forma horizontal, donde cada columna es un ejemplo
# (1, 100)
x_train = x_vals.T  
y_train = y_vals.T  

# Convertir a lista de tuplas [(x1, y1), (x2, y2), ...]
training_data = [(x_train[:, i:i+1], y_train[:, i:i+1]) for i in range(x_train.shape[1])]

# Creación de la red neuronal
net = NeuralNetwork(layer_sizes=[1, 10, 1], activations=['tanh', 'tanh'])


# Entrenar la red neuronal
history = net.train(
    training_data=training_data,
    epochs=1000,
    mini_batch_size=10,
    learning_rate=0.01,
    decay=0.0,
    schedule_type="none",
    verbose=True
)



# Obtener predicciones de la red neuronal
preds = [net.feedforward(x.reshape(-1, 1)) for x in x_vals]
preds = np.array(preds).reshape(-1)

plt.plot(x_vals, y_vals, label='sin(x)')
plt.plot(x_vals, preds, label='NN prediction', linestyle='--')
plt.legend()
plt.title("Aproximación con GD y momentum")
plt.show()



plt.plot(history['loss'])
plt.title("Loss durante el entrenamiento")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()