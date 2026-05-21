import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 /(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)
def ReLU_prime(x):
    return np.where(x > 0, 1, 0)


n = 1000

data = np.random.rand(n, 2)
labels = np.zeros(len(data))


for i in range(len(data)):
    data[i][0] = data[i][0] - 0.5
    data[i][1] = data[i][1] - 0.5
    if data[i][0]**2 + data[i][1]**2 < 0.09:  # Circle at origin with radius 0.3
        labels[i] = 1

        
        
for i in range(len(data)): # Plots 1s as red and 0s as blue
    x = data[i][0]
    y = data[i][1]
    label = labels[i]
    color = "red" if label == 1 else "blue"
    plt.scatter(x, y, c = color)

plt.show()

    
class NeuralNet():
    def __init__(self):
        self.w1 = np.random.randn(2, 10)
        self.b1 = np.random.randn(1, 10)
        self.w2 = np.random.randn(10, 10)
        self.b2 = np.random.randn(1, 10)
        self.w3 = np.random.randn(10, 1)
        self.b3 = np.random.randn(1, 1)
    

        
    def forward(self, x):
        self.z1 = np.array(x) 
        self.a2 = np.matmul(self.z1, self.w1)
        self.a2 = np.add(self.a2, self.b1) 
        self.z2 = self.a2 
        self.a2 = ReLU(self.a2) 
        self.a3 = np.matmul(self.a2, self.w2)
        self.a3 = np.add(self.a3, self.b2)  
        self.z3 = self.a3 
        self.a3 = ReLU(self.a3) 
        self.z4 = np.matmul(self.a3, self.w3)
        self.z4 = np.add(self.z4, self.b3) 
        self.y_hat = sigmoid(self.z4)
        
        return self.y_hat
    
    def backprop(self, x, y):
        y = y.reshape(-1, 1)
        delta_z4 = self.y_hat - y
        delta_z3 = (delta_z4 @ self.w3.T) * ReLU_prime(self.z3)
        delta_z2 =(delta_z3 @ self.w2.T) * ReLU_prime(self.z2)
        
        self.w3_grad =  self.z3.T @ delta_z4 / len(y)
        self.b3_grad = np.mean(delta_z4, axis = 0, keepdims=True)
        self.w2_grad = self.z2.T @ delta_z3 / len(y)
        self.b2_grad = np.mean(delta_z3, axis = 0, keepdims=True)
        self.w1_grad = self.z1.T @ delta_z2 / len(y)
        self.b1_grad = np.mean(delta_z2, axis = 0, keepdims=True)
        
    
    def descent(self, alpha = 0.01):
        self.w1 = self.w1 - alpha * self.w1_grad
        self.w2 = self.w2 - alpha * self.w2_grad
        self.w3 = self.w3 - alpha * self.w3_grad
        self.b1 = self.b1 - alpha * self.b1_grad
        self.b2 = self.b2 - alpha * self.b2_grad
        self.b3 = self.b3 - alpha * self.b3_grad
    
    def loss(self, y):  # Binary Cross Entropy is used as the loss function
        self.y_hat = np.clip(self.y_hat, 1e-5, 1) 
        return - ( y * np.log(self.y_hat) + (1-y) * np.log(1 - self.y_hat))
    
    def accuracy(self, X, Y):
        preds = (model.forward(X) > 0.5)
        accuracy = np.mean(preds == Y) * 100
        return accuracy
    
model = NeuralNet()

X = data
Y = labels.reshape(-1, 1)

epochs = 1000
a = 0.05
for j in range(epochs): 

    model.forward(X)
    model.backprop(X, Y)
    model.descent(a)
    
    if j%100 == 0:
        print(np.mean(model.loss(Y), axis = 0))
        
print(model.accuracy(X, Y))

for i in range(len(X)):  # Correctly identified points are green, incorrect points are red

    x = X[i][0]
    y = X[i][1]
    pred = (model.forward(X[i]) > 0.5)
    correct = (pred == Y[i])
    colour = "green" if correct else "red"
    
    plt.scatter(x, y, c = colour)
plt.show()