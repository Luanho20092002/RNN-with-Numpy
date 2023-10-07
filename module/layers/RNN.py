import numpy as np

class RNN:

    def __init__(self, units, active='relu') -> None:
        self.active = active
        self.units = units
        self.has_params = True
        self.has_optimizer_params =True

    def forward(self, X):
        n, t, d = X.shape
        self.X = X
        if self.has_params:
            self.has_params = False
            self.params = {"Wx": 0.01 * np.random.randn(d, self.units),
                           "Wh": 0.01 * np.random.randn(self.units, self.units),
                           "b":  0.01 * np.random.randn(1, self.units)}
        if self.active == "relu":
            self.drelu = np.zeros((t, n, self.units))
        h = np.zeros((n, self.units))
        self.hs = {0: h}
        
        for i in range(t):
            x = self.X[:, i]
            h = (x @ self.params["Wx"] + self.params["b"]) + (h @ self.params["Wh"]) 
            if self.active == "relu":
                self.drelu[i] = (h > 0)
                h = h * self.drelu[i]
            elif self.active == "tanh":
                h = np.tanh(h)
            self.hs[i + 1] = h
        return h
    

    def backward(self, dh, optimizer):
        dWx = np.zeros_like(self.params["Wx"])
        dWh = np.zeros_like(self.params["Wh"])
        db = np.zeros_like(self.params["b"])
        
        n, t, d = self.X.shape
        for i in reversed(range(t)):
            if self.active == "relu":
                dh = dh * self.drelu[i]
            elif self.active == "tanh":
                dh = (1 - self.hs[i + 1]**2) * dh
            x = self.X[:, i]
            dWx += x.T @ dh
            dWh += self.hs[i].T @ dh
            db += np.sum(dh, axis=0, keepdims=True)
            dh = dh @ self.params["Wh"].T
        optimizer.update(self, self.params.keys(), dWx, dWh, db)
        return dh


