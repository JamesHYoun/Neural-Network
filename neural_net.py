import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import time
from sympy.parsing.sympy_parser import rationalize
import data_gen as dg


class NeuralNet:

    def __init__(self, data, nn_dim, learn_rate, reg_coeff, eps):
        self.nn_dim = nn_dim
        self.num_data = len(data[0])
        self.num_feats = len(data[0][0])
        self.data_x = np.array(data[0])
        self.data_y = np.array(data[1])
        self.bias, self.weights = self.init_bw()
        self.learn_rate = learn_rate
        self.reg_coeff = reg_coeff
        self.eps = eps
        self.output = self.init_output()
        print(self.output)

    def init_bw(self):
        bias = np.array([[0.0 for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim))], dtype=object)
        weights = np.array([[[0.5 for _ in range(self.nn_dim[i+1])] for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim) - 1)], dtype=object)
        return bias, weights

    def init_output(self):
        self.output = np.array([[-1.0 for _ in range(len(self.data_y[0]))] for _ in range(self.num_data)])
        for i in range(self.num_data):
            self.frwd_prop(self.data_x[i], i)
        return self.output

    def activation(self, input):
        return 1 / (1 + np.exp(-input))

    def activ_prime(self, input):
        return -np.exp(-input) / (1 + np.exp(-input))**2

    def frwd_prop(self, input=[], data_i=-1, af_in=[], af_out=[]):
        flag = len(input) == 0
        a = input
        if flag: 
            a = af_out[0] = self.data_x[data_i]
        for i in range(1, len(self.nn_dim)):
            z = np.matmul(a, self.weights[i - 1]) + self.bias[i]
            a = self.activation(z)
            if flag:
                af_in[i] = z
                af_out[i] = a
        if flag or data_i != -1:
            # if flag:
            #     print(f'YAYYYYY')
            self.output[data_i] = a
        return max(a)   
    
    def back_prop(self, data_i, bias, weights, af_in, af_out):
        grad_b = np.multiply(af_out[-1] - self.data_y[data_i], self.activ_prime(af_in[-1]))
        bias[-1] += (1 / self.num_data) * grad_b
        for i in range(len(self.nn_dim) - 2, -1, -1):
            grad_w = np.matmul(np.array([af_out[i]]).T, np.array([grad_b]))
            weights[i] += (1 / self.num_data) * grad_w
            grad_b = np.multiply(np.matmul(self.weights[i], np.array([grad_b]).T).T[0], self.activ_prime(np.array(af_in[i])))
            bias[i] += (1 / self.num_data) * grad_b

    def gradient(self):
        print(f'GRADIENT')
        bias = np.array([[0.0 for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim))], dtype=object)
        weights = np.array([[[0.0 for _ in range(self.nn_dim[i+1])] for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim) - 1)], dtype=object)
        for i in range(self.num_data):
            af_in = np.array([[-1.0 for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim))], dtype=object)
            af_out = np.array([[-1.0 for _ in range(self.nn_dim[i])] for i in range(len(self.nn_dim))], dtype=object)
            self.frwd_prop([], i, af_in, af_out)
            self.back_prop(i, bias, weights, af_in, af_out)
        return bias, weights

    def cost(self):
        return (1 / (2 * self.num_data)) * sum([sum((self.output[i] - self.data_y[i])**2) for i in range(self.num_data)])

    def grad_desc(self):
        curr_cost = self.cost()
        print(curr_cost)
        prev_cost = 2 * curr_cost
        while abs(curr_cost - prev_cost) > self.eps * prev_cost:
            grad = self.gradient()
            self.bias -= self.learn_rate * grad[0]
            self.weights -= self.learn_rate * grad[1]
            print(f'BIASSSS')
            print(self.bias)
            print(f'WEIGHTSSSS')
            print(self.weights)
            prev_cost = curr_cost
            curr_cost = self.cost()
            print(curr_cost)

    def normalize(self):
        self.data_x_org = self.data_x
        data_x = [[-1.0 for _ in range(self.num_feats)] for _ in range(self.num_data)]
        params = [(-1.0, -1.0) for _ in range(self.num_feats)]
        for i in range(self.num_feats):
            col = self.data_x[:][i]
            mean, std = np.mean(col), np.std(col)
            for j in range(self.num_data):
                data_x[j][i] = (self.data_x[j][i] - mean) / std
            params[i] = (mean, std)
        self.data_x = data_x
        self.params = params

    # Needs Work
    def revert(self):
        diff = 0
        for i in range(self.num_feats):
            mean, std = self.params[i][0], self.params[i][1]
            self.weights[i] /= std
            diff += self.weights[i] * mean
        self.bias -= diff
        self.data_x = self.data_x_org

    def main(self):
        self.grad_desc()
        return self.bias, self.weights


# GENERATE DATA

data = dg.get_data(10000, 2, -10, 10)
data_x, data_y = data[0], data[1]


# TRAIN NEURAL NETWORK

nn_dim = [len(data_x[0]), 4, 4, len(data_y[0])]
eps = 10**(-6)
reg_coeff = 1e-2
learn_rate = 2e-6
nn = NeuralNet(data, nn_dim, learn_rate, reg_coeff, eps)
start_time = time.time()
bias, weights = nn.main()
end_time = time.time()
# print(f'Time taken: {end_time - start_time}')
# print(f'bias: {bias}\nweights: {weights}')


# TESTING

# for i in range(len(data_y)):
#     print(f'In: {data_x[i]}  Out: {data_y[i]}')

success = 0
for i in range(len(data_y)):
    if abs(data_y[i] - nn.frwd_prop(data_x[i])) < 0.05 * abs(data_y[i]):
        success += 1
print(f'SUCCESS RATE: {success / len(data_y)}')
