from torch import tensor
import numpy as np
import torch
from numpy.linalg import inv
from scipy.integrate import trapezoid as integrate
from numpy import diff
from torch import nn
import matplotlib.pyplot as plt

from Node_Shape_Function_Net import Node_Shape_Function_Net

class HiDeNN_for_FEM():
    def __init__(self, dom, num_nodes, equation, boundary_conditions=[0,0], optimizer=torch.optim.SGD):
        '''
        dom: 
        '''
        self.num_nodes = num_nodes
        self.dom = dom
        self.dom_end = dom[-1]
        self.dom_start = dom[0]
        self.__nodes = tensor(np.linspace(self.dom_start, self.dom_end, num_nodes), requires_grad=True)
        self.u_1 = boundary_conditions[0]
        self.u_n = boundary_conditions[1]
        self.dx = (self.dom_end-self.dom_start)/len(dom)
        self.num_elem = self.num_nodes - 1
        
        # -ku'' + cu' + bu = f
        self.k = equation[0]
        self.c = equation[1]
        self.b = equation[2]
        self.f_func = lambda x_v: list(map(equation[3], x_v))
        
        self.__node_nets_arr = self.node_nets_func( u_arr=[1 for i in range(num_nodes)] )
        self.shape_func_arr = [func_net.forward(dom) for func_net in self.__node_nets_arr]
        self.shape_func_arr = torch.stack(self.shape_func_arr).detach().numpy().reshape(num_nodes, -1)
        
        #### Calculating initial F and K for initial u's #### 
        
        self.__F = [integrate(shape*self.f_func(dom), dx=self.dx) for shape in self.shape_func_arr[1:-1]]
        self.__K = np.zeros((num_nodes, num_nodes))
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                df_i = diff(func_i)/self.dx
                df_j = diff(func_j)/self.dx
                integrand = self.k * ( df_i * df_j ) + self.c * ( df_i * func_j[1:] ) + self.b * ( func_i[1:] * func_j[1:] )
                self.__K[i, j] = integrate(integrand, dx=self.dx).reshape(-1)

        # Using the approach of strong condition (big number) - Computationally gives the same result as the theoretically correct way
        # https://scicomp.stackexchange.com/questions/20515/how-to-efficiently-implement-dirichlet-boundary-conditions-in-global-sparse-fini
        BigNumber = 10e64
        self.__F = np.append( BigNumber * self.u_1 , self.__F )
        self.__F = np.append( self.__F, BigNumber * self.u_n )
        self.__K[0,0] = BigNumber
        self.__K[-1,-1] = BigNumber
        
        ### Calculating initial displacements and aprox u ### 
        self.displacement_arr = np.array(inv(self.__K) @ self.__F) # alpha = K^-1 x F
        self.u_aprox_arr = self.displacement_arr @ self.shape_func_arr
            
        for i,node in enumerate(self.__node_nets_arr):
            node.u = self.displacement_arr[i]
        
        self.optimizer = optimizer([self.__nodes], lr=1e-3)
        
    def forward(self):
        out_hidden_layers = [node_net.forward(self.dom) for node_net in self.__node_nets_arr]
        
        output_layer = nn.Linear(self.num_nodes, 1, bias=False)
        output_layer.weight = nn.Parameter(torch.ones_like(output_layer.weight, dtype=torch.float64))
        output_layer.requires_grad_(False)
        
        self.u_aprox_arr = output_layer(torch.cat(out_hidden_layers, 1))
        self.u_aprox_arr = self.u_aprox_arr.numpy().reshape(-1)
        return self.u_aprox_arr
    
    def node_nets_func(self,u_arr):
            arr = [
                Node_Shape_Function_Net(
                    self.__nodes[i-1], 
                    self.__nodes[i], 
                    self.__nodes[i+1], 
                    u_arr[i]
                ) for i in range(1,self.num_nodes-1)
            ]
            arr.insert(0,
                Node_Shape_Function_Net(
                    None,
                    self.__nodes[0],
                    self.__nodes[1],
                    u_arr[0]
                )
            )
            arr.append(
                Node_Shape_Function_Net(
                    self.__nodes[-2],
                    self.__nodes[-1],
                    None,
                    u_arr[-1]
                )
            )
            return np.array(arr)
    
    @property 
    def node_nets_arr(self):
        self.__node_nets_arr = self.node_nets_func( self.displacement_arr )
        self.shape_func_arr = [func_net.forward(self.dom) for func_net in self.__node_nets_arr]
        self.shape_func_arr = torch.stack(self.shape_func_arr).detach().numpy().reshape(self.num_nodes, -1)
        return self.__node_nets_arr
    
    @property
    def K(self):
        self.__K = np.zeros((self.num_nodes, self.num_nodes))
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                df_i = diff(func_i)/self.dx
                df_j = diff(func_j)/self.dx
                integrand = self.k * ( df_i * df_j ) + self.c * ( df_i * func_j[1:] ) + self.b * ( func_i[1:] * func_j[1:] )
                self.__K[i, j] = integrate(integrand, dx=self.dx).reshape(-1)

        BigNumber = 10e64
        self.__K[0,0] = BigNumber
        self.__K[-1,-1] = BigNumber
        return self.__K
    
    @property
    def F(self):
        self.__F = [integrate( shape * self.f_func(self.dom), dx=self.dx ) for shape in self.shape_func_arr[1:-1]]

        BigNumber = 10e64
        self.__F = np.append( BigNumber * self.u_1 , self.__F )
        self.__F = np.append( self.__F, BigNumber * self.u_n )
        return self.__F

    @property
    def nodes(self):
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes):
        self.__nodes = nodes
        
        self.__node_nets_arr = self.node_nets_func( u_arr=[1 for i in range(self.num_nodes)] )
        self.shape_func_arr = [func_net.forward(self.dom) for func_net in self.__node_nets_arr]
        self.shape_func_arr = torch.stack(self.shape_func_arr).detach().numpy().reshape(self.num_nodes, -1)
        
        self.__F = [integrate(shape*self.f_func(self.dom), dx=self.dx) for shape in self.shape_func_arr[1:-1]]
        self.__K = np.zeros((self.num_nodes, self.num_nodes))
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                df_i = diff(func_i)/self.dx
                df_j = diff(func_j)/self.dx
                integrand = self.k * ( df_i * df_j ) + self.c * ( df_i * func_j[1:] ) + self.b * ( func_i[1:] * func_j[1:] )
                self.__K[i, j] = integrate(integrand, dx=self.dx).reshape(-1)

        BigNumber = 10e64
        self.__F = np.append( BigNumber * self.u_1 , self.__F )
        self.__F = np.append( self.__F, BigNumber * self.u_n )
        self.__K[0,0] = BigNumber
        self.__K[-1,-1] = BigNumber
        
        ### Calculating initial displacements and aprox u ### 
        self.displacement_arr = np.array(inv(self.__K) @ self.__F) # alpha = K^-1 x F
        self.u_aprox_arr = self.displacement_arr @ self.shape_func_arr
            
        self.__node_nets_arr = self.node_nets_func( self.displacement_arr )
        
    def plot_shape_functions(self):
        figure = plt.figure()
        figure, axs = plt.subplots(self.num_nodes, sharex=True, sharey=True)
        nodes = self.__nodes.detach().numpy()
        figure = plt.xticks(nodes)
        figure = figure.suptitle('Shape Functions')
        for i,func in enumerate(self.shape_func_arr):
            axs[i].plot(self.dom, func)
            axs[i].grid()
        return figure
    
    def plot_u_exact_vs_u_aprox(self, u_exact_arr):
        figure = plt.figure()
        figure = plt.plot(self.dom, u_exact_arr, label='Solução exata')
        nodes = self.__nodes.detach().numpy()
        figure = plt.plot(nodes, self.displacement_arr, 'o--', label=f'Aproximação com {self.num_elem} elementos')
        figure = plt.xticks(nodes)
        figure = plt.grid()
        figure = plt.legend()
        return figure

    def train(self, epochs=1, lossfunc=nn.MSELoss, lr=1e-3):
        
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        lossfunc = lossfunc()
        
        for epoch in range(epochs):
            # -k * u''(x) + c * u'(x) + b * u(x) = f(x)
            dudx = diff(self.forward())/self.dx
            ddudx2 = diff(dudx)/self.dx
            u = self.forward()
            self.optimizer.zero_grad()
            loss = lossfunc( 
                            tensor(-self.k * ddudx2 + self.c * dudx[1:] + self.b * u[2:], requires_grad=True, dtype=torch.float64), 
                            tensor(self.f_func(self.dom)[2:], dtype=torch.float64) )
            loss.backward()
            self.optimizer.step()
        
        return loss
    
        