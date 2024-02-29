#!/usr/bin/env python
# coding: utf-8

#Modules
import numpy as np
import torch
from torch import nn 
from torch import tensor
from torch.autograd import grad
from datetime import datetime
from numpy.linalg import inv
from scipy.integrate import trapezoid as integrate
from numpy import diff
import matplotlib.pyplot as plt


#Classes
#Mesh nodes' neural network class
class Node_Shape_Function_Net(nn.Module):
    def __init__(self, x_previous, x_current, x_later, u_i): 
        super().__init__()
        self.x_current = tensor([[x_current]], dtype=torch.float64).requires_grad_(True)
        self.__u = tensor([[u_i]], dtype=torch.float64).requires_grad_(True)
        self.x_later = tensor([[x_later]], dtype=torch.float64) if x_later!=None else None
        self.x_previous = tensor([[x_previous]], dtype=torch.float64) if x_previous!=None else None
        
        # 1. Constructing the layers of the shape function (the ascent and descent parts) h
        if x_later!=None: # its not the last node
            self.x_later.requires_grad = True
            
            descending_layer1 = nn.Linear(1,1)
            descending_layer1.weight = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # w_12_12\n",
            descending_layer1.bias = nn.Parameter(-self.x_current) # b_2_2\n",
            descending_layer1.requires_grad_(False)
            
            descending_layer2 = nn.Linear(1,1).requires_grad_(False)
            descending_layer2.weight = nn.Parameter( -1 / (self.x_later - self.x_current) ) # w_23_22
            descending_layer2.bias = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # b_3_2
            
            descending_layer2 = nn.Linear(1,1)
            descending_layer2.weight = nn.Parameter( -1 / (self.x_later - self.x_current) ) # w_23_22\n",
            descending_layer2.bias = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # b_3_2\n",
            descending_layer2.requires_grad_(False)
            
            descending_layer3 = nn.Linear(1,1)
            descending_layer3.weight = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # w_34_22\n",
            if x_previous!=None:
                descending_layer3.bias = nn.Parameter(tensor([[-0.5]], dtype=torch.float64)) # b_4_2\n",
            elif x_previous==None: # its the first node\n",
                descending_layer3.bias = nn.Parameter(tensor([[0.]], dtype=torch.float64))
            descending_layer3.requires_grad_(False)
            
        if x_previous!=None: # its not the first node
            self.x_previous.requires_grad = True
            
            ascending_layer1 = nn.Linear(1, 1)
            ascending_layer1.weight = nn.Parameter(tensor([[-1.]], dtype=torch.float64)) # w_12_11
            ascending_layer1.bias = nn.Parameter(self.x_current) # b_2_1
            ascending_layer1.requires_grad_(False)
            
            ascending_layer2 = nn.Linear(1,1)
            ascending_layer2.weight = nn.Parameter( -1 / (self.x_current - self.x_previous)) # w_23_11
            ascending_layer2.bias = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # b_3_1
            ascending_layer2.requires_grad_(False)
            
            ascending_layer3 = nn.Linear(1,1)
            ascending_layer3.weight =  nn.Parameter(tensor([[1.]], dtype=torch.float64)) # w_34_11
            if x_later!=None:
                ascending_layer3.bias = nn.Parameter(tensor([[-0.5]], dtype=torch.float64)) # b_4_1
            elif x_later==None: # its the last node
                ascending_layer3.bias = nn.Parameter(tensor([[0.]], dtype=torch.float64))
            ascending_layer3.requires_grad_(False)

        # 2. Constructing the output layer (displacement)
        w_45_11 = self.__u
        w_45_21 = self.__u
        self.output_layer = nn.Linear(2,1, bias=False)
        self.output_layer.weight = nn.Parameter(torch.cat([w_45_11, w_45_21], 1))
        self.output_layer.requires_grad_(False)
        
        # 3. Assembling the ascent and descent layers
        if x_previous!=None:   
            self.ascendingLayers = nn.Sequential(
                ascending_layer1,
                nn.ReLU(),
                ascending_layer2,
                nn.ReLU(),
                ascending_layer3
            )
        else:
            self.ascendingLayers = -1
            
        if x_later!=None:
            self.descendingLayers = nn.Sequential(
                descending_layer1,
                nn.ReLU(),
                descending_layer2,
                nn.ReLU(),
                descending_layer3
            )
        else:
            self.descendingLayers = -1
    
    @property
    def u(self):
        return self.__u
    
    @u.setter
    def u(self, u_i):
        self.__u = tensor([[u_i]], dtype=torch.float64).requires_grad_(True)
        w_45_11 = self.__u
        w_45_21 = self.__u
        self.output_layer = nn.Linear(2,1, bias=False)
        self.output_layer.weight = nn.Parameter(torch.cat([w_45_11, w_45_21], 1))
        self.output_layer.requires_grad_(False)
        
    def forward(self, dom):
        dom = tensor(np.array([dom]), dtype=torch.float64).reshape((-1,1))
        if self.ascendingLayers == -1:
            out_ascending = torch.zeros_like(dom)
        else:
            out_ascending = self.ascendingLayers(dom)
        
        if self.descendingLayers == -1:
            out_descending = torch.zeros_like(dom)
        else:
            out_descending = self.descendingLayers(dom)
        out = self.output_layer(torch.cat([out_ascending, out_descending], 1))
        return out
        
        
tensor1 = tensor([[1,2,3]]).reshape(-1,1)
tensor2 = tensor([[3,4,5]]).reshape(-1,1)
tensor3 = tensor([[9,9,9]]).reshape(-1,1)
w1 = tensor([[2]])
w2 = tensor([[4]])
torch.cat([tensor1, tensor2, tensor3], 1), torch.cat([w1, w2], 1)


# Main class (assembles the nodes' neural networks)
class HiDeNN_for_FEM():
    def __init__(self, dom, num_nodes, equation, boundary_conditions=[0,0]):
        '''
        dom: 
        '''
        self.num_nodes = num_nodes
        self.dom = dom
        self.dom_end = dom[-1]
        self.dom_start = dom[0]
        self.nodes = np.linspace(self.dom_start, self.dom_end, num_nodes)
        u_1 = boundary_conditions[0]
        u_n = boundary_conditions[1]
        dx = (self.dom_end-self.dom_start)/len(dom)
        self.num_elem = self.num_nodes - 1
        
        # -ku'' + cu' + bu = f
        k = equation[0]
        c = equation[1]
        b = equation[2]
        f_func = equation[3]
        
        self.node_nets_arr = self.node_nets_func( u_arr=[1 for i in range(num_nodes)] )
        self.shape_func_arr = [func_net.forward(dom) for func_net in self.node_nets_arr]
        self.shape_func_arr = torch.stack(self.shape_func_arr).detach().numpy().reshape(num_nodes, -1)
        
        #### Calculating initial F and K for initial u's #### 
        
        self.F = [integrate(shape*f_func(dom), dx=dx) for shape in self.shape_func_arr[1:-1]]
        self.K = np.zeros((num_nodes, num_nodes))
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                df_i = diff(func_i)/dx
                df_j = diff(func_j)/dx
                integrand = k * ( df_i * df_j ) + c * ( df_i * func_j[1:] ) + b * ( func_i[1:] * func_j[1:] )
                self.K[i, j] = integrate(integrand, dx=dx).reshape(-1)

        # Using the approach of strong condition (big number) - Computationally gives the same result as the theoretically correct way
        # https://scicomp.stackexchange.com/questions/20515/how-to-efficiently-implement-dirichlet-boundary-conditions-in-global-sparse-fini
        BigNumber = 10e64
        self.F = np.append( BigNumber * u_1 , self.F )
        self.F = np.append( self.F, BigNumber * u_n )
        self.K[0,0] = BigNumber
        self.K[-1,-1] = BigNumber
        
        ### Calculating initial displacements and aprox u ### 
        self.displacement_arr = np.array(inv(self.K) @ self.F) # alpha = K^-1 x F
        self.u_aprox_arr = self.displacement_arr @ self.shape_func_arr
            
        for i,node in enumerate(self.node_nets_arr):
            node.u = self.displacement_arr[i]
        
        self.parameters_arr = []
        for node in self.node_nets_arr:
            self.parameters_arr.append( node.u )
            self.parameters_arr.append( node.x_current )
            
        self.__optimizer = torch.optim.SGD(self.parameters_arr, lr=1e-3)
        
    def forward(self):
        out_hidden_layers = [node_net.forward(self.dom) for node_net in self.node_nets_arr]
        
        output_layer = nn.Linear(self.num_nodes, 1, bias=False)
        output_layer.weight = nn.Parameter(torch.ones_like(output_layer.weight, dtype=torch.float64))
        output_layer.requires_grad_(False)
        
        self.u_aprox_arr = output_layer(torch.cat(out_hidden_layers, 1))
        self.u_aprox_arr = self.u_aprox_arr.numpy().reshape(-1)
        print('hello')
        return self.u_aprox_arr
    
    def node_nets_func(self,u_arr):
            arr = [
                Node_Shape_Function_Net(
                    self.nodes[i-1], 
                    self.nodes[i], 
                    self.nodes[i+1], 
                    u_arr[i]
                ) for i in range(1,self.num_nodes-1)
            ]
            arr.insert(0,
                Node_Shape_Function_Net(
                    None,
                    self.nodes[0],
                    self.nodes[1],
                    u_arr[0]
                )
            )
            arr.append(
                Node_Shape_Function_Net(
                    self.nodes[-2],
                    self.nodes[-1],
                    None,
                    u_arr[-1]
                )
            )
            return np.array(arr)
        
    def plot_shape_functions(self):
        fig, axs = plt.subplots(self.num_nodes, sharex=True, sharey=True)
        plt.xticks(self.nodes)
        fig.suptitle('Shape Functions')
        for i,func in enumerate(self.shape_func_arr):
            axs[i].plot(self.dom, func)
            axs[i].grid()
        plt.show()
    
    def plot_u_exact_vs_u_aprox(self, u_exact_arr):
        plt.plot(self.dom, u_exact_arr, label='Solução exata')
        plt.plot(self.nodes, self.displacement_arr, 'o--', label=f'Aproximação com {self.num_elem} elementos')
        plt.xticks(self.nodes)
        plt.grid()
        plt.legend()
        plt.show()

    def train_nodes(self):
        self.__optimizer.zero_grad()
        loss = self.lossfunc()
        loss.backward()
        self.__optimizer.step()
        return loss 
        


# # Tests on the model
equation = [1, 0, 0, lambda x: 1] # [k, c, b, f]
bound_conds = [0,0]
dom = np.linspace(0,1,100)
u_exact = [-x**2/2 + 0.5*x for x in dom]

model = HiDeNN_for_FEM(dom, 6, equation, bound_conds)
model.plot_shape_functions()
model.plot_u_exact_vs_u_aprox(u_exact)
model

model.displacement_arr, model.K, model.nodes, model.parameters_arr, len(model.u_aprox_arr)

model.u_aprox_arr

model.forward()