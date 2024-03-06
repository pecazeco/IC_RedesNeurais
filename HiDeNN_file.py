from torch import tensor
from numpy import linspace
import torch
from torch.linalg import solve
from torch import nn
import matplotlib.pyplot as plt
import importlib
import Node_Net_File
importlib.reload(Node_Net_File)
from Node_Net_File import Node_Shape_Function_Net

def diff(y, dx):
    dydx = torch.zeros(len(y)-1)
    for i in range(len(y)-1):
        dydx[i] = (y[i+1]-y[i]) / dx
    return dydx

class HiDeNN_for_FEM():
    def __init__(self, dom, num_nodes, equation, boundary_conditions=[0,0]):
        '''
        dom: 
        '''
        self.num_nodes = num_nodes
        self.dom = tensor(dom, dtype=torch.float64).reshape(1,-1)
        self.dom_end = dom[-1]
        self.dom_start = dom[0]
        self.nodes = tensor(linspace(self.dom_start, self.dom_end, num_nodes), requires_grad=True)
        self.u_1 = boundary_conditions[0]
        self.u_n = boundary_conditions[1]
        self.dx = (self.dom_end-self.dom_start)/len(dom)
        self.num_elem = self.num_nodes - 1
        
        # -ku'' + cu' + bu = f
        self.k = equation[0]
        self.c = equation[1]
        self.b = equation[2]
        self.f_func = lambda x_v: list(map(equation[3], x_v))
        
        self.shape_func_arr = self.node_nets_func( torch.ones((self.num_nodes,1), dtype=torch.float64) )
        
        #### Calculating initial F and K for initial u's #### 
        
        self.__F = torch.trapz( tensor(self.f_func(dom)) * self.shape_func_arr, dx=self.dx ) 
        self.__K = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                
                df_i = diff(func_i, self.dx)
                df_j = diff(func_j, self.dx)
                
                integrand = self.k * ( df_i * df_j ) + self.c * ( df_i * func_j[1:] ) + self.b * ( func_i[1:] * func_j[1:] )
                self.__K[i, j] = torch.trapz(integrand, dx=self.dx).reshape(-1)

        # Using the approach of strong condition (big number) - Computationally gives the same result as the theoretically correct way
        # https://scicomp.stackexchange.com/questions/20515/how-to-efficiently-implement-dirichlet-boundary-conditions-in-global-sparse-fini
        BigNumber = 10e10
        #self.__F = torch.cat( ( tensor([BigNumber * self.u_1]), self.__F ) )
        self.__F[0] = BigNumber * self.u_1
        #self.__F = torch.cat( ( self.__F, tensor([BigNumber * self.u_n]) ) )
        self.__F[-1] = BigNumber * self.u_n
        self.__K[0,0] = BigNumber
        self.__K[-1,-1] = BigNumber
        
        ### Calculating initial displacements and aprox u ### 
        self.displacement_arr = solve(self.__K, self.__F) # alpha = K^-1 x F
        self.u_aprox_arr = self.displacement_arr @ self.shape_func_arr
        
        self.node_nets_arr = self.node_nets_func( self.displacement_arr )    
        
    def forward(self):
        self.node_nets_arr = self.node_nets_func( self.displacement_arr )
        self.shape_func_arr = self.node_nets_func( torch.ones((self.num_nodes,1), dtype=torch.float64) )
        
        output_layer = nn.Linear(self.num_nodes, 1, bias=False)
        output_layer.weight = nn.Parameter(torch.ones_like(output_layer.weight, dtype=torch.float64))
        output_layer.requires_grad_(False)
        
        self.u_aprox_arr = output_layer(torch.transpose(self.node_nets_arr, 0, 1))
        return self.u_aprox_arr
    
    def node_nets_func(self,u_arr):
            shape_tensor = tensor([])
            dom = self.dom.reshape(-1,1)
            for i in range(1, self.num_nodes-1):
                t = Node_Shape_Function_Net(
                        self.nodes[i-1], 
                        self.nodes[i], 
                        self.nodes[i+1], 
                        u_arr[i]
                    ).forward(dom).reshape(1, -1)
                shape_tensor = torch.cat((shape_tensor, t), 0)
                
            t = Node_Shape_Function_Net(
                    None,
                    self.nodes[0],
                    self.nodes[1],
                    u_arr[0]
                ).forward(dom).reshape(1, -1)
            shape_tensor = torch.cat((t, shape_tensor), 0)

            t = Node_Shape_Function_Net(
                    self.nodes[-2],
                    self.nodes[-1],
                    None,
                    u_arr[-1]
                ).forward(dom).reshape(1, -1)
            shape_tensor = torch.cat((shape_tensor, t), 0)
            
            return shape_tensor
     
    def update_node_nets(self):
        self.node_nets_arr = self.node_nets_func( self.displacement_arr )
        self.shape_func_arr = self.node_nets_func( torch.ones((self.num_nodes,1)) )
    
    @property
    def K(self):
        self.__K = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float64)
        for i,func_i in enumerate(self.shape_func_arr):
            for j,func_j in enumerate(self.shape_func_arr):
                
                df_i = diff(func_i, self.dx)
                df_j = diff(func_j, self.dx)
                    
                integrand = self.k * ( df_i * df_j ) + self.c * ( df_i * func_j[1:] ) + self.b * ( func_i[1:] * func_j[1:] )
                self.__K[i, j] = torch.trapz(integrand, dx=self.dx).reshape(-1)

        BigNumber = 10e10
        self.__K[0,0] = BigNumber
        self.__K[-1,-1] = BigNumber
        return self.__K
    
    @property
    def F(self):
        self.__F = torch.trapz( tensor(self.f_func(self.dom)) * self.shape_func_arr, dx=self.dx ) 
        # self.__F = [integrate( shape * tensor(self.f_func(self.dom)), dx=self.dx ) for shape in self.shape_func_arr[1:-1]]

        BigNumber = 10e10
        #self.__F = torch.cat( ( tensor([BigNumber * self.u_1]), self.__F ) )
        self.__F[0] = BigNumber * self.u_1
        #self.__F = torch.cat( ( self.__F, tensor([BigNumber * self.u_n]) ) )
        self.__F[-1] = BigNumber * self.u_n
        return self.__F
        
    def plot_shape_functions(self):
        figure = plt.figure()
        figure, axs = plt.subplots(self.num_nodes, sharex=True, sharey=True)
        nodes = self.nodes.detach().numpy()
        plt.xticks(nodes)
        figure.suptitle('Shape Functions')
        for i,func in enumerate(self.shape_func_arr):
            dom = self.dom.reshape(-1).detach().numpy()
            func = func.detach().numpy()
            axs[i].plot(dom, func)
            axs[i].grid()
        #return figure
    
    def plot_u_exact_vs_u_aprox(self, u_exact_arr):
        figure = plt.figure()
        dom = self.dom.reshape(-1).detach().numpy()
        figure = plt.plot(dom, u_exact_arr, label='Solução exata')
        nodes = self.nodes.detach().numpy()
        figure = plt.plot(nodes, self.displacement_arr, 'o--', label=f'Aproximação com {self.num_elem} elementos')
        figure = plt.xticks(nodes)
        figure = plt.grid()
        figure = plt.legend()
        #return figure
    
    def train(self, epochs=1, lossfunc=nn.MSELoss(), lr=1e-3):        
        for epoch in range(epochs):
            # -k * u''(x) + c * u'(x) + b * u(x) = f(x)
            u = self.forward().reshape(-1)
            
            dudx = diff(u, self.dx)
            ddudx2 = diff(dudx, self.dx)
            
            self.optimizer = torch.optim.SGD([self.nodes], lr=lr)
            self.optimizer.zero_grad()
            loss = lossfunc( 
                            ( -self.k * ddudx2 + self.c * dudx[1:] + self.b * u[2:] - tensor(self.f_func(self.dom[0,2:])) ).requires_grad_(True),
                            tensor([[0]], dtype=torch.float64)
                            )
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            self.shape_func_arr = self.node_nets_func(torch.ones((self.num_nodes,1), dtype=torch.float64))
            
            ### Calculating initial displacements and aprox u ### 
            self.displacement_arr = solve(self.K,self.F) # alpha = K^-1 x F
            self.u_aprox_arr = self.displacement_arr @ self.shape_func_arr
            
            self.node_nets_arr = self.node_nets_func(self.displacement_arr)
                
        return loss
    
        