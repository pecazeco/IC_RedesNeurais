import torch
from torch import tensor
from torch import nn

class Node_Shape_Function_Net(nn.Module):
    def __init__(self, x_previous, x_current, x_later, u_i): 
        super().__init__()
        self.x_current = x_current.reshape(1,1)
        print(self.x_current.grad_fn, 'self.x_current')
        self.u = u_i
        print(self.u.grad_fn, 'self.u')
        self.x_later = x_later.reshape(1,1) if x_later!=None else None
        self.x_previous = x_previous.reshape(1,1) if x_previous!=None else None
        self.layers = {}
        if self.x_later!=None: # its not the last node
            # self.x_later.requires_grad = True
            
            self.layers['descending_layer1_weight'] = tensor([[1.]], dtype=torch.float64) # w_12_12\n",
            self.layers['descending_layer1_bias'] = -self.x_current # b_2_2\n"
            
            self.layers['descending_layer2_weight'] =  -1 / (self.x_later - self.x_current) # w_23_22
            self.layers['descending_layer2_bias'] = tensor([[1.]], dtype=torch.float64) # b_3_2
            
            self.layers['descending_layer2_weight'] = -1 / (self.x_later - self.x_current) # w_23_22\n",
            self.layers['descending_layer2_bias'] = tensor([[1.]], dtype=torch.float64) # b_3_2\n"
            
            self.layers['descending_layer3_weight'] = tensor([[1.]], dtype=torch.float64) # w_34_22\n"
            
            if self.x_previous!=None:
                self.layers['descending_layer3_bias'] = tensor([[-0.5]], dtype=torch.float64) # b_4_2\n"
            elif self.x_previous==None: # its the first node\n",
                self.layers['descending_layer3_bias'] = tensor([[0.]], dtype=torch.float64)
            
        if self.x_previous!=None: # its not the first node
            
            self.layers['ascending_layer1_weight'] = tensor([[-1.]], dtype=torch.float64) # w_12_11
            self.layers['ascending_layer1_bias'] = self.x_current # b_2_1
            
            self.layers['ascending_layer2_weight'] = -1 / (self.x_current - self.x_previous) # w_23_11
            self.layers['ascending_layer2_bias'] = tensor([[1.]], dtype=torch.float64) # b_3_1
            
            self.layers['ascending_layer3_weight'] = tensor([[1.]], dtype=torch.float64) # w_34_11
            
            if self.x_later!=None:
                self.layers['ascending_layer3_bias'] = tensor([[-0.5]], dtype=torch.float64) # b_4_1
            elif self.x_later==None: # its the last node
                self.layers['ascending_layer3_bias'] = tensor([[0.]], dtype=torch.float64)

        # 2. Constructing the output layer (displacement)
        w_45_11 = self.u.reshape(1,1)
        w_45_21 = self.u.reshape(1,1)
        self.layers['output_layer_weight'] = torch.cat([w_45_11, w_45_21],0)
    
    def forward(self, dom):
        #print(dom.grad_fn, 'dom grad_fn')
        
        if self.x_previous == None:
            out_ascending = torch.zeros_like(dom)
        else:
            out_ascending = dom @ self.layers['ascending_layer1_weight'] + self.layers['ascending_layer1_bias']
            out_ascending = torch.relu(out_ascending)
            out_ascending = out_ascending @ self.layers['ascending_layer2_weight'] + self.layers['ascending_layer2_bias']
            out_ascending = torch.relu(out_ascending)
            out_ascending = out_ascending @ self.layers['ascending_layer3_weight'] + self.layers['ascending_layer3_bias']
            #print(out_ascending.grad_fn, 'outascending grad')
        
        if self.x_later == None:
            out_descending = torch.zeros_like(dom)
        else:
            out_descending = dom @ self.layers['descending_layer1_weight'] + self.layers['descending_layer1_bias']
            out_descending = torch.relu(out_descending)
            out_descending = out_descending @ self.layers['descending_layer2_weight'] + self.layers['descending_layer2_bias']
            out_descending = torch.relu(out_descending)
            out_descending = out_descending @ self.layers['descending_layer3_weight'] + self.layers['descending_layer3_bias']
            #print(out_descending.grad_fn, 'outdescending grad')
            
        out = torch.cat([out_ascending, out_descending], 1) @ self.layers['output_layer_weight']
        print(out.grad_fn, 'out grad')
        print(out.is_leaf, 'out')
        
        #return (out, out.grad_fn.data)
        return out
        