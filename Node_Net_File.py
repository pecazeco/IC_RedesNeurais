import torch
from torch import tensor
from torch import nn

class Node_Shape_Function_Net(nn.Module):
    def __init__(self, x_previous, x_current, x_later, u_i): 
        super().__init__()
        self.x_current = x_current.reshape(1,1)
        # print(self.x_current.grad_fn, 'self.x_current')
        self.u = u_i
        # print(self.u.grad_fn, 'self.u')
        self.x_later = x_later.reshape(1,1) if x_later!=None else None
        self.x_previous = x_previous.reshape(1,1) if x_previous!=None else None
        
        self.update_layers()
        
    def update_layers(self):
        if self.x_later!=None: # its not the last node
            # self.x_later.requires_grad = True
            
            descending_layer1 = nn.Linear(1,1)
            descending_layer1.weight = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # w_12_12\n",
            descending_layer1.bias = nn.Parameter(-self.x_current) # b_2_2\n",
            descending_layer1.requires_grad_(False)
            
            descending_layer2 = nn.Linear(1,1)
            descending_layer2.weight = nn.Parameter( -1 / (self.x_later - self.x_current) ) # w_23_22
            descending_layer2.bias = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # b_3_2
            
            descending_layer2 = nn.Linear(1,1)
            descending_layer2.weight = nn.Parameter( -1 / (self.x_later - self.x_current) ) # w_23_22\n",
            descending_layer2.bias = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # b_3_2\n",
            descending_layer2.requires_grad_(False)
            
            descending_layer3 = nn.Linear(1,1)
            descending_layer3.weight = nn.Parameter(tensor([[1.]], dtype=torch.float64)) # w_34_22\n",
            if self.x_previous!=None:
                descending_layer3.bias = nn.Parameter(tensor([[-0.5]], dtype=torch.float64)) # b_4_2\n",
            elif self.x_previous==None: # its the first node\n",
                descending_layer3.bias = nn.Parameter(tensor([[0.]], dtype=torch.float64))
            descending_layer3.requires_grad_(False)
            
        if self.x_previous!=None: # its not the first node
            # self.x_previous.requires_grad = True
            
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
            if self.x_later!=None:
                ascending_layer3.bias = nn.Parameter(tensor([[-0.5]], dtype=torch.float64)) # b_4_1
            elif self.x_later==None: # its the last node
                ascending_layer3.bias = nn.Parameter(tensor([[0.]], dtype=torch.float64))
            ascending_layer3.requires_grad_(False)

        # 2. Constructing the output layer (displacement)
        w_45_11 = self.u.reshape(1,1)
        w_45_21 = self.u.reshape(1,1)
        self.output_layer = nn.Linear(2,1, bias=False)
        self.output_layer.weight = nn.Parameter(torch.cat([w_45_11, w_45_21], 1))
        self.output_layer.requires_grad_(False)
        
        # 3. Assembling the ascent and descent layers
        if self.x_previous!=None:   
            self.ascendingLayers = nn.Sequential(
                ascending_layer1,
                nn.ReLU(),
                ascending_layer2,
                nn.ReLU(),
                ascending_layer3
            )
        else:
            self.ascendingLayers = -1
            
        if self.x_later!=None:
            self.descendingLayers = nn.Sequential(
                descending_layer1,
                nn.ReLU(),
                descending_layer2,
                nn.ReLU(),
                descending_layer3
            )
        else:
            self.descendingLayers = -1     
    
    def forward(self, dom):
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
        