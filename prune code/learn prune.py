# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:30:27 2020
参考两个大佬写的代码做的一点小改进
https://github.com/mightydeveloper/Deep-Compression-PyTorch
https://github.com/mepeichun/Efficient-Neural-Network-Bilibili

剪枝里面有使用意义的代码
供参考
详细改动代码见————
@author: yz
"""

"""
对线性层继承，添加属性mask，及其剪枝功能

"""
class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = None
        
    def set_mask(self):
        if self.mask_flag == False :
            self.mask = Parameter(torch.ones([self.out_features, self.in_features]), requires_grad=False)
            self.weight.data = self.weight.data * self.mask.data
            self.mask_flag = True
        else: print("多次调用")

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        #计算阈值（若小于阈值）重新定义mask————置零
        # Apply new weight and mask!!!
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)   
        
        
"""
对卷积层继承，添加属性mask，及其剪枝功能

"""
class MaskConv2d(nn.Conv2d):
   def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.temp = kernel_size
        super(MaskConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False 
   def set_mask(self):
       if self.mask_flag == False :
            self.mask = Parameter(torch.ones([self.out_channels,self.in_channels,self.temp, self.temp]), requires_grad=False)
            self.mask_flag = True
    
   def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
   def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
   def prune(self, p_id_start=0, p_id_end=1):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        mask[ p_id_start: p_id_end,:,:,:] = 0 
        new_mask = mask 
        #计算阈值（若小于阈值）重新定义mask————置零
        # Apply new weight and mask!!!
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)    
        
"""
选择层进行掩码操作
剪枝的阈值可以自己选，被置零卷积核可以选取

"""

 for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.set_mask()                    #打开并生成
                module.prune(threshold)
        for name, module in self.named_modules():
            if name in ['conv1', 'conv2', 'conv3']:
                module.set_mask()                    #打开并生成
                module.prune(p_id_start,p_id_end)

"""
这一块比较厉害，在训练数据的时候，阻止被置零的权重进行更新


"""
def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():  #model.named_parameters() 返回层的名字及参数
             #   print(name)
             #   print(p)
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()   
                grad_tensor = p.grad.data.cpu().numpy() #数据转换到cpu上 numpy会在cpu上加速运算而Tensor能在GPU上加速运算
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
                ###如果参数w为0，那么他的梯度也被置零，这样做也就是说在后向传播的时候不改变已经置零的值的梯度
            optimizer.step()
            
            
            
            
            
"""
遇到新的方法还是会更新

"""