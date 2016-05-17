require 'nn'

do

    local Linear, parent = torch.class('nn.normalLinear', 'nn.Linear')
    
    -- override the constructor to have the additional range of initialization
    function Linear:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize,outputSize)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:zero()
        else
            self.weight:normal(0,1)
            self.bias:zero()
        end
    end

end
