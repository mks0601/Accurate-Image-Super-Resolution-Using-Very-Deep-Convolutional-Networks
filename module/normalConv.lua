require 'nn'
require 'cudnn'

do

    local SpatialConvolution, parent = torch.class('cudnn.normalConv', 'cudnn.SpatialConvolution')
    
    -- override the constructor to have the additional range of initialization
    function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, mean, std)
        parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function SpatialConvolution:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:zero()
        else
            self.weight:normal(0,1)
            self.bias:zero()
        end
    end

end
