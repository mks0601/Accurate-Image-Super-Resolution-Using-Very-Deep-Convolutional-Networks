require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
require 'module/normalLinear'
dofile 'etc.lua'


model = nn.Sequential()
kernelSz = 3

model:add(cudnn.normalConv(inputDim,fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*inputDim))))
model:add(nn.SpatialBatchNormalization(fDim))
model:add(nn.ReLU(true))

for lid = 1,n do
    model:add(cudnn.normalConv(fDim,fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*fDim))))
    model:add(nn.SpatialBatchNormalization(fDim))
    model:add(nn.ReLU(true))
end

model:add(cudnn.normalConv(fDim,outputDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*fDim))))
model:add(nn.SpatialBatchNormalization(outputDim))

criterion = nn.MSECriterion()
criterion.sizeAverage = false

--print(model)

cudnn.convert(model, cudnn)

model:cuda()
criterion:cuda()


