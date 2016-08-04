require 'torch'
require 'optim'
require 'xlua'
require 'image'
dofile 'etc.lua'

params, gradParams = model:getParameters()
optimState = {
    learningRate = lr,
    learningRateDecay = 0.0,
    weightDecay = wDecay,
    momentum = mmt,
}
optimMethod = optim.sgd
tot_error = 0
cnt_error = 0
epoch = 0

function train(trainData, trainLabel)
    local time = sys.clock()
    
    tot_error = 0
    cnt_error = 0
    local iter_cnt = 0

    model:training()
    shuffle = torch.randperm(trainSz)
    
    local inputs = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    local targets = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)

    for t = 1,trainSz,batchSz do
        
        if t+batchSz-1 > trainSz then
            inputs = torch.CudaTensor(trainSz-t+1,inputDim,inputSz,inputSz)
            targets = torch.CudaTensor(trainSz-t+1,inputDim,inputSz,inputSz)
            curBatchDim = trainSz-t+1
        else
            curBatchDim = batchSz
        end

        for i = t,math.min(t+batchSz-1,trainSz) do
            
            local input = trainData[shuffle[i]]
            local target = trainLabel[shuffle[i]]
            input = torch.reshape(input,inputDim,inputSz,inputSz)
            target = torch.reshape(target,inputDim,inputSz,inputSz)
            
            --[===[
            if t==1 and i<t+20 then
                img = torch.Tensor(inputDim,inputSz,inputSz)
                img[1] = input[1]
                image.save(tostring(i) .. ".jpg",img)
                img[1] = target[1]
                image.save(tostring(i) .. "_.jpg",img)
            end
            --]===]
                        
            inputs[i-t+1]:copy(input)
            targets[i-t+1]:copy(target)
        end
        
        if epoch > 0 and epoch%20 == 0 then
            optimState.learningRate = optimState.learningRate * 0.1
        end
             
        local feval = function(x)
           if x ~= params then
              params:copy(x)
           end

           gradParams:zero()
           local output = model:forward(inputs)
           local err = criterion:forward(output,targets)
           model:backward(inputs,criterion:backward(output,targets))
           err = err/curBatchDim
           tot_error = tot_error + err
           cnt_error = cnt_error + 1

           gradParams:clamp(-lr_theta/optimState.learningRate,lr_theta/optimState.learningRate)
           return err,gradParams
        end

         optimMethod(feval, params, optimState)
        
        if iter_cnt % 1000 == 0 then
            print("epoch: " .. epoch .. "/" .. epochNum .. " batch: " ..  t .. "/" .. trainSz .. " loss: " .. tot_error/cnt_error)
        end
        iter_cnt = iter_cnt + 1

    end
   
    if epoch == epochNum then
        local filename = paths.concat(save_dir, modelName)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end
end


