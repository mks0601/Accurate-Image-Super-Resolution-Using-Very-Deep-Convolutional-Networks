require 'torch'
require 'xlua' 
require 'optim'
require 'image'

local function MSE(x1, x2)
   return (x1 - x2):pow(2):mean()
end

local function PSNR(x1, x2)
   local mse = MSE(x1, x2)
   return 10 * math.log10((255*255) / mse)
end

local function loadTestData(id)
    
    fnameX = test_dir .. 'testX' .. id .. '_' .. testScale .. '.bin'
    fnameY = test_dir .. 'testY' .. id .. '_' .. testScale .. '.bin'
    fnameSize = test_dir .. 'testSize' .. id .. '_' .. testScale .. '.txt'

    local fpX = torch.DiskFile(fnameX,"r"):binary()
    local fpY = torch.DiskFile(fnameY,"r"):binary()
    local fpSize = io.open(fnameSize,'r')

    local hei = fpSize:read()
    local wid = fpSize:read()

    local input = torch.ByteTensor(fpX:readByte(hei*wid)):type('torch.FloatTensor')
    local target = torch.ByteTensor(fpY:readByte(hei*wid)):type('torch.FloatTensor')

    input = torch.reshape(input,inputDim,hei,wid)
    target = torch.reshape(target,outputDim,hei,wid)
    
    input = input/255
    
    fpX:close()
    fpY:close()
    fpSize:close()
    return input,target
end

function test()
    
    if mode == "test" then
        print("model loading...")
        model = torch.load(save_dir .. modelName)
    end
    
    model:evaluate()
    
    print('==> testing:')
    PSNR_sum = 0 
    for did = 1,testDataSz do
        
        local input,target
        input,target = loadTestData(did)

        local sz = input:size()
        input = input:cuda()
        input = torch.reshape(input,1,inputDim,sz[2],sz[3])
        

        local output = model:forward(input)
        input = torch.reshape(input,inputDim,sz[2],sz[3])
        output = torch.reshape(output,outputDim,sz[2],sz[3])
        
        output = output + input
        input = input*255
        output = output*255

        target = target:cuda()
        target = torch.reshape(target,outputDim,sz[2],sz[3])
        
        local cropSz = testScale
        output = image.crop(output:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        target = image.crop(target:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        input = image.crop(input:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        

        image.save("result/input_" .. did .. ".jpg",input/255)
        image.save("result/output_" .. did .. ".jpg",output/255)
        image.save("result/target_" .. did .. ".jpg",target/255)
        
        PSNR_sum = PSNR_sum + PSNR(output,target)
    end

    print('PSNR: ' .. PSNR_sum/testDataSz)

end
