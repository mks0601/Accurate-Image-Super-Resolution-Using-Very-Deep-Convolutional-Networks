require 'torch'
require 'cunn'
require "data.lua"
dofile "etc.lua"
    

local trainData
local trainLabel

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())

dofile "data.lua"
dofile "model.lua"
dofile "train.lua"
dofile "test.lua"


if mode == "train" then
    trainData, trainLabel = load_data()
    fp_err = io.open("result/loss_" .. testScale .. ".txt","a")
    fp_PSNR = io.open("result/PSNR_" .. testScale .. ".txt","a")
    while epoch <= epochNum do
        train(trainData, trainLabel)
        epoch = epoch + 1
        err = tot_error/cnt_error
        fp_err:write(err,"\n")
        test()
        fp_PSNR:write(PSNR_sum/testDataSz,"\n")
    end
    fp_err:close()
    fp_PSNR:close()
end


if mode == "test" then
    test()
end
