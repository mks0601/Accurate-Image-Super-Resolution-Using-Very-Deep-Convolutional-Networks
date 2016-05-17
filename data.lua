require "torch"
require "image"
dofile "etc.lua"


function load_data()
    
    local data
    local label
    
    for sid,scale in pairs(trainScale) do 
    
        fnameX = db_dir .. "trainX_" .. scale .. ".bin"
        fnameY = db_dir .. "trainY_" .. scale .. ".bin"
        print("trainDB loading... scale: " .. scale)
        
        local fpX = torch.DiskFile(fnameX,"r"):binary()
        local fpY = torch.DiskFile(fnameY,"r"):binary()
    
        fpX:seekEnd()
        fpY:seekEnd()
        local len = fpX:position()-1
        local lineNum = len/(inputSz*inputSz*inputDim)
        fpX:seek(1)
        fpY:seek(1)
        
        trainSz = 6*lineNum
        curData = torch.FloatTensor(trainSz, inputSz*inputSz*inputDim)
        curLabel = torch.FloatTensor(trainSz, inputSz*inputSz*inputDim)

        for lid = 1,lineNum do
            
            local dataX = torch.ByteTensor(fpX:readByte(inputSz*inputSz)):type('torch.FloatTensor')        
            local dataY = torch.ByteTensor(fpY:readByte(inputSz*inputSz)):type('torch.FloatTensor')

            --augmentation
            dataOrigin = torch.reshape(dataX,inputDim,inputSz,inputSz)
            dataFlipV = image.flip(dataOrigin,2)  
            dataFlipH = image.flip(dataOrigin,3)
            dataRot1 = image.rotate(dataOrigin,math.pi/2)
            dataRot2 = image.rotate(dataOrigin,math.pi/2*2)
            dataRot3 = image.rotate(dataOrigin,math.pi/2*3)

            labelOrigin = torch.reshape(dataY,inputDim,inputSz,inputSz)
            labelFlipV = image.flip(labelOrigin,2)
            labelFlipH = image.flip(labelOrigin,3)
            labelRot1 = image.rotate(labelOrigin,math.pi/2)
            labelRot2 = image.rotate(labelOrigin,math.pi/2*2)
            labelRot3 = image.rotate(labelOrigin,math.pi/2*3)

            curData[{{(lid-1)*6 + 1},{1,inputSz*inputSz}}] = torch.reshape(dataOrigin,inputDim*inputSz*inputSz)
            curData[{{(lid-1)*6 + 2},{1,inputSz*inputSz}}] = torch.reshape(dataFlipV,inputDim*inputSz*inputSz)
            curData[{{(lid-1)*6 + 3},{1,inputSz*inputSz}}] = torch.reshape(dataFlipH,inputDim*inputSz*inputSz)
            curData[{{(lid-1)*6 + 4},{1,inputSz*inputSz}}] = torch.reshape(dataRot1,inputDim*inputSz*inputSz)
            curData[{{(lid-1)*6 + 5},{1,inputSz*inputSz}}] = torch.reshape(dataRot2,inputDim*inputSz*inputSz)
            curData[{{(lid-1)*6 + 6},{1,inputSz*inputSz}}] = torch.reshape(dataRot3,inputDim*inputSz*inputSz)

            curLabel[{{(lid-1)*6 + 1},{1,inputSz*inputSz}}] = torch.reshape(labelOrigin,inputDim*inputSz*inputSz)
            curLabel[{{(lid-1)*6 + 2},{1,inputSz*inputSz}}] = torch.reshape(labelFlipV,inputDim*inputSz*inputSz)
            curLabel[{{(lid-1)*6 + 3},{1,inputSz*inputSz}}] = torch.reshape(labelFlipH,inputDim*inputSz*inputSz)
            curLabel[{{(lid-1)*6 + 4},{1,inputSz*inputSz}}] = torch.reshape(labelRot1,inputDim*inputSz*inputSz)
            curLabel[{{(lid-1)*6 + 5},{1,inputSz*inputSz}}] = torch.reshape(labelRot2,inputDim*inputSz*inputSz)
            curLabel[{{(lid-1)*6 + 6},{1,inputSz*inputSz}}] = torch.reshape(labelRot3,inputDim*inputSz*inputSz)


            --[===[
            if lid < 20 then
                img = data[{{lid},{}}]
                img = torch.reshape(img,inputDim,inputSz,inputSz)
                image.save(lid .. ".jpg",img/255)
            end
            --]===]

        end
        
        fpX:close()
        fpY:close()

        if sid == 1 then
            data = curData
            label = curLabel
        else
            data = torch.cat(data,curData,1)
            label = torch.cat(label,curLabel,1)
        end

    end 
    
    trainSz = data:size()[1]
    shuffle = torch.randperm(trainSz):type('torch.LongTensor')
    data = data:index(1,shuffle)
    label = label:index(1,shuffle)

    data = data/255
    label = label/255
    label = label - data

    return data, label
end

