function [count, data, label ] = extractPatch(count, data, label, scale, stride, size_input, filepaths, dir)

 for i = 1 : length(filepaths)
    
    image = imread(fullfile(dir,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));                      
   
    im_label = modcrop(image, scale);
    [hei,wid,channel] = size(im_label);
   
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = im_label(x : x+size_input-1, y : y+size_input-1,:);
            
            %imshow(subim_input);
            %waitforbuttonpress;
            
            count=count+1;
            data(:, :, :, count) = im2uint8(subim_input);
            label(:, :, :, count) = im2uint8(subim_label); 
             

        end
    end
    
     %imshow(image);
     %waitforbuttonpress;
end
end

