function[patch] = get_patches(picdims, mids, num_cycles, sd_Gauss, patchFactor)
 onecycle  = picdims(2) ./ fliplr(mids);
 
 sd_pic = onecycle .* sd_Gauss;
 picsize  = onecycle .* num_cycles * patchFactor; % large pics so that gauss comes to zero
 k = (sd_pic./picsize) * 2; 
        
 for md = 1:numel(mids)
    w1 = window(@gausswin, round(picsize(md)),1/k(md)); %previously k = 0.35
    patch{md} = w1*w1';
 end
end
