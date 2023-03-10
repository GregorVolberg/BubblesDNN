function [picdims, facedims, npic, mids] = load_stimuli(picfilename)
tmp  = load(picfilename); 
npic = tmp.struct_npic.npic;
mids = tmp.struct_npic.mids;
picdims = tmp.struct_npic.picdims;
facedims = tmp.struct_npic.facedims;
end
