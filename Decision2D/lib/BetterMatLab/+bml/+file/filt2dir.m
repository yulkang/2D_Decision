function d = filt2dir(filt)
% Extracts the folder from filter string, e.g., with *.ext.
%
% Unlike fileparts, filt2str works even when 
% the folder name contains period,
% and/or when the folder name is not followed by filesep,
% as long as the folder exists.
%
% d = filt2dir(filt)
%
% EXAMPLE:
% % fileparts needs filesep at the end
% >> fileparts('Data/Confusing.Folder') 
% ans =
% Data
%  
% % filt2dir works, as long as the folder exists.
% >> bml.file.filt2dir('Data/Confusing.Folder')
% ans =
% Data/Confusing.Folder
%
% % filt2dir reserves filesep if it is at the end, as needed for rsync.
% >> bml.file.filt2dir('Data/Confusing.Folder/')
% ans =
% Data/Confusing.Folder/
%
% >> bml.file.filt2dir('Data/Confusing.Folder/*.mat')
% ans =
% Data/Confusing.Folder
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

if bml.file.exist_dir(filt)
    d = filt;
else
    d = fileparts(filt);
end