function save_parallel(savefilename,struct_to_save,append_flag)
if nargin<3 || isempty(append_flag)
    append_flag = 0;
end
if append_flag==1
    save(savefilename,'-struct','struct_to_save','-append');
else
    save(savefilename,'-struct','struct_to_save');
end
