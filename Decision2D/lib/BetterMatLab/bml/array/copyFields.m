function obj = copyFields(obj, varargin) % [ix], src, fieldNames, suppressError, excludeFields)
% COPYFIELDS : Copies the latter's properties or fields to the former.
%
% obj = copyFields(obj, structOrObjOrDataset)
% obj = copyFields(obj, structOrObjOrDataset, {fieldName1, fieldName2, ...}, 
%                  suppressError=false, excludeFields=false)
% obj = copyFieldsIx(obj, ix, src, fieldNames, suppressError, excludeFields)
%
% If no field names are specified, copies all fields.
%
% If excludeFields = false, copy only the specified fields.
% If excludeFields = true,  copy all fields except the specified fields.
%
% suppressError: 0: rethrow; 1: warn; 2: ignore
%
% See also: copy_fields, data, PsyLib
%
% 2013-2014 (c) Yul Kang. See help PsyLib for the license.

if isnumeric(varargin{1}) % index specified
    ixmode = true;
    ix  = varargin{1};
    siz = size(ix);
    ix  = ix(:)';
    varargin = varargin(2:end);
else
    ixmode = false;
end

src = varargin{1};
[fieldNames, suppressError, excludeFields] = dealDef(varargin(2:end), ...
    {{}, false, false}, true);

%% Field names
if isempty(fieldNames), 
    fieldNames = fieldnames(src)'; % Copy all fields
end 
if excludeFields,
    fieldNames = setdiff(fieldnames(src)', fieldNames, 'stable');
end
if isa(src, 'dataset'), 
    fieldNames = setdiff(fieldNames, 'Properties', 'stable');
end
    
%% Copy fields
for iField = 1:length(fieldNames)    
    if ixmode
        for ii = ix
            try
                obj(ii).(fieldNames{iField}) = src.(fieldNames{iField});
            catch cError
                switch suppressError
                    case 0
                        rethrow(cError);
                    case 1
                        warning(err_msg(cError));
                end
            end
        end
    else
        try
%             if isa(src, 'dataset')
%                 obj.(fieldNames{iField}) = src(:,fieldNames{iField});        
%             else
                obj.(fieldNames{iField}) = src.(fieldNames{iField});
%             end
        catch cError
            switch suppressError
                case 0
                    rethrow(cError);
                case 1
                    warning(err_msg(cError));
            end
        end
    end
end

%% Reshape into original size
if ixmode
    obj = reshape(obj, siz);
end
end