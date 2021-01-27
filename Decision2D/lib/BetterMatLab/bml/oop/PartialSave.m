classdef PartialSave < matlab.mixin.Copyable
% Abandoned. Just use Transient, and save them separately if necessary.
% Otherwise, results in infinite loop in case of circular reference.
% Workaround is possible but not worth all the time.
% Separate saving of transient properties can be implemented,
% e.g., with keep_on_save, by copying chosen transient properties to
% regular property into saved_properties_.*
%
% PartialSave - Resets designated properties on save.
properties
%     to_reset_ = struct;
%     verbose_reset_ = false; % true; % Set to true for debugging
    saved_property_values_ = struct;
%     properties_to_save_ = {};
%     verbose_save_ = false;
end
methods
%     function empty_on_save(PS, props)
%         % empty_on_save(PS, props)
%         if ~iscell(props), props = {props}; end
%         
%         for ii = 1:length(props)
%             PS.reset_on_save(props{ii}, []);
%         end
%     end
%     function reset_on_save(PS, varargin)
%         % reset_on_save_(PS, varargin)
%         PS.to_reset_ = varargin2S(varargin, PS.to_reset_);
%     end
%     function keep_on_save(PS, props)
%         % keep_on_save_(PS, props)
%         %
%         % Current implementation wastes memory but should work
%         if ~iscell(props)
%             props = {props};
%         end
%         for prop = props(:)'
%             PS.saved_property_values_.(prop{1}) = PS.(prop{1});
%         end
%     end
%     function reset_properties(PS)
% %         if PS.verbose_reset_ % DEBUG
%             fprintf('PartialSave is resetting properties before save for %s: ', ...
%                 class(PS));
%             C = fieldnames(PS.to_reset_);
%             fprintf(' %s', C{:});
%             fprintf('\n');
% %         end
%         copyFields(PS, PS.to_reset_);        
%     end
%     function PS2 = saveobj(PS)
%         try
%             PS2 = deep_copy(PS);
%         catch
%             PS2 = copy(PS);
%         end
%         
%         PS2.reset_properties;        
%     end
end
methods (Static)
%     function PS = loadobj(PS)
%         for f = fieldnames(PS.saved_property_values_)'
%             PS.(f{1}) = PS.saved_property_values_.(f{1});
%         end
%         PS.saved_property_values_ = struct;
%     end
end
end