classdef PartialSaveTree < PartialSave & VisitableTree
properties
%     reset_already_ = false;
end
methods
%     function reset_properties(PS)
%         if ~PS.reset_already_
%             PS.reset_properties@PartialSave;
%             PS.set_reset_already_(true);
%         end
%     end
%     function set_reset_already_(PS, tf)
%         assert(islogical(tf));
%         PS.reset_already_ = tf;
%     end
%     function PS2 = saveobj(PS)
%         Visitor = VisitorToTree;
%         try
%             PS2 = deep_copy(PS);
%         catch
%             PS2 = copy(PS);
%         end
%         if ~PS.reset_already_
%             Visitor.eval_tree_recursive(PS2, @reset_properties);
%         end
%     end
end
end