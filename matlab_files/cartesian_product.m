function A = cartesian_product(varargin)
% ALLCOMB - All combinations
%    B = ALLCOMB(A1,A2,A3,...,AN) returns all combinations of the elements
%    in A1, A2, ..., and AN. B is P-by-N matrix is which P is the product
%    of the number of elements of the N inputs.
%    Empty inputs yields an empty matrix B of size 0-by-N. Note that
%    previous versions (1.x) simply ignored empty inputs.
%
%    Example:
%       allcomb([1 3 5],[-3 8],[0 1]) 
%         1  -3   0
%         1  -3   1
%         1   8   0
%         ...
%         5  -3   1
%         5   8   1
%
%       % Inputs can be character arrays as well:
%       allcomb('abc','XY')
%         aX
%         aY
%         bX
%         bY
%         cX
%         cY
%
%       % or combined (output will be a character array)
%       allcomb('ab',[65 66])
%         aA
%         aB
%         bA
%         bB
%
%    ALLCOMB(A1,..AN,'matlab') causes the first column to change fastest.
%    This is more consistent with matlab indexing. Example:
%    allcomb(1:2,3:4,5:6,'matlab') %->
%      1   3   5
%      2   3   5
%      1   4   5
%      ...
%      2   4   6
%
%    This functionality is also known as the cartesian product.
%
%    See also NCHOOSEK, PERMS, NDGRID
%    and COMBN, KTHCOMBN (Matlab Central FEX)

% for Matlab R13+
% version 2.2 (jan 2012)
% (c) Jos van der Geest
% email: jos@jasen.nl

% History
% 1.1 (feb 2006), removed minor bug when entering empty cell arrays;
%     added option to let the first input run fastest (suggestion by JD)
% 1.2 (jan 2010), using ii as an index on the left-hand for the multiple
%     output by NDGRID. Thanks to Jan Simon, for showing this little trick
% 2.0 (dec 2010). Bruno Luong convinced me that an empty input should
% return an empty output.
% 2.1 (feb 2011). A cell as input argument caused the check on the last
%      argument (specifying the order) to crash.
% 2.2 (jan 2012). removed a superfluous line of code (ischar(..))
% 3.0 (may 2012) removed check for doubles so character arrays are accepted

% error(nargchk(1,Inf,nargin)) ;
narginchk(1,Inf) ;

% check for empty inputs
q = ~cellfun('isempty',varargin) ;
if any(~q),
    warning('ALLCOMB:EmptyInput','Empty inputs result in an empty output.') ;
    A = zeros(0,nargin) ;
else
    
    ni = sum(q) ;
    
    argn = varargin{end} ;

    if ischar(argn) && (strcmpi(argn,'matlab') || strcmpi(argn,'john')),
        % based on a suggestion by JD on the FEX
        ni = ni-1 ;
        ii = 1:ni ;
        q(end) = 0 ;
    else
        % enter arguments backwards, so last one (AN) is changing fastest
        ii = ni:-1:1 ;
    end
    
    if ni==0,
        A = [] ;
    else
        args = varargin(q) ;
        if ni==1,
            A = args{1}(:) ;
        else
            % flip using ii if last column is changing fastest
            [A{ii}] = ndgrid(args{ii}) ;
            % concatenate
            A = reshape(cat(ni+1,A{:}),[],ni) ;
        end
    end
end
