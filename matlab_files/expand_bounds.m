function [Bup,Blo] = expand_bounds(t,B0,a,d,USfunc)

t = t(:);
Bup=B0*(t==t);
s=t>d;

switch USfunc
    case 'Linear'
        Bup(s)=B0-a*(t(s)-d);
    case 'Quadratic'
        Bup(s)=B0-a*(t(s)-d).^2;
    case 'Exponential'
        Bup(s)=B0*exp(-a*(t(s)-d));
    case 'Logistic'
        Bup = B0*1./(1+exp(a*(t-d)));
    case 'Hyperbolic'
        Bup(s) = B0*1./(1+(a*(t(s)-d)));
    case 'Step'
        Bup(s)=B0*1e-3;
    case {'None','Deadline'}
        % Just use Bup itself
    otherwise
        error('USfunc not recognized')
end

% Bounds stop collapsing when the bounds 
% become less than 0.1% of initial height.
Bup(Bup<=B0*1e-3, 1) = B0*1e-3;


Blo=-Bup;