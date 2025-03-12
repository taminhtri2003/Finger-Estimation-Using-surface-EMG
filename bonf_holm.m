function [h, p_adj] = bonf_holm(p_unc, alpha)
%BONF_HOLM  Bonferroni-Holm corrected p-values
%
%   [h, p_adj] = bonf_holm(p_unc, alpha)
%
%   Returns h, a logical array indicating which p-values are significant,
%   given the uncorrected p-values, p_unc, and the FWER, alpha, using the
%   Bonferroni-Holm procedure. Also returns p_adj, the adjusted p-values.

if nargin < 2 || isempty(alpha)
    alpha = 0.05;
end

[p_sorted, sort_idx] = sort(p_unc);
m = length(p_unc);
k = 1:m;
p_adj = min(1, max(p_sorted .* (m - k + 1), [], 2)); % Adjusted p-values

 % Undo the sort
 unsort_idx(sort_idx) = 1:m;
 p_adj = p_adj(unsort_idx);
 h = p_adj <= alpha; % Significant tests

end