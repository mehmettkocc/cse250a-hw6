%{
Author: Mehmet Koc
Date: 11/21/13
Description: HMM Speech Recognition...
only inference no learning, parameters are available
implementing Viterbi's Algorithm
%}
close all
clear
%read data
Ot = importdata('observations.txt') + 1';
A = importdata('transitionMatrix.txt');
B = importdata('emissionMatrix.txt');
pi = importdata('initialStateDistribution.txt');
%%
T = length(Ot);
n = length(pi);
%compute max log-likelihood for all t=1:T
%L = [l1, l2, ..., lT]
L = zeros(n, T);
%also save the most probable state transitions
%phi = [phi1, phi2, ..., phiT] where phi1 is not meaningful
phi = zeros(n, T);
l1 = log(pi) + log(B(:, Ot(1)));
L(:, 1) = l1;
for i = 2:T
    lMat = repmat(L(:, i-1), [1, n]); 
    [maxL, ind] = max(lMat + log(A));
    L(:, i) = maxL' +  log(B(:, Ot(i)));
    phi(:, i) = ind';
end
%%
%best hidden state sequence
S = zeros(T, 1);
[~, S(T)] = max(L(:, T));
%backtracking
for i = (T-1):-1:1
    S(i) = phi(S(i+1), i+1);
end
isRepeated = false(T, 1);
for i = 2:T
   if(S(i) == S(i-1))
       isRepeated(i) = 1;
   end
end
%decode and eliminate repetitions
sentence = S(~isRepeated);
alphabet = char(97:122);
sentenceDecoded = alphabet(sentence);
%%
%plot the most likely sequence of hidden states vs time
figure, plot(S, 'linewidth', 2);
xlim([1, T]); ylim([0, n+1]);
xlabel('Time (t=1:T)'); ylabel('Hidden state value (1 to 26)');
title('The most likely sequence of hidden states vs. time');
