# ================= Definitions of mathematical variables ================================================

# a - single action of the agent
# A - space of all agent actions (control process)
# delta = time step
# D - time grid where D_T = {n*delta, n <= T/delta}
# f - instantaneous reward function
# g - terminal constant

# gamma - dynamic learning rate
# gammaBase - initial learning rate
#
# H - optimal reward function to be used as basis for comparison to actual outcome (ie: error)
# k - step at which agent may make decisions (for discrete model)

# m - error function of deviation from optimal policy
# M - the problem we are trying to solve: E[m(q,X(z),z)] = 0

# q - the solution to the problem we are trying to solve with RL: E[m(q,X(z),z)] = 0
#   - q(z) = the current decision by the agent given the current state of the system 'z'
# Q - the optimal expected gain when the agent starts at 'z' and chooses action 'a' at time 't' is the sum 
#   - of the next expected reward 'R' plus the value of acting optimally starting from the new position 'U' 
#   - at time 't + delta'
#   - satisfies the classical dynamic programming principal (DPP)
#   - solves the equation: E[m(q,X^z,z)] = 0

# rho - discount factor
# R - discounted incremental reward from t to t + delta
# t - current time
# T - final time

# u - current state of the transition probabilities
# U - process defined on filtered probability space which represents the state of the system
# x - current state of transition probability 'u' and 'r'
# X - X(z): a random variable with an unknown distribution
# z - current state of the system: z = (t,u,a)
# Z - Markov chain: discrete state space of the system where Z* = {1,...,d} and Z = [0,T] x U x A
#




# ================================ Functions and Equations ================================================

# f(t,U,A) = ?
# H(q,x,z) = r + rho^delta sup(a') q(t+delta,u,a')
# m(q,x,z) = H(q,x,z) - q(z)

# Q(t,u,a) = E[R + rho^delta * sup(A) Q(t+delta, U, A) | U = u, A = a]
# R = integral from t to 't+delta' of rho * f(s, U, A) ds

# 

# x = (u,r)
# z = (t,u,a)


# ================================ Helper Functions ================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.animation import FuncAnimation

# create functions for self-correcting learning rates

def high(gamma, gammaBase):
    h = np.minimum(gamma + gammaBase, 3*gammaBase)
    return h

def low(gamma, gammaBase):
    l = np.maximum(gamma - gammaBase, gammaBase)
    return l

# create function for actual reward based on agent decision
def q_agent(z):
    '''The solution to the problem we are trying to solve with RL: E[m(q,X(z),z)] = 0 
    q(z) = the current decision by the agent given the current state of the system z'''
    q = z
    return q

# create function for optimal reward from optimal decision
def H_optimal(q,x,z):
    '''Optimal discounted reward function to be used as basis for comparison to actual outcome (ie: error)'''
    h = x + rho * q
    return h

# create function for error term m which we want to minimize
def m_error(q,x,p=None, f=None, c=None, tau=None, T=None,type='drift'):
    '''Error function of deviation from optimal policy. Agent wants to minimize this.
    m(q,x,z1) = H(q,x,z1) - q(z1)'''
    if type == 'drift':
        m = q - x
    elif type == 'placement':
        m = q - (f + tau * c)
    else:
        m = H_optimal(q,x,z) - q
    return m


# create time series

def timeSeries(S0,mu,sigma,n,T, seed, plot='yes'):
    '''Generate GBM with inputs for initial price, mu, sigma, num steps, T. seed= yes/no. plot=yes/no.'''
    S = [S0]
    dS = []
    dt = T/250
    
    if seed == 'yes':
        np.random.seed(1)
        draws = np.random.normal(size=n)
        drift = (mu - 0.5*sigma*sigma)*dt
        stoch = sigma*np.sqrt(dt)*draws
               
        for i in range(n):
            S.append(S[i]*np.exp(drift + stoch[i]))
            dS.append(S[i+1] - S[i])
            
        if plot == 'yes':
            plt.title('time series (n='+ str(n)+')')
            plt.plot(S)
            plt.show()
    else:
        draws = np.random.normal(size=n)
        drift = (mu - 0.5*sigma*sigma)*dt
        stoch = sigma*np.sqrt(dt)*draws
               
        for i in range(n):
            S.append(S[i]*np.exp(drift + stoch[i]))
            dS.append(S[i+1] - S[i])
            
        if plot == 'yes':
            plt.title('time series (n='+ str(n)+')')
            plt.plot(S)
            plt.show()
    return S, dS




def drift(S0,mu,sigma,n,T,seed,plot='yes',episodes=10,w=5):
    '''RL with stochastic learning rate (PASS algo) to estimate the drift of a time series.
    Input: initial price, mu, sigma, num steps, T, seed=yes/no, plot=yes/no, episodes=10, w=5.
    Output: vector of final q-values, plots for TS and L2 error vs iterations.'''
    S,dS = timeSeries(S0,mu,sigma,n,T,seed,plot=plot)
    
    q0 = 1
    gamma0 = .5
    
    q = np.ones(n)*q0
    gamma = np.ones(n)*gamma0
    gammaHat = np.ones(n)*gamma0
    gammaHat[0] = gamma[0]
    Epast = np.zeros(n)
    EpastNorm = []
    count = 0
    x = np.linspace(0,100,episodes-1)
    
    qlist = []
    for episode in range(1,episodes):
        for i in range(n):
            Xdelta = S[i+1] - S[i]
            
            if episode == 1:
                q[i] = q[i] - gammaHat[i] * m_error(q[i],Xdelta)
                
            elif m_error(q[i],Xdelta) * Epast[i] >= 0:
                
                gammaHat[i] = high(gammaHat[i],gamma[i])
                q[i] = q[i] - gammaHat[i] * m_error(q[i],Xdelta)
            
            elif m_error(q[i],Xdelta) * Epast[i] <= 0:
                gammaHat[i] = (low(gammaHat[i],gamma[i]))
                q[i] = q[i] - gammaHat[i] * m_error(q[i],Xdelta)
            
            Epast[i] = (m_error(q[i],Xdelta))
        EpastNorm.append(np.linalg.norm(Epast))
        
        if count > w:
            if np.mean(EpastNorm[i-2*w:i-w]) <= np.mean(EpastNorm[i-w:i]):
                gamma[i:] = np.maximum(gamma[i]/2,.01)
        count += 1
    if plot == 'yes':
        # errors
        plt.title('L2 Error vs Iteration Count')
        plt.plot(EpastNorm,x)
        plt.show()
        
        # scatter
        plt.title('drift vs q-values')
        plt.scatter(dS,q)
        plt.show()
               
    return q







