"""
Code for reproducing the figures and table of the ISIT 2023 paper 
    "Generic Decoding of Restricted Errors"

authors:
    Sebastian Bitzer
    Alessio Pavoni
    Violetta Weger
    Paolo Santini
    Marco Baldi
    Antonia Wachter-Zeh
"""

import collections
from math import*
import scipy.optimize as opt
import numpy as np

#####################
# general functions #
#####################

def round_to_str(t):
    """
    Rounds s.t 4 digit precision
    """
    s = str(round(t,4))
    return (s + "0" * (5 + s.find(".") -len(s)))

def check_constraints(constraints, solution) : 
    return [ (constraint['type'], constraint['fun'](solution)) for constraint in constraints ]

def wrap(f,g) :
    def inner(x):
        return f(g(*x))
    return inner

def xlx(x):
    if x<=0: return - 100*x
    return x*log(x, 2)

def hbin(x):
    """
    exponential coeff of binomial
    """
    if 0<x<1: return -x*log(x, 2) - (1-x)*log(1-x, 2)
    return 0

def trinom(a,b):
    """
    exponential coeff of trinomial
    """
    return - xlx(a) - xlx(b) - xlx(1-a-b)



###########################
# BJMM-specific functions #
###########################

def Niter(R,W,L,P):
    """
    number of iterations given by 1/P_succ
    R: code rate
    W: overall weight
    L: redundancy of small instance
    P: relative weight of small instance
    """
    return -xlx(W)-xlx(1-W) -xlx(1-R-L) + xlx(W-P) + xlx(1-R-L-W+P) - xlx(R+L) + xlx(P) + xlx(R+L-P)

def Lsize(R,L,P,M,Z):
    """
    intermediate list sizes for BJMM2
    R: rate
    L: red small instance
    P: small instance
    """
    Num = (R+L) * trinom(P/(R+L),M/(R+L)) + Z*(P+M) 

    if    Z == log(6,2):  Num +=  10*M # E = E_+ for z = 6

    return Num
    
def ReprNum(R,L, P, E, M, B, C, D, Z):
    """
    intermediate list sizes for BJMM2
    R: rate
    L: redundancy of small instance
    P: weight after merge
    E: number of overlaps to 0
    M: elements from E_+ after merge
    B: overlaps E_+ and E
    D: overlaps E and E
    """
    repr = 0
    if M > C > 0: 
        repr = M*trinom(C/M, (1-C/M)/2) 
        if Z == 2 or Z == log(6,2): repr += C

    t = R+L-P-M
    if t > 0.0: repr += t*hbin(E/t) + Z*E 

    if P > 0.0 and 2*D + 2*B <= P:  
        if  Z == 1 or Z == 2:  repr += P + P*hbin(2*B/P) 
        elif   Z == log(6,2):  repr += P* trinom(2*D/P, 0.5-D/P)#P*hbin(P1/P) + P1*hbin( 2*D/P1) #+ 2*(P/2-D)*hbin(B/(P/2-D))
        
        if      Z == 1:            repr +=    0*B - 10*D # z = 2 does not have add. structure in E 
        elif    Z == 2:            repr +=    2*B - 10*D # z = 4 does not have add. structure in E
        #elif    Z == log(6,2):     repr +=    2*B +  2*D 
        elif    Z == log(6,2):     repr +=  -10*B +  2*D # z = 6 does have additive structure (and we use E_+)
    return repr

def ReprBCJ(R,L, P, E, M, Z):
    """
    intermediate list sizes for BJMM2
    R: rate
    L: redundancy of small instance
    P: weight after merge
    E: number of overlaps to 0
    M: elements from E_+ after merge
    """
    repr = M + P
    t = R+L-P-M
    if t > 0.0: repr += t*hbin(2*E/t) + 2*E
    return repr




#####################################
# functions for reproducing numbers #
#####################################

def make_plot_over_wt(q, z, R, max_trials, prec, num_iter, verb=True):
    """
    use this function for recreating Figure 2, 3 and 4
    """
    Z = log(z, 2)
    Q = log(q, 2)
    for w in range(1,prec):
        W = w/prec
        if hbin(W) +W*Z <= Q*(1-R): # check uniqueness condition
            Cost_opt = 1000
            for _ in range(num_iter):
                Cost2, params, success = optimize_BJMM2(q, z, R, W, max_trials, verb)
                if success and Cost2 < Cost_opt:
                    Cost_opt = Cost2
            print("(",round_to_str(W),",",round_to_str(Cost_opt),") %", round_to_str(Cost_opt/Q))
        else:
            print("uniqueness not given!")

def make_table(max_trials, num_iter, verb=True):
    """
    use this function for recreating Table 1
    """
    zs = [2]*7 + [4]*7 + [6]*3
    qs = [16381]*3 + [32749]*2 + [29, 31, 109,157,197,137,157,173,193,139,157,193]
    ns = [400, 500, 400, 500, 600, 167, 256, 270, 312, 384, 272, 312, 344, 384, 276, 312, 384]
    Rs = [0.75, 0.75, 0.8, 0.75, 0.66, 0.79, 0.8] + [0.5]*3 + [0.2]*7
    Ws = [0.16, 0.13, 0.14, 0.13, 0.14] + [1.0]*2 +[0.34]*3 + [0.6]*7

    claims = [128]*5 + [87, 128, 125, 144, 177, 88,101, 111, 124, 89, 101, 124]

    for i in range(len(qs)):
        z = zs[i]
        q = qs[i]
        n = ns[i]
        R = Rs[i]
        W = Ws[i]
        claim = claims[i]
        
        num_lvl = -1
        Cost_opt = 1000

        for _ in range(num_iter):
            Cost2, _, success2 = optimize_BJMM2(q, z, R, W, max_trials, verb)
            if success2 and Cost2 < Cost_opt:
                Cost_opt = Cost2
                num_lvl = 2

            Cost3, _, success3 = optimize_BJMM3(q, z, R, W, max_trials, verb)
            if success3 and Cost3 < Cost_opt:
                Cost_opt = Cost2
                num_lvl = 3
        
        print(f'z={z}, q={q}, n={n}, R={R}, W={W}, claim={claim}, BJMM_+({num_lvl})={round_to_str(Cost_opt*n)}')
        print('-------------------------------------------------------------------')


