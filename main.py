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
from macros import *
from BCJ import optimize_BCJ
from BJMM2 import optimize_BJMM2
from BJMM3 import optimize_BJMM3

#####################################
# functions for reproducing numbers #
#####################################

def make_plot_over_wt(num_lvl,q, z, R, max_trials, prec, num_iter, verb=False):
    """
    use this function for recreating Figure 2, 3 and 4
    """
    assert num_lvl == 2 or num_lvl == 3
    Z = log(z, 2)
    Q = log(q, 2)
    for w in range(1,prec):
        W = w/prec
        if hbin(W) +W*Z <= Q*(1-R): # check uniqueness condition
            Cost_opt = 1000
            for _ in range(num_iter):
                if num_lvl == 2: Cost, params, success = optimize_BJMM2(q, z, R, W, max_trials, verb)
                if num_lvl == 3: Cost, params, success = optimize_BJMM3(q, z, R, W, max_trials, verb)
                if success and Cost < Cost_opt:
                    Cost_opt = Cost
            print("(",round_to_str(W),",",round_to_str(Cost_opt),") %", round_to_str(Cost_opt/Q))
        else:
            print("uniqueness not given!")

def make_table(max_trials, num_iter, verb=False):
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
            if z == 2 and W == 1.0:
                Cost_BCJ, adic, success = optimize_BCJ(q, z, R, W, max_trials, False)
                if success and Cost_BCJ < Cost_opt:
                    Cost_opt = Cost_BCJ
                    num_lvl = 3
                    if verb:
                        print("== BCJ is best ==")
                        for t in adic:
                            print(t, round_to_str(adic[t]) )
            else:
                Cost2, adic, success2 = optimize_BJMM2(q, z, R, W, max_trials, False)
                if success2 and Cost2 < Cost_opt:
                    Cost_opt = Cost2
                    num_lvl = 2
                    if verb:
                        print("== 2 is best ==")
                        for t in adic:
                            print(t, round_to_str(adic[t]) )

                Cost3, adic, success3 = optimize_BJMM3(q, z, R, W, max_trials, False)
                if success3 and Cost3 < Cost_opt:
                    Cost_opt = Cost3
                    num_lvl = 3
                    if verb:
                        print("== 3 is best ==")
                        for t in adic:
                            print(t, round_to_str(adic[t]) )
        
        if z == 2 and W == 1.0:
            print(f'z={z}, q={q}, n={n}, R={R}, W={W}, claim={claim}, BCJ({num_lvl})={round_to_str(Cost_opt*n)} ({round_to_str(Cost_opt)})')
            print('-------------------------------------------------------------------')
        else:
            print(f'z={z}, q={q}, n={n}, R={R}, W={W}, claim={claim}, BJMM_+({num_lvl})={round_to_str(Cost_opt*n)} ({round_to_str(Cost_opt)})')
            print('-------------------------------------------------------------------')


###################
# reproduce plots #
###################
q = 157
R = 0.5
max_trials = 100
prec = 5
num_iter = 5
verb = False
num_lvl = 3

# plot for z = 2
z = 2
print('------------------------ figure z = 2 ------------------------------------')
make_plot_over_wt(num_lvl,q, z, R, max_trials, prec, num_iter, verb)

# plot for z = 2
z = 4
print('------------------------ figure z = 4 ------------------------------------')
make_plot_over_wt(num_lvl,q, z, R, max_trials, prec, num_iter, verb)

# plot for z = 6
z = 6
print('------------------------ figure z = 6 ------------------------------------')
make_plot_over_wt(num_lvl,q, z, R, max_trials, prec, num_iter, verb)

###################
# reproduce table #
###################
print('----------------------- Parameter Table ---------------------------')
make_table(max_trials, num_iter, False)

