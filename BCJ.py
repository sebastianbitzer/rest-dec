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

set_BCJ = collections.namedtuple('BCJ', 'L p0 p1 p2 p3 a1 a2 a3 m1 m2 m3 l0 l1 l2 l3 l4 niter r0 r1 r2')
def BCJ(f) : return wrap(f,set_BCJ)

scale = 3000


def cost_BCJ(x):
    x = set_BCJ(*x)
    return (max(x.l4, x.l3, 2*x.l3 + x.r2 - x.r1, 2*x.l2 + x.r1 -x.r0, x.l0) +x.niter)*scale #scale up if no successful optimizazion


def optimize_BCJ(q, z, R, W, max_trials, verb=True):
    """
    Optimizes the BCJ algorithm for z = 2 and full weight
    -> uses shifted error vector
    q: field size
    z: restricted size, here z = 2
    R: code rate
    W: weight of error, here full, i.e., W = 1.0
    """

    assert z==2 and W==1.0

    # shift error #
    W = 0.5
    z = 1
    
    Q = log(q, 2)
    Z = log(z, 2)
    Nsol = hbin(W) + W*Z-(1-R)*Q
    assert Nsol<=0.01,('uniqueness of error vector not given!',Nsol)

    cost = cost_BCJ
    mycons = [
    # pi: number of '+1'
    { 'type' : 'ineq', 'fun' : BCJ(lambda x : min(R+x.L, W)    - x.p0)}, # P <= R+L
    { 'type' : 'ineq', 'fun' : BCJ(lambda x : max(1-W-R-x.L,0) + x.p0)}, # P >= W-(1-R-L) = W + R + L -1
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : x.p0/2 + x.a1 - x.p1)},
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : x.p1/2 + x.a2 - x.p2)},
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : x.p2/2 + x.a3 - x.p3)},
    # mi: number of '-1'
    { 'type' : 'eq', 'fun' : BCJ(lambda x :          x.a1 - x.m1)}, 
    { 'type' : 'eq', 'fun' : BCJ(lambda x : x.m1/2 + x.a2 - x.m2)},
    { 'type' : 'eq', 'fun' : BCJ(lambda x : x.m2/2 + x.a3 - x.m3)},
    # success probability
    { 'type' : 'eq', 'fun' :  BCJ(lambda x : Niter(R,W,x.L,x.p0) - x.niter)},
    # number of representations
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : ReprNum(R, x.L, x.p2, x.a3, x.m2, 0.0, 0.0, Z) - x.r2)},
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : ReprNum(R, x.L, x.p1, x.a2, x.m1, 0.0, 0.0, Z) - x.r1)},
    { 'type' : 'eq', 'fun' :   BCJ(lambda x : ReprNum(R, x.L, x.p0, x.a1,  0.0, 0.0, 0.0, Z) - x.r0)},
    { 'type' : 'ineq', 'fun' : BCJ(lambda x : x.r1  - x.r2)},
    { 'type' : 'ineq', 'fun' : BCJ(lambda x : x.r0  - x.r1)}, # r1 <= r0
    { 'type' : 'ineq', 'fun' : BCJ(lambda x : Q*x.L - x.r0)}, # r0 <= Q*L
    # sizes of the lists
    { 'type' : 'eq', 'fun' : BCJ(lambda x : Lsize(R/2, x.L/2, x.p3/2, x.m3/2, Z) - x.l4 )}, #base list 
    { 'type' : 'eq', 'fun' : BCJ(lambda x : 2*x.l4 - x.r2                        - x.l3 )},# conc-merged 
    { 'type' : 'eq', 'fun' : BCJ(lambda x : Lsize(R, x.L, x.p2, x.m2, Z)  - x.r1 - x.l2 )},# repr-merged 
    { 'type' : 'eq', 'fun' : BCJ(lambda x : Lsize(R, x.L, x.p1, x.m1, Z)  - x.r0 - x.l1 )},# repr-merged 
    { 'type' : 'eq', 'fun' : BCJ(lambda x : 2*x.l1                + x.r0 - Q*x.L - x.l0 )},# repr-merged final wt p
    ]
    #        'L p0 p1 p2 p3 a1 a2 a3 m1 m2 m3 l0 l1 l2 l3 l4 niter r0 r1 r2'
    bounds = [(0, 1-R)] + [(max(0, W-(1-R)), W)]+[(0,W)]*3 + [(0,1)]*6  + [(0,10)]*9  
    
    start = [0]*len(bounds)
    for l in range(len(bounds)): # randomize starting point
        start[l] = np.random.uniform(bounds[l][0],bounds[l][1])

    eps = 1e-3
    for _ in range(max_trials):
        result = opt.minimize(cost, start, 
                bounds= bounds, tol=1e-5, 
                constraints=mycons, options={'maxiter':5000})
        adic= set_BCJ(*result.x)._asdict()
        if result.success and  abs( 2*adic['l2'] + adic['r1'] - adic['r0'] - adic['l0']) < eps: 
            break
        else:
            for l in range(len(bounds)):
                start[l] = np.random.uniform(bounds[l][0],bounds[l][1])
            
    astuple = set_BCJ(*result.x)

    if verb:
        print("Validity: ", result.success,"|l0-l1|=",abs( 2*adic['l2'] + adic['r1'] - adic['r0'] - adic['l0']))
        print("q-Time: ", round_to_str(cost(astuple)/scale/Q))
        print("2-Time: ", round_to_str(cost(astuple)/scale))
        for t in adic:
            print(t, round_to_str(adic[t]) )
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    return cost(astuple)/scale, adic, result.success