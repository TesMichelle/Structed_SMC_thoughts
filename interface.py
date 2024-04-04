import numpy as np
from numpy import dot

from scipy import linalg
from scipy.integrate import odeint


class MarginalDistr:

    def __init__(self, la1, la2, mu1, mu2):
        #mu1, mu2 - migration  rates
        #la1, la2 - coalescent rates
        #init_cond - vector of length 3
        self.la1 = la1
        self.la2 = la2

        self.mu1 = mu1
        self.mu2 = mu2

        #self.init_cond = np.array(init_cond)/sum(init_cond)

        self.SetMatrix(0.0)

    def m1(self, t):
        return self.mu1

    def m2(self, t):
        return self.mu2

    def l1(self, t):
        return self.la1

    def l2(self, t):
        return self.la2

    def SetMatrix(self, t):
        self.A = np.matrix( [[-2*self.m1(t)-self.l1(t), 0, self.m2(t)],
                             [0, -2*self.m2(t)-self.l2(t), self.m1(t)],
                             [2*self.m1(t), 2*self.m2(t), -self.m1(t) - self.m2(t)]] )

    def CalcProb(self, t, init_cond):
        MET = linalg.expm( dot(self.A, t) )
        return dot(MET, init_cond)

    def CalcDist(self, num_steps, step, init_cond):
        pr_dist = np.zeros(num_steps+1)
        MET = linalg.expm( dot(self.A, step) )
        for i in range(num_steps):
            pr_dist[i] = self.la1*init_cond[0]+self.la2*init_cond[1]
            init_cond = dot(MET, init_cond)
        pr_dist[num_steps] = self.la1*init_cond[0]+self.la2*init_cond[1]
        return pr_dist

class lineage:
    def __init__(self, a, b, p):
        #a and b indicates ancestral material (1) or non-ancestral (0)
        self.a = a
        self.b = b
        self.p = p#0 or 1 population

    def __str__(self):
        return str(self.a)+str(self.b)+str(self.p)

class state:
    def __init__(self, lng):
        self.lng = lng
        self.l = len(lng)
        self.name()
        self.stat()

    def name(self):
        lng_s = []
        for l in self.lng:
            lng_s.append( str(l.a) + str(l.b) + str(l.p) )
        lng_s.sort()
        self.n = ','.join(lng_s)

    def stat(self):
        self.num = {'a': 0, 'b': 0}
        self.index = {'a':[-1,-1],'b':[-1,-1]}
        ai, bi = 0, 0
        for i, l in enumerate(self.lng):
            if l.a == 1:
                self.num['a'] += 1
                self.index['a'][ai] = i
                ai += 1
            if l.b == 1:
                self.num['b'] += 1
                self.index['b'][bi] = i
                bi += 1

    def __str__(self):
        prstr = self.n + ": " + "num_a = " + str(self.num['a']) + ", " + "num_b = " + str(self.num['b']) + "\n    " + "index_a = " + str(self.index['a'][0:self.num['a']]) + "\n    " + "index_b = " + str(self.index['b'][0:self.num['b']])
        return(prstr)

numStates =[
    [ [1,1,0] ], # добавил два состояния
    [ [1,1,1] ],

    [ [1,1,0],[1,1,0] ],
    [ [1,1,0],[1,1,1] ],
    [ [1,1,1],[1,1,1] ],

    [ [1,0,0],[1,1,0] ],
    [ [1,0,0],[1,1,1] ],
    [ [1,0,1],[1,1,0] ],
    [ [1,0,1],[1,1,1] ],

    [ [0,1,0],[1,1,0] ],
    [ [0,1,0],[1,1,1] ],
    [ [0,1,1],[1,1,0] ],
    [ [0,1,1],[1,1,1] ],

    [ [0,1,0],[1,0,0] ],
    [ [0,1,0],[1,0,1] ],
    [ [0,1,1],[1,0,0] ],
    [ [0,1,1],[1,0,1] ],

    [ [1,0,0],[0,1,0],[1,1,0] ],
    [ [1,0,0],[0,1,0],[1,1,1] ],
    [ [1,0,0],[0,1,1],[1,1,0] ],
    [ [1,0,0],[0,1,1],[1,1,1] ],
    [ [1,0,1],[0,1,0],[1,1,0] ],
    [ [1,0,1],[0,1,0],[1,1,1] ],
    [ [1,0,1],[0,1,1],[1,1,0] ],
    [ [1,0,1],[0,1,1],[1,1,1] ],

    [ [1,0,0],[0,1,0],[0,1,0] ],
    [ [1,0,0],[0,1,0],[0,1,1] ],
    [ [1,0,0],[0,1,1],[0,1,1] ],
    [ [1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,1],[0,1,1],[0,1,1] ],

    [ [1,0,0],[1,0,0],[0,1,0] ],
    [ [1,0,0],[1,0,0],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,0] ],
    [ [1,0,0],[1,0,1],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,0] ],
    [ [1,0,1],[1,0,1],[0,1,1] ],

    [ [1,0,0],[1,0,0],[0,1,0],[0,1,0] ],
    [ [1,0,0],[1,0,0],[0,1,0],[0,1,1] ],
    [ [1,0,0],[1,0,0],[0,1,1],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,0],[1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,1],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,1],[1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,1],[0,1,1] ]
]

Mdim = len(numStates)

states = []
stateToNum = {}
i = 0
for st in numStates:
    lng = []
    for l in st:
        lng.append( lineage(l[0], l[1], l[2]) )
    states.append( state(lng) )
    stateToNum[states[-1].n] = i
    i += 1

def Eq(P, t):
    return dot(dot(M,params(t)),P)

def Coal(st, i, k):
    if i > st.l or k > st.l:
        print("Wrong lineage index.")
    if st.lng[i].p != st.lng[k].p:
        print("Coalescence is impossible.")
    newState = [st.lng[j] for j in range(st.l) if i != j and k != j]
    newState.append(lineage(st.lng[i].a | st.lng[k].a, st.lng[i].b | st.lng[k].b, st.lng[i].p))
    newState = state(newState)
    if newState.l > 0: # добавил наличие поглощающего состояния длины 1
        return stateToNum[newState.n]
    else:
        return -1

#Matrix entry [rho, l1, l2, m1, m2] Thank's...

def Matrix():
    trRate = np.zeros( (Mdim, Mdim, 5), int ) # :,:,0 - recom., :,:,1-2 - coal., :,:,3-4 - migr.
    for s in range(Mdim):
        st = states[s]
        for i in range(st.l):
            #recombination
            if st.lng[i].a & st.lng[i].b:
                newState = [st.lng[j] for j in range(st.l) if i != j]
                newState.append(lineage(1,0,st.lng[i].p))
                newState.append(lineage(0,1,st.lng[i].p))
                newState = state(newState)
                trRate[s, stateToNum[newState.n], 0] += 1
                trRate[s, s, 0] -= 1
            #coalescence
            for k in range(i+1,st.l):
                if st.lng[i].p == st.lng[k].p:
                    newState_id = Coal(st, i, k)
                    if newState_id != -1:
                        trRate[s, newState_id, st.lng[i].p+1 ] += 1
                    trRate[s, s, st.lng[i].p+1 ] -= 1
            #migration
            newState = [st.lng[j] for j in range(st.l) if i != j]
            newState.append( lineage(st.lng[i].a,st.lng[i].b,(st.lng[i].p+1)%2) )
            newState = state(newState)
            trRate[s, stateToNum[newState.n], st.lng[i].p+3 ] += 1
            trRate[s, s, st.lng[i].p+3 ] -= 1
    return( np.transpose(trRate, (1,0,2)) )

M = Matrix()

def FirstCoal(locus, t, params, popID = -1):
    matr = np.zeros( (Mdim, Mdim) )
    for s, st in enumerate(states):
        if not (st.num['a'] == 2 and st.num['b'] == 2 and st.l > 2):
            continue
        lng_i0, lng_i1 = st.index[locus][0], st.index[locus][1]
        if not (popID == -1 and st.lng[lng_i0].p == st.lng[lng_i1].p) and not (popID != -1 and st.lng[lng_i0].p == popID and st.lng[lng_i1].p == popID):
            continue
        newState_id = Coal(st, lng_i0, lng_i1)
        matr[ newState_id, s ] += params(t)[ st.lng[lng_i0].p + 1 ]
    return matr

def SecondCoal(locus, t, params, popID = -1):
    matr = np.zeros( (Mdim, Mdim) )

    locus2 = 'b'
    if locus == 'b':
        locus2 = 'a'

    for s, st in enumerate(states):
        if not (st.num[locus] == 2 and st.num[locus2] == 1):
            continue
        lng_i0, lng_i1 = st.index[locus][0], st.index[locus][1]
        if not (popID == -1 and st.lng[lng_i0].p == st.lng[lng_i1].p) and not (popID != -1 and st.lng[lng_i0].p == popID and st.lng[lng_i1].p == popID):
            continue
        newState_id = Coal(st, lng_i0, lng_i1)
        #print(st.n, states[newState_id].n)
        matr[ newState_id, s ] += params(t)[ st.lng[lng_i0].p + 1 ]
    return matr

def DoubleCoal(t, params, pop_a = -1, pop_b = -1): # полностью новая функция
    matrd = np.zeros( (Mdim, Mdim) )
    matrd2 = np.zeros( (Mdim, Mdim) )
    for s, st in enumerate(states):
        if (pop_a == pop_b and st.num['a'] == 2 and st.num['b'] == 2 and st.l == 2):
            lng_i0, lng_i1 = st.index['a'][0], st.index['a'][1]
            if (pop_a == -1 and st.lng[lng_i0].p == st.lng[lng_i1].p) \
            or (pop_a != -1 and st.lng[lng_i0].p == pop_a and st.lng[lng_i1].p == pop_a):
                newState_id = Coal(st, lng_i0, lng_i1)
                matrd[ newState_id, s ] += params(t)[ st.lng[lng_i0].p + 1 ]
        elif (st.num['a'] == 2 and st.num['b'] == 2 and st.l > 2):
            lng_i0, lng_i1, lng_i2, lng_i3 = st.index['a'][0], st.index['a'][1], st.index['b'][0], st.index['b'][1]
            pa0, pa1, pb0, pb1 = st.lng[lng_i0].p, st.lng[lng_i1].p, st.lng[lng_i2].p, st.lng[lng_i3].p
            if pop_a == -1 or (pop_a == pa0 and pop_b == pb0):
                if st.l == 3 and (pa0 + pa1 + pb0 + pb1) % 4 == 0:
                    newState_id = stateToNum[f'11{pa0}']
                    matrd2[ newState_id, s ] += params(t)[ pa0 + 1 ]**2
                elif st.l == 4 and pa0 == pa1 and pb0 == pb1:
                    newState_id = stateToNum[state([lineage(0, 1, pa0), lineage(1, 0, pb0)]).n]
                    matrd2[ newState_id, s ] += params(t)[ pa0 + 1 ] * params(t)[ pb0 + 1 ]
    return matrd, matrd2

def filename(scenario, dim, dt, rho):
    fn = "data_" + str(dim) + "_" + str(dt) + "_" + str(rho) + ".npy"
    fn = "data/" + scenario + "/" + fn
    return(fn)


def Eq(P, t, params):
    return dot(dot(M,params(t)),P)

def solve_eq(params, times, Pinit):
    sol = odeint(lambda P, t: Eq(P, t, params), Pinit, times)
    return(sol[1,])

def solve_eq2(params, times, Pinit):
    sol = odeint(lambda P, t: Eq(P, t, params), Pinit, times)
    return(sol)

def densityTaTb(method, Pinit, params, Ta, Tb, h, pop_a = -1, pop_b = -1):
    if method == 'ode':
        return densityTaTb_ode(Pinit, params, Ta, Tb, pop_a=pop_a, pop_b=pop_b)
    else:
        return densityTaTb_expm(Pinit, params, Ta, Tb, h, pop_a=pop_a, pop_b=pop_b)

def densityTaTb_ode(Pinit, params, Ta, Tb, pop_a = -1, pop_b = -1):
    if Ta < Tb:
        P = solve_eq(params, [0., Ta], Pinit)
        P = dot(FirstCoal('a', Ta, params, pop_a), P)

        P = solve_eq(params, [Ta, Tb], P)
        P = dot(SecondCoal('b', Tb, params, pop_b), P)
    elif Ta > Tb:
        P = solve_eq(params, [0., Tb], Pinit)
        P = dot(FirstCoal('b', Tb, params, pop_b), P)

        P = solve_eq(params, [Tb, Ta], P)
        P = dot(SecondCoal('a', Ta, params, pop_a), P)
    else:
        P = solve_eq(params, [0, Ta], Pinit)
        P = dot(DoubleCoal(Ta, params, pop_a, pop_b), P)
    return( sum(P) )

# вставил домножение на шаг по времени внутрь функции, изменил саму фкнцию немного
def densityTaTb_expm(Pinit, params, Ta, Tb, h, pop_a = -1, pop_b = -1):
    if Ta < Tb:
        Mexp = linalg.expm( dot(M, params(0))*Ta ) # (M, params) = good transition matrix, MAGIC!!!!!
        P = dot(Mexp, Pinit)
        # print(P)
        P = dot(FirstCoal('a', Ta, params, pop_a), P)
        Mexp = linalg.expm( dot(M, params(0))*(Tb-Ta) )
        P = dot(Mexp, P)
        P = dot(SecondCoal('b', Tb, params, pop_b), P) *h
    elif Ta > Tb:
        Mexp = linalg.expm( dot(M, params(0))*Tb )
        P = dot(Mexp, Pinit)
        P = dot(FirstCoal('b', Tb, params, pop_b), P)
        Mexp = linalg.expm( dot(M, params(0))*(Ta-Tb) )
        P = dot(Mexp, P)
        P = dot(SecondCoal('a', Ta, params, pop_a), P) *h
    else:
        Mexp = linalg.expm( dot(M, params(0))*Ta )
        P = dot(Mexp, Pinit)
        matrd, matrd2 = DoubleCoal(Ta, params, pop_a, pop_b)
        P = dot(matrd, P) + dot(matrd2, P) * h
    return P.sum()
