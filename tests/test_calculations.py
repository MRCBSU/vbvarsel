from variationaltempering.calcparams import *
import numpy as np
# X 'calcAkj_annealed', 
# X 'calcAlphak_annealed', 
# 'calcB_annealed', 
# 'calcC_annealed', 
# X 'calcDelta_annealed',
# 'calcF0', 
# 'calcM_annealed', 
# X 'calcN1_annealed', 
# X 'calcN2_annealed', 
# 'calcS', 
# 'calcXd', 
# 'calcZ_annealed', 
# 'calcbetakj_annealed', 
# 'calcexpF', 
# 'calcexpF0', 
# X 'expPi', 
# 'expSigma', 
# 'expTau',
# X 'normal'


def test_calcAlphak():
    testak = calcAlphak(5,1,1)
    assert testak == 6.0

def test_normal_calc():
    norm = normal(1,1,1)
    assert norm == 0.3989422804014327

def test_n1annealed_calc():
    n1 = calcN1(10,0,0,10)
    assert n1[1] == 0.0 and n1[0] == 1.0

def test_n2annealed_calc():
    n2 = calcN2(1,1,1,1)
    assert n2[1] == -0.5

def test_calc_expi():
    pik = expPi(np.ndarray([1]), np.ndarray([1]))
    assert pik == [0.0]

def test_calc_delta():
    delta_ann = calcDelta([10,10,10], 5, 1)
    assert delta_ann[0] == 1.3636363636363635

def test_calcAkj():
    akj = calcAkj(1, 1, 10, 5, 3.0, 1)
    assert akj == [[28.0]]


