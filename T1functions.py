import numpy as np

def calc_RichardsonNo(g_prime, q_th, q_v):
    '''

    :param g_prime: Reduced gravity
    :param q_th: Turbidity current thickness
    :param q_v: Turbidity flow speed
    :return: Richardson number
    '''

    with np.errstate(divide='ignore', invalid='ignore'):
        Ri = g_prime * q_th / (q_v ** 2)  # Richardson number
    return Ri

def calc_dimlessIncorporationRate(Ri):
    '''

    :param Ri: Richardson number
    :return: Dimensionless incorporation rate
    '''
    return np.nan_to_num(0.075 / np.sqrt(1 + 718 * Ri ** (2.4)))

def calc_rateOfSeaWaterIncorp(U,Estar):

    return U * Estar

def calc_changeIn_q_th(Ew, pt):
    '''
    :param Ew: Rate of seawater incorporation
    :param pt: CA time step
    :return: Change in turbidity current thickness
    '''
    return np.nan_to_num(Ew * pt)

def calc_new_qcj(q_cj, q_th, new_q_th):
    '''

    :param q_cj: jth sediment concentration (turbidity current)
    :param q_th: Turbidity current thickness
    :param new_q_th: Turbidity current thickness (calculated in T_1)
    :return: New value of jth sediment concentration (turbidity current)
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        var = np.nan_to_num(q_cj * (q_th / new_q_th)[:, :, np.newaxis])
    return var