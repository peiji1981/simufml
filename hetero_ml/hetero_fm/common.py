
from typing import Dict

import numpy as np
from numpy import array

def base_fm_forward(batch_x: array, model_params: Dict[str, array]) -> array:
        """ 
        Calculate w*x + (v*x)^2 - (v^2)*(x^2)
        
        Arguments:    
            batch_x: 
                shape [batch_size * feature_size]
            model_params: 
                FM Model parameters. A dict contains:
                - 'w': [feature_size,], 
                - 'embed': [feature_size, embed_size]
                - 'b': [1,]

        Return:
            pred: [batch_size, ]
        """

        def fm_func(batch_x, embed):
            '''
            batch_x: [batch_size, feature_size]
            embed: [feature_size, embed_size]
            '''
            vx = np.multiply(batch_x[...,None], embed[None,]) # shape: [batch_size, feature_size, embed_size]
            square_of_sum = np.sum(vx, 1) ** 2 # shape: [batch_size, embed_size]
            sum_of_square = np.sum(vx**2, 1) # shape: [batch_size, embed_size]
            reduce = np.sum(square_of_sum - sum_of_square, 1) # shape: [batch_size, ]
            return 0.5 * reduce

        w = model_params['w']
        embed = model_params['embed']
        b = model_params.get('b', 0.)

        wx = np.dot(batch_x, w[:,None]).squeeze() # shape: [batch_size, ]
        fm = fm_func(batch_x, embed)
        pred = wx + fm + b

        return pred


def base_fm_vx(batch_x: array, embed: array) -> array:
    """
    Calculate vx = v*x
    
    Arguments:
        batch_x: 
            [batch_size, feature_size]
        embed: 
            [feature_size, embed_size]
    
    Return:
        vx: [batch_size, embed_size]
    """
    return np.dot(batch_x, embed)


def compute_local_grad(
    batch_x, 
    grad_z, 
    en_aggvx,
    embed
):
    # gradient on w
    grad_w = np.multiply(batch_x, grad_z[:,None]).mean(0) # [feature_size, ]

    # gradient on embed
    aggvx_mul_x = np.multiply(batch_x[:,:,None], en_aggvx[:,None,:]) # [batch_size, feature_size, embed_size]
    v_mul_x2 = np.multiply(embed[None,:,:], (batch_x**2)[:,:,None])
    en_grad_v = np.multiply((aggvx_mul_x-v_mul_x2), grad_z[:,None,None]).mean(0) # [feature_size, embed_size]

    # gradient on bias
    grad_b = np.array([grad_z.mean()]) # [1, ]

    return grad_w, en_grad_v, grad_b