r"""
Scheduler
==========

This module implements the :math:`\alpha`-schedulers employed in the conensuse schemes.

"""

import numpy as np
from scipy.special import logsumexp

class scheduler(object):
    r"""scheduler class
    
    This class implements the base scheduler class. It is used to implement the :math:`\alpha`-schedulers
    employed in the consensus schemes.
    
    Parameters
    ----------
    opt : object
        The optimizer for which the :math:`\alpha`-parameter should be updated

    alpha : float, optional
        The initial value of the :math:`\alpha`-parameter. The default is 1.0.

    alpha_max : float, optional
        The maximum value of the :math:`\alpha`-parameter. The default is 100000.0.

    """

    def __init__(self, dyn, var_params):
        self.dyn = dyn
        self.var_params = var_params

    def update(self):
        for var_param in self.var_params:
            var_param.update(self.dyn)


class multiply:
    
    def __init__(self, name = 'alpha',
                 maximum = 1e5, factor = 1.0):
        self.name = name
        self.factor = factor
        self.maximum = maximum
    
    def update(self, dyn):
        r"""Update the :math:`\alpha`-parameter in opt according to the exponential scheduler."""

        old_val = getattr(dyn, self.name)
        new_val = min(self.factor * old_val, self.maximum)
        setattr(dyn, self.name, new_val)
    
    


# # class for alpha_eff scheduler
# class alpha_eff(scheduler_base):
#     r"""alpha_eff scheduler class
    
#     This class implements a scheduler for the :math:`\alpha`-parameter based on the effective number of particles.
#     The :math:`\alpha`-parameter is updated according to the rule
    
#     .. math::
        
#         \alpha_{k+1} = \begin{cases}
#         \alpha_k \cdot r & \text{if } J_{eff} \geq \eta \cdot J \\ 
#         \alpha_k / r & \text{otherwise}
#         \end{cases} 
        
#     where :math:`r`, :math:`\eta` are parameters and :math:`J` is the number of particles. The effictive number of
#     particles is defined as

#     .. math::

#         J_{eff} = \frac{1}{\sum_{i=1}^J w_i^2}
    
#     where :math:`w_i` are the weights of the particles. This was, e.g., employed in [1]_.


    
#     Parameters
#     ----------
#     opt : object
#         The optimizer for which the :math:`\alpha`-parameter should be updated
#     eta : float, optional
#         The parameter :math:`\eta` of the scheduler. The default is 1.0.
#     alpha_max : float, optional
#         The maximum value of the :math:`\alpha`-parameter. The default is 100000.0.
#     factor : float, optional
#         The parameter :math:`r` of the scheduler. The default is 1.05. 

#     References
#     ----------
#     .. [1] Carrillo, J. A., Hoffmann, F., Stuart, A. M., & Vaes, U. (2022). Consensus‐based sampling. Studies in Applied Mathematics, 148(3), 1069-1140. 


#     """
#     def __init__(self, opt, eta=1.0, alpha_max=1e5, factor=1.05):
#         super(alpha_eff, self).__init__(opt, alpha_max = alpha_max)

#         self.eta = eta
#         self.alpha_max = alpha_max
#         self.J_eff = 1.0
#         self.factor=factor
    
#     def update(self,):
#         alpha = self.opt.alpha
        
#         term1 = logsumexp(-alpha*self.opt.V(self.opt.x))
#         term2 = logsumexp(-2*alpha*self.opt.V(self.opt.x))
#         self.J_eff = np.exp(2*term1-term2)
        
#         #w = np.exp(-alpha * self.opt.V(self.opt.x))
#         #self.J_eff = np.sum(w)**2/max(np.linalg.norm(w)**2,1e-7)
        
#         if self.J_eff >= self.eta * self.opt.num_particles:
#             self.opt.alpha = min(alpha*self.factor, self.alpha_max)
#         else:
#             pass
#             #self.opt.alpha /= self.factor