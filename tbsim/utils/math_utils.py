import torch
import numpy as np

def soft_min(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((y-x)/2)
    exp2 = expfun((x-y)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)

def soft_max(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((x-y)/2)
    exp2 = expfun((y-x)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)

def soft_sat(x,x_min=None,x_max=None,gamma=5):
    if x_min is None and x_max is None:
        return x
    elif x_min is None and x_max is not None:
        return soft_min(x,x_max,gamma)
    elif x_min is not None and x_max is None:
        return soft_max(x,x_min,gamma)
    else:
        if isinstance(x_min,torch.Tensor) or isinstance(x_min,np.ndarray):
            assert (x_max>x_min).all()
        else:
            assert x_max>x_min
        xc = x - (x_min+x_max)/2
        if isinstance(x,torch.Tensor):
            return xc/(torch.pow(1+torch.pow(torch.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        elif isinstance(x,np.ndarray):
            return xc/(np.power(1+np.power(np.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        else:
             raise Exception("data type not supported")

