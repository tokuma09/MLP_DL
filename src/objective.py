# reference: https://www.sfu.ca/~ssurjano/optimization.html
import torch


def sphere(x):
    """sphere function

    Bowl-shaped test function

    global minimum: f(x)=0, at x=(0,..., 0)
    input domain: x_i \in [-5.12, 5.12]

    Parameters
    ----------
    x : [type]
        [description]
    """
    return torch.sum(x**2, axis=0)


def booth(x):
    """booth function

    plate-shaped test function(two-dimentional function)

    global minimum: f(x)=0, x=(1, 3)
    input domain: x_i \in [-10, 10]

    Parameters
    ----------
    x : [type]
        [description]
    """
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2


def rosenbrock(x, a=10, b=1):
    """rosenbrock function

    Vally-Shaped test function
    global minimum: f(x)=0, at x=(1,..., 1)
    input domain: x_i \in [-2.048, 2.048]
    Parameters
    ----------
    x : [type]
        [description]
    a : int, optional
        [description], by default 100
    b : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    return a * (x[1] - x[0]**2)**2 + (b - x[0])**2


def ackley(x, a=20, b=0.2, c=2 * torch.pi):
    """ackley function

    Many Local Minima test function

    global minimum: f(x)=0 at x=(0,...,0)
    input domain: x_i \in [-32.768, 32.768]

    Parameters
    ----------
    x : [type]
        [description]
    a : int, optional
        [description], by default 20
    b : float, optional
        [description], by default 0.2
    c : [type], optional
        [description], by default 2*torch.pi

    Returns
    -------
    [type]
        [description]
    """
    first = -a * torch.exp(-b * torch.sqrt(torch.sum(x**2, axis=0) / len(x)))
    second = -torch.exp(torch.sum(torch.cos(c * x), axis=0) / len(x))
    return first + second + a + torch.e


def easom(x):
    """easom function

    steep drop test function(two dimensional function)

    global minimum: f(x) = -1, at x=(\pi, \pi)
    input domain: x_i \in [-100, 100]

    Parameters
    ----------
    x : [type]
        [description]
    """

    first = -torch.cos(x[0])
    second = torch.cos(x[1])
    third = torch.exp(-(x[0] - torch.pi)**2 - (x[1] - torch.pi)**2)
    return first * second * third
