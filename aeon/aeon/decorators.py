"""Misc function/class decorators."""
from typing import Callable


def tab_completion(func: Callable):
    """Create class attributes from the results of some function. Helpful for creating minimal
    classes that mostly exist to provide tab completion.

    Arguments
    ---------
    func : callable
        A function that returns iterable[str].

    Examples
    --------
    @tab_comletion(get_pet_names)
    class Pets:
        ...
    """
    values = func()
    def decorator(cls):
        for val in values:
            setattr(cls, val.upper(), val)
        return cls
    return decorator
