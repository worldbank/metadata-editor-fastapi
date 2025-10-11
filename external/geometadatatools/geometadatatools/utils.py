import warnings
from functools import wraps


def catch_exceptions_as_warnings(func):
    """
    A decorator that catches exceptions, issues them as warnings with the function name,
    and returns an empty dictionary if an exception occurs.

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn(
                f"Returning empty dict since function '{func.__name__}' raised: {e}", RuntimeWarning, stacklevel=2
            )
            return {}

    return wrapper
