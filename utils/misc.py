# *************************************************************
# a decorator used to display the execution time of a function
# *************************************************************
def timed(fn):
    from time import perf_counter
    from functools import wraps

    @wraps(fn)
    def inner(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        elapsed = end - start

        args_ = [str(a) for a in args]
        kwargs_ = ['{0}={1}'.format(k, v) for (k, v) in kwargs.items()]
        all_args = args_ + kwargs_
        args_str = ', '.join(all_args)
        print(f'{fn.__name__}({args_str}) took {elapsed:.6f}s to run.')
        return result

    return inner