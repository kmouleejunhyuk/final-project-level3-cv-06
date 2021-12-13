import timeit

def timer(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] inference time : {timeit.default_timer()-start}")
        return result
    return wrapper