def clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x
