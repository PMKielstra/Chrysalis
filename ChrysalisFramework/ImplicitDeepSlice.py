def deep_slice(L, slices):
    if len(slices) == 0:
        return L
    if slices[0] == None:
        return [deep_slice(LL, slices[1:]) for LL in L]
    return deep_slice(L[slices[0]], slices[1:])

def implicit_deep_slice(L, ts):
    length = max(t[1] for t in ts) + 1
    slices = [None] * length
    for t in ts:
        slices[t[1]] = t[0]
    return deep_slice(L, slices)

ds = deep_slice
ids = implicit_deep_slice
