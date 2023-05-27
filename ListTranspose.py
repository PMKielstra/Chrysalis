def list_transpose(L, ax0, ax1):
    if ax0 > 0:
        return [list_transpose(LL, ax0 - 1, ax1 - 1) for LL in L]
    if ax1 == 0:
        return L
    L = [list_transpose(LL, ax0 - 1, ax1 - 1) for LL in L]
    L = [[LL[i] for LL in L] for i in range(len(L[0]))]
    L = [list_transpose(LL, ax0 - 1, ax1 - 1) for LL in L]
    return L

def push_front_to_back(L, dimens):
    if dimens == 0:
        return L
    L = [push_front_to_back([LL[i] for LL in L], dimens - 1) for i in range(len(L[0]))]
    return L

