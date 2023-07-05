def czip(*ls):
    """Zip, but assert that all lists involved have the same length."""
    for l in ls:
        assert len(l) == len(ls[0])
    return zip(*ls)
