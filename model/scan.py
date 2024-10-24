from einops import rearrange


def s_hw(x):
    """
    1 2 3
    4 5 6 -> 1 2 3 4 5 6 7 8 9
    7 8 9
    """
    return rearrange(x, "b c h w -> b (h w) c")


def sr_hw(x, h, w):
    """
                         1 2 3
    1 2 3 4 5 6 7 8 9 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def s_wh(x):
    """
    1 2 3
    4 5 6 -> 1 4 7 2 5 8 3 6 9
    7 8 9
    """
    return rearrange(x, "b c h w -> b (w h) c")


def sr_wh(x, h, w):
    """
                         1 2 3
    1 4 7 2 5 8 3 6 9 -> 2 5 8
                         3 6 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h)


def s_rhw(x):
    """
    1 2 3
    4 5 6 -> 7 8 9 4 5 6 1 2 3
    7 8 9
    """
    return rearrange(x.flip(2), "b c h w -> b (h w) c")


def sr_rhw(x, h, w):
    """
                         1 2 3
    7 8 9 4 5 6 1 2 3 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).flip(2)


def s_hrw(x):
    """
    1 2 3
    4 5 6 -> 3 2 1 6 5 4 9 8 7
    7 8 9
    """
    return rearrange(x.flip(3), "b c h w -> b (h w) c")


def sr_hrw(x, h, w):
    """
                         1 2 3
    3 2 1 6 5 4 9 8 7 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).flip(3)


def s_rwh(x):
    """
    1 2 3
    4 5 6 -> 3 6 9 2 5 8 1 4 7
    7 8 9
    """
    return rearrange(x.flip(3), "b c h w -> b (w h) c")


def sr_rwh(x, h, w):
    """
                         1 2 3
    3 6 9 2 5 8 1 4 7 -> 2 5 8
                         3 6 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h).flip(3)


def s_wrh(x):
    """
    1 2 3
    4 5 6 -> 7 4 1 8 5 2 9 6 3
    7 8 9
    """
    return rearrange(x.flip(2), "b c h w -> b (w h) c")


def sr_wrh(x, h, w):
    """
                         1 2 3
    7 4 1 8 5 2 9 6 3 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h).flip(2)


from einops import rearrange


def s_hw(x):
    """
    1 2 3
    4 5 6 -> 1 2 3 4 5 6 7 8 9
    7 8 9
    """
    return rearrange(x, "b c h w -> b (h w) c")


def sr_hw(x, h, w):
    """
                         1 2 3
    1 2 3 4 5 6 7 8 9 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def s_wh(x):
    """
    1 2 3
    4 5 6 -> 1 4 7 2 5 8 3 6 9
    7 8 9
    """
    return rearrange(x, "b c h w -> b (w h) c")


def sr_wh(x, h, w):
    """
                         1 2 3
    1 4 7 2 5 8 3 6 9 -> 2 5 8
                         3 6 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h)


def s_rhw(x):
    """
    1 2 3
    4 5 6 -> 7 8 9 4 5 6 1 2 3
    7 8 9
    """
    return rearrange(x.flip(2), "b c h w -> b (h w) c")


def sr_rhw(x, h, w):
    """
                         1 2 3
    7 8 9 4 5 6 1 2 3 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).flip(2)


def s_hrw(x):
    """
    1 2 3
    4 5 6 -> 3 2 1 6 5 4 9 8 7
    7 8 9
    """
    return rearrange(x.flip(3), "b c h w -> b (h w) c")


def sr_hrw(x, h, w):
    """
                         1 2 3
    3 2 1 6 5 4 9 8 7 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).flip(3)


def s_rwh(x):
    """
    1 2 3
    4 5 6 -> 3 6 9 2 5 8 1 4 7
    7 8 9
    """
    return rearrange(x.flip(3), "b c h w -> b (w h) c")


def sr_rwh(x, h, w):
    """
                         1 2 3
    3 6 9 2 5 8 1 4 7 -> 2 5 8
                         3 6 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h).flip(3)


def s_wrh(x):
    """
    1 2 3
    4 5 6 -> 7 4 1 8 5 2 9 6 3
    7 8 9
    """
    return rearrange(x.flip(2), "b c h w -> b (w h) c")


def sr_wrh(x, h, w):
    """
                         1 2 3
    7 4 1 8 5 2 9 6 3 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h).flip(2)


def s_rhrw(x):
    """
    1 2 3
    4 5 6 -> 9 8 7 6 5 4 3 2 1
    7 8 9
    """
    return rearrange(x.flip(2).flip(3), "b c h w -> b (h w) c")


def sr_rhrw(x, h, w):
    """
                         1 2 3
    9 8 7 6 5 4 3 2 1 -> 4 5 6
                         7 8 9
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).flip(2).flip(3)


def s_rwrh(x):
    """
    1 2 3
    4 5 6 -> 9 6 3 8 5 2 7 4 1
    7 8 9
    """
    return rearrange(x.flip(2).flip(3), "b c h w -> b (w h) c")


def sr_rwrh(x, h, w):
    """
                         1 2 3
    9 6 3 8 5 2 7 4 1 -> 2 5 8
                         3 6 9
    """
    return rearrange(x, "b (w h) c -> b c h w", w=w, h=h).flip(3).flip(2)
