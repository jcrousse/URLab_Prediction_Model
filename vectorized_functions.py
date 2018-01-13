import numba

# simple functions, can easily be vectorised (although time saved is marginal compared to other transformations)
@numba.vectorize
def percentage_open(is_open, minutes_to_event):
    out_val = max(0, 60-minutes_to_event) / 60
    if is_open:
        out_val = min(60, minutes_to_event) / 60
    return out_val

@numba.vectorize
def open_flag_from_pct(open_pct):
    out_val = True
    if open_pct < 0.5:
        out_val = False
    return out_val

@numba.vectorize
def dist_to_h(hour, h):
    return min(abs(hour-h), abs(h-hour))