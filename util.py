# Changes seconds to some nice string
def sec_to_str(secs):
    res = str(int(secs % 60)) + "s"
    mins = int(secs / 60)
    if mins > 0:
        res = str(int(mins % 60)) + "m " + res
        hours = int(mins / 60)
        if hours > 0:
            res = str(int(hours % 60)) + "h " + res
    return res



