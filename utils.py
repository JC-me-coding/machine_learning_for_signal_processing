import scipy.io as io
from scipy import signal

def loadmatlabfile(file):
    #file = "data/problem2_5_noise.mat"
    matlab_file = io.loadmat(file)
    list_of_arrays = [x for x in matlab_file.keys() if x not in ['__header__', '__version__', '__globals__']]
    return dict((array, matlab_file[array]) for array in list_of_arrays)

# perform biased cross correlation using library function

def xcorr(x, y, k):
    N = min(len(x),len(y))
    r_xy = (1/N) * signal.correlate(x,y,'full') # reference implementation is unscaled
    return r_xy[N-k-1:N+k]