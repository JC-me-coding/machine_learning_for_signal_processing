import scipy.io as io

def loadmatlabfile(file):
    #file = "data/problem2_5_noise.mat"
    matlab_file = io.loadmat(file)
    list_of_arrays = [x for x in matlab_file.keys() if x not in ['__header__', '__version__', '__globals__']]
    return dict((array, matlab_file[array]) for array in list_of_arrays)

loadmatlabfile("data/problem2_5_noise.mat")
