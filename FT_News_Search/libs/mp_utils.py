## multi process function lib 


class MP_joblib():
    """
    abstract base class for Generator that yields info from each doc in a dir
    :param input: File or Dir
    """
    def __init__(self, inputs, mp_func,mp_func_args=None):
        self.input = inputs
        self.mp_func = mp_func
        self.args = mp_func_args

