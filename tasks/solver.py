from moabb.datasets import Shin2017A, Shin2017B
import multiprocessing as mp
from utils import plot_scores


def apply_algorithm(func, config):
    use_concurrent = config['concurrent']

    subjects = config['subjects']

    dataset = Shin2017B() if config['is_ma'] else Shin2017A()

    args = list(map(lambda x: [x, config, dataset], subjects))

    def concurrent():
        pool = mp.Pool(len(args))

        all_scores = pool.starmap(func, args)

        pool.close()
        pool.join()
        return all_scores

    def single_process():
        return [func(*x) for x in args]

    scores = concurrent() if use_concurrent else single_process()

    # print('single threaded: ', timeit.timeit(fun2, number=3))
    # print('multi process: ', timeit.timeit(fun1, number=3))

    plot_scores(scores, config)