from exploror import exploror
from feedbacker import feedbacker

specs = [2,3]
sizes = [8, 10]
# filename = '/home/fei/bitbucket/QBF/QBFinstances/train10/6_unsat.qdimacs'
filename = '/homes/wang603/QBF/QBFinstances/train10/6_unsat.qdimacs'
sample_size = 40
exp = exploror(sizes[0])
fb = feedbacker(filename, specs, sizes)
exp.set_state([])
while True:
    trials = exp.sample(sample_size)
    fb.evaluate(trials)
    # print(fb.last)
    if fb.unsat:
        print('witness of unsat is {}'.format(fb.witness))
        exit(0)
    else:
        seeds = fb.recommend_seed(exp.seed_size)
        exp.set_state(seeds)
