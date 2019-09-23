class ESTIMATOR:
    PEARSON = 0
    SPEARMAN = 1
    KENDALL = 2
    FREQUENCY = 3


class FEATURE:
    CONTINUOUS = 0
    DISCRETE = 1
    SURVIVAL_EVENT = 2
    SURVIVAL_TIME = 3

class MAP:
    estimator_map = {'pearson'  : ESTIMATOR.PEARSON, 
                     'spearman' : ESTIMATOR.SPEARMAN, 
                     'kendall'  : ESTIMATOR.KENDALL,
                     'frequency': ESTIMATOR.FREQUENCY}
    features_map = {'continuous': FEATURE.CONTINUOUS,
                    'discrete'  : FEATURE.DISCRETE,
                    'event'     : FEATURE.SURVIVAL_EVENT,
                    'time'      : FEATURE.SURVIVAL_TIME}
 

    