import numpy as np


def replace_nans(data):

    data[np.isnan(data)] = np.random.binomial(1, 0.5, np.sum(np.isnan(data)))
    return data

# # Example use
# replace_nans(np.full(10, np.nan))

def get_info_from_fullID(fullID, length=1):
    return {
        'animal': [int(str(fullID)[:-5])] * length,
        'age': [int(str(fullID)[-5:-2])] * length,
        'gender': [int(str(fullID)[-2:-1])] * length,
        'agegroup': [int(str(fullID)[-1:])] * length
    }

# # Example use
# get_info_from_fullID(fullIDs[0], length=5)