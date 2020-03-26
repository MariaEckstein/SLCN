import numpy as np
import pandas as pd


def replace_nans(data):

    data[np.isnan(data)] = np.random.binomial(1, 0.5, np.sum(np.isnan(data)))
    return data

# # Example use
# replace_nans(np.full(10, np.nan))


def get_info_from_fullID(fullID, length=1):

    # Gender
    g = int(str(fullID)[-2:-1])
    if g == 1:
        gender = 'Male'
    elif g == 2:
        gender = 'Female'
    else:
        raise ValueError('Invalid fullID at gender.')

    # Agegroup
    a = int(str(fullID)[-1:])
    if a == 1:
        agegroup = 'Adult'
    elif a == 2:
        agegroup = 'Juvenile'
    else:
        raise ValueError('Invalid fullID at agegroup.')

    # Put everything together
    return {
        'animal': [int(str(fullID)[:-5])] * length,
        'age': [int(str(fullID)[-5:-2])] * length,
        'gender': [gender] * length,
        'agegroup': [agegroup] * length
    }

# # Example use
# get_info_from_fullID(fullIDs[0], length=5)


def add_meta_column(dat):
    dat['meta'] = np.nan
    dat.loc[dat.session <= 2, 'meta'] = '0-2'
    dat.loc[(dat.session >= 3) * (dat.session <= 9), 'meta'] = '3-9'
    dat.loc[(dat.session >= 10) * (dat.session <= 12), 'meta'] = '10-12'
    dat = dat.dropna(subset=['meta'])
    dat['meta'] = pd.Categorical(dat['meta'], ordered=True, categories=('0-2', '3-9', '10-12'))

    return dat

# # Example use
# add_meta_column(true_dat)