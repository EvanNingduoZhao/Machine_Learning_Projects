import numpy as np
import pandas as pd
from apyori import apriori

data = [[2,3,1],
        [2,3,4],
        [2],
        [3],
        [1]
        ]

association_rules = apriori(data,min_support=0.0001,min_confidence=0.2,min_lift=1.1,min_length=2)


for i in association_rules:
    print(i)
    # for j in range(2,len(i)):
    #     print(i[j])
