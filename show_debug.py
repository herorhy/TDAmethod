from matplotlib import pyplot as plt

import test_load
filename = "hs300cfg.csv"
label = [0,0,0,0,0,0,4,3,0,3,2,1,0,3,1,3,0,0,3,3,0,2,1, 4, 0 ,1, 3, 0, 0 ,0 ,1 ,4 ,0 ,4 ,0 ,0 ,3,
 0,0,3,4,0,1 ,3, 2, 0, 3 ,2, 1 ,0 ,0 ,1, 0, 3, 1 ,3 ,0 ,2 ,0 ,0 ,0 ,2 ,0 ,2 ,0 ,0 ,3 ,3 ,0 ,0 ,0 ,0 ,1 ,0
    , 0,2,3,0,0, 3, 2, 3, 1, 3, 1, 1 ,0, 3, 0, 3, 1, 4, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3 ,3 ,0 ,0 ,3 ,0 ,1 ,1 ,0, 3,
 0,1]

for i in range(1,288):
    nameOfcurve, originalSignal = test_load.loadcsv(filename, i, 1)
    print(nameOfcurve)
    plt.show()

"""
num = 0
for i in label:
    num += 1
    if i == 0:
        nameOfcurve, originalSignal = test_load.loadcsv(filename, num, 1)
        print(nameOfcurve)
        plt.show()
"""



