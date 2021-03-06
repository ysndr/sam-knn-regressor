import numpy as np
import sys
import matplotlib.pyplot as plt

from samknnreg import SAMKNNRegressor
from datagen import NormalBlopps 

if __name__ == "__main__":
    generator = NormalBlopps(dims=int(sys.argv[1]) if len(sys.argv)>1 else 1 )
    generator.prepare_for_use()
    data = [generator.next_sample(1500)]
    X = []
    y = []

    for i in range(len(data)):
        X += data[i][0]
        y += data[i][1]
    X = np.array(X).astype('d')/np.amax(X)
    y = np.array(y).astype('d')/np.amax(y)
    X = np.reshape(X, (X.shape[0], -1))
    model = SAMKNNRegressor(show_plots=True)
    model.fit(X, y)
    #print(model.predict(np.array([[3],[8],[15],[79]])))
    print("LTM size:", len(model.LTMX), "STM size:", len(model.STMX))
    if(X.shape[1] == 1):
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="original", s=100, alpha=.1)
        ax.scatter(model.STMX, model.STMy, label="STM", s=100, alpha=.4)
        ax.scatter(model.LTMX, model.LTMy, label="LTM", s=100, alpha=.4)
        ax.legend()
        plt.show()
    elif(X.shape[1] == 2):
        fig, ax = plt.subplots()
        ax.scatter(np.array(X)[:,0], np.array(X)[:,1], label="original", s=100, alpha=.1, c = y)
        ax.scatter(np.array(model.STMX)[:,0], np.array(model.STMX)[:,1], label="STM", s=100, alpha=.4, marker="x", c = model.STMy)
        ax.scatter(np.array(model.LTMX)[:,0], np.array(model.LTMX)[:,1], label="LTM", s=100, alpha=.4, marker="^", c = model.LTMy)
        ax.legend()
        plt.show()
    model.print_model()
    _ = input("Press [enter] to end.")
