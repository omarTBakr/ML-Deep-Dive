import matplotlib.pyplot as plt
import numpy as np
def visualize2d(x, y, scatter=True, title='x vs y' ,same_figure = False):
    if not same_figure:
        plt.figure()
    if scatter:
        plt.scatter(x, y, color='r',marker='o')
    else:
        plt.plot(x, y, 'bo-')
    plt.grid()
    plt.title(title)
    if not same_figure:
        plt.show()





def pair_visualization( x1, y1, x2, y2, title =None ):
    plt.figure()
    visualize2d(x1,y1,scatter= True , same_figure=True,  )
    visualize2d(x2, y2 , same_figure=True , scatter=False  )
    plt.show()




def visualize_trails(X,Y , trials):
    plt.figure()
    plt.scatter(X,Y, label  = 'best fit')

    for i, trial in enumerate(trials,1):
        fittted = trial.poly.transform(X)
        plt.plot(X ,trial.model.predict(fittted) ,label = f'degree {i}')

    plt.legend()
    plt.grid()
    plt.show()
