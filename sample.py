#%%
from skmultiflow.evaluation import EvaluatePrequential
import matplotlib.pyplot as plt

# from skmultiflow.data import ConceptDriftStream, RegressionGenerator, SineGenerator, STAGGERGenerator

from datagen import JumpingSineGenerator, StairsGenerator

#%%
# 1. Create a stream
stream = JumpingSineGenerator(granularity=10)
stream.prepare_for_use()

X, y = stream.next_sample(batch_size=1000)


stairs = StairsGenerator()
stairs.prepare_for_use()

X_st, y_st = stairs.next_sample(batch_size=1000)

#%%
plt.scatter(X,y,s=1)
plt.scatter(X_st, y_st)
plt.show()





#%%
