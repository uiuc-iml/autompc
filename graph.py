import matplotlib.pyplot as plt
import pickle
import numpy as np

infile = open('dadIter0.data','rb')
inputData = pickle.load(infile)
x = inputData[5]
testTraj = inputData[9]

xAxisTimesteps = [0]
for t in range(1, testTraj.obs.shape[0]):
    xAxisTimesteps.append(t)


plt.plot(xAxisTimesteps, testTraj.obs[:,2].tolist(), label = "X Observation")

models = [0,1,2,3,4,5]
#models = np.arange(0,26,8).tolist()

for n in models:
    infile = open('dadIter' + str(n) + '.data','rb')
    inputData = pickle.load(infile)
    x = inputData[7]

    plt.plot(xAxisTimesteps, x.tolist(), label = "X Prediction:  "+ str(n))

#plt.plot(xAxisTimesteps, x.tolist(), label = "X Prediction: 25")

plt.legend()
#plt.xlim([35, 40])
#plt.ylim([8, 10])
plt.yscale('linear')
plt.savefig('trajXIndividual.png', dpi=600, bbox_inches='tight')
plt.clf()

for i in range(6):
    infile = open('dadIter'+ str(i) + '.data','rb')
    inputData = pickle.load(infile)
    print(inputData[3].shape)

model = 1
infile = open('dadIter'+ str(model) + '.data','rb')
inputData = pickle.load(infile)
dYData = inputData[4]

start = 0

if(model > 0):
    infile = open('dadIter'+ str(model - 1) + '.data','rb')
    inputData = pickle.load(infile)
    start = inputData[4].shape[0]

plt.hist(dYData[start:,2], bins=100, alpha=0.5, label="Model " + str(model))

plt.title('Model '+ str(model) + ' dY distribution X')
plt.legend()
ax = plt.gca()  # get the current axes
ax.relim()      # make sure all the data fits
ax.autoscale()
plt.savefig('dYXDistributionIndividual.png', dpi=300, bbox_inches='tight')
plt.clf()






# Useful methods