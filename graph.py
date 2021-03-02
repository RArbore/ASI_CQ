import matplotlib.pyplot as plt
import sys

NUM_TRIALS = 15

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle('color', [cm(1.*i/NUM_TRIALS) for i in range(NUM_TRIALS)])

for t in range(14, 15):
    path = "trial"+str(t+1)+"/during_training_performance.txt"

    f = open(path)
    read = f.read()
    splits = read.split("\n")
    splits.remove("")

    index = []
    value = []
    for s in splits:
        split = s.split(" ")
        index.append(int(split[0]))
        value.append(float(split[int(sys.argv[1])]))

    ax.plot(index, value)#, label=str(t+1))
#plt.legend(loc="upper left")
plt.ylabel("Validation Loss (MSE)")
plt.xlabel("Epoch #")
plt.show()