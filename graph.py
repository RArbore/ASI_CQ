import matplotlib.pyplot as plt
import sys

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)

path =sys.argv[1]+"/during_training_performance.txt"

f = open(path)
read = f.read()
splits = read.split("\n")
splits.remove("")

index = []
value = []
for s in splits:
    split = s.split(" ")
    index.append(int(split[0]))
    value.append(float(split[int(sys.argv[2])]))

ax.plot(index, value)#, label=str(t+1))
#plt.legend(loc="upper left")
if sys.argv[2] == "0":
    plt.ylabel("Epoch #")
elif sys.argv[2] == "1":
    plt.ylabel("Training Loss (MSE)")
elif sys.argv[2] == "2":
    plt.ylabel("Validation Loss (MSE)")
plt.xlabel("Epoch #")
plt.savefig("Figure_"+sys.argv[1].split("_")[0]+"_"+sys.argv[1][-1]+".png")
