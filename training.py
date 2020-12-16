from torchvision import transforms
import random
import torch
import time
import math
import sys
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

DATA_SIZE = -1 #placeholder
BATCH_SIZE = -1 #placeholder
NUM_EPOCHS = -1 #placeholder
NUM_BATCHES = int(DATA_SIZE / BATCH_SIZE)

nf = 16 #number of features: multiplier of channels in CNN

lr = 0.0001
b1 = 0.5
b2 = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu") #can't print from GPU

folder = ""

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layers = torch.nn.sequential( #declaring all of our layers
            #placeholder
        )

    def forward(self, input): #runs the next layer
        return self.layers(input)


def train_model(train_data):
    model = CNN()

    current_milli_time = lambda: int(round(time.time() * 1000))

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    before_time = current_milli_time()

    print("Beginning training")
    print("")

    f = open(folder + "/during_training_performance.txt", "a") #the file where we store the epoch loss

    for epoch in range(0, NUM_EPOCHS):
        os.mkdir(folder + "/epoch"+str(epoch+1))

        epoch_loss = 0
        epoch_before_time = current_milli_time()

        for batch in range(BATCHES_PER_EPOCH):
            opt.zero_grad()
            #placeholder - where we portion the train data into batch data

            output = model(batch_input.to(device))
            loss = #placeholder - loss function
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(BATCHES_PER_EPOCH) #average of all the epoch losses in this patch
            #.item() changes a pytorch tensor to a regular number, only on the CPU, can be expensive

        with torch.no_grad(): #just evaluating it, don't create the graph with .no_grad()
            #placeholder - epoch evaluation

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss)+"\n")

    after_time = current_milli_time() #time of entire process

    torch.save(model.state_dict(), folder + "/model.pt")
    print("")
    f.close()

    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60

    print(str(NUM_EPOCHS) + " epochs took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    return model

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    #sets up the trial folder
    if len(sys.argv) <= 1:
        files = os.listdir(".")
        m = [int(f[5:]) for f in files if len(f) > 5 and f[0:5] == "trial"]
        if len(m) > 0:
            folder = "trial" + str(max(m) + 1)
        else:
            folder = "trial1"
    else: #Otherwise, if we want to set it up manually
        folder = sys.argv[1]

    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")

    train_data = torch.load("TRAIN.pt") #TRAIN.pt is the training data

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(train_data)
