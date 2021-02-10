from torchvision import transforms
import random
import resnet
import torch
import time
import math
import sys
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

NUM_AUGS = 0
DATA_SIZE = 15000 * (NUM_AUGS + 1)
VALID_DATA_SIZE = 100
BATCH_SIZE = 10
NUM_EPOCHS = 100
NUM_BATCHES = int(DATA_SIZE / BATCH_SIZE)

nf = 16 #number of features: multiplier of channels in CNN

palette = 64

lr = 0.0001
b1 = 0.5
b2 = 0.999

saved_images_per_epoch = 10

distance_mag = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu") #can't print from GPU

folder = ""

# test_label = torch.rand(1, 32*3).to(device)

class CNN(torch.nn.Module):

    def __init__(self): #Input: (B, 3, 256, 256), Output: (B, palette * 3)
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential( #declaring all of our conv layers, Input: (B, 3, 256, 256), Output: (B, nf*8, 1, 1)
            torch.nn.Conv2d(3, nf, 3, 1, 1),
            # torch.nn.Dropout2d(0.5),
            # torch.nn.BatchNorm2d(nf),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(nf, nf * 2, 3, 1, 1),
            # torch.nn.Dropout2d(0.5),
            # torch.nn.BatchNorm2d(nf * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            # torch.nn.Dropout2d(0.5),
            # torch.nn.BatchNorm2d(nf * 4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            # torch.nn.Dropout2d(0.5),
            # torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool2d(4),
        )
        self.linear = torch.nn.Sequential( #declaring all of our linear layers, Input: (B, nf*8), Output: (B, palette * 3)
            torch.nn.Linear(nf * 8, nf * 4),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(nf * 4, palette * 3),
            torch.nn.Tanh(),
        )

    #     self.seq1 = torch.nn.Sequential(
    #         torch.nn.Conv2d(3, nf, 3, 2, 0),
    #         torch.nn.ReLU(True),
    #         torch.nn.Conv2d(nf, nf, 3, 1, 0),
    #         torch.nn.ReLU(True),
    #         torch.nn.Conv2d(nf, nf * 2, 3, 1, 1),
    #         torch.nn.ReLU(True),
    #         torch.nn.MaxPool2d(3, 2),
    #         torch.nn.Conv2d(nf * 2, nf * 2, 3, 1, 0),
    #         torch.nn.ReLU(True),
    #         torch.nn.Conv2d(nf * 2, nf * 2, 3, 2, 0),
    #         torch.nn.ReLU(True),
    #         torch.nn.Conv2d(nf * 2, nf * 4, 3, 1, 0),
    #         torch.nn.ReLU(True),
    #         torch.nn.MaxPool2d(8),
    #     )
    #     self.seq2 = torch.nn.Sequential(
    #         torch.nn.Linear(576, palette * 3),
    #         torch.nn.Tanh(),
    #     )

    # def forward(self, input): #runs input through layers of model, returns output
    #     output = (self.seq2(self.seq1(input).view(input.size(0), -1))+1)/2
    #     return output

    def forward(self, input): #runs input through layers of model, returns output
        conv_output = self.conv(input)
        linear_output = (self.linear(conv_output.view(input.size(0), -1))+1)/2
        return linear_output

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

def construct_quantized_images(output, batch):
    B = batch.size(0)

    output_v = output.view(B, palette, 3).to(device)

    # distance = torch.zeros(B, palette, 256, 256).to(device)
    distance = torch.sum((batch.view(B, 1, 3, 256, 256).repeat(1, palette, 1, 1, 1) - output_v[:, :, :].view(B, palette, 3, 1, 1).repeat(1, 1, 1, 256, 256))**2, dim=2)
    # for p in range(palette):
    #     distance[:, p, :, :] = torch.sum((batch - output_v[:, p, :].view(B, 3, 1, 1).repeat(1, 1, 256, 256))**2, dim=1)
    distance = torch.argmin(distance, dim=1)
    selection_one_hot = torch.nn.functional.one_hot(distance, num_classes=palette).permute(0, 3, 1, 2).view(B, palette, 1, 256, 256).repeat((1, 1, 3, 1, 1))
    quantized_batch = output_v.view(B, palette, 3, 1, 1).repeat(1, 1, 1, 256, 256)
    quantized_batch = torch.sum(quantized_batch * selection_one_hot, dim=1)
    # if torch.sum(torch.isnan(quantized_batch)) > 0:
    #     print("distance: ", distance, torch.mean(distance), torch.sum(torch.isnan(distance)))
    #     print("w: ", w, torch.mean(w), torch.sum(torch.isnan(w)))
    #     print(1/0)
    return quantized_batch

def construct_soft_quantized_images(output, batch): #output: (B, palette * 3), batch: (B, 3, 256, 256)
    # output_v = output.view(BATCH_SIZE, palette, 3, 1, 1)
    # output_v = output_v.repeat(1, 1, 1, 256, 256) #(B, palette, 3, 256, 256)
    # batch_v = batch.view(BATCH_SIZE, 1, 3, 256, 256)
    # batch_v = batch_v.repeat(1, palette, 1, 1, 1) #(B, palette, 3, 256, 256)
    # difference = output_v - batch_v #(B, palette, 3, 256, 256)
    # distance = difference[:, :, 0, :, :]**2 + difference[:, :, 1, :, :]**2 + difference[:, :, 2, :, :]**2 #(B, palette, 256, 256)
    # mapping = torch.argmin(distance, dim=1).view(BATCH_SIZE, 1, 1, 256, 256) #(B, 1, 1, 256, 256)
    # mapping = mapping.repeat(1, 1, 3, 1, 1) #(B, 1, 3, 256, 256)
    # quantized_batch = torch.gather(batch_v, 1, mapping) #(B, 1, 3, 256, 256)
    # return quantized_batch.view(BATCH_SIZE, 3, 256, 256)

    B = batch.size(0)

    output_v = output.view(B, palette, 3).to(device)
    # quantized_batch = torch.zeros(B, 3, 256, 256).to(device)

    # distance = torch.zeros(B, palette, 256, 256).to(device)
    # for p in range(palette):
    #     distance[:, p, :, :] = torch.exp(-torch.sum((batch - output_v[:, p, :].view(B, 3, 1, 1).repeat(1, 1, 256, 256))**2, dim=1)*distance_mag)
    distance = torch.exp(-torch.sum((batch.view(B, 1, 3, 256, 256).repeat(1, palette, 1, 1, 1) - output_v[:, :, :].view(B, palette, 3, 1, 1).repeat(1, 1, 1, 256, 256))**2, dim=2)*distance_mag)
    w = torch.sum(distance, dim=1) #(B, 256, 256)
    quantized_batch = torch.sum((distance[:, :, :, :]/(w.view(B, 1, 256, 256).repeat(1, palette, 1, 1))).view(B, palette, 1, 256, 256).repeat((1, 1, 3, 1, 1)) * output_v[:, :, :].view(B, palette, 3, 1, 1).repeat(1, 1, 1, 256, 256), dim=1)
    # for p in range(palette):
    #     quantized_batch[:, :, :, :] += (distance[:, p, :, :]/w).view(B, 1, 256, 256).repeat((1, 3, 1, 1)) * output_v[:, p, :].view(B, 3, 1, 1).repeat(1, 1, 256, 256)
    if torch.sum(torch.isnan(quantized_batch)) > 0:
        # print("distance: ", distance, torch.mean(distance), torch.sum(torch.isnan(distance)))
        # print("w: ", w, torch.mean(w), torch.sum(torch.isnan(w)), (w == 0).nonzero())
        # print("quantized batch: ", quantized_batch, torch.mean(quantized_batch), torch.sum(torch.isnan(quantized_batch)), (torch.isnan(quantized_batch) == True).nonzero())
        # print(distance[4, :, 120, 143])
        # print(distance[4, :, 120, 255])
        # print(batch[4, :, 120, 143], output[4, :].view(1, palette, 3))
        # print(torch.min(batch), torch.max(batch))
        print(1/0)
    # print(torch.min(batch), torch.max(batch))
    return quantized_batch
            

def SSELoss(output, batch):
    quantized_batch = construct_soft_quantized_images(output, batch)
    return torch.mean((quantized_batch - batch)**2)

def train_model(train_data):
    model = CNN()
    # model = resnet.resnet(3, palette * 3)

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
        valid_loss = 0
        epoch_before_time = current_milli_time()

        for batch in range(NUM_BATCHES):
            opt.zero_grad()
            model = model.train()
            batch_input = train_data[(batch * BATCH_SIZE):((batch + 1) * BATCH_SIZE)].to(device)

            output = model(batch_input*2-1)
            loss = SSELoss(output, batch_input)
            loss.backward()
            opt.step()
            epoch_loss += loss.to(cpu).item() / float(NUM_BATCHES) #average of all the epoch losses in this patch
            #.item() changes a pytorch tensor to a regular number, only on the CPU, can be expensive
            if (math.isnan(epoch_loss)):
                print("NaN!")

        with torch.no_grad(): #just evaluating it, don't create the graph with .no_grad()
            model = model.eval()
            batch_input = train_data[DATA_SIZE:DATA_SIZE+saved_images_per_epoch].to(device)
            output = model(batch_input)
            quantized_batch = construct_quantized_images(output, batch_input)
            batch_input = batch_input.to(cpu)
            quantized_batch = quantized_batch.to(cpu)
            for i in range(saved_images_per_epoch):
                save_image(batch_input[i], folder + "/epoch"+str(epoch+1) + "/original_image_"+str(i)+".png")
                save_image(quantized_batch[i], folder + "/epoch"+str(epoch+1) + "/quantized_image_"+str(i)+".png")

            for valid_batch in range(int(VALID_DATA_SIZE/BATCH_SIZE)):
                batch_input = train_data[DATA_SIZE + (valid_batch * BATCH_SIZE):DATA_SIZE + ((valid_batch + 1) * BATCH_SIZE)].to(device)
                output = model(batch_input)
                quantized_batch = construct_quantized_images(output, batch_input)
                valid_loss += torch.mean((quantized_batch - batch_input)**2).to(cpu).item() / float(VALID_DATA_SIZE/BATCH_SIZE)       

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60

        print("["+str(epoch + 1)+"]   Loss : "+str(epoch_loss)+"   Validation Loss : "+str(valid_loss)+"   Took "+str(minutes)+" minute(s) and "+str(seconds)+" second(s).")

        f.write(str(epoch + 1)+" "+str(epoch_loss)+" "+str(valid_loss)+"\n")

    after_time = current_milli_time() #time of entire process

    torch.save(model.to(cpu).state_dict(), folder + "/model.pt")
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

    train_data = torch.clamp(torch.load("TRAIN_aquarium.pt"), 0, 1) #(15100, 3, 256, 256)
    # train_data = torch.cat((torch.load("TRAIN_aquarium.pt"), torch.load("TRAIN_badlands.pt"), torch.load("TRAIN_baseball_field.pt")), dim=0) #TRAIN_x.pt is the data, size: (45000, 3, 256, 256)

    after_time = current_milli_time()
    seconds = math.floor((after_time - before_time) / 1000)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

    model = train_model(train_data)
