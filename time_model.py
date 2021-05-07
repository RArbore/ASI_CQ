#!/usr/bin/env python3

from training import CNN
import torch, time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

model = CNN().to(device)

data = torch.load("TRAIN_aquarium.pt")

current_milli_time = lambda: int(round(time.time() * 1000))

print("Starting")
before_time = current_milli_time()
for i in range(50):
    model(data[i:i+1].to(device))
after_time = current_milli_time()
print((after_time-before_time)/50, "milliseconds")
