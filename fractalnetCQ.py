import torch
import ezras_fractalnet

def create_fractalnet(nf, paletteSize):
    channels = [int(nf/16), int(nf/8), int(nf/4), int(nf/2)]
    num_columns = 4  #Just because thats the example they used in https://arxiv.org/pdf/1605.07648.pdf
    dropout_probs = [0.5, 0.5, 0.5, 0.5]
    loc_drop_prob = 0.5
    glob_drop_ratio = 0.5

    # For reference these are the parameters they used for CIFAR:
    # channels = [64, 128, 256, 512, 512]
    # num_columns = 3
    # dropout_probs = (0.0, 0.1, 0.2, 0.3, 0.4)
    # loc_drop_prob = 0.15
    # glob_drop_ratio = 0.5


    model = ezras_fractalnet.CIFARFractalNet(channels, num_columns, dropout_probs, loc_drop_prob, glob_drop_ratio, paletteSize)
    return model

if __name__ == "__main__":
    create_fractalnet(16, 64)
