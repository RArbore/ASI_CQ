import torch
import tqdm

data_names = [
    "TRAIN_abbey.pt",
    "TRAIN_airport_terminal.pt",
    "TRAIN_alley.pt",
    "TRAIN_amphitheater.pt",
    "TRAIN_amusement_park.pt",
    "TRAIN_aquarium.pt",
    "TRAIN_aqueduct.pt",
    "TRAIN_arch.pt",
    "TRAIN_art_gallery.pt",
    "TRAIN_art_studio.pt",
    "TRAIN_assembly_line.pt",
    "TRAIN_attic.pt",
    "TRAIN_auditorium.pt",
]

i = 7
train_data = None

for name in tqdm.tqdm(data_names):
    if train_data is None:
        train_data = torch.clamp(torch.load(name), 0, 1)[0:1161]
    else:
        train_data = torch.cat((train_data, torch.clamp(torch.load(name), 0, 1)[0:(1162 if i > 0 else 1161)]))
        i -= 1

print("Loaded data with size:", train_data.size())
torch.save(train_data, "TRAIN_DATA.pt")
