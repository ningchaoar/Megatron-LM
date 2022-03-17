import torch
import numpy as np
# pip install transformers -i https://pypi.python.org/simple
from transformers import GPT2Config, GPT2LMHeadModel
from deepspeed.profiling.flops_profiler import FlopsProfiler


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long).to("cuda")
        return input_ids

    def __len__(self):
        return len(self.input_list)

print("Model Initializing")
model_config = GPT2Config.from_json_file("flops/config_medium.json")
model = GPT2LMHeadModel(model_config)
model.train().to("cuda")
prof = FlopsProfiler(model)

regularized_params = []
non_regularized_params = []
for param in model.parameters():
    if param.requires_grad:
        if len(param.shape) == 1:
            non_regularized_params.append(param)
        else:
            regularized_params.append(param)
params = [
    {"params": regularized_params, "weight_decay": 0.01},
    {"params": non_regularized_params, "weight_decay": 0}
]
optimizer = torch.optim.AdamW(params)

max_len = 1024
generated = np.random.randint(low=1, high=50256, size=(128, max_len + 1))
train_dataset = MyDataset(generated, max_len + 1)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
print("DataLoader Initializing")

profile_step = 2
print_profile= True
print("Start Training")
for step, batch in enumerate(data_loader):
    print("Step: {}".format(step))
    input_ids = batch[:, :-1]
    labels = batch[:, 1:]
    # start profiling at training step "profile_step"
    if step == profile_step:
        prof.start_profile()
    # forward() method
    loss = model(input_ids=input_ids, labels=labels)[0]
    # end profiling and print output
    if step == profile_step: # if using multi nodes, check global_rank == 0 as well
        print("Start Profiling")
        prof.stop_profile()
        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        if print_profile:
            prof.print_model_profile(profile_step=profile_step)
        prof.end_profile()
    # runs backpropagation
    loss.backward()
    # weight update
    optimizer.step()
