import torch
import torch.nn as nn
from losses import get_cycle_gen_loss,get_disc_loss
from imageDataset import create_Dataloader
from show_tensor_image import show_tensor_images
from CycleGan import Generator,Discriminator



adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()
dim_A = 3
dim_B = 3
lr = 0.0002
target_shape = 256
device = 'cuda'

gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)



gen_AB = gen_AB.apply(weights_init)
gen_BA = gen_BA.apply(weights_init)
disc_A = disc_A.apply(weights_init)
disc_B = disc_B.apply(weights_init)

resize=(256,256)
batchsize=10
shuffle=True

dataloader=create_Dataloader("folder",resize,batchsize,shuffle)

n_epochs=40
display_step=500

def train():
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    cur_step = 0

    for epoch in range(n_epochs):

        for real_A, real_B in dataloader:

            # real_A = nn.functional.interpolate(real_A, size=target_shape)
            # real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)


            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()


            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()


            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_cycle_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            
            gen_loss.backward()
            gen_opt.step()


            mean_discriminator_loss += disc_A_loss.item() / display_step

            mean_generator_loss += gen_loss.item() / display_step


            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0

            cur_step += 1

train()

print("Save Model:")
if(input().lower()=="y"):
    
    torch.save(gen_AB,"Day_to_Night.pth")
    torch.save(gen_BA,"Day_to_Night.pth")
