# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Main entry point for training AttGAN network."""

import argparse
import datetime
import json
import os
from os.path import join

import torch.utils.data as data

import torch
import torchvision.utils as vutils
from attr2font import DiscriminatorWithClassifier, GeneratorStyle
from data_with_attgan import check_attribute_conflict, CelebA, CelebA_HQ
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter

from vgg_cx import VGG19_CX
import torch.nn as nn
from loss import CXLoss
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../../data/celeba/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='../../data/celeba/list_attr_celeba.txt')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=1, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    parser.add_argument("--attr_embed", type=int, default=64,
                        help="attribute embedding channel, attribute id to attr_embed, must same as image size")
    parser.add_argument("--n_res_blocks", type=int, default=16, help="number of residual blocks in style encoder")
    parser.add_argument("--attention", type=bool, default=True, help="whether use the self attention layer in the generator")
    parser.add_argument("--style_out_channel", type=int, default=128, help="number of style embedding channel")
    parser.add_argument("--n_style", type=int, default=4, help="number of style input images")
    parser.add_argument("--dis_pred", type=bool, default=True, help="whether the discriminator predict the attributes")
    parser.add_argument("--init_epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--log_freq", type=int, default=100, help="frequency of sample training batch")
    # Lambda
    parser.add_argument("--lambda_cx", type=float, default=6.0, help='Contextual loss lambda')
    parser.add_argument("--lambda_attr", type=float, default=20.0, help='discriminator predict attribute loss lambda')
    parser.add_argument("--lambda_GAN", type=float, default=5.0, help='GAN loss lambda')
    parser.add_argument("--lambda_l1", type=float, default=50.0, help='pixel l1 loss lambda')
    
    return parser.parse_args(args)

args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CelebA':
    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs, args.n_style)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs, args.n_style)
if args.data == 'CelebA-HQ':
    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

#### Including vgg network
# CX Loss
if args.lambda_cx > 0:
    criterion_cx = CXLoss(sigma=0.5).to(device)
    vgg19 = VGG19_CX().to(device)
    vgg19.load_model('vgg19-dcbb9e9d.pth')
    vgg19.eval()
    vgg_layers = ['conv3_3', 'conv4_2']

generator = GeneratorStyle(n_style=args.n_style, attr_channel=args.n_attrs,
                               style_out_channel=args.style_out_channel,
                               n_res_blocks=args.n_res_blocks,
                               attention=args.attention)
discriminator = DiscriminatorWithClassifier()
# Attrbute embedding
# attribute: N x 37 -> N x 37 x 64
attribute_embed = nn.Embedding(args.n_attrs, args.attr_embed)

if args.multi_gpu:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    attribute_embed = nn.DataParallel(attribute_embed)
    #attr_unsuper_tolearn = nn.DataParallel(attr_unsuper_tolearn)
generator = generator.to(device)
discriminator = discriminator.to(device)
attribute_embed = attribute_embed.to(device)

# Discriminator output patch shape
patch = (1, args.img_size // 2**4, args.img_size // 2**4)

# optimizers
optimizer_G = torch.optim.Adam([
    {'params': generator.parameters()},
    {'params': attribute_embed.parameters(), 'lr': 1e-3}],
    lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

fixed_img_a, fixed_style_a, fixed_att_a, fixed_style_b, fixed_att_b = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float)

# for i in range(args.n_attrs):
#     tmp = fixed_att_a.clone()
#     tmp[:, i] = 1 - tmp[:, i]
#     tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
#     sample_att_b_list.append(tmp)

mod_sample_att_b_list = valid_dataset.get_style(fixed_att_a)

it = 0
it_per_epoch = len(train_dataset) // args.batch_size

attrid = torch.tensor([i for i in range(args.n_attrs)]).to(device)
attrid = attrid.view(1, attrid.size(0))
attrid = attrid.repeat(args.batch_size, 1)

attrid_val = torch.tensor([i for i in range(args.n_attrs)]).to(device)
attrid_val = attrid_val.view(1, attrid_val.size(0))
attrid_val = attrid_val.repeat(args.n_samples, 1)

# Loss criterion
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
criterion_ce = torch.nn.CrossEntropyLoss().to(device)
criterion_attr = torch.nn.MSELoss().to(device)

criterion_bce = torch.nn.BCELoss().to(device)

labels = valid_dataset.labels

for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    #attgan.set_lr(lr)
    writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_A, styles_A, attr_A_data, styles_B, attr_B_data in progressbar(train_dataloader):
        generator.train()
        discriminator.train()
        
        img_A = img_A.to(device)
        att_a = attr_A_data.to(device)
        
        att_b = attr_B_data.to(device)
        att_b = att_b.contiguous()

        styles_A = styles_A.to(device)
        styles_B = styles_B.to(device)

        valid = torch.ones((img_A.size(0), *patch)).to(device)
        fake = torch.zeros((img_A.size(0), *patch)).to(device)
        
        # Construct attribute
        attr_raw_A = attribute_embed(attrid)
        attr_raw_B = attribute_embed(attrid)

        attr_A_intensity = att_a

        attr_A_intensity_u = attr_A_intensity.unsqueeze(-1)
        attr_A = attr_A_intensity_u * attr_raw_A

        attr_B_intensity = att_b

        attr_B_intensity_u = attr_B_intensity.unsqueeze(-1)
        attr_B = attr_B_intensity_u * attr_raw_B

        delta_intensity = attr_B_intensity - attr_A_intensity
        delta_attr = attr_B - attr_A
        
        if (it+1) % (args.n_d+1) != 0:
            recon_A = generator(img_A, styles_A, delta_intensity, delta_attr)
            pred_real, A_attr_real = discriminator(img_A, attr_B_intensity=attr_B_intensity.detach())
            loss_real = criterion_GAN(pred_real, valid)

            img_fake = generator(img_A, styles_B, delta_intensity, delta_attr)
            pred_fake, A_attr_fake, B_attr_fake = discriminator(recon_A.detach(), img_fake.detach(), attr_B_intensity.detach())

            loss_attr_D = torch.zeros(1).to(device)
            if args.dis_pred:
                loss_attr_D += criterion_attr(attr_A_intensity.detach(), A_attr_real)
                loss_attr_D += criterion_attr(attr_A_intensity.detach(), A_attr_fake)
                loss_attr_D += criterion_attr(attr_B_intensity.detach(), B_attr_fake)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = loss_real + loss_fake + loss_attr_D

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            errD = {
                'd_loss': loss_D.item(), 'd_loss_real': loss_real.item(),
                'd_loss_fake': loss_fake.item(), 'd_loss_attr': loss_attr_D.item()
            }

            add_scalar_dict(writer, errD, it+1, 'D')
        else:
            pred_recon, real_A_attr_fake, recon_A_attr_fake = discriminator(img_A, recon_A, attr_B_intensity=attr_B_intensity)

            if args.lambda_cx > 0:
                vgg_fake_B = vgg19(recon_A)
                vgg_img_B = vgg19(img_A)
            
            loss_GAN = args.lambda_GAN * criterion_GAN(pred_recon, valid)
            
            # Reconstruction loss
            loss_pixel = args.lambda_l1 * criterion_pixel(recon_A, img_A)
            
            # CX loss
            loss_CX = torch.zeros(1).to(device)
            if args.lambda_cx > 0:
                for l in vgg_layers:
                    cx = criterion_cx(vgg_img_B[l], vgg_fake_B[l])
                    loss_CX += cx * args.lambda_cx
            
            # For fake attribute
            img_fake = generator(img_A, styles_B, delta_intensity, delta_attr)
            pred_fake, fake_B_attr_fake = discriminator(img_fake)

            loss_GAN += args.lambda_GAN * criterion_GAN(pred_fake, valid)

            loss_attr = torch.zeros(1).to(device)
            if args.dis_pred:
                loss_attr += args.lambda_attr * criterion_attr(attr_A_intensity, real_A_attr_fake)
                loss_attr += args.lambda_attr * criterion_attr(attr_A_intensity, recon_A_attr_fake)
                loss_attr += args.lambda_attr * criterion_attr(attr_B_intensity, fake_B_attr_fake)

            loss_G = loss_GAN + loss_pixel + loss_CX + loss_attr

            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            errG = {
                'g_loss': loss_G.item(), 'gan_loss': loss_GAN.item(),
                'pixel_loss': loss_pixel.item(), 'attr_loss': loss_attr.item(),
                'cx_loss': loss_CX.item()
            }

            add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
        
        #if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
        #    attgan.saveG(os.path.join(
        #        'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        #    ))
            # attgan.save(os.path.join(
            #     'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % args.sample_interval == 0:
            generator.eval()

            attr_raw_A = attribute_embed(attrid_val)
            attr_raw_B = attribute_embed(attrid_val)

            attr_A_intensity = fixed_att_a

            attr_A_intensity_u = attr_A_intensity.unsqueeze(-1)
            attr_A = attr_A_intensity_u * attr_raw_A

            with torch.no_grad():
                samples = [fixed_img_a]
                for i, (att_val, fixed_style_val) in enumerate(mod_sample_att_b_list):
                    print(att_val.shape, fixed_style_val.shape)

                    att_val = att_val.to(device)
                    fixed_style_val = fixed_style_val.to(device)

                    attr_B_intensity = att_val

                    attr_B_intensity_u = attr_B_intensity.unsqueeze(-1)
                    attr_B = attr_B_intensity_u * attr_raw_B

                    delta_intensity = attr_B_intensity - attr_A_intensity
                    delta_attr = attr_B - attr_A

                    samples.append(generator(fixed_img_a, fixed_style_val, delta_intensity, delta_attr))
                samples = torch.cat(samples, dim=3)
                writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it+1)
                vutils.save_image(samples, os.path.join(
                        'output', args.experiment_name, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=True, range=(-1., 1.))
        it += 1