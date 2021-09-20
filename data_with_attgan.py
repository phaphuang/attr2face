# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
from torch._C import device
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        print(atts)
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    
    def __len__(self):
        return len(self.images)

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, n_style=4):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        if mode == 'train':
            self.images = images[:182000]
            self.labels = labels[:182000]
        if mode == 'valid':
            self.images = images[182000:182637]
            self.labels = labels[182000:182637]
        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)

        self.labels = torch.tensor((self.labels + 1) // 2, dtype=torch.double)
        self.n_style = n_style

        self.selected_attrs = selected_attrs
        self.len_attr = len(selected_attrs)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = self.labels[index]

        # Find style of input image
        try:
            att_a_list = torch.nonzero(att == 1)
            rand_index = torch.randint(0, len(att_a_list), (1,))
            att_a = torch.zeros([self.len_attr], dtype=torch.double)
            att_a[rand_index] = 1
            label_index = self.labels.matmul(att_a)
            label_index = torch.nonzero(label_index == 1)
        except Exception as e:
            print("Exception with error of ", att)
            label_index = torch.nonzero(torch.sum(self.labels, dim=1) == 0)
        
        task_idx_a = random.sample((list(label_index)), self.n_style)
        
        styles_A = []
        if self.n_style == 1:
            styles_A.append(self.tf(Image.open(os.path.join(self.data_path, self.images[task_idx_a[0]]))))
        else:
            for idx in range(len(task_idx_a)):
                styles_A.append(self.tf(Image.open(os.path.join(self.data_path, self.images[task_idx_a[idx]]))))
        
        styles_A = torch.cat(styles_A)

        # Select image b, first random the attribute
        idx = torch.randint(0, self.len_attr, (1,))
        att_b = torch.zeros([self.len_attr], dtype=torch.double)
        att_b[idx] = 1
        label_index = self.labels.matmul(att_b)
        label_index = torch.nonzero(label_index == 1)
        task_idx_b = random.sample((list(label_index)), self.n_style)
        #out_label = self.labels[task_idx[0]].squeeze()
        #print(att_b)
        #print(self.labels[task_idx[0]], self.labels[task_idx[1]], self.labels[task_idx[2]], self.labels[task_idx[3]])
        styles_B = []
        if self.n_style == 1:
            styles_B.append(self.tf(Image.open(os.path.join(self.data_path, self.images[task_idx_b[0]]))))
        else:
            for idx in range(len(task_idx_b)):
                styles_B.append(self.tf(Image.open(os.path.join(self.data_path, self.images[task_idx_b[idx]]))))
            
        styles_B = torch.cat(styles_B)

        return img, styles_A, att.float(), styles_B, att_b.float()
    
    def check_attribute_conflict(self, att_batch, att_name, att_names):
        def _get(att, att_name):
            if att_name in att_names:
                return att[att_names.index(att_name)]
            return None
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value
        att_id = att_names.index(att_name)
        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
                if _get(att, 'Bangs') != 0:
                    _set(att, 1-att[att_id], 'Bangs')
            elif att_name == 'Bangs' and att[att_id] != 0:
                for n in ['Bald', 'Receding_Hairline']:
                    if _get(att, n) != 0:
                        _set(att, 1-att[att_id], n)
                        _set(att, 1-att[att_id], n)
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name and _get(att, n) != 0:
                        _set(att, 1-att[att_id], n)
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name and _get(att, n) != 0:
                        _set(att, 1-att[att_id], n)
            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
                for n in ['Mustache', 'No_Beard']:
                    if n != att_name and _get(att, n) != 0:
                        _set(att, 1-att[att_id], n)
        return att_batch

    def get_style(self, fixed_att_a):

        sample_att_b_list = []

        for i in range(self.len_attr):
            tmp = fixed_att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = self.check_attribute_conflict(tmp, self.selected_attrs[i], self.selected_attrs)
            tmp, tmp_style = self.query_style(tmp, self.selected_attrs[i], self.selected_attrs, i)
            sample_att_b_list.append((tmp, tmp_style))
        
        return sample_att_b_list
    
    def query_style(self, att_batch, att_name, att_names, att_id):

        all_styles_B = []

        for idx, sample in enumerate(att_batch):
            #### Return Style ####
            #print(f"Sample {idx} Attribute: {att_name} Attribute id: {att_id}, Sample: {sample[att_id]}")
            #selected_labels = self.labels[self.labels[:, att_id].to(device) == sample[att_id]]
            #print(selected_labels)
            #task_idx_b = random.sample(list(selected_labels), self.n_style)
            selected_images = self.images[self.labels[:, att_id] == sample[att_id].cpu()]
            sample_images = random.sample(list(selected_images), self.n_style)
            #print(sample_images)

            styles_B = []
            if self.n_style == 1:
                #styles_B.append(self.tf(Image.open(os.path.join(self.data_path, self.images[task_idx_b[0]]))))
                styles_B.append(self.tf(Image.open(os.path.join(self.data_path, selected_images[0]))))
            else:
                for img in sample_images:
                    styles_B.append(self.tf(Image.open(os.path.join(self.data_path, img))))
                
            styles_B = torch.cat(styles_B)
            all_styles_B.append(styles_B.unsqueeze(0))
        
        all_styles_B = torch.cat(all_styles_B, dim=0)
        
        return att_batch, all_styles_B
            

    def __len__(self):
        return self.length

class CelebA_HQ(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int)
        
        images = ['{:d}.jpg'.format(i) for i in range(30000)]
        labels = orig_labels[indices]
        
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length



def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    attrs_default = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )