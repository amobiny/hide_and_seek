from torch.utils.data import Dataset
import csv
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class CheXpertDataSet(Dataset):
    def __init__(self, root, is_train=None, input_size=(600, 600), policy="ones", data_len=None, target_label=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []
        self.is_train = is_train
        self.input_size = input_size
        if self.is_train:
            image_list_file = os.path.join(root, 'train.csv')
        else:
            image_list_file = os.path.join(root, 'valid.csv')

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k = 0
            for line in csvReader:
                k += 1
                image_name = line[0]
                label = line[5:]

                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_names.append(os.path.join(os.path.dirname(root), image_name))
                labels.append(label)

        if target_label is None:
            target_label = list(range(np.array(labels).shape[-1]))

        if data_len is not None:
            self.image_names = image_names[:data_len]
            self.labels = np.squeeze(np.array(labels)[:data_len, target_label])
        else:
            self.image_names = image_names
            self.labels = np.squeeze(np.array(labels)[:, target_label])

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        img = Image.open(image_name).convert('RGB')
        target = self.labels[index]

        if self.is_train:
            img = transforms.Resize((500, 500), Image.BILINEAR)(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(options.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        out_name = image_name.split('/')[-3] + '_' + image_name.split('/')[-2] + '_' + image_name.split('/')[-1].split('.')[0]
        return img, target, out_name

        # image_name = self.image_names[index]
        # image = Image.open(image_name).convert('RGB')
        # label = self.labels[index]
        # if self.transform is not None:
        #     image = self.transform(image)
        # return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
