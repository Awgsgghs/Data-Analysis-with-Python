import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding)
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels,
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()


        # добавьте несколько слоев encoder block
        # это блоки-составляющие энкодер-части сети
        # В ячейке ниже приводится проверка построенной архитектуры.
        # Подумайте, сколько encoder_block и с какими размерами стоит выбрать для корректного решения
        self.encoder = nn.Sequential(
            encoder_block(3, 32, 3, 1),
            encoder_block(32, 64, 3, 1),
            encoder_block(64, 128, 3, 1),
            encoder_block(128, 256, 3, 1),
            encoder_block(256, 512, 3, 1)
        )

        # добавьте несколько слоев decoder block
        # это блоки-составляющие декодер-части сети
        # Не забывайте по необходимости делать ConvTranspose2d и вызов Sigmoid в конце для объединения всех восстановленных изображений и нормализации выходов
        # В ячейке ниже приводится проверка построенной архитектуры.
        # Подумайте, сколько decoder_block и с какими размерами стоит выбрать для корректного решения
        self.decoder = nn.Sequential(
            decoder_block(512,256,3,1),
            decoder_block(256,128,3,1),
            decoder_block(128,64,3,1),
            decoder_block(64,32,3,1),
            decoder_block(32,3,3,1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # downsampling
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction


def create_model():
    return Autoencoder()
