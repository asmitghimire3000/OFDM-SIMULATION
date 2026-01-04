import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_to_bits(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert to color
    img_array = np.array(img)
    h, w, c = img_array.shape  # c will be 3 for RGB
    # Convert each color channel of each pixel to bits
    bits = ''.join(format(pixel, '08b') for row in img_array for col in row for pixel in col)
    return bits, w, h

def bits_to_image(bits, w, h):
    bits_str = ''.join(map(str, bits))
    
    # Convert bits back to integers
    pixels = [int(bits_str[i:i+8], 2) for i in range(0, len(bits_str), 8)]
    
    # Reshape to (height, width, 3 channels)
    img_array = np.array(pixels, dtype=np.uint8).reshape((h, w, 3))
    
    img = Image.fromarray(img_array, mode='RGB')
    return img

def qam16_modulator(bits):
    if len(bits) % 2 != 0:
        raise ValueError("Input length must be even for QPSK.")
    
    dibits = bits.reshape(-1, 2)
    symbol_map = {
        (0, 0): ( 1 + 1j) / np.sqrt(2),
        (0, 1): ( 1 - 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (-1 + 1j) / np.sqrt(2)
    }
    symbols = np.array([symbol_map[tuple(dibit)] for dibit in dibits])
    return symbols