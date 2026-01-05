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

def qam16_demodulator(symbols):
    bits = []
    for symbol in symbols:
        bit_i = 0 if symbol.real > 0 else 1
        bit_q = 0 if symbol.imag > 0 else 1
        bits.append(bit_i)
        bits.append(bit_q)
    return np.array(bits)

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_real = np.random.randn(len(signal)) * np.sqrt(noise_power/2)
    noise_imag = np.random.randn(len(signal)) * np.sqrt(noise_power/2)
    noise = noise_real + 1j * noise_imag
    return signal + noise

def transmitter(bits, fft_size = 64, cplen=16, pilot_freq = 3):
    data_symbols = qam16_modulator(bits)                                                                       

    # print("\n\n\n_____________________DATA SYMBOLS____________________\n",data_symbols)
    print("\n\n\n_________________DATA SYMBOLS LENGTH_________________\n",len(data_symbols))
    pilot_positions = np.arange(0, fft_size, pilot_freq)
    # print(pilot_positions)
    num_pilots = len(pilot_positions)                                                                           # 8
    num_data_symbols_per_ofdm = fft_size - num_pilots                                                           # 56
    
    num_ofdm_symbols = int(np.ceil(len(data_symbols) / num_data_symbols_per_ofdm))                              # int ceil 64 / 56 = 2
    print("\n\n\n_____________________NUMBER OF OFDM SYMBOLS____________________\n",num_ofdm_symbols)

    total_data_symbols_needed = num_ofdm_symbols * num_data_symbols_per_ofdm                                    # 2 * 56 = 112                 
    padded_data_symbols = np.zeros(total_data_symbols_needed, dtype=complex)                                    
    padded_data_symbols[:len(data_symbols)] = data_symbols

    print("\n\n\n________LENGTH OF PADDED DATA SYMBOLS_______________\n",len(padded_data_symbols))
    padded_data_symbols = padded_data_symbols.reshape(num_ofdm_symbols, num_data_symbols_per_ofdm)
    # print("\n\n\n__________PADDED SYMBOLS____________",padded_data_symbols)

    ofdm_freq_domain = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
    
    for i in range(num_ofdm_symbols):
        pilot_value = (1 + 1j) / np.sqrt(2)
        ofdm_freq_domain[i, pilot_positions] = pilot_value
        
        data_positions = np.setdiff1d(np.arange(fft_size), pilot_positions)
        ofdm_freq_domain[i, data_positions] = padded_data_symbols[i]

    # print("\n\n\n__________OFDM FREQUENCY DOMAIN____________\n",ofdm_freq_domain)

    ofdm_time_domain = np.fft.ifft(ofdm_freq_domain, axis=1)
    ofdm_with_cp = np.hstack([ofdm_time_domain[:, -cplen:], ofdm_time_domain])

    # print("\n\n\n__________OFDM WITH CP____________\n",ofdm_with_cp)                                           # First 16 samples are CP and should match last 16 samples of the same OFDM symbol
    tx_signal = ofdm_with_cp.flatten()
    
    print("\n\n\n__________LENGTH OF TX SIGNAL____________\n",len(tx_signal))                                 # 64*2 + 16*2 = 160 

    return tx_signal, data_symbols, pilot_positions, num_ofdm_symbols


def multipath_channel(tx_signal, delay, gain):
    max_delay = max(delay)
    channel_length = max_delay + 1
    channel_impulse_response = np.zeros(channel_length)
    
    for d, g in zip(delay, gain):
        channel_impulse_response[d] += g
    
    print("\n\n\n__________CHANNEL IMPULSE RESPONSE____________\n",channel_impulse_response)
    # print("\n\n\n__________TX_SIGNAL____________\n",(tx_signal))
    
    rx_signal_full = np.convolve(tx_signal, channel_impulse_response, mode='full')
    rx_signal = rx_signal_full[:len(tx_signal)]
    return rx_signal

