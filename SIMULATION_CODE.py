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

def receiver(rx_signal, pilot_positions, num_ofdm_symbols, num_data_symbols=None, fft_size=64, cplen=16):
    ofdm_symbol_length = fft_size + cplen                                                                   # 80
    total_samples_needed = num_ofdm_symbols * ofdm_symbol_length                                            # 80*2 = 160
    # print(total_samples_needed)
    if len(rx_signal) < total_samples_needed:
        rx_signal_trimmed = np.concatenate([rx_signal, np.zeros(total_samples_needed - len(rx_signal))])
    else:
        rx_signal_trimmed = rx_signal[:total_samples_needed]


        # CHANGE TO (2,80)
        rx_signal_reshaped = rx_signal_trimmed.reshape(num_ofdm_symbols, ofdm_symbol_length)
        # print(np.shape(rx_signal_reshaped))

        # REMOVE CP
        rx_symbols_no_cp = rx_signal_reshaped[:, cplen:cplen + fft_size]
        print("\n\n\n___________RX SYMBOLS WITH NO CP_________\n",np.shape(rx_symbols_no_cp))

        recovered_freq_domain = np.fft.fft(rx_symbols_no_cp, axis=1)
        # print("\n\n\n_______RECOVERED SYMBOLS_______\n",recovered_freq_domain)
        print("\n SHAPE \n",np.shape(recovered_freq_domain))

    equalized_symbols = np.zeros_like(recovered_freq_domain)
    data_positions = np.setdiff1d(np.arange(fft_size), pilot_positions)


    for i in range(num_ofdm_symbols):
        received_pilots = recovered_freq_domain[i, pilot_positions]
        # print("\n\n\n________RECEIVED PILOTS_________\n",received_pilots)

        transmitted_pilots = (1 + 1j) / np.sqrt(2)
        
        # Channel estimation at pilot positions
        channel_at_pilots = received_pilots / transmitted_pilots
        # print("\n\n\n__________received_pilots / transmitted_pilots____________\n",channel_at_pilots)
        all_positions = np.arange(fft_size)
        
        # Interpolate real part
        channel_real = np.interp(all_positions, pilot_positions, 
                               np.real(channel_at_pilots),
                               left=np.real(channel_at_pilots[0]),
                               right=np.real(channel_at_pilots[-1]))
        # print("\n\n\n__________CHANNEL REAL INTERPOLATION____________\n",channel_real)
        
        channel_imag = np.interp(all_positions, pilot_positions,
                               np.imag(channel_at_pilots),
                               left=np.imag(channel_at_pilots[0]),
                               right=np.imag(channel_at_pilots[-1]))
        # print("\n\n\n__________CHANNEL IMAGINARY INTERPOLATION____________\n",channel_imag)

        
        # Combine into complex channel estimate
        channel_estimate = channel_real + 1j * channel_imag
        
        # Equalization with regularization
        equalizer = 1.0 / (channel_estimate)
        equalized_symbols[i] = recovered_freq_domain[i] * equalizer

        received_data_symbols = equalized_symbols[:, data_positions].flatten()

        if num_data_symbols is not None:
            received_data_symbols = received_data_symbols[:num_data_symbols]
    
    received_bits = qam16_demodulator(received_data_symbols)

    return received_bits

def calc_error(original_bits, received_bits):
    same = 0
    length = len(original_bits)
    for i in range (length):
        if original_bits[i] != received_bits[i]:
            print(f"Error at position {i}: original {original_bits[i]}, received {received_bits[i]} \n")
        else:
            same += 1
    print("______________CORRECT BITS_________________\n",same)

    errors = np.sum(original_bits != received_bits)
    BER = errors / len(original_bits)
    print("______________BIT ERROR RATE_________________\n", BER * 100)

def main():
    original_image = 'C:\\Users\\Lenovo\\Desktop\\STUDY\\SIGNAL PROCESSING\\OFDM\\image.png' 
    bits, w, h = image_to_bits(original_image)
    # bits = np.random.randint(0, 2, 10000)
    print("_______________LEN OF BITS_________________\n\n", len(bits))

    # Convert bit string to numpy array of integers
    bits_array = np.array(list(bits), dtype=int)

    tx_signal, original_symbols, pilot_positions, num_ofdm_symbols = transmitter(bits_array)

    #ADD MULTIPATH AND NOISE
    #MULTIPATH
    delay = [0, 3, 8 , 12]  
    gain = [1, 0.3, 0.2, 0.1]  

    rx_signal = multipath_channel(tx_signal, delay, gain)
    rx_signal = add_awgn(rx_signal, snr_db=15)  # Add AWGN

    print("\n\n\n________LENGTH OF SIGNAL AFTER PASSING FROM MULTIPATH__________\n",len(rx_signal))

    out_bits= receiver(rx_signal, pilot_positions, num_ofdm_symbols, len(original_symbols), fft_size=64, cplen=16)

    print("\n\n\n________INPUT BITS__________\n",np.array(bits))
    print("\n\n\n________OUTPUT BITS__________\n",out_bits)
    print("\n\n\n________OUTPUT BITS LENGHT__________\n",len(out_bits))

    calc_error(np.array([int(b) for b in bits]), out_bits)

    received_image = bits_to_image(out_bits, w, h)
    received_image.show()
    received_image.save('reconstructed_image.png')

    pass

main()