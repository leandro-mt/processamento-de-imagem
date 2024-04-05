from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

RADIUS = 30

"""
Cria mascara de circuferencia com raio = 30
"""
def get_mask(img, fshift):
    rows, cols = img.shape
    crow, ccol = rows/2 , cols/2

    n = len(fshift)
    y, x = np.ogrid[-crow:n-crow, -ccol:n-ccol]
    mask = x*x + y*y <= RADIUS*RADIUS
    return mask

def inverse_transform(masked_img):
    f_ishift = np.fft.ifftshift(masked_img)
    filtered_img = np.fft.ifft2(f_ishift)
    filtered_img = np.abs(filtered_img)
    return filtered_img

def show_results(filtered_img, magnitude_spectrum):
    plt.subplot(121), plt.imshow(filtered_img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def start(filename):
    # Carregar a imagem usando PIL
    img = Image.open(filename)
    img = np.array(img.convert('L'))

    # CONVERTE PARA DOMINIO DA FREQUENCIA
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # PROCESSAMENTO NO DOMINIO DA FREQUENCIA
    mask = get_mask(img, fshift)
    mask_resized = np.resize(mask, fshift.shape)
    masked_img = fshift * mask_resized

    # RETORNA AO DOMINIO ESPACIAL
    filtered_img = inverse_transform(masked_img)
    magnitude_spectrum = 100 * np.log(1 + np.abs(masked_img))

    show_results(filtered_img, magnitude_spectrum)



filename = "C:/imagens/dog.jpeg"
start(filename)