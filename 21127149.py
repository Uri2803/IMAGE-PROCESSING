# Student Name: Huỳnh Minh Quang
# Student ID: 21127149

#import thư viện 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Input funtion
def Input():
    input_File_Name= input("Enter the image file name: ")
    image = Image.open(input_File_Name)
    # Convert the image to a numpy array
    img = np.array(image)
    return img,  input_File_Name

# Display Image_original and newImage
def Display(image, newImage, key):  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image_original")
    axes[0].axis("off")
    axes[1].imshow(newImage)
    axes[1].set_title("Image_"+key)
    axes[1].axis("off") 
    plt.show()

# Display Image_original and newImage --gray image
def Display_Gray(image, newImage, key):  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image_original")
    axes[0].axis("off")
    axes[1].imshow(newImage, cmap='gray')
    axes[1].set_title("Image_"+key)
    axes[1].axis("off")
    plt.show()
    
#save the changed image file
def Save_img(newImage, input_File_Name, key): 
    output_format = 'png' #format ảnh có thẻ thay đổi thành jpg/pdf/..
    fileName = input_File_Name.split('.')
    fileName = fileName[0] +'_'+ key +'.'+ output_format
    image=Image.fromarray(newImage.astype(np.uint8))
    image.save(fileName)

#Change the brightness
def change_brightness(image, K):
    newImage = np.uint8(np.clip(image + np.array([K], dtype=np.int16), 0, 255))
    return newImage

#Change Contrast
def change_contrast(image, K):
    factor = np.clip(float(K), -255, 255)
    contrast = np.clip(float(K), -255, 255)
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    newImage= np.uint8(np.clip(factor * (image.astype(float) - 128) + 128, 0, 255))
    return newImage

# Flip the image (horizontal - vertical)
def flip_image(image, direction):
    if direction == 'Vertical':
        newImage= np.flipud(image)
    elif direction=='Horizontal':
        newImage=np.fliplr(image)
    else:
        print("Error flip image ")
        exit()
    newImage = np.clip(newImage, 0, 255).astype(np.uint8)
    return newImage

#Convert RGB image to gray image
def rgb_gray(image):
    weight =[0.2126, 0.7152, 0.0722]
    newImage=np.uint8(np.dot(image[..., :3], weight))
    newImage = np.clip(newImage, 0, 255).astype(np.uint8)
    return newImage

#Convert RGB image to sepia image
def rgb_sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    # Calculate new value for each pixel
    newImage= np.dot(image, sepia_filter.T)
    newImage = np.clip(newImage, 0, 255).astype(np.uint8)
    return newImage

# Gaussian function
def Gaussian_func(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Calculate 2D Gaussian filter matrix.
def calc_Gaussian_kernel(kernel_size, sigma):
    x, y = np.meshgrid(np.arange(-kernel_size // 2, kernel_size // 2 + 1), 
                       np.arange(-kernel_size // 2, kernel_size // 2 + 1))
    kernel = Gaussian_func(x, y, sigma)
    kernel /= np.sum(kernel)
    return kernel

# Convolution on 2 dimensional matrices
def convolve_layer(layer, kernel):
    view = kernel.shape + tuple(np.subtract(layer.shape, kernel.shape) + 1)
    submatrices = np.lib.stride_tricks.as_strided(layer, shape = view, strides = layer.strides * 2)
    return np.einsum('ij,ijkl->kl', kernel, submatrices)

# Convolution on a color image consisting of 3 RGB channels
def convolution(img, kernel):
    return np.dstack((convolve_layer(img[:,:,0], kernel),
                      convolve_layer(img[:,:,1], kernel),
                      convolve_layer(img[:,:,2], kernel)))

#Blur image
def blur_image(image, kernel_size):
    kernel = calc_Gaussian_kernel(kernel_size, sigma=(kernel_size-1)/6)
    newImage = convolution(image, kernel)
    newImage = np.clip(newImage, 0, 255).astype(np.uint8)
    return newImage

# Sharpen matrix
def sharpen_kernel():
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

#sharpen image
def sharpen_image(image):
    kernel = sharpen_kernel() # Create a Gaussian filter with sharpen matrix
    newImage = convolution(image, kernel)
    newImage = np.clip(newImage, 0, 255).astype(np.uint8)
    return newImage


#Crop the image from the center to the target size
def crop_center(image, target_height, target_width):
    height, width = image.shape[:2]
    #center image
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2
    end_y = start_y + target_height
    end_x = start_x + target_width
    newImage = image[start_y:end_y, start_x:end_x]
    return newImage

# crop image circle
def crop_circle(image):
    height, width = image.shape[:2]
    center_y, center_x = height // 2, width // 2  #center
    y, x = np.ogrid[:height, :width]
    # equation of the circle
    mask = (x - center_x)**2 + (y - center_y)**2 <= min(center_y, center_x)**2
    newImage=image * mask[:, :, np.newaxis]
    return newImage

#create ellip mask
def create_elliptical_mask(shape, center_x, center_y, major_axis, minor_axis, angle):
    mask = np.zeros(shape, dtype=np.uint8)
    y, x = np.ogrid[:shape[0], :shape[1]]
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Equation of the ellipse
    ellipse = (((x - center_x) * cos_a + (y - center_y) * sin_a) ** 2) / (major_axis ** 2) + \
              (((x - center_x) * sin_a - (y - center_y) * cos_a) ** 2) / (minor_axis ** 2) <= 1
    mask[ellipse] = 255
    return mask

# Crop the image in the frame of 2 diagonal ellip
def crop_ellip_frame(image):
    # Parameters of two diagonal ellipses
    height, width = image.shape[:2]
    center_y, center_x = height // 2, width // 2
    major_axis = int(width * 0.5)
    minor_axis = int(height * 0.3)
    angle = 30 # The inclination of the ellipse (unit is degrees)
    mask1 = create_elliptical_mask(image.shape[:2], center_x, center_y, major_axis, minor_axis, angle)
    mask2 = create_elliptical_mask(image.shape[:2], center_x, center_y, major_axis, minor_axis, -angle)
    mask = mask1 | mask2
    # Apply mask to crop iamge
    newImage=image.copy()
    newImage[mask == 0] = 0
    y, x = np.nonzero(mask)
    min_y, max_y = np.min(y), np.max(y)
    min_x, max_x = np.min(x), np.max(x)
    newImage = newImage[min_y:max_y + 1, min_x:max_x + 1]
    return newImage

# Menu to select functions
def Menu():
    print("\nNhập phím chọn các chức năng xử lý ảnh:")
    print("[0] Chọn tất cả các chức năng và file ảnh")
    print("[1] Thay đổi độ sáng")
    print("[2] Thay đổi độ tương phản")
    print("[3] Lật ảnh (ngang - dọc)")
    print("[4] Chuyển đổi ảnh RGB thành ảnh xám hoặc sepia")
    print("[5] Làm mờ ảnh")
    print("[6] Làm sắc nét ảnh")
    print("[7] Cắt ảnh từ trung tâm với kích thước")
    print("[8] Cắt ảnh thành một khung hình tròn")
    print("[9] Cắt ảnh thành một khung hình elip chéo")
    print("[q] Thoát")

# Menu to choose the direction to flip the image
def Menu_flip():
    print("\nNhập phím để chọn hướng lật ảnh")
    print("[v] Vertical - dọc ")
    print("[h] Horizontal - ngang")
    print("[q] Thoát")
    while True:
        direction = input("Chọn hướng lật ảnh: ")
        if direction=='v':
            return 'Vertical'
        elif direction=='h':
            return 'Horizontal'
        elif direction=='q':
            exit()
        else:
             print("Lựa chọn không hợp lệ")

# Menu to choose whether to convert the image to grayscale or spepia
def Menu_gray_sepia():
    print("\nNhập phím để chọn đổi ảnh RGB thành ảnh xám/sepia ")
    print("[g] Gray - xám ")
    print("[s] Spepia")
    print("[q] Thoát")
    while True:
        direction = input("xám/sepia: ")
        if direction=='g':
            return 'Gray'
        elif direction=='s':
            return 'Spepia'
        elif direction=='q':
            exit()
        else:
             print("Lựa chọn không hợp lệ")

def Choice(choice, image):
    key=''
    newImage=image
    if choice == '1':
        key='brightness'
        # k: -255 -> 255
        K = 70 # can change > 0 increase brightness and vice versa
        newImage=change_brightness(image, K)
    elif choice == '2':
        key='contrast'
         # k: -255 -> 255
        K = 70 # can change >0 increase contrast and vice versa
        newImage=change_contrast(image, K)
    elif choice == '3':
        key='flip'
        direction = Menu_flip()
        newImage=flip_image(image, direction)
    elif choice == '4':
        sub_choice = Menu_gray_sepia()
        if sub_choice == 'Gray':
            key='gray'
            newImage=rgb_gray(image)
        elif sub_choice == 'Spepia':
            key='spepia'
            newImage=rgb_sepia(image)
        else:
            print("Lựa chọn không hợp lệ")
    elif choice == '5':
        key='blur' # Kernel size: from 1 to positive infinity, must be odd
        kernel_size = 11
        newImage=blur_image(image, kernel_size)
    elif choice == '6':
        key='sharpen'
        newImage=sharpen_image(image)
    elif choice == '7':
        height = int(input("\nNhập chiều cao ảnh: "))
        width = int(input("Nhập chiều rộng ảnh: "))
        key='crop('+str(height)+'x'+str(width)+')'
        newImage=crop_center(image, height, width)
    elif choice == '8':
        key='circle'
        newImage=crop_circle(image)
    elif choice == '9':
        key='ellip'
        newImage=crop_ellip_frame(image)
    return newImage, key 
     
# Main         
def main():
    image, inputFileNmae=Input()
    Menu()
    while True:
        choice = input("\nNhập lựa chọn của bạn: ")
        if choice=='0':
            for i in range (1,10):
                choice=str(i)
                newImage, key=Choice(choice, image)
                Save_img(newImage,inputFileNmae, key)
                print("Đã lưu ảnh!")
                # if key=='gray':
                #     Display_Gray(image, newImage, key)
                # else:
                #     Display(image, newImage, key)
        elif choice>'0' and choice <='9':
            newImage, key=Choice(choice, image)
            if key=='gray':
                Display_Gray(image, newImage, key)
            else:
                Display(image, newImage, key)
            Save_img(newImage, inputFileNmae, key)
            print("Đã lưu ảnh!")
        elif choice=='q':
            break
        else:
            print("Lỗi cú pháp. Xin nhập lại")  

if __name__ == '__main__':
   main()