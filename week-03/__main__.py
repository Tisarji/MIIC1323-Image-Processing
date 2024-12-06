# LAB Week 03 Part 1 ------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def negative(r):
#     s = 255 - r
#     return s

# def scale_image(input_img):
#     input_img = input_img / np.max(input_img)
#     input_img = (input_img * 255).astype('int')
#     return input_img

# def plot_results(input_img, output_img, x_values, y_values, save_as):
#     # plotting the graph
#     plt.figure(figsize=(36, 12))
#     plt.subplot(131)
#     plt.imshow(input_img)
#     plt.title('Input Image')
#     plt.axis('off')
#     plt.subplot(132)
#     plt.plot(x_values, y_values)
#     plt.xlabel('Input Pixels')
#     plt.ylabel('Output Pixels')
#     plt.grid(True)
#     plt.subplot(133)
#     plt.imshow(output_img)
#     plt.title('Transformed Image')
#     plt.axis('off')

# img = cv2.imread('week-03/images-teacher-file/Fig0304(a)(breast_digital_Xray).tif')
# x_values = np.linspace(0, 255, 500)
# y_values = negative(x_values)
# img_neg = negative(img)
# plot_results(img, img_neg, x_values, y_values, 'negative')
# plt.show()

# LAB Week 03 Part 2 ------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def logTransform(r, c=1):
#     s = c * np.log(1.0 + r)
#     return s

# def scale_image(input_img):
#     input_img = input_img / np.max(input_img)
#     input_img = (input_img * 255).astype('int')
#     return input_img

# def plot_results(input_img, output_img, x_values, y_values, save_as):
#     # plotting the graph
#     plt.figure(figsize=(36, 12))

#     plt.subplot(131)
#     plt.imshow(input_img)
#     plt.title('Input Image')
#     plt.axis('off')

#     plt.subplot(132)
#     plt.plot(x_values, y_values)
#     plt.xlabel('Input Pixels')
#     plt.ylabel('Output Pixels')
#     plt.grid(True)

#     plt.subplot(133)
#     plt.imshow(output_img)
#     plt.title('Transformed Image')
#     plt.axis('off')

# im = cv2.imread('week-03/images-teacher-file/Fig0305(a)(DFT_no_log).tif')
# if im is None:
#     raise FileNotFoundError("Image file not found. Please check the file path.")

# x_values = np.linspace(0, 255, 500)
# y_values = logTransform(x_values)
# img_log = logTransform(im)
# img_log_scaled = scale_image(img_log)

# plot_results(im, img_log_scaled, x_values, y_values, 'log')
# plt.show()

# LAB Week 03 Part 3 ------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def gammaTransform(r, gamma, c=1):
#     s = c * (r ** gamma)
#     return s

# def scale_image(input_img):
#     input_img = input_img / np.max(input_img)
#     input_img = (input_img * 255).astype('int')
#     return input_img

# def plot_results(input_img, output_img, x_values, y_values, save_as):
#     # plotting the graph
#     plt.figure(figsize=(36, 12))
#     plt.subplot(131)
#     plt.imshow(input_img)
#     plt.title('Input Image')
#     plt.axis('off')
#     plt.subplot(132)
#     plt.plot(x_values, y_values)
#     plt.xlabel('Input Pixels')
#     plt.ylabel('Output Pixels')
#     plt.grid(True)
#     plt.subplot(133)
#     plt.imshow(output_img)
#     plt.title('Transformed Image')
#     plt.axis('off')


# spine_img = cv2.imread('week-03/images-teacher-file/Fig0308(a)(fractured_spine).tif')
# if spine_img is None:
#     raise FileNotFoundError("Image file not found. Please check the file path.")

# img_gamma = gammaTransform(spine_img, 0.4)
# img_gamma_scaled = scale_image(img_gamma)

# x_values = np.linspace(0, 255, 500)
# y_values = gammaTransform(x_values, 0.4)
# plot_results(spine_img, img_gamma_scaled, x_values, y_values, 'gamma_0_4')
# plt.show()

# LAB Week 03 Part 4 ------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def bitPlaneSlicing(r, bit_plane):
#     dec = np.binary_repr(r, width=8)
#     return int(dec[8 - bit_plane])

# def plot_results(input_img, output_img, x_values, y_values, save_as):
#     # plotting the graph
#     plt.figure(figsize=(36, 12))
#     plt.subplot(131)
#     plt.imshow(input_img)
#     plt.title('Input Image')
#     plt.axis('off')
#     plt.subplot(132)
#     plt.plot(x_values, y_values)
#     plt.xlabel('Input Pixels')
#     plt.ylabel('Output Pixels')
#     plt.grid(True)
#     plt.subplot(133)
#     plt.imshow(output_img)
#     plt.title('Transformed Image')
#     plt.axis('off')

# dollar_img = cv2.imread('week-03/images-teacher-file/Fig0314(a)(100-dollars).tif', 0)
# if dollar_img is None:
#     raise FileNotFoundError("Image file not found. Please check the file path.")

# bitPlaneSlicingVec = np.vectorize(bitPlaneSlicing)
# eight_bitplace = bitPlaneSlicingVec(dollar_img, bit_plane = 8)

# plt.imshow(eight_bitplace, cmap="gray")
# plt.show()
