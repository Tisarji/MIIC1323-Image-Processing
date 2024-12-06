from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_histogram(image):
    if isinstance(image, str):
        image = Image.open(image)

    data = np.array(image)

    if len(data.shape) == 3:
        # Initialize histograms
        histogramRed = np.zeros(256)
        histogramGreen = np.zeros(256)
        histogramBlue = np.zeros(256)

        # Update histograms
        for row in data:
            for pixel in row:
                r, g, b = pixel[:3]
                histogramRed[r] += 1
                histogramGreen[g] += 1
                histogramBlue[b] += 1

        return histogramRed, histogramGreen, histogramBlue, data

    elif len(data.shape) == 2:
        # Initialize histogram
        histogramGray = np.zeros(256)

        # Update histogram
        for row in data:
            for pixel in row:
                histogramGray[pixel] += 1

        return histogramGray, data

    else:
        print("Unsupported image format")
        return None, None


def compare_histograms_chi_squared(histogram1, histogram2):
    chi_squared_value = 0.0  # Initialize chi-squared value

    # Iterate through bins in both histograms
    for bin1, bin2 in zip(histogram1, histogram2):
        expected_value = (bin1 + bin2) / 2  # Calculate the expected value

        if expected_value != 0:  # Avoid division by zero
            squared_difference = (bin1 - bin2) ** 2  # Squared difference
            contribution = squared_difference / expected_value  # Contribution to chi-squared
            chi_squared_value += contribution  # Add contribution to the total

    return chi_squared_value

image_path = 'week-02/image/01.jpg'

histRed, histGreen, histBlue, image_data = create_histogram(image_path)

gray_image = Image.open(image_path).convert('L')
histGray, gray_data = create_histogram(gray_image)

chi_squared_red = compare_histograms_chi_squared(histGray, histRed)
chi_squared_green = compare_histograms_chi_squared(histGray, histGreen)
chi_squared_blue = compare_histograms_chi_squared(histGray, histBlue)


plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(image_data)
plt.axis('off')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.plot(histRed, color='red', label='Red')
plt.plot(histGreen, color='green', label='Green')
plt.plot(histBlue, color='blue', label='Blue')
plt.title('RGB Histograms')
plt.legend()

plt.subplot(3, 2, 3)
plt.imshow(gray_data, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')

plt.subplot(3, 2, 4)
plt.plot(histGray, color='black', label='Grayscale')
plt.title('Grayscale Histogram')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(histGray, color='black', label='Grayscale', linewidth=2)
plt.plot(histRed, color='red', linestyle='--', label='Red')
plt.plot(histGreen, color='green', linestyle='--', label='Green')
plt.plot(histBlue, color='blue', linestyle='--', label='Blue')
plt.title('Comparison of Grayscale and RGB Histograms')
plt.legend()

plt.tight_layout()
plt.savefig('week-02/image/histograms_comparison.png')
plt.show()
