import numpy as np
import cv2
import matplotlib.pyplot as plt


# QUESTION 1
def vertical_flip(img, arr):  # To sum up this method it is basically changing the rows with the vertical symmetric rows

    n_iter = 0
    for idx, val in enumerate(arr):
        temp1 = np.array(arr[idx])
        arr[idx] = arr[-1 - idx]
        arr[-1 - idx] = temp1
        n_iter += 1
        if row % 2 == 0:  # And this if-else block to avoid to rows changing which are already changed
            if n_iter == row / 2:
                break
        else:
            if n_iter == (row // 2) + 1:
                break
    img = arr
    plt.title("Vertically Flipped Image")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 2
def horizontal_flip(img, arr):  # It is also basically changing the columns with the horizontal symmetric columns.

    lst = []  # Some kind of a temporary variable which holds the changed values.
    for i, val1 in enumerate(arr):
        n_iter = 0
        for j, val in enumerate(val1):
            temp = val
            val1[j] = val1[-1 - j]
            val1[-1 - j] = temp
            n_iter += 1

            if col % 2 == 0:  # And this if-else block to avoid to columns changing which are already changed
                if n_iter == col // 2:
                    break
            else:
                if n_iter == (col // 2) + 1:
                    break
        lst.append(val1)

    arr = np.array(lst)
    img = arr
    plt.title("Horizontally Flipped Image")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 3
def clockwise_rotate(img, arr):
    """
    This method is basically changing the values in the matrix with an appropriate index for 90 degrees rotate
    """

    arr_2 = np.array(arr)
    for idx, val in enumerate(arr):
        temp = arr[:, idx]
        arr_2[idx, :] = temp[::-1]

    img = arr_2
    plt.title("90 Deg. Clockwise Rotated")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 4
def counter_clockwise_rotate(img, arr):
    """
    This method is basically changing the values in the matrix with an appropriate index for 90 degrees rotate
    """
    arr_2 = np.array(arr)
    for idx, val in enumerate(arr):
        temp = arr[:, (-1 - idx)]
        arr_2[idx, :] = temp

    img = arr_2
    plt.title("90 Deg. Counterclockwise Rotated")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 5
def resize(img, arr):
    """
    In this method, I wanted to get each four values as a mini matrix that their average will give the new value. So,
    firstly I created a new array that will get the specific rows and all the columns. When I get the specific
    row and all columns I wanted to get specific columns too, because as I mentioned before I wanted to get four
    values their average will give the new intensity value. I used loops to get specific rows and columns process
    in order to make it dynamic (to accomplish the code for every row and column). Also, I used iteration and epoch
    numbers to avoid infinite loops.

    """
    lst = []
    epoch = 0
    while epoch <= (row // 2):
        r_iter = 0
        for i, val1 in enumerate(arr):
            arr3 = arr[r_iter:r_iter + 2, :]
            r_iter += 2
            epoch += 1

            if epoch > row:
                break

            n_iter = 0
            epoch_2 = 0

            for j, val2 in enumerate(arr):
                if epoch_2 < (col // 2):
                    nn = n_iter + 2
                    arr4 = arr3[:, n_iter:nn]
                    lst.append((arr4.sum()) // 4)
                    epoch_2 += 1
                    n_iter += 2

                    if epoch_2 > col:
                        idx = 0

                        idx += 1
                        break

    nmp = np.array(lst)
    nmp = nmp[nmp != 0]  # Remove the zeros from the array
    nmp = nmp.reshape((row // 2), (col // 2))  # To make the array a matrix, used reshape

    cv2.imshow("Original", img)  # I displayed the image with opencv because in matplotlib we can't see the true size
    img = nmp

    img = img.astype(np.uint8)
    cv2.imshow("Resized", img)
    cv2.waitKey(0)


# QUESTION 6
def negative_transformation(img, arr):  # Changes the values via negative transformation formula

    for i, val1 in enumerate(arr):
        for j, val2 in enumerate(val1):
            arr[i, j] = 255 - val2

    img = arr
    plt.title("Negative Transformation")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 7
def gamma_transformation(img, arr, c=1, gamma=2.2):
    """
    Changes the values via gamma transformation formula, constant and gamma variables are initialized with default
    values but also they could get as a parameter.
    """

    for i, val1 in enumerate(arr):
        for j, val2 in enumerate(val1):
            arr[i, j] = ((arr[i, j] / 255) ** (1 / gamma)) * 255
    img = arr
    plt.title("Gamma Transformation")
    plt.imshow(img, cmap="gray")
    plt.show()


# QUESTION 8
def histogram(img, arr):  # This method basically counts the number of intensitys.
    i, j, k = 0, 0, 0
    hist = np.zeros(256, dtype=int)  # Full of zeros.

    while k < 256:
        hist[k] = np.count_nonzero(arr == k)  # If it is non-zero then count number is increases 1
        k += 1

    its = np.arange(0, 256, 1)

    plt.bar(its, hist, color='maroon', width=0.5)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title('Histogram')
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("Lenna.png")  # Reading the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting to Grayscale

    plt.title("Original Image")
    plt.imshow(img, cmap="gray")  # Displaying the original image
    plt.figure()

    arr = np.array(img)  # Creating a numpy array via img matrix
    row, col = arr.shape

    resize(img,arr)