import numpy as np
import cv2



def centroid_histogram(labels):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(labels)) + 1)
	(hist, _) = np.histogram(labels, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar



def DegreeTrans(theta):
    res = (theta * 180)/ np.pi 
    return res
 
 # Повернуть градус изображения против часовой стрелки (оригинальный размер)
def rotateImage(src, degree):
         # Центр вращения является центром изображения
    h, w = src.shape[:2]
         # Рассчитать 2D повернутую матрицу аффинного преобразования
    RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1)
    print(RotateMatrix)
         # Аффинное преобразование, цвет фона заполнен белым
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate
 
 # Рассчитать угол с помощью преобразования Хафа
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(midImage, 50, 200, 3)
    lineimage = srcImage.copy()
 
         # Обнаружение прямых линий по преобразованию Хафа
         # Четвертый параметр - это порог, чем больше порог, тем выше точность обнаружения
    lines = cv2.HoughLines(dstImage, 1, np.pi/180, 200)
         # Из-за разных изображений порог установить нелегко, так как он установлен слишком высоко, поэтому линия не может быть обнаружена, порог слишком низкий, линия слишком большая, скорость очень низкая
    sum = 0
    count = 0
         # Нарисуйте каждый отрезок по очереди
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # В качестве угла поворота выберите только наименьший угол
            sum += theta
            count += 1 
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Imagelines", lineimage)
 
         # Усредняя все углы, эффект вращения будет лучше
    average = sum / len(lines)
    ugl = sum / count
    angle = DegreeTrans(average)-90
    print(angle)
    return angle



