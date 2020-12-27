import cv2 as cv
import numpy as np
import os
from time import time
from playsound import playsound

class Currency:
    """ Holds Currency Image Details/Information """
    # Class Attributes
    keypoints = []
    descriptors = []

    def __init__(self, Directory, Value = 0):
        """ Default Class Constructor """
        self.directory = Directory
        self.value = Value

    def CheckGreyscale(self, imgFile):
        """ Returns true if the image is greyscale """
        r,g,b=cv.split(imgFile)
        r_g=np.count_nonzero(abs(r-g))
        r_b=np.count_nonzero(abs(r-b))
        g_b=np.count_nonzero(abs(g-b))
        diffSum=float(r_g+r_b+g_b)
        ratio=diffSum/imgFile.size
        if ratio < 0.005:
            return True
        else:
            return False

    def FeatureExtraction(self, minHessian = 400):
        """ Using SURF to extract keypoints & descriptors from image file """
        imgFile = cv.imread(self.directory)
        # Checks whether or not the imgFile is valid to process.
        isGrey = self.CheckGreyscale(imgFile)
        if isGrey or imgFile is None:
            print('File rejected, Cannot be proccessed!')
            pass
        # Using SURF to extract the features of the image life
        detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        self.keypoints, self.descriptors = detector.detectAndCompute(imgFile, None)
        pass

def loadRefCurrencies(folder = "Dataset"):
    """ Gets Reference Currencies and Extracts their features """
    print("Loading and proccessing reference dataset:")
    RefCurrencies = []
    for value in [10,20,50,100,200]:
        for filename in os.listdir(folder+"/"+str(value)):
            fileDirectory = folder+"/"+str(value)+"/"+filename
            newCurrency = Currency(fileDirectory, value)
            newCurrency.FeatureExtraction()
            RefCurrencies.append(newCurrency)
            print(fileDirectory,"has been added")
    return RefCurrencies

def loadTestCurrencies(folder = "Testset"):
    """ Gets Reference Currencies and Extracts their features """
    print("\nLoading and proccessing test dataset:")
    TestCurrencies = []
    for filename in os.listdir(folder):
        fileDirectory = folder+"/"+filename
        newCurrency = Currency(fileDirectory)
        newCurrency.FeatureExtraction()
        TestCurrencies.append(newCurrency)
        print(fileDirectory,"has been added")
    return TestCurrencies


def Classify(RefCurrencies, TestCurrencies, loweThresh = 0.7, matchThreshold = 4):
    for TestCurrency in TestCurrencies:
        print("\nCurrently Testing |", TestCurrency.directory)
        percentages = []
        for RefCurrency in RefCurrencies:
            # Matching of Descriptors
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            knnMatches = matcher.knnMatch(RefCurrency.descriptors, TestCurrency.descriptors, 2)
            # Lowe's Ratio Test
            goodMatches = []
            for m,n in knnMatches:
                if m.distance < loweThresh * n.distance:
                    goodMatches.append(m)
            # Calculates Percentage
            percentageMatch = (len(goodMatches) / len(RefCurrency.keypoints)) * 100.0
            percentages.append(percentageMatch)
            print(RefCurrency.directory, "| Percentage =", percentageMatch)
        # Finds the currency value depending on the percentages of the descriptor matching.
        maxPercentage = max(percentages)
        outputValue = RefCurrencies[percentages.index(maxPercentage)].value
        if (maxPercentage <= matchThreshold):
            print(TestCurrency.directory, "has been Rejected!")
            playsound("Audio/Rejected.mp3")        
        else:
            TestCurrency.value = outputValue
            print(TestCurrency.directory, "is a", outputValue)
            playsound("Audio/"+str(outputValue)+".mp3")
    pass

def CountCurrency(TestCurrencies):
    """Prints the final totals of each currency detected in terminal"""
    print("-------------------------")
    rejectedCount = sum(c.value == 0 for c in TestCurrencies)
    if (rejectedCount != 0):
        print("Count of Rejected =", rejectedCount)
    for value in [10,20,50,100,200]:
        valueCount = sum(c.value == value for c in TestCurrencies)
        if (valueCount != 0):
            print("Count of", value, "=", valueCount)
    pass

# Start of time benchmark of execution time
StartTime = time()
# Code calls to start the detection process
RefCurrencies = loadRefCurrencies()
TestCurrencies = loadTestCurrencies()
Classify(RefCurrencies, TestCurrencies)
CountCurrency(TestCurrencies)
# End of the benchmark
EndTime = time()
print("Executed in", EndTime-StartTime, "sec")