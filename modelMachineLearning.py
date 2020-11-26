import PyIPSDK
import PyIPSDK.IPSDKIPLUtility as util
import xml.etree.ElementTree as xmlet
import joblib


import MachineLearningFunctions as ml
# Name of the folder which contains the model



def My_function(inImg1,modelName):

	file = xmlet.parse(modelFolder + modelName + "/Settings.mho")
	xmlElement = file.getroot()
	listFeatures, nbLabels = ml.createListOfFeatures(xmlElement, inImg1)
	model = joblib.load(modelFolder + modelName + "/Model")
#	output = mlEval.applyModel(model,listFeatures,nbLabels)
	output = ml.applyModel(model, listFeatures, nbLabels)
	if nbLabels == 2:
		output = util.convertImg(output, PyIPSDK.eIBT_Binary)
	
	return output



modelFolder = "C:/Users/opid17.ESRF/AppData/Roaming/ReactivIP/Explorer/Machine_Learning/Pixel_Classification/"
folder="D:/Shifali/patte/less_frames/"
inImg = PyIPSDK.loadTiffImageFiles(folder,"*.tif|*.tiff")
print(inImg)

outImage = My_function(inImg,"Model 4")
outputImageFileName="D:/Shifali/patte/SegmentedImage.tif"
print(outputImageFileName)
PyIPSDK.saveTiffImageFile(outputImageFileName,outImage)


import PyIPSDK.IPSDKUI as ui
ui.displayImg(outImage)
ui.displayImg(outImage,overlayImage = inImg)