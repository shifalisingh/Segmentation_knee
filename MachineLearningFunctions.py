import PyIPSDK
import PyIPSDK.IPSDKIPLUtility as util
import PyIPSDK.IPSDKIPLGeometricTransform as gtrans
import PyIPSDK.IPSDKIPLArithmetic as arithm
import PyIPSDK.IPSDKIPLMorphology as morpho
import PyIPSDK.IPSDKIPLFiltering as filtering
import PyIPSDK.IPSDKIPLStats as stats
import PyIPSDK.IPSDKIPLLogical as logic
import PyIPSDK.IPSDKIPLGlobalMeasure as glbmsr
import PyIPSDK.IPSDKIPLBinarization as bin

import numpy as np

import xml.etree.ElementTree as xmlet

def SubElement(element,text):

    toReturn=None
    for child in element:
        if child.tag==text:
            toReturn=child
    if toReturn is None:
        toReturn = xmlet.SubElement(element, text)
    return toReturn

def childText(element,tag):

    text=None
    if element != None:
        for child in element:
            if child.tag==tag:
                text=child.text
    return text

def createListOfFeatures(xmlElement,image):

    numberLabelsElement = SubElement(xmlElement, 'NumberLabels')
    numberLabels = int(numberLabelsElement.text)

    featuresElement = SubElement(xmlElement, 'Features')
    allSizes = childText(featuresElement, "All_Sizes")
    allSizes = allSizes.split(",")
    for i in range(len(allSizes)):
        allSizes[i] = int(allSizes[i])

    multiRes = childText(featuresElement, "MultiRes")
    multiRes = multiRes.split(",")
    for i in range(len(multiRes)):
        multiRes[i] = int(multiRes[i])

    dimension = childText(featuresElement, "Dimension")
    if dimension == "3D" and image.getSizeZ() == 1:
        dimension = "2D"

    listFeatures = []
    for child in featuresElement:
        nameFunction = child.tag.replace("_", " ")
        if nameFunction in ["Gaussian", "Mean", "Laplacian Difference", "High Pass", "Variance","Erosion", "Dilatation", "Opening", "Closing"]:
            featuresBooleans = child.text.split(",")
            for j in range(len(featuresBooleans)):
                if featuresBooleans[j] == "True":
                    for res in multiRes:
                        addClassicFeaturesToList(image,nameFunction, allSizes[j], listFeatures, res, dimension)
        if nameFunction == "Neighborhood":
            size = int(child.text)
            for res in multiRes:
                addNeighborhoodToList(image, size, listFeatures, res, dimension)

    return listFeatures,numberLabels

def addClassicFeaturesToList(image,nameFunction, size, listFeatures, res,dimension):

    if image.getSizeC() == 1:
        newFeature = featureFromName(image, nameFunction, size, res,dimension)
        listFeatures.append(newFeature)
    else:
        for c in range(3):
            plan = PyIPSDK.extractPlan(0, c, 0,image)
            newFeature = featureFromName(plan, nameFunction, size, res, dimension)
            listFeatures.append(newFeature)

def addNeighborhoodToList(image,size,listFeatures,res,dimension):

    for i in range(int(-size / 2), int(size / 2) + 1):
        for j in range(int(-size / 2), int(size / 2) + 1):
            if dimension == "2D":
                if image.getSizeC() == 1:
                    newFeature = translation2D(image,i,j,res)
                    listFeatures.append(newFeature)
                else:
                    for c in range(3):
                        plan = PyIPSDK.extractPlan(0, c, 0, image)
                        newFeature = translation2D(plan, i, j, res)
                        listFeatures.append(newFeature)
            else:
                for k in range(int(-size / 2), int(size / 2) + 1):
                    if image.getSizeC() == 1:
                        newFeature = translation3D(image, i, j, k, res)
                        listFeatures.append(newFeature)
                    else:
                        for c in range(3):
                            volume = PyIPSDK.extractVolume( c, 0, image)
                            newFeature = translation3D(volume, i, j, k, res)
                            listFeatures.append(newFeature)

def addClassicFeatures(nameFunction, size, label, dictFeatures, res,dimension):
    if size not in dictFeatures:
        dictFeatures[size] = {}
    if nameFunction not in dictFeatures[size]:
        dictFeatures[size][nameFunction] = {}
    if res not in dictFeatures[size][nameFunction]:
        dictFeatures[size][nameFunction][res] = {}
    if label.image.getSizeC() == 1:
        try:
            newFeature = label.dictFeatures[size][nameFunction][res][0]
        except:
            newFeature = featureFromName(label.image, nameFunction, size, res,dimension)
        if newFeature is None:
            newFeature = featureFromName(label.image, nameFunction, size, res,dimension)
        dictFeatures[size][nameFunction][res][0] = newFeature
        label.listFeatures.append(newFeature)
        textName = nameFunction + " " +str(2*size+1) + "x" + str(2*size+1)
        if label.image.getSizeZ() > 1:
            textName += " " + dimension
        if res != 1:
            textName += " (x 1/"+str(res)+")"
        label.listNamesFeatures.append(textName)
    else:
        for c in range(3):
            try:
                newFeature = label.dictFeatures[size][nameFunction][res][c]
            except:
                plan = PyIPSDK.extractPlan(0, c, 0, label.image)
                newFeature = featureFromName(plan, nameFunction, size, res, dimension)
            if newFeature is None:
                plan = PyIPSDK.extractPlan(0, c, 0, label.image)
                newFeature = featureFromName(plan, nameFunction, size, res, dimension)
            dictFeatures[size][nameFunction][res][c] = newFeature
            label.listFeatures.append(newFeature)
            textName = nameFunction + " " + str(2*size+1) + "x" + str(2*size+1)
            if label.image.getSizeZ() > 1:
                textName += " " + dimension
            if res != 1:
                textName += " (x 1/" + str(res) + ")"
            if c == 0:
                textName += " (red)"
            if c == 1:
                textName += " (green)"
            if c == 2:
                textName += " (blue)"
            label.listNamesFeatures.append(textName)

def featureFromName(image, nameFunction, size, res,dimension):
    if res != 1:
        if dimension == "2D":
            imageResized = gtrans.zoom2dImg(image, 1 / res, 1 / res, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear)
        else:
            imageResized = gtrans.zoom3dImg(image, 1 / res, 1 / res, 1 / res, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear)
    else:
        imageResized = image

    if dimension == "2D":
        if nameFunction == "Gaussian":
            imageFeature = filtering.gaussianSmoothing2dImg(imageResized, size)
        if nameFunction == "Mean":
            imageFeature = filtering.meanSmoothing2dImg(imageResized, size, size)
        if nameFunction == "Laplacian Difference":
            imageFeature = filtering.laplacianDoG2dImg(imageResized, size)
        if nameFunction == "High Pass":
            imageFeature = filtering.highPass2dImg(imageResized, size)
        if nameFunction == "Variance":
            imageFeature = stats.variance2dImg(imageResized, size, size)
        if nameFunction == "Erosion":
            structuringElement = PyIPSDK.circularSEXYInfo(size)
            imageFeature = morpho.erode2dImg(imageResized, structuringElement)
        if nameFunction == "Dilatation":
            structuringElement = PyIPSDK.circularSEXYInfo(size)
            imageFeature = morpho.dilate2dImg(imageResized, structuringElement)
        if nameFunction == "Opening":
            structuringElement = PyIPSDK.circularSEXYInfo(size)
            imageFeature = morpho.opening2dImg(imageResized, structuringElement, PyIPSDK.eBEP_Disable)
        if nameFunction == "Closing":
            structuringElement = PyIPSDK.circularSEXYInfo(size)
            imageFeature = morpho.closing2dImg(imageResized, structuringElement, PyIPSDK.eBEP_Disable)
    else:
        if nameFunction == "Gaussian":
            imageFeature = filtering.gaussianSmoothing3dImg(imageResized, size)
        if nameFunction == "Mean":
            imageFeature = filtering.meanSmoothing3dImg(imageResized, size, size, size)
        if nameFunction == "Laplacian Difference":
            imageFeature = filtering.laplacianDoG3dImg(imageResized, size)
        if nameFunction == "High Pass":
            imageFeature = filtering.highPass3dImg(imageResized, size)
        if nameFunction == "Variance":
            imageFeature = stats.variance3dImg(imageResized, size, size, size)
        if nameFunction == "Erosion":
            structuringElement = PyIPSDK.sphericalSEXYZInfo(size)
            imageFeature = morpho.erode3dImg(imageResized, structuringElement)
        if nameFunction == "Dilatation":
            structuringElement = PyIPSDK.sphericalSEXYZInfo(size)
            imageFeature = morpho.dilate3dImg(imageResized, structuringElement)
        if nameFunction == "Opening":
            structuringElement = PyIPSDK.sphericalSEXYZInfo(size)
            imageFeature = morpho.opening3dImg(imageResized, structuringElement, PyIPSDK.eBEP_Disable)
        if nameFunction == "Closing":
            structuringElement = PyIPSDK.sphericalSEXYZInfo(size)
            imageFeature = morpho.closing3dImg(imageResized, structuringElement, PyIPSDK.eBEP_Disable)

    if res != 1:
        outImage = PyIPSDK.createImage(image)
        if dimension == "2D":
            gtrans.zoom2dImg(imageFeature, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear, outImage)
        else:
            gtrans.zoom3dImg(imageFeature, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear, outImage)
    else:
        outImage = imageFeature

    return outImage

def translation2D(image, x, y, res):

    if res != 1:
        imageResized = gtrans.zoom2dImg(image, 1 / res, 1 / res, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear)
    else:
        imageResized = image

    translatedImage = PyIPSDK.createImage(imageResized)
    transform = PyIPSDK.createWarpMotionTransform2d(PyIPSDK.eGeometricTransform2dType.eGT2DT_Translation, [x, y])
    gtrans.warp2dImg(imageResized, transform, translatedImage)

    if res != 1:
        outImage = PyIPSDK.createImage(image)
        gtrans.zoom2dImg(translatedImage, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear, outImage)
    else:
        outImage = translatedImage

    return outImage

def translation3D(image, x, y, z,res):

    import time
    start = time.time()

    if res != 1:
        imageResized = gtrans.zoom3dImg(image, 1 / res, 1 / res, 1/ res, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear)
    else:
        imageResized = image

    translatedImage = PyIPSDK.createImage(imageResized)
    transform = PyIPSDK.createWarpMotionTransform3d(PyIPSDK.eGeometricTransform3dType.eGT3DT_Translation, [x, y, z])
    gtrans.warp3dImg(imageResized, transform, translatedImage)

    if res != 1:
        outImage = PyIPSDK.createImage(image)
        gtrans.zoom3dImg(translatedImage, PyIPSDK.eZoomInterpolationMethod.eZIM_Linear, outImage)
    else:
        outImage = translatedImage

    print ("Translation",x,y,z," : ",time.time()-start)

    return outImage

def addNeighborhood(size,label,dictFeatures,res,dimension):

    if "Neighborhood" not in dictFeatures:
        dictFeatures["Neighborhood"] = {}
    for i in range(int(-size / 2), int(size / 2) + 1):
        if i not in dictFeatures["Neighborhood"] :
            dictFeatures["Neighborhood"][i] = {}
        for j in range(int(-size / 2), int(size / 2) + 1):
            if j not in dictFeatures["Neighborhood"][i]:
                dictFeatures["Neighborhood"][i][j] = {}
            if dimension == "2D":
                if res not in dictFeatures["Neighborhood"][i][j]:
                    dictFeatures["Neighborhood"][i][j][res] = {}
                if label.image.getSizeC() == 1:
                    try:
                        newFeature = label.dictFeatures["Neighborhood"][i][j][res][0]
                    except:
                        newFeature = translation2D(label.image,i,j,res)
                    if newFeature is None:
                        newFeature = translation2D(label.image,i,j,res)
                    print(i,j)
                    dictFeatures["Neighborhood"][i][j][res][0] = newFeature
                    label.listFeatures.append(newFeature)
                    if i == 0 and j == 0:
                        textName = "Original image"
                    else:
                        textName = "Translation [" + str(i) + "," + str(j) + "]"
                    if res != 1:
                        textName += " (x 1/" + str(res) + ")"
                    label.listNamesFeatures.append(textName)
                else:
                    for c in range(3):
                        try:
                            newFeature = label.dictFeatures["Neighborhood"][i][j][res][c]
                        except:
                            plan = PyIPSDK.extractPlan(0, c, 0, label.image)
                            newFeature = translation2D(plan, i, j, res)
                        if newFeature is None:
                            plan = PyIPSDK.extractPlan(0, c, 0, label.image)
                            newFeature = translation2D(plan, i, j, res)
                        dictFeatures["Neighborhood"][i][j][res][c] = newFeature
                        label.listFeatures.append(newFeature)
                        if i == 0 and j == 0:
                            textName = "Original image"
                        else:
                            textName = "Translation [" + str(i) + "," + str(j) + "]"
                        if res != 1:
                            textName += " (x 1/" + str(res) + ")"
                        if c == 0:
                            textName += " (red)"
                        if c == 1:
                            textName += " (green)"
                        if c == 2:
                            textName += " (blue)"
                        label.listNamesFeatures.append(textName)
            else:
                for k in range(int(-size / 2), int(size / 2) + 1):
                    if k not in dictFeatures["Neighborhood"][i][j]:
                        dictFeatures["Neighborhood"][i][j][k] = {}
                    if res not in dictFeatures["Neighborhood"][i][j][k]:
                        dictFeatures["Neighborhood"][i][j][k][res] = {}
                    if label.image.getSizeC() == 1:
                        try:
                            newFeature = label.dictFeatures["Neighborhood"][i][j][k][res][0]
                        except:
                            newFeature = translation3D(label.image, i, j, k, res)
                        if newFeature is None:
                            newFeature = translation3D(label.image, i, j, k, res)
                        dictFeatures["Neighborhood"][i][j][k][res][0] = newFeature
                        label.listFeatures.append(newFeature)
                        if i == 0 and j == 0 and k == 0:
                            textName = "Original image"
                        else:
                            textName = "Translation [" + str(i) + "," + str(j) + "," + str(k) + "]"
                        if res != 1:
                            textName += " (x 1/" + str(res) + ")"
                        label.listNamesFeatures.append(textName)
                    else:
                        for c in range(3):
                            try:
                                newFeature = label.dictFeatures["Neighborhood"][i][j][k][res][c]
                            except:
                                volume = PyIPSDK.extractVolume( c, 0, label.image)
                                newFeature = translation3D(volume, i, j, k, res)
                            if newFeature is None:
                                volume = PyIPSDK.extractVolume( c, 0, label.image)
                                newFeature = translation3D(volume, i, j, k, res)
                            dictFeatures["Neighborhood"][i][j][k][res][c] = newFeature
                            label.listFeatures.append(newFeature)
                            if i == 0 and j == 0 and k == 0:
                                textName = "Original image"
                            else:
                                textName = "Translation [" + str(i) + "," + str(j) + "," + str(k) + "]"
                            if res != 1:
                                textName += " (x 1/" + str(res) + ")"
                            if c == 0:
                                textName += " (red)"
                            if c == 1:
                                textName += " (green)"
                            if c == 2:
                                textName += " (blue)"
                            label.listNamesFeatures.append(textName)

def computeMask(tree, listImages, labelImage, num,dictMask, maskValue,previousMask=None):

    if tree.children_left[num] != -1:
        dictMask[maskValue] = bin.darkThresholdImg(listImages[tree.feature[num]],tree.threshold[num])
        maskValue += 1
        if maskValue in dictMask:
            # logic.bitwiseNotImg(dictMask[vrb.maskValue-1],dictMask[vrb.maskValue])
            logic.logicalNotImg(dictMask[maskValue-1],dictMask[maskValue])
        else:
            # dictMask[vrb.maskValue] = logic.bitwiseNotImg(dictMask[vrb.maskValue-1])
            dictMask[maskValue] = logic.logicalNotImg(dictMask[maskValue-1])
        if previousMask is not None:
            logic.bitwiseAndImgImg(dictMask[maskValue-1],previousMask,dictMask[maskValue-1])
            logic.bitwiseAndImgImg(dictMask[maskValue],previousMask,dictMask[maskValue])
        maskValue += 1

        currentValue = maskValue
        computeMask(tree, listImages, labelImage, tree.children_left[num],dictMask, maskValue, previousMask=dictMask[currentValue-2])
        computeMask(tree, listImages, labelImage, tree.children_right[num],dictMask, maskValue, previousMask=dictMask[currentValue-1])

    else:
        value = int(np.argmax(tree.value[num][0]))
        if listImages[0].getSizeZ() == 1:
            plan = PyIPSDK.extractPlan(0, 0, value, labelImage)
        else:
            plan = PyIPSDK.extractVolume(0, value, labelImage)
        arithm.addImgImg(plan, previousMask, plan)

def applyModel(model,listImages,nbLabel,probalities = False):

    if listImages[0].getSizeZ() == 1:
        labelImage = PyIPSDK.createImage(PyIPSDK.geometrySeq2d(PyIPSDK.eIBT_UInt16, listImages[0].getSizeX(), listImages[0].getSizeY(), nbLabel))
    else:
        labelImage = PyIPSDK.createImage(PyIPSDK.geometrySeq3d(PyIPSDK.eIBT_UInt16, listImages[0].getSizeX(), listImages[0].getSizeY(),listImages[0].getSizeZ(), nbLabel))
    util.eraseImg(labelImage, 0)

    dictMask = {}
    maskValue = 0

    nbTree = 0
    for tree_idx, est in enumerate(model.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1  # no support for multi-output

        maskValue = 0
        computeMask(tree, listImages, labelImage, 0, dictMask,maskValue)
        nbTree += 1
        # print(vrb.maskValue)

    maxImage = glbmsr.seqProjectionImg(labelImage, PyIPSDK.eProjStatType.ePST_Max)
    diff = arithm.genericSubtractImgImg(labelImage, maxImage)
    maskSeq = bin.thresholdImg(diff, 0, 0)
    if listImages[0].getSizeZ() == 1:
        outLabel = PyIPSDK.createImage(PyIPSDK.geometry2d(PyIPSDK.eIBT_UInt16, listImages[0].getSizeX(), listImages[0].getSizeY()))
    else:
        outLabel = PyIPSDK.createImage(PyIPSDK.geometry3d(PyIPSDK.eIBT_UInt16, listImages[0].getSizeX(), listImages[0].getSizeY(),listImages[0].getSizeZ()))
    util.eraseImg(outLabel, 0)

    for i in range(nbLabel):
        if listImages[0].getSizeZ() == 1:
            plan = PyIPSDK.extractPlan(0, 0, i, maskSeq)
        else:
            plan = PyIPSDK.extractVolume(0, i, maskSeq)
        plan = util.convertImg(plan, PyIPSDK.eIBT_UInt16)

        plan = arithm.multiplyScalarImg(plan, i)

        maskOutImage = bin.thresholdImg(outLabel, 0, 0)
        plan = arithm.multiplyImgImg(plan, maskOutImage)

        outLabel = arithm.addImgImg(plan, outLabel)

    outLabel = util.convertImg(outLabel, PyIPSDK.eIBT_Label16)

    if probalities == False:
        return outLabel
    else:
        imageProbabilities = util.convertImg(maxImage, PyIPSDK.eIBT_Real32)
        imageProbabilities = arithm.multiplyScalarImg(imageProbabilities,1/nbTree)

        return outLabel,imageProbabilities