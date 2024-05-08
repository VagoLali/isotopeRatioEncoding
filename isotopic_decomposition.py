import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.container as container
import seaborn as sns
import pandas as pd

def normalizeToSumAbundances(abundances: list[float]) -> list[float]:
    total = sum(abundances)
    return [ab / total for ab in abundances]


def fullNameOfMassDistribution(numOfD: int) -> str:
    numOfH = 26 - numOfD
    if numOfD == 1:
        numOfD = ''
    return 'C20' + 'H' + str(numOfH) + 'D' + str(numOfD) + 'N2O5'


def shortNameOfMassDistribution(numOfD: int) -> str:
    # numOfH = 26 - numOfD
    if numOfD == 1:
        numOfD = ''
    return 'D' + str(numOfD)


def nrDeutShort(shortName: str) -> float:
    numOfD = shortName[1:]
    if numOfD == '':
        return 1.0
    return float(numOfD)


def nrDeutFull(fullName: str) -> str:
    numOfD = fullName[fullName.find('D') + 1:fullName.find('N')]
    if numOfD == '':
        return 1.0
    return float(numOfD)


def deutDist(md1: (list[float], float), md2: (list[float], float)) -> float:
    return abs(md1.avgDeut - md2.avgDeut)


def normalizeForNdp(md: (list[float], float)) -> (list[float], float):
    norm = np.linalg.norm(md[0])
    return ([w / norm for w in md[0]], md[1])


def NDP(normMassDistr1: list[float], normMassDistr2: list[float]) -> float:
    return np.dot(normMassDistr1, normMassDistr2)


def createMixtureWeights(numberOfUnits: int = 10, nrOfIngredients: int = 3, maxNrOfDiffIng: int = 3) -> list[
    list[float]]:
    numberOfUnitsLeftList = [numberOfUnits]
    weightsList = [[]]
    nrDiffLeftList = [maxNrOfDiffIng]
    for nrOfIngredientsLeft in range(nrOfIngredients, 0, -1):
        weightsListNew = []
        if nrOfIngredientsLeft == 1:
            for (nrLeft, wBeg) in zip(numberOfUnitsLeftList, weightsList):
                weightsListNew += [wBeg + [nrLeft]]
            weightsList = weightsListNew

        else:
            unitLeftListNew = []
            diffLeftListNew = []
            for (nrUnitsLeft, nrDiffLeft, wBeg) in zip(numberOfUnitsLeftList, nrDiffLeftList, weightsList):
                weightsListNew += [wBeg + [0]]
                unitLeftListNew += [nrUnitsLeft]
                diffLeftListNew += [nrDiffLeft]

                if nrUnitsLeft > 0:
                    weightsListNew += [wBeg + [nrUnitsLeft]]
                    unitLeftListNew += [0]
                    diffLeftListNew += [nrDiffLeft - 1]

                if nrDiffLeft > 1:
                    for i in range(1, nrUnitsLeft):
                        weightsListNew += [wBeg + [i]]
                        unitLeftListNew += [nrUnitsLeft - i]
                        diffLeftListNew += [nrDiffLeft - 1]

            numberOfUnitsLeftList = unitLeftListNew
            weightsList = weightsListNew
            nrDiffLeftList = diffLeftListNew
    return [[w / numberOfUnits for w in ww] for ww in weightsList]


def createAllNormMixtures(massDistributions: list[(np.ndarray[float], float)], weightsList: list[list[float]]) -> \
list[(list[float], float)]:
    return [normalizeForNdp(combineMassDistributions(wList, massDistributions)) for wList in weightsList]


def computeNdpMatrix(massDistributions: list[(np.ndarray[float], float)]) -> list[list[float]]:
    return [[NDP(massDistributions[i], massDistributions[j]) if i < j else 0.0 for i in range(len(massDistributions))]
            for j in range(len(massDistributions))]


def computeCrossNdpMatrix(mixtures1: list[list[float]], mixtures2: list[list[float]], triangleOnly: bool = False) -> \
list[list[float]]:
    return [[NDP(mixtures1[i], mixtures2[j]) if ((not triangleOnly) or i < j) else 0.0 for i in range(len(mixtures1))]
            for j in range(len(mixtures2))]


def computeDeutDistMatrix(massDistributions: list[(np.ndarray[float], float)]) -> list[list[float]]:
    return [
        [deutDist(massDistributions[i], massDistributions[j]) if i < j else 0.0 for i in range(len(massDistributions))]
        for j in range(len(massDistributions))]


def computeMixturesAndNdps(numberOfUnits: int, nrOfIngredients: int = 3, maxNrOfDiffIng: int = 3) -> (
list[list[float]], list[list[float]]):
    start = time.time()
    weightsList = createMixtureWeights(numberOfUnits, nrOfIngredients, maxNrOfDiffIng)
    massDistributionsWithDeut = massDistributionOfBuildingBlocks(massDistributionsData, buildingBlocks)[
                                :nrOfIngredients]
    mixtures = createAllNormMixtures(massDistributionsWithDeut, weightsList)
    nrMixtures = len(weightsList)
    ndpMx = computeNdpMatrix(mixtures)
    nrOfNdps = int(nrMixtures * (nrMixtures - 1) / 2)
    end = time.time()
    return ndpMx, weightsList


###########################
### Stats ##########
##############

def pairsWithHighNDP(mx: list[list[float]], componentNames: list[str], weightsList: list[list[float]],
                     threshold: float) -> (list[(str, str)], list[(int, int)]):
    nrOfConstituents = len(componentNames)
    mixNames = ["".join(['{0:.0%}'.format(w[i]) + " " + componentNames[i] + " + " if w[i] > 0 else "" for i in
                         range(nrOfConstituents)])[:-3] for w in weightsList]
    indices = filterIndices(mx, threshold)
    textFormat = [(mixNames[i], mixNames[j]) for (i, j) in indices]
    return textFormat, indices


def heatMap(mx: list[list[float]], vmin: float = 0.0) -> None:
    mask = np.full_like(mx, True, bool)
    mask[np.tril_indices_from(mask, 1)] = False
    with sns.axes_style("white"):
        ax = sns.heatmap(mx, mask=mask, vmin=vmin, vmax=1.0, square=True, cmap="YlGnBu")
        plt.show()


def mxHist(mx: list[list[float]]) -> (np.ndarray[float], np.ndarray[float], container.BarContainer):
    mask = np.full_like(mx, False, bool)
    mask[np.tril_indices_from(mask, 1)] = True
    flatMask = np.array(flatten(mask))
    flatList = np.array(flatten(mx))
    return plt.hist(roundList(flatList[flatMask]))


def flatAndFilt(mx: list[list[float]]) -> list[float]:
    mask = np.full_like(mx, False, bool)
    mask[np.tril_indices_from(mask, 1)] = True
    flatMask = np.array(flatten(mask))
    flatList = np.array(flatten(mx))
    return flatList[flatMask]


def maxNdp(numberOfUnits: int, nrOfIngredients: int = 3, maxNrOfDiffIng: int = 3) -> float:
    ndpMx, weightsList = computeMixturesAndNdps(numberOfUnits, nrOfIngredients, maxNrOfDiffIng)
    return mxMax(ndpMx)


def filterIndices(mx: list[list[float]], threshold: float) -> list[(int, int)]:
    outIndices = []
    for i in range(len(mx)):
        for j in range(len(mx[0])):
            if mx[i][j] >= threshold:
                outIndices += [(i, j)]
    return outIndices


def filterIndicesByNdpAndDd(ndpMx: list[list[float]], ndpThreshold: float, ddMx: list[list[float]], ddLevel: float) -> \
list[(int, int)]:
    outIndices = []
    for i in range(len(ndpMx)):
        for j in range(len(ndpMx[0])):
            if ndpMx[i][j] >= ndpThreshold and ddMx[i][j] == ddLevel:
                outIndices += [(i, j)]
    return outIndices


def filterIndicesByNdp(ndpMx: list[list[float]], ndpThreshold: float) -> list[(int, int)]:
    outIndices = []
    for i in range(len(ndpMx)):
        for j in range(len(ndpMx[0])):
            if ndpMx[i][j] >= ndpThreshold:
                outIndices += [(i, j)]
    return outIndices


##########################

def mxMax(mx: list[list[float]], digits: int = 3) -> float:
    return roundUp(max([max(row) for row in mx]), digits)


def roundUp(x: float, nrDecimals: int = 3) -> float:
    return math.ceil(x * 10 ** nrDecimals) / 10 ** nrDecimals


def flattenAndRound(mx: list[list[float]], numberOfDecimals: int = 3) -> list[float]:
    return [roundUp(item, numberOfDecimals) for row in mx for item in row]


def roundList(l: list[float], numberOfDecimals: int = 3) -> list[float]:
    return [roundUp(item, numberOfDecimals) for item in l]


def flatten(mx: list[list[float]]) -> list[float]:
    return [item for row in mx for item in row]


#########################

def blowUpAbundances(masses: list[float], abundances: list[float], allMasses: set[float], simplify: bool = True) -> \
list[float]:
    out = []
    for ma in allMasses:
        try:
            i = masses.index(ma)
            out += [abundances[i]]
        except:
            out += [0.0]

    if simplify:
        simpleMasses = np.floor(allMasses)
        simpleOut = []
        for i in range(len(allMasses)):
            if (i == 0 or simpleMasses[i - 1] < simpleMasses[i]):
                simpleOut += [out[i]]
            else:
                simpleOut[-1] += out[i]
        return simpleOut
    return out


def combineMassDistributions(weights: list[list[float]], massDistributionsWithDeut: list[(list[float], float)]) -> (
list[float], float):
    out = np.zeros(len(massDistributionsWithDeut[0][0]))
    avgDeut = 0.0
    for w, dd in zip(weights, massDistributionsWithDeut):
        avgDeut += w * dd[1]
        out += w * dd[0]
    return (out, avgDeut)


def massDistributionOfBuildingBlocks(massDistributionsD: pd.core.frame.DataFrame,
                                     buildingBl: pd.core.frame.DataFrame) -> list[(np.ndarray[float], float)]:
    grouped = massDistributionsD.groupby('molecular_formula')
    allMasses = set()
    for key, group in grouped:
        allMasses.update(group['mass'].tolist())
    allMasses = sorted(allMasses)
    theoreticalDistributionsWithKeys = [(key, nrDeutFull(key), normalizeToSumAbundances(
        blowUpAbundances(group['mass'].tolist(), group['abundance'].tolist(), allMasses))) for key, group in grouped]
    theoreticalDistributionsWithKeys.sort(reverse=True,
                                          key=lambda distr: int(distr[0][distr[0].index('H') + 1:distr[0].index('D')]))
    theoreticalDistributionsWithD = [(np.array(dist), deut) for _, deut, dist in theoreticalDistributionsWithKeys]
    actualDistributions = [combineMassDistributions(buildingBl.iloc[i].tolist()[1:], theoreticalDistributionsWithD) for
                           i in range(buildingBl.shape[0])]
    return actualDistributions


def findNeighbors(weights: list[float], numberOfUnits: int) -> list[float]:
    outList = []
    unit = 1. / numberOfUnits
    zeros = np.argwhere(np.array(weights) == 0)
    ones = np.argwhere(np.array(weights) == 1)
    rest = np.argwhere((0 < np.array(weights)) * (np.array(weights) < 1))
    for down in np.append(ones, rest):
        for up in np.append(zeros, rest):
            if up != down:
                out = np.copy(weights)
                out[down] -= unit
                out[up] += unit
                outList += [out]
    return outList
