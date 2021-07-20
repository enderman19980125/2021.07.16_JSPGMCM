import heapq
import numpy as np
import random
from scipy.optimize import fsolve


def getinitialPopulation(length, pSize):
    chromsomes = np.zeros((pSize, length), dtype=np.int)
    for p in range(pSize):
        chromsomes[p, :] = np.random.randint(length)
    return chromsomes


def getEncodeLength(decisionvariables, d):
    lengths = []
    for dvar in dvariables:
        upper = dvar[1]
    low = dvar[0]
    res = fsolve(lambda x: ((upper - low) / d - 2 ** x + 1), 30)
    lengths.append(length)
    print("EncodeLength:", lengths)
    return lengths


def getDecode(population, encodelength, dvariables, delta):
    popsize = population.shape[0]
    length = len(encodelength)
    dVariables = np.zeros((populationsize, length), dtype=np.float)
    for i, popchild in enumerate(population):
        start = 0
        for j, lengthchild in enumerate(encodelength):
            power = lengthchild - 1
            decimal = 0
            for k in range(start, start + lengthchild):
                decimal += populationchild[k] * (2 ** power)
                power -= 1
    lower, upper = decisionvariables[j][0], decisionvariables[j][1]
    dvalue = lower + decimal * (upper - lower) / (2 ** lengthchild - 1)
    dVariables[i][j] = decodevalue


return dVariables


def FitValue(func, decode):
    popusize, decisionvar = decode.shape
    fv = np.zeros((popusize, 1))
    for popunum in range(popusize):
        fv[popunum][0] = func(decode[popunum][0], decode[popunum][1])
    probability = fv / np.sum(fv)
    cum_probability = np.cumsum(probability)
    return fv, cum_probability


def selectNewPopulation(decodepopu, cum_probability):
    m, n = decodepopu.shape
    newPopulation = np.zeros((m, n))
    for i in range(m):
        randomnum = np.random.random()
        for j in range(m):
            if (randomnum < cum_probability[j]):
                newPopulation[i] = decodepopu[j]
                break
    return newPopulation


def crossNewPopulation(newpopu, prob):
    m, n = newpopu.shape
    updatepopulation = np.zeros((m, n), dtype=np.uint8)
    index = random.sample(range(m), n)
    for i in range(m):
        if not index.__contains__(i):
            updatepopulation[i] = newpopu[i]
    j = 0
    while j < n:
        upp[index[j]][0:crossPoint] = newpop[index[j]][0:crossPoint]
        upp[index[j]][crossPoint:] = newpop[index[j + 1]][crossPoint:]
        upp[index[j + 1]][0:crossPoint] = newpop[j + 1][0:crossPoint]
        upp[index[j + 1]][crossPoint:] = newpop[index[j]][crossPoint:]
        j = j + 2
    return updatepopulation


def mutation(crosspopulation, mutaprob):
    mutationpop = np.copy(crosspopulation)
    m, n = crosspopulation.shape
    mutationindex = random.sample(range(m * n), mutationnums)
    for geneindex in mutationindex:
        row, colume = np.uint8(np.floor(geneindex / n)), geneindex % n
        if mutationpop[row][colume] == 0:
            mutationpop[row][colume] = 1
        else:
            mutationpop[row][colume] = 0
    return mutationpop


def findMaxPopulation(population, maxevaluation, maxSize):
    maxevalue = maxevaluation.flatten()
    maxevaluelist = maxevalue.tolist()
    maxIndex = map(maxevaluelist.index, heapq.nlargest(100, maxevaluelist))


def main():
    deta = 0.0001
    initialPopuSize = 100
    EncodeLength = 5
    population = getinitialPopulation(sum(EncodeLength), initialPopuSize)
    maxgeneration = 500
    prob = 0.8
    mutationprob = 0.01


for generation in range(maxgeneration):
    decode = getDecode(population, EncodeLength, decisionVariables, delta)
    newpopulations = selectNewPopulation(population, cum_proba)
    crossPopulations = crossNewPopulation(newpopulations, prob)
    mutationpopulation = mutation(crossPopulations, mutationprob)
    totalpopulation = np.vstack((population, mutationpopulation))
    final_decode = getDecode(totalpopulation, EncodeLength, decisionVariables, delta)
    final_evaluation, final_cumprob = getFitnessValue(fitnessFunction(), final_decode)
    population = findMaxPopulation(totalpopulation, final_evaluation, maxPopuSize)
    optimalvalue.append(np.max(final_evaluation))
    index = np.where(final_evaluation == max(final_evaluation))
    optimalvariables.append(list(final_decode[index[0][0]]))
