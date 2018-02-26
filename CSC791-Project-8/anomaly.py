#!/usr/bin/env python
import sys
import os
import igraph
import numpy as np
import random as rd
rd.seed(0)
import matplotlib.pyplot as plt

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
NUMBER_OF_BITS = 32

def main():
    if len(sys.argv) != 2:
        print "Usage: python anomaly.py <dir_of_data_files>"
        sys.exit(EXIT_FAILURE)
    dir_of_data_files = sys.argv[1] # read in data file name
    graphTimeSeries = load_data(dir_of_data_files) # read data of graph
    similarityTimeSeries = calculate_similarity_timeseries(graphTimeSeries) # calculate similarity for graphs over given time series
    save_timeseries(similarityTimeSeries, dir_of_data_files) # save similarities over time series
    anomalyUpperThres, anomalyLowerThres = calculate_anomaly(similarityTimeSeries) # calculate anomaly threshold
    plot_timeseries(similarityTimeSeries, anomalyUpperThres, anomalyLowerThres, dir_of_data_files) # plot similarities over time series
    return EXIT_SUCCESS

def load_data(dir_of_data_files):
    graphTimeSeries = list()
    countFile = 0
    for file in os.listdir(dir_of_data_files): # read over all text data files
        if file.endswith(".txt"):
            countFile += 1
    dirname = dir_of_data_files.strip('\/').split('/')[-1]
    for index in range(countFile):
        try:
            file = str(index) + '_' + dirname + '.txt'
            #print file
            fp = open(dir_of_data_files.strip('\/') + '/' + file, 'r')
            graphInfo = list()
            for line in fp:
                entry = [int(term) for term in line.split()]
                graphInfo.append(entry)
            graphTimeSeries.append(graphInfo)
            fp.close()
        except IOError:
            print file + " cannot open!"
            continue
    return graphTimeSeries

def calculate_similarity_timeseries(graphTimeSeries):
    similarityTimeSeries = list()
    for index in range(len(graphTimeSeries)-1):
        print index
        graphInfoA = graphTimeSeries[index]
        header = graphInfoA[0]
        graphA = igraph.Graph(header[0], directed=True)
        for edge in graphInfoA[1:]:
            graphA.add_edge(edge[0], edge[1])

        graphInfoB = graphTimeSeries[index+1]
        header = graphInfoB[0]
        graphB = igraph.Graph(header[0], directed=True)
        for edge in graphInfoB[1:]:
            graphB.add_edge(edge[0], edge[1])
        featureA = phi_func(graphA)
        fingerPrintA = fingerprint_hash(featureA)
        featureB = phi_func(graphB)
        fingerPrintB = fingerprint_hash(featureB)
        similarity = sim_SS(fingerPrintA, fingerPrintB)
        similarityTimeSeries.append(similarity)
    return similarityTimeSeries

def sim_SS(fingerPrintA, fingerPrintB): # calculate similarity of two graphs
    vectorA = np.array(fingerPrintA)
    vectorB = np.array(fingerPrintB)
    similarityVector = (vectorA == vectorB)
    similarity = sum(similarityVector.tolist())*1.0/ float(len(similarityVector.tolist()))
    return similarity

def fingerprint_hash(featureSet, numberOfBits=NUMBER_OF_BITS): # get fingerprint by hashing graph features
    hashVector = np.array([0.0 for bit in range(numberOfBits)])
    for entry in featureSet:
        rank = entry[-1]
        rank_choices = [-1.0*rank, rank]
        entry_vector = np.array([rank_choices[rd.randint(0,1)] for bit in range(numberOfBits)])
        hashVector = hashVector + entry_vector
    fingerPrint = map(lambda x: 1 if x>0 else 0, hashVector.tolist())
    return fingerPrint

def phi_func(graph): # extract feature from graph
    featureSet = set()
    pageRank = graph.pagerank()
    for vertex, rank in enumerate(pageRank): # quality of all vertices
        featureSet.add((vertex, rank))
    edgelist = graph.get_edgelist()
    for edge in edgelist: # quality of all edges
        rank = pageRank[edge[0]] / graph.vs[edge[0]].degree(type="out")
        featureSet.add((edge, rank))
    return featureSet

def calculate_anomaly(similarityTimeSeries):
    sumM = 0.0
    Num = len(similarityTimeSeries)
    copySimilarityTimeSeries = list(similarityTimeSeries)
    copySimilarityTimeSeries.sort()
    simMedian = copySimilarityTimeSeries[Num/2] # get similarity median
    for index in range(Num-1): # calculate moving range average M
        sumM = sumM + np.abs(similarityTimeSeries[index+1] - similarityTimeSeries[index])
    avgM = sumM / float(Num-1)
    upperThreshold = simMedian + avgM*3.0
    lowerThreshold = simMedian - avgM*3.0
    return upperThreshold, lowerThreshold

def save_timeseries(similarityTimeSeries, dir_of_data_files):
    fname = dir_of_data_files.strip('\/').split('\/')[-1] + "_time_series.txt"
    try:
        fp = open(fname, 'w')
        for value in similarityTimeSeries:
            fp.write(str(value)+'\n')
        fp.close()
    except:
        print "File cannot open to write!"
        sys.exit(EXIT_FAILURE)
    print "Save File Done!"

def plot_timeseries(similarityTimeSeries, anomalyUpperThres, anomalyLowerThres, dir_of_data_files):
    fname = dir_of_data_files.strip('\/').split('\/')[-1] + "_time_series.png"
    size = len(similarityTimeSeries)
    timeseries = range(size)
    plt.plot(timeseries, similarityTimeSeries, 'bo')
    plt.plot(timeseries, [anomalyLowerThres for i in range(size)], 'r--')
    plt.plot(timeseries, [anomalyUpperThres for i in range(size)], 'r--')
    plt.ylim([0.0, 1.0])
    plt.xlim([0, size])
    plt.ylabel("Similarity")
    plt.xlabel("Time Series")
    plt.savefig(fname)
    print "Plot Done!"


if __name__ == "__main__":
    main()