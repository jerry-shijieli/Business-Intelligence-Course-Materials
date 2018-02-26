#!/usr/bin/env python
import sys
import numpy as np
import igraph
import csv

MAX_NUM_OF_ITERATION = 15
EXIT_FAILURE = -1
EXIT_SUCCESS = 0

def main():
    if (len(sys.argv) != 2):
        print "Usage: python sac1.py <value_of_alpha>"
        sys.exit(EXIT_FAILURE)
    alpha = float(sys.argv[1]) # determine alpha from the command-line argument
    print "alpha =", alpha
    attrList, edgeList = load_data() # read the data of node attributes and edges
    g_base = init_graph(len(attrList), edgeList) # initialize the graph
    igraph.summary(g_base)
    print "phase one"
    g_phase1 = phase_one(g_base, attrList, alpha) # cluster nodes into communities in phase1
    print "phase two"
    communityMemberList = phase_two(g_phase1, attrList, alpha) # cluster community meta-nodes into communities in phase2
    print "save communities"
    saveCommunity(communityMemberList, alpha) # save communities into text files
    print "done"

def load_data():
    fnEdgeList = "./data/fb_caltech_small_edgelist.txt"
    fnAttrList = "./data/fb_caltech_small_attrlist.csv"
    attrList = list()
    edgeList = list()
    try:
        fpEdges = open(fnEdgeList, 'r')
        for line in fpEdges: # read in all edges (v1, v2)
            entry = line.split()
            edge = (int(entry[0]), int(entry[1]))
            edgeList.append(edge)
        fpEdges.close()
        fpAttr = open(fnAttrList, 'r')
        table = csv.reader(fpAttr)
        for index, row in enumerate(table): # read in all attributes
            if index == 0:
                header = row
            else:
                attrVec = [int(term) for term in row]
                attrList.append(attrVec)
        fpAttr.close()
    except IOError:
        print "File does not exists!"
        sys.exit(EXIT_FAILURE)
    return attrList, edgeList

def init_graph(numberOfVertices, edgeList):
    graph = igraph.Graph(numberOfVertices, edgeList)
    graph.es["weight"] = 1 # initial weight of edge
    return graph

def dQ_newman(node, graph, communityID, adjMatrix):
    deltaQ = 0.0
    vertexList = graph.vs.select(community_eq=communityID) # all vertices in the same community
    numberOfEdges = len(graph.es)
    dx = graph.vs[node].degree()
    for vertex in vertexList: # gain of Newman's modularity over all nodes in the community
        dv = vertex.degree()
        deltaQ += (adjMatrix[vertex.index, node] - (dx*dv)/(2.0*numberOfEdges)) / (2.0*numberOfEdges)
    return deltaQ

def dQ_attr(node, graph, simMatrix, communityID):
    deltaQ = 0.0
    vertexList = graph.vs.select(community_eq=communityID) # all vertices in the same community
    for vertex in vertexList: # gain of similarity over all nodes in the community
        deltaQ += simMatrix[node, vertex.index]
    deltaQ /= ((len(vertexList)+0.00001)**2) # normalization by community size
    return deltaQ

def phase_one(graph, attrList, alpha):
    numberOfVertices = len(graph.vs)
    numberOfCommunities = numberOfVertices
    graph.vs["community"] = [index for index in range(numberOfCommunities)] # initialize community
    adjMatrix = graph.get_adjacency("weight")
    simMatrix = [] # similarity matrix of vertex attributes
    for i in range(numberOfVertices):
        row = []
        vec1 = attrList[i]
        for j in range(numberOfVertices):
            vec2 = attrList[j]
            similarity = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            row.append(similarity)
        simMatrix.append(row)
    simMatrix = np.matrix(simMatrix)
    stopFlag = False
    countIteration = 0
    while ((not stopFlag) and countIteration<MAX_NUM_OF_ITERATION): # stop conditions
        stopFlag = True
        countIteration += 1
        for i in range(numberOfVertices):
            gainDict = dict()
            originCommunity = graph.vs[i]["community"]
            graph.vs[i]["community"] = -1
            for j in range(numberOfVertices):
                if True:
                    communityID = graph.vs[j]["community"]
                    dQ = alpha*dQ_newman(i, graph, communityID, adjMatrix) + (1-alpha)*dQ_attr(i, graph, simMatrix, communityID)/numberOfCommunities
                    if dQ > 0: # check positive gain
                        gainDict.update({dQ: j})
            if len(gainDict) > 0: # change community to that of max Q gain
                vertex = gainDict[max(gainDict)]
                max_communityID = graph.vs[vertex]["community"]
                graph.vs[i]["community"] = max_communityID
                stopFlag = False
            else:
                graph.vs[i]["community"] = originCommunity
    resultGraph = graph
    return resultGraph

def phase_two(graph, attrList, alpha):
    communitySet = list(set(graph.vs["community"]))
    numberOfCommunities = len(communitySet)
    metaGraph = igraph.Graph(numberOfCommunities) # meta-graph
    metaAttrList = list() # attribute of meta-nodes
    communityMap = dict()
    adjMatrix = graph.get_adjacency("weight")
    metaEdges = dict()
    for indexA, communityID in enumerate(communitySet):
        vertexListA = graph.vs.select(community_eq=communityID)
        communityAttrList = list()
        for vertex in vertexListA:
            communityAttrList.append(attrList[vertex.index])
        communityAttr = np.array(communityAttrList).mean(axis=0).tolist()
        metaAttrList.append(communityAttr)
        metaGraph.vs[indexA]["community"] = indexA
        communityMap.update({indexA: communityID})
        for communityIDB in (communitySet[indexA:]):
            if communityID != communityIDB:
                indexB = communitySet.index(communityIDB)
                vertexListB = graph.vs.select(community_eq=communityIDB)
                for vA in vertexListA:
                    for vB in vertexListB:
                        if metaEdges.has_key((indexA, indexB)): # meta-edge
                            metaEdges[(indexA, indexB)] += adjMatrix[vA.index, vB.index]
                        else:
                            metaEdges[(indexA, indexB)] = adjMatrix[vA.index, vB.index]
    metaGraph.add_edges(metaEdges.keys())
    metaGraph.es["weight"] = metaEdges.values()
    finalGraph = phase_one(metaGraph, metaAttrList, alpha) # reapply phase one
    finalCommunitySet = set(finalGraph.vs["community"])
    finalCommunityMap = dict()
    for index, fc in enumerate(finalCommunitySet): # retrieve members of final communities
        finalCommunityMap[index] = list()
        finalVertexList = finalGraph.vs.select(community_eq=fc)
        for finalVertex in finalVertexList:
            retrieveVertexList = graph.vs.select(community_eq=communityMap[finalVertex.index])
            finalCommunityMap[index].extend([vertex.index for vertex in retrieveVertexList])
    return finalCommunityMap

def saveCommunity(communityMemberList, alpha):
    if alpha == 0:
        fname = "communities_0.txt"
    elif alpha == 1:
        fname = "communities_1.txt"
    else:
        fname = "communities_5.txt"
    try:
        fp = open(fname, 'w')
        for communityMembers in communityMemberList.values():
            fp.write(str(communityMembers).strip(r'\[\]'))
            fp.write('\n')
        fp.close()
    except IOError:
        print "File fails to open."
        sys.exit(EXIT_FAILURE)

if __name__ == "__main__":
    main()