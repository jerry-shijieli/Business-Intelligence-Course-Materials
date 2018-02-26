from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    indexCount = []
    posCount = []
    negCount = []
    index = 0
    for term in counts:
        indexCount.append(index)
        posCount.append(0)
        negCount.append(0)
        for count in term:
            if count[0] == "positive":
                posCount[index] += count[1]
            elif count[0] == "negative":
                negCount[index] += count[1]
        index += 1
    posLabel = "positive"
    negLabel = "negative"
    plt.plot(indexCount, posCount, 'o', ls='-', color='b', label = posLabel)
    plt.plot(indexCount, negCount, 'o', ls='-', color='g', label = negLabel)
    plt.xlim([min(indexCount)-1, max(indexCount)+1])
    plt.ylim(0, max(max(posCount), max(negCount))*1.4)
    plt.xlabel('Time step')
    plt.ylabel('Word count')
    legend = plt.legend(loc='upper left')
    plt.savefig('plot.png')



def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    fdata = open(filename, "r")
    txtData = fdata.readlines()
    fdata.close()
    words = []
    for line in txtData:
        words.append(line.strip())
    return words



def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    #tweets.pprint()
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    words = tweets.flatMap(lambda tweet: tweet.split(" "))
    def classWord(word):
        if word in pwords:
            return ("positive", 1)
        elif word in nwords:
            return ("negative", 1)
        else :
            return ("positive", 0)
    classifications = words.map(lambda x: classWord(x))
    wordCounts = classifications.reduceByKey(lambda x,y: x + y)
    def updateFunc(newVal, runningCount):
        if runningCount is None:
            runningCount = 0
        return sum(newVal, runningCount)
    totalCounts = classifications.updateStateByKey(updateFunc)
    totalCounts.pprint()
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    wordCounts.foreachRDD(lambda t, rdd: counts.append(rdd.collect()))
    # stream computation for assigned time interval
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)
    # return counts for plot
    return counts


if __name__=="__main__":
    main()
