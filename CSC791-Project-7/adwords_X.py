#!/usr/bin/env python
# unityID: sli41
import sys
import csv
import random
import numpy as np

EXIT_FAILURE = -1
EXIT_SUCCESS = 0
RANDOM_SEED = 2017
NUMBER_OF_QUERY_PERMUTATIONS = 100


def main():
    if len(sys.argv) != 2:  # select matching algorithm from command line argument
        print "Usage: python adwords.py <Matching-Algorithm-Name>"
        sys.exit(EXIT_FAILURE)
    algorithmID = sys.argv[1]
    # read bidder and query data
    query_bidder_price_dict, bidder_budget_dict = load_bidder_profile() # bidder profile
    queries_list = load_queries() # use strip() to remove '\n' for each entry
    optimalRevenue = sum(bidder_budget_dict.values())
    # choose the matching algorithm
    algorithm_dict = {"greedy": greedy_matching, "msvv": msvv_matching, "balance": balance_matching}
    if not algorithm_dict.has_key(algorithmID):
        print "Please enter the correct algorithm name!"
        sys.exit(EXIT_FAILURE)
    else:
        matching_algorithm = algorithm_dict[algorithmID]
    # execute matching algorithm to compute average value of total revenue for 100 random query sequence
    random.seed(RANDOM_SEED)
    total_revenue_record = list()
    for count in range(NUMBER_OF_QUERY_PERMUTATIONS):
        random.shuffle(queries_list)
        totalRevenue = matching_algorithm(queries_list, query_bidder_price_dict, bidder_budget_dict)
        total_revenue_record.append(totalRevenue)
    averageTotalRevenue = np.mean(total_revenue_record)
    # evaluate the algorithm performance
    evaluation_stats(averageTotalRevenue, optimalRevenue)
    return EXIT_SUCCESS


def load_bidder_profile(fname = "bidder_dataset.csv"): # load the advertise bidder profile (ID, query keyword, bid value, budget)
    bidder_profile_list = list() # raw data of bidder profile
    query_bidder_price_dict = dict() # dict {query : (bidderID, bidValue)}
    bidder_budget_dict = dict() # dict {bidderID: bidderBudget}
    try:
        fp = open(fname, 'rt')
        table = csv.reader(fp)
        for row in table:
            bidder_profile_list.append(row)
        fp.close()
        header = bidder_profile_list[0]
        for entry in bidder_profile_list[1:]:
            id =  int(entry[0])
            keyword = entry[1]
            bidVal = float(entry[2])
            budget = entry[3]
            if query_bidder_price_dict.has_key(keyword): # register query -> (bidder, bid value)
                query_bidder_price_dict[keyword].append((id, bidVal))
            else:
                query_bidder_price_dict[keyword] = [(id, bidVal)]
            if budget.isdigit(): # register bidder initial budget
                bidder_budget_dict[id] = float(budget)
    except IOError:
        print "Error: File does not exist!"
        sys.exit(EXIT_FAILURE)
    except IndexError:
        print "Error: Empty file!"
        sys.exit(EXIT_FAILURE)
    return query_bidder_price_dict, bidder_budget_dict


def load_queries(fname = "queries.txt"): # load the search queries
    try:
        fp = open(fname, 'r')
        queries = fp.readlines()
        fp.close()
    except IOError:
        print "Error: File does not exist!"
        sys.exit(EXIT_FAILURE)
    return queries


def greedy_matching(queries_list, query_bidder_price_dict, bidder_budget_dict): # Greedy algorithm
    local_queries_list = list(queries_list)
    local_query_bidder_price_dict = dict(query_bidder_price_dict)
    local_bidder_budget_dict = dict(bidder_budget_dict)
    local_totalRevenue = 0.0
    for query in local_queries_list:
        if not local_query_bidder_price_dict.has_key(query.strip()):
            continue
        bidder_price_list = local_query_bidder_price_dict[query.strip()]
        # check budget
        bidder_list = [entry[0] for entry in bidder_price_list if entry[1]<= local_bidder_budget_dict[entry[0]]] # filter valid bidders
        totalBudget = sum([local_bidder_budget_dict[bidder] for bidder in bidder_list])
        if totalBudget == 0: # check for available budget
            continue
        # find matched bidder
        bidder_price_list = [bidder_price for bidder_price in bidder_price_list if bidder_price[0]
                             in bidder_list] # update by valid bidders
        matched_bidder_profile = max(bidder_price_list, key=lambda x: x[1]) # matched with highest bid value
        matched_bidder_id = matched_bidder_profile[0]
        matched_bid_price = matched_bidder_profile[1]
        local_bidder_budget_dict[matched_bidder_id] -= matched_bid_price
        local_totalRevenue += matched_bid_price
    return local_totalRevenue

def msvv_matching(queries_list, query_bidder_price_dict, bidder_budget_dict): # MSVV algorithm
    local_queries_list = list(queries_list)
    local_query_bidder_price_dict = dict(query_bidder_price_dict)
    local_bidder_budget_dict = dict(bidder_budget_dict)
    local_totalRevenue = 0.0
    for query in local_queries_list:
        if not local_query_bidder_price_dict.has_key(query.strip()):
            continue
        bidder_price_list = local_query_bidder_price_dict[query.strip()]
        # check budget
        bidder_list = [entry[0] for entry in bidder_price_list if entry[1]<=local_bidder_budget_dict[entry[0]]] # filter valid bidders
        totalBudget = sum([local_bidder_budget_dict[bidder] for bidder in bidder_list])
        if totalBudget == 0: # check for available budget
            continue
        # find mathced bidder
        bidder_price_budget_list = [(bidder_price[0], bidder_price[1], local_bidder_budget_dict[bidder_price[0]], bidder_budget_dict[bidder_price[0]]) for bidder_price in bidder_price_list if bidder_price[0] in bidder_list] # update by valid bidders and add required budget info as (id, bid value, unspent budget, original budget)
        matched_bidder_profile = max(bidder_price_budget_list, key=lambda x: x[1]*(1-np.exp((x[3]-x[2])/x[3]-1))) # matched with highest scaled bid = bid_value * (1 - exp(spent_fraction - 1))
        matched_bidder_id = matched_bidder_profile[0]
        matched_bid_price = matched_bidder_profile[1]
        local_bidder_budget_dict[matched_bidder_id] -= matched_bid_price
        local_totalRevenue += matched_bid_price
    return local_totalRevenue

def balance_matching(queries_list, query_bidder_price_dict, bidder_budget_dict): # Balance algorithm
    local_queries_list = list(queries_list)
    local_query_bidder_price_dict = dict(query_bidder_price_dict)
    local_bidder_budget_dict = dict(bidder_budget_dict)
    local_totalRevenue = 0.0
    for query in local_queries_list:
        if not local_query_bidder_price_dict.has_key(query.strip()):
            continue
        bidder_price_list = local_query_bidder_price_dict[query.strip()]
        # check budget validation
        bidder_list = [entry[0] for entry in bidder_price_list if entry[1] <= local_bidder_budget_dict[entry[0]]] # filter valid bidders
        totalBudget = sum([local_bidder_budget_dict[bidder] for bidder in bidder_list])
        if totalBudget == 0: # check for available budget
            continue
        # find matched bidder
        bidder_budget_list = [(bidder_id, bidder_budget_dict[bidder_id]) for bidder_id in bidder_list] # construct the bidder budget list for valid bidders
        matched_bidder_profile = max(bidder_budget_list, key=lambda x: x[1]) # matched with highest unspent budget
        bidder_price_list = [bidder_price for bidder_price in bidder_price_list if bidder_price[0]
                             in bidder_list] # update by valid bidders
        bidder_price_dict = dict(bidder_price_list)
        matched_bidder_id = matched_bidder_profile[0]
        matched_bid_price = bidder_price_dict[matched_bidder_id]
        local_bidder_budget_dict[matched_bidder_id] -= matched_bid_price
        local_totalRevenue += matched_bid_price
    return local_totalRevenue

def evaluation_stats(totalRevenue, optimalRevenue): # compute and print the total revenue and competitive ratio
    try:
        competitive_ratio = totalRevenue / optimalRevenue
        print "revenue \t competitive ratio"
        print "%.2lf \t %.4lf"%(totalRevenue, competitive_ratio)
    except ZeroDivisionError:
        print "Error: divided by zero!"

if __name__ == "__main__":
    main()