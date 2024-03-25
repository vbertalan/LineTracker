import LineTracker as lt
import os

lc = lt.LineTracker()
token = "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"
lc.cluster(logfile = "/home/vbertalan/Downloads/LineTracker/LineTracker/data-10.json", num_clusters = 2, token=token)
#lc.cluster(logfile = "/home/vbertalan/Downloads/LineTracker/LineTracker/ctm-cmcp_sim", 
#           num_clusters = 10, token = token)