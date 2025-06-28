from typing import *

def notit(j):
    L = [0,1,2]
    L.pop(j)
    return L

def ninjaTraining(n: int, points: List[List[int]]) -> int:

    # Write your code here.
    dp = []
    for k in range(n):
        dp.append([-1]*3)
    dp[0] = points[0]
    for i in range(1,n):
        for j in range(3):
            p =  max(dp[i-1][notit(j)[0]],dp[i-1][notit(j)[1]])+points[i][j]
            dp[i][j] = p
    return max(dp[-1]) 

print(ninjaTraining(3,[[1,2,5],[3,1,1],[3,3,3]]))