import heapq

class median_test:

    def __init__(self):
        self.heap = [],[]

    def addNum(self, number):
        small, large = self.heap
        heapq.heappush(small, -heapq.heappushpop(large, number))
        if len(large) < len(small):
            heapq.heappush(large, -heapq.heappop(small))

    def find(self):
        small, large = self.heap
        if len(large) > len(small):
            return float(large[0])
        else:
            return (large[0] - small[0])/2.0

    def show(self):
        small, large = self.heap
        for i in small:
            print(i)
        for j in large:
            print(j)
