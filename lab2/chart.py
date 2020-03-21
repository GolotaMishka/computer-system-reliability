from collections import defaultdict


class Chart:

    def __init__(self, vertices):
        self.chart = defaultdict(list)
        self.V = vertices
        self.all_pathes = list()

    def addEdge(self, u, v):
        self.chart[u].append(v)

    def printPathsRecursive(self, u, d, visited, path):
        visited[u] = True
        path.append(u)

        if u == d:
            a = path.copy()
            self.all_pathes.append(a)
            print(path)
        else:
            for i in self.chart[u]:
                if visited[i] == False:
                    self.printPathsRecursive(i, d, visited, path)

        path.pop()
        visited[u] = False

    def printAllPaths(self, s, d):
        visited = [False] * (self.V)
        path = []
        self.printPathsRecursive(s, d, visited, path)

    def getAllPaths(self):
        return self.all_pathes

