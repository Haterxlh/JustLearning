import sys
import heapq

data = sys.stdin.read().split()
it = iter(data)

n, m, k = int(next(it)), int(next(it)), int(next(it))
a = [-int(next(it)) for _ in range(n)]

heapq.heapify(a)

for i in range(m):
    base = (i + 1) * k
    # 每天取出最多的鸡
    maxv = heapq.heappop(a)
    # 计算剩下的鸡数量
    maxv = (base - maxv) // 2
    # 更新堆
    heapq.heappush(a, (-maxv + base))

res = -sum(a) + n * k * m
print(res)