"""
华为机考 / ACM 模式 快速 IO 模板
=================================
**核心代码模式（LeetCode）与 ACM 模式的 4 个硬差别**：
  1. 没有函数签名给你——输入自己读，输出自己 print
  2. 输入格式不固定——多组用例？第一行 T？一行几个数？带引号？全靠你判断
  3. Python 不快读会 TLE——用 sys.stdin.buffer.read().split()
  4. 循环里 print 会 TLE——结果收集到 list，最后一次 '\\n'.join() 输出

**搬运规则**：把 LeetCode 里 class Solution 的**核心逻辑**搬进 solve()，
删掉 class、删掉 self、删掉类型注解（可选）。不要改算法。

**机考加分项**（面试官会看的）：
  - 顶部 sys.setrecursionlimit(10**6)（涉及 DFS 时必需）
  - 不用 input()，用 sys.stdin.buffer.read()
  - 输出最后一次性 write
"""
import sys


# ============================================================
# 骨架一：单组用例，第一行 n，第二行 n 个整数
#   输入:  5
#          1 2 3 4 5
# ============================================================
def solve_single(data):
    it = iter(data)
    n = int(next(it))
    nums = [int(next(it)) for _ in range(n)]
    return str(sum(nums))


# ============================================================
# 骨架二：第一行 T 表示组数（华为最常见）
#   输入:  2
#          3
#          1 2 3
#          4
#          1 1 1 1
# ============================================================
def solve_T(data):
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        n = int(next(it))
        nums = [int(next(it)) for _ in range(n)]
        out.append(str(sum(nums)))       # 每组一行
    return "\n".join(out)


# ============================================================
# 骨架三：多组用例，读到 EOF 为止（没有 T，最阴险的一种）
#   输入:  3
#          1 2 3
#          2
#          7 8
# ============================================================
def solve_eof(data):
    it = iter(data)
    out = []
    while True:
        try:
            n = int(next(it))
        except StopIteration:
            break
        nums = [int(next(it)) for _ in range(n)]
        out.append(str(sum(nums)))
    return "\n".join(out)


# ============================================================
# 骨架四：一行不定长的数字（没有 n，整行读完）
#   输入:  1 2 3 4 5
# ============================================================
def solve_one_line(data):
    nums = list(map(int, data))          # data 里就是这一行切出来的 token
    return str(max(nums))


# ============================================================
# 骨架五：字符串/数组带引号（LeetCode 风格的用例会这样给）
#   输入:  "abcabcbb"
#   或:    ["eat","tea","tan"]
# ============================================================
def solve_quoted(data):
    # sys.stdin.buffer.read().split() 之后，引号还粘在 token 上，要先剥
    s = data[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip().strip('"').strip("'")
    return str(len(s))


# ============================================================
# 骨架六：二维矩阵
#   输入:  3 4
#          1 2 3 4
#          5 6 7 8
#          9 1 2 3
# ============================================================
def solve_matrix(data):
    it = iter(data)
    n, m = int(next(it)), int(next(it))
    grid = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    return str(sum(sum(r) for r in grid))


# ============================================================
# main：按题目实际格式，把上面某个骨架的解析逻辑拷进来
# ============================================================
def main():
    sys.setrecursionlimit(10**6)                    # DFS / 递归题必需
    data = sys.stdin.buffer.read().split()          # 快读，别用 input()
    if not data:
        return

    # ---- 按题目格式选择解析方式（下面三选一，其余删掉）----
    ans = solve_single(data)      # 第一行 n + 第二行 n 个数
    # ans = solve_T(data)         # 第一行 T
    # ans = solve_eof(data)       # 读到 EOF

    sys.stdout.write(ans + "\n")                    # 一次性写出，别在循环里 print


if __name__ == "__main__":
    main()
