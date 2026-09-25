"""
Day 01 摸底题复核脚本（agent 生成，用于验证 4 道题的**正确性与复杂度**，不是题解）

跑法：python algorithms/_day01_check.py
"""
import importlib.util
import random
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASS, FAIL = 0, 0


def load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solution


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}\n        got ={got!r}\n        want={want!r}")


def norm(groups):
    """把分组结果归一化成可比较的集合集合"""
    return sorted(sorted(g) for g in groups)


print("=" * 78)
print("#1 两数之和  ->  排序 + 双指针")
print("=" * 78)
S1 = load("1.两数之和/1.两数之和.py", "p1")
for nums, tgt, want in [
    ([2, 7, 11, 15], 9, [0, 1]),
    ([3, 2, 4], 6, [1, 2]),
    ([3, 3], 6, [0, 1]),
    ([-3, 4, 3, 90], 0, [0, 2]),
    ([2, 5, 5, 11], 10, [1, 2]),
]:
    r = S1().twoSum(nums, tgt)
    check(f"nums={nums} target={tgt}", r, want)

print("  --- 越界探测（题目保证有解，所以这不算错，但看它坏成什么样）---")
r = S1().twoSum([1, 2], 5)
print(f"  [!!] nums=[1,2] target=5 无解 -> 返回 {r}（应为 [] 或抛异常）")

print()
print("=" * 78)
print("#3 无重复字符的最长子串  ->  list 当队列的滑动窗口")
print("=" * 78)
S3 = load("3.无重复字符的最长子串/3.无重复字符的最长子串.py", "p3")
for s, want in [
    ("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0), (" ", 1),
    ("dvdf", 3), ("abba", 2), ("tmmzuxt", 5), ("au", 2), ("ohvhjdml", 6),
]:
    check(f"s={s!r}", S3().lengthOfLongestSubstring(s), want)

print("  --- 复杂度实测：字母表不设限时（题目限定了字母/数字/符号，所以这是压力测试）---")
alphabet_big = "".join(chr(0x4E00 + i) for i in range(60000))  # 全不重复的字符
for n in (10000, 20000, 40000):
    s = alphabet_big[:n]  # 所有字符互不相同 -> 窗口一路涨到 n
    t0 = time.perf_counter()
    S3().lengthOfLongestSubstring(s)
    dt = time.perf_counter() - t0
    print(f"  n={n:>6}  用时 {dt:7.3f}s")
print("  对照：题目限定字母表 ~95 个字符，窗口长度天然 <=95，所以真实用例能过")

print()
print("=" * 78)
print("#49 字母异位词分组  ->  26 位计数向量做 key")
print("=" * 78)
S49 = load("49.字母异位词分组/49.字母异位词分组.py", "p49")
check("官方示例", norm(S49().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])),
      norm([["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]))
check("空串", norm(S49().groupAnagrams([""])), norm([[""]]))
check("单字符", norm(S49().groupAnagrams(["a"])), norm([["a"]]))
# 关键回归：验证 "," 分隔符真的修掉了 "111" 歧义
a = "abbbbbbbbbbb"   # 1 个 a, 11 个 b
b = "aaaaaaaaaaab"   # 11 个 a, 1 个 b
got = norm(S49().groupAnagrams([a, b]))
check("歧义回归（1个a+11个b vs 11个a+1个b 必须分到两组）", len(got), 2)
t0 = time.perf_counter()
S49().groupAnagrams(["".join(random.choices(string.ascii_lowercase, k=100)) for _ in range(10000)])
print(f"  规模压测 10^4 词 x 100 字符：{time.perf_counter() - t0:.3f}s")

print()
print("=" * 78)
print("#15 三数之和  ->  排序 + 双指针 + 三重去重")
print("=" * 78)
S15 = load("15.三数之和/15.三数之和.py", "p15")
for nums, want in [
    ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
    ([0, 1, 1], []),
    ([0, 0, 0], [[0, 0, 0]]),
    ([], []),
    ([1, 2, 3], []),
    ([-2, 0, 0, 2, 2], [[-2, 0, 2]]),
    ([-4, -2, -2, -2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6], [[-4, -2, 6], [-4, 0, 4], [-4, 1, 3], [-4, 2, 2], [-2, -2, 4], [-2, 0, 2]]),
]:
    check(f"nums={nums}", norm(S15().threeSum(nums)), norm(want))
t0 = time.perf_counter()
S15().threeSum([random.randint(-10**5, 10**5) for _ in range(3000)])
print(f"  规模压测 n=3000（题目上界）：{time.perf_counter() - t0:.3f}s")

print()
print("=" * 78)
print(f"结果：PASS={PASS}  FAIL={FAIL}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
