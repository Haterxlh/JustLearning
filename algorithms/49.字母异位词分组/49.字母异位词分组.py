#
# @lc app=leetcode.cn id=49 lang=python3
#
# [49] 字母异位词分组
#
# https://leetcode.cn/problems/group-anagrams/description/
#
# algorithms
# Medium (69.37%)
# Likes:    2842
# Dislikes: 0
# Total Accepted:    1.8M
# Total Submissions: 2.6M
# Testcase Example:  '["eat","tea","tan","ate","nat","bat"]'
#
# 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
# 
# 
# 
# 示例 1:
# 
# 
# 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# 
# 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
# 
# 解释：
# 
# 
# 在 strs 中没有字符串可以通过重新排列来形成 "bat"。
# 字符串 "nat" 和 "tan" 是字母异位词，因为它们可以重新排列以形成彼此。
# 字符串 "ate" ，"eat" 和 "tea" 是字母异位词，因为它们可以重新排列以形成彼此。
# 
# 
# 
# 示例 2:
# 
# 
# 输入: strs = [""]
# 
# 输出: [[""]]
# 
# 
# 示例 3:
# 
# 
# 输入: strs = ["a"]
# 
# 输出: [["a"]]
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= strs.length <= 10^4
# 0 <= strs[i].length <= 100
# strs[i] 仅包含小写字母
# 
# 
#

# @lc code=start
from collections import defaultdict
class Solution:
    def hashs(self, s: str) -> str:
        hs = [0 for _ in range(ord("z") - ord("a") + 1)]
        for ch in s:
            idx = ord(ch) - ord("a")
            hs[idx] += 1
        return ",".join(map(str, hs))
    
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        res = []
        maps = defaultdict(list)

        for s in strs:
            h = self.hashs(s)
            if h not in maps.keys():
                maps[h] = [s]
            else:
                maps[h].append(s)
        res = maps.values()
        return list(res)
# @lc code=end

