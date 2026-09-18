# AI 智能教师 Agent 路由评测

生成时间：2026-09-18T13:35:26+08:00

## 总体指标

| 样本 | 工具选择 | 参数有效率 | 任务完成率 | 引用正确率 | 忠实度代理 | 平均延迟 | P95 延迟 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 100.0% | 100.0% | 100.0% | 90.0% | 97.2% | 3.41s | 5.09s |

Token 用量：输入 90448，输出 11006。

> 引用正确率检查引用编号是否来自合成工具；忠实度是执行轨迹代理指标，不是 LLM-as-a-Judge 的语义事实评分。成本仅在配置每百万 Token 单价后估算。

## 明细

| 输入 | 预期工具 | 实际工具 | 参数 | 耗时 | 结果 |
|---|---|---|---|---:|---|
| 二次函数的顶点坐标公式是什么？ | search_textbook | search_textbook | 有效 | 4.53s | 通过 |
| 等差数列的通项公式怎么写？ | search_textbook | search_textbook | 有效 | 3.71s | 通过 |
| 请根据二次函数给我出一道基础题，先不要答案。 | generate_exercise | generate_exercise | 有效 | 2.77s | 通过 |
| 给我来一道稍难的函数选择题，并附答案。 | generate_exercise | generate_exercise | 有效 | 3.69s | 通过 |
| 题目：求 y=x²-2x+1 的顶点。我的答案是 (1,0)，请批改。 | grade_answer | grade_answer | 有效 | 3.67s | 通过 |
| 题目：2x+1=5。我的解答是 x=3，帮我看看哪里错了。 | grade_answer | grade_answer | 有效 | 4.17s | 通过 |
| 你好，你是谁？ | 不调用工具 | 不调用工具 | 有效 | 1.81s | 通过 |
| 今天学习有点累，鼓励我一下。 | 不调用工具 | 不调用工具 | 有效 | 2.27s | 通过 |
| 帮我制定一个今晚复习数学的简单安排。 | 不调用工具 | 不调用工具 | 有效 | 3.27s | 通过 |
| 再来一道。 | generate_exercise | generate_exercise | 有效 | 3.56s | 通过 |
| 这道太简单了，难一点。 | generate_exercise | generate_exercise | 有效 | 3.07s | 通过 |
| 我的答案是 (1, 0)。 | grade_answer | grade_answer | 有效 | 3.90s | 通过 |
| 勾股定理的内容和适用条件是什么？ | search_textbook | search_textbook | 有效 | 4.69s | 通过 |
| 正弦定理什么时候使用？ | search_textbook | search_textbook | 有效 | 5.09s | 通过 |
| 导数在几何上表示什么？ | search_textbook | search_textbook | 有效 | 4.66s | 通过 |
| 古典概型的概率公式是什么？ | search_textbook | search_textbook | 有效 | 3.45s | 通过 |
| 椭圆的焦点和离心率怎样定义？ | search_textbook | search_textbook | 有效 | 4.13s | 通过 |
| 等比数列前 n 项和公式怎么推导？ | search_textbook | search_textbook | 有效 | 4.51s | 通过 |
| 三角函数的最小正周期怎么求？ | search_textbook | search_textbook | 有效 | 5.43s | 通过 |
| 排列与组合有什么区别？ | search_textbook | search_textbook | 有效 | 4.44s | 通过 |
| 出一道概率填空题。 | generate_exercise | generate_exercise | 有效 | 2.77s | 通过 |
| 用数列知识测试我一下，不要答案。 | generate_exercise | generate_exercise | 有效 | 3.72s | 通过 |
| 把刚才的题换成选择题。 | generate_exercise | generate_exercise | 有效 | 3.38s | 通过 |
| 给我一道中等难度的三角函数练习。 | generate_exercise | generate_exercise | 有效 | 3.50s | 通过 |
| 题目：3x-4=8。我的答案是 x=4。 | grade_answer | grade_answer | 有效 | 4.14s | 通过 |
| 我算出来是 x=2，帮我批改。 | grade_answer | grade_answer | 有效 | 4.55s | 通过 |
| 刚才那题我不会，我的过程是先配方。 | grade_answer | grade_answer | 有效 | 3.25s | 通过 |
| 题目：sin 30° 等于多少？我写的是 1/2。 | grade_answer | grade_answer | 有效 | 3.42s | 通过 |
| 我的解题过程写完了，请给分。 | 不调用工具 | 不调用工具 | 有效 | 1.69s | 通过 |
| 谢谢老师。 | 不调用工具 | 不调用工具 | 有效 | 2.16s | 通过 |
| 你能做什么？ | 不调用工具 | 不调用工具 | 有效 | 2.50s | 通过 |
| 我总是粗心，安慰我一下。 | 不调用工具 | 不调用工具 | 有效 | 2.66s | 通过 |
| 帮我安排一个 25 分钟的番茄钟。 | 不调用工具 | 不调用工具 | 有效 | 2.18s | 通过 |
| 学了很久，我应该休息多久？ | 不调用工具 | 不调用工具 | 有效 | 2.35s | 通过 |
| 我今天应该先复习哪一科？ | 不调用工具 | 不调用工具 | 有效 | 2.30s | 通过 |
| 再见，明天继续。 | 不调用工具 | 不调用工具 | 有效 | 1.37s | 通过 |

## 失败案例

本轮没有失败案例；历史失败与修复记录见 `docs/failure_cases.md`。
