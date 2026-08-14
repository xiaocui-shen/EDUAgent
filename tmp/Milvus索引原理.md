# 向量库索引原理

## 索引的作用

索引就是数据的目录，其作用是查询快。向量数据库索引的用途：向量相似度检索（图片、文本、Embedding 搜索）。我们平时存向量（图片、文本、语音转成的数字数组），需求是：给一个查询向量，快速找出库里最相似的 N 个向量（近似最近邻搜索 ANN）

## 向量索引

- 按数据量、内存资源

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjQxYzY3NjA5YmZkN2ExZDhhNzkzY2U3NjQ4Nzk1YzRfNTUyY2VlYmViMWFhMmRmODJiOTljOTkwMzA4Yjc5YzhfSUQ6NzY0NjE0MDEyMDc1NTYwNDY4N18xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

- 按数据量

    - \<10 万：FLAT

    - 10 万–100 万：IVF\_FLAT / HNSW / ANNOY（低维 \+ 内存紧）

    - 100 万–亿级、内存一般：IVF\_SQ8 / RNSG（高维）

    - 亿级 \+、内存紧张：IVF\_PQ / DISKANN（超大规模）

    - 线上高并发、低延迟：HNSW



### FLAT

暴力搜索，适合小型百万级数据集，搜索结果完全准确但效率低



### IVF\_FLAT

- 正向（正排）逻辑和倒排逻辑

    - 正向逻辑（Normal）：当你手里有一个向量时，你去计算它离哪个簇中心（分区）最近，然后把它归到那个簇里。*逻辑流向：* 向量 ➡️ 簇中心

    - 倒排逻辑（Inverted）：IVF 在建立索引时，反其道而行之。它是以“簇中心”为主体，手里拿着一份清单，清单上写满了“哪些向量是归我管的”。*逻辑流向：* 簇中心 ➡️ 一堆向量的ID



- IVF\_FLAT，先分类（IVF，用的k\-means聚类算法分nlist个组），再在最像的分类里暴力搜索（FLAT）。用倒排表快速缩小搜索范围（IVF），再在缩小的范围内做无压缩的精确计算（FLAT），在“找得快”和“找得准”之间取得了完美平衡。

    - 建立索引阶段：做聚类，用K\-means算法把所有向量分成若干个分区，并建立倒排表（倒排索引），即【分区 ➡️ 向量id清单】映射表，IVF就指这个过程。

    - 查询阶段：用查询向量跟所有分区的【中心点向量（簇中心）】比对，确定最像的分区，拿到该分区的向量id清单，最后再暴力搜索。

    - 分区越多，向量id清单越少，速度越快，但前期建立索引的时候会慢点。

    - 查询阶段可以多找几个中心点向量相似的分区，可以提高准确率（参数 `nprobe`）。

- IVF 边界向量容易搜不准，边界向量是指刚好落在两个簇中间、离两边中心都很近的向量。IVF 搜索时只打开少数几个簇（nprobe），边界向量容易被分到 “隔壁簇”，导致最近邻被漏掉。

- nlist桶数的计算

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWUzMjBlNzNjZDIyOTJmZjg1YzM1N2Y0NjI3N2UzMDFfY2ViY2VlMjQxN2M4MDQwNGMyNTBjMmQyMzgyMGI2MzJfSUQ6NzY0OTMxMzc0NjE2Mjk5NDEyNV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)



### IVF\_SQ8

IVF\_SQ8，是在IVF\_FLAT基础上增加量化步骤：先聚类分组，再建立倒排映射表，最后做分桶压缩，即把每维度的4字节小数压缩成1字节的整数，此时内存和硬盘里的4字节小数都变成了1字节的整数。这样能够节省约75%的空间，速度更快，但搜索准确度（召回率）也会相应降低。

- SQ8（先分桶再压缩）指的是：

    - **8**【分桶256个】：意思是“8位（8\-bit）”。在计算机里，8位正好等于 1 个Byte字节。把原始向量中每个维度可能出现的数值范围（比如 \-1 到 1），均匀地切分成 256 个小桶（因为 8 位二进制数能表示 0 到 255，刚好 256 个数）。

    - **SQ**（Scalar Quantization）【压缩4字节变1字节】：意思是“标量量化”，简单说就是给向量里的每一个小数（标量）都进行压缩，把原始向量里原本占用 4 个字节（32位浮点数 float32）的精确小数，替换成它所在的那个“桶”的编号（0\-255 的整数）。这个整数只需要 1 个字节就能存下。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWRmOGNjZTBhNmI2OGViODI1ODIyMTAxNWU5MjkyNmZfNTljNjQ3ZWU1NzljOGVhZGFhYTBhZDAxMDY4ODkzZTBfSUQ6NzY0NjE0MDExOTQ0Mjg4NTg0N18xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

​    

### IVF\_PQ

IVF\_PQ，IVF聚类\+倒排过程不变，原始向量数据做“乘积量化（Product Quantization）”，即分段后再做聚类。

- PQ（乘积量化）：切段 \+ 聚类中心做替身。假设你有一个 128 维的长向量，PQ 会这样做：

    - 切段（Product）：先把这 128 维切成 M 个小段（比如切成 4 段，每段就是 32 维）。

    - 聚类找替身（Quantization）：对每一段，系统都会提前准备好 **256 个“标准小段”（也就是聚类中心）**。当要存储你的向量时，系统会看你的第 1 段最像哪个“标准小段”，就记下那个标准小段的编号（0\-255）；再看第 2 段像哪个，再记下编号……以此类推。

    - 压缩结果：原本一个 128 维的向量，最后只存了 4 个编号（比如 \[124, 56, 132, 222\]）。这 4 个编号就是原向量的“压缩替身”。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YmViMTI2YWVhYTQyMWU0ODllNGYyYmRhYzI1OWRlYzlfZGI5YWQxOGUxMDVlNTgxYjU4OGI2YzFjNjdjZDFiMjdfSUQ6NzY0NjE0MDEyMDU1MDE4MjA3OV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

- IVF\_PQ优缺点：压缩率极其恐怖！它能把向量压缩到原大小的几十分之一甚至百分之一，极度节省内存和磁盘空间。因为是用“替身”来代表原向量，细节丢失比 SQ8 更多，所以搜索的准确度（召回率）通常会比 SQ8 再低一些。



### HNSW

FLAT，暴力遍历，最准，数据一多巨慢；IVF（倒排），分区搜索，大数据还行，但边界向量（两簇中间的）容易搜不准；HNSW综合最强，速度、精度、内存平衡最好，现在 Milvus、Qdrant、Chroma、FAISS 新版主力索引都是它。

- HNSW 是专门给向量搜索做的「多层导航近邻地图」，搜相似向量又快又省内存，当前主流向量数据库全靠它。HNSW 是 分层的、可导航的小世界图索引：

    - S = Skip List（跳表）：多层链表，高层当 “快速通道”，低层走细节

    - NSW = Navigable Small World（可导航小世界图）：每个点都连一堆相近邻居

    - H = Hierarchical（分层）：把上面的图，分成好多层

- 现实比喻：把每一个向量 = 一座房子，目标：找和「目标房子」距离最近的几栋房子。

    - 单层 NSW（底层逻辑：小世界图）：每栋房子，都会主动认识身边一批邻居（相近向量）。找最近房子的流程：

        - 随便挑一栋房子当起点

        - 问它：你邻居里谁离目标更近？

        - 走到更近的那栋，再重复问邻居。

        - 一路越走越近，最后找到最近的一批。

        - 优点：比暴力遍历快很多

        - 缺点：数据量极大时，还是要走很多步。

    - 加上分层 H = HNSW（最终形态），给这片房子修多层高速路，从上到下：

        - 最高层（顶层）：房子最少，全是「地标大楼」，连接最远，全局快速跳转（高速公路）

        - 中间层：房子变多，路段变密（城市主干道）

        - 最底层：包含所有房子，邻居都是极近的（小区小路）

    - 完整搜索流程：

        - 从最高层出发：随便找个地标楼，顺着高层邻居大跨度跳跃，快速落到「目标大致区域」

        - 逐层往下走：每下一层，范围缩得更小，一步步逼近

        - 到最底层：在全部房子里，精细找距离最近的结果

        - 全程：先高速跑远路，再小路精搜，速度拉满。

- 插入新向量（新增房子）流程

    - 从顶层开始，逐层找到这个新向量该落在哪个区域

    - 在每一层里，给它连上周围相近的邻居

    - 层数越高，每个房子连的邻居越少；层数越低，邻居越多

    - 简单说：新来的房子，每层都给自己找邻居、融入路网。

- 关键特性

    - 速度极快：分层跳跃减少对比次数，百万 / 亿级向量也能毫秒返回结果。

    - 近似搜索（不是 100% 精准）： 它找的是大概率最像的，不是数学上绝对最近。 可以调参数：想要更准就放慢一点，想要更快就牺牲一点点精度。

    - 内存占用友好： 只存每个向量的邻居关系，比很多索引结构省空间。

- 最常用参数

    - `ef_construction`：建索引时找多少邻居（越大索引越准、建库越慢）

    - `ef_search`：搜索时遍历多少候选（越大搜索越准、速度略降）

    - `M`：每个节点默认连多少个邻居

- 总结

    - 每个向量是一个点，互相连接邻近点 = 小世界图

    - 把图分成多层，高层快跑、低层细找 = HNSW

    - 核心逻辑：高层粗定位，底层精搜索

    - 特点：快、准、内存适中，工业界首选向量索引。



### 关于剪枝 \[拓展\]

HNSW 的剪枝＝每个节点最多只留 M 条 “有用” 的邻居边，多了就删掉，保证图又快又省内存、还好导航。

- 为什么要剪枝？剪枝＝把多余、没用、重复的邻居删掉，只留最优质的 M 个。回到房子比喻：

    - 每个房子（向量）可以认识很多邻居

    - 如果不限制：每个房子连几百几千个邻居 → 内存炸、搜索变慢

    - HNSW 规定：每层每个房子最多连 M 个邻居（M 一般 12\~48）

- 什么时候剪枝？只在建索引 / 插入新向量时剪，搜索时不剪。流程：

    - 新来一个向量（新房）

    - 在图里先找一堆候选邻居（ef\_construction 个）

    - 从这一堆里挑最好的 M 个，剩下全删掉 → 剪枝

    - 同时：老房子多了新邻居后，如果超过 M，也会把最差的那条删掉

- 怎么剪？两种常用策略

    1. 简单剪枝（Simple）：留最近的

        - 候选邻居按距离从小到大排序

        - 直接留最近的 M 个，远的全砍

        - 优点：快、简单

        - 缺点：邻居可能都挤在一个方向，“视野窄”

    2. 启发式、多样性剪枝（Heuristic）：留 “四面八方” 的

        - 不光看距离，还要看方向不重复

        - 规则大概是：新邻居不能跟已选邻居太像 / 太近

        - 效果：邻居散得开、覆盖广，不容易陷入局部死角，搜索更准

        - 例子：绿色是新房，候选很多。简单剪枝会全选左边一簇；多样性剪枝会左右各留几个，保证两边都能走到。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDVjNTljMjA1OTM0NTUyNDMxMmUzMTFkN2Q1YWM0MzZfM2Y3MzM4OGM4M2NlYzg2NGVjYWFiMTc5NWUzZjk3MjhfSUQ6NzY0NjMxNTUyMTc3NDYwMzQ3NV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

​            

- 核心参数：M（最大邻居数）：每层每个节点最多保留的邻居数（剪枝上限） 

    - 越大：图越密、搜索越准、内存越大、建库越慢

    - 越小：图越疏、搜索越快、内存越小、精度略降

    - 经验：M=16/24 最常用；高维 / 高精度用 32\~48

- 剪枝到底剪的是什么？

    - 剪掉太远的邻居（没用）

    - 剪掉太重复的邻居（同一方向、冗余）

    - 保留最近 \+ 分布最散的 M 个邻居

- 剪枝的意义作用

    - ✅ 内存可控

    - ✅ 搜索更快

    - ✅ 图更 “好导航”，不容易迷路

    - ❌ 极端情况下可能丢极少数最优边（近似搜索代价）

        - 两个邻居（向量）相连成一条线，这个线就是边

- 总结 HNSW 剪枝：插入时给每个节点 “限流” 到 M 条边，优先留近的、再留散的，让图又快又省还好走。



### RNSG \[拓展\]

> Refined Navigating Spreading\-out Graph
> 
> 

HNSW 的 “精简省内存版”，单层导航图；适合动态数据（增删改友好、疑似好友、可能认识的人），适用场景：

- 千万级、内存紧张、又要速度的场景

- 频繁插入更新的向量库



### DiskANN \[拓展\]

- Disk\-based ANN：把图索引放磁盘，内存只放压缩摘要，支撑百亿级超大数据。其原理是 Vamana 图 \+ PQ 压缩：

    - Vamana 图：类似 RNSG/HNSW 的图，但整张图存在 SSD，不占内存。

    - PQ 压缩：每个向量压缩成 32～64 字节的 “摘要”，全量放进内存，用来快速算近似距离Milvus。

- 查询流程： 

    - 内存里用 PQ 摘要快速遍历图、找候选；

    - 按需从磁盘读取少量节点的全量向量做精确重排。



### ANNOY \[拓展\]

> Approximate Nearest Neighbors Oh Yeah
> 
> 

搞几十套不同的划分规则，每次查询靠多套规则快速圈出一小批候选，再从中挑相似的，避免全量比对。

规则就是 “随机找俩点，画条中线切开”，反复切，切到小块里只剩几个点；多做几套这种切法 = 多棵树。每棵树的每一刀都是重新随机选两个点，所以每棵树的切法都不一样、互相独立。

要查询X的相似结果的时候，对每棵树： 从根开始，按这棵树的规则（每一层的中线）判断 X 该走左边还是右边；一路走到叶子，得到这个小块里的所有点；再把所有树找到的小块里的点全部汇总、去重；最后在这批少量候选里算精确距离，挑最像的 TopK

优点：占内存极少，建索引快，数据能放磁盘。缺点：树少了找不准，树多了变慢；大数据量表现一般。适合场景：数据量不大、机器内存小、平时很少增删改数据。



## ANNS近似最近邻 \[拓展\]

- Milvus中只有 FLAT 是 KNN（暴力），其余所有索引都是 近似最近邻ANNS：

    - IVF\_FLAT / IVF\_SQ8 / IVF\_PQ

    - HNSW / RNSG

    - ANNOY

    - DiskANN 等

- ANNS的概念

    - NNS（Nearest Neighbor Search）：给定一个查询向量，在库里找 “距离最近” 的向量。

    - KNN（K\-Nearest Neighbor）：精确最近邻，暴力全扫，高维 \+ 大数据集下极慢（亿级向量可能秒 / 分钟级）

    - ANNS（Approximate Nearest Neighbor Search，近似最近邻）： 不保证 “绝对最近”，只保证 “很可能很近”；在可接受精度损失（比如 95%\~99% 召回）下，换取几个数量级的速度提升（毫秒级）

- 为啥用ANNS

    - 向量是 “高维” 的，传统索引B\+ 树、二叉树、哈希表无法有效缩小搜索范围，完全失效

    - 数据规模巨大：百万、千万、亿级是常态（RAG、推荐、图像检索：亿级向量很常见），ANNS更快

    - 业务场景 “允许近似”，更看重速度，向量检索的典型场景：

        - RAG：找 “语义相似” 的文档，不需要绝对最相似

        - 图片检索：找 “长得像” 的图，差一点没关系

        - 推荐：找 “兴趣相近” 的用户 / 商品，近似足够

    - Milvus 的定位就是面向生产的高性能向量库，所以用ANNS近似最近邻索引。



## 向量类型对应索引 \[拓展\]

其他索引可查看官方文档 https://milvus\.io/docs/zh/index\-explained\.md

- 索引选择总结

    - 普通 RAG / 语义搜索 → FLOAT\_VECTOR \+ HNSW索引（99% 项目）

    - 想省空间 → FLOAT16 / BF16 / INT8

    - 关键词搜索 → SPARSE\_FLOAT\_VECTOR

    - 图像哈希 / 指纹 → BINARY\_VECTOR



## 向量相似度

- 在 Milvus 中，**相似度度量用于衡量向量之间的相似性**。选择一个好的距离度量方法可以显著提高分类和聚类的性能。根据输入数据的形式，选择特定的相似度度量方法可以获得最优的性能。Milvus 中，对于计算俩个向量代表内容的相似度，可以选择以下度量方式：

    - **欧几里得距离（L2），欧式距离**（勾股定理），计算**结果越小越接近**。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjlkNzlkYjc1YTAzYmQxYjZmOGE0YThiNzY5NDQxZDNfZTUxMmJjNzdkNjUwNTgzOTZkNDE1YTBiZDQzMGQ3ZDVfSUQ6NzY0NjMwODEwNTUxNzIwNjc0Ml8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

    - **内积（IP）**，计算**结果越大越接近**；**归一化后与余弦相似度 \(COSINE\)一致**：越接近1越相似，\-1则完全相反。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTA5ODVkYWYyY2MxZWNlY2JkZTI3MDlkYTg4OTcxMzlfNzNhMjZkYjEyMTgwMDkxM2Q3YTFlYTlhMGQ2N2YyMTBfSUQ6NzY0NjMxMDMzODM0MjIyNjkwM18xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

    - **余弦相似度 \(COSINE\)，越接近1越相似，\-1则完全相反**。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODg3OGEzYjIzYmM4N2NlMTc2MWNmNzZhMzYxMmM0YWVfZmFmZjc4NGM3YTcxODNjZGUyMTUzYzJiYjg2MTgyY2FfSUQ6NzY0NjMxMjU0MTA2ODgyMzc0MF8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

- 归一化计算

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODVkYzRmNzBiYjVlZWI2Y2Y3NDJmNThiYzU0ZWNlZTVfZDk4OWI2OGFmNmMyZTg1MTUyMmMyNTE5MDFjNmE2ZjVfSUQ6NzY0NjMxNDcxMDE1ODAzNTkyNF8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

- 大部分开源 Embedding 模型默认选 COSINE余弦相似度。如果开启 IVF、HNSW 索引，度量方式要在建索引时就指定，不能事后改。归一化向量后，`COSINE` 和 `IP` 结果等价，IP 计算开销更低。

    - 做文本 / 语义搜索 → 直接用 余弦 COSINE（90% 大模型嵌入场景）

    - 向量已经归一化、想提速 → 用 IP

    - 做坐标、点位、形态特征 → 用 L2



## 向量与余弦相似度\[拓展\]

> 拓展：对向量进行余弦相似度计算
> 
> 

### 向量Vector\[拓展\]

向量（Vector）就是文本的 “数学身份证” ，它把一段文字的语义信息，转换成一串固定长度的数字列表，让计算机能 “看懂” 文字的含义并可以做相似度计算。其本质是**在高纬空间坐标系中，语义表达的方向。**

文本转换成向量的计算（文本嵌入过程），可借助文本嵌入模型实现，如text\-embedding\-v1模型。

向量的维度，就是这串数字的长度，表示一段文本在多个抽象语义特征方面的强度，越长越准，但也越费资源。使用text\-embedding\-v1文本嵌入模型，可以从文本提取语义特征并转换为固定长度1536个的数字，这些数字表示某段文本在1536个主题上的得分（强度）。维度数代表模型用多少个抽象语义特征来描述文本，维度越多，做语义匹配越精准，但性能压力也会增大。语义维度举例如下：

向量的匹配，就是识别2段文本是否是表述相似的含义，可以通过计算二者的向量**余弦相似度**来完成。

```Shell
# 下列案例中向量为示例，仅描述概念，非真实向量）
# 通过余弦相似度算法可以计算得到：A和B相似度0.999789，A和C相似度0.361446
# 相似度越高，语义越接近
# A： “如何快速学打篮球” [0.2, 0.5, 0.8]
# B： “打篮球怎么学得快” [0.18, 0.52, 0.79]
# C： “运动后吃什么好呢” [0.9, 0.1, 0.2]
```



### 余弦相似度\[拓展\]

向量的数字，表示在高维空间中的**方向**和**长度**\.而余弦相似度主要就是撇除长度的影响，得到方向的夹角。夹角越小越相似，即方向越相同。夹角为0度时，方向完全相同（语义同向），余弦相似度即为1，两段文本的语义就高度相同，即文本语义高度一致、表达方向完全相同。这里以二维向量举例，二维向量如下图所示。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGQyYWE1MDA2Y2M1YTYxMWQ0ZjM1NDBiNzM3ODVmZjdfYjIzNmM1YmViY2RmNzZkNjlmZGU1YmU0MzA4ZjIxMDJfSUQ6NzY0NjM1MjU1MDY2NDg4MzM4Nl8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

```Shell
**两段文字-->两个向量-->两个箭头（在高纬空间坐标系中，语义表达的方向）**
1. 箭头**方向越接近**，意思越像，余弦相似度**越接近 1**
2. 箭头**完全相反**，意思完全不同，相似度**接近 - 1**
3. 箭头**垂直没关系**，毫无关联，相似度**等于 0**
**余弦相似度就看两个向量夹角大小，夹角越小，越相似**
```

- 余弦相似度的计算公式有如下三种数学表达方式，其本质都是一样的。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=N2UwN2NjN2E1ODA2Y2VjZDg5M2I0OTc0NjZmZTljN2RfNjA5YjMzMTkzNjEzYWJmNDRhYjRlY2M3YzlhZGM4ZTZfSUQ6NzY0NjM1MjU1MjE4MjgyODIyOV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

​    

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjcyZjcxMTRiMGU5NjZlYzZkMzkwMDE1ZmQ2YTJmZjVfNTJlYzAxNDY2NTdlZjBmM2EwYzVhZWY1YTdlMWE2YjZfSUQ6NzY0NjM1MjU1MDA5MDUyNTYzOV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

​    

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWE0N2U0ZjE0MzM1ZjFmOGFiMThjYjNmODU0ZDRiNzFfYjZmZjFmODkzNjYzYmEyYTk2NjgxMDJhOWExNmNiNWVfSUQ6NzY0NjM1MjU1MDk1NDUwMzExNV8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)

    - 分子：**两个向量对应数字相乘再全部相加（点积：同维度数字乘积之和）**
    
    - 分母：**A 所有数平方的和再开根号 × B 所有数平方的和再开根号（即A模长\*B模长）**
    
    - 余弦相似度 = 两个向量的**点积 **÷ 两个向量**模长**的乘积
    
    - 结果就是相似度，范围：**\[\-1，1\]**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Mzc4Njc2NzUzYzY1NWUwYTJhNmY0ZjBlMmMxN2UyNjFfZjA3MmUzY2UyMmVhNDM2MDg1ZDlkYTM1ZGM2NjZkYzRfSUQ6NzY0NjM1MjU1MTgyMjMzMDgzNl8xNzg0MDg2NjczOjE3ODQxNzMwNzNfVjM)



### 计算余弦相似度\[拓展\]

```Python
import numpy as np
"""
计算两个向量的余弦相似度（衡量方向相似性，剔除长度影响）

参数：
    vec_a (np.array): 向量A
    vec_b (np.array): 向量B
返回：
    float: 余弦相似度结果（范围[-1,1]，越接近1方向越一致）
公式：
  - 分子：两个向量对应数字相乘再全部相加（点积：同维度数字乘积之和）
  - 分母：A 所有数平方的和再开根号 × B 所有数平方的和再开根号（即A模长*B模长）
  - 余弦相似度 = 两个向量的点积 ÷ 两个向量模长的乘积

A: [0.5, 0.5]
B: [0.7, 0.7]
C: [0.7, 0.5]
D: [-0.6, -0.5]
"""
def get_dot(vec_a, vec_b):
    *"""计算2个向量的点积，即：2个向量同维度数字乘积之和"""*
*    *if len(vec_a) != len(vec_b):
        raise ValueError("2个向量必须维度数量相同")

    dot_sum = 0
    *# zip(vec_a, vec_b): 把两个向量按位置两两配对*
*    # 每次遍历出的a和b就是维度相同的两个数字*
*    *for a, b in zip(vec_a, vec_b):
        ab = a * b *# 同维度数字乘积*
*        *dot_sum += ab *# 累加和*

*    *return dot_sum

def get_norm(vec):
    *"""计算单个向量的模长：对向量的每个数字求平方、再求和、再开根号"""*
*    *sum_square = 0
    for v in vec:
        vv = v * v *# 每个数字的平方*
*        *sum_square += vv *# 再求和*

*    # numpy的sqrt函数完成开根号*
*    *ret = np.sqrt(sum_square)
    return ret

def cosine_similarity(vec_a, vec_b):
    *"""余弦相似度：2个向量的点积 除以 2个向量模长的乘积"""*
*    *result = get_dot(vec_a, vec_b) / (get_norm(vec_a) * get_norm(vec_b))
    return result

if __name__ == '__main__':
    vec_a = [0.5, 0.5]
    vec_b = [0.7, 0.7]
    vec_c = [0.7, 0.5]
    vec_d = [-0.6, -0.5]

    print("ab:", cosine_similarity(vec_a, vec_b))
    print("ac:", cosine_similarity(vec_a, vec_c))
    print("ad:", cosine_similarity(vec_a, vec_d))
```



