# DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration 

**Title (ZH)**: 双空间过滤与强化刚性注册 

**Authors**: Jiayi Li, Yuxin Yao, Qiuhang Lu, Juyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17034)  

**Abstract**: Rigid registration, aiming to estimate a rigid transformation to align source and target data, play a crucial role in applications such as SLAM and 3D reconstruction. However, noisy, partially overlapping data and the need for real-time processing pose major challenges for rigid registration. Considering that feature-based matching can handle large transformation differences but suffers from limited accuracy, while local geometry-based matching can achieve fine-grained local alignment but relies heavily on a good initial transformation, we propose a novel dual-space paradigm to fully leverage the strengths of both approaches. First, we introduce an efficient filtering mechanism that incorporates a computationally lightweight single-point RANSAC algorithm followed by a refinement module to eliminate unreliable feature-based correspondences. Subsequently, we treat filtered correspondences as anchor points, extract geometric proxies, and formulates an effective objective function with a tailored solver to estimate the transformation. Experiments verify our method's effectiveness, as shown by achieving up to a 32x CPU-time speedup over MAC on KITTI with comparable accuracy. 

**Abstract (ZH)**: 基于双空间 paradigm 的刚体注册方法：融合特征匹配与局部几何匹配优势 

---
# COSMO-Bench: A Benchmark for Collaborative SLAM Optimization 

**Title (ZH)**: COSMO-Bench: 一种协作SLAM优化基准 

**Authors**: Daniel McGann, Easton R. Potokar, Michael Kaess  

**Link**: [PDF](https://arxiv.org/pdf/2508.16731)  

**Abstract**: Recent years have seen a focus on research into distributed optimization algorithms for multi-robot Collaborative Simultaneous Localization and Mapping (C-SLAM). Research in this domain, however, is made difficult by a lack of standard benchmark datasets. Such datasets have been used to great effect in the field of single-robot SLAM, and researchers focused on multi-robot problems would benefit greatly from dedicated benchmark datasets. To address this gap, we design and release the Collaborative Open-Source Multi-robot Optimization Benchmark (COSMO-Bench) -- a suite of 24 datasets derived from a state-of-the-art C-SLAM front-end and real-world LiDAR data. Data DOI: this https URL 

**Abstract (ZH)**: 近年来，分布式优化算法在多robot协作的同时定位与建图（C-SLAM）研究中受到了广泛关注。然而，由于缺乏标准基准数据集，该领域的研究面临较大困难。单robot SLAM领域中已经有效地应用了此类数据集，专注于多robot问题的研究人员将受益于专门的数据集。为填补这一空白，我们设计并发布了协作开源多robot优化基准（COSMO-Bench）——一套基于最先进的C-SLAM前端和真实LiDAR数据的24个数据集。数据DOI: this https URL。 

---
# A Synthetic Dataset for Manometry Recognition in Robotic Applications 

**Title (ZH)**: 一种用于机器人应用的食道测压识别合成数据集 

**Authors**: Pedro Antonio Rabelo Saraiva, Enzo Ferreira de Souza, Joao Manoel Herrera Pinheiro, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17468)  

**Abstract**: This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos-Predict2 world-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real-only baseline. These findings highlight the viability of a synthetic-first approach as an efficient, cost-effective, and safe alternative for developing reliable perception systems in safety-critical and resource-constrained industrial applications. 

**Abstract (ZH)**: 本研究解决了在复杂工业环境中，如 offshore 油平台，训练鲁棒对象检测模型时面临的数据稀缺和高昂采集成本的挑战。在这些危险环境中收集真实世界数据的实际和经济障碍往往阻碍了自主检测系统的开发。为克服这一难题，我们在此工作中提出并验证了一种结合过程渲染和AI驱动视频生成的混合数据合成管道。我们的方法利用BlenderProc创建具有精确标注的 photorealistic 图像，并采用控制域随机化，同时结合 NVIDIA 的 Cosmos-Predict2 世界基础模型，生成具有时间多样性且物理上合理的视频序列，捕捉到罕见视角和不利条件。我们证明，使用综合数据集训练的基于 YOLO 的检测网络，该数据集结合了真实图像和我们合成的数据，相对于仅使用真实世界数据训练的模型实现了更优性能。值得注意的是，真实数据与合成数据各占一半的混合数据集达到了最高的准确率，超越了仅使用真实数据的基线。这些发现突显了合成数据优先方法在安全关键和资源受限的工业应用中作为高效、经济和安全替代方案的可行性。 

---
# Collaborative-Online-Learning-Enabled Distributionally Robust Motion Control for Multi-Robot Systems 

**Title (ZH)**: 基于协作在线学习的分布鲁棒多机器人系统运动控制 

**Authors**: Chao Ning, Han Wang, Longyan Li, Yang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17173)  

**Abstract**: This paper develops a novel COllaborative-Online-Learning (COOL)-enabled motion control framework for multi-robot systems to avoid collision amid randomly moving obstacles whose motion distributions are partially observable through decentralized data streams. To address the notable challenge of data acquisition due to occlusion, a COOL approach based on the Dirichlet process mixture model is proposed to efficiently extract motion distribution information by exchanging among robots selected learning structures. By leveraging the fine-grained local-moment information learned through COOL, a data-stream-driven ambiguity set for obstacle motion is constructed. We then introduce a novel ambiguity set propagation method, which theoretically admits the derivation of the ambiguity sets for obstacle positions over the entire prediction horizon by utilizing obstacle current positions and the ambiguity set for obstacle motion. Additionally, we develop a compression scheme with its safety guarantee to automatically adjust the complexity and granularity of the ambiguity set by aggregating basic ambiguity sets that are close in a measure space, thereby striking an attractive trade-off between control performance and computation time. Then the probabilistic collision-free trajectories are generated through distributionally robust optimization problems. The distributionally robust obstacle avoidance constraints based on the compressed ambiguity set are equivalently reformulated by deriving separating hyperplanes through tractable semi-definite programming. Finally, we establish the probabilistic collision avoidance guarantee and the long-term tracking performance guarantee for the proposed framework. The numerical simulations are used to demonstrate the efficacy and superiority of the proposed approach compared with state-of-the-art methods. 

**Abstract (ZH)**: 基于狄利克雷过程混合模型的COllaborative-Online-Learning (COOL) 启发的多机器人系统避碰运动控制框架 

---
# Social Identity in Human-Agent Interaction: A Primer 

**Title (ZH)**: 人类与智能体互动中的社会身份：入门指南 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2508.16609)  

**Abstract**: Social identity theory (SIT) and social categorization theory (SCT) are two facets of the social identity approach (SIA) to understanding social phenomena. SIT and SCT are models that describe and explain how people interact with one another socially, connecting the individual to the group through an understanding of underlying psychological mechanisms and intergroup behaviour. SIT, originally developed in the 1970s, and SCT, a later, more general offshoot, have been broadly applied to a range of social phenomena among people. The rise of increasingly social machines embedded in daily life has spurned efforts on understanding whether and how artificial agents can and do participate in SIA activities. As agents like social robots and chatbots powered by sophisticated large language models (LLMs) advance, understanding the real and potential roles of these technologies as social entities is crucial. Here, I provide a primer on SIA and extrapolate, through case studies and imagined examples, how SIT and SCT can apply to artificial social agents. I emphasize that not all human models and sub-theories will apply. I further argue that, given the emerging competence of these machines and our tendency to be taken in by them, we experts may need to don the hat of the uncanny killjoy, for our own good. 

**Abstract (ZH)**: 社会身份理论（SIT）和社会分类理论（SCT）是社会身份方法（SIA）的两个方面，用于理解社会现象。社会身份理论和社会分类理论是描述和解释人们如何通过理解基本的心理机制和群体间行为进行社会互动的模型。社会身份理论（SIT）最初在20世纪70年代发展，而社会分类理论（SCT）则是后来的一个更为广泛的应用分支，两者都被广泛应用于人类社会的各种现象中。随着嵌入日常生活中的社会机器越来越多，我们开始努力理解人工代理是否以及如何参与社会身份方法的活动。随着像社会机器人和由复杂大型语言模型（LLMs）驱动的聊天机器人的发展，理解这些技术作为社会实体的真实和潜在角色至关重要。在这里，我对社会身份方法提供了一个概述，并通过案例研究和假设的示例，探讨社会身份理论（SIT）和社会分类理论（SCT）如何适用于人工社会代理。我强调，并非所有的人类模型和次理论都适用。进一步地，鉴于这些机器日益增长的能力以及我们倾向于被它们所迷惑的倾向，我们这些专家可能需要戴上怪异的 killjoy 的帽子，这对我们自己是有益的。 

---
# Hermes 4 Technical Report 

**Title (ZH)**: 赫梅斯4技术报告 

**Authors**: Ryan Teknium, Roger Jin, Jai Suphavadeeprasit, Dakota Mahan, Jeffrey Quesnelle, Joe Li, Chen Guang, Shannon Sands, Karan Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2508.18255)  

**Abstract**: We present Hermes 4, a family of hybrid reasoning models that combine structured, multi-turn reasoning with broad instruction-following ability. We describe the challenges encountered during data curation, synthesis, training, and evaluation, and outline the solutions employed to address these challenges at scale. We comprehensively evaluate across mathematical reasoning, coding, knowledge, comprehension, and alignment benchmarks, and we report both quantitative performance and qualitative behavioral analysis. To support open research, all model weights are published publicly at this https URL 

**Abstract (ZH)**: 我们介绍Hermes 4，这是一种结合结构化多轮推理和广泛指令遵循能力的混合推理模型系列。我们在数据整理、合成、训练和评估过程中遇到的挑战进行了描述，并概述了大规模解决这些挑战所采用的解决方案。我们在数学推理、编码、知识、理解及对齐基准上进行了全面评估，并报告了定量性能和定性行为分析结果。为了支持开放研究，所有模型权重已在以下网址公开发布：https://github.com/alibaba/Hermes。 

---
# Efficient Computation of Blackwell Optimal Policies using Rational Functions 

**Title (ZH)**: 使用有理函数计算Blackwell最优策略的高效方法 

**Authors**: Dibyangshu Mukherjee, Shivaram Kalyanakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18252)  

**Abstract**: Markov Decision Problems (MDPs) provide a foundational framework for modelling sequential decision-making across diverse domains, guided by optimality criteria such as discounted and average rewards. However, these criteria have inherent limitations: discounted optimality may overly prioritise short-term rewards, while average optimality relies on strong structural assumptions. Blackwell optimality addresses these challenges, offering a robust and comprehensive criterion that ensures optimality under both discounted and average reward frameworks. Despite its theoretical appeal, existing algorithms for computing Blackwell Optimal (BO) policies are computationally expensive or hard to implement.
In this paper we describe procedures for computing BO policies using an ordering of rational functions in the vicinity of $1$. We adapt state-of-the-art algorithms for deterministic and general MDPs, replacing numerical evaluations with symbolic operations on rational functions to derive bounds independent of bit complexity. For deterministic MDPs, we give the first strongly polynomial-time algorithms for computing BO policies, and for general MDPs we obtain the first subexponential-time algorithm. We further generalise several policy iteration algorithms, extending the best known upper bounds from the discounted to the Blackwell criterion. 

**Abstract (ZH)**: 马尔可夫决策问题（MDPs）为跨不同领域的序贯决策建模提供了一个基础框架，这些决策由折现奖励和平均奖励等最优性标准指导。然而，这些标准存在固有的局限性：折现最优性可能过度优先考虑短期奖励，而平均最优性则依赖于强有力的结构假设。Blackwell最优性解决了这些挑战，提供了一个在折现和平均奖励框架下都能确保最优性的稳健和全面的标准。尽管在理论上具有吸引力，现有的计算Blackwell最优（BO）策略的算法在计算上可能非常昂贵或难以实现。

在本文中，我们描述了使用接近1的有理函数排序来计算BO策略的过程。我们适应了最先进的确定性和通用MDP算法，将数值评估替换为有理函数的符号操作，从而得到与位复杂度无关的界。对于确定性MDP，我们首次提出了计算BO策略的强多项式时间算法，并对于通用MDP，我们获得了首个次指数时间算法。我们还进一步推广了多种策略迭代算法，将已知的最佳上界从折现标准扩展到Blackwell标准。 

---
# Disentangling the Factors of Convergence between Brains and Computer Vision Models 

**Title (ZH)**: 解开大脑与计算机视觉模型收敛因素的分离分析 

**Authors**: Joséphine Raugel, Marc Szafraniec, Huy V. Vo, Camille Couprie, Patrick Labatut, Piotr Bojanowski, Valentin Wyart, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2508.18226)  

**Abstract**: Many AI models trained on natural images develop representations that resemble those of the human brain. However, the factors that drive this brain-model similarity remain poorly understood. To disentangle how the model, training and data independently lead a neural network to develop brain-like representations, we trained a family of self-supervised vision transformers (DINOv3) that systematically varied these different factors. We compare their representations of images to those of the human brain recorded with both fMRI and MEG, providing high resolution in spatial and temporal analyses. We assess the brain-model similarity with three complementary metrics focusing on overall representational similarity, topographical organization, and temporal dynamics. We show that all three factors - model size, training amount, and image type - independently and interactively impact each of these brain similarity metrics. In particular, the largest DINOv3 models trained with the most human-centric images reach the highest brain-similarity. This emergence of brain-like representations in AI models follows a specific chronology during training: models first align with the early representations of the sensory cortices, and only align with the late and prefrontal representations of the brain with considerably more training. Finally, this developmental trajectory is indexed by both structural and functional properties of the human cortex: the representations that are acquired last by the models specifically align with the cortical areas with the largest developmental expansion, thickness, least myelination, and slowest timescales. Overall, these findings disentangle the interplay between architecture and experience in shaping how artificial neural networks come to see the world as humans do, thus offering a promising framework to understand how the human brain comes to represent its visual world. 

**Abstract (ZH)**: 许多在自然图像上训练的AI模型形成了与人类大脑相似的表示。然而，驱动这种大脑模型相似性的因素仍然知之甚少。为了独立地分析模型、训练和数据如何引导神经网络发展出类似大脑的表示，我们训练了一种系统地变化这些不同因素的自监督视觉变压器（DINOv3）家族。我们将它们对图像的表示与使用fMRI和MEG记录的人类大脑图像进行比较，提供高分辨率的空间和时间分析。我们使用三个互补的度量标准来评估大脑模型相似性，这些度量标准分别关注整体表示相似性、拓扑组织和时间动态。我们展示了模型大小、训练量和图像类型这三个因素如何各自独立地并相互作用地影响这些大脑相似性度量。特别是，用最以人类为中心的图像训练的最大DINOv3模型达到最高的大脑相似度。这些类似大脑的表示在训练中出现了特定的顺序：模型首先与感觉皮层的早期表示对齐，只有在大量训练后才与脑的晚期和前额皮层的表示对齐。最终，这一发育轨迹由人类皮层的结构和功能特性指数化：由模型最后获得的表示特异性地与发育扩张最大、厚度最大、髓鞘化最少和时间尺度最慢的皮层区域对齐。总体而言，这些发现解缠了架构和经验在塑造人工神经网络如何以人类方式看待世界之间的互动关系，从而为理解人类大脑如何表示其视觉世界提供了一个有前景的框架。 

---
# Interpretable Early Failure Detection via Machine Learning and Trace Checking-based Monitoring 

**Title (ZH)**: 基于机器学习和跟踪检查的可解释早期故障检测方法 

**Authors**: Andrea Brunello, Luca Geatti, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17786)  

**Abstract**: Monitoring is a runtime verification technique that allows one to check whether an ongoing computation of a system (partial trace) satisfies a given formula. It does not need a complete model of the system, but it typically requires the construction of a deterministic automaton doubly exponential in the size of the formula (in the worst case), which limits its practicality. In this paper, we show that, when considering finite, discrete traces, monitoring of pure past (co)safety fragments of Signal Temporal Logic (STL) can be reduced to trace checking, that is, evaluation of a formula over a trace, that can be performed in time polynomial in the size of the formula and the length of the trace. By exploiting such a result, we develop a GPU-accelerated framework for interpretable early failure detection based on vectorized trace checking, that employs genetic programming to learn temporal properties from historical trace data. The framework shows a 2-10% net improvement in key performance metrics compared to the state-of-the-art methods. 

**Abstract (ZH)**: 基于向量化轨迹检查的GPU加速可解释早期故障检测框架：纯过去（协）安全性片段的Signal Temporal Logic监控减少为轨迹检查 

---
# A Taxonomy of Transcendence 

**Title (ZH)**: 超越性的分类 

**Authors**: Natalie Abreu, Edwin Zhang, Eran Malach, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17669)  

**Abstract**: Although language models are trained to mimic humans, the resulting systems display capabilities beyond the scope of any one person. To understand this phenomenon, we use a controlled setting to identify properties of the training data that lead a model to transcend the performance of its data sources. We build on previous work to outline three modes of transcendence, which we call skill denoising, skill selection, and skill generalization. We then introduce a knowledge graph-based setting in which simulated experts generate data based on their individual expertise. We highlight several aspects of data diversity that help to enable the model's transcendent capabilities. Additionally, our data generation setting offers a controlled testbed that we hope is valuable for future research in the area. 

**Abstract (ZH)**: 尽管语言模型被训练成模拟人类，但产生的系统展示了超出单一人类能力范围的能力。为理解这一现象，我们利用一个控制环境来识别导致模型超越数据源性能的训练数据属性。我们基于前人工作概述了三种超越模式，称为技能去噪、技能选择和技能泛化。然后，我们引入了一个基于知识图谱的环境，在该环境中模拟专家根据其专业领域生成数据。我们强调了有助于增强模型超越能力的数据多样性方面的多个方面。此外，我们的数据生成设置提供了一个可控的测试平台，我们希望这在未来该领域的研究中具有价值。 

---
# Evaluating Movement Initiation Timing in Ultimate Frisbee via Temporal Counterfactuals 

**Title (ZH)**: 通过时间反事实评估飞盘运动启动时机 

**Authors**: Shunsuke Iwashita, Ning Ding, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2508.17611)  

**Abstract**: Ultimate is a sport where points are scored by passing a disc and catching it in the opposing team's end zone. In Ultimate, the player holding the disc cannot move, making field dynamics primarily driven by other players' movements. However, current literature in team sports has ignored quantitative evaluations of when players initiate such unlabeled movements in game situations. In this paper, we propose a quantitative evaluation method for movement initiation timing in Ultimate Frisbee. First, game footage was recorded using a drone camera, and players' positional data was obtained, which will be published as UltimateTrack dataset. Next, players' movement initiations were detected, and temporal counterfactual scenarios were generated by shifting the timing of movements using rule-based approaches. These scenarios were analyzed using a space evaluation metric based on soccer's pitch control reflecting the unique rules of Ultimate. By comparing the spatial evaluation values across scenarios, the difference between actual play and the most favorable counterfactual scenario was used to quantitatively assess the impact of movement timing.
We validated our method and show that sequences in which the disc was actually thrown to the receiver received higher evaluation scores than the sequences without a throw.
In practical verifications, the higher-skill group displays a broader distribution of time offsets from the model's optimal initiation point.
These findings demonstrate that the proposed metric provides an objective means of assessing movement initiation timing, which has been difficult to quantify in unlabeled team sport plays. 

**Abstract (ZH)**: Ultimate飞盘运动中投接盘得分，持盘球员不能移动，场上动态主要由其他球员的移动驱动。然而，当前团队运动领域的文献忽略了在比赛中球员何时发起未标记移动的定量评价。本文提出了一种针对Ultimate飞盘的运动启动时间的定量评价方法。首先，使用无人机摄像机录制比赛 footage，并获取球员的位置数据，这些数据将作为UltimateTrack数据集发布。接着，检测球员的移动发起，并通过基于规则的方法生成时间上的反事实情景。这些情景使用基于足球控球区域的评价度量进行分析，该度量反映了Ultimate的独特规则。通过比较情景间的空间评价值，实际比赛与最有利的反事实情景之间的差异被用来定量评估运动时间的影响。我们验证了该方法，结果显示实际传接盘序列获得了更高的评价得分。在实际验证中，高技能组显示了与模型最优启动点时间偏移分布更广。这些发现表明，提出的方法提供了一种客观评价未标记团队运动中运动发起时间的方法。 

---
# Consciousness as a Functor 

**Title (ZH)**: 意识作为一种函子 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17561)  

**Abstract**: We propose a novel theory of consciousness as a functor (CF) that receives and transmits contents from unconscious memory into conscious memory. Our CF framework can be seen as a categorial formulation of the Global Workspace Theory proposed by Baars. CF models the ensemble of unconscious processes as a topos category of coalgebras. The internal language of thought in CF is defined as a Multi-modal Universal Mitchell-Benabou Language Embedding (MUMBLE). We model the transmission of information from conscious short-term working memory to long-term unconscious memory using our recently proposed Universal Reinforcement Learning (URL) framework. To model the transmission of information from unconscious long-term memory into resource-constrained short-term memory, we propose a network economic model. 

**Abstract (ZH)**: 我们提出了一种新的意识理论，即意识函子(CF)理论，用于接收和传递来自无意识记忆的内容到意识记忆。我们的CF框架可以被视为Baars提出的全局工作空间理论的范畴表述。CF将无意识过程ensemble建模为煤gebra范畴的topos类别。CF中的内部语言定义为多模态通用Mitchell-Benabou嵌入语言(MUMBLE)。我们使用最近提出的通用强化学习(URL)框架来建模信息从意识短时工作记忆到长期无意识记忆的传递。为了建模信息从长期无意识记忆向资源受限的短时记忆的传递，我们提出了一种网络经济模型。标题：意识函子理论：从无意识记忆到意识记忆的信息传递模型 

---
# Solving Constrained Stochastic Shortest Path Problems with Scalarisation 

**Title (ZH)**: 求解约束随机最短路径问题的标量化方法 

**Authors**: Johannes Schmalz, Felipe Trevizan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17446)  

**Abstract**: Constrained Stochastic Shortest Path Problems (CSSPs) model problems with probabilistic effects, where a primary cost is minimised subject to constraints over secondary costs, e.g., minimise time subject to monetary budget. Current heuristic search algorithms for CSSPs solve a sequence of increasingly larger CSSPs as linear programs until an optimal solution for the original CSSP is found. In this paper, we introduce a novel algorithm CARL, which solves a series of unconstrained Stochastic Shortest Path Problems (SSPs) with efficient heuristic search algorithms. These SSP subproblems are constructed with scalarisations that project the CSSP's vector of primary and secondary costs onto a scalar cost. CARL finds a maximising scalarisation using an optimisation algorithm similar to the subgradient method which, together with the solution to its associated SSP, yields a set of policies that are combined into an optimal policy for the CSSP. Our experiments show that CARL solves 50% more problems than the state-of-the-art on existing benchmarks. 

**Abstract (ZH)**: 约束随机最短路径问题（CSSPs）模型在存在概率效应的情况下，最小化主要成本的同时满足次要成本的约束，例如在预算限制下最小化时间。现有的启发式搜索算法针对CSSPs通过求解一系列逐步加大的线性规划问题来寻找原始CSSP的最优解。本文提出了一种新型算法CARL，该算法使用高效的启发式搜索算法求解一系列无约束随机最短路径问题（SSPs）。CARL利用标量化方法构造这些SSP子问题，将CSSP的向量形式的主要和次要成本投影到一个标量成本。CARL使用类似于次梯度方法的优化算法找到一个最大化标量化的方法，结合其相关的SSP解，生成一组策略并组合成CSSP的最优策略。实验结果显示，CARL在现有基准测试中解决了比最先进的算法多50%的问题。 

---
# L-XAIDS: A LIME-based eXplainable AI framework for Intrusion Detection Systems 

**Title (ZH)**: L-XAIDS：一种基于LIME的可解释人工智能入侵检测系统框架 

**Authors**: Aoun E Muhammad, Kin-Choong Yow, Nebojsa Bacanin-Dzakula, Muhammad Attique Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17244)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and their applications in critical industries such as healthcare, fin-tech and cybersecurity have led to a surge in research in explainability in AI. Innovative research methods are being explored to extract meaningful insight from blackbox AI systems to make the decision-making technology transparent and interpretable. Explainability becomes all the more critical when AI is used in decision making in domains like fintech, healthcare and safety critical systems such as cybersecurity and autonomous vehicles. However, there is still ambiguity lingering on the reliable evaluations for the users and nature of transparency in the explanations provided for the decisions made by black-boxed AI. To solve the blackbox nature of Machine Learning based Intrusion Detection Systems, a framework is proposed in this paper to give an explanation for IDSs decision making. This framework uses Local Interpretable Model-Agnostic Explanations (LIME) coupled with Explain Like I'm five (ELI5) and Decision Tree algorithms to provide local and global explanations and improve the interpretation of IDSs. The local explanations provide the justification for the decision made on a specific input. Whereas, the global explanations provides the list of significant features and their relationship with attack traffic. In addition, this framework brings transparency in the field of ML driven IDS that might be highly significant for wide scale adoption of eXplainable AI in cyber-critical systems. Our framework is able to achieve 85 percent accuracy in classifying attack behaviour on UNSW-NB15 dataset, while at the same time displaying the feature significance ranking of the top 10 features used in the classification. 

**Abstract (ZH)**: Recent developments in Artificial Intelligence (AI) and their applications in critical industries such as healthcare, finance, and cybersecurity have led to a surge in research on AI explainability. Innovative research methods are being explored to extract meaningful insights from black-box AI systems, making decision-making technologies transparent and interpretable. Explainability is particularly crucial when AI is used in decision-making for domains such as finance, healthcare, and safety-critical systems like cybersecurity and autonomous vehicles. However, there is still ambiguity in reliable evaluations for users regarding the transparency in the explanations provided by black-box AI. To address the black-box nature of Machine Learning-based Intrusion Detection Systems (IDSs), this paper proposes a framework to explain IDSs’ decision-making processes. The framework utilizes Local Interpretable Model-Agnostic Explanations (LIME), Explain Like I'm Five (ELI5), and Decision Tree algorithms to provide both local and global explanations, thereby improving the interpretability of IDSs. Local explanations justify the decision made on a specific input, while global explanations provide a list of significant features and their relationship with attack traffic. Additionally, this framework introduces transparency in ML-driven IDS, which may be highly significant for the wide-scale adoption of Explainable AI in cyber-critical systems. Our framework achieves 85% accuracy in classifying attack behavior on the UNSW-NB15 dataset while simultaneously displaying the feature significance ranking of the top 10 features used in classification. 

---
# MC3G: Model Agnostic Causally Constrained Counterfactual Generation 

**Title (ZH)**: MC3G: 模型无关的因果约束反事实生成 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.17221)  

**Abstract**: Machine learning models increasingly influence decisions in high-stakes settings such as finance, law and hiring, driving the need for transparent, interpretable outcomes. However, while explainable approaches can help understand the decisions being made, they may inadvertently reveal the underlying proprietary algorithm: an undesirable outcome for many practitioners. Consequently, it is crucial to balance meaningful transparency with a form of recourse that clarifies why a decision was made and offers actionable steps following which a favorable outcome can be obtained. Counterfactual explanations offer a powerful mechanism to address this need by showing how specific input changes lead to a more favorable prediction. We propose Model-Agnostic Causally Constrained Counterfactual Generation (MC3G), a novel framework that tackles limitations in the existing counterfactual methods. First, MC3G is model-agnostic: it approximates any black-box model using an explainable rule-based surrogate model. Second, this surrogate is used to generate counterfactuals that produce a favourable outcome for the original underlying black box model. Third, MC3G refines cost computation by excluding the ``effort" associated with feature changes that occur automatically due to causal dependencies. By focusing only on user-initiated changes, MC3G provides a more realistic and fair representation of the effort needed to achieve a favourable outcome. We show that MC3G delivers more interpretable and actionable counterfactual recommendations compared to existing techniques all while having a lower cost. Our findings highlight MC3G's potential to enhance transparency, accountability, and practical utility in decision-making processes that incorporate machine-learning approaches. 

**Abstract (ZH)**: 基于因果约束的模型无关反事实生成（MC3G）：提高机器学习决策过程的透明性、问责制和实用性 

---
# Explainable Counterfactual Reasoning in Depression Medication Selection at Multi-Levels (Personalized and Population) 

**Title (ZH)**: 抑郁药物选择多维度（个性化和群体层面）的可解释反事实推理 

**Authors**: Xinyu Qin, Mark H. Chignell, Alexandria Greifenberger, Sachinthya Lokuge, Elssa Toumeh, Tia Sternat, Martin Katzman, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17207)  

**Abstract**: Background: This study investigates how variations in Major Depressive Disorder (MDD) symptoms, quantified by the Hamilton Rating Scale for Depression (HAM-D), causally influence the prescription of SSRIs versus SNRIs. Methods: We applied explainable counterfactual reasoning with counterfactual explanations (CFs) to assess the impact of specific symptom changes on antidepressant choice. Results: Among 17 binary classifiers, Random Forest achieved highest performance (accuracy, F1, precision, recall, ROC-AUC near 0.85). Sample-based CFs revealed both local and global feature importance of individual symptoms in medication selection. Conclusions: Counterfactual reasoning elucidates which MDD symptoms most strongly drive SSRI versus SNRI selection, enhancing interpretability of AI-based clinical decision support systems. Future work should validate these findings on more diverse cohorts and refine algorithms for clinical deployment. 

**Abstract (ZH)**: 背景：本研究探讨了通过Hamilton抑郁评定量表（HAM-D）量化的主要抑郁症症状变异如何因果影响选择SSRIs与SNRIs。方法：我们使用可解释的反事实推理与反事实解释（CFs）来评估特定症状变化对抗抑郁药选择的影响。结果：在17个二元分类器中，随机森林表现最佳（准确率、F1值、精确率、召回率、ROC-AUC接近0.85）。基于样本的反事实解释揭示了个体症状在药物选择中的局部和全局特征重要性。结论：反事实推理阐明了哪些主要抑郁症症状最强烈地驱动SSRI与SNRI的选择，增强了基于AI的临床决策支持系统的可解释性。未来工作应在更多样化的队列中验证这些发现并细化临床部署算法。 

---
# MaRVL-QA: A Benchmark for Mathematical Reasoning over Visual Landscapes 

**Title (ZH)**: MaRVL-QA：视觉景观上的数学推理基准 

**Authors**: Nilay Pande, Sahiti Yerramilli, Jayant Sravan Tamarapalli, Rynaa Grover  

**Link**: [PDF](https://arxiv.org/pdf/2508.17180)  

**Abstract**: A key frontier for Multimodal Large Language Models (MLLMs) is the ability to perform deep mathematical and spatial reasoning directly from images, moving beyond their established success in semantic description. Mathematical surface plots provide a rigorous testbed for this capability, as they isolate the task of reasoning from the semantic noise common in natural images. To measure progress on this frontier, we introduce MaRVL-QA (Mathematical Reasoning over Visual Landscapes), a new benchmark designed to quantitatively evaluate these core reasoning skills. The benchmark comprises two novel tasks: Topological Counting, identifying and enumerating features like local maxima; and Transformation Recognition, recognizing applied geometric transformations. Generated from a curated library of functions with rigorous ambiguity filtering, our evaluation on MaRVL-QA reveals that even state-of-the-art MLLMs struggle significantly, often resorting to superficial heuristics instead of robust spatial reasoning. MaRVL-QA provides a challenging new tool for the research community to measure progress, expose model limitations, and guide the development of MLLMs with more profound reasoning abilities. 

**Abstract (ZH)**: 多模态大语言模型的关键前沿领域是能够直接从图像中进行深入的数学和空间推理，超越其在语义描述上的已有成就。数学曲面图提供了这种能力的严格测试平台，因为它们将推理任务与自然图像中常见的语义噪声隔离开来。为了衡量这一领域的进展，我们引入了MaRVL-QA（基于视觉景观的数学推理问题集），这是一个新的基准，旨在定量评估这些核心推理能力。该基准包括两个新的任务：拓扑计数，识别和枚举局部极大值等特征；以及几何变换识别，识别应用的几何变换。来源于具有严格模糊性过滤的函数库，我们在MaRVL-QA上的评估表明，即使是最先进的多模态大语言模型也面临着显著挑战，常常依赖于表面的启发法而非稳健的空间推理。MaRVL-QA为研究社区提供了一个具有挑战性的新工具，用于衡量进展、揭示模型的局限性，并指导开发具有更深层次推理能力的多模态大语言模型。 

---
# Rethinking How AI Embeds and Adapts to Human Values: Challenges and Opportunities 

**Title (ZH)**: 重新思考AI如何嵌入和适应人类价值观：挑战与机遇 

**Authors**: Sz-Ting Tzeng, Frank Dignum  

**Link**: [PDF](https://arxiv.org/pdf/2508.17104)  

**Abstract**: The concepts of ``human-centered AI'' and ``value-based decision'' have gained significant attention in both research and industry. However, many critical aspects remain underexplored and require further investigation. In particular, there is a need to understand how systems incorporate human values, how humans can identify these values within systems, and how to minimize the risks of harm or unintended consequences. In this paper, we highlight the need to rethink how we frame value alignment and assert that value alignment should move beyond static and singular conceptions of values. We argue that AI systems should implement long-term reasoning and remain adaptable to evolving values. Furthermore, value alignment requires more theories to address the full spectrum of human values. Since values often vary among individuals or groups, multi-agent systems provide the right framework for navigating pluralism, conflict, and inter-agent reasoning about values. We identify the challenges associated with value alignment and indicate directions for advancing value alignment research. In addition, we broadly discuss diverse perspectives of value alignment, from design methodologies to practical applications. 

**Abstract (ZH)**: “以人为本的AI”和“基于价值的决策”概念在研究和行业领域获得了广泛关注，但许多关键方面仍有待探索并需要进一步研究。特别是，需要理解系统如何融入人类价值观、人类如何在系统中识别这些价值观以及如何最小化危害或意外后果的风险。在本文中，我们强调需要重新思考价值对齐的方式，并认为价值对齐应超越静态和单一的价值观观念。我们argue认为，AI系统应采用长期推理，并保持对演变价值观的适应性。此外，由于价值观在个体或群体之间往往不同，多智能体系统提供了应对多元主义、冲突以及智能体间关于价值观的推理的适当框架。我们指出了价值对齐面临的挑战，并指出了推进价值对齐研究的方向。此外，我们从设计方法论到实际应用广泛讨论了价值对齐的多种视角。 

---
# Solving the Min-Max Multiple Traveling Salesmen Problem via Learning-Based Path Generation and Optimal Splitting 

**Title (ZH)**: 基于学习路径生成和最优分割的MinMax多旅行售货员问题求解 

**Authors**: Wen Wang, Xiangchen Wu, Liang Wang, Hao Hu, Xianping Tao, Linghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17087)  

**Abstract**: This study addresses the Min-Max Multiple Traveling Salesmen Problem ($m^3$-TSP), which aims to coordinate tours for multiple salesmen such that the length of the longest tour is minimized. Due to its NP-hard nature, exact solvers become impractical under the assumption that $P \ne NP$. As a result, learning-based approaches have gained traction for their ability to rapidly generate high-quality approximate solutions. Among these, two-stage methods combine learning-based components with classical solvers, simplifying the learning objective. However, this decoupling often disrupts consistent optimization, potentially degrading solution quality. To address this issue, we propose a novel two-stage framework named \textbf{Generate-and-Split} (GaS), which integrates reinforcement learning (RL) with an optimal splitting algorithm in a joint training process. The splitting algorithm offers near-linear scalability with respect to the number of cities and guarantees optimal splitting in Euclidean space for any given path. To facilitate the joint optimization of the RL component with the algorithm, we adopt an LSTM-enhanced model architecture to address partial observability. Extensive experiments show that the proposed GaS framework significantly outperforms existing learning-based approaches in both solution quality and transferability. 

**Abstract (ZH)**: 本研究探讨了Min-Max Multiple Traveling Salesmen Problem ($m^3$-TSP)，目标是协调多名销售人员的行程，使得最长的行程长度最小化。由于该问题的NP难性质，在P≠NP的假设下，精确求解器变得不实用。因此，基于学习的方法因其能够快速生成高质量近似解而受到关注。在这类方法中，两阶段方法将基于学习的组件与经典求解器结合，简化了学习目标。然而，这种分离往往会打断一致的优化，可能降低解的质量。为了解决这一问题，我们提出了一种名为Generate-and-Split（GaS）的新型两阶段框架，该框架将强化学习（RL）与最优分割算法结合，在联合训练过程中进行集成。分割算法在城市数量方面具有近线性可扩展性，并能够确保在欧几里得空间中对任意给定路径实现最优分割。为促进RL组件与算法的联合优化，我们采用LSTM增强的模型架构来解决部分可观测性问题。大量实验表明，提出的GaS框架在解的质量和可迁移性方面显著优于现有基于学习的方法。 

---
# Complexity in finitary argumentation (extended version) 

**Title (ZH)**: 有限论辩的复杂性（扩展版本） 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2508.16986)  

**Abstract**: Abstract argumentation frameworks (AFs) provide a formal setting to analyze many forms of reasoning with conflicting information. While the expressiveness of general infinite AFs make them a tempting tool for modeling many kinds of reasoning scenarios, the computational intractability of solving infinite AFs limit their use, even in many theoretical applications.
We investigate the complexity of computational problems related to infinite but finitary argumentations frameworks, that is, infinite AFs where each argument is attacked by only finitely many others. Our results reveal a surprising scenario. On one hand, we see that the assumption of being finitary does not automatically guarantee a drop in complexity. However, for the admissibility-based semantics, we find a remarkable combinatorial constraint which entails a dramatic decrease in complexity.
We conclude that for many forms of reasoning, the finitary infinite AFs provide a natural setting for reasoning which balances well the competing goals of being expressive enough to be applied to many reasoning settings while being computationally tractable enough for the analysis within the framework to be useful. 

**Abstract (ZH)**: 无穷但可数的论辩框架的计算复杂性研究：一种表达性和计算效率之间的平衡 

---
# Route-and-Execute: Auditable Model-Card Matching and Specialty-Level Deployment 

**Title (ZH)**: 执行路线：可审计的模型卡匹配与专科级部署 

**Authors**: Shayan Vassef, Soorya Ram Shimegekar, Abhay Goyal, Koustuv Saha, Pi Zonooz, Navin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16839)  

**Abstract**: Clinical workflows are fragmented as a patchwork of scripts and task-specific networks that often handle triage, task selection, and model deployment. These pipelines are rarely streamlined for data science pipeline, reducing efficiency and raising operational costs. Workflows also lack data-driven model identification (from imaging/tabular inputs) and standardized delivery of model outputs. In response, we present a practical, healthcare-first framework that uses a single vision-language model (VLM) in two complementary roles. First (Solution 1), the VLM acts as an aware model-card matcher that routes an incoming image to the appropriate specialist model via a three-stage workflow (modality -> primary abnormality -> model-card id). Checks are provided by (i) stagewise prompts that allow early exit via None/Normal/Other and (ii) a stagewise answer selector that arbitrates between the top-2 candidates at each stage, reducing the chance of an incorrect selection and aligning the workflow with clinical risk tolerance. Second (Solution 2), we fine-tune the VLM on specialty-specific datasets ensuring a single model covers multiple downstream tasks within each specialty, maintaining performance while simplifying deployment. Across gastroenterology, hematology, ophthalmology, and pathology, our single-model deployment matches or approaches specialized baselines.
Compared with pipelines composed of many task-specific agents, this approach shows that one VLM can both decide and do. It may reduce effort by data scientists, shorten monitoring, increase the transparency of model selection (with per-stage justifications), and lower integration overhead. 

**Abstract (ZH)**: 临床工作流程碎片化为一系列脚本和任务特定网络的拼贴，常用于处理分诊、任务选择和模型部署。这些管道很少专门针对数据科学管道优化，从而降低了效率并增加了运营成本。工作流程中缺乏基于数据的模型识别（从影像/表格式输入）以及模型输出的标准交付。为此，我们提出了一种实用的、以医疗健康为主的框架，该框架利用单一的视觉-语言模型（VLM）在两种互补的角色中发挥作用。首先（解决方案1），VLM 作为一个知情的模型卡匹配器，通过三阶段工作流（模态->主要异常->模型卡ID）将输入图像路由到适当的专科模型，通过阶段提示允许早期退出（无/正常/其他）和阶段回答选择器在每个阶段选择前二名候选人之间仲裁，从而减少错误选择并使工作流与临床风险容忍度保持一致。其次（解决方案2），我们针对各专科特定的数据集微调VLM，确保一个模型能够涵盖每个专科内的多个下游任务，从而保持性能并简化部署。在胃肠病学、血液学、眼科学和病理学领域，我们的单模型部署匹配或接近专科基线。与由多个任务特定代理组成的管道相比，这种方法显示了一个VLM可以两者兼备。它可以通过减少数据科学家的努力、缩短监控时间、增加模型选择的透明度（每个阶段的解释）并降低集成成本来发挥作用。 

---
# PuzzleJAX: A Benchmark for Reasoning and Learning 

**Title (ZH)**: PuzzleJAX: 一个推理与学习的标准评测基准 

**Authors**: Sam Earle, Graham Todd, Yuchen Li, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2508.16821)  

**Abstract**: We introduce PuzzleJAX, a GPU-accelerated puzzle game engine and description language designed to support rapid benchmarking of tree search, reinforcement learning, and LLM reasoning abilities. Unlike existing GPU-accelerated learning environments that provide hard-coded implementations of fixed sets of games, PuzzleJAX allows dynamic compilation of any game expressible in its domain-specific language (DSL). This DSL follows PuzzleScript, which is a popular and accessible online game engine for designing puzzle games. In this paper, we validate in PuzzleJAX several hundred of the thousands of games designed in PuzzleScript by both professional designers and casual creators since its release in 2013, thereby demonstrating PuzzleJAX's coverage of an expansive, expressive, and human-relevant space of tasks. By analyzing the performance of search, learning, and language models on these games, we show that PuzzleJAX can naturally express tasks that are both simple and intuitive to understand, yet often deeply challenging to master, requiring a combination of control, planning, and high-level insight. 

**Abstract (ZH)**: PuzzleJAX：一种加速的谜题游戏引擎及描述语言，用于支持树搜索、强化学习和LLM推理能力的快速基准测试 

---
# Explainable AI for Predicting and Understanding Mathematics Achievement: A Cross-National Analysis of PISA 2018 

**Title (ZH)**: 可解释的人工智能在预测和理解数学成就中的应用：基于PISA 2018的跨国家分析 

**Authors**: Liu Liu, Rui Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.16747)  

**Abstract**: Understanding the factors that shape students' mathematics performance is vital for designing effective educational policies. This study applies explainable artificial intelligence (XAI) techniques to PISA 2018 data to predict math achievement and identify key predictors across ten countries (67,329 students). We tested four models: Multiple Linear Regression (MLR), Random Forest (RF), CATBoost, and Artificial Neural Networks (ANN), using student, family, and school variables. Models were trained on 70% of the data (with 5-fold cross-validation) and tested on 30%, stratified by country. Performance was assessed with R^2 and Mean Absolute Error (MAE). To ensure interpretability, we used feature importance, SHAP values, and decision tree visualizations. Non-linear models, especially RF and ANN, outperformed MLR, with RF balancing accuracy and generalizability. Key predictors included socio-economic status, study time, teacher motivation, and students' attitudes toward mathematics, though their impact varied across countries. Visual diagnostics such as scatterplots of predicted vs actual scores showed RF and CATBoost aligned closely with actual performance. Findings highlight the non-linear and context-dependent nature of achievement and the value of XAI in educational research. This study uncovers cross-national patterns, informs equity-focused reforms, and supports the development of personalized learning strategies. 

**Abstract (ZH)**: 理解塑造学生数学表现的因素对于设计有效的教育政策至关重要。本研究采用可解释的人工智能（XAI）技术对PISA 2018数据进行分析，预测数学成就并识别十个参与国家（67,329名学生）的关键预测因子。我们测试了四种模型：多元线性回归（MLR）、随机森林（RF）、CATBoost和人工神经网络（ANN），使用学生、家庭和学校变量。模型在70%的数据上进行训练（5折交叉验证），并在30%的数据上进行测试，按国家分层。性能用R²和平均绝对误差（MAE）进行评估。为了确保可解释性，我们使用了特征重要性、SHAP值和决策树可视化。特别是RF和ANN等非线性模型优于MLR，RF在准确性和普适性方面取得了平衡。关键预测因子包括社会经济地位、学习时间、教师动机以及学生对数学的态度，但其影响因国家而异。散点图等可视化诊断显示RF和CATBoost与实际表现高度一致。研究结果强调了成就的非线性和情境依赖性，并突显了XAI在教育研究中的价值。本研究揭示了跨国家的模式，为促进教育公平的改革提供了信息，并支持个性化学习策略的开发。 

---
# Revisiting Rule-Based Stuttering Detection: A Comprehensive Analysis of Interpretable Models for Clinical Applications 

**Title (ZH)**: 基于规则的颤抖检测 revisit：可解释模型在临床应用中的全面分析 

**Authors**: Eric Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16681)  

**Abstract**: Stuttering affects approximately 1% of the global population, impacting communication and quality of life. While recent advances in deep learning have pushed the boundaries of automatic speech dysfluency detection, rule-based approaches remain crucial for clinical applications where interpretability and transparency are paramount. This paper presents a comprehensive analysis of rule-based stuttering detection systems, synthesizing insights from multiple corpora including UCLASS, FluencyBank, and SEP-28k. We propose an enhanced rule-based framework that incorporates speaking-rate normalization, multi-level acoustic feature analysis, and hierarchical decision structures. Our approach achieves competitive performance while maintaining complete interpretability-critical for clinical adoption. We demonstrate that rule-based systems excel particularly in prolongation detection (97-99% accuracy) and provide stable performance across varying speaking rates. Furthermore, we show how these interpretable models can be integrated with modern machine learning pipelines as proposal generators or constraint modules, bridging the gap between traditional speech pathology practices and contemporary AI systems. Our analysis reveals that while neural approaches may achieve marginally higher accuracy in unconstrained settings, rule-based methods offer unique advantages in clinical contexts where decision auditability, patient-specific tuning, and real-time feedback are essential. 

**Abstract (ZH)**: stuttering 影响全球约 1%的人口，影响沟通和生活质量。尽管深度学习近期在自动语音不畅检测方面取得了进展，但基于规则的方法仍对于强调可解释性和透明度的临床应用至关重要。本文综合分析了多种基于规则的 stuttering 检测系统，包括 UCLASS、FluencyBank 和 SEP-28k 数据集中的洞察。我们提出了一种增强的基于规则的框架，结合了发音速率规范化、多级声学特征分析和分层决策结构。该方法在保持完全可解释性的同时实现了竞争性的性能，这对于临床应用至关重要。我们证明基于规则的系统在延长音检测方面表现出色（准确率 97-99%），并在不同发音速率下表现出稳定的性能。此外，我们展示了这些可解释模型如何与现代机器学习管道集成，作为建议生成器或约束模块，从而弥合传统言语病理学实践与现代人工智能系统之间的差距。我们的分析表明，在不受约束的环境中，神经方法可能在准确率上略高，但基于规则的方法在临床环境中具有独特优势，因为这些环境中决策审计、患者特定调整和实时反馈至关重要。 

---
# ANO : Faster is Better in Noisy Landscape 

**Title (ZH)**: ANO : 在噪声环境中，更快更好 

**Authors**: Adrien Kegreisz  

**Link**: [PDF](https://arxiv.org/pdf/2508.18258)  

**Abstract**: Stochastic optimizers are central to deep learning, yet widely used methods such as Adam and Adan can degrade in non-stationary or noisy environments, partly due to their reliance on momentum-based magnitude estimates. We introduce Ano, a novel optimizer that decouples direction and magnitude: momentum is used for directional smoothing, while instantaneous gradient magnitudes determine step size. This design improves robustness to gradient noise while retaining the simplicity and efficiency of first-order methods. We further propose Anolog, which removes sensitivity to the momentum coefficient by expanding its window over time via a logarithmic schedule. We establish non-convex convergence guarantees with a convergence rate similar to other sign-based methods, and empirically show that Ano provides substantial gains in noisy and non-stationary regimes such as reinforcement learning, while remaining competitive on low-noise tasks such as standard computer vision benchmarks. 

**Abstract (ZH)**: 随机优化器是深度学习的核心，然而广泛使用的Adam和Adan等方法在非平稳或噪声环境中可能会退化，部分原因是它们依赖于基于动量的幅度估算。我们提出了一种新型优化器Ano，将方向和幅度解耦：动量用于方向平滑，而瞬时梯度幅度决定步长大小。这种设计提高了对梯度噪声的鲁棒性，同时保留了一阶方法的简洁性和效率。我们还提出了Anolog，通过使用对数时间表逐步扩展动量窗口，消除了对动量系数的敏感性。我们建立了非凸收敛保证，并且实验表明，在强化学习等噪声和非平稳环境中，Ano提供了显著的改进，同时在低噪声任务如标准计算机视觉基准测试中保持竞争力。 

---
# KillChainGraph: ML Framework for Predicting and Mapping ATT&CK Techniques 

**Title (ZH)**: KillChainGraph：用于预测和映射ATT&CK技术的机器学习框架 

**Authors**: Chitraksh Singh, Monisha Dhanraj, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18230)  

**Abstract**: The escalating complexity and volume of cyberattacks demand proactive detection strategies that go beyond traditional rule-based systems. This paper presents a phase-aware, multi-model machine learning framework that emulates adversarial behavior across the seven phases of the Cyber Kill Chain using the MITRE ATT&CK Enterprise dataset. Techniques are semantically mapped to phases via ATTACK-BERT, producing seven phase-specific datasets. We evaluate LightGBM, a custom Transformer encoder, fine-tuned BERT, and a Graph Neural Network (GNN), integrating their outputs through a weighted soft voting ensemble. Inter-phase dependencies are modeled using directed graphs to capture attacker movement from reconnaissance to objectives. The ensemble consistently achieved the highest scores, with F1-scores ranging from 97.47% to 99.83%, surpassing GNN performance (97.36% to 99.81%) by 0.03%--0.20% across phases. This graph-driven, ensemble-based approach enables interpretable attack path forecasting and strengthens proactive cyber defense. 

**Abstract (ZH)**: 不断提高的网络攻击复杂性和体积要求超越传统规则系统的前瞻检测策略。本文提出了一种相位意识的多模型机器学习框架，利用MITRE ATT&CK Enterprise数据集在网络杀伤链的七个阶段中模拟对手行为。通过ATTACK-BERT将技术语义映射到各个阶段，产生了七个阶段特定的数据集。我们评估了LightGBM、一个自定义Transformer编码器、微调的BERT以及图神经网络（GNN），通过加权软投票集成整合其输出。使用有向图建模阶段间的依赖关系，以捕捉攻击者从侦察到目标的行为。集成框架在所有阶段均表现出最高性能，F1得分范围从97.47%到99.83%，相较于GNN（97.36%到99.81%）在各个阶段高出0.03%–0.20%。这种基于图的集成方法实现了可解释的攻击路径预测，并增强了主动网络防御。 

---
# Deep Learning and Matrix Completion-aided IoT Network Localization in the Outlier Scenarios 

**Title (ZH)**: 深度学习和矩阵填充辅助的物联网网络定位方法在异常场景中 

**Authors**: Sunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.18225)  

**Abstract**: In this paper, we propose a deep learning and matrix completion aided approach for recovering an outlier contaminated Euclidean distance matrix D in IoT network localization. Unlike conventional localization techniques that search the solution over a whole set of matrices, the proposed technique restricts the search to the set of Euclidean distance matrices. Specifically, we express D as a function of the sensor coordinate matrix X that inherently satisfies the unique properties of D, and then jointly recover D and X using a deep neural network. To handle outliers effectively, we model them as a sparse matrix L and add a regularization term of L into the optimization problem. We then solve the problem by alternately updating X, D, and L. Numerical experiments demonstrate that the proposed technique can recover the location information of sensors accurately even in the presence of outliers. 

**Abstract (ZH)**: 基于深学习和矩阵完成的物联网网络定位中受污染欧几里得距离矩阵恢复方法 

---
# Why Synthetic Isn't Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation 

**Title (ZH)**: 合成的还没成为现实：接触中心对话生成的诊断框架 

**Authors**: Rishikesh Devanathan, Varun Nathan, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18210)  

**Abstract**: Synthetic transcript generation is critical in contact center domains, where privacy and data scarcity limit model training and evaluation. Unlike prior synthetic dialogue generation work on open-domain or medical dialogues, contact center conversations are goal-oriented, role-asymmetric, and behaviorally complex, featuring disfluencies, ASR noise, and compliance-driven agent actions. In deployments where transcripts are unavailable, standard pipelines still yield derived call attributes such as Intent Summaries, Topic Flow, and QA Evaluation Forms. We leverage these as supervision signals to guide generation. To assess the quality of such outputs, we introduce a diagnostic framework of 18 linguistically and behaviorally grounded metrics for comparing real and synthetic transcripts. We benchmark four language-agnostic generation strategies, from simple prompting to characteristic-aware multi-stage approaches, alongside reference-free baselines. Results reveal persistent challenges: no method excels across all traits, with notable deficits in disfluency, sentiment, and behavioral realism. Our diagnostic tool exposes these gaps, enabling fine-grained evaluation and stress testing of synthetic dialogue across languages. 

**Abstract (ZH)**: 合成转录生成在客服中心领域至关重要，由于隐私和数据稀缺限制了模型的训练和评估。与先前针对开放领域或医疗对话的合成对话生成工作不同，客服中心的对话具有目标导向性、角色不对称性和行为复杂性，包含口吃、ASR噪音和合规驱动的代理行动。在转录不可用的部署中，标准管道仍能生成诸如意图摘要、话题流和问答评估表等衍生通话属性。我们利用这些作为监督信号来指导生成。为了评估这些输出的质量，我们引入了一种包含18个语用和行为依据的诊断框架，用于比较真实和合成的转录。我们将四种语言无关的生成策略，从简单的提示到具有特征意识的多阶段方法，与无参考基线进行了基准测试。结果揭示了持续的挑战：没有任何方法在所有特性上都表现出色，尤其是在口吃、情感和行为现实性方面存在明显缺陷。我们的诊断工具暴露了这些差距，使我们可以对跨语言的合成对话进行精细评估和压力测试。 

---
# Amortized Sampling with Transferable Normalizing Flows 

**Title (ZH)**: 转移可变形正则化流的渐进采样 

**Authors**: Charlie B. Tan, Majdi Hassan, Leon Klein, Saifuddin Syed, Dominique Beaini, Michael M. Bronstein, Alexander Tong, Kirill Neklyudov  

**Link**: [PDF](https://arxiv.org/pdf/2508.18175)  

**Abstract**: Efficient equilibrium sampling of molecular conformations remains a core challenge in computational chemistry and statistical inference. Classical approaches such as molecular dynamics or Markov chain Monte Carlo inherently lack amortization; the computational cost of sampling must be paid in-full for each system of interest. The widespread success of generative models has inspired interest into overcoming this limitation through learning sampling algorithms. Despite performing on par with conventional methods when trained on a single system, learned samplers have so far demonstrated limited ability to transfer across systems. We prove that deep learning enables the design of scalable and transferable samplers by introducing Prose, a 280 million parameter all-atom transferable normalizing flow trained on a corpus of peptide molecular dynamics trajectories up to 8 residues in length. Prose draws zero-shot uncorrelated proposal samples for arbitrary peptide systems, achieving the previously intractable transferability across sequence length, whilst retaining the efficient likelihood evaluation of normalizing flows. Through extensive empirical evaluation we demonstrate the efficacy of Prose as a proposal for a variety of sampling algorithms, finding a simple importance sampling-based finetuning procedure to achieve superior performance to established methods such as sequential Monte Carlo on unseen tetrapeptides. We open-source the Prose codebase, model weights, and training dataset, to further stimulate research into amortized sampling methods and finetuning objectives. 

**Abstract (ZH)**: 高效的分子构象均衡采样仍然是计算化学和统计推断中的核心挑战。通过Prose实现大规模且可转移的采样算法ucid 

---
# The Computational Complexity of Satisfiability in State Space Models 

**Title (ZH)**: 状态空间模型中的 satisfiability 计算复杂性 

**Authors**: Eric Alsmann, Martin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2508.18162)  

**Abstract**: We analyse the complexity of the satisfiability problem ssmSAT for State Space Models (SSM), which asks whether an input sequence can lead the model to an accepting configuration. We find that ssmSAT is undecidable in general, reflecting the computational power of SSM. Motivated by practical settings, we identify two natural restrictions under which ssmSAT becomes decidable and establish corresponding complexity bounds. First, for SSM with bounded context length, ssmSAT is NP-complete when the input length is given in unary and in NEXPTIME (and PSPACE-hard) when the input length is given in binary. Second, for quantised SSM operating over fixed-width arithmetic, ssmSAT is PSPACE-complete resp. in EXPSPACE depending on the bit-width encoding. While these results hold for diagonal gated SSM we also establish complexity bounds for time-invariant SSM. Our results establish a first complexity landscape for formal reasoning in SSM and highlight fundamental limits and opportunities for the verification of SSM-based language models. 

**Abstract (ZH)**: 我们分析了状态空间模型（SSM）的满足性问题ssmSAT的复杂性，该问题询问是否有一个输入序列可以使模型达到接受配置。我们发现ssmSAT通常是不可判定的，反映了SSM的计算能力。受到实际应用的启发，我们识别了两种自然的限制条件，在这两种条件下ssmSAT变为可判定，并建立了相应的时间复杂度界限。首先，对于具有有界上下文长度的SSM，当输入长度以 unary 形式给出时，ssmSAT 是 NP 完全问题；当输入长度以二进制形式给出时，ssmSAT 是 NEXPTIME（和 PSPACE-硬）问题。其次，对于操作固定宽度算术的量化 SSM，ssmSAT 是 PSPACE 完全问题，具体取决于位宽编码，可能在 EXPSPACE 中运行。尽管这些结果适用于对角门控 SSM，我们还为时不变 SSM 建立了复杂性界限。我们的结果为 SSM 中的形式化推理提供了首个复杂性景观，并突显了 SSM 基础语言模型验证的基本限制和机会。 

---
# Assessing the Noise Robustness of Class Activation Maps: A Framework for Reliable Model Interpretability 

**Title (ZH)**: 评估类激活图的噪声鲁棒性：一种可靠的模型可解释性框架 

**Authors**: Syamantak Sarkar, Revoti P. Bora, Bhupender Kaushal, Sudhish N George, Kiran Raja  

**Link**: [PDF](https://arxiv.org/pdf/2508.18154)  

**Abstract**: Class Activation Maps (CAMs) are one of the important methods for visualizing regions used by deep learning models. Yet their robustness to different noise remains underexplored. In this work, we evaluate and report the resilience of various CAM methods for different noise perturbations across multiple architectures and datasets. By analyzing the influence of different noise types on CAM explanations, we assess the susceptibility to noise and the extent to which dataset characteristics may impact explanation stability. The findings highlight considerable variability in noise sensitivity for various CAMs. We propose a robustness metric for CAMs that captures two key properties: consistency and responsiveness. Consistency reflects the ability of CAMs to remain stable under input perturbations that do not alter the predicted class, while responsiveness measures the sensitivity of CAMs to changes in the prediction caused by such perturbations. The metric is evaluated empirically across models, different perturbations, and datasets along with complementary statistical tests to exemplify the applicability of our proposed approach. 

**Abstract (ZH)**: Class Activation Maps (CAMs)在不同噪声干扰下的鲁棒性研究：多架构与多数据集上的评估与报告 

---
# Towards Continual Visual Anomaly Detection in the Medical Domain 

**Title (ZH)**: 医学领域持续视觉异常检测的研究 

**Authors**: Manuel Barusco, Francesco Borsatti, Nicola Beda, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2508.18013)  

**Abstract**: Visual Anomaly Detection (VAD) seeks to identify abnormal images and precisely localize the corresponding anomalous regions, relying solely on normal data during training. This approach has proven essential in domains such as manufacturing and, more recently, in the medical field, where accurate and explainable detection is critical. Despite its importance, the impact of evolving input data distributions over time has received limited attention, even though such changes can significantly degrade model performance. In particular, given the dynamic and evolving nature of medical imaging data, Continual Learning (CL) provides a natural and effective framework to incrementally adapt models while preserving previously acquired knowledge. This study explores for the first time the application of VAD models in a CL scenario for the medical field. In this work, we utilize a CL version of the well-established PatchCore model, called PatchCoreCL, and evaluate its performance using BMAD, a real-world medical imaging dataset with both image-level and pixel-level annotations. Our results demonstrate that PatchCoreCL is an effective solution, achieving performance comparable to the task-specific models, with a forgetting value less than a 1%, highlighting the feasibility and potential of CL for adaptive VAD in medical imaging. 

**Abstract (ZH)**: 视觉异常检测在持续学习框架下的医疗领域应用研究 

---
# Previously on... Automating Code Review 

**Title (ZH)**: 此前研究... 自动化代码审查 

**Authors**: Robert Heumüller, Frank Ortmeier  

**Link**: [PDF](https://arxiv.org/pdf/2508.18003)  

**Abstract**: Modern Code Review (MCR) is a standard practice in software engineering, yet it demands substantial time and resource investments. Recent research has increasingly explored automating core review tasks using machine learning (ML) and deep learning (DL). As a result, there is substantial variability in task definitions, datasets, and evaluation procedures. This study provides the first comprehensive analysis of MCR automation research, aiming to characterize the field's evolution, formalize learning tasks, highlight methodological challenges, and offer actionable recommendations to guide future research. Focusing on the primary code review tasks, we systematically surveyed 691 publications and identified 24 relevant studies published between May 2015 and April 2024. Each study was analyzed in terms of tasks, models, metrics, baselines, results, validity concerns, and artifact availability. In particular, our analysis reveals significant potential for standardization, including 48 task metric combinations, 22 of which were unique to their original paper, and limited dataset reuse. We highlight challenges and derive concrete recommendations for examples such as the temporal bias threat, which are rarely addressed so far. Our work contributes to a clearer overview of the field, supports the framing of new research, helps to avoid pitfalls, and promotes greater standardization in evaluation practices. 

**Abstract (ZH)**: 现代代码审查自动化研究综述：特征、挑战与标准化建议 

---
# A Feminist Account of Intersectional Algorithmic Fairness 

**Title (ZH)**: 女性主义视角下的交集算法公平性 

**Authors**: Marie Mirsch, Laila Wegner, Jonas Strube, Carmen Leicht-Scholten  

**Link**: [PDF](https://arxiv.org/pdf/2508.17944)  

**Abstract**: Intersectionality has profoundly influenced research and political action by revealing how interconnected systems of privilege and oppression influence lived experiences, yet its integration into algorithmic fairness research remains limited. Existing approaches often rely on single-axis or formal subgroup frameworks that risk oversimplifying social realities and neglecting structural inequalities. We propose Substantive Intersectional Algorithmic Fairness, extending Green's (2022) notion of substantive algorithmic fairness with insights from intersectional feminist theory. Building on this foundation, we introduce ten desiderata within the ROOF methodology to guide the design, assessment, and deployment of algorithmic systems in ways that address systemic inequities while mitigating harms to intersectionally marginalized communities. Rather than prescribing fixed operationalizations, these desiderata encourage reflection on assumptions of neutrality, the use of protected attributes, the inclusion of multiply marginalized groups, and enhancing algorithmic systems' potential. Our approach emphasizes that fairness cannot be separated from social context, and that in some cases, principled non-deployment may be necessary. By bridging computational and social science perspectives, we provide actionable guidance for more equitable, inclusive, and context-sensitive intersectional algorithmic practices. 

**Abstract (ZH)**: 实质性交叠算法公平：一种基于交叠女权理论的系统化指导框架 

---
# AMELIA: A Family of Multi-task End-to-end Language Models for Argumentation 

**Title (ZH)**: AMELIA：论辩领域多任务端到端语言模型系列 

**Authors**: Henri Savigny, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17926)  

**Abstract**: Argument mining is a subfield of argumentation that aims to automatically extract argumentative structures and their relations from natural language texts. This paper investigates how a single large language model can be leveraged to perform one or several argument mining tasks. Our contributions are two-fold. First, we construct a multi-task dataset by surveying and converting 19 well-known argument mining datasets from the literature into a unified format. Second, we explore various training strategies using Meta AI's Llama-3.1-8B-Instruct model: (1) fine-tuning on individual tasks, (2) fine-tuning jointly on multiple tasks, and (3) merging models fine-tuned separately on individual tasks. Our experiments show that task-specific fine-tuning significantly improves individual performance across all tasks. Moreover, multi-task fine-tuning maintains strong performance without degradation, suggesting effective transfer learning across related tasks. Finally, we demonstrate that model merging offers a viable compromise: it yields competitive performance while mitigating the computational costs associated with full multi-task fine-tuning. 

**Abstract (ZH)**: 论文本论证结构与关系的自动提取：单一大型语言模型在多任务下的应用探究 

---
# A Defect Classification Framework for AI-Based Software Systems (AI-ODC) 

**Title (ZH)**: 基于AI的软件系统缺陷分类框架（AI-ODC） 

**Authors**: Mohammed O. Alannsary  

**Link**: [PDF](https://arxiv.org/pdf/2508.17900)  

**Abstract**: Artificial Intelligence has gained a lot of attention recently, it has been utilized in several fields ranging from daily life activities, such as responding to emails and scheduling appointments, to manufacturing and automating work activities. Artificial Intelligence systems are mainly implemented as software solutions, and it is essential to discover and remove software defects to assure its quality using defect analysis which is one of the major activities that contribute to software quality. Despite the proliferation of AI-based systems, current defect analysis models fail to capture their unique attributes. This paper proposes a framework inspired by the Orthogonal Defect Classification (ODC) paradigm and enables defect analysis of Artificial Intelligence systems while recognizing its special attributes and characteristics. This study demonstrated the feasibility of modifying ODC for AI systems to classify its defects. The ODC was adjusted to accommodate the Data, Learning, and Thinking aspects of AI systems which are newly introduced classification dimensions. This adjustment involved the introduction of an additional attribute to the ODC attributes, the incorporation of a new severity level, and the substitution of impact areas with characteristics pertinent to AI systems. The framework was showcased by applying it to a publicly available Machine Learning bug dataset, with results analyzed through one-way and two-way analysis. The case study indicated that defects occurring during the Learning phase were the most prevalent and were significantly linked to high-severity classifications. In contrast, defects identified in the Thinking phase had a disproportionate effect on trustworthiness and accuracy. These findings illustrate AIODC's capability to identify high-risk defect categories and inform focused quality assurance measures. 

**Abstract (ZH)**: 人工智能近年来引起了广泛关注，已被应用于从日常活动到制造业和自动化工作的多个领域。人工智能系统主要作为软件解决方案实施，发现并消除软件缺陷以保证其质量是缺陷分析的关键活动之一。尽管人工智能基系统得到了广泛应用，当前的缺陷分析模型仍未捕捉到其独特属性。本文提出了一种灵感来自正交缺陷分类（ODC）范式的框架，能够在识别和分析人工智能系统缺陷的同时，认识到其特殊属性和特征。该研究证明了将ODC修改应用于人工智能系统以对其进行分类是可行的。ODC被调整以适应人工智能系统的数据、学习和思考三大方面，引入了新的严重程度等级，并用与人工智能系统相关的重要特征替代影响区域。该框架通过将其应用于一个公开的机器学习错误数据集得到展示，结果通过单向和双向分析进行分析。案例研究显示，在学习阶段出现的缺陷最为普遍，并且与高严重性分类密切相关。相比之下，识别于思考阶段的缺陷对可信性和准确性产生了不成比例的影响。这些发现表明AIODC能够识别高风险缺陷类别，并为有针对性的质量保证措施提供信息。 

---
# FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation 

**Title (ZH)**: FasterVoiceGrad: 基于对抗扩散转换蒸馏的一步扩散语音转换加速方法 

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17868)  

**Abstract**: A diffusion-based voice conversion (VC) model (e.g., VoiceGrad) can achieve high speech quality and speaker similarity; however, its conversion process is slow owing to iterative sampling. FastVoiceGrad overcomes this limitation by distilling VoiceGrad into a one-step diffusion model. However, it still requires a computationally intensive content encoder to disentangle the speaker's identity and content, which slows conversion. Therefore, we propose FasterVoiceGrad, a novel one-step diffusion-based VC model obtained by simultaneously distilling a diffusion model and content encoder using adversarial diffusion conversion distillation (ADCD), where distillation is performed in the conversion process while leveraging adversarial and score distillation training. Experimental evaluations of one-shot VC demonstrated that FasterVoiceGrad achieves competitive VC performance compared to FastVoiceGrad, with 6.6-6.9 and 1.8 times faster speed on a GPU and CPU, respectively. 

**Abstract (ZH)**: 基于扩散的声音转换（VC）模型（如VoiceGrad）可以实现高质量的语音和高speaker相似度，但由于迭代采样的原因，其转换过程较慢。FastVoiceGrad通过将VoiceGrad提炼成一步扩散模型来克服这一限制。然而，它仍然需要一个计算密集的内容编码器来分离说话人身份和内容，这会减慢转换速度。因此，我们提出了FasterVoiceGrad，这是一种通过对抗扩散转换提炼同时提炼扩散模型和内容编码器获得的新颖一步扩散基于VC模型，其中在转换过程中利用对抗性和分数提炼训练进行提炼。实验结果显示，FasterVoiceGrad在单次发声转换上的性能与FastVoiceGrad相当，分别在GPU和CPU上快6.6-6.9倍和1.8倍。 

---
# Ada-TransGNN: An Air Quality Prediction Model Based On Adaptive Graph Convolutional Networks 

**Title (ZH)**: Ada-TransGNN：基于自适应图卷积网络的空气质量预测模型 

**Authors**: Dan Wang, Feng Jiang, Zhanquan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17867)  

**Abstract**: Accurate air quality prediction is becoming increasingly important in the environmental field. To address issues such as low prediction accuracy and slow real-time updates in existing models, which lead to lagging prediction results, we propose a Transformer-based spatiotemporal data prediction method (Ada-TransGNN) that integrates global spatial semantics and temporal behavior. The model constructs an efficient and collaborative spatiotemporal block set comprising a multi-head attention mechanism and a graph convolutional network to extract dynamically changing spatiotemporal dependency features from complex air quality monitoring data. Considering the interaction relationships between different monitoring points, we propose an adaptive graph structure learning module, which combines spatiotemporal dependency features in a data-driven manner to learn the optimal graph structure, thereby more accurately capturing the spatial relationships between monitoring points. Additionally, we design an auxiliary task learning module that enhances the decoding capability of temporal relationships by integrating spatial context information into the optimal graph structure representation, effectively improving the accuracy of prediction results. We conducted comprehensive evaluations on a benchmark dataset and a novel dataset (Mete-air). The results demonstrate that our model outperforms existing state-of-the-art prediction models in short-term and long-term predictions. 

**Abstract (ZH)**: 基于Transformer的空间时序数据预测方法（Ada-TransGNN）：融合全局空间 semantics 和时序行为的空气质量预测 

---
# Limits of message passing for node classification: How class-bottlenecks restrict signal-to-noise ratio 

**Title (ZH)**: 消息传递在节点分类中的限制：类瓶颈如何限制信噪比 

**Authors**: Jonathan Rubin, Sahil Loomba, Nick S. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2508.17822)  

**Abstract**: Message passing neural networks (MPNNs) are powerful models for node classification but suffer from performance limitations under heterophily (low same-class connectivity) and structural bottlenecks in the graph. We provide a unifying statistical framework exposing the relationship between heterophily and bottlenecks through the signal-to-noise ratio (SNR) of MPNN representations. The SNR decomposes model performance into feature-dependent parameters and feature-independent sensitivities. We prove that the sensitivity to class-wise signals is bounded by higher-order homophily -- a generalisation of classical homophily to multi-hop neighbourhoods -- and show that low higher-order homophily manifests locally as the interaction between structural bottlenecks and class labels (class-bottlenecks). Through analysis of graph ensembles, we provide a further quantitative decomposition of bottlenecking into underreaching (lack of depth implying signals cannot arrive) and oversquashing (lack of breadth implying signals arriving on fewer paths) with closed-form expressions. We prove that optimal graph structures for maximising higher-order homophily are disjoint unions of single-class and two-class-bipartite clusters. This yields BRIDGE, a graph ensemble-based rewiring algorithm that achieves near-perfect classification accuracy across all homophily regimes on synthetic benchmarks and significant improvements on real-world benchmarks, by eliminating the ``mid-homophily pitfall'' where MPNNs typically struggle, surpassing current standard rewiring techniques from the literature. Our framework, whose code we make available for public use, provides both diagnostic tools for assessing MPNN performance, and simple yet effective methods for enhancing performance through principled graph modification. 

**Abstract (ZH)**: 统一统计框架下消息传递神经网络在异质性和结构瓶颈中的性能分析与优化 

---
# Limitations of Normalization in Attention Mechanism 

**Title (ZH)**: 注意力机制中归一化的限制 

**Authors**: Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2508.17821)  

**Abstract**: This paper investigates the limitations of the normalization in attention mechanisms. We begin with a theoretical framework that enables the identification of the model's selective ability and the geometric separation involved in token selection. Our analysis includes explicit bounds on distances and separation criteria for token vectors under softmax scaling. Through experiments with pre-trained GPT-2 model, we empirically validate our theoretical results and analyze key behaviors of the attention mechanism. Notably, we demonstrate that as the number of selected tokens increases, the model's ability to distinguish informative tokens declines, often converging toward a uniform selection pattern. We also show that gradient sensitivity under softmax normalization presents challenges during training, especially at low temperature settings. These findings advance current understanding of softmax-based attention mechanism and motivate the need for more robust normalization and selection strategies in future attention architectures. 

**Abstract (ZH)**: 本文探讨了注意力机制中归一化方法的局限性。我们以一个理论框架为基础，能够识别模型的选择能力和词元选择中的几何分离。我们的分析包括在softmax缩放下的词元向量距离和分离标准的显式边界。通过使用预训练的GPT-2模型进行实验，我们 empirically 验证了理论结果，并分析了注意力机制的关键行为。值得注意的是，我们证明随着选定词元数量的增加，模型区分信息性词元的能力下降，通常会朝向均匀选择模式收敛。我们还展示在softmax归一化下的梯度敏感性会在训练中遇到挑战，特别是在低温度设置下。这些发现推进了对基于softmax的注意力机制的理解，并激发了对未来注意力架构中更稳健的归一化和选择策略的需求。 

---
# UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization 

**Title (ZH)**: UniSino：基于物理驱动的基础模型用于通用CT sinogram标准化 

**Authors**: Xingyu Ai, Shaoyu Wang, Zhiyuan Jia, Ao Xu, Hongming Shan, Jianhua Ma, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17816)  

**Abstract**: During raw-data acquisition in CT imaging, diverse factors can degrade the collected sinograms, with undersampling and noise leading to severe artifacts and noise in reconstructed images and compromising diagnostic accuracy. Conventional correction methods rely on manually designed algorithms or fixed empirical parameters, but these approaches often lack generalizability across heterogeneous artifact types. To address these limitations, we propose UniSino, a foundation model for universal CT sinogram standardization. Unlike existing foundational models that operate in image domain, UniSino directly standardizes data in the projection domain, which enables stronger generalization across diverse undersampling scenarios. Its training framework incorporates the physical characteristics of sinograms, enhancing generalization and enabling robust performance across multiple subtasks spanning four benchmark datasets. Experimental results demonstrate thatUniSino achieves superior reconstruction quality both single and mixed undersampling case, demonstrating exceptional robustness and generalization in sinogram enhancement for CT imaging. The code is available at: this https URL. 

**Abstract (ZH)**: 在CT成像的数据采集过程中，多种因素会导致采集到的sinogram降质，采样不足和噪声会导致重建图像出现严重的伪影和噪声，从而影响诊断准确性。常规的校正方法依赖于人工设计的算法或固定的经验参数，但这些方法往往缺乏对异质伪影类型的普适性。为了解决这些限制，我们提出UniSino，这是一种用于通用CT sinogram标准化的基础模型。与现有的在图像域工作的基础模型不同，UniSino直接在投影域标准化数据，这使其能够在多种欠采样场景中展现出更强的泛化能力。其训练框架结合了sinogram的物理特性，增强了泛化能力，并能够跨四个基准数据集的多项子任务获得稳健的表现。实验结果表明，UniSino在单一和混合欠采样情况下均能实现优秀的重建质量，显示出在CT成像中对sinogram增强的卓越稳健性和泛化能力。代码可在以下链接获取：this https URL。 

---
# Proximal Supervised Fine-Tuning 

**Title (ZH)**: proximal 监督微调 

**Authors**: Wenhong Zhu, Ruobing Xie, Rui Wang, Xingwu Sun, Di Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17784)  

**Abstract**: Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization. 

**Abstract (ZH)**: 监督微调（SFT）往往导致泛化能力较差，其中基础能力在针对新任务或领域进行微调后退化。受强化学习（RL）中信任区域策略优化（TRPO）和近端策略优化（PPO）的启发，我们提出了近端监督微调（PSFT）。该微调目标结合了信任区域的优点，有效地在监督微调过程中限制策略漂移，同时保持竞争性的微调效果。通过将监督微调视为具有恒定正优势的策略梯度方法的特殊情形，我们推导出PSFT，该方法稳定优化过程并提高泛化能力，同时为后续的后训练阶段保留进一步优化的空间。跨数学和人类价值领域的实验表明，PSFT在领域内与监督微调表现相当，在领域外泛化效果更优，并且在长时间训练过程中保持稳定，不会导致熵崩溃，为后续优化提供了更坚实的基础。 

---
# Algebraic Approach to Ridge-Regularized Mean Squared Error Minimization in Minimal ReLU Neural Network 

**Title (ZH)**: 基于代数方法的最小ReLU神经网络中岭正则化均方误差最小化研究 

**Authors**: Ryoya Fukasaku, Yutaro Kabata, Akifumi Okuno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17783)  

**Abstract**: This paper investigates a perceptron, a simple neural network model, with ReLU activation and a ridge-regularized mean squared error (RR-MSE). Our approach leverages the fact that the RR-MSE for ReLU perceptron is piecewise polynomial, enabling a systematic analysis using tools from computational algebra. In particular, we develop a Divide-Enumerate-Merge strategy that exhaustively enumerates all local minima of the RR-MSE. By virtue of the algebraic formulation, our approach can identify not only the typical zero-dimensional minima (i.e., isolated points) obtained by numerical optimization, but also higher-dimensional minima (i.e., connected sets such as curves, surfaces, or hypersurfaces). Although computational algebraic methods are computationally very intensive for perceptrons of practical size, as a proof of concept, we apply the proposed approach in practice to minimal perceptrons with a few hidden units. 

**Abstract (ZH)**: 本文研究了带有ReLU激活和岭正则化均方误差（RR-MSE）的感知器，这是一种简单的神经网络模型。我们的方法利用了ReLU感知器的RR-MSE为分段多项式的事实，从而利用计算代数中的工具进行系统的分析。特别是，我们开发了一种划分-枚举-合并策略，可以穷尽地枚举RR-MSE的所有局部最小值。由于采用代数表示，我们的方法不仅可以识别数值优化得到的典型零维最小值（即孤立点），还可以识别高维最小值（如曲线、曲面或超曲面等连接集合）。尽管计算代数方法对于实际规模的感知器来说计算上非常密集，作为概念验证，我们将所提出的方法应用于具有少量隐藏单元的最小感知器进行实践研究。 

---
# DiffusionGS: Generative Search with Query Conditioned Diffusion in Kuaishou 

**Title (ZH)**: DiffusionGS: 基于查询条件扩散的生成搜索在快手 

**Authors**: Qinyao Li, Xiaoyang Zheng, Qihang Zhao, Ke Xu, Zhongbo Sun, Chao Wang, Chenyi Lei, Han Li, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17754)  

**Abstract**: Personalized search ranking systems are critical for driving engagement and revenue in modern e-commerce and short-video platforms. While existing methods excel at estimating users' broad interests based on the filtered historical behaviors, they typically under-exploit explicit alignment between a user's real-time intent (represented by the user query) and their past actions. In this paper, we propose DiffusionGS, a novel and scalable approach powered by generative models. Our key insight is that user queries can serve as explicit intent anchors to facilitate the extraction of users' immediate interests from long-term, noisy historical behaviors. Specifically, we formulate interest extraction as a conditional denoising task, where the user's query guides a conditional diffusion process to produce a robust, user intent-aware representation from their behavioral sequence. We propose the User-aware Denoising Layer (UDL) to incorporate user-specific profiles into the optimization of attention distribution on the user's past actions. By reframing queries as intent priors and leveraging diffusion-based denoising, our method provides a powerful mechanism for capturing dynamic user interest shifts. Extensive offline and online experiments demonstrate the superiority of DiffusionGS over state-of-the-art methods. 

**Abstract (ZH)**: 个性化搜索排名系统对于推动现代电商平台和短视频平台的用户参与和收入至关重要。虽然现有方法在基于过滤的历史行为估计用户广泛的兴趣方面表现出色，但它们通常未能充分利用用户实时意图（由用户查询表示）与过去行为之间的显式对齐。在本文中，我们提出了一种基于生成模型的novel和可扩展的方法DiffusionGS。我们的核心见解是，用户查询可以作为显式的意图锚点，帮助从长期的噪声历史行为中提取用户的即时兴趣。具体而言，我们将兴趣提取公式化为一个条件去噪任务，其中用户的查询引导一个条件扩散过程，从用户的行为序列中生成稳健的、意图意识强的表示。我们提出了用户意识去噪层（UDL）来将用户特定的特征融入到注意力分布优化中，以便在用户的过去行为上进行。通过将查询重新定义为意图先验，并利用基于扩散的去噪方法，我们的方法提供了捕获用户兴趣动态变化的强大机制。广泛的离线和在线实验表明，DiffusionGS在对比最先进的方法中具有明显优势。 

---
# EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models 

**Title (ZH)**: EEG-FM-Bench：一种全面的脑电基础模型系统评估基准 

**Authors**: Wei Xiong, Jiangtong Li, Jie Li, Kun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17742)  

**Abstract**: Electroencephalography (EEG) foundation models are poised to significantly advance brain signal analysis by learning robust representations from large-scale, unlabeled datasets. However, their rapid proliferation has outpaced the development of standardized evaluation benchmarks, which complicates direct model comparisons and hinders systematic scientific progress. This fragmentation fosters scientific inefficiency and obscures genuine architectural advancements. To address this critical gap, we introduce EEG-FM-Bench, the first comprehensive benchmark for the systematic and standardized evaluation of EEG foundation models (EEG-FMs). Our contributions are threefold: (1) we curate a diverse suite of downstream tasks and datasets from canonical EEG paradigms, implementing standardized processing and evaluation protocols within a unified open-source framework; (2) we benchmark prominent state-of-the-art foundation models to establish comprehensive baseline results for a clear comparison of the current landscape; (3) we perform qualitative analyses of the learned representations to provide insights into model behavior and inform future architectural design. Through extensive experiments, we find that fine-grained spatio-temporal feature interaction, multitask unified training and neuropsychological priors would contribute to enhancing model performance and generalization capabilities. By offering a unified platform for fair comparison and reproducible research, EEG-FM-Bench seeks to catalyze progress and guide the community toward the development of more robust and generalizable EEG-FMs. Code is released at this https URL. 

**Abstract (ZH)**: EEG基础模型评价基准EEG-FM-Bench：系统标准评估 EEG 基础模型 

---
# Robustness Feature Adapter for Efficient Adversarial Training 

**Title (ZH)**: 稳健性特征适配器用于高效对抗训练 

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17680)  

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale. 

**Abstract (ZH)**: 基于适配器的方法：在特征空间中高效对抗训练的同时解决鲁棒过拟合问题 

---
# Consistent Opponent Modeling of Static Opponents in Imperfect-Information Games 

**Title (ZH)**: 静态对手在不完美信息游戏中的一致对手建模 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2508.17671)  

**Abstract**: The goal of agents in multi-agent environments is to maximize total reward against the opposing agents that are encountered. Following a game-theoretic solution concept, such as Nash equilibrium, may obtain a strong performance in some settings; however, such approaches fail to capitalize on historical and observed data from repeated interactions against our opponents. Opponent modeling algorithms integrate machine learning techniques to exploit suboptimal opponents utilizing available data; however, the effectiveness of such approaches in imperfect-information games to date is quite limited. We show that existing opponent modeling approaches fail to satisfy a simple desirable property even against static opponents drawn from a known prior distribution; namely, they do not guarantee that the model approaches the opponent's true strategy even in the limit as the number of game iterations approaches infinity. We develop a new algorithm that is able to achieve this property and runs efficiently by solving a convex minimization problem based on the sequence-form game representation using projected gradient descent. The algorithm is guaranteed to efficiently converge to the opponent's true strategy given observations from gameplay and possibly additional historical data if it is available. 

**Abstract (ZH)**: 多代理环境中智能体的目标是最大化与对手智能体互动时的总奖励。遵循博弈论解决方案概念，如纳什均衡，在某些情况下可以获得强大的性能；然而，此类方法未能利用在多次互动中观察到的历史和数据。对手建模算法结合机器学习技术来利用可用数据exploit suboptimal对手；然而，到目前为止，此类方法在不完全信息博弈中的有效性相当有限。我们证明，现有的对手建模方法即使在从已知先验分布中抽取静态对手的情况下，也无法满足一个简单的 desirable属性，即它们不能保证模型在游戏迭代次数趋于无穷大时接近对手的真实策略。我们开发了一种新的算法，该算法能够实现这一属性并通过基于序列形式博弈表示的凸最小化问题求解，使用投影梯度下降法高效运行。该算法可以通过游戏观察和可能的附加历史数据，保证有效地收敛到对手的真实策略。 

---
# Few-Shot Pattern Detection via Template Matching and Regression 

**Title (ZH)**: 基于模板匹配和回归的少样本模式检测 

**Authors**: Eunchan Jo, Dahyun Kang, Sanghyun Kim, Yunseon Choi, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.17636)  

**Abstract**: We address the problem of few-shot pattern detection, which aims to detect all instances of a given pattern, typically represented by a few exemplars, from an input image. Although similar problems have been studied in few-shot object counting and detection (FSCD), previous methods and their benchmarks have narrowed patterns of interest to object categories and often fail to localize non-object patterns. In this work, we propose a simple yet effective detector based on template matching and regression, dubbed TMR. While previous FSCD methods typically represent target exemplars as spatially collapsed prototypes and lose structural information, we revisit classic template matching and regression. It effectively preserves and leverages the spatial layout of exemplars through a minimalistic structure with a small number of learnable convolutional or projection layers on top of a frozen backbone We also introduce a new dataset, dubbed RPINE, which covers a wider range of patterns than existing object-centric datasets. Our method outperforms the state-of-the-art methods on the three benchmarks, RPINE, FSCD-147, and FSCD-LVIS, and demonstrates strong generalization in cross-dataset evaluation. 

**Abstract (ZH)**: 我们提出了一个基于模板匹配和回归的简单而有效的检测器TMR，以解决少样本模式检测问题，该问题旨在从输入图像中检测给定模式的所有实例，这些模式通常由少数几个示例表示。虽然类似问题已经在少样本对象计数和检测（FSCD）中研究过，但之前的 方法和基准往往将兴趣模式局限于对象类别，并且难以定位非对象模式。在本文中，我们提出了一种基于模板匹配和回归的简单而有效的检测器TMR。尽管之前的FSCD方法通常将目标示例表示为空间压缩的原型，从而丢失结构信息，我们重新审视了经典的模板匹配和回归方法，能够通过一个简约结构保留和利用示例的空间布局，该结构在冻结的主干之上仅包含少量的学习卷积或投影层。我们还引入了一个新的数据集RPINE，该数据集涵盖了现有对象为中心的数据集之外的更广泛的模式。我们的方法在RPINE、FSCD-147和FSCD-LVIS三个基准上优于现有方法，并在跨数据集评估中显示出了强大的泛化能力。 

---
# RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System 

**Title (ZH)**: RubikSQL: 作为工业级NL2SQL系统的终身学习代理知识库 

**Authors**: Zui Chen, Han Li, Xinhao Zhang, Xiaoyu Chen, Chunyin Dong, Yifeng Wang, Xin Cai, Su Zhang, Ziqi Li, Chi Ding, Jinxu Li, Shuai Wang, Dousheng Zhao, Sanhai Gao, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17590)  

**Abstract**: We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research. 

**Abstract (ZH)**: 我们提出了RubikSQL，一种针对实际企业级NL2SQL中关键挑战（如隐含意图和领域特定术语）的新颖NL2SQL系统。RubikSQL将NL2SQL视为一个终身学习任务，要求同时进行知识库(KB)维护和SQL生成。RubikSQL通过数据库建模、结构化信息提取、代理规则挖掘以及增强思维链（CoT）的SQL建模技术系统地构建和精炼其知识库。然后，RubikSQL采用多智能体工作流利用这一精心构建的知识库生成准确的SQL。RubikSQL在KaggleDBQA和BIRD Mini-Dev数据集上实现了SOTA性能。最后，我们发布了RubikBench基准，这是一种专门设计用于捕获工业NL2SQL场景核心特征的新基准，为未来的研究提供了宝贵的资源。 

---
# MetaGen: A DSL, Database, and Benchmark for VLM-Assisted Metamaterial Generation 

**Title (ZH)**: MetaGen: 一种VLM辅助 metamaterial 生成的DSL、数据库和基准测试 

**Authors**: Liane Makatura, Benjamin Jones, Siyuan Bian, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2508.17568)  

**Abstract**: Metamaterials are micro-architected structures whose geometry imparts highly tunable-often counter-intuitive-bulk properties. Yet their design is difficult because of geometric complexity and a non-trivial mapping from architecture to behaviour. We address these challenges with three complementary contributions. (i) MetaDSL: a compact, semantically rich domain-specific language that captures diverse metamaterial designs in a form that is both human-readable and machine-parsable. (ii) MetaDB: a curated repository of more than 150,000 parameterized MetaDSL programs together with their derivatives-three-dimensional geometry, multi-view renderings, and simulated elastic properties. (iii) MetaBench: benchmark suites that test three core capabilities of vision-language metamaterial assistants-structure reconstruction, property-driven inverse design, and performance prediction. We establish baselines by fine-tuning state-of-the-art vision-language models and deploy an omni-model within an interactive, CAD-like interface. Case studies show that our framework provides a strong first step toward integrated design and understanding of structure-representation-property relationships. 

**Abstract (ZH)**: metamaterials是通过几何结构赋予高度可调的常反直觉的大尺度性质的微架构结构。然而，其设计因几何复杂性和从架构到行为的非平凡映射而充满挑战。我们通过三项互补贡献应对这些挑战。(i) MetaDSL：一种紧凑且语义丰富的领域特定语言，以既便于人类阅读又便于机器解析的形式捕获各种metamaterial设计。(ii) MetaDB：一个收录超过150,000个参数化MetaDSL程序及其导数（三维几何结构、多视图渲染和模拟弹性性质）的精心整理数据库。(iii) MetaBench：用于测试视觉-语言metamaterial辅助系统核心能力的基准测试套件——结构重建、性能预测和属性驱动的逆向设计。我们通过微调最先进的视觉-语言模型建立基线，并在一个交互式的、类似CAD的界面中部署了一个全能模型。案例研究表明，我们的框架为结构-表示-属性关系的集成设计和理解提供了坚实的第一步。 

---
# In-Context Algorithm Emulation in Fixed-Weight Transformers 

**Title (ZH)**: 固定权重变压器中的现场算法模拟 

**Authors**: Jerry Yao-Chieh Hu, Hude Liu, Jennifer Yuntong Zhang, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17550)  

**Abstract**: We prove that a minimal Transformer architecture with frozen weights is capable of emulating a broad class of algorithms by in-context prompting. In particular, for any algorithm implementable by a fixed-weight attention head (e.g. one-step gradient descent or linear/ridge regression), there exists a prompt that drives a two-layer softmax attention module to reproduce the algorithm's output with arbitrary precision. This guarantee extends even to a single-head attention layer (using longer prompts if necessary), achieving architectural minimality. Our key idea is to construct prompts that encode an algorithm's parameters into token representations, creating sharp dot-product gaps that force the softmax attention to follow the intended computation. This construction requires no feed-forward layers and no parameter updates. All adaptation happens through the prompt alone. These findings forge a direct link between in-context learning and algorithmic emulation, and offer a simple mechanism for large Transformers to serve as prompt-programmable libraries of algorithms. They illuminate how GPT-style foundation models may swap algorithms via prompts alone, establishing a form of algorithmic universality in modern Transformer models. 

**Abstract (ZH)**: 我们证明，具有冻结权重的最小Transformer架构可以通过上下文内提示来模拟广泛类别的算法。特别是，对于任何可通过固定权重attention头实现的算法（例如，单步梯度下降或线性/岭回归），存在一个提示，可以驱动两层softmax attention模块来以任意精度复制该算法的输出。这一保证甚至扩展到单头attention层（必要时使用更长的提示），实现架构上的最小化。我们的核心思想是构造提示，将算法的参数编码到 token 表征中，从而创建尖锐的点积差距，迫使softmax attention 跟随预期的计算。这一构造无需前馈层且无需参数更新，所有适应仅通过提示完成。这些发现直接链接了上下文学习与算法模拟，并提供了一种简单的机制，使大型Transformer充当可编程算法库。它们揭示了如何通过提示本身在GPT风格的基础模型之间交换算法，确立了现代Transformer模型中的算法普遍性形式。 

---
# An experimental approach: The graph of graphs 

**Title (ZH)**: 基于实验的方法：图的图谱 

**Authors**: Zsombor Szádoczki, Sándor Bozóki, László Sipos, Zsófia Galambosi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17520)  

**Abstract**: One of the essential issues in decision problems and preference modeling is the number of comparisons and their pattern to ask from the decision maker. We focus on the optimal patterns of pairwise comparisons and the sequence including the most (close to) optimal cases based on the results of a color selection experiment. In the test, six colors (red, green, blue, magenta, turquoise, yellow) were evaluated with pairwise comparisons as well as in a direct manner, on color-calibrated tablets in ISO standardized sensory test booths of a sensory laboratory. All the possible patterns of comparisons resulting in a connected representing graph were evaluated against the complete data based on 301 individual's pairwise comparison matrices (PCMs) using the logarithmic least squares weight calculation technique. It is shown that the empirical results, i.e., the empirical distributions of the elements of PCMs, are quite similar to the former simulated outcomes from the literature. The obtained empirically optimal patterns of comparisons were the best or the second best in the former simulations as well, while the sequence of comparisons that contains the most (close to) optimal patterns is exactly the same. In order to enhance the applicability of the results, besides the presentation of graph of graphs, and the representing graphs of the patterns that describe the proposed sequence of comparisons themselves, the recommendations are also detailed in a table format as well as in a Java application. 

**Abstract (ZH)**: 决策问题和偏好建模中一个关键问题是如何优化询问决策者配对比较的数量及其模式。基于颜色选取实验的结果，我们关注最优的配对比较模式以及包括最多（接近）最优情况的序列。在实验中，六种颜色（红色、绿色、蓝色、品红色、青色、黄色）通过配对比较和直接评价，在ISO标准化感官测试舱的色彩校准平板上进行了评估。所有生成连通表示图的所有可能的比较模式均使用基于301名个体的配对比较矩阵（PCMs）和对数最小二乘权重计算技术进行了评估。实验结果显示，实际情况中配对比较矩阵元素的经验分布与文献中的仿真实验结果非常相似。获得的经验最优比较模式在之前的仿真实验中均为最佳或次最佳，而包含最多（接近）最优模式的比较序列也完全相同。为了提高结果的应用性，除了展示图与图的关系图和描述所提比较序列的表示图外，还在表格和Java应用程序中详细提供了建议。 

---
# TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification 

**Title (ZH)**: TANDEM: 时空注意力引导的神经微分方程模型用于时间序列分类 

**Authors**: YongKyung Oh, Dong-Young Lim, Sungil Kim, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17519)  

**Abstract**: Handling missing data in time series classification remains a significant challenge in various domains. Traditional methods often rely on imputation, which may introduce bias or fail to capture the underlying temporal dynamics. In this paper, we propose TANDEM (Temporal Attention-guided Neural Differential Equations for Missingness), an attention-guided neural differential equation framework that effectively classifies time series data with missing values. Our approach integrates raw observation, interpolated control path, and continuous latent dynamics through a novel attention mechanism, allowing the model to focus on the most informative aspects of the data. We evaluate TANDEM on 30 benchmark datasets and a real-world medical dataset, demonstrating its superiority over existing state-of-the-art methods. Our framework not only improves classification accuracy but also provides insights into the handling of missing data, making it a valuable tool in practice. 

**Abstract (ZH)**: 时间序列分类中缺失数据的处理仍然是各个领域的一项重大挑战。传统方法通常依赖于插补，这可能会引入偏差或无法捕捉潜在的时间动态。在本文中，我们提出了一种名为TANDEM（Temporal Attention-guided Neural Differential Equations for Missingness）的方法，这是一种通过新颖的注意力机制整合原始观测、插补控制路径和连续潜在动态的神经微分方程框架，有效分类带有缺失值的时间序列数据。我们在30个基准数据集和一个真实世界的医疗数据集上评估了TANDEM，展示了其在分类准确性上的优越性，并提供了对缺失数据处理的见解，使其成为实践中的有力工具。 

---
# FedKLPR: Personalized Federated Learning for Person Re-Identification with Adaptive Pruning 

**Title (ZH)**: FedKLPR：带有自适应剪枝的个性化联邦学习在行人重识别中的应用 

**Authors**: Po-Hsien Yu, Yu-Syuan Tseng, Shao-Yi Chien  

**Link**: [PDF](https://arxiv.org/pdf/2508.17431)  

**Abstract**: Person re-identification (Re-ID) is a fundamental task in intelligent surveillance and public safety. Federated learning (FL) offers a privacy-preserving solution by enabling collaborative model training without centralized data collection. However, applying FL to real-world re-ID systems faces two major challenges: statistical heterogeneity across clients due to non-IID data distributions, and substantial communication overhead caused by frequent transmission of large-scale models. To address these issues, we propose FedKLPR, a lightweight and communication-efficient federated learning framework for person re-identification. FedKLPR introduces four key components. First, the KL-Divergence Regularization Loss (KLL) constrains local models by minimizing the divergence from the global feature distribution, effectively mitigating the effects of statistical heterogeneity and improving convergence stability under non-IID conditions. Secondly, KL-Divergence-Prune Weighted Aggregation (KLPWA) integrates pruning ratio and distributional similarity into the aggregation process, thereby improving the robustness of the global model while significantly reducing communication overhead. Furthermore, sparse Activation Skipping (SAS) mitigates the dilution of critical parameters during the aggregation of pruned client models by excluding zero-valued weights from the update process. Finally, Cross-Round Recovery (CRR) introduces a dynamic pruning control mechanism that halts pruning when necessary, enabling deeper compression while maintaining model accuracy. Experimental results on eight benchmark datasets demonstrate that FedKLPR achieves significant communication reduction. Compared with the state-of-the-art, FedKLPR reduces 33\%-38\% communication cost on ResNet-50 and 20\%-40\% communication cost on ResNet-34, while maintaining model accuracy within 1\% degradation. 

**Abstract (ZH)**: 联邦学习框架FedKLPR：针对人员再识别的轻量级和通信高效方法 

---
# Convergence and Generalization of Anti-Regularization for Parametric Models 

**Title (ZH)**: 参数模型中的反正则化收敛性和泛化性分析 

**Authors**: Dongseok Kim, Wonjun Jeong, Gisung Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17412)  

**Abstract**: We propose Anti-regularization (AR), which adds a sign-reversed reward term to the loss to intentionally increase model expressivity in the small-sample regime, and then attenuates this intervention with a power-law decay as the sample size grows. We formalize spectral safety and trust-region conditions, and design a lightweight stability safeguard that combines a projection operator with gradient clipping, ensuring stable intervention under stated assumptions. Our analysis spans linear smoothers and the Neural Tangent Kernel (NTK) regime, providing practical guidance on selecting the decay exponent by balancing empirical risk against variance. Empirically, AR reduces underfitting while preserving generalization and improving calibration in both regression and classification. Ablation studies confirm that the decay schedule and the stability safeguard are critical to preventing overfitting and numerical instability. We further examine a degrees-of-freedom targeting schedule that keeps per-sample complexity approximately constant. AR is simple to implement and reproducible, integrating cleanly into standard empirical risk minimization pipelines. It enables robust learning in data- and resource-constrained settings by intervening only when beneficial and fading away when unnecessary. 

**Abstract (ZH)**: 我们提出反正则化(AR)，该方法在损失中添加了一个符号反转的奖励项，以故意增加小样本情况下的模型表达性，然后随着样本数量的增长，通过幂律衰减来减弱这种干预。我们形式化了光谱安全性与信赖域条件，并设计了一种轻量级的稳定性保障措施，结合投影算子与梯度剪裁，确保在满足特定假设时模型干预的稳定性。我们的分析跨越了线性平滑器和神经 tangent 核 (NTK) 状态，提供了平衡经验风险与方差以选择衰减指数的实用指导。实验证明，AR 可以减少欠拟合现象，同时保持泛化能力和提高校准性，在回归和分类任务中均表现良好。消融实验表明，衰减计划和稳定性保障措施对于防止过拟合和数值不稳定性至关重要。我们还研究了一种自由度目标计划，以保持每样本复杂度相对恒定。AR 实现简单且可再现，能够无缝集成到标准的经验风险最小化流水线中。它能够在数据和资源受限的环境中实现稳健学习，仅在有益时进行干预，而在不需要时逐渐减弱。 

---
# Neural Proteomics Fields for Super-resolved Spatial Proteomics Prediction 

**Title (ZH)**: 神经蛋白质组学领域用于超分辨空间蛋白质组学预测 

**Authors**: Bokai Zhao, Weiyang Shi, Hanqing Chao, Zijiang Yang, Yiyang Zhang, Ming Song, Tianzi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17389)  

**Abstract**: Spatial proteomics maps protein distributions in tissues, providing transformative insights for life sciences. However, current sequencing-based technologies suffer from low spatial resolution, and substantial inter-tissue variability in protein expression further compromises the performance of existing molecular data prediction methods. In this work, we introduce the novel task of spatial super-resolution for sequencing-based spatial proteomics (seq-SP) and, to the best of our knowledge, propose the first deep learning model for this task--Neural Proteomics Fields (NPF). NPF formulates seq-SP as a protein reconstruction problem in continuous space by training a dedicated network for each tissue. The model comprises a Spatial Modeling Module, which learns tissue-specific protein spatial distributions, and a Morphology Modeling Module, which extracts tissue-specific morphological features. Furthermore, to facilitate rigorous evaluation, we establish an open-source benchmark dataset, Pseudo-Visium SP, for this task. Experimental results demonstrate that NPF achieves state-of-the-art performance with fewer learnable parameters, underscoring its potential for advancing spatial proteomics research. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 空间蛋白质组学映射组织中的蛋白质分布，为生命科学提供了变革性的洞察。然而，现有的基于测序的空间蛋白质组学技术具有较低的空间分辨率，并且组织间蛋白质表达的显著差异进一步降低了现有分子数据分析方法的性能。为解决这一问题，我们提出了空间超分辨率任务，即基于测序的空间蛋白质组学(seq-SP)，并提出了首个针对此任务的深度学习模型——神经蛋白质场(NPF)模型。NPF通过为每个组织训练一个专门的网络，将seq-SP公式化为连续空间中的蛋白质重建问题。模型包括一个空间建模模块，该模块学习组织特异性蛋白质的空间分布，以及一个形态学建模模块，该模块提取组织特异性形态特征。此外，为了方便严格评估，我们建立了开源基准数据集Pseudo-Visium SP。实验结果表明，NPF在更少的可学习参数下实现了最先进的性能，证明了其在推动空间蛋白质组学研究方面潜在的重要性。我们的代码和数据集在该网址公开：[this https URL]。 

---
# The Arabic Generality Score: Another Dimension of Modeling Arabic Dialectness 

**Title (ZH)**: 阿拉伯通用性评分：阿拉伯方言建模的另一维度 

**Authors**: Sanad Shaban, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17347)  

**Abstract**: Arabic dialects form a diverse continuum, yet NLP models often treat them as discrete categories. Recent work addresses this issue by modeling dialectness as a continuous variable, notably through the Arabic Level of Dialectness (ALDi). However, ALDi reduces complex variation to a single dimension. We propose a complementary measure: the Arabic Generality Score (AGS), which quantifies how widely a word is used across dialects. We introduce a pipeline that combines word alignment, etymology-aware edit distance, and smoothing to annotate a parallel corpus with word-level AGS. A regression model is then trained to predict AGS in context. Our approach outperforms strong baselines, including state-of-the-art dialect ID systems, on a multi-dialect benchmark. AGS offers a scalable, linguistically grounded way to model lexical generality, enriching representations of Arabic dialectness. 

**Abstract (ZH)**: 阿拉伯方言构成一个多元连续体，然而NLP模型常将其视为离散类别。最近的研究通过将方言特征建模为连续变量来解决这一问题，突出表现为阿拉伯方言连续性水平（ALDi）模型。然而，ALDi将复杂的变体简化为单一维度。我们提出一个补充指标：阿拉伯词汇普适性评分（AGS），量化一个词在不同方言中的使用范围。我们引入了一种管道，结合词汇对齐、语源学感知编辑距离和平滑技术，标注平行语料库中的词级AGS。然后训练回归模型以预测上下文中词级AGS。我们的方法在多方言基准测试中优于Strong Baselines，包括最先进的方言识别系统。AGS提供了一种可扩展、基于语言学的词汇普适性建模方式，丰富了阿拉伯方言特征的表现。 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Title (ZH)**: 使用知识图谱从事实到法律 capturing 法庭判决中的法律推理路径 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

**Abstract (ZH)**: 法院判决揭示了法律规则如何被解释和应用于事实的过程，为理解结构化的法律推理提供了基础。然而，现有的自动化法律推理捕获方法，包括大型语言模型，往往无法识别相关的法律背景，不能准确追踪事实与法律规范之间的关系，并可能误代表司法推理的层次结构。这些限制阻碍了捕捉法院在实践中如何适用法律的能力。本文通过构建来自648份日本行政法院判决的法律知识图谱，解决了这些挑战。我们的方法使用基于提示的大语言模型提取法律推理的组件，规范化对法律规定的引用，并通过法律推理本体将事实、规范和法律应用连接起来。生成的图谱捕捉了实际法院判决中法律推理的完整结构，使隐含的推理明确化并可机器读取。我们使用专家标注的数据评估了系统，并发现它在从事实中检索相关法律规定的准确性上超过了大型语言模型基线和检索增强方法。 

---
# Omne-R1: Learning to Reason with Memory for Multi-hop Question Answering 

**Title (ZH)**: Omne-R1：基于记忆进行多跳问答的推理学习 

**Authors**: Boyuan Liu, Feng Ji, Jiayan Nan, Han Zhao, Weiling Chen, Shihao Xu, Xing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17330)  

**Abstract**: This paper introduces Omne-R1, a novel approach designed to enhance multi-hop question answering capabilities on schema-free knowledge graphs by integrating advanced reasoning models. Our method employs a multi-stage training workflow, including two reinforcement learning phases and one supervised fine-tuning phase. We address the challenge of limited suitable knowledge graphs and QA data by constructing domain-independent knowledge graphs and auto-generating QA pairs. Experimental results show significant improvements in answering multi-hop questions, with notable performance gains on more complex 3+ hop questions. Our proposed training framework demonstrates strong generalization abilities across diverse knowledge domains. 

**Abstract (ZH)**: Omne-R1：一种用于无模式知识图谱多跳问答的先进推理模型集成方法 

---
# Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality 

**Title (ZH)**: 二叉树：通过优化通信局部性增强集体操作 

**Authors**: Daniele De Sensi, Saverio Pasqualoni, Lorenzo Piarulli, Tommaso Bonato, Seydou Ba, Matteo Turisini, Jens Domke, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2508.17311)  

**Abstract**: Communication locality plays a key role in the performance of collective operations on large HPC systems, especially on oversubscribed networks where groups of nodes are fully connected internally but sparsely linked through global connections. We present Bine (binomial negabinary) trees, a family of collective algorithms that improve communication locality. Bine trees maintain the generality of binomial trees and butterflies while cutting global-link traffic by up to 33%. We implement eight Bine-based collectives and evaluate them on four large-scale supercomputers with Dragonfly, Dragonfly+, oversubscribed fat-tree, and torus topologies, achieving up to 5x speedups and consistent reductions in global-link traffic across different vector sizes and node counts. 

**Abstract (ZH)**: 通信局部性在大型HPC系统中集体操作的性能中发挥着关键作用，尤其是在超订阅网络中，内部节点完全连接但通过全局连接稀疏连接。我们提出了Bine（二项负二进制）树，这是一种提高通信局部性的集体算法家族，保持了二项树和蝶形的通用性，并将全局连接流量最多降低了33%。我们在具有Dragonfly、Dragonfly+、超订阅网状网络和环状拓扑的四种大规模超级计算机上实现了八种Bine基集体算法，并实现了最高5倍的加速性能，并在不同向量尺寸和节点数下一致地减少了全局连接流量。 

---
# Deep Learning-Assisted Detection of Sarcopenia in Cross-Sectional Computed Tomography Imaging 

**Title (ZH)**: 深度学习辅助横断面计算机断层扫描影像中的肌少症检测 

**Authors**: Manish Bhardwaj, Huizhi Liang, Ashwin Sivaharan, Sandip Nandhra, Vaclav Snasel, Tamer El-Sayed, Varun Ojha  

**Link**: [PDF](https://arxiv.org/pdf/2508.17275)  

**Abstract**: Sarcopenia is a progressive loss of muscle mass and function linked to poor surgical outcomes such as prolonged hospital stays, impaired mobility, and increased mortality. Although it can be assessed through cross-sectional imaging by measuring skeletal muscle area (SMA), the process is time-consuming and adds to clinical workloads, limiting timely detection and management; however, this process could become more efficient and scalable with the assistance of artificial intelligence applications. This paper presents high-quality three-dimensional cross-sectional computed tomography (CT) images of patients with sarcopenia collected at the Freeman Hospital, Newcastle upon Tyne Hospitals NHS Foundation Trust. Expert clinicians manually annotated the SMA at the third lumbar vertebra, generating precise segmentation masks. We develop deep-learning models to measure SMA in CT images and automate this task. Our methodology employed transfer learning and self-supervised learning approaches using labelled and unlabeled CT scan datasets. While we developed qualitative assessment models for detecting sarcopenia, we observed that the quantitative assessment of SMA is more precise and informative. This approach also mitigates the issue of class imbalance and limited data availability. Our model predicted the SMA, on average, with an error of +-3 percentage points against the manually measured SMA. The average dice similarity coefficient of the predicted masks was 93%. Our results, therefore, show a pathway to full automation of sarcopenia assessment and detection. 

**Abstract (ZH)**: 肌少症是与肌肉质量和功能逐渐减退相关的疾病，与术后长期住院、移动障碍和死亡率增加等不良手术结果有关。虽然可以通过测量骨骼肌面积（SMA）来横向成像评估肌少症，但这一过程耗时且增加了临床工作负担，限制了及时检测和管理；然而，在人工智能应用的辅助下，这一过程可以变得更加高效和可扩展。本文展示了在纽卡斯尔皇家弗里曼医院 NHS 基金会信托的肌少症患者中收集的高质量三维横截面计算机断层扫描（CT）图像。专家临床医师手动在第三腰椎处标注了骨骼肌面积（SMA），生成了精确的分割掩模。我们开发了深度学习模型来测量CT图像中的SMA并自动化这一任务。我们的方法使用了迁移学习和半监督学习方法，利用有标签和无标签的CT扫描数据集。虽然我们为检测肌少症开发了定性评估模型，但我们发现对骨骼肌面积（SMA）的定量评估更为精确和具有信息量。此外，该方法还缓解了类别不平衡和数据可用性有限的问题。我们的模型在平均误差为±3个百分点的情况下预测了SMA，预测掩模的平均骰子相似系数为93%。因此，我们的结果显示了一条实现肌少症评估和检测完全自动化的途径。 

---
# Provable Generalization in Overparameterized Neural Nets 

**Title (ZH)**: 过参数化神经网络的可证明泛化能力 

**Authors**: Aviral Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17256)  

**Abstract**: Deep neural networks often contain far more parameters than training examples, yet they still manage to generalize well in practice. Classical complexity measures such as VC-dimension or PAC-Bayes bounds usually become vacuous in this overparameterized regime, offering little explanation for the empirical success of models like Transformers. In this work, I explore an alternative notion of capacity for attention-based models, based on the effective rank of their attention matrices. The intuition is that, although the parameter count is enormous, the functional dimensionality of attention is often much lower. I show that this quantity leads to a generalization bound whose dependence on sample size matches empirical scaling laws observed in large language models, up to logarithmic factors. While the analysis is not a complete theory of overparameterized learning, it provides evidence that spectral properties of attention, rather than raw parameter counts, may be the right lens for understanding why these models generalize. 

**Abstract (ZH)**: 注意力模型的过参数化泛化能力：基于注意力矩阵有效秩的容量度量 

---
# CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models 

**Title (ZH)**: CoViPAL：分层上下文化视觉词元剪枝大型视觉语言模型 

**Authors**: Zicong Tang, Ziyang Ma, Suqing Wang, Zuchao Li, Lefei Zhang, Hai Zhao, Yun Li, Qianren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17243)  

**Abstract**: Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）处理由文本标记和从图像或视频提取的视觉标记组成的多模态输入。由于丰富的视觉信息，单张图像可以生成数千个视觉标记，导致预填充阶段计算成本高，并且在解码过程中产生显著的内存开销。现有方法尝试剪枝冗余的视觉标记，揭示了视觉表示中的大量冗余。然而，这些方法在浅层网络中常常难以处理，因为缺乏足够的上下文信息。我们认为，即使在浅层网络中，许多视觉标记本质上也是冗余的，并且可以通过适当的上下文信号安全且有效地剪枝。在本文中，我们提出了一种逐层上下文化视觉标记剪枝方法CoViPAL，该方法采用可插即用剪枝模块（PPM）在这些标记被LVLM处理之前预测并移除冗余的视觉标记。PPM轻量级、模型无关，并独立于LVLM架构，确保可以无缝集成到各种模型中。在多个基准上的广泛实验表明，CoViPAL在相等的标记预算下优于无训练剪枝方法，并且在具有类似监督的情况下超过了基于训练的方法。CoViPAL提供了一种可扩展且高效的解决方案，可以在不牺牲准确性的前提下提高LVLM的推理效率。 

---
# ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation 

**Title (ZH)**: ClaimGen-CN：大规模中文法律索赔生成数据集 

**Authors**: Siying Zhou, Yiquan Wu, Hui Chen, Xavier Hu, Kun Kuang, Adam Jatowt, Ming Hu, Chunyan Zheng, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17234)  

**Abstract**: Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available. 

**Abstract (ZH)**: 基于给定案件事实的法律索赔生成问题研究：从现实法律纠纷构建ClaimGen-CN数据集及评估方法 

---
# Multi-Metric Preference Alignment for Generative Speech Restoration 

**Title (ZH)**: 多指标偏好对齐的生成语音恢复 

**Authors**: Junan Zhang, Xueyao Zhang, Jing Yang, Yuancheng Wang, Fan Fan, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17229)  

**Abstract**: Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:this https URL 

**Abstract (ZH)**: Recent生成模型在语音恢复任务中取得了显著进展，但其训练目标往往与人类的感知偏好不一致，导致质量不佳。虽然在文本和图像生成等其他生成领域中，后训练对齐已被证明是有效的，但在生成语音恢复中的应用仍然 largely 未被探索。本研究探讨了将基于偏好的后训练应用于该任务所面临的挑战，重点关注如何定义稳健的偏好信号并收集高质量数据以避免奖励欺骗。为了解决这些挑战，我们提出了一种多指标偏好对齐策略。我们构建了一个新的数据集GenSR-Pref，包含80,000个偏好对，其中每个选择的样本都得到了互补的多种指标的一致青睐，这些指标涵盖了感知质量、信号保真度、内容一致性和音色保持。这种方法确保了全面的偏好信号。通过我们的数据集应用直接偏好优化（DPO），在三种不同的生成范式：自回归模型（AR）、遮蔽生成模型（MGM）和流动匹配模型（FM）上的各种恢复基准上，我们在客观和主观评估中观察到一致且显著的性能提升。消融研究证实，与单一指标方法相比，我们的多指标策略在减轻奖励欺骗方面更具优越性。此外，我们展示了我们的对齐模型可以作为强大的“数据注释器”，生成高质量的伪标签，作为在数据稀缺场景下（如歌唱声音恢复）传统区别性模型的监督信号。 

---
# GPG-HT: Generalized Policy Gradient with History-Aware Decision Transformer for Probabilistic Path Planning 

**Title (ZH)**: GPG-HT：具有历史意识决策变换器的广义策略梯度方法用于概率路径规划 

**Authors**: Xing Wei, Yuqi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17218)  

**Abstract**: With the rapidly increased number of vehicles in urban areas, existing road infrastructure struggles to accommodate modern traffic demands, resulting in the issue of congestion. This highlights the importance of efficient path planning strategies. However, most recent navigation models focus solely on deterministic or time-dependent networks, while overlooking the correlations and the stochastic nature of traffic flows. In this work, we address the reliable shortest path problem within stochastic transportation networks under certain dependencies. We propose a path planning solution that integrates the decision Transformer with the Generalized Policy Gradient (GPG) framework. Based on the decision Transformer's capability to model long-term dependencies, our proposed solution improves the accuracy and stability of path decisions. Experimental results on the Sioux Falls Network (SFN) demonstrate that our approach outperforms previous baselines in terms of on-time arrival probability, providing more accurate path planning solutions. 

**Abstract (ZH)**: 在具有特定依赖性的随机运输网络中可靠的最短路径问题及其路径规划解决方案 

---
# Scaling Graph Transformers: A Comparative Study of Sparse and Dense Attention 

**Title (ZH)**: 扩展图变换器：稀疏与密集注意力的比较研究 

**Authors**: Leon Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17175)  

**Abstract**: Graphs have become a central representation in machine learning for capturing relational and structured data across various domains. Traditional graph neural networks often struggle to capture long-range dependencies between nodes due to their local structure. Graph transformers overcome this by using attention mechanisms that allow nodes to exchange information globally. However, there are two types of attention in graph transformers: dense and sparse. In this paper, we compare these two attention mechanisms, analyze their trade-offs, and highlight when to use each. We also outline current challenges and problems in designing attention for graph transformers. 

**Abstract (ZH)**: 图已成为机器学习中用于捕捉各种领域中关系和结构化数据的核心表示。传统图神经网络往往难以捕捉由于其局部结构导致的长距离节点依赖关系。图变压器通过使用允许节点进行全局信息交换的注意力机制来克服这一问题。然而，图变压器中有两种类型的注意力机制：密集型和稀疏型。在本文中，我们比较了这两种注意力机制，分析了它们的权衡，并指出了在每种情况下使用它们的情形。我们还概述了设计图变压器注意力机制当前面临的挑战和问题。 

---
# ONG: Orthogonal Natural Gradient Descent 

**Title (ZH)**: ONG：正交自然梯度下降 

**Authors**: Yajat Yadav, Jathin Korrapati, Patrick Mendoza  

**Link**: [PDF](https://arxiv.org/pdf/2508.17169)  

**Abstract**: Orthogonal gradient descent has emerged as a powerful method for continual learning tasks. However, its Euclidean projections overlook the underlying information-geometric structure of the space of distributions parametrized by neural networks, which can lead to suboptimal convergence in learning tasks. To counteract this, we combine it with the idea of the natural gradient and present ONG (Orthogonal Natural Gradient Descent). ONG preconditions each new task gradient with an efficient EKFAC approximation of the inverse Fisher information matrix, yielding updates that follow the steepest descent direction under a Riemannian metric. To preserve performance on previously learned tasks, ONG projects these natural gradients onto the orthogonal complement of prior task gradients. We provide a theoretical justification for this procedure, introduce the ONG algorithm, and benchmark its performance on the Permuted and Rotated MNIST datasets. All code for our experiments/reproducibility can be found at this https URL. 

**Abstract (ZH)**: 正交自然梯度下降已成为连续学习任务中一种强大的方法。然而，其欧几里得投影忽略了由神经网络参数化的分布空间下的信息几何结构，可能导致学习任务中次优的收敛效果。为解决这一问题，我们将自然梯度的思想与正交梯度下降相结合，提出了一种称为ONG（正交自然梯度下降）的方法。ONG利用EKFAC近似逆费舍尔信息矩阵对每个新任务梯度进行预条件化，生成在黎曼度量下沿最陡下降方向的更新。为了在保持先前学习任务性能的同时适应新任务，ONG将这些自然梯度投影到先前任务梯度的正交补空间。我们为这种操作提供了理论依据，并介绍了ONG算法，同时在Permuted和Rotated MNIST数据集上展示了其性能。所有实验代码及可重复性代码可在以下链接找到：this https URL。 

---
# Error analysis for the deep Kolmogorov method 

**Title (ZH)**: 深层柯尔莫戈罗夫方法的误差分析 

**Authors**: Iulian Cîmpean, Thang Do, Lukas Gonon, Arnulf Jentzen, Ionel Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17167)  

**Abstract**: The deep Kolmogorov method is a simple and popular deep learning based method for approximating solutions of partial differential equations (PDEs) of the Kolmogorov type. In this work we provide an error analysis for the deep Kolmogorov method for heat PDEs. Specifically, we reveal convergence with convergence rates for the overall mean square distance between the exact solution of the heat PDE and the realization function of the approximating deep neural network (DNN) associated with a stochastic optimization algorithm in terms of the size of the architecture (the depth/number of hidden layers and the width of the hidden layers) of the approximating DNN, in terms of the number of random sample points used in the loss function (the number of input-output data pairs used in the loss function), and in terms of the size of the optimization error made by the employed stochastic optimization method. 

**Abstract (ZH)**: 基于深度学习的深层柯尔莫戈罗夫方法是一种用于近似柯尔莫戈罗夫类型偏微分方程（PDEs）解的简单而流行的深度学习方法。本文提供了深层柯尔莫戈罗夫方法在热方程中的误差分析，具体而言，我们揭示了近似解与热方程精确解之间的总体均方距离的收敛性及其收敛速率，以及这些距离与逼近深度神经网络（DNN）结构大小（深度和隐藏层数量及宽度）、损失函数中使用的随机样本点的数量（损失函数中输入-输出数据对的数量）以及所使用随机优化方法的优化误差大小之间的关系。 

---
# SACA: Selective Attention-Based Clustering Algorithm 

**Title (ZH)**: 基于选择性注意力的聚类算法 

**Authors**: Meysam Shirdel Bilehsavar, Razieh Ghaedi, Samira Seyed Taheri, Xinqi Fan, Christian O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2508.17150)  

**Abstract**: Clustering algorithms are widely used in various applications, with density-based methods such as Density-Based Spatial Clustering of Applications with Noise (DBSCAN) being particularly prominent. These algorithms identify clusters in high-density regions while treating sparser areas as noise. However, reliance on user-defined parameters often poses optimization challenges that require domain expertise. This paper presents a novel density-based clustering method inspired by the concept of selective attention, which minimizes the need for user-defined parameters under standard conditions. Initially, the algorithm operates without requiring user-defined parameters. If parameter adjustment is needed, the method simplifies the process by introducing a single integer parameter that is straightforward to tune. The approach computes a threshold to filter out the most sparsely distributed points and outliers, forms a preliminary cluster structure, and then reintegrates the excluded points to finalize the results. Experimental evaluations on diverse data sets highlight the accessibility and robust performance of the method, providing an effective alternative for density-based clustering tasks. 

**Abstract (ZH)**: 基于选择性注意力的新型密度基于聚类方法：在标准条件下减少用户定义参数的需求 

---
# Two Birds with One Stone: Enhancing Uncertainty Quantification and Interpretability with Graph Functional Neural Process 

**Title (ZH)**: 一石二鸟：通过图功能神经过程提高不确定性量化和可解释性 

**Authors**: Lingkai Kong, Haotian Sun, Yuchen Zhuang, Haorui Wang, Wenhao Mu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17097)  

**Abstract**: Graph neural networks (GNNs) are powerful tools on graph data. However, their predictions are mis-calibrated and lack interpretability, limiting their adoption in critical applications. To address this issue, we propose a new uncertainty-aware and interpretable graph classification model that combines graph functional neural process and graph generative model. The core of our method is to assume a set of latent rationales which can be mapped to a probabilistic embedding space; the predictive distribution of the classifier is conditioned on such rationale embeddings by learning a stochastic correlation matrix. The graph generator serves to decode the graph structure of the rationales from the embedding space for model interpretability. For efficient model training, we adopt an alternating optimization procedure which mimics the well known Expectation-Maximization (EM) algorithm. The proposed method is general and can be applied to any existing GNN architecture. Extensive experiments on five graph classification datasets demonstrate that our framework outperforms state-of-the-art methods in both uncertainty quantification and GNN interpretability. We also conduct case studies to show that the decoded rationale structure can provide meaningful explanations. 

**Abstract (ZH)**: 图神经网络（GNNs）是图数据的强大工具。然而，它们的预测结果缺乏校准且缺乏可解释性，限制了其在关键应用中的采用。为了解决这一问题，我们提出了一种新的集不确定性和可解释性于一体的图分类模型，该模型结合了图函数神经过程和图生成模型。我们的方法的核心是假设一组潜在的理据，这些理据可以映射到概率嵌入空间；分类器的预测分布通过学习一个随机相关矩阵，以这些理据嵌入为条件。图生成器用于从嵌入空间解码理据的图结构以提高模型的可解释性。为了高效训练模型，我们采用了交替优化程序，该程序模仿了著名的期望最大化（EM）算法。所提出的方法是通用的，可以应用于任何现有的GNN架构。在五个图分类数据集上的大量实验表明，我们的框架在不确定性量化和GNN可解释性方面均优于现有最先进的方法。我们还进行了案例研究，以展示解码后的理据结构可以提供有意义的解释。 

---
# Enhancing Knowledge Tracing through Leakage-Free and Recency-Aware Embeddings 

**Title (ZH)**: 通过泄漏免费和近期意识嵌入增强知识追踪 

**Authors**: Yahya Badran, Christine Preisach  

**Link**: [PDF](https://arxiv.org/pdf/2508.17092)  

**Abstract**: Knowledge Tracing (KT) aims to predict a student's future performance based on their sequence of interactions with learning content. Many KT models rely on knowledge concepts (KCs), which represent the skills required for each item. However, some of these models are vulnerable to label leakage, in which input data inadvertently reveal the correct answer, particularly in datasets with multiple KCs per question.
We propose a straightforward yet effective solution to prevent label leakage by masking ground-truth labels during input embedding construction in cases susceptible to leakage. To accomplish this, we introduce a dedicated MASK label, inspired by masked language modeling (e.g., BERT), to replace ground-truth labels. In addition, we introduce Recency Encoding, which encodes the step-wise distance between the current item and its most recent previous occurrence. This distance is important for modeling learning dynamics such as forgetting, which is a fundamental aspect of human learning, yet it is often overlooked in existing models. Recency Encoding demonstrates improved performance over traditional positional encodings on multiple KT benchmarks.
We show that incorporating our embeddings into KT models like DKT, DKT+, AKT, and SAKT consistently improves prediction accuracy across multiple benchmarks. The approach is both efficient and widely applicable. 

**Abstract (ZH)**: 知识追踪（KT）旨在基于学生与学习内容的交互序列预测其未来表现。许多KT模型依赖于知识概念（KCs），用以表示每个项目所需的能力。然而，这些模型中的一些容易受到标签泄露的影响，在含有多个KC的问题数据集中尤为明显，输入数据会无意中透露正确答案。

我们提出了一种简单有效的方法，通过在可能发生泄露的情况下，在输入嵌入构建过程中屏蔽真实标签来防止标签泄露。为此，我们引入了一个专用的MASK标签，受蒙面语言模型（如BERT）的启发，用于替代真实标签。此外，我们引入了最近性编码，这是一种将当前项目与其最近前一个出现之间的逐步距离进行编码的方法。这个距离对于建模遗忘等学习动态非常重要，而遗忘是人类学习的基本方面，但在现有模型中常被忽略。最近性编码在多个知识追踪基准测试中表现出优于传统位置编码的效果。

我们将我们的嵌入整合到DKT、DKT+、AKT和SAKT等模型中，在多个基准测试上一致提高了预测准确性。该方法既高效又具有广泛适用性。 

---
# Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation 

**Title (ZH)**: 零样本跨模态文档检索_via_跨模态问题生成 

**Authors**: Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17079)  

**Abstract**: Rapid advances in Multimodal Large Language Models (MLLMs) have expanded information retrieval beyond purely textual inputs, enabling retrieval from complex real world documents that combine text and visuals. However, most documents are private either owned by individuals or confined within corporate silos and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross modal pre questions (preQs) before retrieval. Unlike earlier multimodal retrievers that compare embeddings in a single vector space, PREMIR leverages preQs from multiple complementary modalities to expand the scope of matching to the token level. Experiments show that PREMIR achieves state of the art performance on out of distribution benchmarks, including closed domain and multilingual settings, outperforming strong baselines across all retrieval metrics. We confirm the contribution of each component through in depth ablation studies, and qualitative analyses of the generated preQs further highlight the model's robustness in real world settings. 

**Abstract (ZH)**: 快速发展的多模态大型语言模型（MLLMs）已将信息检索扩大到超出纯文本输入的范围，使从结合文本和视觉的信息复杂现实文档中检索成为可能。然而，大多数文档是私有的，要么属于个人所有，要么被限制在企业孤岛内，当前的检索器在面对未见领域或语言时遇到困难。为此，我们提出了PREMIR，这是一个简单而有效的框架，利用MLLM的广泛知识在检索前生成跨模态预问题（preQs）。与早期的多模态检索器在单一矢量空间中比较嵌入不同，PREMIR 利用来自多种互补模态的预问题来扩大匹配范围至标记级别。实验证明，PREMIR 在分布外基准测试中，包括封闭领域和多语言设置中，各项检索指标均优于强大基线，表现出色。通过深入的消融研究和生成预问题的定性分析，我们确认了每个组件的贡献，并进一步突显了模型在现实世界设置中的稳健性。 

---
# Optimizing Neural Networks with Learnable Non-Linear Activation Functions via Lookup-Based FPGA Acceleration 

**Title (ZH)**: 基于查找表的FPGA加速实现可学习非线性激活函数优化神经网络 

**Authors**: Mengyuan Yin, Benjamin Chen Ming Choong, Chuping Qu, Rick Siow Mong Goh, Weng-Fai Wong, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17069)  

**Abstract**: Learned activation functions in models like Kolmogorov-Arnold Networks (KANs) outperform fixed-activation architectures in terms of accuracy and interpretability; however, their computational complexity poses critical challenges for energy-constrained edge AI deployments. Conventional CPUs/GPUs incur prohibitive latency and power costs when evaluating higher order activations, limiting deployability under ultra-tight energy budgets. We address this via a reconfigurable lookup architecture with edge FPGAs. By coupling fine-grained quantization with adaptive lookup tables, our design minimizes energy-intensive arithmetic operations while preserving activation fidelity. FPGA reconfigurability enables dynamic hardware specialization for learned functions, a key advantage for edge systems that require post-deployment adaptability. Evaluations using KANs - where unique activation functions play a critical role - demonstrate that our FPGA-based design achieves superior computational speed and over $10^4$ times higher energy efficiency compared to edge CPUs and GPUs, while maintaining matching accuracy and minimal footprint overhead. This breakthrough positions our approach as a practical enabler for energy-critical edge AI, where computational intensity and power constraints traditionally preclude the use of adaptive activation networks. 

**Abstract (ZH)**: 基于Kolmogorov-Arnold网络的可学习激活函数在准确性和可解释性上优于固定激活函数的模型，但在计算复杂性上对能源受限的边缘AI部署构成了关键挑战。传统的CPU/GPU在评估高阶激活时会导致无法接受的延迟和功耗，限制了在超紧凑能源预算下的可部署性。我们通过在边缘FPGA上实现可重构查找表架构来解决这一问题。结合精细量化和自适应查找表，我们的设计在减少能耗密集型算术运算的同时保留了激活函数的精度。FPGA的可重构性使我们能够动态 specialize 学习到的函数硬件，这是对需要部署后适应性的边缘系统的关键优势。使用KANs的评估表明，与边缘CPU和GPU相比，我们的基于FPGA的设计实现了更优的计算速度和超过$10^4$倍的能耗效率，并且保持了匹配的准确性和最小的占位面积开销。这一突破使我们的方法成为一种实用方案，能够在计算强度和功率约束传统上排除可适应激活网络的能源关键边缘AI中发挥作用。 

---
# TabResFlow: A Normalizing Spline Flow Model for Probabilistic Univariate Tabular Regression 

**Title (ZH)**: TabResFlow：一种用于概率单变量表格回归的正则化 spline 流模型 

**Authors**: Kiran Madhusudhanan, Vijaya Krishna Yalavarthi, Jonas Sonntag, Maximilian Stubbemann, Lars Schmidt-Thieme  

**Link**: [PDF](https://arxiv.org/pdf/2508.17056)  

**Abstract**: Tabular regression is a well-studied problem with numerous industrial applications, yet most existing approaches focus on point estimation, often leading to overconfident predictions. This issue is particularly critical in industrial automation, where trustworthy decision-making is essential. Probabilistic regression models address this challenge by modeling prediction uncertainty. However, many conventional methods assume a fixed-shape distribution (typically Gaussian), and resort to estimating distribution parameters. This assumption is often restrictive, as real-world target distributions can be highly complex. To overcome this limitation, we introduce TabResFlow, a Normalizing Spline Flow model designed specifically for univariate tabular regression, where commonly used simple flow networks like RealNVP and Masked Autoregressive Flow (MAF) are unsuitable. TabResFlow consists of three key components: (1) An MLP encoder for each numerical feature. (2) A fully connected ResNet backbone for expressive feature extraction. (3) A conditional spline-based normalizing flow for flexible and tractable density estimation. We evaluate TabResFlow on nine public benchmark datasets, demonstrating that it consistently surpasses existing probabilistic regression models on likelihood scores. Our results demonstrate 9.64% improvement compared to the strongest probabilistic regression model (TreeFlow), and on average 5.6 times speed-up in inference time compared to the strongest deep learning alternative (NodeFlow). Additionally, we validate the practical applicability of TabResFlow in a real-world used car price prediction task under selective regression. To measure performance in this setting, we introduce a novel Area Under Risk Coverage (AURC) metric and show that TabResFlow achieves superior results across this metric. 

**Abstract (ZH)**: 表格回归是一个研究充分且在工业中有广泛应用的问题，但大多数现有方法专注于点估计，常常导致过于自信的预测。在工业自动化中，这种问题尤为关键，因为可靠的决策至关重要。概率回归模型通过建模预测不确定性来应对这一挑战。然而，许多传统方法假设固定形状分布（通常是高斯分布），并依赖于估计分布参数。这一假设往往是限制性的，因为现实世界的目标分布可以非常复杂。为克服这一限制，我们引入了TabResFlow，这是一种专门用于单变量表格回归的规范化插值流模型，常用的简单流网络如RealNVP和掩码自回归流（MAF）在此情况下并不适用。TabResFlow包括三个关键组件：（1）每个数值特征的MLP编码器。（2）完全连接的ResNet骨干网，用于表达性特征提取。（3）条件插值基规范化流，用于灵活且可处理的概率密度估计。我们在九个公开基准数据集上评估了TabResFlow，结果显示它在似然度得分上始终优于现有概率回归模型。我们的结果显示，与最强的概率回归模型（TreeFlow）相比，TabResFlow在该指标上提高了9.64%，并且与最强的深度学习替代品（NodeFlow）相比，平均推理时间快了5.6倍。此外，我们在选定回归的实际二手车价格预测任务中验证了TabResFlow的实用适用性。为了衡量在这种环境下的性能，我们引入了一个新的风险覆盖区域下的面积（Area Under Risk Coverage, AURC）指标，并展示了TabResFlow在该指标上取得了更好的结果。 

---
# GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation 

**Title (ZH)**: GRADE: 生成多跳问答和细粒度难度矩阵以评估语境关联检索系统 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely adopted in knowledge-intensive NLP tasks, but current evaluations often overlook the structural complexity and multi-step reasoning required in real-world scenarios. These benchmarks overlook key factors such as the interaction between retrieval difficulty and reasoning depth. To address this gap, we propose \textsc{GRADE}, a novel evaluation framework that models task difficulty along two orthogonal dimensions: (1) reasoning depth, defined by the number of inference steps (hops), and (2) semantic distance between the query and its supporting evidence. We construct a synthetic multi-hop QA dataset from factual news articles by extracting knowledge graphs and augmenting them through semantic clustering to recover missing links, allowing us to generate diverse and difficulty-controlled queries. Central to our framework is a 2D difficulty matrix that combines generator-side and retriever-side difficulty. Experiments across multiple domains and models show that error rates strongly correlate with our difficulty measures, validating their diagnostic utility. \textsc{GRADE} enables fine-grained analysis of RAG performance and provides a scalable foundation for evaluating and improving multi-hop reasoning in real-world applications. 

**Abstract (ZH)**: GRADE：一种新颖的多跳推理评估框架 

---
# Score Matching on Large Geometric Graphs for Cosmology Generation 

**Title (ZH)**: Large几何图上得分匹配生成 cosmology 

**Authors**: Diana-Alexandra Onutu, Yue Zhao, Joaquin Vanschoren, Vlado Menkovski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16990)  

**Abstract**: Generative models are a promising tool to produce cosmological simulations but face significant challenges in scalability, physical consistency, and adherence to domain symmetries, limiting their utility as alternatives to $N$-body simulations. To address these limitations, we introduce a score-based generative model with an equivariant graph neural network that simulates gravitational clustering of galaxies across cosmologies starting from an informed prior, respects periodic boundaries, and scales to full galaxy counts in simulations. A novel topology-aware noise schedule, crucial for large geometric graphs, is introduced. The proposed equivariant score-based model successfully generates full-scale cosmological point clouds of up to 600,000 halos, respects periodicity and a uniform prior, and outperforms existing diffusion models in capturing clustering statistics while offering significant computational advantages. This work advances cosmology by introducing a generative model designed to closely resemble the underlying gravitational clustering of structure formation, moving closer to physically realistic and efficient simulators for the evolution of large-scale structures in the universe. 

**Abstract (ZH)**: 基于图神经网络的等变评分生成模型在宇宙学模拟中的应用 

---
# THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics 

**Title (ZH)**: 主题：通过语义股票表示和时间动态提升主题投资 

**Authors**: Hoyoung Lee, Wonbin Ahn, Suhwan Park, Jaehoon Lee, Minjae Kim, Sungdong Yoo, Taeyoon Lim, Woohyung Lim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.16936)  

**Abstract**: Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes. 

**Abstract (ZH)**: 主题投资旨在构建与结构性趋势相一致的投资组合，但由于行业边界重叠和市场动态变化，选择相关的股票仍然具有挑战性。为应对这一挑战，我们构建了主题表示集（TRS），该扩展数据集以现实世界的主题ETF为起点，并通过纳入行业分类和财务新闻来克服其覆盖面的限制。最终数据集既包括主题与组成股票的显式映射，也包括每种主题的丰富文本概况。在此数据集的基础上，我们引入了THEME分层对比学习框架。通过将主题和股票的文本概况表示为嵌入，THEME首先利用它们的分层关系实现语义对齐。随后，通过结合单一股票回报的时序精细校正阶段来细化这些语义嵌入。最终的股票表示旨在有效检索具有强烈回报潜力的主题对齐资产。实证结果表明，THEME在多个检索指标上优于强 baseline，并显著提高了投资组合构建性能。通过联合建模来自文本的主题关系和来自回报的市场动态，THEME提供了一种可扩展且适应性强的解决方案，以应对复杂的投资主题。 

---
# Degree of Staleness-Aware Data Updating in Federated Learning 

**Title (ZH)**: staleness感知的数据更新在联邦学习中的程度aware机制 

**Authors**: Tao Liu, Xuehe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16931)  

**Abstract**: Handling data staleness remains a significant challenge in federated learning with highly time-sensitive tasks, where data is generated continuously and data staleness largely affects model performance. Although recent works attempt to optimize data staleness by determining local data update frequency or client selection strategy, none of them explore taking both data staleness and data volume into consideration. In this paper, we propose DUFL(Data Updating in Federated Learning), an incentive mechanism featuring an innovative local data update scheme manipulated by three knobs: the server's payment, outdated data conservation rate, and clients' fresh data collection volume, to coordinate staleness and volume of local data for best utilities. To this end, we introduce a novel metric called DoS(the Degree of Staleness) to quantify data staleness and conduct a theoretic analysis illustrating the quantitative relationship between DoS and model performance. We model DUFL as a two-stage Stackelberg game with dynamic constraint, deriving the optimal local data update strategy for each client in closed-form and the approximately optimal strategy for the server. Experimental results on real-world datasets demonstrate the significant performance of our approach. 

**Abstract (ZH)**: 处理数据过时仍然是联邦学习中具有时间敏感任务时的一个重大挑战，其中数据连续生成且数据过时大大影响模型性能。尽管近期工作尝试通过确定局部数据更新频率或客户端选择策略来优化数据过时，但它们都没有同时考虑数据过时和数据量。在本文中，我们提出了DUFL（联邦学习中的数据更新机制），这是一种新颖的基于三个调节器（服务器支付、过时数据保留率和客户端新鲜数据收集量）控制的局部数据更新方案，以协调局部数据的过时和数量，实现最优效用。为此，我们引入了一个新的度量标准DoS（数据过时程度）来量化数据过时，并进行理论分析以说明DoS与模型性能之间的量化关系。我们将DUFL建模为具有动态约束的两阶段Stackelberg博弈，推导出了每个客户端的闭式最佳局部数据更新策略和服务器的近似最优策略。实验结果表明，我们的方法具有显著的性能优势。 

---
# Tri-Accel: Curvature-Aware Precision-Adaptive and Memory-Elastic Optimization for Efficient GPU Usage 

**Title (ZH)**: Tri-Accel: 曲率意识的精度自适应和内存弹性优化以提高GPU使用效率 

**Authors**: Mohsen Sheibanian, Pouya Shaeri, Alimohammad Beigi, Ryan T. Woo, Aryan Keluskar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16905)  

**Abstract**: Deep neural networks are increasingly bottlenecked by the cost of optimization, both in terms of GPU memory and compute time. Existing acceleration techniques, such as mixed precision, second-order methods, and batch size scaling, are typically used in isolation. We present Tri-Accel, a unified optimization framework that co-adapts three acceleration strategies along with adaptive parameters during training: (1) Precision-Adaptive Updates that dynamically assign mixed-precision levels to layers based on curvature and gradient variance; (2) Sparse Second-Order Signals that exploit Hessian/Fisher sparsity patterns to guide precision and step size decisions; and (3) Memory-Elastic Batch Scaling that adjusts batch size in real time according to VRAM availability. On CIFAR-10 with ResNet-18 and EfficientNet-B0, Tri-Accel achieves up to 9.9% reduction in training time and 13.3% lower memory usage, while improving accuracy by +1.1 percentage points over FP32 baselines. Tested on CIFAR-10/100, our approach demonstrates adaptive learning behavior, with efficiency gradually improving over the course of training as the system learns to allocate resources more effectively. Compared to static mixed-precision training, Tri-Accel maintains 78.1% accuracy while reducing memory footprint from 0.35GB to 0.31GB on standard hardware. The framework is implemented with custom Triton kernels, whose hardware-aware adaptation enables automatic optimization without manual hyperparameter tuning, making it practical for deployment across diverse computational environments. This work demonstrates how algorithmic adaptivity and hardware awareness can be combined to improve scalability in resource-constrained settings, paving the way for more efficient neural network training on edge devices and cost-sensitive cloud deployments. 

**Abstract (ZH)**: 深层神经网络的优化日益受到GPU内存和计算时间成本的限制。现有的加速技术，如混合精度、二次方法和批量大小放大，通常单独使用。我们提出了Tri-Accel，这是一种统一的优化框架，在训练过程中协同调整三种加速策略及其自适应参数：（1）精度自适应更新，根据曲率和梯度方差动态为各层分配混合精度级别；（2）稀疏二次信号，利用海森矩阵/鱼类子的稀疏模式来指导精度和步长的决策；（3）弹性批量大小调整，根据VRAM可用性实时调整批量大小。在CIFAR-10上使用ResNet-18和EfficientNet-B0，Tri-Accel实现了高达9.9%的训练时间减少和13.3%的更低内存使用，同时在FP32基线基础上提高准确率1.1个百分点。在CIFAR-10/100上测试时，该方法显示自适应学习行为，随着系统在训练过程中学习更有效地分配资源，效率逐渐提高。与静态混合精度训练相比，Tri-Accel在标准硬件上将内存占用从0.35GB减少到0.31GB的同时保持78.1%的准确性。该框架使用自定义的Triton内核实现，其硬件感知的自适应性能够实现自动优化，无需手动调整超参数，使其能够在多样化的计算环境中得到更广泛的应用。这项工作展示了如何将算法自适应性和硬件感知性相结合，以改善资源受限环境中的可扩展性，为边缘设备和成本敏感的云端部署高效神经网络训练铺平道路。 

---
# TriagerX: Dual Transformers for Bug Triaging Tasks with Content and Interaction Based Rankings 

**Title (ZH)**: TriagerX: 基于内容和交互的双Transformer漏洞 triaging 任务排序方法 

**Authors**: Md Afif Al Mamun, Gias Uddin, Lan Xia, Longyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16860)  

**Abstract**: Pretrained Language Models or PLMs are transformer-based architectures that can be used in bug triaging tasks. PLMs can better capture token semantics than traditional Machine Learning (ML) models that rely on statistical features (e.g., TF-IDF, bag of words). However, PLMs may still attend to less relevant tokens in a bug report, which can impact their effectiveness. In addition, the model can be sub-optimal with its recommendations when the interaction history of developers around similar bugs is not taken into account. We designed TriagerX to address these limitations. First, to assess token semantics more reliably, we leverage a dual-transformer architecture. Unlike current state-of-the-art (SOTA) baselines that employ a single transformer architecture, TriagerX collects recommendations from two transformers with each offering recommendations via its last three layers. This setup generates a robust content-based ranking of candidate developers. TriagerX then refines this ranking by employing a novel interaction-based ranking methodology, which considers developers' historical interactions with similar fixed bugs. Across five datasets, TriagerX surpasses all nine transformer-based methods, including SOTA baselines, often improving Top-1 and Top-3 developer recommendation accuracy by over 10%. We worked with our large industry partner to successfully deploy TriagerX in their development environment. The partner required both developer and component recommendations, with components acting as proxies for team assignments-particularly useful in cases of developer turnover or team changes. We trained TriagerX on the partner's dataset for both tasks, and it outperformed SOTA baselines by up to 10% for component recommendations and 54% for developer recommendations. 

**Abstract (ZH)**: 预训练语言模型在软件 bug triaging 任务中的应用：TriagerX的设计与优化 

---
# WildSpoof Challenge Evaluation Plan 

**Title (ZH)**: WildSpoof挑战评估计划 

**Authors**: Yihan Wu, Jee-weon Jung, Hye-jin Shim, Xin Cheng, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16858)  

**Abstract**: The WildSpoof Challenge aims to advance the use of in-the-wild data in two intertwined speech processing tasks. It consists of two parallel tracks: (1) Text-to-Speech (TTS) synthesis for generating spoofed speech, and (2) Spoofing-robust Automatic Speaker Verification (SASV) for detecting spoofed speech. While the organizers coordinate both tracks and define the data protocols, participants treat them as separate and independent tasks. The primary objectives of the challenge are: (i) to promote the use of in-the-wild data for both TTS and SASV, moving beyond conventional clean and controlled datasets and considering real-world scenarios; and (ii) to encourage interdisciplinary collaboration between the spoofing generation (TTS) and spoofing detection (SASV) communities, thereby fostering the development of more integrated, robust, and realistic systems. 

**Abstract (ZH)**: WildSpoof挑战旨在推进野生数据在两个相互关联的语音处理任务中的应用。该挑战包含两个并行赛道：(1) 从文本到语音(TTS)合成以生成欺骗性语音，以及(2) 抗欺骗性的自动说话人验证(SASV)以检测欺骗性语音。尽管组织者协调这两个赛道并定义数据协议，但参与者将它们视为独立任务来对待。该挑战的主要目标是：(i) 促进野生数据在TTS和SASV中的应用，超越传统的清洁和受控数据集，考虑实际场景；以及(ii) 鼓励欺骗性生成(TTS)和欺骗性检测(SASV)社区之间的跨学科合作，从而促进更集成、更 robust 和更现实系统的开发。 

---
# DevLicOps: A Framework for Mitigating Licensing Risks in AI-Generated Code 

**Title (ZH)**: DevLicOps：一种用于缓解AI生成代码许可风险的框架 

**Authors**: Pratyush Nidhi Sharma, Lauren Wright, Anne Herfurth, Munsif Sokiyna, Pratyaksh Nidhi Sharma, Sethu Das, Mikko Siponen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16853)  

**Abstract**: Generative AI coding assistants (ACAs) are widely adopted yet pose serious legal and compliance risks. ACAs can generate code governed by restrictive open-source licenses (e.g., GPL), potentially exposing companies to litigation or forced open-sourcing. Few developers are trained in these risks, and legal standards vary globally, especially with outsourcing. Our article introduces DevLicOps, a practical framework that helps IT leaders manage ACA-related licensing risks through governance, incident response, and informed tradeoffs. As ACA adoption grows and legal frameworks evolve, proactive license compliance is essential for responsible, risk-aware software development in the AI era. 

**Abstract (ZH)**: 生成式AI编码助手（ACAs）在广泛应用的同时也带来了严重的法律和合规风险。ACAs可以生成受限制的开源许可证代码（例如GPL），可能使公司面临诉讼或被迫开源。很少有开发者接受过这些风险的培训，而且在全球范围内，尤其是在外包情况下，法律标准差异很大。本文介绍了DevLicOps，这是一种实用框架，通过治理、事件响应和明智的选择帮助IT领导者管理ACA相关的许可风险。随着ACA的广泛应用和法律框架的不断演变，在AI时代进行负责任的风险意识软件开发需要积极的许可合规。 

---
# A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems 

**Title (ZH)**: 语音认证和防欺骗系统面临的威胁综述 

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2508.16843)  

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems. 

**Abstract (ZH)**: 现代语音认证系统（VAS）及其抗欺骗措施（CMs）的威胁 landscape 及研究进展：包括数据中毒、对抗性攻击、深度合成和对抗性欺骗攻击的综合回顾 

---
# Physics-Inspired Spatial Temporal Graph Neural Networks for Predicting Industrial Chain Resilience 

**Title (ZH)**: 基于物理启发的空间时间图神经网络工业链韧性预测 

**Authors**: Bicheng Wang, Junping Wang, Yibo Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.16836)  

**Abstract**: Industrial chain plays an increasingly important role in the sustainable development of national economy. However, as a typical complex network, data-driven deep learning is still in its infancy in describing and analyzing the resilience of complex networks, and its core is the lack of a theoretical framework to describe the system dynamics. In this paper, we propose a physically informative neural symbolic approach to describe the evolutionary dynamics of complex networks for resilient prediction. The core idea is to learn the dynamics of the activity state of physical entities and integrate it into the multi-layer spatiotemporal co-evolution network, and use the physical information method to realize the joint learning of physical symbol dynamics and spatiotemporal co-evolution topology, so as to predict the industrial chain resilience. The experimental results show that the model can obtain better results and predict the elasticity of the industry chain more accurately and effectively, which has certain practical significance for the development of the industry. 

**Abstract (ZH)**: 工业链在国民经济可持续发展中发挥着越来越重要的作用。然而，作为典型的复杂网络，基于数据的深度学习在描述和分析复杂网络的韧性方面仍处于初级阶段，其核心问题是缺乏一个描述系统动力学的理论框架。在本文中，我们提出了一种物理信息神经符号方法来描述复杂网络的动力学演变以实现韧性预测。核心思想是学习物理实体活动状态的动力学并将其实现到多层时空共演化网络中，并通过物理信息方法实现物理符号动力学与时空共演化拓扑的联合学习，从而预测工业链韧性。实验结果表明，该模型可以取得更好的效果，并更准确有效地预测工业链的弹性，具有一定的实用意义。 

---
# Out of Distribution Detection for Efficient Continual Learning in Quality Prediction for Arc Welding 

**Title (ZH)**: 离分布检测在高效持续学习中的质量预测应用：电弧焊接 

**Authors**: Yannik Hahn, Jan Voets, Antonin Koenigsfeld, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16832)  

**Abstract**: Modern manufacturing relies heavily on fusion welding processes, including gas metal arc welding (GMAW). Despite significant advances in machine learning-based quality prediction, current models exhibit critical limitations when confronted with the inherent distribution shifts that occur in dynamic manufacturing environments. In this work, we extend the VQ-VAE Transformer architecture - previously demonstrating state-of-the-art performance in weld quality prediction - by leveraging its autoregressive loss as a reliable out-of-distribution (OOD) detection mechanism. Our approach exhibits superior performance compared to conventional reconstruction methods, embedding error-based techniques, and other established baselines. By integrating OOD detection with continual learning strategies, we optimize model adaptation, triggering updates only when necessary and thereby minimizing costly labeling requirements. We introduce a novel quantitative metric that simultaneously evaluates OOD detection capability while interpreting in-distribution performance. Experimental validation in real-world welding scenarios demonstrates that our framework effectively maintains robust quality prediction capabilities across significant distribution shifts, addressing critical challenges in dynamic manufacturing environments where process parameters frequently change. This research makes a substantial contribution to applied artificial intelligence by providing an explainable and at the same time adaptive solution for quality assurance in dynamic manufacturing processes - a crucial step towards robust, practical AI systems in the industrial environment. 

**Abstract (ZH)**: 现代制造业高度依赖于焊接工艺，包括气体金属弧焊（GMAW）。尽管基于机器学习的质量预测取得了显著进展，但当前模型在面对动态制造环境中固有的分布偏移时表现出关键的局限性。在本文中，我们通过利用VQ-VAE Transformer架构的自回归损失作为可靠的离群值检测（OOD）机制，扩展了该架构——先前在焊接质量预测任务中展示了最先进的性能。我们的方法在性能上优于传统的重构方法、基于误差的技术以及其他现有的基准。通过将离群值检测与连续学习策略相结合，我们优化了模型的适应性，仅在必要时触发更新，从而最大限度地减少了昂贵的标签要求。我们引入了一个新颖的定量指标，同时评估离群值检测能力和解释聚类内性能。在实际焊接场景下的实验验证表明，我们的框架有效地在显著的分布偏移下维持了稳健的质量预测能力，解决了动态制造环境中过程参数频繁变化的关键挑战。这项研究通过提供一个可解释且适应性强的解决方案，显著地促进了应用于动态制造过程的质量保证，这是工业环境中稳健、实用AI系统的一个关键步骤。 

---
# Understanding and Tackling Over-Dilution in Graph Neural Networks 

**Title (ZH)**: 理解并应对图神经网络中的过度稀释问题 

**Authors**: Junhyun Lee, Veronika Thost, Bumsoo Kim, Jaewoo Kang, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16829)  

**Abstract**: Message Passing Neural Networks (MPNNs) hold a key position in machine learning on graphs, but they struggle with unintended behaviors, such as over-smoothing and over-squashing, due to irregular data structures. The observation and formulation of these limitations have become foundational in constructing more informative graph representations. In this paper, we delve into the limitations of MPNNs, focusing on aspects that have previously been overlooked. Our observations reveal that even within a single layer, the information specific to an individual node can become significantly diluted. To delve into this phenomenon in depth, we present the concept of Over-dilution and formulate it with two dilution factors: intra-node dilution for attribute-level and inter-node dilution for node-level representations. We also introduce a transformer-based solution that alleviates over-dilution and complements existing node embedding methods like MPNNs. Our findings provide new insights and contribute to the development of informative representations. The implementation and supplementary materials are publicly available at this https URL. 

**Abstract (ZH)**: 消息传递神经网络（MPNNs）在图上的机器学习中占据关键位置，但由于不规则的数据结构，它们在不经意的行为，如过度平滑和过度挤压方面存在局限。这些局限性观察和建模已成为构建更具信息量的图表示的基础。在本文中，我们深入探讨了MPNNs的局限性，重点关注先前被忽略的方面。我们的观察表明，即使在单层中，单个节点特有的信息也可能显著稀释。为深入探讨这一现象，我们提出了过度稀释的概念，并通过两种稀释因子对其进行建模：节点内稀释用于属性级别和节点间稀释用于节点级表示。我们还介绍了一种基于transformer的解决方案，该解决方案缓解过度稀释并补充现有的节点嵌入方法，如MPNNs。我们的发现提供了新的见解，并促进了更具信息量的表示的发展。相关实现和补充材料可在以下网址获取：this https URL。 

---
# Exploring the Impact of Generative Artificial Intelligence on Software Development in the IT Sector: Preliminary Findings on Productivity, Efficiency and Job Security 

**Title (ZH)**: 探索生成式人工智能对信息技术sector软件开发的影响：关于生产力、效率和就业安全的初步发现 

**Authors**: Anton Ludwig Bonin, Pawel Robert Smolinski, Jacek Winiarski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16811)  

**Abstract**: This study investigates the impact of Generative AI on software development within the IT sector through a mixed-method approach, utilizing a survey developed based on expert interviews. The preliminary results of an ongoing survey offer early insights into how Generative AI reshapes personal productivity, organizational efficiency, adoption, business strategy and job insecurity. The findings reveal that 97% of IT workers use Generative AI tools, mainly ChatGPT. Participants report significant personal productivity gain and perceive organizational efficiency improvements that correlate positively with Generative AI adoption by their organizations (r = .470, p < .05). However, increased organizational adoption of AI strongly correlates with heightened employee job security concerns (r = .549, p < .001). Key adoption challenges include inaccurate outputs (64.2%), regulatory compliance issues (58.2%) and ethical concerns (52.2%). This research offers early empirical insights into Generative AI's economic and organizational implications. 

**Abstract (ZH)**: 本研究通过混合方法探讨生成式AI在信息技术sector软件开发中的影响，基于专家访谈开发了一份调查问卷。正在进行的调查初步结果提供了生成式AI如何重塑个人生产力、组织效率、采纳、商业策略和工作安全感的早期见解。研究发现，97%的IT工作者使用生成式AI工具，主要为ChatGPT。参与者报告个人生产力显著提升，并感知到其组织中生成式AI采纳带来的组织效率改善（r = .470, p < .05）。然而，组织中AI采纳的增加与员工工作安全感的担忧显著正相关（r = .549, p < .001）。关键采纳挑战包括不准确的输出（64.2%）、合规问题（58.2%）和伦理问题（52.2%）。本研究提供了生成式AI在经济和组织方面的早期实证见解。 

---
# FAIRWELL: Fair Multimodal Self-Supervised Learning for Wellbeing Prediction 

**Title (ZH)**: FAIRWELL: 公平的多模态自监督学习以预测幸福感 

**Authors**: Jiaee Cheong, Abtin Mogharabin, Paul Liang, Hatice Gunes, Sinan Kalkan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16748)  

**Abstract**: Early efforts on leveraging self-supervised learning (SSL) to improve machine learning (ML) fairness has proven promising. However, such an approach has yet to be explored within a multimodal context. Prior work has shown that, within a multimodal setting, different modalities contain modality-unique information that can complement information of other modalities. Leveraging on this, we propose a novel subject-level loss function to learn fairer representations via the following three mechanisms, adapting the variance-invariance-covariance regularization (VICReg) method: (i) the variance term, which reduces reliance on the protected attribute as a trivial solution; (ii) the invariance term, which ensures consistent predictions for similar individuals; and (iii) the covariance term, which minimizes correlational dependence on the protected attribute. Consequently, our loss function, coined as FAIRWELL, aims to obtain subject-independent representations, enforcing fairness in multimodal prediction tasks. We evaluate our method on three challenging real-world heterogeneous healthcare datasets (i.e. D-Vlog, MIMIC and MODMA) which contain different modalities of varying length and different prediction tasks. Our findings indicate that our framework improves overall fairness performance with minimal reduction in classification performance and significantly improves on the performance-fairness Pareto frontier. 

**Abstract (ZH)**: 利用自监督学习提高多模态机器学习公平性的新进展：FAIRWELL方法 

---
# CellEcoNet: Decoding the Cellular Language of Pathology with Deep Learning for Invasive Lung Adenocarcinoma Recurrence Prediction 

**Title (ZH)**: CellEcoNet: 使用深度学习解码病理学的细胞语言以预测侵袭性肺腺癌的复发 

**Authors**: Abdul Rehman Akbar, Usama Sajjad, Ziyu Su, Wencheng Li, Fei Xing, Jimmy Ruiz, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16742)  

**Abstract**: Despite surgical resection, ~70% of invasive lung adenocarcinoma (ILA) patients recur within five years, and current tools fail to identify those needing adjuvant therapy. To address this unmet clinical need, we introduce CellEcoNet, a novel spatially aware deep learning framework that models whole slide images (WSIs) through natural language analogy, defining a "language of pathology," where cells act as words, cellular neighborhoods become phrases, and tissue architecture forms sentences. CellEcoNet learns these context-dependent meanings automatically, capturing how subtle variations and spatial interactions derive recurrence risk. On a dataset of 456 H&E-stained WSIs, CellEcoNet achieved superior predictive performance (AUC:77.8% HR:9.54), outperforming IASLC grading system (AUC:71.4% HR:2.36), AJCC Stage (AUC:64.0% HR:1.17) and state-of-the-art computational methods (AUCs:62.2-67.4%). CellEcoNet demonstrated fairness and consistent performance across diverse demographic and clinical subgroups. Beyond prognosis, CellEcoNet marks a paradigm shift by decoding the tumor microenvironment's cellular "language" to reveal how subtle cell variations encode recurrence risk. 

**Abstract (ZH)**: 基于空间感知的深度学习框架CellEcoNet在肺癌腺癌复发风险预测中的应用：超越当前工具实现精准亚群管理 

---
# AI Product Value Assessment Model: An Interdisciplinary Integration Based on Information Theory, Economics, and Psychology 

**Title (ZH)**: 基于信息理论、经济学和心理学的AI产品价值评估模型 

**Authors**: Yu yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16714)  

**Abstract**: In recent years, breakthroughs in artificial intelligence (AI) technology have triggered global industrial transformations, with applications permeating various fields such as finance, healthcare, education, and manufacturing. However, this rapid iteration is accompanied by irrational development, where enterprises blindly invest due to technology hype, often overlooking systematic value assessments. This paper develops a multi-dimensional evaluation model that integrates information theory's entropy reduction principle, economics' bounded rationality framework, and psychology's irrational decision theories to quantify AI product value. Key factors include positive dimensions (e.g., uncertainty elimination, efficiency gains, cost savings, decision quality improvement) and negative risks (e.g., error probability, impact, and correction costs). A non-linear formula captures factor couplings, and validation through 10 commercial cases demonstrates the model's effectiveness in distinguishing successful and failed products, supporting hypotheses on synergistic positive effects, non-linear negative impacts, and interactive regulations. Results reveal value generation logic, offering enterprises tools to avoid blind investments and promote rational AI industry development. Future directions include adaptive weights, dynamic mechanisms, and extensions to emerging AI technologies like generative models. 

**Abstract (ZH)**: 近年来，人工智能（AI）技术突破引发了全球产业变革，其应用遍及金融、医疗、教育和制造等多个领域。然而，这一快速迭代过程中伴随着盲目发展，企业因技术 hype 盲目投资，往往忽视了系统的价值评估。本文开发了一种多维评估模型，将信息论的熵减原则、经济学的有限理性框架和心理学的非理性决策理论相结合，以量化AI产品的价值。关键因素包括正面维度（如不确定性消除、效率提升、成本节省、决策质量改进）和负面风险（如错误概率、影响程度和修正成本）。非线性公式捕捉了因素间的耦合，通过10个商用案例验证，该模型在区分成功与失败产品方面显示出有效性，并支持协同正面效应、非线性负面影响和互动规制的假说。结果揭示了价值创造逻辑，为企业提供了避免盲目投资、促进理性AI产业发展工具。未来方向包括自适应权重、动态机制及其向生成模型等新兴AI技术的拓展。 

---
# Generative Artificial Intelligence and Agents in Research and Teaching 

**Title (ZH)**: 生成式人工智能与智能代理在研究和教学中的应用 

**Authors**: Jussi S. Jauhiainen, Aurora Toppari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16701)  

**Abstract**: This study provides a comprehensive analysis of the development, functioning, and application of generative artificial intelligence (GenAI) and large language models (LLMs), with an emphasis on their implications for research and education. It traces the conceptual evolution from artificial intelligence (AI) through machine learning (ML) and deep learning (DL) to transformer architectures, which constitute the foundation of contemporary generative systems. Technical aspects, including prompting strategies, word embeddings, and probabilistic sampling methods (temperature, top-k, and top-p), are examined alongside the emergence of autonomous agents. These elements are considered in relation to both the opportunities they create and the limitations and risks they entail.
The work critically evaluates the integration of GenAI across the research process, from ideation and literature review to research design, data collection, analysis, interpretation, and dissemination. While particular attention is given to geographical research, the discussion extends to wider academic contexts. A parallel strand addresses the pedagogical applications of GenAI, encompassing course and lesson design, teaching delivery, assessment, and feedback, with geography education serving as a case example.
Central to the analysis are the ethical, social, and environmental challenges posed by GenAI. Issues of bias, intellectual property, governance, and accountability are assessed, alongside the ecological footprint of LLMs and emerging technological strategies for mitigation. The concluding section considers near- and long-term futures of GenAI, including scenarios of sustained adoption, regulation, and potential decline. By situating GenAI within both scholarly practice and educational contexts, the study contributes to critical debates on its transformative potential and societal responsibilities. 

**Abstract (ZH)**: 本研究对生成人工智能（GenAI）和大型语言模型（LLM）的发展、运作和应用进行了全面分析，并强调了它们对研究和教育的影响。该研究从人工智能（AI）到机器学习（ML）、深度学习（DL）再到变换器架构的逐步演进进行了追溯，后者构成了当代生成系统的基石。研究不仅探讨了这些技术层面的内容，包括提示策略、词嵌入和概率采样方法（温度、top-k和top-p），而且还分析了自主智能体的出现。这些元素与其带来的机会以及涉及的局限性和风险共同被纳入考量。

该研究批判性地评估了GenAI在研究过程中的整合，从构想和文献综述到研究设计、数据收集、分析、解释和传播等各个环节。特别关注了地理研究领域，同时也延伸到更广泛的学术背景下。另一条线索则探讨了GenAI的教学应用，包括课程和教学设计、教学实施、评估和反馈，地理教育被用作案例研究。

分析的核心在于GenAI提出的伦理、社会和环境挑战。评估了偏见、知识产权、治理和问责制等问题，同时关注了LLM的生态足迹和新兴的技术缓解策略。研究的最后一部分考虑了GenAI的近期和远期未来，包括持续采用、监管和潜在下降的场景。通过对GenAI既置于学术实践又置于教育背景中的探讨，该研究为有关其变革潜力及其社会责任的批判性辩论做出了贡献。 

---
# DecoMind: A Generative AI System for Personalized Interior Design Layouts 

**Title (ZH)**: DecoMind: 个性化室内设计布局的生成型AI系统 

**Authors**: Reema Alshehri, Rawan Alotaibi, Leen Almasri, Rawan Altaweel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16696)  

**Abstract**: This paper introduces a system for generating interior design layouts based on user inputs, such as room type, style, and furniture preferences. CLIP extracts relevant furniture from a dataset, and a layout that contains furniture and a prompt are fed to Stable Diffusion with ControlNet to generate a design that incorporates the selected furniture. The design is then evaluated by classifiers to ensure alignment with the user's inputs, offering an automated solution for realistic interior design. 

**Abstract (ZH)**: 基于用户输入生成室内设计布局的系统：CLIP提取相关家具并与Stable Diffusion和ControlNet协作生成设计，并通过分类器评估以确保与用户输入一致，提供一种现实主义室内设计的自动化解决方案。 

---
# Making AI Inevitable: Historical Perspective and the Problems of Predicting Long-Term Technological Change 

**Title (ZH)**: 让AI不可避免：历史视角与预测长期技术变革的问题 

**Authors**: Mark Fisher, John Severini  

**Link**: [PDF](https://arxiv.org/pdf/2508.16692)  

**Abstract**: This study demonstrates the extent to which prominent debates about the future of AI are best understood as subjective, philosophical disagreements over the history and future of technological change rather than as objective, material disagreements over the technologies themselves. It focuses on the deep disagreements over whether artificial general intelligence (AGI) will prove transformative for human society; a question that is analytically prior to that of whether this transformative effect will help or harm humanity. The study begins by distinguishing two fundamental camps in this debate. The first of these can be identified as "transformationalists," who argue that continued AI development will inevitably have a profound effect on society. Opposed to them are "skeptics," a more eclectic group united by their disbelief that AI can or will live up to such high expectations. Each camp admits further "strong" and "weak" variants depending on their tolerance for epistemic risk. These stylized contrasts help to identify a set of fundamental questions that shape the camps' respective interpretations of the future of AI. Three questions in particular are focused on: the possibility of non-biological intelligence, the appropriate time frame of technological predictions, and the assumed trajectory of technological development. In highlighting these specific points of non-technical disagreement, this study demonstrates the wide range of different arguments used to justify either the transformationalist or skeptical position. At the same time, it highlights the strong argumentative burden of the transformationalist position, the way that belief in this position creates competitive pressures to achieve first-mover advantage, and the need to widen the concept of "expertise" in debates surrounding the future development of AI. 

**Abstract (ZH)**: 这一研究展示了对未来AI的广泛关注最好被理解为关于技术变革历史和未来的主观哲学分歧，而非对技术本身的具体物质分歧。该研究集中在对通用人工智能（AGI）是否会对人类社会产生变革性影响的深刻分歧上；这是一个比这种变革性影响是利是害更为基本的问题。研究一开始便区分了这场辩论中的两大基本阵营。第一个阵营可被识别为“变革主义者”，他们认为持续的AI发展将不可避免地对社会产生深远影响。反对他们的是“怀疑论者”，这是一个更为多元的群体，统一于他们不相信AI能或会达到如此高的期望。每个阵营又进一步分为“强硬派”和“温和派”变体，这取决于他们对认识论风险的容忍度。这些理论上的对比有助于识别塑造两大阵营对AI未来理解的若干核心问题。特别是关注三个问题：非生物学智能的可能性、技术预测的适当时间框架以及技术发展的假设轨迹。通过强调这些具体的非技术性分歧，该研究展示了支持变革主义者或怀疑论者的不同论点的广泛范围。同时，该研究强调了变革主义者立场的强大论点论证负担、这种立场如何创造了抢先优势的竞争压力以及在关于AI未来发展辩论中扩展“专家”概念的必要性。 

---
# STGAtt: A Spatial-Temporal Unified Graph Attention Network for Traffic Flow Forecasting 

**Title (ZH)**: STGAtt：一种用于交通流量预测的时空统一图注意力网络 

**Authors**: Zhuding Liang, Jianxun Cui, Qingshuang Zeng, Feng Liu, Nenad Filipovic, Tijana Geroski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16685)  

**Abstract**: Accurate and timely traffic flow forecasting is crucial for intelligent transportation systems. This paper presents a novel deep learning model, the Spatial-Temporal Unified Graph Attention Network (STGAtt). By leveraging a unified graph representation and an attention mechanism, STGAtt effectively captures complex spatial-temporal dependencies. Unlike methods relying on separate spatial and temporal dependency modeling modules, STGAtt directly models correlations within a Spatial-Temporal Unified Graph, dynamically weighing connections across both dimensions. To further enhance its capabilities, STGAtt partitions traffic flow observation signal into neighborhood subsets and employs a novel exchanging mechanism, enabling effective capture of both short-range and long-range correlations. Extensive experiments on the PEMS-BAY and SHMetro datasets demonstrate STGAtt's superior performance compared to state-of-the-art baselines across various prediction horizons. Visualization of attention weights confirms STGAtt's ability to adapt to dynamic traffic patterns and capture long-range dependencies, highlighting its potential for real-world traffic flow forecasting applications. 

**Abstract (ZH)**: 准确及时的交通流预测对于智能交通系统至关重要。本文提出了一种新颖的深度学习模型，即空间-时间统一图注意力网络（STGAtt）。通过利用统一的图表示和注意力机制，STGAtt 有效地捕捉了复杂的空-时依赖关系。与依赖于独立的空间和时间依赖性建模模块的方法不同，STGAtt 直接在空间-时间统一图中建模内部的联系，并动态权衡两个维度上的连接。为了进一步提升其能力，STGAtt 将交通流观测信号划分为邻域子集，并采用了一种新型的交换机制，能够有效地捕捉短程和远程的依赖关系。在 PEMS-BAY 和 SHMetro 数据集上的 extensive 实验表明，STGAtt 在各种预测时间范围内的性能优于最先进的基线方法。注意力权重的可视化结果证实了 STGAtt 适应动态交通模式并捕捉远程依赖性的能力，突显了其在实际交通流预测应用中的潜力。 

---
# MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation 

**Title (ZH)**: MedRepBench: 医学报告解读的综合基准 

**Authors**: Fangxin Shang, Yuan Xia, Dalu Yang, Yahui Wang, Binglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16674)  

**Abstract**: Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding. 

**Abstract (ZH)**: 医疗报告解读在医疗保健中扮演着至关重要的角色，能够提供面向患者的解释并促进临床系统之间的有效信息流动。尽管近期的视觉-语言模型和大规模语言模型展示了通用文档理解能力，但在评估医疗报告结构化解读质量方面仍然缺乏标准化基准。我们介绍了MedRepBench，这是一个基于1900份去标识化的现实世界中文医疗报告构建的综合基准，涵盖多个科室、患者人口统计特征和获取格式。该基准主要用于评估端到端的视觉-语言模型在结构化医疗报告理解中的性能。为了进行可控比较，我们还引入了一个仅文本评估设置，结合高质量的OCR输出和语言模型，以估计在最小化字符识别错误时的上界性能。我们的评估框架支持两种互补的协议：（1）基于客观指标的字段级别召回率的评估，以及（2）使用强大语言模型作为评分代理的自动主观评估，以评估事实性、可解释性和推理质量。根据客观指标，我们进一步设计了一个奖励函数，并应用组相对策略优化（GRPO）来提升中规模的视觉-语言模型，实现了高达6%的召回率提升。我们还观察到，尽管OCR+LLM流水线表现出色，但仍存在布局盲视和延迟问题，这促使我们进一步向稳健的整体视觉报告理解方向发展。 

---
# The AI Model Risk Catalog: What Developers and Researchers Miss About Real-World AI Harms 

**Title (ZH)**: AI模型风险目录：开发者和研究人员忽视的现实世界AI危害 

**Authors**: Pooja S. B. Rao, Sanja Šćepanović, Dinesh Babu Jayagopi, Mauro Cherubini, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2508.16672)  

**Abstract**: We analyzed nearly 460,000 AI model cards from Hugging Face to examine how developers report risks. From these, we extracted around 3,000 unique risk mentions and built the \emph{AI Model Risk Catalog}. We compared these with risks identified by researchers in the MIT Risk Repository and with real-world incidents from the AI Incident Database. Developers focused on technical issues like bias and safety, while researchers emphasized broader social impacts. Both groups paid little attention to fraud and manipulation, which are common harms arising from how people interact with AI. Our findings show the need for clearer, structured risk reporting that helps developers think about human-interaction and systemic risks early in the design process. The catalog and paper appendix are available at: this https URL. 

**Abstract (ZH)**: 我们分析了Hugging Face上的近46万个AI模型卡片，以考察开发者如何报告风险。我们从中提取了大约3000个独特的风险提及，构建了《AI模型风险目录》。我们将这些风险与MIT风险存储库中识别的风险以及AI事故数据库中的实际事件进行了比较。开发人员主要关注技术问题如偏见和安全性，而研究人员则更侧重于更广泛的社会影响。两组对欺诈和操纵这类常见的交互造成的危害关注度不高。我们的研究结果表明，需要更清晰和结构化的风险报告，以帮助开发者在设计初期就考虑到人类交互和系统性风险。该目录及其论文附录可在以下链接获取：this https URL。 

---
# Reflective Paper-to-Code Reproduction Enabled by Fine-Grained Verification 

**Title (ZH)**: 细粒度验证支持的反思性纸笔代码重现 

**Authors**: Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, Da Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16671)  

**Abstract**: Reproducing machine learning papers is essential for scientific progress but remains challenging for both humans and automated agents. Existing agent-based methods often struggle to fully and accurately reproduce implementation details such as mathematical formulas and algorithmic logic. Previous studies show that reflection with explicit feedback improves agent performance. However, current paper reproduction methods fail to effectively adopt this strategy. This gap mainly arises from the diverse paper patterns, complex method modules, and varied configurations encountered in research papers. Motivated by how humans use systematic checklists to efficiently debug complex code, we propose \textbf{RePro}, a \textbf{Re}flective Paper-to-Code \textbf{Repro}duction framework that automatically extracts a paper's fingerprint, referring to a comprehensive set of accurate and atomic criteria serving as high-quality supervisory signals. The framework first generates code based on the extracted information, and then leverages the fingerprint within iterative verification and refinement loop. This approach systematically detects discrepancies and produces targeted revisions to align generated code with the paper's implementation details. Extensive experiments on the PaperBench Code-Dev benchmark have been conducted, RePro achieves 13.0\% performance gap over baselines, and it correctly revises complex logical and mathematical criteria in reflecting, on which the effectiveness is obvious. 

**Abstract (ZH)**: 机器学习论文的再现对于科学进步至关重要，但对人类和自动化代理而言仍具有挑战性。现有的基于代理的方法往往难以全面准确地再现实现细节，如数学公式和算法逻辑。先前的研究表明，带有显反馈的反思可以提升代理性能。然而，当前的论文再现方法未能有效采用这一策略。这一差距主要源于研究论文中遇到的多样论文模式、复杂的方法模块和不同的配置。受人类如何使用系统化检查列表高效调试复杂代码的启发，我们提出了一种名为RePro的Reflective Paper-to-Code Reproduction框架，自动提取论文指纹，参考一个全面的、准确且原子的标准集合作为高质量的监督信号。该框架首先基于提取的信息生成代码，然后利用指纹在迭代验证和改进循环中进行验证和细化。这种方法系统性地检测不一致之处并生成针对性修订，以使生成的代码与论文的实现细节保持一致。在PaperBench Code-Dev基准测试上的 extensive 实验表明，RePro 的性能相较于baseline提升了13.0%，在反映过程中正确修订了复杂的逻辑和数学标准，其有效性显而易见。 

---
# Situational Awareness as the Imperative Capability for Disaster Resilience in the Era of Complex Hazards and Artificial Intelligence 

**Title (ZH)**: 态势感知作为复杂灾害与人工智能时代灾害韧性的重要能力 

**Authors**: Hongrak Pak, Ali Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16669)  

**Abstract**: Disasters frequently exceed established hazard models, revealing blind spots where unforeseen impacts and vulnerabilities hamper effective response. This perspective paper contends that situational awareness (SA)-the ability to perceive, interpret, and project dynamic crisis conditions-is an often overlooked yet vital capability for disaster resilience. While risk mitigation measures can reduce known threats, not all hazards can be neutralized; truly adaptive resilience hinges on whether organizations rapidly detect emerging failures, reconcile diverse data sources, and direct interventions where they matter most. We present a technology-process-people roadmap, demonstrating how real-time hazard nowcasting, interoperable workflows, and empowered teams collectively transform raw data into actionable insight. A system-of-systems approach enables federated data ownership and modular analytics, so multiple agencies can share timely updates without sacrificing their distinct operational models. Equally crucial, structured sense-making routines and cognitive load safeguards help humans remain effective decision-makers amid data abundance. By framing SA as a socio-technical linchpin rather than a peripheral add-on, this paper spotlights the urgency of elevating SA to a core disaster resilience objective. We conclude with recommendations for further research-developing SA metrics, designing trustworthy human-AI collaboration, and strengthening inclusive data governance-to ensure that communities are equipped to cope with both expected and unexpected crises. 

**Abstract (ZH)**: 灾害频繁超出已建立的危害模型，揭示出存在未预见影响和脆弱性的盲点，这些盲点阻碍了有效的应对。本文观点认为，情况感知（SA）——即感知、解释和预测动态危机条件的能力——是灾害韧性中一个常被忽视但至关重要的能力。虽然风险管理措施可以降低已知威胁，但并非所有危害都能被消除；真正适应性的韧性取决于组织能否迅速检测到新兴的失败、协调多元数据源，并将干预措施集中在最关键的地方。本文提出了一条技术-流程-人员的道路图，展示出如何通过实时危害现在casting、互操作的工作流程以及赋能的团队将原始数据转化为可行动的洞见。系统中的系统的方法使联邦数据拥有权和模块化分析得以实现，从而多个机构可以共享及时更新而不牺牲其独特的操作模型。同样重要的是，结构化的意义构建范式和认知负担的安全措施有助于人们在数据过剩的情况下保持有效的决策能力。通过将SA框架为社会技术的关键节点而非外围附加部分，本文突显了提高SA作为核心灾害韧性目标的紧迫性。最后，本文提出了进一步研究的建议，包括开发SA指标、设计值得信赖的人工智能合作以及加强包容性数据治理，以确保社区能够应对预期和未预期的危机。 

---
# HiCL: Hippocampal-Inspired Continual Learning 

**Title (ZH)**: hippocampal启发的连续学习 

**Authors**: Kushal Kapoor, Wyatt Mackey, Yiannis Aloimonos, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16651)  

**Abstract**: We propose HiCL, a novel hippocampal-inspired dual-memory continual learning architecture designed to mitigate catastrophic forgetting by using elements inspired by the hippocampal circuitry. Our system encodes inputs through a grid-cell-like layer, followed by sparse pattern separation using a dentate gyrus-inspired module with top-k sparsity. Episodic memory traces are maintained in a CA3-like autoassociative memory. Task-specific processing is dynamically managed via a DG-gated mixture-of-experts mechanism, wherein inputs are routed to experts based on cosine similarity between their normalized sparse DG representations and learned task-specific DG prototypes computed through online exponential moving averages. This biologically grounded yet mathematically principled gating strategy enables differentiable, scalable task-routing without relying on a separate gating network, and enhances the model's adaptability and efficiency in learning multiple sequential tasks. Cortical outputs are consolidated using Elastic Weight Consolidation weighted by inter-task similarity. Crucially, we incorporate prioritized replay of stored patterns to reinforce essential past experiences. Evaluations on standard continual learning benchmarks demonstrate the effectiveness of our architecture in reducing task interference, achieving near state-of-the-art results in continual learning tasks at lower computational costs. 

**Abstract (ZH)**: HiCL：一种启发自海马体的新型双记忆连续学习架构及其在减轻灾难性遗忘方面的应用 

---
# LatentFlow: Cross-Frequency Experimental Flow Reconstruction from Sparse Pressure via Latent Mapping 

**Title (ZH)**: 潜在流形：通过潜在映射从稀疏压力进行跨频率实验流重建 

**Authors**: Junle Liu, Chang Liu, Yanyu Ke, Qiuxiang Huang, Jiachen Zhao, Wenliang Chen, K.T. Tse, Gang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16648)  

**Abstract**: Acquiring temporally high-frequency and spatially high-resolution turbulent wake flow fields in particle image velocimetry (PIV) experiments remains a significant challenge due to hardware limitations and measurement noise. In contrast, temporal high-frequency measurements of spatially sparse wall pressure are more readily accessible in wind tunnel experiments. In this study, we propose a novel cross-modal temporal upscaling framework, LatentFlow, which reconstructs high-frequency (512 Hz) turbulent wake flow fields by fusing synchronized low-frequency (15 Hz) flow field and pressure data during training, and high-frequency wall pressure signals during inference. The first stage involves training a pressure-conditioned $\beta$-variation autoencoder ($p$C-$\beta$-VAE) to learn a compact latent representation that captures the intrinsic dynamics of the wake flow. A secondary network maps synchronized low-frequency wall pressure signals into the latent space, enabling reconstruction of the wake flow field solely from sparse wall pressure. Once trained, the model utilizes high-frequency, spatially sparse wall pressure inputs to generate corresponding high-frequency flow fields via the $p$C-$\beta$-VAE decoder. By decoupling the spatial encoding of flow dynamics from temporal pressure measurements, LatentFlow provides a scalable and robust solution for reconstructing high-frequency turbulent wake flows in data-constrained experimental settings. 

**Abstract (ZH)**: 基于交叉模态时间上尺度的LatentFlow：在数据受限实验条件下重构高频湍涡尾流场 

---
# Few-shot Class-incremental Fault Diagnosis by Preserving Class-Agnostic Knowledge with Dual-Granularity Representations 

**Title (ZH)**: 基于双粒度表示保留无类别依赖知识的少量样本类别增量故障诊断 

**Authors**: Zhendong Yang, Jie Wang, Liansong Zong, Xiaorong Liu, Quan Qian, Shiqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16634)  

**Abstract**: Few-Shot Class-Incremental Fault Diagnosis (FSC-FD), which aims to continuously learn from new fault classes with only a few samples without forgetting old ones, is critical for real-world industrial systems. However, this challenging task severely amplifies the issues of catastrophic forgetting of old knowledge and overfitting on scarce new data. To address these challenges, this paper proposes a novel framework built upon Dual-Granularity Representations, termed the Dual-Granularity Guidance Network (DGGN). Our DGGN explicitly decouples feature learning into two parallel streams: 1) a fine-grained representation stream, which utilizes a novel Multi-Order Interaction Aggregation module to capture discriminative, class-specific features from the limited new samples. 2) a coarse-grained representation stream, designed to model and preserve general, class-agnostic knowledge shared across all fault types. These two representations are dynamically fused by a multi-semantic cross-attention mechanism, where the stable coarse-grained knowledge guides the learning of fine-grained features, preventing overfitting and alleviating feature conflicts. To further mitigate catastrophic forgetting, we design a Boundary-Aware Exemplar Prioritization strategy. Moreover, a decoupled Balanced Random Forest classifier is employed to counter the decision boundary bias caused by data imbalance. Extensive experiments on the TEP benchmark and a real-world MFF dataset demonstrate that our proposed DGGN achieves superior diagnostic performance and stability compared to state-of-the-art FSC-FD approaches. Our code is publicly available at this https URL 

**Abstract (ZH)**: Few-Shot 类内增量故障诊断 (FSC-FD)：基于双粒度表示的双粒度引导网络 

---
# Adaptive Variance-Penalized Continual Learning with Fisher Regularization 

**Title (ZH)**: 适应性方差惩罚的持续学习与费希尔正则化 

**Authors**: Krisanu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16632)  

**Abstract**: The persistent challenge of catastrophic forgetting in neural networks has motivated extensive research in continual learning . This work presents a novel continual learning framework that integrates Fisher-weighted asymmetric regularization of parameter variances within a variational learning paradigm. Our method dynamically modulates regularization intensity according to parameter uncertainty, achieving enhanced stability and performance. Comprehensive evaluations on standard continual learning benchmarks including SplitMNIST, PermutedMNIST, and SplitFashionMNIST demonstrate substantial improvements over existing approaches such as Variational Continual Learning and Elastic Weight Consolidation . The asymmetric variance penalty mechanism proves particularly effective in maintaining knowledge across sequential tasks while improving model accuracy. Experimental results show our approach not only boosts immediate task performance but also significantly mitigates knowledge degradation over time, effectively addressing the fundamental challenge of catastrophic forgetting in neural networks 

**Abstract (ZH)**: 神经网络中灾难性遗忘的持久挑战推动了持续学习研究的广泛开展。本工作提出了一种新的持续学习框架，该框架在变分学习范式中集成 Fisher 权重加权非对称正则化参数方差，动态调节正则化强度以参数不确定性为基础，实现增强的稳定性和性能。在包括 SplitMNIST、PermutedMNIST 和 SplitFashionMNIST 的标准持续学习基准上的全面评估显示，本方法在与现有方法如变分持续学习和弹性权重巩固相比时表现出显著的改进。非对称方差惩罚机制特别有效，能够在顺序任务中维持知识并提高模型准确性。实验结果表明，本方法不仅提升了当前任务的性能，还显著减少了随时间推移的知识退化，有效解决了神经网络中灾难性遗忘的基本挑战。 

---
# The Impact of Artificial Intelligence on Human Thought 

**Title (ZH)**: 人工智能对人类思维的影响 

**Authors**: Rénald Gesnot  

**Link**: [PDF](https://arxiv.org/pdf/2508.16628)  

**Abstract**: This research paper examines, from a multidimensional perspective (cognitive, social, ethical, and philosophical), how AI is transforming human thought. It highlights a cognitive offloading effect: the externalization of mental functions to AI can reduce intellectual engagement and weaken critical thinking. On the social level, algorithmic personalization creates filter bubbles that limit the diversity of opinions and can lead to the homogenization of thought and polarization. This research also describes the mechanisms of algorithmic manipulation (exploitation of cognitive biases, automated disinformation, etc.) that amplify AI's power of influence. Finally, the question of potential artificial consciousness is discussed, along with its ethical implications. The report as a whole underscores the risks that AI poses to human intellectual autonomy and creativity, while proposing avenues (education, transparency, governance) to align AI development with the interests of humanity. 

**Abstract (ZH)**: 这篇研究论文从认知、社会、伦理和哲学多维度探讨了AI如何变革人类思维。它突出了认知卸载效应：将心理功能外移到AI可以减少智力参与并削弱批判性思维。在社会层面，算法个性化创造出信息茧房，限制了意见的多样性，并可能导致思想同质化和极化。该研究还描述了算法操控机制（利用认知偏差、自动化假信息等），这些机制放大了AI的影响能力。最后，讨论了潜在的人工意识问题及其伦理影响。整个报告强调了AI对人类智力自主性和创造力的潜在风险，并提出教育、透明度和治理等途径，以确保AI发展符合人类利益。 

---
# Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection 

**Title (ZH)**: 数据和上下文至关重要：面向基于AI的软件漏洞检测的泛化研究 

**Authors**: Rijha Safdar, Danyail Mateen, Syed Taha Ali, M. Umer Ashfaq, Wajahat Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.16625)  

**Abstract**: The performance of AI-based software vulnerability detection systems is often limited by their poor generalization to unknown codebases. In this research, we explore the impact of data quality and model architecture on the generalizability of vulnerability detection systems. By generalization we mean ability of high vulnerability detection performance across different C/C++ software projects not seen during training. Through a series of experiments, we demonstrate that improvements in dataset diversity and quality substantially enhance detection performance. Additionally, we compare multiple encoder-only and decoder-only models, finding that encoder based models outperform in terms of accuracy and generalization. Our model achieves 6.8% improvement in recall on the benchmark BigVul[1] dataset, also outperforming on unseen projects, hence showing enhanced generalizability. These results highlight the role of data quality and model selection in the development of robust vulnerability detection systems. Our findings suggest a direction for future systems having high cross-project effectiveness. 

**Abstract (ZH)**: 基于AI的软件漏洞检测系统在未知代码库上的性能往往受限于其较差的泛化能力。本研究探讨了数据质量和模型架构对漏洞检测系统泛化能力的影响。通过一系列实验，我们证明了数据集多样性和质量的提升显著提高了检测性能。此外，我们比较了多种编码器-only和解码器-only模型，发现基于编码器的模型在准确性与泛化能力上表现更优。我们的模型在基准数据集BigVul上召回率提高了6.8%，并且在未见过的项目上也表现出色，从而展示了增强的泛化能力。这些结果突显了数据质量和模型选择在开发稳健的漏洞检测系统中的作用。我们的发现指出了未来系統高跨项目效应的一个发展方向。 

---
# The GPT-4o Shock Emotional Attachment to AI Models and Its Impact on Regulatory Acceptance: A Cross-Cultural Analysis of the Immediate Transition from GPT-4o to GPT-5 

**Title (ZH)**: GPT-4o 情感黏着于AI模型及其对监管接受度的影响：从GPT-4o到GPT-5的即时过渡的跨文化分析 

**Authors**: Hiroki Naito  

**Link**: [PDF](https://arxiv.org/pdf/2508.16624)  

**Abstract**: In August 2025, a major AI company's immediate, mandatory transition from its previous to its next-generation model triggered widespread public reactions. I collected 150 posts in Japanese and English from multiple social media platforms and video-sharing services between August 8-9, 2025, and qualitatively analyzed expressions of emotional attachment and resistance. Users often described GPT-4o as a trusted partner or AI boyfriend, suggesting person-like bonds. Japanese posts were dominated by loss-oriented narratives, whereas English posts included more anger, meta-level critique, and memes.A preliminary quantitative check showed a statistically significant difference in attachment coding between Japanese and English posts, with substantially higher attachment observed in the Japanese data. The findings suggest that for attachment-heavy models, even safety-oriented changes can face rapid, large-scale resistance that narrows the practical window for behavioral control. If future AI robots capable of inducing emotional bonds become widespread in the physical world, such attachment could surpass the ability to enforce regulation at an even earlier stage than in digital settings. Policy options include gradual transitions, parallel availability, and proactive measurement of attachment thresholds and points of no return to prevent emotional dynamics from outpacing effective governance. 

**Abstract (ZH)**: 2025年8月，一家主要AI公司在其从上一代模型立即、强制过渡到下一代模型时引发了广泛的社会反应。我收集了2025年8月8日至9日多个社交平台和视频分享服务上的150条日文和英文帖子，并对其进行了定性分析，探讨了情感依附和抵抗的表达。用户常将GPT-4o描述为可信赖的伙伴或AI男友，暗示人类似的关系。日文帖子主要集中在悲伤叙事上，而英文帖子则包含更多愤怒、元层面的批评和梗图。初步的定量检查显示，日文和英文帖子在依附编码上的差异具有统计学意义，日文数据中的依附程度显著更高。研究结果表明，对于依附导向的模型，即使是安全导向的更改也可能迅速引发大规模的抵抗，从而缩短行为控制的可行窗口。如果未来具备诱发情感联系的AI机器人在现实世界中普及，这种依附可能比在数字设置中更早地超越监管能力。政策选项包括逐步过渡、并行可用性及主动测量依附阈值和临界点，以防止情感动态超越有效的治理。 

---
# A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction 

**Title (ZH)**: 基于检索增强的空间时间框架的交通预测 

**Authors**: Weilin Ruan, Xilin Dang, Ziyu Zhou, Sisuo Lyu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16623)  

**Abstract**: Traffic prediction is a cornerstone of modern intelligent transportation systems and a critical task in spatio-temporal forecasting. Although advanced Spatio-temporal Graph Neural Networks (STGNNs) and pre-trained models have achieved significant progress in traffic prediction, two key challenges remain: (i) limited contextual capacity when modeling complex spatio-temporal dependencies, and (ii) low predictability at fine-grained spatio-temporal points due to heterogeneous patterns. Inspired by Retrieval-Augmented Generation (RAG), we propose RAST, a universal framework that integrates retrieval-augmented mechanisms with spatio-temporal modeling to address these challenges. Our framework consists of three key designs: 1) Decoupled Encoder and Query Generator to capture decoupled spatial and temporal features and construct a fusion query via residual fusion; 2) Spatio-temporal Retrieval Store and Retrievers to maintain and retrieve vectorized fine-grained patterns; and 3) Universal Backbone Predictor that flexibly accommodates pre-trained STGNNs or simple MLP predictors. Extensive experiments on six real-world traffic networks, including large-scale datasets, demonstrate that RAST achieves superior performance while maintaining computational efficiency. 

**Abstract (ZH)**: 基于检索增强的时空交通预测框架（RAST） 

---
# STRelay: A Universal Spatio-Temporal Relaying Framework for Location Prediction with Future Spatiotemporal Contexts 

**Title (ZH)**: STRelay: 一种考虑未来时空上下文的通用时空 relay 预测框架 

**Authors**: Bangchao Deng, Lianhua Ji, Chunhua Chen, Xin Jing, Ling Ding, Bingqing QU, Pengyang Wang, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16620)  

**Abstract**: Next location prediction is a critical task in human mobility modeling, enabling applications like travel planning and urban mobility management. Existing methods mainly rely on historical spatiotemporal trajectory data to train sequence models that directly forecast future locations. However, they often overlook the importance of the future spatiotemporal contexts, which are highly informative for the future locations. For example, knowing how much time and distance a user will travel could serve as a critical clue for predicting the user's next location. Against this background, we propose \textbf{STRelay}, a universal \textbf{\underline{S}}patio\textbf{\underline{T}}emporal \textbf{\underline{Relay}}ing framework explicitly modeling the future spatiotemporal context given a human trajectory, to boost the performance of different location prediction models. Specifically, STRelay models future spatiotemporal contexts in a relaying manner, which is subsequently integrated with the encoded historical representation from a base location prediction model, enabling multi-task learning by simultaneously predicting the next time interval, next moving distance interval, and finally the next location. We evaluate STRelay integrated with four state-of-the-art location prediction base models on four real-world trajectory datasets. Results demonstrate that STRelay consistently improves prediction performance across all cases by 3.19\%-11.56\%. Additionally, we find that the future spatiotemporal contexts are particularly helpful for entertainment-related locations and also for user groups who prefer traveling longer distances. The performance gain on such non-daily-routine activities, which often suffer from higher uncertainty, is indeed complementary to the base location prediction models that often excel at modeling regular daily routine patterns. 

**Abstract (ZH)**: 下一步位置预测是人类移动建模中的关键任务，能够支持旅行规划和城市移动管理等应用。现有方法主要依赖历史时空轨迹数据训练序列模型直接预测未来位置，但通常忽视了对未来时空上下文的重要性，而这些上下文对于预测未来位置非常有用。例如，知道用户将要花费多少时间和距离移动，可以作为预测用户下一步位置的关键线索。基于此，我们提出了一种新的时空接力框架STRelay，该框架明确建模给定人类轨迹的未来时空上下文，以提升不同位置预测模型的性能。具体而言，STRelay通过接力的方式建模未来时空上下文，并将其与基础位置预测模型的编码历史表示融合，从而通过同时预测下一时段、下一步移动距离和最终位置来实现多任务学习。我们将STRelay分别与四种最新位置预测基础模型在四种真实世界轨迹数据集上进行评估。结果显示，STRelay在所有情况下均能一致地提高预测性能，增幅为3.19%-11.56%。此外，我们发现未来时空上下文特别有助于娱乐相关位置的预测，也对偏好长途旅行的用户群体有益。对于这类非日常活动，由于其不确定性较高，性能提升确实补充了基础位置预测模型在建模日常规律模式方面的优势。 

---
# Negative Shanshui: Real-time Interactive Ink Painting Synthesis 

**Title (ZH)**: 负山水：实时交互式水墨画合成 

**Authors**: Aven-Le Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16612)  

**Abstract**: This paper presents Negative Shanshui, a real-time interactive AI synthesis approach that reinterprets classical Chinese landscape ink painting, i.e., shanshui, to engage with ecological crises in the Anthropocene. Negative Shanshui optimizes a fine-tuned Stable Diffusion model for real-time inferences and integrates it with gaze-driven inpainting, frame interpolation; it enables dynamic morphing animations in response to the viewer's gaze and presents as an interactive virtual reality (VR) experience. The paper describes the complete technical pipeline, covering the system framework, optimization strategies, gaze-based interaction, and multimodal deployment in an art festival. Further analysis of audience feedback collected during its public exhibition highlights how participants variously engaged with the work through empathy, ambivalence, and critical reflection. 

**Abstract (ZH)**: 这篇论文介绍了 Negative Shanshui，这是一种实时互动的AI合成方法，重新解读了传统的中国山水墨画，即山水画，以应对人类世的生态危机。Negative Shanshui 对微调后的 Stable Diffusion 模型进行优化以实现实时推理，并将其与注视驱动的修复、帧插值相结合，能够根据观众的注视生成动态形态动画，并作为互动虚拟现实（VR）体验呈现。论文描述了完整的技术流程，涵盖了系统框架、优化策略、基于注视的交互以及在艺术节中的多模态部署。进一步分析其公共展览期间收集的观众反馈表明，参与者通过共情、矛盾和批判性反思等各种方式与作品进行了互动。 

---
# To Explain Or Not To Explain: An Empirical Investigation Of AI-Based Recommendations On Social Media Platforms 

**Title (ZH)**: 是否需要解释：基于AI的推荐在社交媒体平台上的实证研究 

**Authors**: AKM Bahalul Haque, A.K.M. Najmul Islam, Patrick Mikalef  

**Link**: [PDF](https://arxiv.org/pdf/2508.16610)  

**Abstract**: AI based social media recommendations have great potential to improve the user experience. However, often these recommendations do not match the user interest and create an unpleasant experience for the users. Moreover, the recommendation system being a black box creates comprehensibility and transparency issues. This paper investigates social media recommendations from an end user perspective. For the investigation, we used the popular social media platform Facebook and recruited regular users to conduct a qualitative analysis. We asked participants about the social media content suggestions, their comprehensibility, and explainability. Our analysis shows users mostly require explanation whenever they encounter unfamiliar content and to ensure their online data security. Furthermore, the users require concise, non-technical explanations along with the facility of controlled information flow. In addition, we observed that explanations impact the users perception of transparency, trust, and understandability. Finally, we have outlined some design implications and presented a synthesized framework based on our data analysis. 

**Abstract (ZH)**: 基于AI的社会媒体推荐具有大幅提升用户体验的潜力。然而，这些推荐往往不能匹配用户兴趣，给用户带来不愉快的体验。此外，由于推荐系统是一个黑箱，这造成了可解释性和透明度的问题。本文从最终用户的角度探讨社会媒体推荐。我们使用流行的社交媒体平台Facebook并招募常规用户进行定性分析，询问参与者关于社交媒体内容建议、可解释性和可理解性的看法。我们的分析显示，用户在遇到不熟悉的内容时大多需要解释，并且需要确保在线数据安全。此外，用户需要简洁、非技术性的解释，并希望有控制信息流动的便利。最后，我们提出了设计建议，并基于数据分析提出了一套综合框架。 

---
# "Accessibility people, you go work on that thing of yours over there": Addressing Disability Inclusion in AI Product Organizations 

**Title (ZH)**: “让残疾人能够使用，你们就去处理那边的那个事情吧”：在AI产品组织中推动残疾人包容性 

**Authors**: Sanika Moharana, Cynthia L. Bennett, Erin Buehler, Michael Madaio, Vinita Tibdewal, Shaun K. Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.16607)  

**Abstract**: The rapid emergence of generative AI has changed the way that technology is designed, constructed, maintained, and evaluated. Decisions made when creating AI-powered systems may impact some users disproportionately, such as people with disabilities. In this paper, we report on an interview study with 25 AI practitioners across multiple roles (engineering, research, UX, and responsible AI) about how their work processes and artifacts may impact end users with disabilities. We found that practitioners experienced friction when triaging problems at the intersection of responsible AI and accessibility practices, navigated contradictions between accessibility and responsible AI guidelines, identified gaps in data about users with disabilities, and gathered support for addressing the needs of disabled stakeholders by leveraging informal volunteer and community groups within their company. Based on these findings, we offer suggestions for new resources and process changes to better support people with disabilities as end users of AI. 

**Abstract (ZH)**: 生成式人工智能的快速兴起改变了技术的设计、构建、维护和评估方式。在创造人工智能驱动系统时做出的决策可能不对等地影响某些用户，例如残疾人。本文报道了对跨多个角色（工程、研究、用户体验和负责任的人工智能）的25名人工智能从业者进行的访谈研究，探讨他们的工作流程和产出如何影响残疾用户。研究发现，从业者在负责任的人工智能和可达性实践交叉问题的优先排序过程中遇到了摩擦，导航了可达性与负责任的人工智能指南之间的矛盾，识别了关于残疾用户的数据缺口，并通过在其公司内利用非正式的志愿者和社区小组来争取支持，以解决残疾相关方的需求。基于这些发现，我们提出了新的资源和流程改进建议，以更好地支持残疾人为人工智能的最终用户。 

---
# Humans Perceive Wrong Narratives from AI Reasoning Texts 

**Title (ZH)**: 人类从AI推理文本中感知到错误的故事线 

**Authors**: Mosh Levy, Zohar Elyoseph, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.16599)  

**Abstract**: A new generation of AI models generates step-by-step reasoning text before producing an answer. This text appears to offer a human-readable window into their computation process, and is increasingly relied upon for transparency and interpretability. However, it is unclear whether human understanding of this text matches the model's actual computational process. In this paper, we investigate a necessary condition for correspondence: the ability of humans to identify which steps in a reasoning text causally influence later steps. We evaluated humans on this ability by composing questions based on counterfactual measurements and found a significant discrepancy: participant accuracy was only 29.3%, barely above chance (25%), and remained low (42%) even when evaluating the majority vote on questions with high agreement. Our results reveal a fundamental gap between how humans interpret reasoning texts and how models use it, challenging its utility as a simple interpretability tool. We argue that reasoning texts should be treated as an artifact to be investigated, not taken at face value, and that understanding the non-human ways these models use language is a critical research direction. 

**Abstract (ZH)**: 一种新型的AI模型在生成答案之前会产生逐步推理文本。这种文本似乎为人们的计算过程提供了一个可读窗口，并越来越多地被依赖以实现透明性和可解释性。然而，尚不清楚人类对这种文本的理解是否与模型的实际计算过程相符。本文探讨了对应关系的一个必要条件：人类识别推理文本中对后续步骤有因果影响的步骤的能力。通过基于假设测量构建问题来评估这一能力，我们发现了显著的差距：参与者准确率仅为29.3%，几乎与随机概率（25%）相当，并且在高一致性的问题中，即使评估多个答案的多数投票，准确率也仅提高至42%。我们的结果揭示了人类解读推理文本与模型使用之间的基本差异，质疑其作为简单解释工具的有效性。我们认为，推理文本应被视为需要研究的产物，而非直接接受，并且理解这些模型非人类的语言使用方式是关键的研究方向。 

---
# Bridging Foundation Models and Efficient Architectures: A Modular Brain Imaging Framework with Local Masking and Pretrained Representation Learning 

**Title (ZH)**: 融合基础模型与高效架构：一种基于局部掩码与预训练表示学习的模块化脑成像框架 

**Authors**: Yanwen Wang, Xinglin Zhao, Yijin Song, Xiaobo Liu, Yanrong Hao, Rui Cao, Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16597)  

**Abstract**: Functional connectivity (FC) derived from resting-state fMRI plays a critical role in personalized predictions such as age and cognitive performance. However, applying foundation models(FM) to fMRI data remains challenging due to its high dimensionality, computational complexity, and the difficulty in capturing complex spatiotemporal dynamics and indirect region-of-interest (ROI) interactions. To address these limitations, we propose a modular neuroimaging framework that integrates principles from FM with efficient, domain-specific architectures. Our approach begins with a Local Masked Autoencoder (LMAE) for pretraining, which reduces the influence of hemodynamic response function (HRF) dynamics and suppresses noise. This is followed by a Random Walk Mixture of Experts (RWMOE) module that clusters features across spatial and temporal dimensions, effectively capturing intricate brain interactions. Finally, a state-space model (SSM)-based predictor performs downstream task inference. Evaluated on the Cambridge Centre for Ageing and Neuroscience (Cam-CAN) dataset, our framework achieved mean absolute errors (MAEs) of 5.343 for age prediction and 2.940 for fluid intelligence, with Pearson correlation coefficients (PCCs) of 0.928 and 0.887, respectively-outperforming existing state-of-the-art methods. Visualization of expert distribution weights further enhances interpretability by identifying key brain regions. This work provides a robust, interpretable alternative to LLM-based approaches for fMRI analysis, offering novel insights into brain aging and cognitive function. 

**Abstract (ZH)**: 功能性连接（FC）从静息态fMRI中提取，在年龄和个人认知性能的个性化预测中发挥关键作用。然而，将基础模型（FM）应用于fMRI数据由于其高维度、计算复杂性和复杂时空动态及间接感兴趣区（ROI）相互作用的捕捉困难而具有挑战性。为解决这些限制，我们提出了一种模块化神经成像框架，将基础模型的原则与高效、领域特定的架构相结合。该方法首先使用局部掩蔽自动编码器（LMAE）进行预训练，以减轻血流动力学反应函数（HRF）动态的影响并抑制噪声，随后使用随机游走混合专家（RWMOE）模块在空间和时间维度上聚类特征，有效地捕捉复杂的脑部相互作用。最后，基于状态空间模型（SSM）的预测器执行下游任务推理。在剑桥老化与神经科学中心（Cam-CAN）数据集上的评估表明，该框架在年龄预测中的平均绝对误差（MAE）为5.343，在流体智力预测中的MAE为2.940，分别对应的皮尔逊相关系数（PCC）为0.928和0.887，均优于现有最先进的方法。专家权重分布图的可视化进一步增强了可解释性，通过识别关键脑区。该项工作提供了一种鲁棒且可解释的替代语言模型（LLM）方法，用于fMRI分析，为脑老化和认知功能提供了新的见解。 

---
# ARL-Based Multi-Action Market Making with Hawkes Processes and Variable Volatility 

**Title (ZH)**: 基于ARL的带有赫克尔斯过程和可变波动性的多行动市场制作 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16589)  

**Abstract**: We advance market-making strategies by integrating Adversarial Reinforcement Learning (ARL), Hawkes Processes, and variable volatility levels while also expanding the action space available to market makers (MMs). To enhance the adaptability and robustness of these strategies -- which can quote always, quote only on one side of the market or not quote at all -- we shift from the commonly used Poisson process to the Hawkes process, which better captures real market dynamics and self-exciting behaviors. We then train and evaluate strategies under volatility levels of 2 and 200. Our findings show that the 4-action MM trained in a low-volatility environment effectively adapts to high-volatility conditions, maintaining stable performance and providing two-sided quotes at least 92\% of the time. This indicates that incorporating flexible quoting mechanisms and realistic market simulations significantly enhances the effectiveness of market-making strategies. 

**Abstract (ZH)**: 通过结合对手 reinforcement 学习（ARL）、霍克尔斯过程和可变波动水平，我们推进了市场制作策略，并扩展了市场制作商的行动空间。为了增强这些策略的适应性和鲁棒性——这些策略可以一直报价、仅在市场一侧报价或根本不报价——我们从常用的泊松过程转向了霍克尔斯过程，这更好地捕捉了实际市场动态和自激发行为。我们还在波动水平为 2 和 200 的环境下训练和评估了这些策略。研究发现，低波动环境下训练的四行动市场制作商能够有效适应高波动条件，保持稳定性能，并至少有 92% 的时间提供双边报价。这表明，引入灵活的报价机制和现实的市场模拟显著提高了市场制作策略的有效性。 

---
# Robust Market Making: To Quote, or not To Quote 

**Title (ZH)**: 稳健的市场制作：报价，还是不报价 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16588)  

**Abstract**: Market making is a popular trading strategy, which aims to generate profit from the spread between the quotes posted at either side of the market. It has been shown that training market makers (MMs) with adversarial reinforcement learning allows to overcome the risks due to changing market conditions and to lead to robust performances. Prior work assumes, however, that MMs keep quoting throughout the trading process, but in practice this is not required, even for ``registered'' MMs (that only need to satisfy quoting ratios defined by the market rules). In this paper, we build on this line of work and enrich the strategy space of the MM by allowing to occasionally not quote or provide single-sided quotes. Towards this end, in addition to the MM agents that provide continuous bid-ask quotes, we have designed two new agents with increasingly richer action spaces. The first has the option to provide bid-ask quotes or refuse to quote. The second has the option to provide bid-ask quotes, refuse to quote, or only provide single-sided ask or bid quotes. We employ a model-driven approach to empirically compare the performance of the continuously quoting MM with the two agents above in various types of adversarial environments. We demonstrate how occasional refusal to provide bid-ask quotes improves returns and/or Sharpe ratios. The quoting ratios of well-trained MMs can basically meet any market requirements, reaching up to 99.9$\%$ in some cases. 

**Abstract (ZH)**: 市场做市是一种流行的交易策略，旨在通过市场两侧提供的报价差来获利。已有研究表明，使用对抗强化学习训练做市商（MMs）可以应对市场变化带来的风险，并实现稳健的业绩。此前的工作假设MMs在整个交易过程中持续报价，但在实践中，即使对于“注册”MMs（只需满足市场规则定义的报价比率），这并不是必需的。在本文中，我们在此基础上扩展了MM的战略空间，允许MM偶尔不报价或提供单边报价。为此，除了提供连续双边报价的MM代理，我们还设计了两个新的代理，具有越来越丰富的行动空间。第一个代理可以选择提供双边报价或拒绝报价。第二个代理可以选择提供双边报价、拒绝报价或仅提供单边报价。我们采用基于模型的方法，比较连续报价的MM与上述两个代理在各种对抗环境中的表现。我们展示了偶尔拒绝提供双边报价如何提高回报率和/或夏普比率。经过充分训练的MM的报价比率足以满足任何市场要求，在某些情况下可达99.9%。 

---
