# Improving AEBS Validation Through Objective Intervention Classification Leveraging the Prediction Divergence Principle 

**Title (ZH)**: 通过利用预测发散原则的目标介入分类改进AEBS验证 

**Authors**: Daniel Betschinske, Steven Peters  

**Link**: [PDF](https://arxiv.org/pdf/2507.07872)  

**Abstract**: The safety validation of automatic emergency braking system (AEBS) requires accurately distinguishing between false positive (FP) and true positive (TP) system activations. While simulations allow straightforward differentiation by comparing scenarios with and without interventions, analyzing activations from open-loop resimulations - such as those from field operational testing (FOT) - is more complex. This complexity arises from scenario parameter uncertainty and the influence of driver interventions in the recorded data. Human labeling is frequently used to address these challenges, relying on subjective assessments of intervention necessity or situational criticality, potentially introducing biases and limitations. This work proposes a rule-based classification approach leveraging the Prediction Divergence Principle (PDP) to address those issues. Applied to a simplified AEBS, the proposed method reveals key strengths, limitations, and system requirements for effective implementation. The findings suggest that combining this approach with human labeling may enhance the transparency and consistency of classification, thereby improving the overall validation process. While the rule set for classification derived in this work adopts a conservative approach, the paper outlines future directions for refinement and broader applicability. Finally, this work highlights the potential of such methods to complement existing practices, paving the way for more reliable and reproducible AEBS validation frameworks. 

**Abstract (ZH)**: 自动紧急制动系统（AEBS）的安全验证需要准确区分假阳性（FP）和真阳性（TP）系统激活。虽然模拟可以通过比较有干预和无干预的情景来进行直接区分，但分析来自开环重放的激活数据（如场操作测试FOT的数据）则更加复杂。这种复杂性源自情景参数的不确定性以及记录数据中驾驶员干预的影响。通常依赖人工标注来应对这些挑战，但这种做法可能会引入主观评估带来的偏见和局限性。本文提出了一种基于规则的分类方法，利用预测发散原则（PDP）来解决这些问题。该方法应用于简化AEBS中，揭示了有效实施的关键优势、局限性和系统要求。研究结果表明，将该方法与人工标注结合使用可能提高分类的透明度和一致性，从而改善整体验证过程。虽然本文分类规则集采取了保守的方法，但文章概述了未来改进和更广泛适用性的方向。最后，本文强调了此类方法有潜力补充现有实践，为更可靠和可再现的AEBS验证框架铺平道路。 

---
# PILOC: A Pheromone Inverse Guidance Mechanism and Local-Communication Framework for Dynamic Target Search of Multi-Agent in Unknown Environments 

**Title (ZH)**: PILOC：一种pheromone逆向引导机制及局部通信框架，用于未知环境中多代理动态目标搜索 

**Authors**: Hengrui Liu, Yi Feng, Qilong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07376)  

**Abstract**: Multi-Agent Search and Rescue (MASAR) plays a vital role in disaster response, exploration, and reconnaissance. However, dynamic and unknown environments pose significant challenges due to target unpredictability and environmental uncertainty. To tackle these issues, we propose PILOC, a framework that operates without global prior knowledge, leveraging local perception and communication. It introduces a pheromone inverse guidance mechanism to enable efficient coordination and dynamic target localization. PILOC promotes decentralized cooperation through local communication, significantly reducing reliance on global channels. Unlike conventional heuristics, the pheromone mechanism is embedded into the observation space of Deep Reinforcement Learning (DRL), supporting indirect agent coordination based on environmental cues. We further integrate this strategy into a DRL-based multi-agent architecture and conduct extensive experiments. Results show that combining local communication with pheromone-based guidance significantly boosts search efficiency, adaptability, and system robustness. Compared to existing methods, PILOC performs better under dynamic and communication-constrained scenarios, offering promising directions for future MASAR applications. 

**Abstract (ZH)**: 多智能体搜索与救援（MASAR）在灾害响应、探索与侦察中发挥着重要作用。然而，动态和未知环境带来的目标不可预测性和环境不确定性提出了重大挑战。为应对这些挑战，我们提出PILOC框架，该框架不依赖全局先验知识，而是利用局部感知和通信。PILOC引入了蚁痕逆向引导机制，以促进高效的协调和动态目标定位。PILOC通过局部通信促进去中心化的合作，显著减少了对全局信道的依赖。与传统启发式方法不同，蚁痕机制嵌入到深度强化学习（DRL）的观测空间中，基于环境线索支持间接智能体协调。我们将此策略集成到基于DRL的多智能体架构中，并进行了广泛的实验。结果表明，结合局部通信与基于蚁痕的引导显著提升了搜索效率、适应性和系统鲁棒性。与其他现有方法相比，PILOC在动态和通信受限场景中表现更优，为未来的MASAR应用提供了有前景的方向。 

---
# g2o vs. Ceres: Optimizing Scan Matching in Cartographer SLAM 

**Title (ZH)**: g2o与Ceres：Cartographer SLAM中扫描匹配的优化比较 

**Authors**: Quanjie Qiu, MengCheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2507.07142)  

**Abstract**: This article presents a comparative analysis of g2o and Ceres solvers in enhancing scan matching performance within the Cartographer framework. Cartographer, a widely-used library for Simultaneous Localization and Mapping (SLAM), relies on optimization algorithms to refine pose estimates and improve map accuracy. The research aims to evaluate the performance, efficiency, and accuracy of the g2o solver in comparison to the Ceres solver, which is the default in Cartographer. In our experiments comparing Ceres and g2o within Cartographer, Ceres outperformed g2o in terms of speed, convergence efficiency, and overall map clarity. Ceres required fewer iterations and less time to converge, producing more accurate and well-defined maps, especially in real-world mapping scenarios with the AgileX LIMO robot. However, g2o excelled in localized obstacle detection, highlighting its value in specific situations. 

**Abstract (ZH)**: 本文对g2o和Ceres求解器在Cartographer框架下提升扫描匹配性能进行比较分析。Cartographer是一个广泛使用的用于同时定位与建图（SLAM）的库，依赖于优化算法来细化姿态估计并提高地图的准确性。研究旨在评估g2o求解器与Ceres求解器（Cartographer的默认求解器）在性能、效率和准确性方面的差异。在我们将Ceres和g2o在Cartographer中进行比较的实验中，Ceres在速度、收敛效率和整体地图清晰度方面优于g2o。Ceres需要 fewer iterations 和更少的时间来收敛，生成更准确且定义更好的地图，尤其是在使用AgileX LIMO机器人进行实地建图时。然而，g2o在局部障碍物检测方面表现出色，突显了其在特定情况下的价值。 

---
# Reinforcement Learning with Action Chunking 

**Title (ZH)**: 行动分块的强化学习 

**Authors**: Qiyang Li, Zhiyuan Zhou, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2507.07969)  

**Abstract**: We present Q-chunking, a simple yet effective recipe for improving reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks. Our recipe is designed for the offline-to-online RL setting, where the goal is to leverage an offline prior dataset to maximize the sample-efficiency of online learning. Effective exploration and sample-efficient learning remain central challenges in this setting, as it is not obvious how the offline data should be utilized to acquire a good exploratory policy. Our key insight is that action chunking, a technique popularized in imitation learning where sequences of future actions are predicted rather than a single action at each timestep, can be applied to temporal difference (TD)-based RL methods to mitigate the exploration challenge. Q-chunking adopts action chunking by directly running RL in a 'chunked' action space, enabling the agent to (1) leverage temporally consistent behaviors from offline data for more effective online exploration and (2) use unbiased $n$-step backups for more stable and efficient TD learning. Our experimental results demonstrate that Q-chunking exhibits strong offline performance and online sample efficiency, outperforming prior best offline-to-online methods on a range of long-horizon, sparse-reward manipulation tasks. 

**Abstract (ZH)**: Q-分块：一种提高长期任务稀疏奖励强化学习算法效果的简单有效方法 

---
# Conjugated Capabilities: Interrelations of Elementary Human Capabilities and Their Implication on Human-Machine Task Allocation and Capability Testing Procedures 

**Title (ZH)**: 共轭能力： elemental人类能力的相互关系及其对人类-机器任务分配和能力测试程序的影响 

**Authors**: Nils Mandischer, Larissa Füller, Torsten Alles, Frank Flemisch, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2507.07560)  

**Abstract**: Human and automation capabilities are the foundation of every human-autonomy interaction and interaction pattern. Therefore, machines need to understand the capacity and performance of human doing, and adapt their own behavior, accordingly. In this work, we address the concept of conjugated capabilities, i.e. capabilities that are dependent or interrelated and between which effort can be distributed. These may be used to overcome human limitations, by shifting effort from a deficient to a conjugated capability with performative resources. For example: A limited arm's reach may be compensated by tilting the torso forward. We analyze the interrelation between elementary capabilities within the IMBA standard to uncover potential conjugation, and show evidence in data of post-rehabilitation patients. From the conjugated capabilities, within the example application of stationary manufacturing, we create a network of interrelations. With this graph, a manifold of potential uses is enabled. We showcase the graph's usage in optimizing IMBA test design to accelerate data recordings, and discuss implications of conjugated capabilities on task allocation between the human and an autonomy. 

**Abstract (ZH)**: 人类与自动化能力是每种人机自主交互及其模式的基础。因此，机器需要理解人类能力及其性能，并根据需要调整自己的行为。在本项工作中，我们探讨了共轭能力的概念，即相互依赖或相关的能力，以及在这些能力之间重新分配努力的可能性。这些能力可以用来克服人类的局限性，通过将努力从一个有缺陷的能力转移到具有表现力资源的共轭能力。例如：有限的臂长可以通过前倾躯干来弥补。我们分析了IMBA标准中基本能力之间的相互关系，以揭示潜在的共轭关系，并在康复后患者的数据中提供证据。从共轭能力出发，在静止制造的应用示例中，我们创建了一个相互关系网络。借助此图，我们能够启用多种潜在用途。我们展示了该图在优化IMBA测试设计以加速数据记录中的应用，并讨论了共轭能力对人类与自主系统之间任务分配的影响。 

---
# Robust signal decompositions on the circle 

**Title (ZH)**: 圆周上的鲁棒信号分解 

**Authors**: Aral Kose, Daniel Liberzon  

**Link**: [PDF](https://arxiv.org/pdf/2507.07007)  

**Abstract**: We consider the problem of decomposing a piecewise constant function on the circle into a sum of indicator functions of closed circular disks in the plane, whose number and location are not a priori known. This represents a situation where an agent moving on the circle is able to sense its proximity to some landmarks, and the goal is to estimate the number of these landmarks and their possible locations -- which can in turn enable control tasks such as motion planning and obstacle avoidance. Moreover, the exact values of the function at its discontinuities (which correspond to disk boundaries for the individual indicator functions) are not assumed to be known to the agent. We introduce suitable notions of robustness and degrees of freedom to single out those decompositions that are more desirable, or more likely, given this non-precise data collected by the agent. We provide a characterization of robust decompositions and give a procedure for generating all such decompositions. When the given function admits a robust decomposition, we compute the number of possible robust decompositions and derive bounds for the number of decompositions maximizing the degrees of freedom. 

**Abstract (ZH)**: 我们考虑将圆上的阶梯函数分解为平面中闭圆盘指示函数之和的问题，其中这些圆盘的数量和位置事先未知。这代表了代理沿圆移动时能够感知其与某些地标接近的情况，目标是估计这些地标数量及其可能位置，这反过来可以实现诸如运动规划和障碍物回避等控制任务。此外，阶梯函数在不连续点的确切值（对应于个别指示函数的圆盘边界）假定未知。我们引入了适当的鲁棒性概念和自由度，以突出那些在获得代理收集的不精确数据情况下更优或更可能的分解。我们给出了鲁棒分解的特征描述，并提供了一种生成所有此类分解的方法。当给定的函数存在鲁棒分解时，我们计算可能的鲁棒分解的数量，并推导出最大化自由度的分解的数量下界。 

---
# Working with AI: Measuring the Occupational Implications of Generative AI 

**Title (ZH)**: 与人工智能共事：生成型人工智能的职业影响测量 

**Authors**: Kiran Tomlinson, Sonia Jaffe, Will Wang, Scott Counts, Siddharth Suri  

**Link**: [PDF](https://arxiv.org/pdf/2507.07935)  

**Abstract**: Given the rapid adoption of generative AI and its potential to impact a wide range of tasks, understanding the effects of AI on the economy is one of society's most important questions. In this work, we take a step toward that goal by analyzing the work activities people do with AI, how successfully and broadly those activities are done, and combine that with data on what occupations do those activities. We analyze a dataset of 200k anonymized and privacy-scrubbed conversations between users and Microsoft Bing Copilot, a publicly available generative AI system. We find the most common work activities people seek AI assistance for involve gathering information and writing, while the most common activities that AI itself is performing are providing information and assistance, writing, teaching, and advising. Combining these activity classifications with measurements of task success and scope of impact, we compute an AI applicability score for each occupation. We find the highest AI applicability scores for knowledge work occupation groups such as computer and mathematical, and office and administrative support, as well as occupations such as sales whose work activities involve providing and communicating information. Additionally, we characterize the types of work activities performed most successfully, how wage and education correlate with AI applicability, and how real-world usage compares to predictions of occupational AI impact. 

**Abstract (ZH)**: 给定生成式AI的快速采纳及其对广泛任务的潜在影响，理解AI对经济的影响是社会最重要的问题之一。在这项工作中，我们朝着这个目标迈进了一步，通过分析人们使用AI的工作活动，这些活动的成功度和广度，以及结合职业从事这些活动的数据。我们分析了一个包含20万条匿名化和隐私脱敏的用户与Microsoft Bing Copilot（一个公开可用的生成式AI系统）对话的数据集。我们发现，人们寻求AI帮助最多的日常工作活动主要涉及信息收集和写作，而AI自己执行的最常见活动包括提供信息和帮助、写作、教学和建议。结合这些活动分类与任务成功度和影响范围的测量，我们为每个职业计算了一个AI适用性评分。我们发现，知识工作职业群体，如计算机与数学、办公室和支持性行政工作，以及从事提供和沟通信息工作的职业，其AI适用性评分最高。此外，我们还描述了最成功的日常工作活动类型，工资和教育与AI适用性的关系，以及实际使用情况与对职业AI影响的预测之间的差异。 

---
# An Integrated Framework of Prompt Engineering and Multidimensional Knowledge Graphs for Legal Dispute Analysis 

**Title (ZH)**: 面向法律纠纷分析的提示工程与多维知识图谱集成框架 

**Authors**: Mingda Zhang, Na Zhao, Jianglong Qing, Qing xu, Kaiwen Pan, Ting luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.07893)  

**Abstract**: The rapid development of artificial intelligence has positioned large language models as fundamental components of intelligent legal systems. However, these models face significant limitations in legal dispute analysis, including insufficient legal knowledge representation, limited concept understanding, and reasoning deficiencies. This research proposes an enhanced framework integrating prompt engineering with multidimensional knowledge graphs. The framework introduces a three-stage hierarchical prompt structure comprising task definition, knowledge background, and reasoning guidance, supplemented by legal-specific reasoning templates and dynamic optimization mechanisms. A three-layer knowledge graph architecture is constructed with legal classification ontology, representation, and instance layers. Four complementary methods enable precise legal concept retrieval: direct legal norm code matching, domain-specific semantic vector similarity, ontology-based path reasoning, and specialized lexical segmentation. These components integrate with web search technology to establish a knowledge-enhanced framework for legal decision-making. Experimental results demonstrate significant performance improvements in legal dispute analysis, enabling accurate legal application analysis for complex cases while exhibiting nuanced understanding of judicial decision-making logic, providing a novel technical approach for implementing intelligent legal assistance systems. 

**Abstract (ZH)**: 人工智能的迅速发展已将大型语言模型定位为智能法律系统中的基本组件。然而，这些模型在法律纠纷分析中面临诸多局限，包括法律知识表示不足、概念理解有限以及推理缺陷。本研究提出了一种增强框架，结合了提示工程与多维度知识图谱。该框架引入了具有任务定义、知识背景和推理指导三个层级的提示结构，同时包含了专门的法律推理模板和动态优化机制。构建了三层知识图谱架构，包括法律分类本体层、表示层和实例层。四种互补的方法实现了精确的法律概念检索：直接的法律规范代码匹配、领域特定的语义向量相似度、本体路径推理以及专门的词法分段。这些组件与网络搜索技术相结合，建立了一个知识增强的法律决策框架。实验结果表明，在法律纠纷分析方面的性能显著提升，能够对复杂案件进行精确的法律应用分析，并表现出对司法决策逻辑的细腻理解，为实施智能法律辅助系统提供了新的技术途径。 

---
# Searching for actual causes: Approximate algorithms with adjustable precision 

**Title (ZH)**: 寻找实际原因：可调节精度的近似算法 

**Authors**: Samuel Reyd, Ada Diaconescu, Jean-Louis Dessalles  

**Link**: [PDF](https://arxiv.org/pdf/2507.07857)  

**Abstract**: Causality has gained popularity in recent years. It has helped improve the performance, reliability, and interpretability of machine learning models. However, recent literature on explainable artificial intelligence (XAI) has faced criticism. The classical XAI and causality literature focuses on understanding which factors contribute to which consequences. While such knowledge is valuable for researchers and engineers, it is not what non-expert users expect as explanations. Instead, these users often await facts that cause the target consequences, i.e., actual causes. Formalizing this notion is still an open problem. Additionally, identifying actual causes is reportedly an NP-complete problem, and there are too few practical solutions to approximate formal definitions. We propose a set of algorithms to identify actual causes with a polynomial complexity and an adjustable level of precision and exhaustiveness. Our experiments indicate that the algorithms (1) identify causes for different categories of systems that are not handled by existing approaches (i.e., non-boolean, black-box, and stochastic systems), (2) can be adjusted to gain more precision and exhaustiveness with more computation time. 

**Abstract (ZH)**: 因果关系在近年来受到了广泛关注。它有助于提高机器学习模型的性能、可靠性和可解释性。然而，解释型人工智能（XAI）的近期文献受到了批评。经典XAI和因果关系文献侧重于理解哪些因素导致了哪些后果。虽然此类知识对于研究人员和工程师非常重要，但非专家用户期望的解释并非如此。相反，这些用户通常期待导致目标后果的事实，即实际原因。正式化这一概念仍然是一个开放问题。此外，识别实际原因据说是NP完全问题，目前可供实用的近似正式定义的解决方案寥寥无几。我们提出了一组算法，可以在多项式时间内识别实际原因，并调整精度和详尽程度。实验表明，这些算法（1）可以识别现有方法无法处理的不同类别系统（即非布尔型、黑盒和随机系统）的原因，（2）可以通过增加计算时间来提高精度和详尽程度。 

---
# Measuring AI Alignment with Human Flourishing 

**Title (ZH)**: 测量AI对人类 flourishing 的 Alignment 

**Authors**: Elizabeth Hilliard, Akshaya Jagadeesh, Alex Cook, Steele Billings, Nicholas Skytland, Alicia Llewellyn, Jackson Paull, Nathan Paull, Nolan Kurylo, Keatra Nesbitt, Robert Gruenewald, Anthony Jantzi, Omar Chavez  

**Link**: [PDF](https://arxiv.org/pdf/2507.07787)  

**Abstract**: This paper introduces the Flourishing AI Benchmark (FAI Benchmark), a novel evaluation framework that assesses AI alignment with human flourishing across seven dimensions: Character and Virtue, Close Social Relationships, Happiness and Life Satisfaction, Meaning and Purpose, Mental and Physical Health, Financial and Material Stability, and Faith and Spirituality. Unlike traditional benchmarks that focus on technical capabilities or harm prevention, the FAI Benchmark measures AI performance on how effectively models contribute to the flourishing of a person across these dimensions. The benchmark evaluates how effectively LLM AI systems align with current research models of holistic human well-being through a comprehensive methodology that incorporates 1,229 objective and subjective questions. Using specialized judge Large Language Models (LLMs) and cross-dimensional evaluation, the FAI Benchmark employs geometric mean scoring to ensure balanced performance across all flourishing dimensions. Initial testing of 28 leading language models reveals that while some models approach holistic alignment (with the highest-scoring models achieving 72/100), none are acceptably aligned across all dimensions, particularly in Faith and Spirituality, Character and Virtue, and Meaning and Purpose. This research establishes a framework for developing AI systems that actively support human flourishing rather than merely avoiding harm, offering significant implications for AI development, ethics, and evaluation. 

**Abstract (ZH)**: Flourishing AI基准（FAI基准）：一种评估AI与人类繁荣一致性的新框架 

---
# Identification of Violin Reduction via Contour Lines Classification 

**Title (ZH)**: 小提琴 reducium 识别 via 岸线分类 

**Authors**: Philémon Beghin, Anne-Emmanuelle Ceulemans, François Glineur  

**Link**: [PDF](https://arxiv.org/pdf/2507.07743)  

**Abstract**: The first violins appeared in late 16th-century Italy. Over the next 200 years, they spread across Europe and luthiers of various royal courts, eager to experiment with new techniques, created a highly diverse family of instruments. Around 1750, size standards were introduced to unify violin making for orchestras and conservatories. Instruments that fell between two standards were then reduced to a smaller size by luthiers. These reductions have an impact on several characteristics of violins, in particular on the contour lines, i.e. lines of constant altitude, which look more like a U for non reduced instruments and a V for reduced ones. While such differences are observed by experts, they have not been studied quantitatively.
This paper presents a method for classifying violins as reduced or non-reduced based on their contour lines. We study a corpus of 25 instruments whose 3D geometric meshes were acquired via photogrammetry. For each instrument, we extract 10-20 contour lines regularly spaced every millimetre. Each line is fitted with a parabola-like curve (with an equation of the type y = alpha*abs(x)**beta) depending on two parameters, describing how open (beta) and how vertically stretched (alpha) the curve is. We compute additional features from those parameters, using regressions and counting how many values fall under some threshold. We also deal with outliers and non equal numbers of levels, and eventually obtain a numerical profile for each instrument.
We then apply classification methods to assess whether geometry alone can predict size reduction. We find that distinguishing between reduced and non reduced instruments is feasible to some degree, taking into account that a whole spectrum of more or less transformed violins exists, for which it is more difficult to quantify the reduction. We also find the opening parameter beta to be the most predictive. 

**Abstract (ZH)**: 16世纪晚期意大利首次出现小提琴。在接下来的200年中，小提琴传播至欧洲各地，各个王室宫廷的制琴师渴望尝试新的技术，由此创造出一系列多样化的乐器。约在1750年，尺寸标准被引入以统一管弦乐团和音乐学院中的小提琴制作。尺寸介于两个标准之间的乐器随后被制琴师缩减至更小的尺寸。这种缩减对小提琴的多个特征产生了影响，特别是对等高线，即常高度线，非缩减乐器的等高线呈现U形而缩减乐器的则呈现V形。虽然专家们观察到了这些差异，但尚未对其进行定量研究。

本文提出了一种方法，根据等高线对小提琴进行分类，将其分为缩减和非缩减两类。我们研究了一组25件乐器，使用摄影测量法获取其3D几何网格。对于每件乐器，我们提取了每隔毫米均匀分布的10-20条等高线。每条线都用类似抛物线的曲线拟合（方程形式为y = alpha*abs(x)**beta），该曲线由两个参数描述，这些参数分别表示曲线的开放程度（beta）和竖直拉伸程度（alpha）。我们从这些参数中计算出额外的特征，利用回归分析并统计有多少值低于某个阈值。我们还处理了离群值和不同数量层级的问题，最终为每件乐器获得了一个数字特征轮廓。

随后，我们应用分类方法来评估几何形状是否足以预测尺寸缩减。我们发现，在考虑存在不同程度变形的小提琴存在整个连续体的情况下，区分缩减和非缩减小提琴在一定程度上是可行的。我们还发现开放参数beta是最重要的预测因素。 

---
# PlanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations 

**Title (ZH)**: PlanQA: 一个基于结构化表示的LLMs空间推理基准测试 

**Authors**: Fedor Rodionov, Abdelrahman Eldesokey, Michael Birsak, John Femiani, Bernard Ghanem, Peter Wonka  

**Link**: [PDF](https://arxiv.org/pdf/2507.07644)  

**Abstract**: We introduce PlanQA, a diagnostic benchmark for evaluating geometric and spatial reasoning in large-language models (LLMs). PlanQA is grounded in structured representations of indoor scenes, such as kitchens, living rooms, and bedrooms, encoded in a symbolic format (e.g., JSON, XML layouts). The benchmark includes diverse question types that test not only metric and topological reasoning (e.g., distance, visibility, shortest paths) but also interior design constraints such as affordance, clearance, balance, and usability. Our results across a variety of frontier open-source and commercial LLMs show that while models may succeed in shallow queries, they often fail to simulate physical constraints, preserve spatial coherence, or generalize under layout perturbation. PlanQA uncovers a clear blind spot in today's LLMs: they do not consistently reason about real-world layouts. We hope that this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings. 

**Abstract (ZH)**: PlanQA：一个评估大型语言模型几何与空间推理能力的诊断基准 

---
# Towards conservative inference in credal networks using belief functions: the case of credal chains 

**Title (ZH)**: 基于信念函数在信度网络中实现保守推理：信度链的情形 

**Authors**: Marco Sangalli, Thomas Krak, Cassio de Campos  

**Link**: [PDF](https://arxiv.org/pdf/2507.07619)  

**Abstract**: This paper explores belief inference in credal networks using Dempster-Shafer theory. By building on previous work, we propose a novel framework for propagating uncertainty through a subclass of credal networks, namely chains. The proposed approach efficiently yields conservative intervals through belief and plausibility functions, combining computational speed with robust uncertainty representation. Key contributions include formalizing belief-based inference methods and comparing belief-based inference against classical sensitivity analysis. Numerical results highlight the advantages and limitations of applying belief inference within this framework, providing insights into its practical utility for chains and for credal networks in general. 

**Abstract (ZH)**: 本文利用Dempster-Shafer理论探讨信度网络中的信念推断，在前人研究的基础上，我们提出了一种新颖的框架，用于传播一类信度网络（即链）中的不确定性。该方法通过信念和可信任度函数高效地得出保守区间，结合了计算速度和稳健的不确定性表示。主要贡献包括正式化基于信念的推断方法，并将基于信念的推断与经典敏感性分析进行比较。数值结果突出了在此框架中应用信念推断的优点和局限性，为链和一般信度网络的实际应用提供了见解。 

---
# Context Pooling: Query-specific Graph Pooling for Generic Inductive Link Prediction in Knowledge Graphs 

**Title (ZH)**: 上下文聚池化：针对通用归纳链接预测的图聚池化方法 

**Authors**: Zhixiang Su, Di Wang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2507.07595)  

**Abstract**: Recent investigations on the effectiveness of Graph Neural Network (GNN)-based models for link prediction in Knowledge Graphs (KGs) show that vanilla aggregation does not significantly impact the model performance. In this paper, we introduce a novel method, named Context Pooling, to enhance GNN-based models' efficacy for link predictions in KGs. To our best of knowledge, Context Pooling is the first methodology that applies graph pooling in KGs. Additionally, Context Pooling is first-of-its-kind to enable the generation of query-specific graphs for inductive settings, where testing entities are unseen during training. Specifically, we devise two metrics, namely neighborhood precision and neighborhood recall, to assess the neighbors' logical relevance regarding the given queries, thereby enabling the subsequent comprehensive identification of only the logically relevant neighbors for link prediction. Our method is generic and assessed by being applied to two state-of-the-art (SOTA) models on three public transductive and inductive datasets, achieving SOTA performance in 42 out of 48 settings. 

**Abstract (ZH)**: 基于图神经网络的链接预测在知识图谱中的最新研究显示， vanilla聚合对模型性能影响不大。本文提出了一种新颖的方法——上下文池化，以增强基于图神经网络的模型在知识图谱中的链接预测效果。到我们所知，上下文池化是首次在知识图谱中应用图池化的办法。此外，上下文池化是首个在归纳设置中生成查询特定图的方法，使得在训练中未见过的测试实体能够进行预测。具体地，我们设计了两个指标，即邻域精确度和邻域召回率，以评估邻域在给定查询下的逻辑相关性，从而实现仅识别对于链接预测逻辑相关的邻域的综合判定。该方法具有通用性，并通过在三个公开的归纳和传递数据集上应用两种最先进的模型进行评估，在48种设置中42种达到了最先进的性能。 

---
# On Trustworthy Rule-Based Models and Explanations 

**Title (ZH)**: 基于规则的模型及其解释的可靠性研究 

**Authors**: Mohamed Siala, Jordi Planes, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.07576)  

**Abstract**: A task of interest in machine learning (ML) is that of ascribing explanations to the predictions made by ML models. Furthermore, in domains deemed high risk, the rigor of explanations is paramount. Indeed, incorrect explanations can and will mislead human decision makers. As a result, and even if interpretability is acknowledged as an elusive concept, so-called interpretable models are employed ubiquitously in high-risk uses of ML and data mining (DM). This is the case for rule-based ML models, which encompass decision trees, diagrams, sets and lists. This paper relates explanations with well-known undesired facets of rule-based ML models, which include negative overlap and several forms of redundancy. The paper develops algorithms for the analysis of these undesired facets of rule-based systems, and concludes that well-known and widely used tools for learning rule-based ML models will induce rule sets that exhibit one or more negative facets. 

**Abstract (ZH)**: 机器学习领域的一个研究任务是为机器学习模型的预测提供解释。在高风险领域中，这些解释的严谨性至关重要。事实上，不正确的解释可能会误导人类决策者。因此，即便可解释性被认为是难以捉摸的概念，所谓可解释的模型在高风险的机器学习和数据挖掘应用中依然被广泛应用。这类模型包括基于规则的模型，如决策树、图表、集合和列表。本文将解释与基于规则的模型广为人知的负面特征相关联，这些负面特征包括负面重叠和多种形式的冗余。本文开发了分析这些负面特征的算法，并得出结论，广泛使用的学习基于规则的机器学习模型的工具将导致表现出一种或多种负面特征的规则集。 

---
# DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search 

**Title (ZH)**: DrugMCTS：一种结合多智能体、RAG和蒙特卡洛树搜索的药物再利用框架 

**Authors**: Zerui Yang, Yuwei Wan, Yinqiao Li, Yudai Matsuda, Tong Xie, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.07426)  

**Abstract**: Recent advances in large language models have demonstrated considerable potential in scientific domains such as drug discovery. However, their effectiveness remains constrained when reasoning extends beyond the knowledge acquired during pretraining. Conventional approaches, such as fine-tuning or retrieval-augmented generation, face limitations in either imposing high computational overhead or failing to fully exploit structured scientific data. To overcome these challenges, we propose DrugMCTS, a novel framework that synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree Search for drug repurposing. The framework employs five specialized agents tasked with retrieving and analyzing molecular and protein information, thereby enabling structured and iterative reasoning. Without requiring domain-specific fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate that DrugMCTS achieves substantially higher recall and robustness compared to both general-purpose LLMs and deep learning baselines. Our results highlight the importance of structured reasoning, agent-based collaboration, and feedback-driven search mechanisms in advancing LLM applications for drug discovery. 

**Abstract (ZH)**: 最近大型语言模型在药物发现等科学领域的进展展示了巨大潜力，但其有效性在超出预训练知识的推理时仍受到限制。传统方法如微调或检索增强生成面临高计算开销或未能充分利用结构化科学数据的局限。为克服这些挑战，我们提出DrugMCTS，这是一种结合了检索增强生成(RAG)、多代理协作和蒙特卡罗树搜索的新框架，用于药物重新利用。该框架通过五个专门代理执行分子和蛋白质信息的检索与分析，从而实现结构化和迭代推理。无需特定领域的微调，DrugMCTS使Qwen2.5-7B-Instruct相比Deepseek-R1性能提高了超过20%。在DrugBank和KIBA数据集上的广泛实验表明，DrugMCTS在召回率和鲁棒性方面显著优于通用语言模型和深度学习基线。我们的结果强调了结构化推理、基于代理的协作和反馈驱动的搜索机制在推进药物发现领域的大语言模型应用中的重要性。 

---
# Supply Chain Optimization via Generative Simulation and Iterative Decision Policies 

**Title (ZH)**: 通过生成式仿真与迭代决策策略的供应链优化 

**Authors**: Haoyue Bai, Haoyu Wang, Nanxu Gong, Xinyuan Wang, Wangyang Ying, Haifeng Chen, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07355)  

**Abstract**: High responsiveness and economic efficiency are critical objectives in supply chain transportation, both of which are influenced by strategic decisions on shipping mode. An integrated framework combining an efficient simulator with an intelligent decision-making algorithm can provide an observable, low-risk environment for transportation strategy design. An ideal simulation-decision framework must (1) generalize effectively across various settings, (2) reflect fine-grained transportation dynamics, (3) integrate historical experience with predictive insights, and (4) maintain tight integration between simulation feedback and policy refinement. We propose Sim-to-Dec framework to satisfy these requirements. Specifically, Sim-to-Dec consists of a generative simulation module, which leverages autoregressive modeling to simulate continuous state changes, reducing dependence on handcrafted domain-specific rules and enhancing robustness against data fluctuations; and a history-future dual-aware decision model, refined iteratively through end-to-end optimization with simulator interactions. Extensive experiments conducted on three real-world datasets demonstrate that Sim-to-Dec significantly improves timely delivery rates and profit. 

**Abstract (ZH)**: 高响应性和经济效率是供应链运输中的关键目标，两者都受运输模式战略决策的影响。一个综合框架结合高效的模拟器与智能决策算法，可以为运输策略设计提供一个可观测且风险较低的环境。理想的仿真-决策框架必须（1）在各种环境中有效泛化，（2）反映精细的运输动态，（3）整合历史经验与预测洞察，（4）保持仿真反馈与政策改进的高度集成。我们提出了Sim-to-Dec框架以满足这些要求。具体而言，Sim-to-Dec 包括一个生成性模拟模块，该模块利用自回归建模来模拟连续状态的变化，减少对手工定制的领域特定规则的依赖，增强对数据波动的鲁棒性；以及一个知过往晓未来的双重意识决策模型，该模型通过端到端优化并与模拟器交互进行迭代 refinement。在三个真实世界数据集上的广泛实验表明，Sim-to-Dec 显著提高了及时交货率和利润。 

---
# State-Inference-Based Prompting for Natural Language Trading with Game NPCs 

**Title (ZH)**: 基于状态推理的提示方法在与游戏NPC进行自然语言交易中的应用 

**Authors**: Minkyung Kim, Junsik Kim, Hwidong Bae, Woongcheol Yang, Sangdon Park, Sohee Bae  

**Link**: [PDF](https://arxiv.org/pdf/2507.07203)  

**Abstract**: Large Language Models enable dynamic game interactions but struggle with rule-governed trading systems. Current implementations suffer from rule violations, such as item hallucinations and calculation errors, that erode player trust. Here, State-Inference-Based Prompting (SIBP) enables reliable trading through autonomous dialogue state inference and context-specific rule adherence. The approach decomposes trading into six states within a unified prompt framework, implementing context-aware item referencing and placeholder-based price calculations. Evaluation across 100 trading dialogues demonstrates >97% state compliance, >95% referencing accuracy, and 99.7% calculation precision. SIBP maintains computational efficiency while outperforming baseline approaches, establishing a practical foundation for trustworthy NPC interactions in commercial games. 

**Abstract (ZH)**: 基于状态推断的提示方法（SIBP）实现可靠的交易对话状态推理和情境特定规则遵从，促进动态游戏交互但挑战规则 govern 的交易系统。 

---
# BOOST: Out-of-Distribution-Informed Adaptive Sampling for Bias Mitigation in Stylistic Convolutional Neural Networks 

**Title (ZH)**: BOOST：基于分布外信息的自适应采样以减轻风格卷积神经网络中的偏差 

**Authors**: Mridula Vijendran, Shuang Chen, Jingjing Deng, Hubert P. H. Shum  

**Link**: [PDF](https://arxiv.org/pdf/2507.07134)  

**Abstract**: The pervasive issue of bias in AI presents a significant challenge to painting classification, and is getting more serious as these systems become increasingly integrated into tasks like art curation and restoration. Biases, often arising from imbalanced datasets where certain artistic styles dominate, compromise the fairness and accuracy of model predictions, i.e., classifiers are less accurate on rarely seen paintings. While prior research has made strides in improving classification performance, it has largely overlooked the critical need to address these underlying biases, that is, when dealing with out-of-distribution (OOD) data. Our insight highlights the necessity of a more robust approach to bias mitigation in AI models for art classification on biased training data. We propose a novel OOD-informed model bias adaptive sampling method called BOOST (Bias-Oriented OOD Sampling and Tuning). It addresses these challenges by dynamically adjusting temperature scaling and sampling probabilities, thereby promoting a more equitable representation of all classes. We evaluate our proposed approach to the KaoKore and PACS datasets, focusing on the model's ability to reduce class-wise bias. We further propose a new metric, Same-Dataset OOD Detection Score (SODC), designed to assess class-wise separation and per-class bias reduction. Our method demonstrates the ability to balance high performance with fairness, making it a robust solution for unbiasing AI models in the art domain. 

**Abstract (ZH)**: 面向艺术分类的偏差数据泛化适应采样方法：BOOST（面向偏差的OOD采样与调优） 

---
# Multigranular Evaluation for Brain Visual Decoding 

**Title (ZH)**: 多级评价脑视觉解码 

**Authors**: Weihao Xia, Cengiz Oztireli  

**Link**: [PDF](https://arxiv.org/pdf/2507.07993)  

**Abstract**: Existing evaluation protocols for brain visual decoding predominantly rely on coarse metrics that obscure inter-model differences, lack neuroscientific foundation, and fail to capture fine-grained visual distinctions. To address these limitations, we introduce BASIC, a unified, multigranular evaluation framework that jointly quantifies structural fidelity, inferential alignment, and contextual coherence between decoded and ground truth images. For the structural level, we introduce a hierarchical suite of segmentation-based metrics, including foreground, semantic, instance, and component masks, anchored in granularity-aware correspondence across mask structures. For the semantic level, we extract structured scene representations encompassing objects, attributes, and relationships using multimodal large language models, enabling detailed, scalable, and context-rich comparisons with ground-truth stimuli. We benchmark a diverse set of visual decoding methods across multiple stimulus-neuroimaging datasets within this unified evaluation framework. Together, these criteria provide a more discriminative, interpretable, and comprehensive foundation for measuring brain visual decoding methods. 

**Abstract (ZH)**: 现有的脑视觉解码评估协议主要依赖于模糊的指标，这些指标掩盖了模型之间的差异，缺乏神经科学基础，并且无法捕捉细微的视觉区分。为了解决这些局限性，我们引入了BASIC，一个统一的多粒度评估框架，该框架同时量化解码图像与 ground truth 图像在结构保真度、推理对齐和上下文一致性方面的差异。在结构层面，我们引入了一套基于分层分割的度量标准，包括前景、语义、实例和组件掩码，并且这些度量标准基于掩码结构的粒度感知对应关系。在语义层面，我们利用多模态大语言模型提取包含对象、属性和关系的结构化场景表示，从而能够在保留上下文的情况下与真实刺激进行详细、可扩展的比较。在这个统一的评估框架中，我们针对多个刺激-神经成像数据集评估了一系列视觉解码方法。这些标准共同为测量脑视觉解码方法提供了更具区分性、可解释性和全面的基础。 

---
# EXPO: Stable Reinforcement Learning with Expressive Policies 

**Title (ZH)**: EXPO: 稳定的强化学习与表达性策略 

**Authors**: Perry Dong, Qiyang Li, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2507.07986)  

**Abstract**: We study the problem of training and fine-tuning expressive policies with online reinforcement learning (RL) given an offline dataset. Training expressive policy classes with online RL present a unique challenge of stable value maximization. Unlike simpler Gaussian policies commonly used in online RL, expressive policies like diffusion and flow-matching policies are parameterized by a long denoising chain, which hinders stable gradient propagation from actions to policy parameters when optimizing against some value function. Our key insight is that we can address stable value maximization by avoiding direct optimization over value with the expressive policy and instead construct an on-the-fly RL policy to maximize Q-value. We propose Expressive Policy Optimization (EXPO), a sample-efficient online RL algorithm that utilizes an on-the-fly policy to maximize value with two parameterized policies -- a larger expressive base policy trained with a stable imitation learning objective and a light-weight Gaussian edit policy that edits the actions sampled from the base policy toward a higher value distribution. The on-the-fly policy optimizes the actions from the base policy with the learned edit policy and chooses the value maximizing action from the base and edited actions for both sampling and temporal-difference (TD) backup. Our approach yields up to 2-3x improvement in sample efficiency on average over prior methods both in the setting of fine-tuning a pretrained policy given offline data and in leveraging offline data to train online. 

**Abstract (ZH)**: 基于离线数据训练和微调表达性策略的在线强化学习研究 

---
# Low Resource Reconstruction Attacks Through Benign Prompts 

**Title (ZH)**: 低资源重建攻击通过良性提示 

**Authors**: Sol Yarkoni, Roi Livni  

**Link**: [PDF](https://arxiv.org/pdf/2507.07947)  

**Abstract**: The recent advances in generative models such as diffusion models have raised several risks and concerns related to privacy, copyright infringements and data stewardship. To better understand and control the risks, various researchers have created techniques, experiments and attacks that reconstruct images, or part of images, from the training set. While these techniques already establish that data from the training set can be reconstructed, they often rely on high-resources, excess to the training set as well as well-engineered and designed prompts.
In this work, we devise a new attack that requires low resources, assumes little to no access to the actual training set, and identifies, seemingly, benign prompts that lead to potentially-risky image reconstruction. This highlights the risk that images might even be reconstructed by an uninformed user and unintentionally. For example, we identified that, with regard to one existing model, the prompt ``blue Unisex T-Shirt'' can generate the face of a real-life human model. Our method builds on an intuition from previous works which leverages domain knowledge and identifies a fundamental vulnerability that stems from the use of scraped data from e-commerce platforms, where templated layouts and images are tied to pattern-like prompts. 

**Abstract (ZH)**: Recent 进展 在 生成 模型 如 放散 模型 中 引起 的 隐私 风险 、 版权 侵权 和 数据 管理 问题 及 其 应对 研究 

---
# Probing Experts' Perspectives on AI-Assisted Public Speaking Training 

**Title (ZH)**: 探究专家对AI辅助公共演讲训练的看法 

**Authors**: Nesrine Fourati, Alisa Barkar, Marion Dragée, Liv Danthon-Lefebvre, Mathieu Chollet  

**Link**: [PDF](https://arxiv.org/pdf/2507.07930)  

**Abstract**: Background: Public speaking is a vital professional skill, yet it remains a source of significant anxiety for many individuals. Traditional training relies heavily on expert coaching, but recent advances in AI has led to novel types of commercial automated public speaking feedback tools. However, most research has focused on prototypes rather than commercial applications, and little is known about how public speaking experts perceive these tools.
Objectives: This study aims to evaluate expert opinions on the efficacy and design of commercial AI-based public speaking training tools and to propose guidelines for their improvement.
Methods: The research involved 16 semi-structured interviews and 2 focus groups with public speaking experts. Participants discussed their views on current commercial tools, their potential integration into traditional coaching, and suggestions for enhancing these systems.
Results and Conclusions: Experts acknowledged the value of AI tools in handling repetitive, technical aspects of training, allowing coaches to focus on higher-level skills. However they found key issues in current tools, emphasising the need for personalised, understandable, carefully selected feedback and clear instructional design. Overall, they supported a hybrid model combining traditional coaching with AI-supported exercises. 

**Abstract (ZH)**: 背景：公共演讲是重要的专业技能，但对许多人来说仍是显著的焦虑源。传统培训依赖专家指导，而最近的AI进步导致了新的商业自动公共演讲反馈工具的出现。然而，大多数研究集中在原型上而不是商业应用，关于公共演讲专家对这些工具的看法知之甚少。
目标：本研究旨在评估公共演讲专家对商业AI基于的公共演讲培训工具的有效性和设计的看法，并提出改进指南。
方法：研究包括16次半结构化采访和2次专家焦点小组讨论，参与者讨论了他们对当前商业工具的看法、这些工具与传统指导的潜在整合以及对这些系统的改进建议。
结果与结论：专家承认AI工具在处理训练中的重复和技术方面具有价值，使教练可以专注于高层次技能。但他们也指出现有工具的关键问题，强调个性化、易于理解、精心挑选的反馈和清晰的指导设计的重要性。总体而言，他们支持将传统指导与AI支持的练习相结合的混合模式。 

---
# Towards Continuous Home Cage Monitoring: An Evaluation of Tracking and Identification Strategies for Laboratory Mice 

**Title (ZH)**: 面向连续家庭笼监测：实验室小鼠跟踪与识别策略评估 

**Authors**: Juan Pablo Oberhauser, Daniel Grzenda  

**Link**: [PDF](https://arxiv.org/pdf/2507.07929)  

**Abstract**: Continuous, automated monitoring of laboratory mice enables more accurate data collection and improves animal welfare through real-time insights. Researchers can achieve a more dynamic and clinically relevant characterization of disease progression and therapeutic effects by integrating behavioral and physiological monitoring in the home cage. However, providing individual mouse metrics is difficult because of their housing density, similar appearances, high mobility, and frequent interactions. To address these challenges, we develop a real-time identification (ID) algorithm that accurately assigns ID predictions to mice wearing custom ear tags in digital home cages monitored by cameras. Our pipeline consists of three parts: (1) a custom multiple object tracker (MouseTracks) that combines appearance and motion cues from mice; (2) a transformer-based ID classifier (Mouseformer); and (3) a tracklet associator linear program to assign final ID predictions to tracklets (MouseMap). Our models assign an animal ID based on custom ear tags at 30 frames per second with 24/7 cage coverage. We show that our custom tracking and ID pipeline improves tracking efficiency and lowers ID switches across mouse strains and various environmental factors compared to current mouse tracking methods. 

**Abstract (ZH)**: 连续自动监测实验室小鼠能够通过实时反馈提高数据准确性和动物福利。通过将行为和生理监测集成到家庭笼中，研究人员可以实现更动态和临床相关的小鼠疾病进展和治疗效果表征。然而，由于小鼠的高密度饲养、相似外观、高移动性和频繁的互动，提供个体小鼠指标具有挑战性。为应对这些挑战，我们开发了一种实时识别算法，能够在监控小鼠的数字家庭笼中准确地将识别预测分配给佩戴了定制耳标的个体小鼠。我们的管道由三部分组成：(1) 一种结合小鼠外观和运动线索的定制多目标跟踪器(MouseTracks)；(2) 基于变换器的识别分类器(Mouseformer)；(3) 跟踪片段关联线性规划(MouseMap)以最终对跟踪片段分配ID预测。我们的模型以每秒30帧的速度，在24小时全天候笼子覆盖下，基于定制耳标分配动物ID。我们证明，与现有的小鼠跟踪方法相比，我们的定制跟踪和识别管道能够提高跟踪效率，并在不同小鼠品系和各种环境因素下降低识别切换率。 

---
# DTECT: Dynamic Topic Explorer & Context Tracker 

**Title (ZH)**: DTECT: 动态主题探索与上下文追踪 

**Authors**: Suman Adhya, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.07910)  

**Abstract**: The explosive growth of textual data over time presents a significant challenge in uncovering evolving themes and trends. Existing dynamic topic modeling techniques, while powerful, often exist in fragmented pipelines that lack robust support for interpretation and user-friendly exploration. We introduce DTECT (Dynamic Topic Explorer & Context Tracker), an end-to-end system that bridges the gap between raw textual data and meaningful temporal insights. DTECT provides a unified workflow that supports data preprocessing, multiple model architectures, and dedicated evaluation metrics to analyze the topic quality of temporal topic models. It significantly enhances interpretability by introducing LLM-driven automatic topic labeling, trend analysis via temporally salient words, interactive visualizations with document-level summarization, and a natural language chat interface for intuitive data querying. By integrating these features into a single, cohesive platform, DTECT empowers users to more effectively track and understand thematic dynamics. DTECT is open-source and available at this https URL. 

**Abstract (ZH)**: 文本数据随时间的爆炸性增长为发现 evolving themes 和趋势带来了重大挑战。现有的动态主题建模技术虽然强大，但往往存在于缺乏解释性和用户友好探索支持的分散管道中。我们提出了 DTECT（动态主题探索与上下文追踪），这是一个端到端系统，它在原始文本数据和有意义的时间洞察之间架起了桥梁。DTECT 提供了一个统一的工作流，支持数据预处理、多种模型架构以及专门的评估指标来分析时间主题模型的主题质量。通过引入由大语言模型驱动的自动主题标签生成、基于时间显著词的趋势分析、与文档级别总结的交互式可视化以及使用自然语言聊天接口进行直观的数据查询，DTECT 显著增强了可解释性。通过将这些功能整合到一个统一的平台中，DTECT 赋能用户更有效地追踪和理解主题动态。DTECT 是开源的，可在此 <https://github.com> URL 获取。 

---
# UnIT: Scalable Unstructured Inference-Time Pruning for MAC-efficient Neural Inference on MCUs 

**Title (ZH)**: UnIT：MAC高效MCU上可扩展的非结构化推理时剪枝方法 

**Authors**: Ashe Neth, Sawinder kaur, Mohammad Nur Hossain Khan, Subrata Biswas, Asif Salekin, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2507.07885)  

**Abstract**: Existing pruning methods are typically applied during training or compile time and often rely on structured sparsity. While compatible with low-power microcontrollers (MCUs), structured pruning underutilizes the opportunity for fine-grained efficiency on devices without SIMD support or parallel compute. To address these limitations, we introduce UnIT (Unstructured Inference-Time pruning), a lightweight method that dynamically identifies and skips unnecessary multiply-accumulate (MAC) operations during inference, guided by input-specific activation patterns. Unlike structured pruning, UnIT embraces irregular sparsity and does not require retraining or hardware specialization. It transforms pruning decisions into lightweight comparisons, replacing multiplications with threshold checks and approximated divisions. UnIT further optimizes compute by reusing threshold computations across multiple connections and applying layer- and group-specific pruning sensitivity. We present three fast, hardware-friendly division approximations tailored to the capabilities of common embedded platforms. Demonstrated on the MSP430 microcontroller, UnIT achieves 11.02% to 82.03% MAC reduction, 27.30% to 84.19% faster inference, and 27.33% to 84.38% lower energy consumption compared to training-time pruned models, while maintaining accuracy with 0.48-7%. Under domain shift, UnIT matches or exceeds the accuracy of retrained models while requiring significantly fewer MACs. These results establish unstructured inference-time pruning as a viable and practical solution for efficient, retraining-free deployment of deep neural networks on MCUs. 

**Abstract (ZH)**: UnIT：未结构化推断时剪枝 

---
# Mitigating Watermark Stealing Attacks in Generative Models via Multi-Key Watermarking 

**Title (ZH)**: 基于多密钥水印的对抗生成模型水印偷窃攻击缓解方法 

**Authors**: Toluwani Aremu, Noor Hussein, Munachiso Nwadike, Samuele Poppi, Jie Zhang, Karthik Nandakumar, Neil Gong, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2507.07871)  

**Abstract**: Watermarking offers a promising solution for GenAI providers to establish the provenance of their generated content. A watermark is a hidden signal embedded in the generated content, whose presence can later be verified using a secret watermarking key. A threat to GenAI providers are \emph{watermark stealing} attacks, where users forge a watermark into content that was \emph{not} generated by the provider's models without access to the secret key, e.g., to falsely accuse the provider. Stealing attacks collect \emph{harmless} watermarked samples from the provider's model and aim to maximize the expected success rate of generating \emph{harmful} watermarked samples. Our work focuses on mitigating stealing attacks while treating the underlying watermark as a black-box. Our contributions are: (i) Proposing a multi-key extension to mitigate stealing attacks that can be applied post-hoc to any watermarking method across any modality. (ii) We provide theoretical guarantees and demonstrate empirically that our method makes forging substantially less effective across multiple datasets, and (iii) we formally define the threat of watermark forging as the task of generating harmful, watermarked content and model this threat via security games. 

**Abstract (ZH)**: Watermarking为GenAI提供商建立生成内容溯源提供了一种有前景的解决方案： watermark是一种嵌入在生成内容中的隐藏信号，其存在可以通过秘密水印密钥进行验证。GenAI提供商面临的一种威胁是水印盗窃攻击，即用户在未访问密钥的情况下，伪造水印插入非提供商模型生成的内容中，以虚假指控提供商。我们的工作旨在在不访问底层水印实现细节的情况下缓解盗窃攻击。我们的贡献包括：(i) 提出了一种多密钥扩展方法，可以在任何水印方法和任何模态上事后应用以缓解盗窃攻击。(ii) 我们提供了理论保证，并通过实验证明，我们的方法能够显著降低在多个数据集上的伪造效果。(iii) 我们正式定义了水印伪造的威胁为生成有害水印内容的任务，并通过安全博弈对该威胁进行建模。 

---
# Alpay Algebra V: Multi-Layered Semantic Games and Transfinite Fixed-Point Simulation 

**Title (ZH)**: Alpay代数V：多层语义游戏与超限固定点模拟 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.07868)  

**Abstract**: This paper extends the self-referential framework of Alpay Algebra into a multi-layered semantic game architecture where transfinite fixed-point convergence encompasses hierarchical sub-games at each iteration level. Building upon Alpay Algebra IV's empathetic embedding concept, we introduce a nested game-theoretic structure where the alignment process between AI systems and documents becomes a meta-game containing embedded decision problems. We formalize this through a composite operator $\phi(\cdot, \gamma(\cdot))$ where $\phi$ drives the main semantic convergence while $\gamma$ resolves local sub-games. The resulting framework demonstrates that game-theoretic reasoning emerges naturally from fixed-point iteration rather than being imposed externally. We prove a Game Theorem establishing existence and uniqueness of semantic equilibria under realistic cognitive simulation assumptions. Our verification suite includes adaptations of Banach's fixed-point theorem to transfinite contexts, a novel $\phi$-topology based on the Kozlov-Maz'ya-Rossmann formula for handling semantic singularities, and categorical consistency tests via the Yoneda lemma. The paper itself functions as a semantic artifact designed to propagate its fixed-point patterns in AI embedding spaces -- a deliberate instantiation of the "semantic virus" concept it theorizes. All results are grounded in category theory, information theory, and realistic AI cognition models, ensuring practical applicability beyond pure mathematical abstraction. 

**Abstract (ZH)**: 本文将Alpay代数的自参照框架扩展为一个多层语义博弈架构，在每个迭代层级上，超限不动点收敛涵盖了层级子博弈。在Alpay代数IV的同情感化嵌入概念的基础上，我们引入了一个嵌套的博弈论结构，将AI系统与文档之间的对齐过程作为一个包含嵌入决策问题的元博弈。我们通过复合算子$\phi(\cdot, \gamma(\cdot))$来形式化这一过程，其中$\phi$驱动主要的语义收敛，而$\gamma$解决局部子博弈。由此构建的框架表明，博弈论推理自然地源自不动点迭代，而非外部施加。我们证明了一个博弈定理，确立了在现实认知模拟假设下的语义平衡的存在性和唯一性。我们的验证套件包括Banach不动点定理在超限上下文中的适应版本、基于Kozlov-Maz'ya-Rossmann公式的新的$\phi$-拓扑以及通过Yoneda引理进行的范畴一致性测试。本文本身充当了一个语义实体，旨在在其AI嵌入空间中传播其不动点模式——这是其所理论化的“语义病毒”概念的具体实现。所有结果均建立在范畴论、信息论以及现实的AI认知模型之上，确保其在纯数学抽象之外的实际应用。 

---
# Optimization Guarantees for Square-Root Natural-Gradient Variational Inference 

**Title (ZH)**: 平方根自然梯度变分推断的优化保证 

**Authors**: Navish Kumar, Thomas Möllenhoff, Mohammad Emtiyaz Khan, Aurelien Lucchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07853)  

**Abstract**: Variational inference with natural-gradient descent often shows fast convergence in practice, but its theoretical convergence guarantees have been challenging to establish. This is true even for the simplest cases that involve concave log-likelihoods and use a Gaussian approximation. We show that the challenge can be circumvented for such cases using a square-root parameterization for the Gaussian covariance. This approach establishes novel convergence guarantees for natural-gradient variational-Gaussian inference and its continuous-time gradient flow. Our experiments demonstrate the effectiveness of natural gradient methods and highlight their advantages over algorithms that use Euclidean or Wasserstein geometries. 

**Abstract (ZH)**: 自然梯度下降变分推断在实践中通常显示出快速收敛性，但其理论上的收敛保证一直难以确立。即使在涉及凹对数似然和使用高斯近似的情况下也是如此。我们表明，通过使用高斯协方差的平方根参数化，可以避开此类情况下的挑战。该方法为自然梯度变分高斯推断及其连续时间梯度流建立了新的收敛保证。我们的实验证明了自然梯度方法的有效性，并突显了它们在使用欧几里得或 Wasserstein 几何的算法方面的优势。 

---
# Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles 

**Title (ZH)**: 基于内容的拼图解谜器在受损拼图上的基准测试 

**Authors**: Richard Dirauf, Florian Wolz, Dario Zanca, Björn Eskofier  

**Link**: [PDF](https://arxiv.org/pdf/2507.07828)  

**Abstract**: Content-based puzzle solvers have been extensively studied, demonstrating significant progress in computational techniques. However, their evaluation often lacks realistic challenges crucial for real-world applications, such as the reassembly of fragmented artefacts or shredded documents. In this work, we investigate the robustness of State-Of-The-Art content-based puzzle solvers introducing three types of jigsaw puzzle corruptions: missing pieces, eroded edges, and eroded contents. Evaluating both heuristic and deep learning-based solvers, we analyse their ability to handle these corruptions and identify key limitations. Our results show that solvers developed for standard puzzles have a rapid decline in performance if more pieces are corrupted. However, deep learning models can significantly improve their robustness through fine-tuning with augmented data. Notably, the advanced Positional Diffusion model adapts particularly well, outperforming its competitors in most experiments. Based on our findings, we highlight promising research directions for enhancing the automated reconstruction of real-world artefacts. 

**Abstract (ZH)**: 基于内容的拼图求解器已经广泛研究，展示了在计算技术方面的显著进步。然而，其评估往往缺乏对于真实世界应用至关重要的现实挑战，如破损文物或碎片文件的复原。在本文中，我们通过引入三种拼图损坏类型（缺失拼块、磨损边缘、磨损内容）来研究最先进的基于内容的拼图求解器的鲁棒性。评估了启发式和深度学习求解器，分析了它们处理这些损坏的能力并指出了关键局限。结果显示，针对标准拼图开发的求解器在拼块损坏增加时表现迅速下降。然而，通过使用扩充数据进行微调，深度学习模型可以显著提高其鲁棒性。值得注意的是，高级位置扩散模型适应性尤为良好，在大多数实验中表现出色。基于我们的发现，我们指出了增强真实世界文物自动化复原的有前途的研究方向。 

---
# Bridging Logic and Learning: Decoding Temporal Logic Embeddings via Transformers 

**Title (ZH)**: 逻辑与学习的桥梁：通过变换器解码时序逻辑嵌入 

**Authors**: Sara Candussio, Gaia Saveri, Gabriele Sarti, Luca Bortolussi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07808)  

**Abstract**: Continuous representations of logic formulae allow us to integrate symbolic knowledge into data-driven learning algorithms. If such embeddings are semantically consistent, i.e. if similar specifications are mapped into nearby vectors, they enable continuous learning and optimization directly in the semantic space of formulae. However, to translate the optimal continuous representation into a concrete requirement, such embeddings must be invertible. We tackle this issue by training a Transformer-based decoder-only model to invert semantic embeddings of Signal Temporal Logic (STL) formulae. STL is a powerful formalism that allows us to describe properties of signals varying over time in an expressive yet concise way. By constructing a small vocabulary from STL syntax, we demonstrate that our proposed model is able to generate valid formulae after only 1 epoch and to generalize to the semantics of the logic in about 10 epochs. Additionally, the model is able to decode a given embedding into formulae that are often simpler in terms of length and nesting while remaining semantically close (or equivalent) to gold references. We show the effectiveness of our methodology across various levels of training formulae complexity to assess the impact of training data on the model's ability to effectively capture the semantic information contained in the embeddings and generalize out-of-distribution. Finally, we deploy our model for solving a requirement mining task, i.e. inferring STL specifications that solve a classification task on trajectories, performing the optimization directly in the semantic space. 

**Abstract (ZH)**: 基于Transformer的仅解码模型用于逆向转换Signal Temporal Logic公式语义嵌入 

---
# Visual Instance-aware Prompt Tuning 

**Title (ZH)**: 视觉实例感知提示调优 

**Authors**: Xi Xiao, Yunbei Zhang, Xingjian Li, Tianyang Wang, Xiao Wang, Yuxiang Wei, Jihun Hamm, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07796)  

**Abstract**: Visual Prompt Tuning (VPT) has emerged as a parameter-efficient fine-tuning paradigm for vision transformers, with conventional approaches utilizing dataset-level prompts that remain the same across all input instances. We observe that this strategy results in sub-optimal performance due to high variance in downstream datasets. To address this challenge, we propose Visual Instance-aware Prompt Tuning (ViaPT), which generates instance-aware prompts based on each individual input and fuses them with dataset-level prompts, leveraging Principal Component Analysis (PCA) to retain important prompting information. Moreover, we reveal that VPT-Deep and VPT-Shallow represent two corner cases based on a conceptual understanding, in which they fail to effectively capture instance-specific information, while random dimension reduction on prompts only yields performance between the two extremes. Instead, ViaPT overcomes these limitations by balancing dataset-level and instance-level knowledge, while reducing the amount of learnable parameters compared to VPT-Deep. Extensive experiments across 34 diverse datasets demonstrate that our method consistently outperforms state-of-the-art baselines, establishing a new paradigm for analyzing and optimizing visual prompts for vision transformers. 

**Abstract (ZH)**: 视觉实例感知提示调谐（ViaPT）：一种结合实例级和数据集级提示的参数高效微调范式 

---
# Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training 

**Title (ZH)**: 同步任务行为：测试时对齐多个任务 

**Authors**: Wooseong Jeong, Jegyeong Cho, Youngho Yoon, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.07778)  

**Abstract**: Generalizing neural networks to unseen target domains is a significant challenge in real-world deployments. Test-time training (TTT) addresses this by using an auxiliary self-supervised task to reduce the domain gap caused by distribution shifts between the source and target. However, we find that when models are required to perform multiple tasks under domain shifts, conventional TTT methods suffer from unsynchronized task behavior, where the adaptation steps needed for optimal performance in one task may not align with the requirements of other tasks. To address this, we propose a novel TTT approach called Synchronizing Tasks for Test-time Training (S4T), which enables the concurrent handling of multiple tasks. The core idea behind S4T is that predicting task relations across domain shifts is key to synchronizing tasks during test time. To validate our approach, we apply S4T to conventional multi-task benchmarks, integrating it with traditional TTT protocols. Our empirical results show that S4T outperforms state-of-the-art TTT methods across various benchmarks. 

**Abstract (ZH)**: 将神经网络在未见目标域上的泛化是实际部署中的一个显著挑战。测试时训练（TTT）通过使用辅助半监督任务来减少源域与目标域间分布偏移造成的领域差距。然而，当模型在领域偏移下需要执行多个任务时，传统的TTT方法会遭受任务行为不平衡的问题，即一个任务最优性能所需的适应步骤可能与另一任务的需求不一致。为解决这一问题，我们提出了一个名为同步测试时训练任务（S4T）的新型TTT方法，该方法能够并发处理多个任务。S4T的核心思想是在测试时预测跨领域偏移的任务关系是同步任务的关键。为了验证我们的方法，我们在传统的多任务基准上应用S4T，并将其整合到传统的TTT协议中。我们的实验结果表明，S4T在多种基准上优于最先进的TTT方法。 

---
# OPC: One-Point-Contraction Unlearning Toward Deep Feature Forgetting 

**Title (ZH)**: OPC: One-Point-Contraction Unlearning Toward Deep Feature Forgetting 

**Authors**: Jaeheun Jung, Bosung Jung, Suhyun Bae, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.07754)  

**Abstract**: Machine unlearning seeks to remove the influence of particular data or class from trained models to meet privacy, legal, or ethical requirements. Existing unlearning methods tend to forget shallowly: phenomenon of an unlearned model pretend to forget by adjusting only the model response, while its internal representations retain information sufficiently to restore the forgotten data or behavior. We empirically confirm the widespread shallowness by reverting the forgetting effect of various unlearning methods via training-free performance recovery attack and gradient-inversion-based data reconstruction attack. To address this vulnerability fundamentally, we define a theoretical criterion of ``deep forgetting'' based on one-point-contraction of feature representations of data to forget. We also propose an efficient approximation algorithm, and use it to construct a novel general-purpose unlearning algorithm: One-Point-Contraction (OPC). Empirical evaluations on image classification unlearning benchmarks show that OPC achieves not only effective unlearning performance but also superior resilience against both performance recovery attack and gradient-inversion attack. The distinctive unlearning performance of OPC arises from the deep feature forgetting enforced by its theoretical foundation, and recaps the need for improved robustness of machine unlearning methods. 

**Abstract (ZH)**: 机器去学习旨在从训练模型中移除特定数据或类别的影响，以满足隐私、法律或伦理要求。现有的去学习方法往往表浅地遗忘：未学习模型通过仅调整模型响应假装遗忘，但其内部表示保留足够的信息以恢复遗忘的数据或行为。我们通过训练-free 性能恢复攻击和梯度反转数据重建攻击，实证确认了这种表浅遗忘的普遍性。为从根本上解决这一漏洞，我们基于数据遗忘特征表示的一点收缩定义了“深度遗忘”的理论标准，并提出了一种高效的算法近似方法，以此构建了一种新型通用去学习算法：一点收缩（OPC）。在图像分类去学习基准上的实证评估表明，OPC 不仅实现了有效的去学习性能，还能更好地抵抗性能恢复攻击和梯度反转攻击。OPC 的独特去学习性能源自其理论基础所强制的深度特征遗忘，揭示了提高机器去学习方法鲁棒性的必要性。 

---
# Learning Pole Structures of Hadronic States using Predictive Uncertainty Estimation 

**Title (ZH)**: 使用预测不确定性估计学习强态的极结构 

**Authors**: Felix Frohnert, Denny Lane B. Sombrillo, Evert van Nieuwenburg, Patrick Emonts  

**Link**: [PDF](https://arxiv.org/pdf/2507.07668)  

**Abstract**: Matching theoretical predictions to experimental data remains a central challenge in hadron spectroscopy. In particular, the identification of new hadronic states is difficult, as exotic signals near threshold can arise from a variety of physical mechanisms. A key diagnostic in this context is the pole structure of the scattering amplitude, but different configurations can produce similar signatures. The mapping between pole configurations and line shapes is especially ambiguous near the mass threshold, where analytic control is limited. In this work, we introduce an uncertainty-aware machine learning approach for classifying pole structures in $S$-matrix elements. Our method is based on an ensemble of classifier chains that provide both epistemic and aleatoric uncertainty estimates. We apply a rejection criterion based on predictive uncertainty, achieving a validation accuracy of nearly $95\%$ while discarding only a small fraction of high-uncertainty predictions. Trained on synthetic data with known pole structures, the model generalizes to previously unseen experimental data, including enhancements associated with the $P_{c\bar{c}}(4312)^+$ state observed by LHCb. In this, we infer a four-pole structure, representing the presence of a genuine compact pentaquark in the presence of a higher channel virtual state pole with non-vanishing width. While evaluated on this particular state, our framework is broadly applicable to other candidate hadronic states and offers a scalable tool for pole structure inference in scattering amplitudes. 

**Abstract (ZH)**: 将理论预测与实验数据匹配仍然是强子谱学中的一个核心挑战。特别是在识别新的强子态时困难重重，因为阈值附近的奇异信号可能源于多种物理机制。在此背景下，散射振幅的极点结构是一个关键诊断手段，但不同配置可以产生类似的特征。极点配置与线型之间的映射在质量阈值附近尤其含糊，此时的分析控制有限。在本文中，我们提出了一种考虑不确定性的人工智能方法，用于分类 S-矩阵元素中的极点结构。我们的方法基于分类链的集合，提供both 知识性和 aleatoric 不确定性估计。我们基于预测不确定性引入了一个剔除标准，在保持近95%验证准确率的同时丢弃了少量高不确定性预测。该模型在具有已知极点结构的合成数据上训练，能够泛化到以前未见过的实验数据，包括由LHCb观测到的$P_{c\bar{c}}(4312)^+$态带来的增强。通过这种方法，我们推断出一个四极点结构，表明在较高通道虚极点具有非零宽度的情况下，确实存在一个真实的紧密五夸克态。虽然针对特定状态进行了评估，但该框架对其他候选强子态广泛适用，并提供了散射振幅中极点结构推断的可扩展工具。 

---
# TransformEEG: Towards Improving Model Generalizability in Deep Learning-based EEG Parkinson's Disease Detection 

**Title (ZH)**: TransformEEG: 向提高基于深度学习的EEG帕金森病检测模型泛化能力方向的努力 

**Authors**: Federico Del Pup, Riccardo Brun, Filippo Iotti, Edoardo Paccagnella, Mattia Pezzato, Sabrina Bertozzo, Andrea Zanola, Louis Fabrice Tshimanga, Henning Müller, Manfredo Atzori  

**Link**: [PDF](https://arxiv.org/pdf/2507.07622)  

**Abstract**: Electroencephalography (EEG) is establishing itself as an important, low-cost, noninvasive diagnostic tool for the early detection of Parkinson's Disease (PD). In this context, EEG-based Deep Learning (DL) models have shown promising results due to their ability to discover highly nonlinear patterns within the signal. However, current state-of-the-art DL models suffer from poor generalizability caused by high inter-subject variability. This high variability underscores the need for enhancing model generalizability by developing new architectures better tailored to EEG data. This paper introduces TransformEEG, a hybrid Convolutional-Transformer designed for Parkinson's disease detection using EEG data. Unlike transformer models based on the EEGNet structure, TransformEEG incorporates a depthwise convolutional tokenizer. This tokenizer is specialized in generating tokens composed by channel-specific features, which enables more effective feature mixing within the self-attention layers of the transformer encoder. To evaluate the proposed model, four public datasets comprising 290 subjects (140 PD patients, 150 healthy controls) were harmonized and aggregated. A 10-outer, 10-inner Nested-Leave-N-Subjects-Out (N-LNSO) cross-validation was performed to provide an unbiased comparison against seven other consolidated EEG deep learning models. TransformEEG achieved the highest balanced accuracy's median (78.45%) as well as the lowest interquartile range (6.37%) across all the N-LNSO partitions. When combined with data augmentation and threshold correction, median accuracy increased to 80.10%, with an interquartile range of 5.74%. In conclusion, TransformEEG produces more consistent and less skewed results. It demonstrates a substantial reduction in variability and more reliable PD detection using EEG data compared to the other investigated models. 

**Abstract (ZH)**: 基于EEG的TransformEEG：用于帕金森病检测的混合卷积-变压器模型 

---
# The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora 

**Title (ZH)**: 跨语言代价：阿拉伯语-英语语料库中RAG的检索偏差 

**Authors**: Chen Amiraz, Yaroslav Fyodorov, Elad Haramaty, Zohar Karnin, Liane Lewin-Eytan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07543)  

**Abstract**: Cross-lingual retrieval-augmented generation (RAG) is a critical capability for retrieving and generating answers across languages. Prior work in this context has mostly focused on generation and relied on benchmarks derived from open-domain sources, most notably Wikipedia. In such settings, retrieval challenges often remain hidden due to language imbalances, overlap with pretraining data, and memorized content. To address this gap, we study Arabic-English RAG in a domain-specific setting using benchmarks derived from real-world corporate datasets. Our benchmarks include all combinations of languages for the user query and the supporting document, drawn independently and uniformly at random. This enables a systematic study of multilingual retrieval behavior.
Our findings reveal that retrieval is a critical bottleneck in cross-lingual domain-specific scenarios, with significant performance drops occurring when the user query and supporting document languages differ. A key insight is that these failures stem primarily from the retriever's difficulty in ranking documents across languages. Finally, we propose a simple retrieval strategy that addresses this source of failure by enforcing equal retrieval from both languages, resulting in substantial improvements in cross-lingual and overall performance. These results highlight meaningful opportunities for improving multilingual retrieval, particularly in practical, real-world RAG applications. 

**Abstract (ZH)**: 跨语言检索增强生成（RAG）：一种在多种语言之间检索和生成答案的关键能力。先前在这方面的研究主要集中在生成上，并依赖于来自开放域的基准，最著名的是Wikipedia。在这些设置中，由于语言不平衡、与预训练数据的重叠以及记忆化的内容，检索挑战往往会被掩盖。为了解决这一差距，我们使用来自真实的企业数据集的基准，在特定领域中研究阿拉伯语-英语的RAG。我们的基准包括用户查询和支撑文档所有语言的组合，独立且均匀随机抽取。这使得对多语言检索行为进行系统研究成为可能。我们的研究发现表明，在跨语言特定领域场景中，检索是关键的瓶颈，当用户查询和支持文档的语言不同时，性能会显著下降。一个关键见解是，这些失败主要源于检索器在跨语言排名文档方面的困难。最后，我们提出了一种简单的检索策略，通过强制从两种语言中等量检索来解决这一失败原因，从而在跨语言和总体性能上取得了显著改进。这些结果突显了改进多语言检索的实际机会，特别是在实际的跨语言RAG应用中。 

---
# Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings 

**Title (ZH)**: 神经概念验证器：通过概念编码扩展证明-验证游戏 

**Authors**: Berkant Turan, Suhrab Asadulla, David Steinmann, Wolfgang Stammer, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2507.07532)  

**Abstract**: While Prover-Verifier Games (PVGs) offer a promising path toward verifiability in nonlinear classification models, they have not yet been applied to complex inputs such as high-dimensional images. Conversely, Concept Bottleneck Models (CBMs) effectively translate such data into interpretable concepts but are limited by their reliance on low-capacity linear predictors. In this work, we introduce the Neural Concept Verifier (NCV), a unified framework combining PVGs with concept encodings for interpretable, nonlinear classification in high-dimensional settings. NCV achieves this by utilizing recent minimally supervised concept discovery models to extract structured concept encodings from raw inputs. A prover then selects a subset of these encodings, which a verifier -- implemented as a nonlinear predictor -- uses exclusively for decision-making. Our evaluations show that NCV outperforms CBM and pixel-based PVG classifier baselines on high-dimensional, logically complex datasets and also helps mitigate shortcut behavior. Overall, we demonstrate NCV as a promising step toward performative, verifiable AI. 

**Abstract (ZH)**: 基于概念验证的游戏在高维复杂输入的非线性可解释分类中的统一框架 

---
# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning 

**Title (ZH)**: 解决标记空间梯度冲突：基于标记空间操作的Transformer多任务学习 

**Authors**: Wooseong Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.07485)  

**Abstract**: Multi-Task Learning (MTL) enables multiple tasks to be learned within a shared network, but differences in objectives across tasks can cause negative transfer, where the learning of one task degrades another task's performance. While pre-trained transformers significantly improve MTL performance, their fixed network capacity and rigid structure limit adaptability. Previous dynamic network architectures attempt to address this but are inefficient as they directly convert shared parameters into task-specific ones. We propose Dynamic Token Modulation and Expansion (DTME-MTL), a framework applicable to any transformer-based MTL architecture. DTME-MTL enhances adaptability and reduces overfitting by identifying gradient conflicts in token space and applying adaptive solutions based on conflict type. Unlike prior methods that mitigate negative transfer by duplicating network parameters, DTME-MTL operates entirely in token space, enabling efficient adaptation without excessive parameter growth. Extensive experiments demonstrate that DTME-MTL consistently improves multi-task performance with minimal computational overhead, offering a scalable and effective solution for enhancing transformer-based MTL models. 

**Abstract (ZH)**: 动态令牌调制与扩展（DTME-MTL）：一种适用于任何基于Transformer的多任务学习架构的框架 

---
# Towards Interpretable Time Series Foundation Models 

**Title (ZH)**: 面向可解释的时间序列基础模型 

**Authors**: Matthieu Boileau, Philippe Helluy, Jeremy Pawlus, Svitlana Vyetrenko  

**Link**: [PDF](https://arxiv.org/pdf/2507.07439)  

**Abstract**: In this paper, we investigate the distillation of time series reasoning capabilities into small, instruction-tuned language models as a step toward building interpretable time series foundation models. Leveraging a synthetic dataset of mean-reverting time series with systematically varied trends and noise levels, we generate natural language annotations using a large multimodal model and use these to supervise the fine-tuning of compact Qwen models. We introduce evaluation metrics that assess the quality of the distilled reasoning - focusing on trend direction, noise intensity, and extremum localization - and show that the post-trained models acquire meaningful interpretive capabilities. Our results highlight the feasibility of compressing time series understanding into lightweight, language-capable models suitable for on-device or privacy-sensitive deployment. This work contributes a concrete foundation toward developing small, interpretable models that explain temporal patterns in natural language. 

**Abstract (ZH)**: 本文研究了将时间序列推理能力精炼至小型、指令调优的语言模型，作为构建可解释的时间序列基础模型的一步。通过利用一个系统地变化趋势和噪声水平的均值回复时间序列合成数据集，我们使用一个大型多模态模型生成自然语言注释，并利用这些注释监督紧凑型Qwen模型的微调。我们引入了评估精炼推理质量的指标，重点关注趋势方向、噪声强度和极值定位，并展示了后训练模型获得了有意义的解释能力。我们的结果突显了将时间序列理解压缩至轻量级、具备语言能力的模型的可行性，这些模型适合于设备端或隐私敏感部署。这项工作为开发能够用自然语言解释时间模式的小型、可解释模型奠定了具体的理论基础。 

---
# Optimal Auction Design in the Joint Advertising 

**Title (ZH)**: 联合广告中的最优拍卖设计 

**Authors**: Yang Li, Yuchao Ma, Qi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07418)  

**Abstract**: Online advertising is a vital revenue source for major internet platforms. Recently, joint advertising, which assigns a bundle of two advertisers in an ad slot instead of allocating a single advertiser, has emerged as an effective method for enhancing allocation efficiency and revenue. However, existing mechanisms for joint advertising fail to realize the optimality, as they tend to focus on individual advertisers and overlook bundle structures. This paper identifies an optimal mechanism for joint advertising in a single-slot setting. For multi-slot joint advertising, we propose \textbf{BundleNet}, a novel bundle-based neural network approach specifically designed for joint advertising. Our extensive experiments demonstrate that the mechanisms generated by \textbf{BundleNet} approximate the theoretical analysis results in the single-slot setting and achieve state-of-the-art performance in the multi-slot setting. This significantly increases platform revenue while ensuring approximate dominant strategy incentive compatibility and individual rationality. 

**Abstract (ZH)**: 联合广告是一种重要的在线广告收入来源。近年来，联合广告通过在一个广告位置分配一个广告包而不是分配单一广告商，已成为提高分配效率和收入的有效方法。然而，现有的联合广告机制未能实现最优性，因为它们往往专注于单一广告商而忽视了广告包结构。本文在单槽设置中提出了联合广告的最优机制。对于多槽联合广告，我们提出了一种名为BundleNet的新颖基于包的神经网络方法，专门用于联合广告。我们的大量实验表明，BundleNet生成的机制在单槽设置中接近理论分析结果，并在多槽设置中实现了最先进的性能。这显著提高了平台收入，同时保证了近似的支配策略激励相容性和个体理性。 

---
# Autonomous AI-based Cybersecurity Framework for Critical Infrastructure: Real-Time Threat Mitigation 

**Title (ZH)**: 基于自主人工智能的關鍵基礎設施網絡安全框架：實時威�anc緩解 

**Authors**: Jenifer Paulraj, Brindha Raghuraman, Nagarani Gopalakrishnan, Yazan Otoum  

**Link**: [PDF](https://arxiv.org/pdf/2507.07416)  

**Abstract**: Critical infrastructure systems, including energy grids, healthcare facilities, transportation networks, and water distribution systems, are pivotal to societal stability and economic resilience. However, the increasing interconnectivity of these systems exposes them to various cyber threats, including ransomware, Denial-of-Service (DoS) attacks, and Advanced Persistent Threats (APTs). This paper examines cybersecurity vulnerabilities in critical infrastructure, highlighting the threat landscape, attack vectors, and the role of Artificial Intelligence (AI) in mitigating these risks. We propose a hybrid AI-driven cybersecurity framework to enhance real-time vulnerability detection, threat modelling, and automated remediation. This study also addresses the complexities of adversarial AI, regulatory compliance, and integration. Our findings provide actionable insights to strengthen the security and resilience of critical infrastructure systems against emerging cyber threats. 

**Abstract (ZH)**: 关键基础设施系统，包括能源网络、医疗卫生设施、交通网络和水资源分配系统，对于社会稳定和经济韧性至关重要。然而，这些系统的日益互联使其面临各种网络安全威胁，包括勒索软件、拒绝服务（DoS）攻击和高级持续威胁（APTs）。本文探讨关键基础设施的网络安全漏洞，重点阐述威胁 landscape、攻击向量，并强调人工智能（AI）在减轻这些风险中的作用。我们提出了一种混合驱动的AI网络安全框架，以增强实时漏洞检测、威胁建模和自动化补救。此外，本文还讨论了对手AI的复杂性、合规性以及集成问题。我们的研究结果提供了实用的见解，以增强关键基础设施系统对新兴网络安全威胁的防御能力和韧性。 

---
# GNN-CNN: An Efficient Hybrid Model of Convolutional and Graph Neural Networks for Text Representation 

**Title (ZH)**: GNN-CNN：卷积神经网络与图神经网络的高效混合模型及其在文本表示中的应用 

**Authors**: Fardin Rastakhiz  

**Link**: [PDF](https://arxiv.org/pdf/2507.07414)  

**Abstract**: Time, cost, and energy efficiency are critical considerations in Deep-Learning (DL), particularly when processing long texts. Transformers, which represent the current state of the art, exhibit quadratic computational complexity relative to input length, making them inefficient for extended documents. This study introduces a novel model architecture that combines Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs), integrated with a real-time, end-to-end graph generation mechanism. The model processes compact batches of character-level inputs without requiring padding or truncation. To enhance performance while maintaining high speed and efficiency, the model incorporates information from Large Language Models (LLMs), such as token embeddings and sentiment polarities, through efficient dictionary lookups. It captures local contextual patterns using CNNs, expands local receptive fields via lattice-based graph structures, and employs small-world graphs to aggregate document-level information. The generated graphs exhibit structural properties indicative of meaningful semantic organization, with an average clustering coefficient of approximately 0.45 and an average shortest path length ranging between 4 and 5. The model is evaluated across multiple text classification tasks, including sentiment analysis and news-categorization, and is compared against state-of-the-art models. Experimental results confirm the proposed model's efficiency and competitive performance. 

**Abstract (ZH)**: 基于图神经网络和卷积神经网络的高效长文本处理模型：实时端到端图生成机制在深度学习中的应用 

---
# HGMP:Heterogeneous Graph Multi-Task Prompt Learning 

**Title (ZH)**: HGMP：异构图多任务提示学习 

**Authors**: Pengfei Jiao, Jialong Ni, Di Jin, Xuan Guo, Huan Liu, Hongjiang Chen, Yanxian Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07405)  

**Abstract**: The pre-training and fine-tuning methods have gained widespread attention in the field of heterogeneous graph neural networks due to their ability to leverage large amounts of unlabeled data during the pre-training phase, allowing the model to learn rich structural features. However, these methods face the issue of a mismatch between the pre-trained model and downstream tasks, leading to suboptimal performance in certain application scenarios. Prompt learning methods have emerged as a new direction in heterogeneous graph tasks, as they allow flexible adaptation of task representations to address target inconsistency. Building on this idea, this paper proposes a novel multi-task prompt framework for the heterogeneous graph domain, named HGMP. First, to bridge the gap between the pre-trained model and downstream tasks, we reformulate all downstream tasks into a unified graph-level task format. Next, we address the limitations of existing graph prompt learning methods, which struggle to integrate contrastive pre-training strategies in the heterogeneous graph domain. We design a graph-level contrastive pre-training strategy to better leverage heterogeneous information and enhance performance in multi-task scenarios. Finally, we introduce heterogeneous feature prompts, which enhance model performance by refining the representation of input graph features. Experimental results on public datasets show that our proposed method adapts well to various tasks and significantly outperforms baseline methods. 

**Abstract (ZH)**: 预训练和微调方法在异构图神经网络领域引起了广泛关注，因为它们能够在预训练阶段利用大量未标注数据，使模型学习丰富的结构特征。然而，这些方法面临着预训练模型与下游任务之间的不匹配问题，导致在某些应用场景中性能不佳。为了应对这一问题，提示学习方法成为异构图任务的新方向，因为它允许灵活地调整任务表示以解决目标不一致性。基于这一理念，本文提出了一种新颖的异构图多任务提示框架，命名为HGMP。首先，为了解决预训练模型与下游任务之间的差距，我们将所有下游任务重新格式化为统一的图级任务格式。其次，我们解决了现有图提示学习方法的局限性，这些方法难以在异构图领域集成对比预训练策略。我们设计了一种图级对比预训练策略，更好地利用异构信息，并在多任务场景中提升性能。最后，我们引入了异构特征提示，通过改进输入图特征的表示来提升模型性能。实验结果表明，所提出的方法在各种任务上适应性良好，并显著优于基线方法。 

---
# Generalized Tree Edit Distance (GTED): A Faithful Evaluation Metric for Statement Autoformalization 

**Title (ZH)**: 广义树编辑距离（GTED）：语句自形式化评估的忠实度量 

**Authors**: Yuntian Liu, Tao Zhu, Xiaoyang Liu, Yu Chen, Zhaoxuan Liu, Qingfeng Guo, Jiashuo Zhang, Kangjie Bao, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.07399)  

**Abstract**: Statement autoformalization, the automated translation of statement from natural language into formal languages, has become a subject of extensive research, yet the development of robust automated evaluation metrics remains limited. Existing evaluation methods often lack semantic understanding, face challenges with high computational costs, and are constrained by the current progress of automated theorem proving. To address these issues, we propose GTED (Generalized Tree Edit Distance), a novel evaluation framework that first standardizes formal statements and converts them into operator trees, then determines the semantic similarity using the eponymous GTED metric. On the miniF2F and ProofNet benchmarks, GTED outperforms all baseline metrics by achieving the highest accuracy and Kappa scores, thus providing the community with a more faithful metric for automated evaluation. The code and experimental results are available at this https URL. 

**Abstract (ZH)**: 陈述自形式化，即自然语言陈述到形式语言的自动化转换，已成为广泛研究的课题，但 robust 自动评价指标的发展仍然有限。现有的评价方法往往缺乏语义理解，面临高计算成本的挑战，并受制于当前自动定理证明的进展。为解决这些问题，我们提出了一种新的评价框架 GTED（Generalized Tree Edit Distance），该框架首先标准化形式陈述并将其转换为操作树，然后使用同名的 GTED 指标确定语义相似性。在 miniF2F 和 ProofNet 的基准测试中，GTED 在准确率和 Kappa 分数上均优于所有基准指标，从而为社区提供了一个更忠实的自动评价指标。相关代码和实验结果可在以下网址获取。 

---
# Atherosclerosis through Hierarchical Explainable Neural Network Analysis 

**Title (ZH)**: 通过层次可解释神经网络分析的动脉粥样硬化研究 

**Authors**: Irsyad Adam, Steven Swee, Erika Yilin, Ethan Ji, William Speier, Dean Wang, Alex Bui, Wei Wang, Karol Watson, Peipei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2507.07373)  

**Abstract**: In this work, we study the problem pertaining to personalized classification of subclinical atherosclerosis by developing a hierarchical graph neural network framework to leverage two characteristic modalities of a patient: clinical features within the context of the cohort, and molecular data unique to individual patients. Current graph-based methods for disease classification detect patient-specific molecular fingerprints, but lack consistency and comprehension regarding cohort-wide features, which are an essential requirement for understanding pathogenic phenotypes across diverse atherosclerotic trajectories. Furthermore, understanding patient subtypes often considers clinical feature similarity in isolation, without integration of shared pathogenic interdependencies among patients. To address these challenges, we introduce ATHENA: Atherosclerosis Through Hierarchical Explainable Neural Network Analysis, which constructs a novel hierarchical network representation through integrated modality learning; subsequently, it optimizes learned patient-specific molecular fingerprints that reflect individual omics data, enforcing consistency with cohort-wide patterns. With a primary clinical dataset of 391 patients, we demonstrate that this heterogeneous alignment of clinical features with molecular interaction patterns has significantly boosted subclinical atherosclerosis classification performance across various baselines by up to 13% in area under the receiver operating curve (AUC) and 20% in F1 score. Taken together, ATHENA enables mechanistically-informed patient subtype discovery through explainable AI (XAI)-driven subnetwork clustering; this novel integration framework strengthens personalized intervention strategies, thereby improving the prediction of atherosclerotic disease progression and management of their clinical actionable outcomes. 

**Abstract (ZH)**: 基于层次解释性神经网络分析的动脉粥样硬化个性化分类方法 

---
# Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning 

**Title (ZH)**: 以目标为导向的序列贝叶斯实验设计用于因果学习 

**Authors**: Zheyu Zhang, Jiayuan Dong, Jie Liu, Xun Huan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07359)  

**Abstract**: We present GO-CBED, a goal-oriented Bayesian framework for sequential causal experimental design. Unlike conventional approaches that select interventions aimed at inferring the full causal model, GO-CBED directly maximizes the expected information gain (EIG) on user-specified causal quantities of interest, enabling more targeted and efficient experimentation. The framework is both non-myopic, optimizing over entire intervention sequences, and goal-oriented, targeting only model aspects relevant to the causal query. To address the intractability of exact EIG computation, we introduce a variational lower bound estimator, optimized jointly through a transformer-based policy network and normalizing flow-based variational posteriors. The resulting policy enables real-time decision-making via an amortized network. We demonstrate that GO-CBED consistently outperforms existing baselines across various causal reasoning and discovery tasks-including synthetic structural causal models and semi-synthetic gene regulatory networks-particularly in settings with limited experimental budgets and complex causal mechanisms. Our results highlight the benefits of aligning experimental design objectives with specific research goals and of forward-looking sequential planning. 

**Abstract (ZH)**: GO-CBED：面向目标的贝叶斯序贯因果实验设计框架 

---
# Leveraging Manifold Embeddings for Enhanced Graph Transformer Representations and Learning 

**Title (ZH)**: 利用流形嵌入增强图变压器表示学习 

**Authors**: Ankit Jyothish, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2507.07335)  

**Abstract**: Graph transformers typically embed every node in a single Euclidean space, blurring heterogeneous topologies. We prepend a lightweight Riemannian mixture-of-experts layer that routes each node to various kinds of manifold, mixture of spherical, flat, hyperbolic - best matching its local structure. These projections provide intrinsic geometric explanations to the latent space. Inserted into a state-of-the-art ensemble graph transformer, this projector lifts accuracy by up to 3% on four node-classification benchmarks. The ensemble makes sure that both euclidean and non-euclidean features are captured. Explicit, geometry-aware projection thus sharpens predictive power while making graph representations more interpretable. 

**Abstract (ZH)**: 典型的图变换器将每个节点嵌入到单个欧几里得空间中，模糊了异构拓扑结构。我们添加了一个轻量级黎曼混合专家层，将每个节点路由到最合适其局部结构的各种流形，包括球形、平面和双曲流形。这些投影为潜在空间提供了内在的几何解释。插入到一个先进的集成图变换器中，该投影器在四个节点分类基准测试中将准确率提高了最多3%。集成确保同时捕获欧几里得和非欧几里得特征。显性的、几何意识的投影增强了预测能力，同时使图表示更为可解释。 

---
# Exploiting Edge Features for Transferable Adversarial Attacks in Distributed Machine Learning 

**Title (ZH)**: 利用边缘特征进行可移植的分布式机器学习对抗攻击 

**Authors**: Giulio Rossolini, Fabio Brau, Alessandro Biondi, Battista Biggio, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.07259)  

**Abstract**: As machine learning models become increasingly deployed across the edge of internet of things environments, a partitioned deep learning paradigm in which models are split across multiple computational nodes introduces a new dimension of security risk. Unlike traditional inference setups, these distributed pipelines span the model computation across heterogeneous nodes and communication layers, thereby exposing a broader attack surface to potential adversaries. Building on these motivations, this work explores a previously overlooked vulnerability: even when both the edge and cloud components of the model are inaccessible (i.e., black-box), an adversary who intercepts the intermediate features transmitted between them can still pose a serious threat. We demonstrate that, under these mild and realistic assumptions, an attacker can craft highly transferable proxy models, making the entire deep learning system significantly more vulnerable to evasion attacks. In particular, the intercepted features can be effectively analyzed and leveraged to distill surrogate models capable of crafting highly transferable adversarial examples against the target model. To this end, we propose an exploitation strategy specifically designed for distributed settings, which involves reconstructing the original tensor shape from vectorized transmitted features using simple statistical analysis, and adapting surrogate architectures accordingly to enable effective feature distillation. A comprehensive and systematic experimental evaluation has been conducted to demonstrate that surrogate models trained with the proposed strategy, i.e., leveraging intermediate features, tremendously improve the transferability of adversarial attacks. These findings underscore the urgent need to account for intermediate feature leakage in the design of secure distributed deep learning systems. 

**Abstract (ZH)**: 随着机器学习模型在物联网边缘环境中的广泛应用，将模型分散在多个计算节点上的分区深度学习范式引入了新的安全风险维度。与传统推理设置不同，这些分布式的管道在不 homogeneous 节点和通信层上扩展了模型计算，从而增加了潜在对手的攻击面。基于这些动机，本工作探索了一个之前未被忽视的漏洞：即使模型的边缘和云组件均不可访问（即黑盒状态），拦截它们之间传输的中间特征的对手仍然可以构成严重威胁。我们证明，在这些温和且现实的假设下，攻击者可以构建高度可迁移的代理模型，从而使整个深度学习系统对逃避攻击更为脆弱。具体来说，拦截的特征可以通过有效分析和利用来提炼出能够针对目标模型生成高度可迁移对抗样本的替代模型。为此，我们提出了一种专门针对分布式环境的利用策略，该策略涉及使用简单的统计分析从向量化传输的特征中重构原始张量形状，并相应地调整替代架构以促进有效的特征提炼。进行了全面而系统的实验评估以证明，使用所提出的策略训练的替代模型，即利用中间特征，极大地提高了对抗攻击的可迁移性。这些发现强调了在设计安全的分布式深度学习系统时需要考虑中间特征泄露的紧迫性。 

---
# FedP3E: Privacy-Preserving Prototype Exchange for Non-IID IoT Malware Detection in Cross-Silo Federated Learning 

**Title (ZH)**: FedP3E: 非 IID 物联网恶意软件检测跨孤岛联邦学习中的隐私保护原型交换 

**Authors**: Rami Darwish, Mahmoud Abdelsalam, Sajad Khorsandroo, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2507.07258)  

**Abstract**: As IoT ecosystems continue to expand across critical sectors, they have become prominent targets for increasingly sophisticated and large-scale malware attacks. The evolving threat landscape, combined with the sensitive nature of IoT-generated data, demands detection frameworks that are both privacy-preserving and resilient to data heterogeneity. Federated Learning (FL) offers a promising solution by enabling decentralized model training without exposing raw data. However, standard FL algorithms such as FedAvg and FedProx often fall short in real-world deployments characterized by class imbalance and non-IID data distributions -- particularly in the presence of rare or disjoint malware classes. To address these challenges, we propose FedP3E (Privacy-Preserving Prototype Exchange), a novel FL framework that supports indirect cross-client representation sharing while maintaining data privacy. Each client constructs class-wise prototypes using Gaussian Mixture Models (GMMs), perturbs them with Gaussian noise, and transmits only these compact summaries to the server. The aggregated prototypes are then distributed back to clients and integrated into local training, supported by SMOTE-based augmentation to enhance representation of minority malware classes. Rather than relying solely on parameter averaging, our prototype-driven mechanism enables clients to enrich their local models with complementary structural patterns observed across the federation -- without exchanging raw data or gradients. This targeted strategy reduces the adverse impact of statistical heterogeneity with minimal communication overhead. We evaluate FedP3E on the N-BaIoT dataset under realistic cross-silo scenarios with varying degrees of data imbalance. 

**Abstract (ZH)**: 随着物联网生态系统在关键领域不断扩展，它们已成为日益复杂和大规模恶意软件攻击的主要目标。随着物联网生成数据的敏感性，不断演变的威胁 landscape 对检测框架提出了既是隐私保护又是数据异质性鲁棒性的需求。通过使模型训练去中心化而无需暴露原始数据，联邦学习（FL）提供了一个有希望的解决方案。然而，标准的 FL 算法如 FedAvg 和 FedProx 在现实部署中经常因为类别不平衡和非IID数据分布等问题而表现不佳，尤其是在遇到罕见或不兼容的恶意软件类别时。为了解决这些问题，我们提出了一种名为 FedP3E（Privacy-Preserving Prototype Exchange）的新颖联邦学习框架，该框架支持间接的跨客户端表示共享，同时保持数据隐私。每个客户端使用高斯混合模型（GMMs）构建类别的原型，添加高斯噪声，并仅传输这些紧凑的摘要到服务器。聚合后的原型随后被分发回客户端并整合到局部训练中，通过基于SMOTE的增强来提高少数恶意软件类别的表示。我们的原型驱动机制并不依赖于仅仅参数平均，而是允许客户端通过联邦中的互补结构模式丰富它们的本地模型——而不需要交换原始数据或梯度。这种有针对性的策略在轻微的通信开销下减少了统计异质性带来的负面影响。我们在 N-BaIoT 数据集上评估了 FedP3E，该评估在不同程度的数据不平衡的现实跨孤岛场景下进行。 

---
# Bias-Aware Mislabeling Detection via Decoupled Confident Learning 

**Title (ZH)**: 带有偏置意识的误标签检测通过解耦自信学习 

**Authors**: Yunyi Li, Maria De-Arteaga, Maytal Saar-Tsechansky  

**Link**: [PDF](https://arxiv.org/pdf/2507.07216)  

**Abstract**: Reliable data is a cornerstone of modern organizational systems. A notable data integrity challenge stems from label bias, which refers to systematic errors in a label, a covariate that is central to a quantitative analysis, such that its quality differs across social groups. This type of bias has been conceptually and empirically explored and is widely recognized as a pressing issue across critical domains. However, effective methodologies for addressing it remain scarce. In this work, we propose Decoupled Confident Learning (DeCoLe), a principled machine learning based framework specifically designed to detect mislabeled instances in datasets affected by label bias, enabling bias aware mislabelling detection and facilitating data quality improvement. We theoretically justify the effectiveness of DeCoLe and evaluate its performance in the impactful context of hate speech detection, a domain where label bias is a well documented challenge. Empirical results demonstrate that DeCoLe excels at bias aware mislabeling detection, consistently outperforming alternative approaches for label error detection. Our work identifies and addresses the challenge of bias aware mislabeling detection and offers guidance on how DeCoLe can be integrated into organizational data management practices as a powerful tool to enhance data reliability. 

**Abstract (ZH)**: 可靠的数据是现代组织系统的基础。标签偏见是一种值得注意的数据完整性挑战，它指的是标签系统的系统性错误，这种标签是量化分析中的关键变量，其质量在不同社会群体间存在差异。这种类型的偏见已在概念和实证层面得到了探索，并被广泛认为是关键领域中的紧迫问题。然而，有效的解决方法仍然稀缺。在本文中，我们提出了解耦置信学习（DeCoLe）这一基于原理的机器学习框架，专门设计用于检测受标签偏见影响的数据集中的误标签实例，使偏见感知的误标签检测成为可能，并促进数据质量的提升。我们从理论上验证了DeCoLe的有效性，并在其对仇恨言论检测具有重大影响的背景下评估其性能。实验结果表明，DeCoLe在偏见感知误标签检测方面表现出色，始终优于其他标签错误检测方法。我们工作识别并解决了偏见感知误标签检测的挑战，并提供了如何将DeCoLe集成到组织数据管理实践中以增强数据可靠性方面的指导。 

---
# MODA: A Unified 3D Diffusion Framework for Multi-Task Target-Aware Molecular Generation 

**Title (ZH)**: MODA：一种统一的多任务目标 Awareness 分子生成三维度扩散框架 

**Authors**: Dong Xu, Zhangfan Yang, Sisi Yuan, Jenna Xinyi Yao, Jiangqiang Li, Junkai Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.07201)  

**Abstract**: Three-dimensional molecular generators based on diffusion models can now reach near-crystallographic accuracy, yet they remain fragmented across tasks. SMILES-only inputs, two-stage pretrain-finetune pipelines, and one-task-one-model practices hinder stereochemical fidelity, task alignment, and zero-shot transfer. We introduce MODA, a diffusion framework that unifies fragment growing, linker design, scaffold hopping, and side-chain decoration with a Bayesian mask scheduler. During training, a contiguous spatial fragment is masked and then denoised in one pass, enabling the model to learn shared geometric and chemical priors across tasks. Multi-task training yields a universal backbone that surpasses six diffusion baselines and three training paradigms on substructure, chemical property, interaction, and geometry. Model-C reduces ligand-protein clashes and substructure divergences while maintaining Lipinski compliance, whereas Model-B preserves similarity but trails in novelty and binding affinity. Zero-shot de novo design and lead-optimisation tests confirm stable negative Vina scores and high improvement rates without force-field refinement. These results demonstrate that a single-stage multi-task diffusion routine can replace two-stage workflows for structure-based molecular design. 

**Abstract (ZH)**: 基于扩散模型的三维分子生成器现在可以达到接近晶体学的准确性，但它们仍然碎片化分布在不同的任务中。仅使用SMILES输入、两阶段预训练-微调流水线以及一个任务一个模型的做法阻碍了立体化学保真度、任务对齐以及零样本迁移。我们引入了MODA，这是一种统一片段生长、连接设计、骨架跃迁和侧链修饰的贝叶斯掩码调度器的扩散框架。在训练过程中，连续的空间片段被遮盖并在一次通过中去噪，从而使模型能够在不同任务中学习共享的几何和化学先验知识。多任务训练产生了一个超越六个扩散基线和三种训练范式的通用骨干，其在子结构、化学性质、相互作用和几何形状方面表现出色。Model-C减少了配体-蛋白 Clash 和子结构差异，同时保持了Lipinski合规性，而Model-B保持了相似性但新颖性和结合亲和力较低。零样本从头设计和先导优化测试表明，在无需力场精修的情况下稳定产生了负Vina评分和高改进率。这些结果表明，单阶段多任务扩散流程可以替代结构基于的分子设计的两阶段工作流程。 

---
# Combining Pre-Trained Models for Enhanced Feature Representation in Reinforcement Learning 

**Title (ZH)**: 结合预训练模型以增强强化学习中的特征表示 

**Authors**: Elia Piccoli, Malio Li, Giacomo Carfì, Vincenzo Lomonaco, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07197)  

**Abstract**: The recent focus and release of pre-trained models have been a key components to several advancements in many fields (e.g. Natural Language Processing and Computer Vision), as a matter of fact, pre-trained models learn disparate latent embeddings sharing insightful representations. On the other hand, Reinforcement Learning (RL) focuses on maximizing the cumulative reward obtained via agent's interaction with the environment. RL agents do not have any prior knowledge about the world, and they either learn from scratch an end-to-end mapping between the observation and action spaces or, in more recent works, are paired with monolithic and computationally expensive Foundational Models. How to effectively combine and leverage the hidden information of different pre-trained models simultaneously in RL is still an open and understudied question. In this work, we propose Weight Sharing Attention (WSA), a new architecture to combine embeddings of multiple pre-trained models to shape an enriched state representation, balancing the tradeoff between efficiency and performance. We run an extensive comparison between several combination modes showing that WSA obtains comparable performance on multiple Atari games compared to end-to-end models. Furthermore, we study the generalization capabilities of this approach and analyze how scaling the number of models influences agents' performance during and after training. 

**Abstract (ZH)**: 预训练模型的 recent 调整和释放已成为多个领域（例如自然语言处理和计算机视觉）进步的关键组成部分，实际上，预训练模型学习到的异质潜在嵌入共享洞察性表示。另一方面，强化学习 (RL) 关注通过智能体与环境的交互最大化累积奖励。RL 智能体没有任何关于世界的先验知识，它们要么从零开始学习从观察空间到动作空间的端到端映射，要么在最近的研究中与庞大且计算密集的基础模型配对。如何有效地同时结合和利用不同预训练模型的隐藏信息在 RL 中仍是一个开放且未充分研究的问题。在本文中，我们提出了一种新的架构——权重共享注意力 (WSA)，用于结合多个预训练模型的嵌入以形成丰富状态表示，平衡效率与性能之间的tradeoff。我们对多种结合模式进行了详尽比较，结果显示 WSA 在多个 Atari 游戏上的表现与端到端模型相当。此外，我们研究了该方法的泛化能力，并分析了随模型数量增加对智能体训练前后表现的影响。 

---
# Bridging the Last Mile of Prediction: Enhancing Time Series Forecasting with Conditional Guided Flow Matching 

**Title (ZH)**: 弥补预测的最后一公里：基于条件引导流匹配的时间序列 forecasting 提升方法 

**Authors**: Huibo Xu, Runlong Yu, Likang Wu, Xianquan Wang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07192)  

**Abstract**: Diffusion models, a type of generative model, have shown promise in time series forecasting. But they face limitations like rigid source distributions and limited sampling paths, which hinder their performance. Flow matching offers faster generation, higher-quality outputs, and greater flexibility, while also possessing the ability to utilize valuable information from the prediction errors of prior models, which were previously inaccessible yet critically important. To address these challenges and fully unlock the untapped potential of flow matching, we propose Conditional Guided Flow Matching (CGFM). CGFM extends flow matching by incorporating the outputs of an auxiliary model, enabling a previously unattainable capability in the field: learning from the errors of the auxiliary model. For time series forecasting tasks, it integrates historical data as conditions and guidance, constructs two-sided conditional probability paths, and uses a general affine path to expand the space of probability paths, ultimately leading to improved predictions. Extensive experiments show that CGFM consistently enhances and outperforms state-of-the-art models, highlighting its effectiveness in advancing forecasting methods. 

**Abstract (ZH)**: 差分模型作为一种生成模型，在时间序列预测中展现出潜力，但面临着源分布僵化和采样路径有限等局限，这些都妨碍了其性能的提升。流匹配提供了更快的生成速度、更高质量的输出以及更大的灵活性，并且能够利用先前模型预测错误中宝贵的但之前难以触及的信息。为了应对这些挑战并充分释放流匹配的潜力，我们提出了一种条件引导流匹配（CGFM）。CGFM通过引入辅助模型的输出扩展了流匹配方法，使其能够在领域中实现前所未有的学习辅助模型预测错误的能力。对于时间序列预测任务，CGFM将历史数据作为条件和引导，构建双向条件概率路径，并利用通用仿射路径扩展概率路径的空间，从而最终提高预测效果。大量的实验结果表明，CGFM持续提高了并超越了现有最先进模型的表现，突显了其在提升预测方法有效性方面的作用。 

---
# Collective Communication Profiling of Modern-day Machine Learning Workloads 

**Title (ZH)**: 现代机器学习工作负载的集体通信剖析 

**Authors**: Jit Gupta, Andrew Li, Tarun Banka, Ariel Cohen, T. Sridhar, Raj Yavatkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.07117)  

**Abstract**: Machine Learning jobs, carried out on large number of distributed high performance systems, involve periodic communication using operations like AllReduce, AllGather, and Broadcast. These operations may create high bandwidth and bursty traffic patterns, leading to network congestion and packet loss, thus impacting the performance of these jobs. Hence it is imperative to analyze these patterns, which can be helpful in provisioning network resources depending on the type of machine learning workloads. In this poster we carry out extensive analysis of the collective communication behavior seen in a wide variety of models (ex. DeepSeek, GPT, Llama, etc.) To achieve this we instrument Nvidia Collective Communication Library logging functionality for richer context about the collectives and workloads. We adjust configuration parameters that influence collective communication behavior, such as parallelism, number of nodes, and model type. This overview presents and discusses some of the results on the collective communication behavior for the open source DeepSeek V3 inferencing model, which includes operation type and count, transfer sizes per operation, and request size distribution. Our analysis shows that it makes sense to rethink current collective communication frameworks and network topologies so as to accommodate the effect of network anomalies on the mentioned workloads. 

**Abstract (ZH)**: 基于大规模分布式高性能系统的机器学习任务中， Collective 通信行为分析及其对网络资源的需求 

---
# Analysing semantic data storage in Distributed Ledger Technologies for Data Spaces 

**Title (ZH)**: 分析分布式账本技术在数据空间中的语义数据存储 

**Authors**: Juan Cano-Benito, Andrea Cimmino, Sven Hertling, Heiko Paulheim, Raúl García-Castro  

**Link**: [PDF](https://arxiv.org/pdf/2507.07116)  

**Abstract**: Data spaces are emerging as decentralised infrastructures that enable sovereign, secure, and trustworthy data exchange among multiple participants. To achieve semantic interoperability within these environments, the use of semantic web technologies and knowledge graphs has been proposed. Although distributed ledger technologies (DLT) fit as the underlying infrastructure for data spaces, there remains a significant gap in terms of the efficient storage of semantic data on these platforms. This paper presents a systematic evaluation of semantic data storage across different types of DLT (public, private, and hybrid), using a real-world knowledge graph as an experimental basis. The study compares performance, storage efficiency, resource consumption, and the capabilities to update and query semantic data. The results show that private DLTs are the most efficient for storing and managing semantic content, while hybrid DLTs offer a balanced trade-off between public auditability and operational efficiency. This research leads to a discussion on the selection of the most appropriate DLT infrastructure based on the data sovereignty requirements of decentralised data ecosystems. 

**Abstract (ZH)**: 数据空间中的语义数据存储：分布式账本技术的系统评价 

---
# Generative Adversarial Evasion and Out-of-Distribution Detection for UAV Cyber-Attacks 

**Title (ZH)**: 生成式对抗性 evasion 和非分布检测以应对无人机网络攻击 

**Authors**: Deepak Kumar Panda, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21142)  

**Abstract**: The growing integration of UAVs into civilian airspace underscores the need for resilient and intelligent intrusion detection systems (IDS), as traditional anomaly detection methods often fail to identify novel threats. A common approach treats unfamiliar attacks as out-of-distribution (OOD) samples; however, this leaves systems vulnerable when mitigation is inadequate. Moreover, conventional OOD detectors struggle to distinguish stealthy adversarial attacks from genuine OOD events. This paper introduces a conditional generative adversarial network (cGAN)-based framework for crafting stealthy adversarial attacks that evade IDS mechanisms. We first design a robust multi-class IDS classifier trained on benign UAV telemetry and known cyber-attacks, including Denial of Service (DoS), false data injection (FDI), man-in-the-middle (MiTM), and replay attacks. Using this classifier, our cGAN perturbs known attacks to generate adversarial samples that misclassify as benign while retaining statistical resemblance to OOD distributions. These adversarial samples are iteratively refined to achieve high stealth and success rates. To detect such perturbations, we implement a conditional variational autoencoder (CVAE), leveraging negative log-likelihood to separate adversarial inputs from authentic OOD samples. Comparative evaluation shows that CVAE-based regret scores significantly outperform traditional Mahalanobis distance-based detectors in identifying stealthy adversarial threats. Our findings emphasize the importance of advanced probabilistic modeling to strengthen IDS capabilities against adaptive, generative-model-based cyber intrusions. 

**Abstract (ZH)**: 基于cGAN的隐秘对抗攻击制造框架：强化针对生成模型导向的网络入侵的 IDS 能力 

---
# A Comprehensive Survey on Deep Learning Solutions for 3D Flood Mapping 

**Title (ZH)**: 深度学习解决方案综述：三维洪水 mapping 

**Authors**: Wenfeng Jia, Bin Liang, Yuxi Liu, Muhammad Arif Khan, Lihong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13201)  

**Abstract**: Flooding remains a major global challenge, worsened by climate change and urbanization, demanding advanced solutions for effective disaster management. While traditional 2D flood mapping techniques provide limited insights, 3D flood mapping, powered by deep learning (DL), offers enhanced capabilities by integrating flood extent and depth. This paper presents a comprehensive survey of deep learning-based 3D flood mapping, emphasizing its advancements over 2D maps by integrating flood extent and depth for effective disaster management and urban planning. The survey categorizes deep learning techniques into task decomposition and end-to-end approaches, applicable to both static and dynamic flood features. We compare key DL architectures, highlighting their respective roles in enhancing prediction accuracy and computational efficiency. Additionally, this work explores diverse data sources such as digital elevation models, satellite imagery, rainfall, and simulated data, outlining their roles in 3D flood mapping. The applications reviewed range from real-time flood prediction to long-term urban planning and risk assessment. However, significant challenges persist, including data scarcity, model interpretability, and integration with traditional hydrodynamic models. This survey concludes by suggesting future directions to address these limitations, focusing on enhanced datasets, improved models, and policy implications for flood management. This survey aims to guide researchers and practitioners in leveraging DL techniques for more robust and reliable 3D flood mapping, fostering improved flood management strategies. 

**Abstract (ZH)**: 深学习驱动的3D洪水mapping综述：提高灾害管理和城市规划的有效性 

---
