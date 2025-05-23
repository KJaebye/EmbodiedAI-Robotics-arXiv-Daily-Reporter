# Dispersion is (Almost) Optimal under (A)synchrony 

**Title (ZH)**: 异步环境下分散式算法几乎最优 

**Authors**: Ajay D. Kshemkalyani, Manish Kumar, Anisur Rahaman Molla, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16216)  

**Abstract**: The dispersion problem has received much attention recently in the distributed computing literature. In this problem, $k\leq n$ agents placed initially arbitrarily on the nodes of an $n$-node, $m$-edge anonymous graph of maximum degree $\Delta$ have to reposition autonomously to reach a configuration in which each agent is on a distinct node of the graph. Dispersion is interesting as well as important due to its connections to many fundamental coordination problems by mobile agents on graphs, such as exploration, scattering, load balancing, relocation of self-driven electric cars (robots) to recharge stations (nodes), etc. The objective has been to provide a solution that optimizes simultaneously time and memory complexities. There exist graphs for which the lower bound on time complexity is $\Omega(k)$. Memory complexity is $\Omega(\log k)$ per agent independent of graph topology. The state-of-the-art algorithms have (i) time complexity $O(k\log^2k)$ and memory complexity $O(\log(k+\Delta))$ under the synchronous setting [DISC'24] and (ii) time complexity $O(\min\{m,k\Delta\})$ and memory complexity $O(\log(k+\Delta))$ under the asynchronous setting [OPODIS'21]. In this paper, we improve substantially on this state-of-the-art. Under the synchronous setting as in [DISC'24], we present the first optimal $O(k)$ time algorithm keeping memory complexity $O(\log (k+\Delta))$. Under the asynchronous setting as in [OPODIS'21], we present the first algorithm with time complexity $O(k\log k)$ keeping memory complexity $O(\log (k+\Delta))$, which is time-optimal within an $O(\log k)$ factor despite asynchrony. Both results were obtained through novel techniques to quickly find empty nodes to settle agents, which may be of independent interest. 

**Abstract (ZH)**: 分布式计算文献中最近对分散问题给予了广泛关注。在该问题中，初始时有$k \leq n$个代理随机放置在具有$n$个节点、$m$条边且最大度为$\Delta$的匿名图的节点上，它们需要自主重新定位，以达到每个代理占据图的一个不同节点的配置。分散问题因其与基于图上移动代理的许多基本协调问题（如探索、散播、负载均衡、自主电动汽车（机器人）移至充电站（节点）的再定位等）的联系而有趣且重要。目标是提供一个同时优化时间复杂度和空间复杂度的解决方案。已知存在图形使得时间复杂度的下界为$\Omega(k)$。空间复杂度对每个代理来说是独立于图形拓扑的$\Omega(\log k)$。最新的算法在同步设置下（如DISC'24）时间复杂度为$O(k\log^2 k)$，空间复杂度为$O(\log(k+\Delta))$，在异步设置下（如OPODIS'21）时间复杂度为$O(\min\{m,k\Delta\})$，空间复杂度为$O(\log(k+\Delta))$。本文在同步设置下显著改进了上述最新成果，提出了第一个保持空间复杂度$O(\log (k+\Delta))$的最优$O(k)$时间算法。在异步设置下，我们提出了第一个时间复杂度为$O(k\log k)$且空间复杂度保持在$O(\log (k+\Delta))$的时间最优算法（尽管异步环境下，最优时间复杂度理论上有$O(\log k)$的因子）。这两个结果通过新颖的技术快速找到空闲节点安放代理而获得，这些技术自身可能独立具有重要性。 

---
# PEnGUiN: Partially Equivariant Graph NeUral Networks for Sample Efficient MARL 

**Title (ZH)**: PEnGUiN: 部分等变图神经网络在样本高效多智能体 reinforcement 学习中的应用 

**Authors**: Joshua McClellan, Greyson Brothers, Furong Huang, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15615)  

**Abstract**: Equivariant Graph Neural Networks (EGNNs) have emerged as a promising approach in Multi-Agent Reinforcement Learning (MARL), leveraging symmetry guarantees to greatly improve sample efficiency and generalization. However, real-world environments often exhibit inherent asymmetries arising from factors such as external forces, measurement inaccuracies, or intrinsic system biases. This paper introduces \textit{Partially Equivariant Graph NeUral Networks (PEnGUiN)}, a novel architecture specifically designed to address these challenges. We formally identify and categorize various types of partial equivariance relevant to MARL, including subgroup equivariance, feature-wise equivariance, regional equivariance, and approximate equivariance. We theoretically demonstrate that PEnGUiN is capable of learning both fully equivariant (EGNN) and non-equivariant (GNN) representations within a unified framework. Through extensive experiments on a range of MARL problems incorporating various asymmetries, we empirically validate the efficacy of PEnGUiN. Our results consistently demonstrate that PEnGUiN outperforms both EGNNs and standard GNNs in asymmetric environments, highlighting their potential to improve the robustness and applicability of graph-based MARL algorithms in real-world scenarios. 

**Abstract (ZH)**: 部分同构图神经网络（PEnGUiN）：一种应对多智能体 reinforcement 学习中固有不对称性的新型架构 

---
# Reinforcement Learning-based Heuristics to Guide Domain-Independent Dynamic Programming 

**Title (ZH)**: 基于强化学习的启发式方法以指导领域无关的动态规划 

**Authors**: Minori Narita, Ryo Kuroiwa, J. Christopher Beck  

**Link**: [PDF](https://arxiv.org/pdf/2503.16371)  

**Abstract**: Domain-Independent Dynamic Programming (DIDP) is a state-space search paradigm based on dynamic programming for combinatorial optimization. In its current implementation, DIDP guides the search using user-defined dual bounds. Reinforcement learning (RL) is increasingly being applied to combinatorial optimization problems and shares several key structures with DP, being represented by the Bellman equation and state-based transition systems. We propose using reinforcement learning to obtain a heuristic function to guide the search in DIDP. We develop two RL-based guidance approaches: value-based guidance using Deep Q-Networks and policy-based guidance using Proximal Policy Optimization. Our experiments indicate that RL-based guidance significantly outperforms standard DIDP and problem-specific greedy heuristics with the same number of node expansions. Further, despite longer node evaluation times, RL guidance achieves better run-time performance than standard DIDP on three of four benchmark domains. 

**Abstract (ZH)**: 基于动态规划的领域独立动态规划 (DIDP) 是一种基于动态规划的组合优化状态空间搜索范式。我们提出使用强化学习来获得一种启发式函数以指导DIDP的搜索。我们开发了两种基于强化学习的指导方法：基于深度Q网络的价值函数指导和基于接近策略优化的策略指导。实验表明，基于强化学习的指导显著优于标准DIDP和相同节点扩展次数的问题特定贪婪启发式方法。此外，尽管节点评估时间较长，但基于强化学习的指导在三个基准领域中仍能实现更好的运行时性能。 

---
# Enhancing Software Quality Assurance with an Adaptive Differential Evolution based Quantum Variational Autoencoder-Transformer Model 

**Title (ZH)**: 基于自适应差分进化量子变分自编码-变换模型的软件质量 assurance 提升方法 

**Authors**: Seshu Babu Barma, Mohanakrishnan Hariharan, Satish Arvapalli  

**Link**: [PDF](https://arxiv.org/pdf/2503.16335)  

**Abstract**: An AI-powered quality engineering platform uses artificial intelligence to boost software quality assessments through automated defect prediction and optimized performance alongside improved feature extraction. Existing models result in difficulties addressing noisy data types together with imbalances, pattern recognition complexities, ineffective feature extraction, and generalization weaknesses. To overcome those existing challenges in this research, we develop a new model Adaptive Differential Evolution based Quantum Variational Autoencoder-Transformer Model (ADE-QVAET), that combines a Quantum Variational Autoencoder-Transformer (QVAET) to obtain high-dimensional latent features and maintain sequential dependencies together with contextual relationships, resulting in superior defect prediction accuracy. Adaptive Differential Evolution (ADE) Optimization utilizes an adaptive parameter tuning method that enhances model convergence and predictive performance. ADE-QVAET integrates advanced AI techniques to create a robust solution for scalable and accurate software defect prediction that represents a top-level AI-driven technology for quality engineering applications. The proposed ADE-QVAET model attains high accuracy, precision, recall, and f1-score during the training percentage (TP) 90 of 98.08%, 92.45%, 94.67%, and 98.12%. 

**Abstract (ZH)**: 一种基于AI的质量工程平台通过自动缺陷预测和优化性能来提升软件质量评估，同时改进特征提取。现有模型难以解决噪声数据类型、数据不平衡、模式识别复杂性、无效特征提取和泛化能力弱等问题。为克服这些挑战，我们开发了一种新的模型——自适应差分进化基于量子变分自编码器-转换器模型（ADE-QVAET），该模型结合了量子变分自编码器-转换器（QVAET）以获取高维潜在特征并保持顺序依赖性和上下文关系，从而实现卓越的缺陷预测准确性。自适应差分进化（ADE）优化利用自适应参数调优方法，增强模型收敛性和预测性能。ADE-QVAET结合先进的AI技术，为可扩展和准确的软件缺陷预测提供了稳健的解决方案，并代表了质量工程应用顶级的AI驱动技术。所提出的ADE-QVAET模型在训练百分比（TP）90时达到高精度、精准度、召回率和F1分数分别为98.08%、92.45%、94.67%和98.12%。 

---
# Speeding up design and making to reduce time-to-project and time-to-market: an AI-Enhanced approach in engineering education 

**Title (ZH)**: 加快设计与制造以减少项目时间和市场时间：工程教育中的AI增强方法 

**Authors**: Giovanni Adorni, Daniele Grosso  

**Link**: [PDF](https://arxiv.org/pdf/2503.16307)  

**Abstract**: This paper explores the integration of AI tools, such as ChatGPT and GitHub Copilot, in the Software Architecture for Embedded Systems course. AI-supported workflows enabled students to rapidly prototype complex projects, emphasizing real-world applications like SLAM robotics. Results demon-started enhanced problem-solving, faster development, and more sophisticated outcomes, with AI augmenting but not replacing human decision-making. 

**Abstract (ZH)**: 本文探讨了在嵌入式系统课程中整合AI工具（如ChatGPT和GitHub Copilot）的方法。AI支持的工作流程使学生能够快速原型设计复杂项目，强调了诸如SLAM机器人等实际应用。结果显示，这种做法提高了问题解决能力、加快了开发速度，并产生了更为复杂的结果，AI辅助但并未取代人类决策。 

---
# Logic Explanation of AI Classifiers by Categorical Explaining Functors 

**Title (ZH)**: 由范畴解释函子提供的AI分类器逻辑解释 

**Authors**: Stefano Fioravanti, Francesco Giannini, Paolo Frazzetto, Fabio Zanasi, Pietro Barbiero  

**Link**: [PDF](https://arxiv.org/pdf/2503.16203)  

**Abstract**: The most common methods in explainable artificial intelligence are post-hoc techniques which identify the most relevant features used by pretrained opaque models. Some of the most advanced post hoc methods can generate explanations that account for the mutual interactions of input features in the form of logic rules. However, these methods frequently fail to guarantee the consistency of the extracted explanations with the model's underlying reasoning. To bridge this gap, we propose a theoretically grounded approach to ensure coherence and fidelity of the extracted explanations, moving beyond the limitations of current heuristic-based approaches. To this end, drawing from category theory, we introduce an explaining functor which structurally preserves logical entailment between the explanation and the opaque model's reasoning. As a proof of concept, we validate the proposed theoretical constructions on a synthetic benchmark verifying how the proposed approach significantly mitigates the generation of contradictory or unfaithful explanations. 

**Abstract (ZH)**: 可解释人工智能中最常见的方法是后 hoc 技术，这些技术识别预训练的不透明模型中最重要的特征。一些最先进的后 hoc 方法可以通过逻辑规则的形式生成解释输入特征相互作用的解释。然而，这些方法常常无法保证提取的解释与模型内部推理的一致性和忠实性。为了解决这一问题，我们提出一种理论支持的方法来确保提取解释的连贯性和忠实性，超越当前基于启发式的方法的限制。为此，我们借鉴范畴论，引入了一个解释泛函，结构上保持解释与不透明模型推理之间的逻辑蕴含。作为概念验证，我们在一个合成基准上验证了所提出理论构造的有效性，证明了该方法显著减少了生成矛盾或不忠实解释的情况。 

---
# Beyond Local Selection: Global Cut Selection for Enhanced Mixed-Integer Programming 

**Title (ZH)**: 超越局部选择：全局割选择以增强混合整数规划 

**Authors**: Shuli Zeng, Sijia Zhang, Shaoang Li, Feng Wu, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15847)  

**Abstract**: In mixed-integer programming (MIP) solvers, cutting planes are essential for Branch-and-Cut (B&C) algorithms as they reduce the search space and accelerate the solving process. Traditional methods rely on hard-coded heuristics for cut plane selection but fail to leverage problem-specific structural features. Recent machine learning approaches use neural networks for cut selection but focus narrowly on the efficiency of single-node within the B&C algorithm, without considering the broader contextual information. To address this, we propose Global Cut Selection (GCS), which uses a bipartite graph to represent the search tree and combines graph neural networks with reinforcement learning to develop cut selection strategies. Unlike prior methods, GCS applies cutting planes across all nodes, incorporating richer contextual information. Experiments show GCS significantly improves solving efficiency for synthetic and large-scale real-world MIPs compared to traditional and learning-based methods. 

**Abstract (ZH)**: 全局割选择（GCS）：基于图神经网络和强化学习的割选择方法 

---
# Ranking Counterfactual Explanations 

**Title (ZH)**: -counterfactual 解释的排名 

**Authors**: Suryani Lim, Henri Prade, Gilles Richard  

**Link**: [PDF](https://arxiv.org/pdf/2503.15817)  

**Abstract**: AI-driven outcomes can be challenging for end-users to understand. Explanations can address two key questions: "Why this outcome?" (factual) and "Why not another?" (counterfactual). While substantial efforts have been made to formalize factual explanations, a precise and comprehensive study of counterfactual explanations is still lacking. This paper proposes a formal definition of counterfactual explanations, proving some properties they satisfy, and examining the relationship with factual explanations. Given that multiple counterfactual explanations generally exist for a specific case, we also introduce a rigorous method to rank these counterfactual explanations, going beyond a simple minimality condition, and to identify the optimal ones. Our experiments with 12 real-world datasets highlight that, in most cases, a single optimal counterfactual explanation emerges. We also demonstrate, via three metrics, that the selected optimal explanation exhibits higher representativeness and can explain a broader range of elements than a random minimal counterfactual. This result highlights the effectiveness of our approach in identifying more robust and comprehensive counterfactual explanations. 

**Abstract (ZH)**: AI驱动的成果对终端用户来说可能难以理解。解释可以解决两个关键问题：“为什么是这个结果？”（事实性）和“为什么不是另一个？”（反事实性）。尽管在形式化事实性解释方面已做出了大量努力，但对反事实性解释的精确和全面研究仍显不足。本文提出了反事实解释的形式化定义，证明了它们满足的一些性质，并探讨了它们与事实性解释的关系。由于特定情况下通常存在多个反事实解释，我们还介绍了一种严谨的方法来对这些反事实解释进行排序，超越了简单的最小性条件，以识别最优的解释。我们的实验结果表明，在大多数情况下，会涌现出一个最优的反事实解释。我们还通过三个指标展示了所选的最优解释具有更高的代表性和可以解释更广泛的元素，不同于随机的最小反事实。这一结果突显了我们方法在识别更稳健和全面的反事实解释方面的有效性。 

---
# ECLAIR: Enhanced Clarification for Interactive Responses 

**Title (ZH)**: ECLAIR: 增强的交互式响应澄清方法 

**Authors**: John Murzaku, Zifan Liu, Md Mehrab Tanjim, Vaishnavi Muppala, Xiang Chen, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15739)  

**Abstract**: We present ECLAIR (Enhanced CLArification for Interactive Responses), a novel unified and end-to-end framework for interactive disambiguation in enterprise AI assistants. ECLAIR generates clarification questions for ambiguous user queries and resolves ambiguity based on the user's this http URL introduce a generalized architecture capable of integrating ambiguity information from multiple downstream agents, enhancing context-awareness in resolving ambiguities and allowing enterprise specific definition of agents. We further define agents within our system that provide domain-specific grounding information. We conduct experiments comparing ECLAIR to few-shot prompting techniques and demonstrate ECLAIR's superior performance in clarification question generation and ambiguity resolution. 

**Abstract (ZH)**: ECLAIR：增强的交互澄清框架以实现企业AI助理的交互去模糊处理 

---
# Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment 

**Title (ZH)**: 努力图谱：量化AI使用风险以评估漏洞 

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß  

**Link**: [PDF](https://arxiv.org/pdf/2503.16392)  

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.
Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.
To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work. 

**Abstract (ZH)**: 基于AI软件的广泛可用性，利用其高度自动化和复杂模式识别能力的风险可能显著增加。使用AI进行攻击而非AI资产的攻击行为被称为进攻性AI。

当前的研究探索了进攻性AI的使用方式及其使用方式的分类方法。同时，组织内部基于AI的资产的威胁建模方法也在逐步发展。然而，仍存在一些需要填补的空白。首先，需要量化构成AI威胁的因素。其次，需要创建分析AI攻击风险的威胁模型，以评估所有组织资产的漏洞。特别是在云环境中，这种需求尤为迫切和具有挑战性，因为复杂的基础设施和访问控制环境普遍存在。能够量化并进一步分析进攻性AI所带来的威胁，使分析师能够对漏洞进行排名并优先实施主动防御措施。

为了解决这些空白，本文介绍了努力图（Graph of Effort）这一直观、灵活且有效的威胁建模方法，用于分析攻击者利用进攻性AI进行漏洞利用所需的努力。虽然威胁模型功能强大并提供了宝贵的支撑，但其设计选择在未来工作中的需要进一步的实证验证。 

---
# Neural Networks: According to the Principles of Grassmann Algebra 

**Title (ZH)**: 神经网络：根据 Grassmann 代数原理 

**Authors**: Z. Zarezadeh, N. Zarezadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16364)  

**Abstract**: In this paper, we explore the algebra of quantum idempotents and the quantization of fermions which gives rise to a Hilbert space equal to the Grassmann algebra associated with the Lie algebra. Since idempotents carry representations of the algebra under consideration, they form algebraic varieties and smooth manifolds in the natural topology. In addition to the motivation of linking up mathematical physics with machine learning, it is also shown that by using idempotents and invariant subspace of the corresponding algebras, these representations encode and perhaps provide a probabilistic interpretation of reasoning and relational paths in geometrical terms. 

**Abstract (ZH)**: 本文探索量子幂等元的代数及其所对应的张量规范化旋子，这产生了与李代数关联的葛森代数相等的希尔伯特空间。由于幂等元携带了所考虑代数的表示，它们在自然拓扑下形成了代数簇和光滑流形。除了将数学物理与机器学习联系起来的动力外，还展示了通过使用幂等元和相应代数的不变子空间，这些表示编码了基于几何术语的推理和关系路径的可能概率解释。 

---
# HiQ-Lip: The First Quantum-Classical Hierarchical Method for Global Lipschitz Constant Estimation of ReLU Networks 

**Title (ZH)**: HiQ-Lip: 全局ReLU网络Lipschitz常数估算的第一种量子-古典分级方法 

**Authors**: Haoqi He, Yan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16342)  

**Abstract**: Estimating the global Lipschitz constant of neural networks is crucial for understanding and improving their robustness and generalization capabilities. However, precise calculations are NP-hard, and current semidefinite programming (SDP) methods face challenges such as high memory usage and slow processing speeds. In this paper, we propose \textbf{HiQ-Lip}, a hybrid quantum-classical hierarchical method that leverages Coherent Ising Machines (CIMs) to estimate the global Lipschitz constant. We tackle the estimation by converting it into a Quadratic Unconstrained Binary Optimization (QUBO) problem and implement a multilevel graph coarsening and refinement strategy to adapt to the constraints of contemporary quantum hardware. Our experimental evaluations on fully connected neural networks demonstrate that HiQ-Lip not only provides estimates comparable to state-of-the-art methods but also significantly accelerates the computation process. In specific tests involving two-layer neural networks with 256 hidden neurons, HiQ-Lip doubles the solving speed and offers more accurate upper bounds than the existing best method, LiPopt. These findings highlight the promising utility of small-scale quantum devices in advancing the estimation of neural network robustness. 

**Abstract (ZH)**: 基于混合量子-经典层次方法的HiQ-Lip：利用相干伊辛机估计神经网络的全局利普希茨常数 

---
# Knowledge-guided machine learning model with soil moisture for corn yield prediction under drought conditions 

**Title (ZH)**: 基于土壤湿度的知识引导机器学习模型在干旱条件下玉米产量预测 

**Authors**: Xiaoyu Wang, Yijia Xu, Jingyi Huang, Zhengwei Yang, Zhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16328)  

**Abstract**: Remote sensing (RS) techniques, by enabling non-contact acquisition of extensive ground observations, have become a valuable tool for corn yield prediction. Traditional process-based (PB) models are limited by fixed input features and struggle to incorporate large volumes of RS data. In contrast, machine learning (ML) models are often criticized for being ``black boxes'' with limited interpretability. To address these limitations, we used Knowledge-Guided Machine Learning (KGML), which combined the strengths of both approaches and fully used RS data. However, previous KGML methods overlooked the crucial role of soil moisture in plant growth. To bridge this gap, we proposed the Knowledge-Guided Machine Learning with Soil Moisture (KGML-SM) framework, using soil moisture as an intermediate variable to emphasize its key role in plant development. Additionally, based on the prior knowledge that the model may overestimate under drought conditions, we designed a drought-aware loss function that penalizes predicted yield in drought-affected areas. Our experiments showed that the KGML-SM model outperformed other ML models. Finally, we explored the relationships between drought, soil moisture, and corn yield prediction, assessing the importance of various features and analyzing how soil moisture impacts corn yield predictions across different regions and time periods. 

**Abstract (ZH)**: 基于土壤湿度的指导机器学习框架（KGML-SM）在玉米产量预测中的应用 

---
# Diffusion-augmented Graph Contrastive Learning for Collaborative Filter 

**Title (ZH)**: 基于扩散增强图对比学习的协作过滤 

**Authors**: Fan Huang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16290)  

**Abstract**: Graph-based collaborative filtering has been established as a prominent approach in recommendation systems, leveraging the inherent graph topology of user-item interactions to model high-order connectivity patterns and enhance recommendation performance. Recent advances in Graph Contrastive Learning (GCL) have demonstrated promising potential to alleviate data sparsity issues by improving representation learning through contrastive view generation and mutual information maximization. However, existing approaches lack effective data augmentation strategies. Structural augmentation risks distorting fundamental graph topology, while feature-level perturbation techniques predominantly employ uniform noise scales that fail to account for node-specific characteristics. To solve these challenges, we propose Diffusion-augmented Contrastive Learning (DGCL), an innovative framework that integrates diffusion models with contrastive learning for enhanced collaborative filtering. Our approach employs a diffusion process that learns node-specific Gaussian distributions of representations, thereby generating semantically consistent yet diversified contrastive views through reverse diffusion sampling. DGCL facilitates adaptive data augmentation based on reconstructed representations, considering both semantic coherence and node-specific features. In addition, it explores unrepresented regions of the latent sparse feature space, thereby enriching the diversity of contrastive views. Extensive experimental results demonstrate the effectiveness of DGCL on three public datasets. 

**Abstract (ZH)**: 基于图的协同过滤已确立为推荐系统中的突出方法，通过利用用户项交互的内在图拓扑来建模高阶连接模式并提升推荐性能。最近在图对比学习（GCL）方面的进展展示了通过对比视图生成和最大互信息来改进表示学习以缓解数据稀疏问题的潜在优势。然而，现有方法缺乏有效的数据增强策略。结构增强有损地扭曲基本图拓扑，而特征层面的扰动技术主要采用均匀噪声尺度，未能考虑节点特定特性。为了解决这些挑战，我们提出了扩散增强对比学习（DGCL）框架，该框架将扩散模型与对比学习相结合以增强协同过滤。我们的方法通过学习节点特定的高斯分布进行扩散过程，从而通过逆向扩散采样生成语义一致但多样的对比视图。DGCL根据重建后的表示实现自适应数据增强，同时考虑语义连贯性和节点特定特性。此外，它探索了潜在稀疏特征空间的未表示区域，从而丰富了对比视图的多样性。广泛的实验结果在三个公开数据集上验证了DGCL的有效性。 

---
# AI Agents in Cryptoland: Practical Attacks and No Silver Bullet 

**Title (ZH)**: AI代理在加密世界中的攻防：没有万能灵药 

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16248)  

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness, yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating. Our findings indicate that prompt-based defenses are insufficient, as malicious inputs can corrupt an agent's stored context, creating cascading vulnerabilities across interactions and platforms. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible. 

**Abstract (ZH)**: AI代理与Web3生态系统的集成利用了其互补潜力以增强自主性和开放性，但同时也引入了未充分探索的安全风险，因为这些代理会动态与金融协议和不可变智能合约交互。本文探讨了在实际场景中 adversarial 攻击威胁下基于区块链的金融生态系统中AI代理的脆弱性。我们引入了上下文操纵的概念——这是一个全面的攻击向量，利用未受保护的上下文面，包括输入通道、内存模块和外部数据源。通过对ElizaOS的实证分析——这是一个用于自动化Web3操作的去中心化AI代理框架——我们展示了攻击者如何通过注入恶意指令到提示或历史交互记录中操纵上下文，导致意外资产转移和协议违规，这些都可能是财政灾难性的。我们的研究发现表明，基于提示的防御措施是不足的，因为恶意输入可以破坏代理存储的上下文，从而在交互和平台之间引发级联漏洞。本研究强调了开发既安全又具有信托责任的AI代理的迫切需求。 

---
# Flight Testing an Optionally Piloted Aircraft: a Case Study on Trust Dynamics in Human-Autonomy Teaming 

**Title (ZH)**: 可选有人驾驶飞行测试：人机团队信任动态案例研究 

**Authors**: Jeremy C.-H. Wang, Ming Hou, David Dunwoody, Marko Ilievski, Justin Tomasi, Edward Chao, Carl Pigeon  

**Link**: [PDF](https://arxiv.org/pdf/2503.16227)  

**Abstract**: This paper examines how trust is formed, maintained, or diminished over time in the context of human-autonomy teaming with an optionally piloted aircraft. Whereas traditional factor-based trust models offer a static representation of human confidence in technology, here we discuss how variations in the underlying factors lead to variations in trust, trust thresholds, and human behaviours. Over 200 hours of flight test data collected over a multi-year test campaign from 2021 to 2023 were reviewed. The dispositional-situational-learned, process-performance-purpose, and IMPACTS homeostasis trust models are applied to illuminate trust trends during nominal autonomous flight operations. The results offer promising directions for future studies on trust dynamics and design-for-trust in human-autonomy teaming. 

**Abstract (ZH)**: 本文探讨了在 Optionally Piloted Aircraft 的人类-自主团队环境中，信任是如何随时间形成、维持或减弱的。传统的基于因素的信任模型提供了技术信任的静态表示，而本文讨论了底层因素的变化如何导致信任、信任阈值和人类行为的变化。通过对2021年至2023年多年度试验 campaign 收集的超过200小时飞行测试数据的审查，应用了 dispositional-situational-learned、process-performance-purpose 和 IMPACTS 自稳态信任模型，以阐明名义自主飞行操作期间的信任趋势。研究结果为未来关于信任动态和信任设计的研究提供了有力的方向。 

---
# PromptMobile: Efficient Promptus for Low Bandwidth Mobile Video Streaming 

**Title (ZH)**: PromptMobile: 适用于低带宽移动视频流传输的高效提示方案 

**Authors**: Liming Liu, Jiangkai Wu, Haoyang Wang, Peiheng Wang, Xinggong Zhang, Zongming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16112)  

**Abstract**: Traditional video compression algorithms exhibit significant quality degradation at extremely low bitrates. Promptus emerges as a new paradigm for video streaming, substantially cutting down the bandwidth essential for video streaming. However, Promptus is computationally intensive and can not run in real-time on mobile devices. This paper presents PromptMobile, an efficient acceleration framework tailored for on-device Promptus. Specifically, we propose (1) a two-stage efficient generation framework to reduce computational cost by 8.1x, (2) a fine-grained inter-frame caching to reduce redundant computations by 16.6\%, (3) system-level optimizations to further enhance efficiency. The evaluations demonstrate that compared with the original Promptus, PromptMobile achieves a 13.6x increase in image generation speed. Compared with other streaming methods, PromptMobile achives an average LPIPS improvement of 0.016 (compared with H.265), reducing 60\% of severely distorted frames (compared to VQGAN). 

**Abstract (ZH)**: 传统视频压缩算法在极低比特率下会显著降低视频质量。Promptus作为一种新的视频流媒体 paradigm，极大地减少了视频流媒体所需的带宽。然而，Promptus计算密集型且无法在移动设备上实时运行。本文提出了PromptMobile，一个针对 Promptus 的高效加速框架。具体来说，我们提出了一种两阶段有效生成框架，将计算成本降低8.1倍，一种细粒度的帧间缓存机制，将冗余计算减少16.6%，以及系统级优化进一步提高效率。评估结果显示，与原始的 Promptus 相比，PromptMobile 的图像生成速度提高了13.6倍。与其它流媒体方法相比，PromptMobile 在 LPIPS 上平均改进了0.016（与 H.265 相比），减少了60%严重失真的帧（与 VQGAN 相比）。 

---
# AIMI: Leveraging Future Knowledge and Personalization in Sparse Event Forecasting for Treatment Adherence 

**Title (ZH)**: AIMI：在稀疏事件预测中的未来知识利用和个人化方法以提高治疗依从性 

**Authors**: Abdullah Mamun, Diane J. Cook, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16091)  

**Abstract**: Adherence to prescribed treatments is crucial for individuals with chronic conditions to avoid costly or adverse health outcomes. For certain patient groups, intensive lifestyle interventions are vital for enhancing medication adherence. Accurate forecasting of treatment adherence can open pathways to developing an on-demand intervention tool, enabling timely and personalized support. With the increasing popularity of smartphones and wearables, it is now easier than ever to develop and deploy smart activity monitoring systems. However, effective forecasting systems for treatment adherence based on wearable sensors are still not widely available. We close this gap by proposing Adherence Forecasting and Intervention with Machine Intelligence (AIMI). AIMI is a knowledge-guided adherence forecasting system that leverages smartphone sensors and previous medication history to estimate the likelihood of forgetting to take a prescribed medication. A user study was conducted with 27 participants who took daily medications to manage their cardiovascular diseases. We designed and developed CNN and LSTM-based forecasting models with various combinations of input features and found that LSTM models can forecast medication adherence with an accuracy of 0.932 and an F-1 score of 0.936. Moreover, through a series of ablation studies involving convolutional and recurrent neural network architectures, we demonstrate that leveraging known knowledge about future and personalized training enhances the accuracy of medication adherence forecasting. Code available: this https URL. 

**Abstract (ZH)**: 基于机器智能的依从性预测与干预（AIMI） 

---
# Redefining Toxicity: An Objective and Context-Aware Approach for Stress-Level-Based Detection 

**Title (ZH)**: 重新定义毒性：一种基于压力级别且客观情境感知的检测方法 

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16072)  

**Abstract**: The fundamental problem of toxicity detection lies in the fact that the term "toxicity" is ill-defined. Such uncertainty causes researchers to rely on subjective and vague data during model training, which leads to non-robust and inaccurate results, following the 'garbage in - garbage out' paradigm. This study introduces a novel, objective, and context-aware framework for toxicity detection, leveraging stress levels as a key determinant of toxicity. We propose new definition, metric and training approach as a parts of our framework and demonstrate it's effectiveness using a dataset we collected. 

**Abstract (ZH)**: 毒性的检测基本问题在于“毒性和”的概念界定不清。这种不确定性导致研究人员在模型训练过程中依赖主观和模糊的数据，从而产生不 robust 和不准确的结果，遵循“垃圾进-垃圾出”的原则。本研究提出了一种新颖的、客观的、上下文感知的毒性和检测框架，利用压力水平作为毒性和的关键决定因素。我们提出了一种新的定义、度量标准和训练方法作为该框架的一部分，并通过我们收集的数据集展示了其有效性。 

---
# PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval 

**Title (ZH)**: PromptHash: 基于提示驱动的协作跨模态学习适应性哈希检索 

**Authors**: Qiang Zou, Shuli Cheng, Jiayi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16064)  

**Abstract**: Cross-modal hashing is a promising approach for efficient data retrieval and storage optimization. However, contemporary methods exhibit significant limitations in semantic preservation, contextual integrity, and information redundancy, which constrains retrieval efficacy. We present PromptHash, an innovative framework leveraging affinity prompt-aware collaborative learning for adaptive cross-modal hashing. We propose an end-to-end framework for affinity-prompted collaborative hashing, with the following fundamental technical contributions: (i) a text affinity prompt learning mechanism that preserves contextual information while maintaining parameter efficiency, (ii) an adaptive gated selection fusion architecture that synthesizes State Space Model with Transformer network for precise cross-modal feature integration, and (iii) a prompt affinity alignment strategy that bridges modal heterogeneity through hierarchical contrastive learning. To the best of our knowledge, this study presents the first investigation into affinity prompt awareness within collaborative cross-modal adaptive hash learning, establishing a paradigm for enhanced semantic consistency across modalities. Through comprehensive evaluation on three benchmark multi-label datasets, PromptHash demonstrates substantial performance improvements over existing approaches. Notably, on the NUS-WIDE dataset, our method achieves significant gains of 18.22% and 18.65% in image-to-text and text-to-image retrieval tasks, respectively. The code is publicly available at this https URL. 

**Abstract (ZH)**: 跨模态哈希是高效数据检索和存储优化的一种有前途的方法。然而，当代方法在语义保真度、语境完整性以及信息冗余方面表现出显著的局限性，这限制了检索效果。我们提出了PromptHash，这是一种利用亲和度提示感知协作学习机制的创新框架，用于自适应跨模态哈希。我们提出了一种端到端的亲和度提示协同哈希框架，具有以下基本技术贡献：(i) 一种文本亲和度提示学习机制，能够在保持参数效率的同时保留语境信息，(ii) 一种自适应门控选择融合架构，将状态空间模型与Transformer网络结合以精确合成跨模态特征集成，以及(iii) 一种提示亲和度对齐策略，通过层次对比学习弥合模态异质性。据我们所知，这是首次对协同跨模态自适应哈希学习中的亲和度提示意识进行研究，建立了提升各模态间语义一致性的新范式。通过在三个基准多标签数据集上的全面评估，PromptHash展现了相对于现有方法的显著性能提升。特别是在NUS-WIDE数据集上，我们的方法分别在图像到文本和文本到图像检索任务中取得了18.22%和18.65%的显著提升。代码已在以下网址公开：this https URL。 

---
# Two-stage Incomplete Utterance Rewriting on Editing Operation 

**Title (ZH)**: 两阶段不完整语句编辑重写 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu, Yaxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16063)  

**Abstract**: Previous work on Incomplete Utterance Rewriting (IUR) has primarily focused on generating rewritten utterances based solely on dialogue context, ignoring the widespread phenomenon of coreference and ellipsis in dialogues. To address this issue, we propose a novel framework called TEO (\emph{Two-stage approach on Editing Operation}) for IUR, in which the first stage generates editing operations and the second stage rewrites incomplete utterances utilizing the generated editing operations and the dialogue context. Furthermore, an adversarial perturbation strategy is proposed to mitigate cascading errors and exposure bias caused by the inconsistency between training and inference in the second stage. Experimental results on three IUR datasets show that our TEO outperforms the SOTA models significantly. 

**Abstract (ZH)**: 基于两阶段编辑操作的对话中不完整表述重写框架TEO 

---
# Expert Race: A Flexible Routing Strategy for Scaling Diffusion Transformer with Mixture of Experts 

**Title (ZH)**: 专家赛跑：一种用于扩展扩散变换器的混合专家灵活路由策略 

**Authors**: Yike Yuan, Ziyu Wang, Zihao Huang, Defa Zhu, Xun Zhou, Jingyi Yu, Qiyang Min  

**Link**: [PDF](https://arxiv.org/pdf/2503.16057)  

**Abstract**: Diffusion models have emerged as mainstream framework in visual generation. Building upon this success, the integration of Mixture of Experts (MoE) methods has shown promise in enhancing model scalability and performance. In this paper, we introduce Race-DiT, a novel MoE model for diffusion transformers with a flexible routing strategy, Expert Race. By allowing tokens and experts to compete together and select the top candidates, the model learns to dynamically assign experts to critical tokens. Additionally, we propose per-layer regularization to address challenges in shallow layer learning, and router similarity loss to prevent mode collapse, ensuring better expert utilization. Extensive experiments on ImageNet validate the effectiveness of our approach, showcasing significant performance gains while promising scaling properties. 

**Abstract (ZH)**: 基于专家混合的方法在增强扩散变压器模型的可扩展性和性能方面的研究：Race-DiT模型及其灵活路由策略 

---
# Temporal-Spatial Attention Network (TSAN) for DoS Attack Detection in Network Traffic 

**Title (ZH)**: 基于时空注意力网络（TSAN）的网络流量DoS攻击检测 

**Authors**: Bisola Faith Kayode, Akinyemi Sadeeq Akintola, Oluwole Fagbohun, Egonna Anaesiuba-Bristol, Onyekachukwu Ojumah, Oluwagbade Odimayo, Toyese Oloyede, Aniema Inyang, Teslim Kazeem, Habeeb Alli, Udodirim Ibem Offia, Prisca Chinazor Amajuoyi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16047)  

**Abstract**: Denial-of-Service (DoS) attacks remain a critical threat to network security, disrupting services and causing significant economic losses. Traditional detection methods, including statistical and rule-based models, struggle to adapt to evolving attack patterns. To address this challenge, we propose a novel Temporal-Spatial Attention Network (TSAN) architecture for detecting Denial of Service (DoS) attacks in network traffic. By leveraging both temporal and spatial features of network traffic, our approach captures complex traffic patterns and anomalies that traditional methods might miss. The TSAN model incorporates transformer-based temporal encoding, convolutional spatial encoding, and a cross-attention mechanism to fuse these complementary feature spaces. Additionally, we employ multi-task learning with auxiliary tasks to enhance the model's robustness. Experimental results on the NSL-KDD dataset demonstrate that TSAN outperforms state-of-the-art models, achieving superior accuracy, precision, recall, and F1-score while maintaining computational efficiency for real-time deployment. The proposed architecture offers an optimal balance between detection accuracy and computational overhead, making it highly suitable for real-world network security applications. 

**Abstract (ZH)**: 时空注意力网络（TSAN）架构用于检测网络流量中的拒绝服务（DoS）攻击 

---
# Open Science and Artificial Intelligence for supporting the sustainability of the SRC Network: The espSRC case 

**Title (ZH)**: 开放科学与人工智_agent_for支撑SRC网络的可持续性:espSRC案例 

**Authors**: J. Garrido, S. Sánchez-Expósito, A. Ruiz-Falcó, J. Ruedas, M. Á. Mendoza, V. Vázquez, M. Parra, J. Sánchez, I. Labadie, L. Darriba, J. Moldón, M. Rodriguez-Álvarez, J. Díaz, L. Verdes-Montenegro  

**Link**: [PDF](https://arxiv.org/pdf/2503.16045)  

**Abstract**: The SKA Observatory (SKAO), a landmark project in radio astronomy, seeks to address fundamental questions in astronomy. To process its immense data output, approximately 700 PB/year, a global network of SKA Regional Centres (SR-CNet) will provide the infrastructure, tools, computational power needed for scientific analysis and scientific support. The Spanish SRC (espSRC) focuses on ensuring the sustainability of this network by reducing its environmental impact, integrating green practices into data platforms, and developing Open Science technologies to enable reproducible research. This paper discusses and summarizes part of the research and development activities that the team is conducting to reduce the SRC energy consumption at the espSRC and SRCNet. The paper also discusses fundamental research on trusted repositories to support Open Science practices. 

**Abstract (ZH)**: SKA望远镜 observatory（SKAO），一项射电天文学领域的里程碑项目，旨在解答天文学中的基本问题。为了处理其巨大的数据输出，约每年700 PB，SKA区域中心（SKA Regional Centres, SR-CNet）全球网络将提供必要的基础设施、工具和计算能力以支持科学分析和科学支持。西班牙区域中心（Spanish SRC, espSRC）致力于通过减少能源消耗、整合绿色实践于数据平台以及开发开放科学技术来确保该网络的可持续性，以促进可再现研究。本文讨论并总结了团队正在进行的部分研究和开发活动，以降低espSRC和SRCNet的能源消耗。此外，本文还讨论了支持开放科学实践的受信任存储库的基本研究。 

---
# Incomplete Utterance Rewriting with Editing Operation Guidance and Utterance Augmentation 

**Title (ZH)**: 基于编辑操作指导和话语扩增的不完整句子重写 

**Authors**: Zhiyu Cao, Peifeng Li, Yaxin Fan, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16043)  

**Abstract**: Although existing fashionable generation methods on Incomplete Utterance Rewriting (IUR) can generate coherent utterances, they often result in the inclusion of irrelevant and redundant tokens in rewritten utterances due to their inability to focus on critical tokens in dialogue context. Furthermore, the limited size of the training datasets also contributes to the insufficient training of the IUR model. To address the first issue, we propose a multi-task learning framework EO-IUR (Editing Operation-guided Incomplete Utterance Rewriting) that introduces the editing operation labels generated by sequence labeling module to guide generation model to focus on critical tokens. Furthermore, we introduce a token-level heterogeneous graph to represent dialogues. To address the second issue, we propose a two-dimensional utterance augmentation strategy, namely editing operation-based incomplete utterance augmentation and LLM-based historical utterance augmentation. The experimental results on three datasets demonstrate that our EO-IUR outperforms previous state-of-the-art (SOTA) baselines in both open-domain and task-oriented dialogue. The code will be available at this https URL. 

**Abstract (ZH)**: 尽管现有的不完备话语修正方法可以在不完整话语重写（IUR）中生成连贯的话语，但由于它们无法关注对话上下文中的关键词汇，常常导致生成的话语包含无关和冗余的词汇。此外，训练数据集规模有限也导致IUR模型训练不足。为解决第一个问题，我们提出了一种多任务学习框架EO-IUR（编辑操作引导的不完备话语修正），该框架通过引入由序列标注模块生成的编辑操作标签来引导生成模型关注关键词汇。为进一步解决第二个问题，我们提出了两种话语增强策略，即基于编辑操作的不完备话语增强和基于大型语言模型的历史话语增强。在三个数据集上的实验结果表明，我们的EO-IUR在开放式和任务导向对话中均优于先前的最优基线方法。代码将在此链接处提供。 

---
# Denoising-based Contractive Imitation Learning 

**Title (ZH)**: 基于去噪收缩模仿学习 

**Authors**: Macheng Shen, Jishen Peng, Zefang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15918)  

**Abstract**: A fundamental challenge in imitation learning is the \emph{covariate shift} problem. Existing methods to mitigate covariate shift often require additional expert interactions, access to environment dynamics, or complex adversarial training, which may not be practical in real-world applications. In this paper, we propose a simple yet effective method (DeCIL) to mitigate covariate shift by incorporating a denoising mechanism that enhances the contraction properties of the state transition mapping. Our approach involves training two neural networks: a dynamics model ( f ) that predicts the next state from the current state, and a joint state-action denoising policy network ( d ) that refines this state prediction via denoising and outputs the corresponding action. We provide theoretical analysis showing that the denoising network acts as a local contraction mapping, reducing the error propagation of the state transition and improving stability. Our method is straightforward to implement and can be easily integrated with existing imitation learning frameworks without requiring additional expert data or complex modifications to the training procedure. Empirical results demonstrate that our approach effectively improves success rate of various imitation learning tasks under noise perturbation. 

**Abstract (ZH)**: 仿生学习中的一个基本挑战是协变量偏移问题。现有缓解协变量偏移的方法通常需要额外的专家交互、访问环境动力学或复杂的对抗性训练，这些在实际应用中可能并不实用。在本文中，我们提出了一种简单而有效的方法（DeCIL），通过引入一种去噪机制来增强状态转换映射的收缩性质以缓解协变量偏移。我们的方法包括训练两个神经网络：一个动力学模型（f），它根据当前状态预测下一个状态；以及一个联合状态-动作去噪策略网络（d），它通过去噪来细化这个状态预测并输出相应的动作。我们提供了理论分析以证明去噪网络作为局部收缩映射的作用，从而减少状态转换过程中的误差传播并提高稳定性。该方法易于实现，并且可以轻松与现有的仿生学习框架集成，无需额外的专家数据或复杂的训练程序修改。实验证明，该方法在噪声干扰下有效提高了各种仿生学习任务的成功率。 

---
# A multi-model approach using XAI and anomaly detection to predict asteroid hazards 

**Title (ZH)**: 基于XAI和异常检测的多模型方法预测小行星危害 

**Authors**: Amit Kumar Mondal, Nafisha Aslam, Prasenjit Maji, Hemanta Kumar Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2503.15901)  

**Abstract**: The potential for catastrophic collision makes near-Earth asteroids (NEAs) a serious concern. Planetary defense depends on accurately classifying potentially hazardous asteroids (PHAs), however the complexity of the data hampers conventional techniques. This work offers a sophisticated method for accurately predicting hazards by combining machine learning, deep learning, explainable AI (XAI), and anomaly detection. Our approach extracts essential parameters like size, velocity, and trajectory from historical and real-time asteroid data. A hybrid algorithm improves prediction accuracy by combining several cutting-edge models. A forecasting module predicts future asteroid behavior, and Monte Carlo simulations evaluate the likelihood of collisions. Timely mitigation is made possible by a real-time alarm system that notifies worldwide monitoring stations. This technique enhances planetary defense efforts by combining real-time alarms with sophisticated predictive modeling. 

**Abstract (ZH)**: 近地小行星（NEAs）潜在的 catastrophic 碰撞使其成为严重关切对象。行星防御依赖于准确分类潜在危险小行星（PHAs），然而数据的复杂性阻碍了传统技术。本工作提出了一种结合机器学习、深度学习、可解释AI（XAI）和异常检测的复杂方法，以精确预测潜在风险。该方法从历史和实时小行星数据中提取关键参数，如大小、速度和轨道。混合算法通过结合多种先进模型提高预测准确性。预报模块预测未来小行星行为，蒙特卡洛模拟评估碰撞的可能性。实时警报系统使得及时的缓解措施成为可能，通知全球监测站。该技术通过结合实时警报和复杂的预测建模增强了行星防御努力。 

---
# Time After Time: Deep-Q Effect Estimation for Interventions on When and What to do 

**Title (ZH)**: 反复的时间：基于深度Q效应估计的何时以及做什么的干预研究 

**Authors**: Yoav Wald, Mark Goldstein, Yonathan Efroni, Wouter A.C. van Amsterdam, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2503.15890)  

**Abstract**: Problems in fields such as healthcare, robotics, and finance requires reasoning about the value both of what decision or action to take and when to take it. The prevailing hope is that artificial intelligence will support such decisions by estimating the causal effect of policies such as how to treat patients or how to allocate resources over time. However, existing methods for estimating the effect of a policy struggle with \emph{irregular time}. They either discretize time, or disregard the effect of timing policies. We present a new deep-Q algorithm that estimates the effect of both when and what to do called Earliest Disagreement Q-Evaluation (EDQ). EDQ makes use of recursion for the Q-function that is compatible with flexible sequence models, such as transformers. EDQ provides accurate estimates under standard assumptions. We validate the approach through experiments on survival time and tumor growth tasks. 

**Abstract (ZH)**: 医疗、机器人技术及金融等领域中存在的问题要求对何时采取何种决策及其效果进行推理。现有的希望是，人工智能能通过估计政策（如如何治疗患者或如何在时间上分配资源）的效果来支持这样的决策。然而，现有的政策效果估计方法在处理不规则时间时存在困难。它们要么离散化时间，要么忽视政策时间效应。我们提出了一种新的深度Q算法，称为最早分歧Q评估（EDQ），它能同时估计何时及采取何种行动的效果。EDQ利用递归构建Q函数，兼容灵活的序列模型（如变换器）。在标准假设下，EDQ能提供准确的估计。我们通过生存时间和肿瘤生长任务的实验验证了该方法。 

---
# LeanTTA: A Backpropagation-Free and Stateless Approach to Quantized Test-Time Adaptation on Edge Devices 

**Title (ZH)**: LeanTTA: 一种无反向传播且无状态的边缘设备上量化测试时适应方法 

**Authors**: Cynthia Dong, Hong Jia, Young D. Kwon, Georgios Rizos, Cecilia Mascolo  

**Link**: [PDF](https://arxiv.org/pdf/2503.15889)  

**Abstract**: While there are many advantages to deploying machine learning models on edge devices, the resource constraints of mobile platforms, the dynamic nature of the environment, and differences between the distribution of training versus in-the-wild data make such deployments challenging. Current test-time adaptation methods are often memory-intensive and not designed to be quantization-compatible or deployed on low-resource devices. To address these challenges, we present LeanTTA, a novel backpropagation-free and stateless framework for quantized test-time adaptation tailored to edge devices. Our approach minimizes computational costs by dynamically updating normalization statistics without backpropagation, which frees LeanTTA from the common pitfall of relying on large batches and historical data, making our method robust to realistic deployment scenarios. Our approach is the first to enable further computational gains by combining partial adaptation with quantized module fusion. We validate our framework across sensor modalities, demonstrating significant improvements over state-of-the-art TTA methods, including a 15.7% error reduction, peak memory usage of only 11.2MB for ResNet18, and fast adaptation within an order-of-magnitude of normal inference speeds on-device. LeanTTA provides a robust solution for achieving the right trade offs between accuracy and system efficiency in edge deployments, addressing the unique challenges posed by limited data and varied operational conditions. 

**Abstract (ZH)**: LeanTTA：面向边缘设备的无回传状态less量化测试时自适应框架 

---
# TruthLens: Explainable DeepFake Detection for Face Manipulated and Fully Synthetic Data 

**Title (ZH)**: TruthLens: 可解释的深度假新闻检测方法及其对人脸篡改与全合成数据的应用 

**Authors**: Rohit Kundu, Athula Balachandran, Amit K. Roy-Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.15867)  

**Abstract**: Detecting DeepFakes has become a crucial research area as the widespread use of AI image generators enables the effortless creation of face-manipulated and fully synthetic content, yet existing methods are often limited to binary classification (real vs. fake) and lack interpretability. To address these challenges, we propose TruthLens, a novel and highly generalizable framework for DeepFake detection that not only determines whether an image is real or fake but also provides detailed textual reasoning for its predictions. Unlike traditional methods, TruthLens effectively handles both face-manipulated DeepFakes and fully AI-generated content while addressing fine-grained queries such as "Does the eyes/nose/mouth look real or fake?"
The architecture of TruthLens combines the global contextual understanding of multimodal large language models like PaliGemma2 with the localized feature extraction capabilities of vision-only models like DINOv2. This hybrid design leverages the complementary strengths of both models, enabling robust detection of subtle manipulations while maintaining interpretability. Extensive experiments on diverse datasets demonstrate that TruthLens outperforms state-of-the-art methods in detection accuracy (by 2-14%) and explainability, in both in-domain and cross-data settings, generalizing effectively across traditional and emerging manipulation techniques. 

**Abstract (ZH)**: 检测深度伪造已成为一个至关重要的研究领域，随着AI图像生成器的广泛使用，使得面部操纵和完全合成的内容得以轻松创建，但现有方法往往仅限于二元分类（真实 vs. 伪造）且缺乏可解释性。为应对这些挑战，我们提出TruthLens，这是一个新颖且高度通用的深度伪造检测框架，不仅确定图像是真实还是伪造的，还提供详细的文本推理解释其预测。与传统方法不同，TruthLens 能够有效处理面部操纵的深度伪造和完全由AI生成的内容，同时解决细粒度问题，如“眼睛/鼻子/嘴巴看起来是真实还是伪造的？”。

TruthLens的架构结合了多模态大型语言模型PaliGemma2的全局上下文理解和仅视觉模型DINOv2的局部特征提取能力。这种混合设计利用了两种模型互补的优势，实现对细微篡改的 robust 检测，同时保持可解释性。在多种数据集上的广泛实验表明，TruthLens 在检测准确性和可解释性方面均优于现有最先进的方法（提高2-14%），并在领域内外数据集上表现出了有效的泛化能力，能够应对传统和新兴篡改技术。 

---
# Active management of battery degradation in wireless sensor network using deep reinforcement learning for group battery replacement 

**Title (ZH)**: 使用深度强化学习进行群体电池更换的无线传感器网络中电池退化主动管理 

**Authors**: Jong-Hyun Jeonga, Hongki Jo, Qiang Zhou, Tahsin Afroz Hoque Nishat, Lang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15865)  

**Abstract**: Wireless sensor networks (WSNs) have become a promising solution for structural health monitoring (SHM), especially in hard-to-reach or remote locations. Battery-powered WSNs offer various advantages over wired systems, however limited battery life has always been one of the biggest obstacles in practical use of the WSNs, regardless of energy harvesting methods. While various methods have been studied for battery health management, existing methods exclusively aim to extend lifetime of individual batteries, lacking a system level view. A consequence of applying such methods is that batteries in a WSN tend to fail at different times, posing significant difficulty on planning and scheduling of battery replacement trip. This study investigate a deep reinforcement learning (DRL) method for active battery degradation management by optimizing duty cycle of WSNs at the system level. This active management strategy effectively reduces earlier failure of battery individuals which enable group replacement without sacrificing WSN performances. A simulated environment based on a real-world WSN setup was developed to train a DRL agent and learn optimal duty cycle strategies. The performance of the strategy was validated in a long-term setup with various network sizes, demonstrating its efficiency and scalability. 

**Abstract (ZH)**: 无线传感器网络（WSNs）已成为结构健康监测（SHM）的一个有前景的解决方案，特别是在难以到达或偏远的位置。基于电池的WSNs相较于有线系统具有多种优势，然而电池寿命有限始终是实际应用中的一个主要障碍，无论是否采用能量采集方法。尽管已经研究了多种电池健康管理方法，但现有方法仅专注于延长单个电池的寿命，缺乏系统层面的整体视角。这种方法的应用导致WSNs中的电池可能在不同时间失效，给电池更换计划和调度带来了重大挑战。本研究探讨了一种通过优化WSNs的系统级工作周期来主动管理电池退化状态的深度强化学习（DRL）方法。这种主动管理策略有效减少了单个电池过早失效的情况，使得能够进行团体更换而不牺牲WSNs的整体性能。基于实际WSNs配置的模拟环境被开发用于训练DRL代理并学习最优工作周期策略。通过长期设置下的各种网络规模验证了该策略的性能，展示了其效率和可扩展性。 

---
# Blend the Separated: Mixture of Synergistic Experts for Data-Scarcity Drug-Target Interaction Prediction 

**Title (ZH)**: 分离结合：协同专家混合在数据稀缺的药物-靶标相互作用预测中的应用 

**Authors**: Xinlong Zhai, Chunchen Wang, Ruijia Wang, Jiazheng Kang, Shujie Li, Boyu Chen, Tengfei Ma, Zikai Zhou, Cheng Yang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15796)  

**Abstract**: Drug-target interaction prediction (DTI) is essential in various applications including drug discovery and clinical application. There are two perspectives of input data widely used in DTI prediction: Intrinsic data represents how drugs or targets are constructed, and extrinsic data represents how drugs or targets are related to other biological entities. However, any of the two perspectives of input data can be scarce for some drugs or targets, especially for those unpopular or newly discovered. Furthermore, ground-truth labels for specific interaction types can also be scarce. Therefore, we propose the first method to tackle DTI prediction under input data and/or label scarcity. To make our model functional when only one perspective of input data is available, we design two separate experts to process intrinsic and extrinsic data respectively and fuse them adaptively according to different samples. Furthermore, to make the two perspectives complement each other and remedy label scarcity, two experts synergize with each other in a mutually supervised way to exploit the enormous unlabeled data. Extensive experiments on 3 real-world datasets under different extents of input data scarcity and/or label scarcity demonstrate our model outperforms states of the art significantly and steadily, with a maximum improvement of 53.53%. We also test our model without any data scarcity and it still outperforms current methods. 

**Abstract (ZH)**: 药物-靶标相互作用预测（DTI）在药物发现和临床应用中至关重要。存在两种广泛用于DTI预测的输入数据视角：内在数据表示药物或靶标是如何构建的，外在数据表示药物或靶标与其他生物实体的关联。然而，对于某些药物或靶标，任何一种视角的输入数据可能都很稀缺，尤其是在这些药物或靶标不常用或新被发现的情况下。此外，特定相互作用类型的_ground-truth标签也可能稀缺。因此，我们提出了首个在输入数据和/或标签稀缺条件下解决DTI预测的方法。为了使模型在仅有一种视角的输入数据时仍可运行，我们设计了两个分别处理内在和外在数据的专业模块，并根据不同样本自适应地融合它们。此外，为了使两种视角互补并弥补标签稀缺性，这两个专业模块以互监督的方式协同工作，利用大量未标记的数据。在不同程度的输入数据和/或标签稀缺条件下，对三个真实世界数据集的广泛实验表明，我们的模型显著且稳健地优于当前最先进的方法，最大改进幅度达53.53%。我们还在没有任何数据稀缺的情况下测试了该模型，其性能仍优于当前方法。 

---
# MobiFuse: Learning Universal Human Mobility Patterns through Cross-domain Data Fusion 

**Title (ZH)**: MobiFuse: 通过跨域数据融合学习通用的人类移动模式 

**Authors**: Haoxuan Ma, Xishun Liao, Yifan Liu, Qinhua Jiang, Chris Stanford, Shangqing Cao, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.15779)  

**Abstract**: Human mobility modeling is critical for urban planning and transportation management, yet existing datasets often lack the resolution and semantic richness required for comprehensive analysis. To address this, we proposed a cross-domain data fusion framework that integrates multi-modal data of distinct nature and spatio-temporal resolution, including geographical, mobility, socio-demographic, and traffic information, to construct a privacy-preserving and semantically enriched human travel trajectory dataset. This framework is demonstrated through two case studies in Los Angeles (LA) and Egypt, where a domain adaptation algorithm ensures its transferability across diverse urban contexts. Quantitative evaluation shows that the generated synthetic dataset accurately reproduces mobility patterns observed in empirical data. Moreover, large-scale traffic simulations for LA County based on the generated synthetic demand align well with observed traffic. On California's I-405 corridor, the simulation yields a Mean Absolute Percentage Error of 5.85% for traffic volume and 4.36% for speed compared to Caltrans PeMS observations. 

**Abstract (ZH)**: 跨域数据融合框架在洛杉矶和埃及的城市规划与交通管理中的应用研究 

---
# Can one size fit all?: Measuring Failure in Multi-Document Summarization Domain Transfer 

**Title (ZH)**: 适合所有尺寸吗？：多文档总结领域迁移中的失败度量 

**Authors**: Alexandra DeLucia, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2503.15768)  

**Abstract**: Abstractive multi-document summarization (MDS) is the task of automatically summarizing information in multiple documents, from news articles to conversations with multiple speakers. The training approaches for current MDS models can be grouped into four approaches: end-to-end with special pre-training ("direct"), chunk-then-summarize, extract-then-summarize, and inference with GPT-style models. In this work, we evaluate MDS models across training approaches, domains, and dimensions (reference similarity, quality, and factuality), to analyze how and why models trained on one domain can fail to summarize documents from another (News, Science, and Conversation) in the zero-shot domain transfer setting. We define domain-transfer "failure" as a decrease in factuality, higher deviation from the target, and a general decrease in summary quality. In addition to exploring domain transfer for MDS models, we examine potential issues with applying popular summarization metrics out-of-the-box. 

**Abstract (ZH)**: 抽象多文档总结（MDS）是自动从多篇文档中总结信息的任务，范围从新闻文章到多讲话者对话。当前MDS模型的训练方法可以归为四种：端到端带有特殊预训练（“直接”）、分块再总结、抽取再总结，以及基于GPT风格的推理方法。在本研究中，我们评估了不同训练方法、领域及维度（参考相似度、质量和事实性）下的MDS模型，以分析并探讨为何在零样本领域迁移设置中，一个领域的模型无法有效总结其他领域的文档（新闻、科学和对话）。我们定义领域迁移“失败”为事实性下降、偏离目标程度增加以及总结质量总体下降。除了探索MDS模型的领域迁移外，我们还考察了直接应用流行总结评估指标可能存在的问题。 

---
# ATTENTION2D: Communication Efficient Distributed Self-Attention Mechanism 

**Title (ZH)**: 注意力二维：通信高效的分布式自我注意力机制 

**Authors**: Venmugil Elango  

**Link**: [PDF](https://arxiv.org/pdf/2503.15758)  

**Abstract**: Transformer-based models have emerged as a leading architecture for natural language processing, natural language generation, and image generation tasks. A fundamental element of the transformer architecture is self-attention, which allows the model to capture intricate dependencies within the data. However, the self-attention mechanism also incurs significant computational and memory costs, particularly for long sequences.
In this paper, we introduce ATTENTION2D, a novel approach that exploits parallelism along two dimensions - query and key/value - of the self-attention operation. This method enables efficient distribution and parallelization of computations across multiple devices. Our approach facilitates asymptotically faster training and inference phases compared to previous methods, without relying on approximations or incurring additional computational or memory overheads. Furthermore, unlike existing techniques that struggle to scale with an increasing number of processing units, our approach effectively scales with additional processing units.
Our experimental results confirm the effectiveness of our method in improving communication efficiency and scalability. Compared to Ring Attention, our approach demonstrated up to a 5x performance boost on a GPT-3-like model using 64 NVIDIA A100 GPUs across 16 nodes, and up to a 9.4x performance boost on 64 NVIDIA H100 GPUs across 64 nodes. 

**Abstract (ZH)**: 基于Transformer的模型已成为自然语言处理、自然语言生成和图像生成任务的主要架构。Transformer架构中的基本要素是自我注意力机制，它允许模型捕获数据中的复杂依赖关系。然而，自我注意力机制也会产生显著的计算和内存成本，尤其是在处理长序列时。

在本文中，我们提出了ATTENTION2D，这是一种新颖的方法，该方法利用了自我注意力操作过程中的两个维度——查询和键/值维度的并行性。这种方法能够在多个设备上实现计算的高效分布和并行化。我们的方法相较于以前的方法能够在不依赖近似或增加额外计算和内存开销的情况下，实现渐近更快的训练和推理阶段。此外，与现有技术不同，我们的方法能够有效地随着处理单元数量的增加而扩展。

我们的实验结果证实了该方法在提高通信效率和扩展性方面的有效性。与Ring Attention相比，在64个NVIDIA A100 GPU（分布在16个节点上）的GPT-3类似模型上，我们的方法表现出高达5倍的性能提升；在64个NVIDIA H100 GPU（分布在64个节点上）上，我们的方法表现出高达9.4倍的性能提升。 

---
# Predicting Multi-Agent Specialization via Task Parallelizability 

**Title (ZH)**: 基于任务并行性的多代理专业化预测 

**Authors**: Elizabeth Mieczkowski, Ruaridh Mon-Williams, Neil Bramley, Christopher G. Lucas, Natalia Velez, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2503.15703)  

**Abstract**: Multi-agent systems often rely on specialized agents with distinct roles rather than general-purpose agents that perform the entire task independently. However, the conditions that govern the optimal degree of specialization remain poorly understood. In this work, we propose that specialist teams outperform generalist ones when environmental constraints limit task parallelizability -- the potential to execute task components concurrently. Drawing inspiration from distributed systems, we introduce a heuristic to predict the relative efficiency of generalist versus specialist teams by estimating the speed-up achieved when two agents perform a task in parallel rather than focus on complementary subtasks. We validate this heuristic through three multi-agent reinforcement learning (MARL) experiments in Overcooked-AI, demonstrating that key factors limiting task parallelizability influence specialization. We also observe that as the state space expands, agents tend to converge on specialist strategies, even when generalist ones are theoretically more efficient, highlighting potential biases in MARL training algorithms. Our findings provide a principled framework for interpreting specialization given the task and environment, and introduce a novel benchmark for evaluating whether MARL finds optimal strategies. 

**Abstract (ZH)**: 多智能体系统往往依赖于具有明确角色的专业智能体，而非能够独立完成整个任务的一般智能体。然而，最优专业化程度的条件仍不清楚。在本项工作中，我们提出，在环境约束限制任务并行化能力的情况下，专业团队的表现优于通用团队。借鉴分布式系统原理，我们提出了一种启发式方法，通过估计两个智能体并行执行任务与专注于互补子任务时所获加速的差异，来预测专业团队与通用团队的相对效率。我们通过Overcooked-AI中的三个多智能体强化学习实验验证了这一启发式方法，展示了限制任务并行化的关键因素影响专业化程度。我们还观察到，随着状态空间的扩大，智能体倾向于采用专业策略，即使理论上通用策略更高效，这揭示了MARL训练算法可能存在偏差。我们的研究为给定任务和环境下的专业化提供了一个原理性的框架，并引入了一个新的基准来评估MARL是否找到了最优策略。 

---
# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction 

**Title (ZH)**: UI-Vision: 以桌面为中心的GUI视觉感知与交互基准 

**Authors**: Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M. Tamer Özsu, Aishwarya Agrawal, David Vazquez, Christopher Pal, Perouz Taslakian, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15661)  

**Abstract**: Autonomous agents that navigate Graphical User Interfaces (GUIs) to automate tasks like document editing and file management can greatly enhance computer workflows. While existing research focuses on online settings, desktop environments, critical for many professional and everyday tasks, remain underexplored due to data collection challenges and licensing issues. We introduce UI-Vision, the first comprehensive, license-permissive benchmark for offline, fine-grained evaluation of computer use agents in real-world desktop environments. Unlike online benchmarks, UI-Vision provides: (i) dense, high-quality annotations of human demonstrations, including bounding boxes, UI labels, and action trajectories (clicks, drags, and keyboard inputs) across 83 software applications, and (ii) three fine-to-coarse grained tasks-Element Grounding, Layout Grounding, and Action Prediction-with well-defined metrics to rigorously evaluate agents' performance in desktop environments. Our evaluation reveals critical limitations in state-of-the-art models like UI-TARS-72B, including issues with understanding professional software, spatial reasoning, and complex actions like drag-and-drop. These findings highlight the challenges in developing fully autonomous computer use agents. By releasing UI-Vision as open-source, we aim to advance the development of more capable agents for real-world desktop tasks. 

**Abstract (ZH)**: 自主导航图形用户界面的代理能够自动化文档编辑和文件管理等任务，极大地提升了计算机工作流程。虽然现有研究主要集中在在线环境中，但许多专业和日常任务所需的桌面环境由于数据收集挑战和许可问题仍被忽视。我们引入了UI-Vision，这是首个全面且许可宽松的基准，用于离线、细致地评估计算机使用代理在真实桌面环境中的性能。与在线基准不同，UI-Vision 提供了：（i）包含边界框、UI 标签和操作轨迹（点击、拖拽和键盘输入）的高质量人类演示密集标注，覆盖83个软件应用程序，以及（ii）三个从细粒度到粗粒度的任务——元素定位、布局定位和动作预测，具有明确的评价指标，以严格评估代理在桌面环境中的表现。我们的评估揭示了诸如UI-TARS-72B等最先进的模型的关键局限性，包括理解和处理专业软件、空间推理以及拖拽等复杂操作的问题。这些发现突显了开发完全自主的计算机使用代理面临的挑战。通过将UI-Vision开源，我们希望促进更强大代理的发展，用于真实的桌面任务。 

---
# Survey on Generalization Theory for Graph Neural Networks 

**Title (ZH)**: 图神经网络泛化理论综述 

**Authors**: Antonis Vasileiou, Stefanie Jegelka, Ron Levie, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2503.15650)  

**Abstract**: Message-passing graph neural networks (MPNNs) have emerged as the leading approach for machine learning on graphs, attracting significant attention in recent years. While a large set of works explored the expressivity of MPNNs, i.e., their ability to separate graphs and approximate functions over them, comparatively less attention has been directed toward investigating their generalization abilities, i.e., making meaningful predictions beyond the training data. Here, we systematically review the existing literature on the generalization abilities of MPNNs. We analyze the strengths and limitations of various studies in these domains, providing insights into their methodologies and findings. Furthermore, we identify potential avenues for future research, aiming to deepen our understanding of the generalization abilities of MPNNs. 

**Abstract (ZH)**: 消息传递图神经网络（MPNNs）已成为图上机器学习的主导方法，近年来吸引了大量关注。尽管大量研究探索了MPNNs的表达能力，即它们区分图形和近似其上函数的能力，但相对较少的研究关注其泛化能力，即在训练数据之外进行有意义的预测。在此，我们系统地回顾了现有文献中关于MPNNs泛化能力的研究。我们分析了这些领域各种研究的优势和局限性，提供了有关其方法和发现的见解。此外，我们确定了未来研究的潜在方向，以加深我们对MPNNs泛化能力的理解。 

---
# Zero-Knowledge Federated Learning: A New Trustworthy and Privacy-Preserving Distributed Learning Paradigm 

**Title (ZH)**: 零知识联邦学习：一种新的可信赖和隐私保护的分布式学习范式 

**Authors**: Yuxin Jin, Taotao Wang, Qing Yang, Long Shi, Shengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15550)  

**Abstract**: Federated Learning (FL) has emerged as a promising paradigm in distributed machine learning, enabling collaborative model training while preserving data privacy. However, despite its many advantages, FL still contends with significant challenges -- most notably regarding security and trust. Zero-Knowledge Proofs (ZKPs) offer a potential solution by establishing trust and enhancing system integrity throughout the FL process. Although several studies have explored ZKP-based FL (ZK-FL), a systematic framework and comprehensive analysis are still lacking. This article makes two key contributions. First, we propose a structured ZK-FL framework that categorizes and analyzes the technical roles of ZKPs across various FL stages and tasks. Second, we introduce a novel algorithm, Verifiable Client Selection FL (Veri-CS-FL), which employs ZKPs to refine the client selection process. In Veri-CS-FL, participating clients generate verifiable proofs for the performance metrics of their local models and submit these concise proofs to the server for efficient verification. The server then selects clients with high-quality local models for uploading, subsequently aggregating the contributions from these selected clients. By integrating ZKPs, Veri-CS-FL not only ensures the accuracy of performance metrics but also fortifies trust among participants while enhancing the overall efficiency and security of FL systems. 

**Abstract (ZH)**: 联邦学习(Federated Learning)作为一种分布式机器学习的有前途的范式，能够在保护数据隐私的同时实现协作模型训练。然而，尽管联邦学习具有许多优势，仍面临诸多挑战，特别是在安全性和可信度方面。零知识证明(Zero-Knowledge Proofs, ZKPs)提供了一种潜在的解决方案，通过在整个联邦学习过程中建立信任和增强系统完整性。尽管已有若干研究探讨了基于零知识证明的联邦学习(ZK-FL)，但系统框架和全面分析仍然缺乏。本文做出两项关键贡献。首先，我们提出了一种结构化的ZK-FL框架，对ZKPs在不同联邦学习阶段和技术任务中的技术角色进行了分类和分析。其次，我们引入了一种新的算法——可验证客户端选择联邦学习(Verifiable Client Selection Federated Learning, Veri-CS-FL)，该算法利用零知识证明来优化客户端选择过程。在Veri-CS-FL中，参与的客户端生成其本地模型性能指标的可验证证明，并将这些简洁的证明提交给服务器进行高效验证。服务器随后选择具有高质量本地模型的客户端以进行上传，并汇总这些选定客户端的贡献。通过整合零知识证明，Veri-CS-FL不仅确保了性能指标的准确性，还增强了参与者之间的信任，同时提高了整个联邦学习系统的效率和安全性。 

---
# Rendering Transparency to Ranking in Educational Assessment via Bayesian Comparative Judgement 

**Title (ZH)**: 基于贝叶斯比较判断的教育评估中透明度排序的研究 

**Authors**: Andy Gray, Alma Rahat, Stephen Lindsay, Jen Pearson, Tom Crick  

**Link**: [PDF](https://arxiv.org/pdf/2503.15549)  

**Abstract**: Ensuring transparency in educational assessment is increasingly critical, particularly post-pandemic, as demand grows for fairer and more reliable evaluation methods. Comparative Judgement (CJ) offers a promising alternative to traditional assessments, yet concerns remain about its perceived opacity. This paper examines how Bayesian Comparative Judgement (BCJ) enhances transparency by integrating prior information into the judgement process, providing a structured, data-driven approach that improves interpretability and accountability.
BCJ assigns probabilities to judgement outcomes, offering quantifiable measures of uncertainty and deeper insights into decision confidence. By systematically tracking how prior data and successive judgements inform final rankings, BCJ clarifies the assessment process and helps identify assessor disagreements. Multi-criteria BCJ extends this by evaluating multiple learning outcomes (LOs) independently, preserving the richness of CJ while producing transparent, granular rankings aligned with specific assessment goals. It also enables a holistic ranking derived from individual LOs, ensuring comprehensive evaluations without compromising detailed feedback.
Using a real higher education dataset with professional markers in the UK, we demonstrate BCJ's quantitative rigour and ability to clarify ranking rationales. Through qualitative analysis and discussions with experienced CJ practitioners, we explore its effectiveness in contexts where transparency is crucial, such as high-stakes national assessments. We highlight the benefits and limitations of BCJ, offering insights into its real-world application across various educational settings. 

**Abstract (ZH)**: 确保教育评估的透明性在大流行后变得日益重要，随着对更公平、更可靠评价方法的需求不断增加，比较判断（CJ）提供了一种有前景的替代方案，但对其透明度仍存在担忧。本文研究了贝叶斯比较判断（BCJ）如何通过整合_prior_信息来增强透明性，提供了一个结构化、数据驱动的方法，以提高可解释性和责任感。BCJ为判断结果分配概率，提供可量化的不确定性度量和对决策信心的更深层次洞察。通过系统地跟踪先验数据和后续判断如何影响最终排名，BCJ阐明了评价过程，并有助于识别评估者之间的分歧。多标准BCJ在此基础上通过独立评估多个学习成果（LO），保留了CJ的丰富性，同时生成与特定评估目标相一致的透明、细化的排名。它还能够从单个LO获得一个整体排名，确保全面评价同时不牺牲详细的反馈。使用英国专业标记的真实高等教育数据集，我们展示了BCJ的量化严谨性和对排名理据的澄清能力。通过定性分析和与经验丰富的CJ实践者的讨论，我们探讨了其在透明性至关重要的背景下，如高风险国家评估中的有效性。我们指出了BCJ的优势和局限性，提供了其在各种教育环境中实际应用的见解。 

---
# A Logic of Uncertain Interpretation 

**Title (ZH)**: 不确定解释逻辑 

**Authors**: Adam Bjorndahl  

**Link**: [PDF](https://arxiv.org/pdf/2503.15544)  

**Abstract**: We introduce a logical framework for reasoning about "uncertain interpretations" and investigate two key applications: a new semantics for implication capturing a kind of "meaning entailment", and a conservative notion of "evidentially supported" belief that takes the form of a Dempster-Shafer belief function. 

**Abstract (ZH)**: 我们引入了一种逻辑框架来推理“不确定解释”，并探讨了两种关键应用：一种新的蕴含语义，捕捉某种“意义蕴含”；以及一种保守的“证据支持”的信念概念，表现为德莫佛-舍费尔信念函数。 

---
# Identifying Likely-Reputable Blockchain Projects on Ethereum 

**Title (ZH)**: 识别很可能具备良好声誉的以太坊区块链项目 

**Authors**: Cyrus Malik, Josef Bajada, Joshua Ellul  

**Link**: [PDF](https://arxiv.org/pdf/2503.15542)  

**Abstract**: Identifying reputable Ethereum projects remains a critical challenge within the expanding blockchain ecosystem. The ability to distinguish between legitimate initiatives and potentially fraudulent schemes is non-trivial. This work presents a systematic approach that integrates multiple data sources with advanced analytics to evaluate credibility, transparency, and overall trustworthiness. The methodology applies machine learning techniques to analyse transaction histories on the Ethereum blockchain.
The study classifies accounts based on a dataset comprising 2,179 entities linked to illicit activities and 3,977 associated with reputable projects. Using the LightGBM algorithm, the approach achieves an average accuracy of 0.984 and an average AUC of 0.999, validated through 10-fold cross-validation. Key influential factors include time differences between transactions and received_tnx.
The proposed methodology provides a robust mechanism for identifying reputable Ethereum projects, fostering a more secure and transparent investment environment. By equipping stakeholders with data-driven insights, this research enables more informed decision-making, risk mitigation, and the promotion of legitimate blockchain initiatives. Furthermore, it lays the foundation for future advancements in trust assessment methodologies, contributing to the continued development and maturity of the Ethereum ecosystem. 

**Abstract (ZH)**: 识别值得信赖的以太坊项目仍然是扩展中的区块链生态系统中的一项关键挑战。区分合法项目和潜在欺诈方案并非易事。本文提出了一种系统性方法，将多种数据源与高级分析技术相结合，以评估信誉、透明度和整体可信度。该方法利用机器学习技术分析以太坊区块链上的交易历史。

该研究基于包含2,179个与非法活动相关的实体和3,977个与值得信赖项目相关的实体的数据集，对账户进行了分类。通过使用LightGBM算法，该方法在10折交叉验证中实现了平均准确率0.984和平均AUC值0.999。关键影响因素包括交易之间的时差和received_tnx。

所提出的方法为识别值得信赖的以太坊项目提供了 robust 机制，促进了更安全和透明的投资环境。通过为利益相关者提供数据驱动的洞察，该研究使决策更加明智，风险控制更加有效，并促进合法区块链项目的推广。此外，该研究为基础信任评估方法的进一步发展奠定了基础，有助于以太坊生态系统的持续发展和成熟。 

---
# There must be encapsulated nonconceptual content in vision 

**Title (ZH)**: 视觉中必须存在非概念化内容。 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.15538)  

**Abstract**: In this paper I want to propose an argument to support Jerry Fodor's thesis (Fodor 1983) that input systems are modular and thus informationally encapsulated. The argument starts with the suggestion that there is a "grounding problem" in perception, i. e. that there is a problem in explaining how perception that can yield a visual experience is possible, how sensation can become meaningful perception of something for the subject. Given that visual experience is actually possible, this invites a transcendental argument that explains the conditions of its possibility. I propose that one of these conditions is the existence of a visual module in Fodor's sense that allows the step from sensation to object-identifying perception, thus enabling visual experience. It seems to follow that there is informationally encapsulated nonconceptual content in visual perception. 

**Abstract (ZH)**: 本文旨在提出一个论据，以支持杰里·福多（Fodor 1983）的观点，即输入系统是模块化的，因此是信息封装的。该论据从感知存在“基础问题”的建议开始，即解释如何可能产生视觉经验，如何使感觉转化为主体对事物有意义的感知存在困难。鉴于视觉经验实际上是可能的，这促使我们进行一个先验论证来解释其可能性的条件。我提出其中一个条件是福多意义上存在的视觉模块，它使从感觉过渡到对象识别的感知成为可能，从而使得视觉经验成为可能。据此可以认为视觉感知中存在信息封装的非概念性内容。 

---
# A Beautiful Mind: Principles and Strategies for AI-Augmented Human Reasoning 

**Title (ZH)**: 美丽的大脑：AI增强人类推理的原则与策略 

**Authors**: Sean Koon  

**Link**: [PDF](https://arxiv.org/pdf/2503.15530)  

**Abstract**: Amidst the race to create more intelligent machines, this paper asserts a critical need to invest in human reasoning so that people can manage the many new challenges and opportunities of the future. As people face accelerating changes and complexities in our society, there is a risk that we will rely on AI in ways that reduce our own agency as humans. This paper outlines a human-centered augmented reasoning paradigm by 1. Articulating fundamental principles for augmented reasoning tools, emphasizing their ergonomic, pre-conclusive, directable, exploratory, enhancing, and integrated nature; 2. Proposing a 'many tasks, many tools' approach to ensuring human control, and 3. Offering examples of interaction modes that can serve as bridges between human reasoning and AI algorithms. 

**Abstract (ZH)**: 在创建更智能机器的竞争中，本文强调了投资于人类推理以应对未来众多新挑战和机遇的迫切需要。随着社会加速变化和复杂性增加，存在一种风险，即我们可能会以减少自身人类自主性的姿态过度依赖AI。本文通过以下三个方面提出了以人类为中心的增强推理范式：1. 阐述增强推理工具的基本原则，强调其人体工学性、非结论性、可指引性、探索性、增强性以及整合性；2. 提出“多种任务，多种工具”的方法以确保人类控制；3. 提供示例以作为人类推理与AI算法之间的桥梁交互模式。 

---
# Complying with the EU AI Act: Innovations in Explainable and User-Centric Hand Gesture Recognition 

**Title (ZH)**: 遵循欧盟人工智能法案：可解释性和用户中心的手势识别创新 

**Authors**: Sarah Seifi, Tobias Sukianto, Cecilia Carbonelli, Lorenzo Servadei, Robert Wille  

**Link**: [PDF](https://arxiv.org/pdf/2503.15528)  

**Abstract**: The EU AI Act underscores the importance of transparency, user-centricity, and robustness in AI systems, particularly for high-risk systems. In response, we present advancements in XentricAI, an explainable hand gesture recognition (HGR) system designed to meet these regulatory requirements. XentricAI adresses fundamental challenges in HGR, such as the opacity of black-box models using explainable AI methods and the handling of distributional shifts in real-world data through transfer learning techniques. We extend an existing radar-based HGR dataset by adding 28,000 new gestures, with contributions from multiple users across varied locations, including 24,000 out-of-distribution gestures. Leveraging this real-world dataset, we enhance XentricAI's capabilities by integrating a variational autoencoder module for improved gesture anomaly detection, incorporating user-specific thresholding. This integration enables the identification of 11.50% more anomalous gestures. Our extensive evaluations demonstrate a 97.5% sucess rate in characterizing these anomalies, significantly improving system explainability. Furthermore, the implementation of transfer learning techniques has shown a substantial increase in user adaptability, with an average improvement of at least 15.17%. This work contributes to the development of trustworthy AI systems by providing both technical advancements and regulatory compliance, offering a commercially viable solution that aligns with the EU AI Act requirements. 

**Abstract (ZH)**: 欧盟AI法案强调了透明性、用户中心性和鲁棒性在AI系统中的重要性，特别是对于高风险系统。为此，我们提出了XentricAI的进展，这是一种解释性的手势识别（HGR）系统，旨在满足这些监管要求。XentricAI解决了手势识别中的根本挑战，如使用解释性AI方法解决黑盒模型的不透明性问题，并通过迁移学习技术处理实际数据中的分布移变。我们扩展了一个现有的雷达基手势识别数据集，添加了28,000个新手势，这些手势来自多个用户，遍布不同的地点，其中24,000个属于分布外手势。利用这个实际数据集，我们通过集成变分自编码器模块提高了XentricAI的手势异常检测能力，并引入了用户特定的阈值。这种集成使得能够识别出11.50%更多的异常手势。广泛的评估显示，该系统在特征化这些异常方面成功率达到97.5%，显著提高了系统的解释性。此外，迁移学习技术的实施显示了用户适应性的显著提高，平均改进幅度至少为15.17%。这项工作通过提供技术和合规改进，促进了值得信赖的AI系统的发展，提供了一个符合欧盟AI法案要求的商业可行解决方案。 

---
# Exploring the Panorama of Anxiety Levels: A Multi-Scenario Study Based on Human-Centric Anxiety Level Detection and Personalized Guidance 

**Title (ZH)**: 探究焦虑水平全景：基于以人为中心的焦虑水平检测与个性化指导的多场景研究 

**Authors**: Longdi Xian, Junhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15527)  

**Abstract**: More and more people are experiencing pressure from work, life, and education. These pressures often lead to an anxious state of mind, or even the early symptoms of suicidal ideation. With the advancement of artificial intelligence (AI) technology, large language models have become one of the most prominent technologies. They are often used for detecting psychological disorders. However, current studies primarily provide categorization results without offering interpretable explanations for these results. To address this gap, this study adopts a person-centered perspective and focuses on GPT-generated multi-scenario simulated conversations. These simulated conversations were selected as data samples for the study. Various transformer-based encoder models were utilized to develop a classification model capable of identifying different levels of anxiety. Additionally, a knowledge base focusing on anxiety was constructed using LangChain and GPT-4. When analyzing classification results, this knowledge base was able to provide explanations and reasons most relevant to the interlocutor's anxiety situation. The study demonstrates that the proposed model achieves over 94% accuracy in categorical prediction, and the advice provided is highly personalized and relevant. 

**Abstract (ZH)**: 越来越多的人正经历着工作、生活和教育带来的压力，这些压力常常导致焦虑情绪，甚至出现自杀念头的早期症状。随着人工智能技术的发展，大型语言模型已成为最具代表性的技术之一，常被用于检测心理障碍。然而，现有研究主要提供分类结果，缺乏对这些结果的可解释性分析。为解决这一问题，本研究采用以人为核心的观点，专注于GPT生成的多场景模拟对话。这些模拟对话被选作研究的数据样本。利用各种基于变换器的编码器模型，开发了一种分类模型，能够识别不同级别的焦虑。此外，通过LangChain和GPT-4构建了一个关注焦虑的知识库，在分析分类结果时，该知识库能够提供与对话者焦虑情况最相关的解释和原因。研究显示，所提出的模型在类别预测中的准确率超过94%，提供的建议高度个性化且相关。 

---
# The Use of Artificial Intelligence Tools in Assessing Content Validity: A Comparative Study with Human Experts 

**Title (ZH)**: 使用人工智能工具评估内容效度：与人类专家的比较研究 

**Authors**: Hatice Gurdil, Hatice Ozlem Anadol, Yesim Beril Soguksu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15525)  

**Abstract**: In this study, it was investigated whether AI evaluators assess the content validity of B1-level English reading comprehension test items in a manner similar to human evaluators. A 25-item multiple-choice test was developed, and these test items were evaluated by four human and four AI evaluators. No statistically significant difference was found between the scores given by human and AI evaluators, with similar evaluation trends observed. The Content Validity Ratio (CVR) and the Item Content Validity Index (I-CVI) were calculated and analyzed using the Wilcoxon Signed-Rank Test, with no statistically significant difference. The findings revealed that in some cases, AI evaluators could replace human evaluators. However, differences in specific items were thought to arise from varying interpretations of the evaluation criteria. Ensuring linguistic clarity and clearly defining criteria could contribute to more consistent evaluations. In this regard, the development of hybrid evaluation systems, in which AI technologies are used alongside human experts, is recommended. 

**Abstract (ZH)**: 本研究 investigate 了 AI 评价者是否以类似人类评价者的方式评估 B1 级英语阅读理解测试项目的内容有效性。开发了一项包含 25 道多项选择题的测试，这些测试项目由四名人类评价者和四名 AI 评价者进行了评估。 human 和 AI 评价者给出的分数没有统计学显著差异，且评价趋势相似。使用威尔科克森符号秩检验计算并分析了内容有效性比率 (CVR) 和项目内容有效性指数 (I-CVI)，同样没有统计学显著差异。研究发现，在某些情况下，AI 评价者可以替代人类评价者。然而，特定项目的差异被认为源于对评价标准的不同解释。为了提高一致性，确保语言清晰并明确界定标准是有帮助的。因此，建议开发人机结合的评价系统，其中 AI 技术与人类专家相结合。 

---
# Analysis of AI Effectiveness in Reducing Human Errors in Processing Transportation Requests 

**Title (ZH)**: 分析AI在减少处理运输请求中的人为错误方面的有效性 

**Authors**: Oleksandr Korostin  

**Link**: [PDF](https://arxiv.org/pdf/2503.15517)  

**Abstract**: This article examines the characteristics of human errors in processing transportation requests. The role of artificial intelligence (AI) in maritime transportation is explored. The main methods and technologies used for automating and optimizing the handling of transportation requests are analyzed, along with their impact on reducing the number of errors. Examples of successful AI implementation in large companies are provided, confirming the positive influence of these technologies on overall operational efficiency and customer service levels. 

**Abstract (ZH)**: 本文研究了处理运输请求中的人为错误特征，探讨了人工智能在海运运输中的作用，并分析了自动化和优化运输请求处理的主要方法和技术及其减少错误数量的影响。提供了大型公司成功实施人工智能的例子，证实了这些技术对整体运营效率和客户服务水平的积极影响。 

---
# In Pursuit of Predictive Models of Human Preferences Toward AI Teammates 

**Title (ZH)**: 追求预测人类对AI队友偏好的模型 

**Authors**: Ho Chit Siu, Jaime D. Peña, Yutai Zhou, Ross E. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2503.15516)  

**Abstract**: We seek measurable properties of AI agents that make them better or worse teammates from the subjective perspective of human collaborators. Our experiments use the cooperative card game Hanabi -- a common benchmark for AI-teaming research. We first evaluate AI agents on a set of objective metrics based on task performance, information theory, and game theory, which are measurable without human interaction. Next, we evaluate subjective human preferences toward AI teammates in a large-scale (N=241) human-AI teaming experiment. Finally, we correlate the AI-only objective metrics with the human subjective preferences. Our results refute common assumptions from prior literature on reinforcement learning, revealing new correlations between AI behaviors and human preferences. We find that the final game score a human-AI team achieves is less predictive of human preferences than esoteric measures of AI action diversity, strategic dominance, and ability to team with other AI. In the future, these correlations may help shape reward functions for training human-collaborative AI. 

**Abstract (ZH)**: 我们从人类合作者的主观视角出发，寻求衡量AI代理作为队友优劣的可测量属性。我们的实验使用合作纸牌游戏Hanabi——这是AI合作者研究的常见基准。首先，我们根据任务性能、信息论和博弈论对AI代理进行客观度量评估，这些度量无需人类交互即可实现。接下来，我们在大规模（N=241）的人机团队试验中评估人类对AI队友的主观偏好。最后，我们将仅基于AI的客观度量与人类的主观偏好进行关联。我们的结果反驳了先前关于强化学习文献中的常见假设，揭示了AI行为与人类偏好之间新的关联性。我们发现，人类-AI团队最终的游戏得分不如AI行为的神秘度量（如行为多样性、战略优势和与其他AI协作的能力）对人类偏好的预测能力强。未来，这些关联可能有助于塑造训练人类协作AI的奖励函数。 

---
# Towards Computer-Using Personal Agents 

**Title (ZH)**: 面向计算机使用的人工智能个人代理 

**Authors**: Piero A. Bonatti, John Domingue, Anna Lisa Gentile, Andreas Harth, Olaf Hartig, Aidan Hogan, Katja Hose, Ernesto Jimenez-Ruiz, Deborah L. McGuinness, Chang Sun, Ruben Verborgh, Jesse Wright  

**Link**: [PDF](https://arxiv.org/pdf/2503.15515)  

**Abstract**: Computer-Using Agents (CUA) enable users to automate increasingly-complex tasks using graphical interfaces such as browsers. As many potential tasks require personal data, we propose Computer-Using Personal Agents (CUPAs) that have access to an external repository of the user's personal data. Compared with CUAs, CUPAs offer users better control of their personal data, the potential to automate more tasks involving personal data, better interoperability with external sources of data, and better capabilities to coordinate with other CUPAs in order to solve collaborative tasks involving the personal data of multiple users. 

**Abstract (ZH)**: 使用计算机的个人代理（CUPA）通过图形界面如浏览器使用户能够自动化执行日益复杂的任务。由于许多潜在的任务需要个人数据，我们提出使用外部个人数据仓库的个人代理（CUPA）以增强用户对其个人数据的控制、实现更多涉及个人数据的任务自动化、提高与外部数据源的互操作性以及更好地协调与其他CUPA以解决涉及多名用户个人数据的协作任务。 

---
# Superhuman AI Disclosure: Impacts on Toxicity, Fairness, and Trust Vary by Expertise and Persona Attributes 

**Title (ZH)**: 超人类AI披露：毒性、公平性和信任的影响因专家水平和角色属性而异 

**Authors**: Jaymari Chua, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15514)  

**Abstract**: As artificial intelligence demonstrates surpassing human performance across real-world tasks, disclosing superhuman capabilities poses challenges for fairness, accountability, and trust. To investigate how transparency impacts attitudes and perceptions, we introduce a grounded and validated set of synthetic personas reflecting diverse fairness concerns and technology acceptance levels. Then we evaluate responses in two contrasting domains: (1) a competitive player in StarCraft II, where strategy and high-skill gameplay often elicit toxic interactions, and (2) a cooperative personal-assistant in providing information. Across numerous interactions spanning persona profiles, we test non-disclosure versus explicit superhuman labelling under controlled game outcomes and usage contexts. Our findings reveal sharp domain-specific effects: in StarCraft II, explicitly labelling AI as superhuman, novice personas who learned of it reported lower toxicity and higher fairness-attributing defeat to advanced skill rather than hidden cheating-whereas expert personas found the disclosure statements irksome but still less deceptive than non-disclosure. Conversely, in the LLM as personal-assistant setting, disclosure of superhuman capabilities improved perceived trustworthiness, though it risked AI overreliance among certain persona segments. We release Dataset X-containing persona cards-including profile attributes, disclosure prompts, and detailed interaction logs, accompanied by reproducible protocols and disclaimers for adapting them to diverse tasks. Our results demonstrate that transparency is not a cure-all: while it reduces suspicion and enhances trust in cooperative contexts, it may inflame resistance or disappointment in competitive domains. 

**Abstract (ZH)**: 随着人工智能在实际任务中展现出超越人类的表现，披露超人类能力对公平性、责任制和信任提出了挑战。为了探究透明性如何影响态度和认知，我们引入了一套基于实证并经过验证的合成人物角色，这些角色反映了多样化的公平问题和技术接受水平。然后我们在两个对比领域进行了评估：（1）《星际争霸II》的竞技玩家，其中策略和高水平的游戏常常引发有毒互动；（2）作为提供信息的合作个人助手。我们在多种人物角色档案的交互中，测试了不披露与显式标签超人类能力在受控游戏结果和使用环境下的效果。我们的发现揭示了特定领域的影响：在《星际争霸II》中，明确标记AI为超人类，初学者人物认为其负面行为减少，更倾向于将失败归因于高级技能而非隐藏作弊——而专家人物则对披露声明感到不悦，但仍比不披露更不具有欺骗性。相反，在大型语言模型作为个人助手的设置中，披露超人类能力提高了可信度，但某些人物角色群体可能会过度依赖AI。我们发布了Dataset X——包含人物卡片的资料集，包括人物属性、披露提示及详细的交互日志，并附有可复制的协议和免责声明以适应各种任务。我们的研究结果表明，透明性并非万能解：虽然它在合作情境中减少疑虑并增强信任，但在竞争领域可能会加剧抵抗或失望。 

---
# Beyond Accuracy, SHAP, and Anchors -- On the difficulty of designing effective end-user explanations 

**Title (ZH)**: 超越准确性、SHAP和Anchors——关于设计有效用户解释的困难 

**Authors**: Zahra Abba Omar, Nadia Nahar, Jacob Tjaden, Inès M. Gilles, Fikir Mekonnen, Jane Hsieh, Christian Kästner, Alka Menon  

**Link**: [PDF](https://arxiv.org/pdf/2503.15512)  

**Abstract**: Modern machine learning produces models that are impossible for users or developers to fully understand -- raising concerns about trust, oversight and human dignity. Transparency and explainability methods aim to provide some help in understanding models, but it remains challenging for developers to design explanations that are understandable to target users and effective for their purpose. Emerging guidelines and regulations set goals but may not provide effective actionable guidance to developers. In a controlled experiment with 124 participants, we investigate whether and how specific forms of policy guidance help developers design explanations for an ML-powered screening tool for diabetic retinopathy. Contrary to our expectations, we found that participants across the board struggled to produce quality explanations, comply with the provided policy requirements for explainability, and provide evidence of compliance. We posit that participant noncompliance is in part due to a failure to imagine and anticipate the needs of their audience, particularly non-technical stakeholders. Drawing on cognitive process theory and the sociological imagination to contextualize participants' failure, we recommend educational interventions. 

**Abstract (ZH)**: 现代机器学习产生的模型难以让用户或开发者完全理解——这引发了关于信任、监管和人类尊严的担忧。透明性和可解释性方法旨在帮助理解模型，但开发者为目标用户设计可理解且有效的解释仍然具有挑战性。新兴的指导方针和法规设定目标，但未必能为开发者提供有效的可操作指导。在一项涉及124名参与者的受控实验中，我们研究了特定形式的政策指引是否以及如何帮助开发者设计一个糖尿病视网膜病变机器学习筛查工具的解释。与我们的预期相反，我们发现所有参与者的解释质量低劣，未能遵守提供的可解释性要求，也无法提供合规证据。我们认为，参与者未能遵守指引的部分原因是未能设想和预见其受众的需求，特别是非技术利益相关者的需求。结合认知过程理论和社会学想象力来解释参与者的表现不佳，我们建议实施教育干预措施。 

---
# MapColorAI: Designing Contextually Relevant Choropleth Map Color Schemes Using a Large Language Model 

**Title (ZH)**: MapColorAI: 使用大型语言模型设计上下文相关的大比例尺地图颜色方案 

**Authors**: Nai Yang, Yijie Wang, Fan Wu, Zhiwei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.15502)  

**Abstract**: Choropleth maps, which utilize color schemes to visualize spatial patterns and trends, are simple yet effective tools for geographic data analysis. As such, color scheme design is a critical aspect of choropleth map creation. The traditional coloring methods offered by GIS tools such as ArcGIS and QGIS are not user-friendly for non-professionals. On the one hand, these tools provide numerous color schemes, making it hard to decide which one best matches the theme. On the other hand, it is difficult to fulfill some ambiguous and personalized coloring needs of users, such as requests for 'summer-like' map colors. To address these shortcomings, we develop a novel system that leverages a large language model and map color design principles to generate contextually relevant and user-aligned choropleth map color schemes. The system follows a three-stage process: Data processing, which provides an overview of the data and classifies the data into meaningful classes; Color Concept Design, where the color theme and color mode are conceptualized based on data characteristics and user intentions; and Color Scheme Design, where specific colors are assigned to classes based on generated color theme, color mode, and user requirements. Our system incorporates an interactive interface, providing necessary visualization for choropleth map color design and allowing users to customize and refine color choices flexibly. Through user studies and evaluations, the system demonstrates acceptable usability, accuracy, and flexibility, with users highlighting the tool's efficiency and ease of use. 

**Abstract (ZH)**: choropleth 图表：一种利用颜色方案可视化空间模式和趋势的简单而有效的地理数据工具，其色彩方案设计是choropleth 图表创建的关键方面。GIS 工具如 ArcGIS 和 QGIS 提供的传统着色方法对于非专业人士不够用户友好。一方面，这些工具提供了众多的颜色方案，使得难以决定哪个方案最符合主题。另一方面，很难满足用户的模糊和个性化着色需求，例如“夏日”风格的颜色请求。为解决这些不足，我们开发了一个全新的系统，该系统利用大规模语言模型和地图颜色设计原则来生成上下文相关且用户对齐的choropleth 图表颜色方案。该系统遵循三阶段过程：数据处理，提供数据概览并按有意义的类别对数据进行分类；颜色概念设计，基于数据特性和用户意图构想颜色主题和颜色模式；颜色方案设计，根据生成的颜色主题、颜色模式和用户要求为类别分配特定颜色。该系统集成了交互式界面，为choropleth 图表颜色设计提供必要的可视化，允许用户灵活地自定义和调整颜色选择。通过用户研究和评估，该系统展示了可接受的易用性、准确性和灵活性，用户还强调了该工具的高效性和易用性。 

---
# Development of an Inclusive Educational Platform Using Open Technologies and Machine Learning: A Case Study on Accessibility Enhancement 

**Title (ZH)**: 使用开放技术和机器学习开发的包容性教育平台：无障碍性增强案例研究 

**Authors**: Jimi Togni  

**Link**: [PDF](https://arxiv.org/pdf/2503.15501)  

**Abstract**: This study addresses the pressing challenge of educational inclusion for students with special needs by proposing and developing an inclusive educational platform. Integrating machine learning, natural language processing, and cross-platform interfaces, the platform features key functionalities such as speech recognition functionality to support voice commands and text generation via voice input; real-time object recognition using the YOLOv5 model, adapted for educational environments; Grapheme-to-Phoneme (G2P) conversion for Text-to-Speech systems using seq2seq models with attention, ensuring natural and fluent voice synthesis; and the development of a cross-platform mobile application in Flutter with on-device inference execution using TensorFlow Lite. The results demonstrated high accuracy, usability, and positive impact in educational scenarios, validating the proposal as an effective tool for educational inclusion. This project underscores the importance of open and accessible technologies in promoting inclusive and quality education. 

**Abstract (ZH)**: 本研究通过提出并发展一个包容性教育平台，应对特殊需求学生教育包容性的迫切挑战。该平台集成机器学习、自然语言处理和跨平台接口，具备语音识别功能支持语音命令和通过语音输入生成文本；使用YOLOv5模型进行实时物体识别，适应教育环境；使用带有注意力机制的seq2seq模型进行字母到音素转换以确保语音合成自然流畅；并开发了一个使用Flutter的跨平台移动应用程序，在设备端执行TensorFlow Lite推理。结果表明该平台在教育场景中具有高准确性、易用性和积极影响，验证了该提案是一个有效的教育包容工具。本项目强调开放和可访问技术在促进包容性和高质量教育方面的重要性。 

---
# Approach to Visual Attractiveness of Event Space Through Data-Driven Environment and Spatial Perception 

**Title (ZH)**: 基于数据驱动环境与空间感知的事件空间视觉吸引力评估方法 

**Authors**: Aliffi Majiid, Riaz-Ul-Haque Mian, Kouki Kurohara, Yen-Khang Nguyen-Tran  

**Link**: [PDF](https://arxiv.org/pdf/2503.15499)  

**Abstract**: Revitalizing Japan's remote areas has become a crucial task, and Matsue City exemplifies this effort in its temporary event spaces, created through collective efforts to foster urban vibrancy and bring together residents and visitors. This research examines the relationship between data-driven in-sights using generative AI and visual attractiveness by evaluating tempo-rary events in Matsue City, particularly considering the cognitive-cultural differences in processing visual information of the participants. The first phase employs semantic keyword extraction from interviews, categorizing responses into physical elements, activities, and atmosphere. The second phase analyzes spatial perception through three categories: layout hierar-chy, product visibility, and visual attention. The correlation indicates that successful event design requires a balance between spatial efficiency and diverse needs, with a spatial organization that optimizes visitor flow and visibility strategies considering cultural and demographic diversity. These findings contribute to understanding the urban quality of temporary event spaces and offer a replicable framework for enhancing the visual appeal of events in remote areas throughout Japan. 

**Abstract (ZH)**: 日本偏远地区的 revitalization 成为了关键任务，松江市通过集体努力创建临时活动空间，促进城市活力并汇聚居民和游客，成为了这一努力的典范。本研究通过生成AI生成的数据驱动见解与视觉吸引力之间的关系，评估松江市临时活动的特点，特别考虑参与者在处理视觉信息方面的认知文化差异。第一阶段从访谈中提取语义关键词，将回应分类为物理元素、活动和氛围。第二阶段通过布局层次、产品可见性和视觉注意力三个维度分析空间感知。相关性表明，成功的活动设计需要在空间效率和多样化需求之间取得平衡，考虑文化与人口多样性的情况下优化访客流动和视觉策略。这些发现有助于理解临时活动空间的城市品质，并为增强日本偏远地区活动的视觉吸引力提供可复制的框架。 

---
# Revival: Collaborative Artistic Creation through Human-AI Interactions in Musical Creativity 

**Title (ZH)**: Revival: 通过人机互动在音乐创作中的协作艺术创作 

**Authors**: Keon Ju M. Lee, Philippe Pasquier, Jun Yuri  

**Link**: [PDF](https://arxiv.org/pdf/2503.15498)  

**Abstract**: Revival is an innovative live audiovisual performance and music improvisation by our artist collective K-Phi-A, blending human and AI musicianship to create electronic music with audio-reactive visuals. The performance features real-time co-creative improvisation between a percussionist, an electronic music artist, and AI musical agents. Trained in works by deceased composers and the collective's compositions, these agents dynamically respond to human input and emulate complex musical styles. An AI-driven visual synthesizer, guided by a human VJ, produces visuals that evolve with the musical landscape. Revival showcases the potential of AI and human collaboration in improvisational artistic creation. 

**Abstract (ZH)**: Revival：K-Phi-A的创新现场音视频表演与音乐即兴创作，融合人类与AI音乐才能，呈现音频反应视觉并与人类即兴互动的电子音乐。 

---
# The Impact of Big Five Personality Traits on AI Agent Decision-Making in Public Spaces: A Social Simulation Study 

**Title (ZH)**: 五大人格特质对公共空间中AI代理决策的影响：一项社会仿真研究 

**Authors**: Mingjun Ren, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15497)  

**Abstract**: This study investigates how the Big Five personality traits influence decision-making processes in AI agents within public spaces. Using AgentVerse framework and GPT-3.5-turbo, we simulated interactions among 10 AI agents, each embodying different dimensions of the Big Five personality traits, in a classroom environment responding to misinformation. The experiment assessed both public expressions ([Speak]) and private thoughts ([Think]) of agents, revealing significant correlations between personality traits and decision-making patterns. Results demonstrate that Openness to Experience had the strongest impact on information acceptance, with curious agents showing high acceptance rates and cautious agents displaying strong skepticism. Extraversion and Conscientiousness also showed notable influence on decision-making, while Neuroticism and Agreeableness exhibited more balanced responses. Additionally, we observed significant discrepancies between public expressions and private thoughts, particularly in agents with friendly and extroverted personalities, suggesting that social context influences decision-making behavior. Our findings contribute to understanding how personality traits shape AI agent behavior in social settings and have implications for developing more nuanced and context-aware AI systems. 

**Abstract (ZH)**: 本研究探讨了五大人格特质如何影响公共空间中AI代理的决策过程。我们使用AgentVerse框架和GPT-3.5-turbo模拟了10个AI代理在教室环境中的互动，每个代理代表五大人格特质的不同维度，应对错误信息。实验评估了代理的公开表达（[Speak]）和内心想法（[Think]），揭示了人格特质与决策模式之间的显著相关性。研究结果表明，开放性对信息接受的影响最大，好奇的代理显示出高的接受率，谨慎的代理表现出强烈的怀疑态度。外向性和尽责性也对决策产生了显著影响，而神经质和宜人性则表现出较为平衡的响应。此外，我们还观察到公共表达和内心想法之间存在显著差异，特别是在友好和外向的人格特质的代理中，这表明社会背景影响决策行为。本研究为理解人格特质如何塑造社会环境中AI代理的行为提供了见解，并为开发更具微妙性和情境意识的AI系统提供了启示。 

---
# Entwicklung einer Webanwendung zur Generierung von skolemisierten RDF Daten für die Verwaltung von Lieferketten 

**Title (ZH)**: 开发一种生成斯科勒姆化RDF数据的Web应用以管理供应链 

**Authors**: Roman Laas  

**Link**: [PDF](https://arxiv.org/pdf/2503.15495)  

**Abstract**: Für eine frühzeitige Erkennung von Lieferengpässen müssen Lieferketten in einer geeigneten digitalen Form vorliegen, damit sie verarbeitet werden können. Der für die Datenmodellierung benötigte Arbeitsaufwand ist jedoch, gerade IT-fremden Personen, nicht zuzumuten. Es wurde deshalb im Rahmen dieser Arbeit eine Webanwendung entwickelt, welche die zugrunde liegende Komplexität für den Benutzer verschleiern soll. Konkret handelt es sich dabei um eine grafische Benutzeroberfläche, auf welcher Templates instanziiert und miteinander verknüpft werden können. Für die Definition dieser Templates wurden in dieser Arbeit geeignete Konzepte erarbeitet und erweitert. Zur Erhebung der Benutzerfreundlichkeit der Webanwendung wurde abschließend eine Nutzerstudie mit mehreren Testpersonen durchgeführt. Diese legte eine Vielzahl von nützlichen Verbesserungsvorschlägen offen.
--
For early detection of supply bottlenecks, supply chains must be available in a suitable digital form so that they can be processed. However, the amount of work required for data modeling cannot be expected of people who are not familiar with IT topics. Therefore, a web application was developed in the context of this thesis, which is supposed to disguise the underlying complexity for the user. Specifically, this is a graphical user interface on which templates can be instantiated and linked to each other. Suitable concepts for the definition of these templates were developed and extended in this thesis. Finally, a user study with several test persons was conducted to determine the usability of the web application. This revealed a large number of useful suggestions for improvement. 

**Abstract (ZH)**: early detection of supply bottlenecks through a user-friendly web application 

---
# World of ScoreCraft: Novel Multi Scorer Experiment on the Impact of a Decision Support System in Sleep Staging 

**Title (ZH)**: ScoreCraft 世界：新型多评分者实验，探讨决策支持系统对睡眠分期的影响 

**Authors**: Benedikt Holm, Arnar Óskarsson, Björn Elvar Þorleifsson, Hörður Þór Hafsteinsson, Sigríður Sigurðardóttir, Heiður Grétarsdóttir, Kenan Hoelke, Gabriel Marc Marie Jouan, Thomas Penzel, Erna Sif Arnardottir, María Óskarsdóttir  

**Link**: [PDF](https://arxiv.org/pdf/2503.15492)  

**Abstract**: Manual scoring of polysomnography (PSG) is a time intensive task, prone to inter scorer variability that can impact diagnostic reliability. This study investigates the integration of decision support systems (DSS) into PSG scoring workflows, focusing on their effects on accuracy, scoring time, and potential biases toward recommendations from artificial intelligence (AI) compared to human generated recommendations. Using a novel online scoring platform, we conducted a repeated measures study with sleep technologists,
who scored traditional and self applied PSGs. Participants were occasionally presented with recommendations labeled as either human or AI generated. We found that traditional PSGs tended to be scored slightly more accurately than self applied PSGs, but this difference was not statistically significant. Correct recommendations significantly improved scoring accuracy for both PSG types, while incorrect recommendations reduced accuracy. No significant bias was observed toward or against AI generated recommendations compared to human generated recommendations. These findings highlight the potential of AI to enhance PSG scoring reliability. However, ensuring the accuracy of AI outputs is critical to maximizing its benefits. Future research should explore the long term impacts of DSS on scoring workflows and strategies for integrating AI in clinical practice. 

**Abstract (ZH)**: PSG打分的人工决策支持系统集成：对准确性、打分时间和人工智能推荐偏倚的影响研究 

---
