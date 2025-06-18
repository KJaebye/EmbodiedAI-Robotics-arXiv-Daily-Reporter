# SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning 

**Title (ZH)**: SENIOR：基于偏好的强化学习中高效查询选择与偏好引导探索 

**Authors**: Hexian Ni, Tao Lu, Haoyuan Hu, Yinghao Cai, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14648)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) methods provide a solution to avoid reward engineering by learning reward models based on human preferences. However, poor feedback- and sample- efficiency still remain the problems that hinder the application of PbRL. In this paper, we present a novel efficient query selection and preference-guided exploration method, called SENIOR, which could select the meaningful and easy-to-comparison behavior segment pairs to improve human feedback-efficiency and accelerate policy learning with the designed preference-guided intrinsic rewards. Our key idea is twofold: (1) We designed a Motion-Distinction-based Selection scheme (MDS). It selects segment pairs with apparent motion and different directions through kernel density estimation of states, which is more task-related and easy for human preference labeling; (2) We proposed a novel preference-guided exploration method (PGE). It encourages the exploration towards the states with high preference and low visits and continuously guides the agent achieving the valuable samples. The synergy between the two mechanisms could significantly accelerate the progress of reward and policy learning. Our experiments show that SENIOR outperforms other five existing methods in both human feedback-efficiency and policy convergence speed on six complex robot manipulation tasks from simulation and four real-worlds. 

**Abstract (ZH)**: 基于偏好强化学习的高效查询选择与偏好引导探索方法：SENIOR 

---
# Barrier Method for Inequality Constrained Factor Graph Optimization with Application to Model Predictive Control 

**Title (ZH)**: 不等式约束因子图优化的屏障方法及其在模型预测控制中的应用 

**Authors**: Anas Abdelkarim, Holger Voos, Daniel Görges  

**Link**: [PDF](https://arxiv.org/pdf/2506.14341)  

**Abstract**: Factor graphs have demonstrated remarkable efficiency for robotic perception tasks, particularly in localization and mapping applications. However, their application to optimal control problems -- especially Model Predictive Control (MPC) -- has remained limited due to fundamental challenges in constraint handling. This paper presents a novel integration of the Barrier Interior Point Method (BIPM) with factor graphs, implemented as an open-source extension to the widely adopted g2o framework. Our approach introduces specialized inequality factor nodes that encode logarithmic barrier functions, thereby overcoming the quadratic-form limitations of conventional factor graph formulations. To the best of our knowledge, this is the first g2o-based implementation capable of efficiently handling both equality and inequality constraints within a unified optimization backend. We validate the method through a multi-objective adaptive cruise control application for autonomous vehicles. Benchmark comparisons with state-of-the-art constraint-handling techniques demonstrate faster convergence and improved computational efficiency. (Code repository: this https URL) 

**Abstract (ZH)**: 因子图在机器人感知任务中展现了显著的效率，特别是在定位和建图应用中。然而，由于处理约束的基本挑战，它们在最优控制问题——尤其是模型预测控制（MPC）——中的应用仍受到限制。本文提出了一种将障碍内部点方法（BIPM）与因子图相结合的新型集成方法，并作为对广泛采用的g2o框架的开源扩展实现。我们的方法引入了专门的不等式因子节点，编码对数障碍函数，从而克服了传统因子图公式化的二次形式限制。据我们所知，这是首个基于g2o框架能够高效处理等式和不等式约束的统一优化后端的实现。该方法通过自主车辆的多目标自适应巡航控制应用进行了验证。基准比较显示，与最先进的约束处理技术相比，该方法具有更快的收敛性和更好的计算效率。（代码库：this https URL） 

---
# Uncertainty-Driven Radar-Inertial Fusion for Instantaneous 3D Ego-Velocity Estimation 

**Title (ZH)**: 基于不确定性驱动的雷达-惯导融合的即时三维 ego 速度 estimation 

**Authors**: Prashant Kumar Rai, Elham Kowsari, Nataliya Strokina, Reza Ghabcheloo  

**Link**: [PDF](https://arxiv.org/pdf/2506.14294)  

**Abstract**: We present a method for estimating ego-velocity in autonomous navigation by integrating high-resolution imaging radar with an inertial measurement unit. The proposed approach addresses the limitations of traditional radar-based ego-motion estimation techniques by employing a neural network to process complex-valued raw radar data and estimate instantaneous linear ego-velocity along with its associated uncertainty. This uncertainty-aware velocity estimate is then integrated with inertial measurement unit data using an Extended Kalman Filter. The filter leverages the network-predicted uncertainty to refine the inertial sensor's noise and bias parameters, improving the overall robustness and accuracy of the ego-motion estimation. We evaluated the proposed method on the publicly available ColoRadar dataset. Our approach achieves significantly lower error compared to the closest publicly available method and also outperforms both instantaneous and scan matching-based techniques. 

**Abstract (ZH)**: 我们提出了一种通过集成高分辨率成像雷达和惯性测量单元来估计自主导航中自我速度的方法。该提出的方案通过运用神经网络处理复杂的雷达原始数据，并同时估计瞬时线性自我速度及其相关的不确定性，解决了传统雷达基自我运动估计技术的局限性。然后，使用扩展卡尔曼滤波器将该不确定性意识下的速度估计值与惯性测量单元数据结合。滤波器利用网络预测的不确定性来细化惯性传感器的噪声和偏置参数，从而提高整体的鲁棒性和精度。我们在公开可用的ColoRadar数据集上评估了所提出的方法。我们的方法在误差方面显著低于最接近的公开可用方法，并且在瞬时速度估计和扫描匹配技术方面也表现出更好的性能。 

---
# Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior 

**Title (ZH)**: 马尔可夫 regime 切换智能驾驶模型及其可解释的跟随行为 

**Authors**: Chengyuan Zhang, Cathy Wu, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.14762)  

**Abstract**: Accurate and interpretable car-following models are essential for traffic simulation and autonomous vehicle development. However, classical models like the Intelligent Driver Model (IDM) are fundamentally limited by their parsimonious and single-regime structure. They fail to capture the multi-modal nature of human driving, where a single driving state (e.g., speed, relative speed, and gap) can elicit many different driver actions. This forces the model to average across distinct behaviors, reducing its fidelity and making its parameters difficult to interpret. To overcome this, we introduce a regime-switching framework that allows driving behavior to be governed by different IDM parameter sets, each corresponding to an interpretable behavioral mode. This design enables the model to dynamically switch between interpretable behavioral modes, rather than averaging across diverse driving contexts. We instantiate the framework using a Factorial Hidden Markov Model with IDM dynamics (FHMM-IDM), which explicitly separates intrinsic driving regimes (e.g., aggressive acceleration, steady-state following) from external traffic scenarios (e.g., free-flow, congestion, stop-and-go) through two independent latent Markov processes. Bayesian inference via Markov chain Monte Carlo (MCMC) is used to jointly estimate the regime-specific parameters, transition dynamics, and latent state trajectories. Experiments on the HighD dataset demonstrate that FHMM-IDM uncovers interpretable structure in human driving, effectively disentangling internal driver actions from contextual traffic conditions and revealing dynamic regime-switching patterns. This framework provides a tractable and principled solution to modeling context-dependent driving behavior under uncertainty, offering improvements in the fidelity of traffic simulations, the efficacy of safety analyses, and the development of more human-centric ADAS. 

**Abstract (ZH)**: 精确且可解释的跟随车模型对于交通仿真和自动驾驶车辆开发至关重要。然而，经典的模型如智能驾驶员模型（IDM）因其简洁性和单体制的结构而从根本上受到限制。它们未能捕捉到人类驾驶行为的多模态特性，即单一驾驶状态（如速度、相对速度和跟车距离）可以引发多种不同的驾驶员行为。这迫使模型在不同的行为之间取平均，降低了其仿真精度，并使模型参数难以解释。为了解决这一问题，我们提出了一种制度转换框架，使得驾驶行为可以由不同的IDM参数集控制，每个参数集对应一个可解释的行为模式。该设计使模型能够动态地在不同的可解释行为模式之间切换，而不是将多种驾驶情境中的行为进行平均。我们使用因子隐马尔可夫模型与IDM动力学相结合（FHMM-IDM）来实例化该框架，通过两个独立的潜在马尔可夫过程显式地将内在的驾驶制度（如积极加速、稳定跟车）与外部的交通场景（如自由流、拥堵、走走停停）区分开来。通过马尔可夫链蒙特卡洛（MCMC）贝叶斯推理，联合估计各制度的参数、转换动态以及潜在状态轨迹。在HighD数据集上的实验表明，FHMM-IDM揭示了人类驾驶中的可解释结构，有效地区分了内部驾驶行为和外部交通条件，并揭示了动态的制度转换模式。该框架提供了一个在不确定性下建模情境依赖的驾驶行为的可行且原理性的解决方案，提高了交通仿真的精度，增强了安全分析的效能，并促进了更加以人为本的ADAS的发展。 

---
# A Novel Indicator for Quantifying and Minimizing Information Utility Loss of Robot Teams 

**Title (ZH)**: 一种量化和最小化机器人团队信息 utility 损失的新指标 

**Authors**: Xiyu Zhao, Qimei Cui, Wei Ni, Quan Z. Sheng, Abbas Jamalipour, Guoshun Nan, Xiaofeng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14237)  

**Abstract**: The timely exchange of information among robots within a team is vital, but it can be constrained by limited wireless capacity. The inability to deliver information promptly can result in estimation errors that impact collaborative efforts among robots. In this paper, we propose a new metric termed Loss of Information Utility (LoIU) to quantify the freshness and utility of information critical for cooperation. The metric enables robots to prioritize information transmissions within bandwidth constraints. We also propose the estimation of LoIU using belief distributions and accordingly optimize both transmission schedule and resource allocation strategy for device-to-device transmissions to minimize the time-average LoIU within a robot team. A semi-decentralized Multi-Agent Deep Deterministic Policy Gradient framework is developed, where each robot functions as an actor responsible for scheduling transmissions among its collaborators while a central critic periodically evaluates and refines the actors in response to mobility and interference. Simulations validate the effectiveness of our approach, demonstrating an enhancement of information freshness and utility by 98%, compared to alternative methods. 

**Abstract (ZH)**: 团队内部机器人之间及时交换信息至关重要，但受限于有限的无线容量。不能及时传递信息可能导致估计算法误差，影响机器人之间的协作效果。本文提出了一种新的度量标准——信息有用性损失（LoIU），以量化对于合作至关重要的信息的新鲜度和有用性。该度量标准使机器人能够在带宽受限的情况下优先传递重要信息。同时，我们利用信念分布估计LoIU，并优化设备到设备传输的传输时间和资源分配策略，以最小化机器人团队内的平均LoIU。我们开发了一种半去中心化的多代理深度确定性策略梯度框架，其中每个机器人作为一个执行者，负责调度与其他合作机器人的通信，而中央评论者定期评估和改进执行者，以响应移动性和干扰。仿真结果验证了我们方法的有效性，相比其他方法，信息的新鲜度和有用性提高了98%。 

---
# Scaling Algorithm Distillation for Continuous Control with Mamba 

**Title (ZH)**: Scaling Algorithm Distillation for Continuous Control with Mamba 

**Authors**: Samuel Beaussant, Mehdi Mounsif  

**Link**: [PDF](https://arxiv.org/pdf/2506.13892)  

**Abstract**: Algorithm Distillation (AD) was recently proposed as a new approach to perform In-Context Reinforcement Learning (ICRL) by modeling across-episodic training histories autoregressively with a causal transformer model. However, due to practical limitations induced by the attention mechanism, experiments were bottlenecked by the transformer's quadratic complexity and limited to simple discrete environments with short time horizons. In this work, we propose leveraging the recently proposed Selective Structured State Space Sequence (S6) models, which achieved state-of-the-art (SOTA) performance on long-range sequence modeling while scaling linearly in sequence length. Through four complex and continuous Meta Reinforcement Learning environments, we demonstrate the overall superiority of Mamba, a model built with S6 layers, over a transformer model for AD. Additionally, we show that scaling AD to very long contexts can improve ICRL performance and make it competitive even with a SOTA online meta RL baseline. 

**Abstract (ZH)**: 基于S6模型的算法蒸馏在长期序列建模中的应用：超越变压器模型进行上下文强化学习 

---
# Enhancing Symbolic Machine Learning by Subsymbolic Representations 

**Title (ZH)**: 通过亚符号表示增强符号机器学习 

**Authors**: Stephen Roth, Lennart Baur, Derian Boer, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2506.14569)  

**Abstract**: The goal of neuro-symbolic AI is to integrate symbolic and subsymbolic AI approaches, to overcome the limitations of either. Prominent systems include Logic Tensor Networks (LTN) or DeepProbLog, which offer neural predicates and end-to-end learning. The versatility of systems like LTNs and DeepProbLog, however, makes them less efficient in simpler settings, for instance, for discriminative machine learning, in particular in domains with many constants. Therefore, we follow a different approach: We propose to enhance symbolic machine learning schemes by giving them access to neural embeddings. In the present paper, we show this for TILDE and embeddings of constants used by TILDE in similarity predicates. The approach can be fine-tuned by further refining the embeddings depending on the symbolic theory. In experiments in three real-world domain, we show that this simple, yet effective, approach outperforms all other baseline methods in terms of the F1 score. The approach could be useful beyond this setting: Enhancing symbolic learners in this way could be extended to similarities between instances (effectively working like kernels within a logical language), for analogical reasoning, or for propositionalization. 

**Abstract (ZH)**: 神经符号AI的目标是整合符号AI和次符号AI的方法，以克服各自局限性。 prominently 系统包括逻辑张量网络（LTN）或DeepProbLog，它们提供了神经谓词并实现了端到端学习。然而，如LTNs和DeepProbLog等系统的灵活性使得它们在简单设置中效率较低，尤其是在许多常量的领域里，尤其是对于有区别的机器学习。因此，我们采用了一种不同的方法：我们建议通过赋予符号机器学习方案访问神经嵌入的能力来增强它们。在本文中，我们展示了这一方法在TILDE及其在相似谓词中使用的常量嵌入中的应用。该方法可以通过进一步细化嵌入来根据符号理论进行调整。在三个真实世界的领域中进行的实验表明，这种方法在F1得分上优于所有基线方法。这种方法可能适用于更广泛的情境：以这种方式增强符号学习器可以扩展到实例之间的相似性（类似于逻辑语言中的核功能）、类比推理或命题化中。 

---
# QUEST: Quality-aware Semi-supervised Table Extraction for Business Documents 

**Title (ZH)**: QUEST：面向质量的半监督商业文档表格提取 

**Authors**: Eliott Thomas, Mickael Coustaty, Aurelie Joseph, Gaspar Deloin, Elodie Carel, Vincent Poulain D'Andecy, Jean-Marc Ogier  

**Link**: [PDF](https://arxiv.org/pdf/2506.14568)  

**Abstract**: Automating table extraction (TE) from business documents is critical for industrial workflows but remains challenging due to sparse annotations and error-prone multi-stage pipelines. While semi-supervised learning (SSL) can leverage unlabeled data, existing methods rely on confidence scores that poorly reflect extraction quality. We propose QUEST, a Quality-aware Semi-supervised Table extraction framework designed for business documents. QUEST introduces a novel quality assessment model that evaluates structural and contextual features of extracted tables, trained to predict F1 scores instead of relying on confidence metrics. This quality-aware approach guides pseudo-label selection during iterative SSL training, while diversity measures (DPP, Vendi score, IntDiv) mitigate confirmation bias. Experiments on a proprietary business dataset (1000 annotated + 10000 unannotated documents) show QUEST improves F1 from 64% to 74% and reduces empty predictions by 45% (from 12% to 6.5%). On the DocILE benchmark (600 annotated + 20000 unannotated documents), QUEST achieves a 50% F1 score (up from 42%) and reduces empty predictions by 19% (from 27% to 22%). The framework's interpretable quality assessments and robustness to annotation scarcity make it particularly suited for business documents, where structural consistency and data completeness are paramount. 

**Abstract (ZH)**: 基于质量感知的自动表格提取框架：适用于商业文档的QUEST 

---
# GUI-Robust: A Comprehensive Dataset for Testing GUI Agent Robustness in Real-World Anomalies 

**Title (ZH)**: GUI-稳健：一个全面的数据集，用于在实际异常中测试GUI代理的稳健性 

**Authors**: Jingqi Yang, Zhilong Song, Jiawei Chen, Mingli Song, Sheng Zhou, linjun sun, Xiaogang Ouyang, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14477)  

**Abstract**: The development of high-quality datasets is crucial for benchmarking and advancing research in Graphical User Interface (GUI) agents. Despite their importance, existing datasets are often constructed under idealized conditions, overlooking the diverse anomalies frequently encountered in real-world deployments. To address this limitation, we introduce GUI-Robust, a novel dataset designed for comprehensive GUI agent evaluation, explicitly incorporating seven common types of anomalies observed in everyday GUI interactions. Furthermore, we propose a semi-automated dataset construction paradigm that collects user action sequences from natural interactions via RPA tools and then generate corresponding step and task descriptions for these actions with the assistance of MLLMs. This paradigm significantly reduces annotation time cost by a factor of over 19 times. Finally, we assess state-of-the-art GUI agents using the GUI-Robust dataset, revealing their substantial performance degradation in abnormal scenarios. We anticipate that our work will highlight the importance of robustness in GUI agents and inspires more future research in this direction. The dataset and code are available at this https URL.. 

**Abstract (ZH)**: 高質量數據集的開發對於 avaliação 和促進圖形用戶界面（GUI）代理的研究至關重要。儘管其重要性，現有數據集往往是根據理想化的條件構建的，忽略了真實部署中頻頻遇到的多樣化的異常。為了解決這一局限，我們介紹了 GUI-Robust，一個新的數據集，旨在全面評價 GUI 代理，並將七種常見的日常 GUI 交互異常Explicitly整合其中。此外，我們提出了一種半自動化的數據集構建框架，通過 RPA 工具收集用戶操作序列，然後利用 MLLMs 給出這些操作對應的操作步驟和任務描述。該框架將注釋時間成本顯著降低超過19倍。最後，我們使用 GUI-Robust 數據集評估現有先進的 GUI 代理，在異常場景中揭示了其顯著Performance下降。我們預期本工作將突出強健性在 GUI 代理中的重要性，並 Inspirer更多的未來研究。數據集和代碼可在以下链接获得：这个链接。 

---
# AST-Enhanced or AST-Overloaded? The Surprising Impact of Hybrid Graph Representations on Code Clone Detection 

**Title (ZH)**: AST增强还是AST过载？混合图表示对代码克隆检测的意外影响 

**Authors**: Zixian Zhang, Takfarinas Saber  

**Link**: [PDF](https://arxiv.org/pdf/2506.14470)  

**Abstract**: As one of the most detrimental code smells, code clones significantly increase software maintenance costs and heighten vulnerability risks, making their detection a critical challenge in software engineering. Abstract Syntax Trees (ASTs) dominate deep learning-based code clone detection due to their precise syntactic structure representation, but they inherently lack semantic depth. Recent studies address this by enriching AST-based representations with semantic graphs, such as Control Flow Graphs (CFGs) and Data Flow Graphs (DFGs). However, the effectiveness of various enriched AST-based representations and their compatibility with different graph-based machine learning techniques remains an open question, warranting further investigation to unlock their full potential in addressing the complexities of code clone detection. In this paper, we present a comprehensive empirical study to rigorously evaluate the effectiveness of AST-based hybrid graph representations in Graph Neural Network (GNN)-based code clone detection. We systematically compare various hybrid representations ((CFG, DFG, Flow-Augmented ASTs (FA-AST)) across multiple GNN architectures. Our experiments reveal that hybrid representations impact GNNs differently: while AST+CFG+DFG consistently enhances accuracy for convolution- and attention-based models (Graph Convolutional Networks (GCN), Graph Attention Networks (GAT)), FA-AST frequently introduces structural complexity that harms performance. Notably, GMN outperforms others even with standard AST representations, highlighting its superior cross-code similarity detection and reducing the need for enriched structures. 

**Abstract (ZH)**: 作为最具破坏性的代码气味之一，代码克隆显著增加了软件维护成本并提高了漏洞风险，使其检测成为软件工程中的关键挑战。在基于深度学习的代码克隆检测中，抽象语法树（AST）因其精确的语法结构表示占据主导地位，但也缺乏语义深度。近期研究通过将控制流图（CFG）和数据流图（DFG）等语义图融入基于AST的表示来解决这一问题。然而，各种增强的基于AST的表示及其与不同图机器学习技术的兼容性仍然存在问题，需要进一步研究以充分发挥其在解决代码克隆检测复杂性方面的潜力。在本文中，我们进行了一项全面的经验研究，以严格评估基于图神经网络（GNN）的代码克隆检测中基于AST的混合图表示的有效性。我们系统地比较了各种混合表示（包括CFG、DFG、流增强的AST（FA-AST））在多种GNN架构下的效果。我们的实验揭示了混合表示对GNN的影响不同：虽然AST+CFG+DFG一致地增强卷积和注意力模型（图卷积网络（GCN）、图注意力网络（GAT））的准确性，但FA-AST经常引入结构复杂性从而损害性能。值得注意的是，GMN即使使用标准AST表示也优于其他方法，突显了其优越的跨代码相似性检测能力，从而减少了对复杂结构的需求。 

---
# Mxplainer: Explain and Learn Insights by Imitating Mahjong Agents 

**Title (ZH)**: Mxplainer: 通过模仿麻雀代理来解释和学习洞察 

**Authors**: Lingfeng Li, Yunlong Lu, Yongyi Wang, Qifan Zheng, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14246)  

**Abstract**: People need to internalize the skills of AI agents to improve their own capabilities. Our paper focuses on Mahjong, a multiplayer game involving imperfect information and requiring effective long-term decision-making amidst randomness and hidden information. Through the efforts of AI researchers, several impressive Mahjong AI agents have already achieved performance levels comparable to those of professional human players; however, these agents are often treated as black boxes from which few insights can be gleaned. This paper introduces Mxplainer, a parameterized search algorithm that can be converted into an equivalent neural network to learn the parameters of black-box agents. Experiments conducted on AI and human player data demonstrate that the learned parameters provide human-understandable insights into these agents' characteristics and play styles. In addition to analyzing the learned parameters, we also showcase how our search-based framework can locally explain the decision-making processes of black-box agents for most Mahjong game states. 

**Abstract (ZH)**: 人们需要内化AI代理的技能以提高自身能力。本文专注于麻将这种涉及不完全信息的多人游戏，需要在随机性和隐藏信息中进行有效的长期决策。通过AI研究者的努力，已经有一些麻将AI代理达到了专业人类玩家的水平；然而，这些代理往往被视为黑盒，从中得出的见解有限。本文介绍了一种参数化搜索算法Mxplainer，可以通过转换为等效的神经网络来学习黑盒代理的参数。实验表明，学习到的参数为人类提供了对这些代理特性和游戏风格的可理解见解。除了分析学习到的参数外，我们还展示了基于搜索的框架如何在大多数麻将游戏状态下局部解释黑盒代理的决策过程。 

---
# Causes in neuron diagrams, and testing causal reasoning in Large Language Models. A glimpse of the future of philosophy? 

**Title (ZH)**: 神经图中原因的揭示，以及在大型语言模型中测试因果推理：哲学未来的瞥见？ 

**Authors**: Louis Vervoort, Vitaly Nikolaev  

**Link**: [PDF](https://arxiv.org/pdf/2506.14239)  

**Abstract**: We propose a test for abstract causal reasoning in AI, based on scholarship in the philosophy of causation, in particular on the neuron diagrams popularized by D. Lewis. We illustrate the test on advanced Large Language Models (ChatGPT, DeepSeek and Gemini). Remarkably, these chatbots are already capable of correctly identifying causes in cases that are hotly debated in the literature. In order to assess the results of these LLMs and future dedicated AI, we propose a definition of cause in neuron diagrams with a wider validity than published hitherto, which challenges the widespread view that such a definition is elusive. We submit that these results are an illustration of how future philosophical research might evolve: as an interplay between human and artificial expertise. 

**Abstract (ZH)**: 我们提出了一种基于因果哲学研究的AI抽象因果推理测试，特别参考了D.刘易斯普及的神经元图示。我们在先进的大型语言模型（ChatGPT、DeepSeek和Gemini）上进行了测试。令人惊讶的是，这些聊天机器人已经能够正确识别文献中高度争议的因果关系案例。为了评估这些LLM及其未来专门化AI的结果，我们提出了一种比以往发表的更为广泛的在神经元图示中定义因果的概念，挑战了因果定义难以捉摸的普遍观点。我们认为这些结果是未来哲学研究如何演进的一个例证：即人类与人工智能专长之间的互动过程。 

---
# Situational-Constrained Sequential Resources Allocation via Reinforcement Learning 

**Title (ZH)**: 基于情景约束的序列资源分配强化学习方法 

**Authors**: Libo Zhang, Yang Chen, Toru Takisaka, Kaiqi Zhao, Weidong Li, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14125)  

**Abstract**: Sequential Resource Allocation with situational constraints presents a significant challenge in real-world applications, where resource demands and priorities are context-dependent. This paper introduces a novel framework, SCRL, to address this problem. We formalize situational constraints as logic implications and develop a new algorithm that dynamically penalizes constraint violations. To handle situational constraints effectively, we propose a probabilistic selection mechanism to overcome limitations of traditional constraint reinforcement learning (CRL) approaches. We evaluate SCRL across two scenarios: medical resource allocation during a pandemic and pesticide distribution in agriculture. Experiments demonstrate that SCRL outperforms existing baselines in satisfying constraints while maintaining high resource efficiency, showcasing its potential for real-world, context-sensitive decision-making tasks. 

**Abstract (ZH)**: 随机动态资源分配：情境约束下的序列资源分配提出在实际应用中一个显著挑战，其中资源需求和优先级依赖于具体情境。本文提出了一种新颖的框架SCRL以应对这一问题。我们将情境约束形式化为逻辑蕴含，并开发了一种新的算法以动态惩罚约束违反。为了有效处理情境约束，我们提出了一种概率选择机制来克服传统约束强化学习（CRL）方法的局限性。我们在两种场景下评估了SCRL：灾 pandemic 中的医疗资源分配以及农业中的农药分配。实验结果表明，SCRL 在满足约束条件的同时保持了高资源效率，展示了其在实际情境敏感决策任务中的潜力。 

---
# Machine Mirages: Defining the Undefined 

**Title (ZH)**: 机器幻象：定义未定义项 

**Authors**: Hamidou Tembine  

**Link**: [PDF](https://arxiv.org/pdf/2506.13990)  

**Abstract**: As multimodal machine intelligence systems started achieving average animal-level and average human-level fluency in many measurable tasks in processing images, language, and sound, they began to exhibit a new class of cognitive aberrations: machine mirages. These include delusion, illusion, confabulation, hallucination, misattribution error, semantic drift, semantic compression, exaggeration, causal inference failure, uncanny valley of perception, bluffing-patter-bullshitting, cognitive stereotypy, pragmatic misunderstanding, hypersignification, semantic reheating-warming, simulated authority effect, fallacious abductive leap, contextual drift, referential hallucination, semiotic Frankenstein effect, calibration failure, spurious correlation, bias amplification, concept drift sensitivity, misclassification under uncertainty, adversarial vulnerability, overfitting, prosodic misclassification, accent bias, turn boundary failure, semantic boundary confusion, noise overfitting, latency-induced decision drift, ambiguity collapse and other forms of error that mimic but do not replicate human or animal fallibility. This article presents some of the errors and argues that these failures must be explicitly defined and systematically assessed. Understanding machine mirages is essential not only for improving machine intelligence reliability but also for constructing a multiscale ethical, co-evolving intelligence ecosystem that respects the diverse forms of life, cognition, and expression it will inevitably touch. 

**Abstract (ZH)**: 多模态机器智能系统的机器 mirages 及其认知偏差：定义与评估 

---
# Integrating Knowledge Graphs and Bayesian Networks: A Hybrid Approach for Explainable Disease Risk Prediction 

**Title (ZH)**: 知识图谱与贝叶斯网络集成：一种可解释的疾病风险预测混合方法 

**Authors**: Mbithe Nzomo, Deshendran Moodley  

**Link**: [PDF](https://arxiv.org/pdf/2506.13920)  

**Abstract**: Multimodal electronic health record (EHR) data is useful for disease risk prediction based on medical domain knowledge. However, general medical knowledge must be adapted to specific healthcare settings and patient populations to achieve practical clinical use. Additionally, risk prediction systems must handle uncertainty from incomplete data and non-deterministic health outcomes while remaining explainable. These challenges can be alleviated by the integration of knowledge graphs (KGs) and Bayesian networks (BNs). We present a novel approach for constructing BNs from ontology-based KGs and multimodal EHR data for explainable disease risk prediction. Through an application use case of atrial fibrillation and real-world EHR data, we demonstrate that the approach balances generalised medical knowledge with patient-specific context, effectively handles uncertainty, is highly explainable, and achieves good predictive performance. 

**Abstract (ZH)**: 基于本体的知识图谱和贝叶斯网络构建可解释的多模态电子健康记录疾病风险预测方法 

---
# Evaluating Explainability: A Framework for Systematic Assessment and Reporting of Explainable AI Features 

**Title (ZH)**: 评估可解释性：一个系统评估和报告可解释AI特征的框架 

**Authors**: Miguel A. Lago, Ghada Zamzmi, Brandon Eich, Jana G. Delfino  

**Link**: [PDF](https://arxiv.org/pdf/2506.13917)  

**Abstract**: Explainability features are intended to provide insight into the internal mechanisms of an AI device, but there is a lack of evaluation techniques for assessing the quality of provided explanations. We propose a framework to assess and report explainable AI features. Our evaluation framework for AI explainability is based on four criteria: 1) Consistency quantifies the variability of explanations to similar inputs, 2) Plausibility estimates how close the explanation is to the ground truth, 3) Fidelity assesses the alignment between the explanation and the model internal mechanisms, and 4) Usefulness evaluates the impact on task performance of the explanation. Finally, we developed a scorecard for AI explainability methods that serves as a complete description and evaluation to accompany this type of algorithm. We describe these four criteria and give examples on how they can be evaluated. As a case study, we use Ablation CAM and Eigen CAM to illustrate the evaluation of explanation heatmaps on the detection of breast lesions on synthetic mammographies. The first three criteria are evaluated for clinically-relevant scenarios. Our proposed framework establishes criteria through which the quality of explanations provided by AI models can be evaluated. We intend for our framework to spark a dialogue regarding the value provided by explainability features and help improve the development and evaluation of AI-based medical devices. 

**Abstract (ZH)**: 可解释性特征旨在提供对AI设备内部机制的洞察，但缺乏评估所提供解释质量的技术。我们提出了一种评估和报告可解释AI特征的框架。我们的AI可解释性评估框架基于四个标准：1) 一致性量化对相似输入的解释变异性；2) 可信度估计解释与真实情况的接近程度；3) 忠实度评估解释与模型内部机制的对齐程度；4) 实用性评估解释对任务性能的影响。最后，我们开发了一种AI可解释性方法评分卡，作为此类算法的完整描述和评估。我们描述了这四个标准，并给出如何评估它们的例子。作为案例研究，我们使用Ablation CAM和Eigen CAM来说明在合成乳腺X线摄影中检测乳腺病变时解释热图的评估。前三项标准在临床相关的场景中进行评估。我们提出的框架建立了评估AI模型提供的解释质量的标准。我们希望我们的框架能够激发关于解释性特征价值的对话，并帮助改善基于AI的医疗设备的开发和评估。 

---
# Bridging Pattern-Aware Complexity with NP-Hard Optimization: A Unifying Framework and Empirical Study 

**Title (ZH)**: 基于模式感知复杂性的NP难优化问题统一框架与实验研究 

**Authors**: Olivier Saidi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13810)  

**Abstract**: NP hard optimization problems like the Traveling Salesman Problem (TSP) defy efficient solutions in the worst case, yet real-world instances often exhibit exploitable patterns. We propose a novel patternaware complexity framework that quantifies and leverages structural regularities e.g., clustering, symmetry to reduce effective computational complexity across domains, including financial forecasting and LLM optimization. With rigorous definitions, theorems, and a meta learning driven solver pipeline, we introduce metrics like Pattern Utilization Efficiency (PUE) and achieve up to 79 percent solution quality gains in TSP benchmarks (22 to 2392 cities). Distinct from theoretical NP hardness, our approach offers a unified, practical lens for pattern-driven efficiency. 

**Abstract (ZH)**: NP难优化问题如旅行商问题(TSP)在最坏情况下难以高效求解，但实际问题实例往往具有可利用的模式。我们提出了一种新颖的模式感知复杂性框架，该框架量化并利用了结构上的规律性，例如聚类、对称性，以降低跨不同领域的有效计算复杂性，包括金融预测和大语言模型优化。通过严格的定义、定理以及基于元学习的求解器管道，我们引入了模式利用效率(PUE)等度量标准，在TSP基准测试中实现了高达79%的解的质量提升（22至2392个城市）。不同于理论上的NP难性，我们的方法提供了一种统一的、实用的模式驱动效率视角。 

---
# Causality in the human niche: lessons for machine learning 

**Title (ZH)**: 人类生态位中的因果关系：机器学习的启示 

**Authors**: Richard D. Lange, Konrad P. Kording  

**Link**: [PDF](https://arxiv.org/pdf/2506.13803)  

**Abstract**: Humans interpret the world around them in terms of cause and effect and communicate their understanding of the world to each other in causal terms. These causal aspects of human cognition are thought to underlie humans' ability to generalize and learn efficiently in new domains, an area where current machine learning systems are weak. Building human-like causal competency into machine learning systems may facilitate the construction of effective and interpretable AI. Indeed, the machine learning community has been importing ideas on causality formalized by the Structural Causal Model (SCM) framework, which provides a rigorous formal language for many aspects of causality and has led to significant advances. However, the SCM framework fails to capture some salient aspects of human causal cognition and has likewise not yet led to advances in machine learning in certain critical areas where humans excel. We contend that the problem of causality in the ``human niche'' -- for a social, autonomous, and goal-driven agent sensing and acting in the world in which humans live -- is quite different from the kind of causality captured by SCMs. For example, everyday objects come in similar types that have similar causal properties, and so humans readily generalize knowledge of one type of object (cups) to another related type (bowls) by drawing causal analogies between objects with similar properties, but such analogies are at best awkward to express in SCMs. We explore how such causal capabilities are adaptive in, and motivated by, the human niche. By better appreciating properties of human causal cognition and, crucially, how those properties are adaptive in the niche in which humans live, we hope that future work at the intersection of machine learning and causality will leverage more human-like inductive biases to create more capable, controllable, and interpretable systems. 

**Abstract (ZH)**: 人类在关于因果性的认知基础上理解并以因果性术语交流周围的世界，这种认知上的因果性被认为是人类在新领域中泛化和高效学习的能力基础，而当前的机器学习系统在这方面的表现较弱。将人类类似的因果能力构建到机器学习系统中可能有助于构建更有效且可解释的AI。实际上，机器学习社区正在引入通过结构因果模型（SCM）框架正式化的因果性理念，SCM框架为许多因果性方面提供了严谨的形式语言，并取得了显著进展。然而，SCM框架未能捕捉人类因果认知的一些显著方面，也尚未在某些人类表现优异的关键领域推动机器学习的发展。我们认为，在“人类生态位”——一个由感知和行动于其中的社会、自主且目标驱动的智能体所感知和作用的世界——中的因果问题与SCM所捕捉的因果性类型是不同的。例如，日常物体有相似的类型且具有相似的因果属性，因此人类可以很容易地通过在具有相似属性的物体之间构造因果类比来从一种类型的知识（杯子）推广到另一种相关类型的知识（碗），但这类类比在SCM中表达起来最多也显得笨拙。我们探讨了在人类生态位中此类因果能力的适应性和动机。通过更好地理解和感知人类因果认知的属性，特别是这些属性在人类所生活的生态位中的适应性，我们希望未来结合机器学习和因果性的研究能够利用更为人类类似的归纳偏置来创造出更强大、可控且可解释的系统。 

---
# Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization 

**Title (ZH)**: 神经连接组中前向排序的反馈弧最小化方法 

**Authors**: Soroush Vahidi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13799)  

**Abstract**: We present a suite of scalable algorithms for minimizing feedback arcs in large-scale weighted directed graphs, with the goal of revealing biologically meaningful feedforward structure in neural connectomes. Using the FlyWire Connectome Challenge dataset, we demonstrate the effectiveness of our ranking strategies in maximizing the total weight of forward-pointing edges. Our methods integrate greedy heuristics, gain-aware local refinements, and global structural analysis based on strongly connected components. Experiments show that our best solution improves the forward edge weight over previous top-performing methods. All algorithms are implemented efficiently in Python and validated using cloud-based execution on Google Colab Pro+. 

**Abstract (ZH)**: 我们提出了一套可扩展的算法，用于在大规模加权有向图中最小化反馈弧，旨在揭示神经连接组中的生物意义前馈结构。通过FlyWire连接组挑战数据集，我们展示了我们的排名策略在最大化前向指针边的总权重方面的有效性。我们的方法结合了贪心启发式、基于强连通分量的获益感知局部优化以及全局结构分析。实验表明，我们提出的方法在前向边权重方面优于之前表现最佳的方法。所有算法均高效地用Python实现，并在基于云的Google Colab Pro+上进行了验证。 

---
# BotTrans: A Multi-Source Graph Domain Adaptation Approach for Social Bot Detection 

**Title (ZH)**: BotTrans：多源图域适应的社会机器人检测方法 

**Authors**: Boshen Shi, Yongqing Wang, Fangda Guo, Jiangli Shao, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13795)  

**Abstract**: Transferring extensive knowledge from relevant social networks has emerged as a promising solution to overcome label scarcity in detecting social bots and other anomalies with GNN-based models. However, effective transfer faces two critical challenges. Firstly, the network heterophily problem, which is caused by bots hiding malicious behaviors via indiscriminately interacting with human users, hinders the model's ability to learn sufficient and accurate bot-related knowledge from source domains. Secondly, single-source transfer might lead to inferior and unstable results, as the source network may embody weak relevance to the task and provide limited knowledge. To address these challenges, we explore multiple source domains and propose a multi-source graph domain adaptation model named \textit{BotTrans}. We initially leverage the labeling knowledge shared across multiple source networks to establish a cross-source-domain topology with increased network homophily. We then aggregate cross-domain neighbor information to enhance the discriminability of source node embeddings. Subsequently, we integrate the relevance between each source-target pair with model optimization, which facilitates knowledge transfer from source networks that are more relevant to the detection task. Additionally, we propose a refinement strategy to improve detection performance by utilizing semantic knowledge within the target domain. Extensive experiments on real-world datasets demonstrate that \textit{BotTrans} outperforms the existing state-of-the-art methods, revealing its efficacy in leveraging multi-source knowledge when the target detection task is unlabeled. 

**Abstract (ZH)**: 从相关社交网络中转移 extensive 知识以克服基于 GNN 的模型在检测社交机器人和其他异常时面临的标签稀缺问题，已经成为了有前途的解决方案。然而，有效的转移面临着两个关键挑战。首先，由于机器人通过无选择地与用户交互来隐藏恶意行为而引发的网络非同质性问题，阻碍了模型从源领域学习足够的准确的机器人相关知识的能力。其次，单源转移可能导致较差且不稳定的结果，因为源网络可能与任务关联较弱并提供有限的知识。为解决这些挑战，我们探索了多个源领域，并提出了一种名为 \textit{BotTrans} 的多源图域适应模型。我们最初利用跨多个源网络共享的标签知识，建立具有增加网络同质性的跨源域拓扑结构。然后，我们聚合跨域邻居信息以增强源节点表示的区分性。接着，我们将每对源-目标的相关性与模型优化集成，从而促进更多相关于检测任务的源网络的知识转移。此外，我们提出了一种改进策略，通过利用目标域内的语义知识来提升检测性能。在实际数据集上的大量实验表明，\textit{BotTrans} 在目标检测任务未标记的情况下，能够更好地利用多源知识，优于现有的先进方法。 

---
# Med-REFL: Medical Reasoning Enhancement via Self-Corrected Fine-grained Reflection 

**Title (ZH)**: Med-REFL: 医学推理增强 via 自我修正细粒度反思 

**Authors**: Zongxian Yang, Jiayu Qian, Zegao Peng, Haoyu Zhang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13793)  

**Abstract**: Large reasoning models have recently made significant strides in mathematical and code reasoning, yet their success has not transferred smoothly to the medical domain. While multiple factors contribute to this disparity, a critical issue is the inadequate focus on the quality of intermediate reflection steps, which is particularly crucial in high-stakes medical scenarios. To address this challenge, we propose Med-REFL, a \underline{\textbf{Med}}ical \underline{\textbf{R}}easoning \underline{\textbf{E}}nhancement via self-corrected \underline{\textbf{F}}ine-grained ref\underline{\textbf{L}}ection. Our method leverages a tree-of-thought approach to decompose medical questions into fine-grained reasoning paths, quantitatively evaluating each step and its subsequent reflections. These assessments enable automatic construction of direct preference optimization data, reducing reliance on expensive expert annotations while guiding models to identify and correct reasoning errors. Experimental results on the MedQA-USMLE benchmark demonstrate Med-REFL achieves consistent improvements, with average gains up to 4.11\%. Notably, it further boosts the state-of-the-art performance of 7B/8B models by an additional 4.13\%. Furthermore, Med-REFL exhibits strong generalization capabilities and robustness across several challenging medical question-answering datasets. Our work illustrates that prioritizing reflection quality leads to more accurate and trustworthy reasoning in medical AI applications. Checkpoints, code, and data can be found \href{this https URL}{here}. 

**Abstract (ZH)**: 大型推理模型在数学和代码推理领域取得了显著进展，但在医疗领域的成功未能顺利转移。尽管有多种因素导致这种差异，但关键问题是未能充分关注中间推理步骤的质量，尤其是在高风险医疗场景中这一点尤为重要。为应对这一挑战，我们提出了一种Med-REFL方法，即通过自我校正的细粒度反思增强医学推理。该方法利用树状思维方法将医学问题分解为细粒度的推理路径，定量评估每一步及后续反思。这些评估能够自动构建直接偏好优化数据，减少对昂贵专家注释的依赖，同时指导模型识别并纠正推理错误。Med-REFL在MedQA-USMLE基准上的实验结果表明，该方法实现了一致的改进，平均增益高达4.11%。此外，它还进一步提升了7B/8B模型的最新性能，增益达4.13%。此外，Med-REFL在多个具有挑战性的医学问答数据集中表现出强大的泛化能力和鲁棒性。我们的工作表明，在医学AI应用中优先考虑反思质量可以提高推理的准确性与可靠性。更多信息和资源，请访问此处。 

---
# ICE-ID: A Novel Historical Census Data Benchmark Comparing NARS against LLMs, \& a ML Ensemble on Longitudinal Identity Resolution 

**Title (ZH)**: ICE-ID: 一种新型历史人口普查数据基准，比较NARS与LLMs及 longitudinal身份解析的机器学习集成方法 

**Authors**: Gonçalo Hora de Carvalho, Lazar S. Popov, Sander Kaatee, Kristinn R. Thórisson, Tangrui Li, Pétur Húni Björnsson, Jilles S. Dibangoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.13792)  

**Abstract**: We introduce ICE-ID, a novel benchmark dataset for historical identity resolution, comprising 220 years (1703-1920) of Icelandic census records. ICE-ID spans multiple generations of longitudinal data, capturing name variations, demographic changes, and rich genealogical links. To the best of our knowledge, this is the first large-scale, open tabular dataset specifically designed to study long-term person-entity matching in a real-world population. We define identity resolution tasks (within and across census waves) with clearly documented metrics and splits. We evaluate a range of methods: handcrafted rule-based matchers, a ML ensemble as well as LLMs for structured data (e.g. transformer-based tabular networks) against a novel approach to tabular data called NARS (Non-Axiomatic Reasoning System) - a general-purpose AI framework designed to reason with limited knowledge and resources. Its core is Non-Axiomatic Logic (NAL), a term-based logic. Our experiments show that NARS is suprisingly simple and competitive with other standard approaches, achieving SOTA at our task. By releasing ICE-ID and our code, we enable reproducible benchmarking of identity resolution approaches in longitudinal settings and hope that ICE-ID opens new avenues for cross-disciplinary research in data linkage and historical analytics. 

**Abstract (ZH)**: ICE-ID：一种用于历史身份解析的新型基准数据集 

---
# Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations 

**Title (ZH)**: 严格和透明的人类基线推荐与报告检查表在模型评估中的应用 

**Authors**: Kevin L. Wei, Patricia Paskov, Sunishchal Dev, Michael J. Byun, Anka Reuel, Xavier Roberts-Gaal, Rachel Calcott, Evie Coxon, Chinmay Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2506.13776)  

**Abstract**: In this position paper, we argue that human baselines in foundation model evaluations must be more rigorous and more transparent to enable meaningful comparisons of human vs. AI performance, and we provide recommendations and a reporting checklist towards this end. Human performance baselines are vital for the machine learning community, downstream users, and policymakers to interpret AI evaluations. Models are often claimed to achieve "super-human" performance, but existing baselining methods are neither sufficiently rigorous nor sufficiently well-documented to robustly measure and assess performance differences. Based on a meta-review of the measurement theory and AI evaluation literatures, we derive a framework with recommendations for designing, executing, and reporting human baselines. We synthesize our recommendations into a checklist that we use to systematically review 115 human baselines (studies) in foundation model evaluations and thus identify shortcomings in existing baselining methods; our checklist can also assist researchers in conducting human baselines and reporting results. We hope our work can advance more rigorous AI evaluation practices that can better serve both the research community and policymakers. Data is available at: this https URL 

**Abstract (ZH)**: 在这一立场声明中，我们argue认为，在基础模型评估中的人类基线必须更加严格和透明，以促进人类与AI表现的有意义比较，并为此提出了建议和报告清单。人类表现基线对于机器学习社区、下游用户和决策者理解AI评估至关重要。模型常被声称实现“超人”性能，但现有的基线方法既不够严格也不够详细，无法可靠地衡量和评估性能差异。基于对度量理论和AI评估文献的元评审，我们提出了一种框架，涵盖了设计、执行和报告人类基线的建议。我们将这些建议综合成一份清单，用于系统性地审查115项基础模型评估中的基线（研究），从而识别现有基线方法的不足；这份清单还可以帮助研究人员开展人类基线评估并报告结果。我们希望我们的工作能够推进更严谨的AI评估实践，更好地服务于研究社区和决策者。数据可在以下链接获得：this https URL。 

---
# Representing Time-Continuous Behavior of Cyber-Physical Systems in Knowledge Graphs 

**Title (ZH)**: 在知识图谱中表示网络物理系统的时间连续行为 

**Authors**: Milapji Singh Gill, Tom Jeleniewski, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.13773)  

**Abstract**: Time-continuous dynamic models are essential for various Cyber-Physical System (CPS) applications. To ensure effective usability in different lifecycle phases, such behavioral information in the form of differential equations must be contextualized and integrated with further CPS information. While knowledge graphs provide a formal description and structuring mechanism for this task, there is a lack of reusable ontological artifacts and methods to reduce manual instantiation effort. Hence, this contribution introduces two artifacts: Firstly, a modular semantic model based on standards is introduced to represent differential equations directly within knowledge graphs and to enrich them semantically. Secondly, a method for efficient knowledge graph generation is presented. A validation of these artifacts was conducted in the domain of aviation maintenance. Results show that differential equations of a complex Electro-Hydraulic Servoactuator can be formally represented in a knowledge graph and be contextualized with other lifecycle data, proving the artifacts' practical applicability. 

**Abstract (ZH)**: 时间连续动态模型对于各种网络物理系统（CPS）应用至关重要。为了在不同生命周期阶段确保有效的可用性，这种以微分方程形式的行为信息必须进行上下文化和与进一步的CPS信息集成。虽然知识图谱提供了进行这一任务的正式描述和结构化机制，但仍缺乏可重用的本体 artefact 和减少手动实例化努力的方法。因此，本贡献介绍了两种 artefact：首先，基于标准的模块化语义模型被引入，用于直接在知识图谱中表示微分方程并对其进行语义丰富化；其次，提出了一种高效的知识图谱生成方法。这些 artefact 在航空维修领域进行了验证，结果显示，复杂的电气液压伺服作动器的微分方程可以形式化地表示在知识图谱中，并与其他生命周期数据进行上下文化整合，证明了这些 artefact 的实际适用性。 

---
# 'Memory States' from Almost Nothing: Representing and Computing in a Non-associative Algebra 

**Title (ZH)**: 几乎无信息中的记忆状态：在非联想代数中的表示与计算 

**Authors**: Stefan Reimann  

**Link**: [PDF](https://arxiv.org/pdf/2506.13768)  

**Abstract**: This note presents a non-associative algebraic framework for the representation and computation of information items in high-dimensional space. This framework is consistent with the principles of spatial computing and with the empirical findings in cognitive science about memory. Computations are performed through a process of multiplication-like binding and non-associative interference-like bundling. Models that rely on associative bundling typically lose order information, which necessitates the use of auxiliary order structures, such as position markers, to represent sequential information that is important for cognitive tasks. In contrast, the non-associative bundling proposed allows the construction of sparse representations of arbitrarily long sequences that maintain their temporal structure across arbitrary lengths. In this operation, noise is a constituent element of the representation of order information, rather than a means of obscuring it. The non-associative nature of the proposed framework results in the representation of a single sequence by two distinct states. The L-state, generated through left-associative bundling, continuously updates and emphasises a recency effect, while the R-state, formed through right-associative bundling, encodes finite sequences or chunks, capturing a primacy effect. The construction of these states may be associated with activity in the prefrontal cortex in relation to short-term memory and hippocampal encoding in long-term memory, respectively. The accuracy of retrieval is contingent upon a decision-making process that is based on the mutual information between the memory states and the cue. The model is able to replicate the Serial Position Curve, which reflects the empirical recency and primacy effects observed in cognitive experiments. 

**Abstract (ZH)**: 一种非结合代数框架下的高维空间中信息项的表示与计算 

---
# Exploring Speaker Diarization with Mixture of Experts 

**Title (ZH)**: 探索专家混合模型在演讲者分离中的应用 

**Authors**: Gaobin Yang, Maokui He, Shutong Niu, Ruoyu Wang, Hang Chen, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.14750)  

**Abstract**: In this paper, we propose a novel neural speaker diarization system using memory-aware multi-speaker embedding with sequence-to-sequence architecture (NSD-MS2S), which integrates a memory-aware multi-speaker embedding module with a sequence-to-sequence architecture. The system leverages a memory module to enhance speaker embeddings and employs a Seq2Seq framework to efficiently map acoustic features to speaker labels. Additionally, we explore the application of mixture of experts in speaker diarization, and introduce a Shared and Soft Mixture of Experts (SS-MoE) module to further mitigate model bias and enhance performance. Incorporating SS-MoE leads to the extended model NSD-MS2S-SSMoE. Experiments on multiple complex acoustic datasets, including CHiME-6, DiPCo, Mixer 6 and DIHARD-III evaluation sets, demonstrate meaningful improvements in robustness and generalization. The proposed methods achieve state-of-the-art results, showcasing their effectiveness in challenging real-world scenarios. 

**Abstract (ZH)**: 基于记忆aware多说话人嵌入的序列到序列演讲者聚类系统（NSD-MS2S及其扩展模型NSD-MS2S-SSMoE）：探索专家混合在演讲者聚类中的应用 

---
# Adaptive Accompaniment with ReaLchords 

**Title (ZH)**: 实时和声适应性伴奏 

**Authors**: Yusong Wu, Tim Cooijmans, Kyle Kastner, Adam Roberts, Ian Simon, Alexander Scarlatos, Chris Donahue, Cassie Tarakajian, Shayegan Omidshafiei, Aaron Courville, Pablo Samuel Castro, Natasha Jaques, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14723)  

**Abstract**: Jamming requires coordination, anticipation, and collaborative creativity between musicians. Current generative models of music produce expressive output but are not able to generate in an \emph{online} manner, meaning simultaneously with other musicians (human or otherwise). We propose ReaLchords, an online generative model for improvising chord accompaniment to user melody. We start with an online model pretrained by maximum likelihood, and use reinforcement learning to finetune the model for online use. The finetuning objective leverages both a novel reward model that provides feedback on both harmonic and temporal coherency between melody and chord, and a divergence term that implements a novel type of distillation from a teacher model that can see the future melody. Through quantitative experiments and listening tests, we demonstrate that the resulting model adapts well to unfamiliar input and produce fitting accompaniment. ReaLchords opens the door to live jamming, as well as simultaneous co-creation in other modalities. 

**Abstract (ZH)**: Jamming 要求音乐家之间进行协调、预见和合作性创作。当前的音乐生成模型能够产生具有表现力的输出，但无法以在线方式生成，即无法与其他 musician（包括人类或其他 machine）同时进行。我们提出 ReaLchords，这是一种在线生成模型，用于即兴生成和弦伴奏以配合用户旋律。我们通过最大似然预训练一种在线模型，并使用强化学习对其进行微调以适应在线使用。微调目标结合了新颖的奖励模型，该模型提供了关于和声与旋律之间以及时间连贯性的反馈，以及一个发散项，该发散项采用了可以预见到未来旋律的教师模型的新型蒸馏方法。通过定量实验和聆听测试，我们展示了该模型能够很好地适应不熟悉的数据，并生成合适的伴奏。ReaLchords 为现场即兴演奏打开了大门，并且在其他创作模式下也开启了同步创作的可能性。 

---
# Refining music sample identification with a self-supervised graph neural network 

**Title (ZH)**: 基于自监督图神经网络的音乐样本识别 refinement 

**Authors**: Aditya Bhattacharjee, Ivan Meresman Higgs, Mark Sandler, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2506.14684)  

**Abstract**: Automatic sample identification (ASID), the detection and identification of portions of audio recordings that have been reused in new musical works, is an essential but challenging task in the field of audio query-based retrieval. While a related task, audio fingerprinting, has made significant progress in accurately retrieving musical content under "real world" (noisy, reverberant) conditions, ASID systems struggle to identify samples that have undergone musical modifications. Thus, a system robust to common music production transformations such as time-stretching, pitch-shifting, effects processing, and underlying or overlaying music is an important open challenge.
In this work, we propose a lightweight and scalable encoding architecture employing a Graph Neural Network within a contrastive learning framework. Our model uses only 9% of the trainable parameters compared to the current state-of-the-art system while achieving comparable performance, reaching a mean average precision (mAP) of 44.2%.
To enhance retrieval quality, we introduce a two-stage approach consisting of an initial coarse similarity search for candidate selection, followed by a cross-attention classifier that rejects irrelevant matches and refines the ranking of retrieved candidates - an essential capability absent in prior models. In addition, because queries in real-world applications are often short in duration, we benchmark our system for short queries using new fine-grained annotations for the Sample100 dataset, which we publish as part of this work. 

**Abstract (ZH)**: 自动样本识别（ASID）：在基于音频查询的检索领域，这是一种用于检测和识别新音乐作品中重复使用的音频片段的必要但具挑战性的任务。虽然相关任务音频指纹识别已在“真实世界”（噪声、混响）条件下取得了显著进展，但ASID系统在识别经过音乐修改的样本方面仍然遇到困难。因此，能够抵抗常见的音乐制作变换（如时间拉伸、音高变化、效果处理和背景或叠加音乐）的系统是一个重要的开放挑战。

在本工作中，我们提出了一种轻量级且可扩展的编码架构，该架构结合了图神经网络和对比学习框架。与当前最先进的系统相比，我们的模型仅使用了9%的可训练参数，但在性能上达到了可比水平，实现了44.2%的平均平均精度（mAP）。

为了提高检索质量，我们引入了一种两阶段方法，首先进行粗略相似度搜索以选择候选样本，然后使用交叉注意力分类器拒绝无关匹配，并细化检索候选的排名——这一能力在之前模型中是不存在的。此外，由于实际应用中的查询往往较短，我们在本工作中发布针对Sample100数据集的新细粒度注释，并在此基础上对短查询进行系统基准测试。 

---
# Design an Editable Speech-to-Sign-Language Transformer System: A Human-Centered AI Approach 

**Title (ZH)**: 设计一种可编辑的语音到手语转换器系统：以人为中心的AI方法 

**Authors**: Yingchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14677)  

**Abstract**: This paper presents a human-centered, real-time, user-adaptive speech-to-sign language animation system that integrates Transformer-based motion generation with a transparent, user-editable JSON intermediate layer. The framework overcomes key limitations in prior sign language technologies by enabling direct user inspection and modification of sign segments, thus enhancing naturalness, expressiveness, and user agency. Leveraging a streaming Conformer encoder and autoregressive Transformer-MDN decoder, the system synchronizes spoken input into upper-body and facial motion for 3D avatar rendering. Edits and user ratings feed into a human-in-the-loop optimization loop for continuous improvement. Experiments with 20 deaf signers and 5 interpreters show that the editable interface and participatory feedback significantly improve comprehension, naturalness, usability, and trust, while lowering cognitive load. With sub-20 ms per-frame inference on standard hardware, the system is ready for real-time communication and education. This work illustrates how technical and participatory innovation together enable accessible, explainable, and user-adaptive AI for sign language technology. 

**Abstract (ZH)**: 基于人类中心、实时、用户自适应的语音到手语动画系统：Transformer基于的动作生成与透明、用户可编辑的JSON中间层整合 

---
# Accurate and scalable exchange-correlation with deep learning 

**Title (ZH)**: 深度学习驱动的准确且可扩展的交换相关泛函 

**Authors**: Giulia Luise, Chin-Wei Huang, Thijs Vogels, Derk P. Kooi, Sebastian Ehlert, Stephanie Lanius, Klaas J. H. Giesbertz, Amir Karton, Deniz Gunceler, Megan Stanley, Wessel P. Bruinsma, Lin Huang, Xinran Wei, José Garrido Torres, Abylay Katbashev, Bálint Máté, Sékou-Oumar Kaba, Roberto Sordillo, Yingrong Chen, David B. Williams-Young, Christopher M. Bishop, Jan Hermann, Rianne van den Berg, Paola Gori-Giorgi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14665)  

**Abstract**: Density Functional Theory (DFT) is the most widely used electronic structure method for predicting the properties of molecules and materials. Although DFT is, in principle, an exact reformulation of the Schrödinger equation, practical applications rely on approximations to the unknown exchange-correlation (XC) functional. Most existing XC functionals are constructed using a limited set of increasingly complex, hand-crafted features that improve accuracy at the expense of computational efficiency. Yet, no current approximation achieves the accuracy and generality for predictive modeling of laboratory experiments at chemical accuracy -- typically defined as errors below 1 kcal/mol. In this work, we present Skala, a modern deep learning-based XC functional that bypasses expensive hand-designed features by learning representations directly from data. Skala achieves chemical accuracy for atomization energies of small molecules while retaining the computational efficiency typical of semi-local DFT. This performance is enabled by training on an unprecedented volume of high-accuracy reference data generated using computationally intensive wavefunction-based methods. Notably, Skala systematically improves with additional training data covering diverse chemistry. By incorporating a modest amount of additional high-accuracy data tailored to chemistry beyond atomization energies, Skala achieves accuracy competitive with the best-performing hybrid functionals across general main group chemistry, at the cost of semi-local DFT. As the training dataset continues to expand, Skala is poised to further enhance the predictive power of first-principles simulations. 

**Abstract (ZH)**: 基于深度学习的Skala交换相关泛函：通过直接从数据学习绕过昂贵的手工设计特征实现化学精度 

---
# Rigor in AI: Doing Rigorous AI Work Requires a Broader, Responsible AI-Informed Conception of Rigor 

**Title (ZH)**: AI的严谨性：进行严谨的AI工作需要一种更广泛且负责任的AI驱动的严谨性观念 

**Authors**: Alexandra Olteanu, Su Lin Blodgett, Agathe Balayn, Angelina Wang, Fernando Diaz, Flavio du Pin Calmon, Margaret Mitchell, Michael Ekstrand, Reuben Binns, Solon Barocas  

**Link**: [PDF](https://arxiv.org/pdf/2506.14652)  

**Abstract**: In AI research and practice, rigor remains largely understood in terms of methodological rigor -- such as whether mathematical, statistical, or computational methods are correctly applied. We argue that this narrow conception of rigor has contributed to the concerns raised by the responsible AI community, including overblown claims about AI capabilities. Our position is that a broader conception of what rigorous AI research and practice should entail is needed. We believe such a conception -- in addition to a more expansive understanding of (1) methodological rigor -- should include aspects related to (2) what background knowledge informs what to work on (epistemic rigor); (3) how disciplinary, community, or personal norms, standards, or beliefs influence the work (normative rigor); (4) how clearly articulated the theoretical constructs under use are (conceptual rigor); (5) what is reported and how (reporting rigor); and (6) how well-supported the inferences from existing evidence are (interpretative rigor). In doing so, we also aim to provide useful language and a framework for much-needed dialogue about the AI community's work by researchers, policymakers, journalists, and other stakeholders. 

**Abstract (ZH)**: 在人工智能研究与实践中，严谨性主要仍被理解为方法论严谨性——例如，数学、统计或计算方法是否正确应用。我们认为，这一狭隘的严谨性概念导致了负责任的人工智能社区提出的一些担忧，包括夸大人工智能的能力。我们认为，需要一种更广泛的关于严谨性人工智能研究与实践应包含的内容的概念。我们相信，除了更广泛地理解（1）方法论严谨性——还应包括（2）如何被现有知识指导所做的工作（知识论严谨性）；（3）学科、社区或个人规范、标准或信念如何影响工作（规范性严谨性）；（4）正在使用的基本理论概念是否清晰阐明（概念性严谨性）；（5）报告的内容及其方式（报告严谨性）；以及（6）现有证据得出的推论是否得到充分支持（解释性严谨性）。我们还希望为此提供有用的语汇和框架，以促进研究人员、政策制定者、记者和其他利益相关者非常需要的关于人工智能社区工作的对话。 

---
# Navigating the growing field of research on AI for software testing -- the taxonomy for AI-augmented software testing and an ontology-driven literature survey 

**Title (ZH)**: 导航不断增长的AI在软件测试领域研究——基于本体的AI增强软件测试分类与文献综述 

**Authors**: Ina K. Schieferdecker  

**Link**: [PDF](https://arxiv.org/pdf/2506.14640)  

**Abstract**: In industry, software testing is the primary method to verify and validate the functionality, performance, security, usability, and so on, of software-based systems. Test automation has gained increasing attention in industry over the last decade, following decades of intense research into test automation and model-based testing. However, designing, developing, maintaining and evolving test automation is a considerable effort. Meanwhile, AI's breakthroughs in many engineering fields are opening up new perspectives for software testing, for both manual and automated testing. This paper reviews recent research on AI augmentation in software test automation, from no automation to full automation. It also discusses new forms of testing made possible by AI. Based on this, the newly developed taxonomy, ai4st, is presented and used to classify recent research and identify open research questions. 

**Abstract (ZH)**: 在工业领域，软件测试是验证和确认基于软件系统的功能、性能、安全性和易用性等的主要方法。过去十年中，随着对测试自动化和模型驱动测试多年深入研究的进展，软件测试自动化逐渐获得广泛关注。然而，设计、开发、维护和演化测试自动化是一项巨大的努力。同时，AI在许多工程领域的突破为软件测试，无论是手动测试还是自动化测试，开启了新的视角。本文回顾了从无自动化到完全自动化的AI在软件测试自动化中的最新研究，讨论了AI使可能的新测试形式，并基于此，介绍了新的分类框架ai4st，用于分类最新研究并确定开放研究问题。 

---
# ACM Survey Draft on Formalising Software Requirements with Large Language Models 

**Title (ZH)**: ACM调查草稿：使用大型语言模型形式化软件需求 

**Authors**: Arshad Beg, Diarmuid O'Donoghue, Rosemary Monahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14627)  

**Abstract**: This draft is a working document, having a summary of nighty-four (94) papers with additional sections on Traceability of Software Requirements (Section 4), Formal Methods and Its Tools (Section 5), Unifying Theories of Programming (UTP) and Theory of Institutions (Section 6). Please refer to abstract of [7,8]. Key difference of this draft from our recently anticipated ones with similar titles, i.e. AACS 2025 [7] and SAIV 2025 [8] is:
[7] is a two page submission to ADAPT Annual Conference, Ireland. Submitted on 18th of March, 2025, it went through the light-weight blind review and accepted for poster presentation. Conference was held on 15th of May, 2025.
[8] is a nine page paper with additional nine pages of references and summary tables, submitted to Symposium on AI Verification (SAIV 2025) on 24th of April, 2025. It went through rigorous review process. The uploaded version on arXiv.org [8] is the improved one of the submission, after addressing the specific suggestions to improve the paper. 

**Abstract (ZH)**: 这是一部工作文档，包含了九十四篇论文的摘要，并增设了软件需求可追溯性（第4节）、形式方法及其工具（第5节）、编程统一理论（UTP）与机构理论（第6节）的相关部分。请参见[7,8]的摘要。与我们先前类似标题的预期文件，即AACS 2025 [7]和SAIV 2025 [8]的不同之处在于： 

---
# Low-code to fight climate change: the Climaborough project 

**Title (ZH)**: 低代码对抗气候变化：Climaborough项目 

**Authors**: Aaron Conrardy, Armen Sulejmani, Cindy Guerlain, Daniele Pagani, David Hick, Matteo Satta, Jordi Cabot  

**Link**: [PDF](https://arxiv.org/pdf/2506.14623)  

**Abstract**: The EU-funded Climaborough project supports European cities to achieve carbon neutrality by 2030. Eleven cities in nine countries will deploy in real conditions products and services fostering climate transition in their local environment. The Climaborough City Platform is being developed to monitor the cities' overall progress towards their climate goals by aggregating historic and real-time data and displaying the results in user-friendly dashboards that will be used by non-technical experts to evaluate the effectiveness of local experimental initiatives, identify those that yield significant impact, and assess the potential consequences of scaling them up to a broader level. In this paper, we explain how we have put in place a low-code/no-code strategy in Climaborough in response to the project's aim to quickly deploy climate dashboards. A low-code strategy is used to accelerate the development of the dashboards. The dashboards embed a no-code philosophy that enables all types of citizen profiles to configure and adapt the dashboard to their specific needs. 

**Abstract (ZH)**: 欧盟资助的Climaborough项目支持欧洲城市在2030年前实现碳中和 

---
# Object-Centric Neuro-Argumentative Learning 

**Title (ZH)**: 对象中心神经论证学习 

**Authors**: Abdul Rahman Jacob, Avinash Kori, Emanuele De Angelis, Ben Glocker, Maurizio Proietti, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.14577)  

**Abstract**: Over the last decade, as we rely more on deep learning technologies to make critical decisions, concerns regarding their safety, reliability and interpretability have emerged. We introduce a novel Neural Argumentative Learning (NAL) architecture that integrates Assumption-Based Argumentation (ABA) with deep learning for image analysis. Our architecture consists of neural and symbolic components. The former segments and encodes images into facts using object-centric learning, while the latter applies ABA learning to develop ABA frameworks enabling predictions with images. Experiments on synthetic data show that the NAL architecture can be competitive with a state-of-the-art alternative. 

**Abstract (ZH)**: 近年来，随着我们越来越多地依赖深度学习技术进行关键决策，对其安全、可靠性和可解释性的问题日益凸显。我们提出了一种新颖的神经论辩学习（NAL）架构，将基于假设的论辩（ABA）与深度学习结合用于图像分析。该架构包含神经和符号两个组件。前者使用对象中心学习对图像进行分割并编码为事实，后者应用ABA学习构建ABA框架，以实现基于图像的预测。在合成数据上的实验表明，NAL架构可以与当前最先进的替代方案竞争。 

---
# Controlling Context: Generative AI at Work in Integrated Circuit Design and Other High-Precision Domains 

**Title (ZH)**: 控制语境：生成式AI在集成电路设计及其他高精度领域中的应用 

**Authors**: Emanuel Moss, Elizabeth Watkins, Christopher Persaud, Passant Karunaratne, Dawn Nafus  

**Link**: [PDF](https://arxiv.org/pdf/2506.14567)  

**Abstract**: Generative AI tools have become more prevalent in engineering workflows, particularly through chatbots and code assistants. As the perceived accuracy of these tools improves, questions arise about whether and how those who work in high-precision domains might maintain vigilance for errors, and what other aspects of using such tools might trouble their work. This paper analyzes interviews with hardware and software engineers, and their collaborators, who work in integrated circuit design to identify the role accuracy plays in their use of generative AI tools and what other forms of trouble they face in using such tools. The paper inventories these forms of trouble, which are then mapped to elements of generative AI systems, to conclude that controlling the context of interactions between engineers and the generative AI tools is one of the largest challenges they face. The paper concludes with recommendations for mitigating this form of trouble by increasing the ability to control context interactively. 

**Abstract (ZH)**: 生成式AI工具在工程工作流中的应用日益普遍，尤其是在聊天机器人和代码助手方面的应用。随着这些工具被认为准确性的提高，人们不禁思考高精度领域工作的从业者如何保持对潜在错误的警惕，以及使用此类工具还会遇到哪些其他问题。本文通过访谈集成电路设计的硬件和软件工程师及其合作者，分析生成式AI工具在他们工作中的作用，以及使用此类工具还会遇到的其他问题。本文列出了这些问题，并将其映射到生成式AI系统的各个要素，得出控制工程师与生成式AI工具交互的上下文是他们面临的一大挑战。文章最后提出建议，通过增强交互式的上下文控制能力来减轻这种问题。 

---
# Aligning Evaluation with Clinical Priorities: Calibration, Label Shift, and Error Costs 

**Title (ZH)**: 临床优先级对齐的评估：校准、标签转移和错误成本 

**Authors**: Gerardo A. Flores, Alyssa H. Smith, Julia A. Fukuyama, Ashia C. Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.14540)  

**Abstract**: Machine learning-based decision support systems are increasingly deployed in clinical settings, where probabilistic scoring functions are used to inform and prioritize patient management decisions. However, widely used scoring rules, such as accuracy and AUC-ROC, fail to adequately reflect key clinical priorities, including calibration, robustness to distributional shifts, and sensitivity to asymmetric error costs. In this work, we propose a principled yet practical evaluation framework for selecting calibrated thresholded classifiers that explicitly accounts for the uncertainty in class prevalences and domain-specific cost asymmetries often found in clinical settings. Building on the theory of proper scoring rules, particularly the Schervish representation, we derive an adjusted variant of cross-entropy (log score) that averages cost-weighted performance over clinically relevant ranges of class balance. The resulting evaluation is simple to apply, sensitive to clinical deployment conditions, and designed to prioritize models that are both calibrated and robust to real-world variations. 

**Abstract (ZH)**: 基于机器学习的决策支持系统在临床环境中越来越广泛部署，其中概率评分函数用于指导和优先处理患者的管理决策。然而，常用的评分规则，如准确率和AUC-ROC，无法充分反映临床优先考虑的关键因素，包括校准、分布变化的稳健性和不对称错误成本的敏感性。在本文中，我们提出了一种既原则性强又实用的评估框架，用于选择校准的分类器阈值，该框架明确考虑了类流行率的不确定性以及临床环境中常见的领域特定成本不对称性。基于适当评分规则的理论，特别是舒尔维希表示法，我们推导出一种调整后的交叉熵（对数得分）变体，它在临床相关范围内的类平衡上平均成本加权性能。该评估方法简单易行，对临床部署条件敏感，并旨在优先考虑既校准又对现实世界变化具有稳健性的模型。 

---
# Complete Characterization for Adjustment in Summary Causal Graphs of Time Series 

**Title (ZH)**: 时间序列摘要因果图中调整的完全表征 

**Authors**: Clément Yvernes, Emilie Devijver, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2506.14534)  

**Abstract**: The identifiability problem for interventions aims at assessing whether the total causal effect can be written with a do-free formula, and thus be estimated from observational data only. We study this problem, considering multiple interventions, in the context of time series when only an abstraction of the true causal graph, in the form of a summary causal graph, is available. We propose in particular both necessary and sufficient conditions for the adjustment criterion, which we show is complete in this setting, and provide a pseudo-linear algorithm to decide whether the query is identifiable or not. 

**Abstract (ZH)**: 干预的可识别性问题旨在评估总因果效应是否可以用do-free公式表示，从而仅从观察数据中进行估计。在仅拥有真实因果图的抽象形式（总结因果图）的时间序列背景下，我们研究了这一问题，特别是在考虑多个干预的情况下。我们特别提出了调整准则的必要且充分条件，并表明在该背景下该准则完备。此外，我们提供了一个伪线性算法来决定查询是否可识别。 

---
# Sharp Generalization Bounds for Foundation Models with Asymmetric Randomized Low-Rank Adapters 

**Title (ZH)**: 具有不对称随机低秩适配器的 foundation 模型的精确泛化边界 

**Authors**: Anastasis Kratsios, Tin Sum Cheng, Aurelien Lucchi, Haitz Sáez de Ocáriz Borde  

**Link**: [PDF](https://arxiv.org/pdf/2506.14530)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a widely adopted parameter-efficient fine-tuning (PEFT) technique for foundation models. Recent work has highlighted an inherent asymmetry in the initialization of LoRA's low-rank factors, which has been present since its inception and was presumably derived experimentally. This paper focuses on providing a comprehensive theoretical characterization of asymmetric LoRA with frozen random factors. First, while existing research provides upper-bound generalization guarantees based on averages over multiple experiments, the behaviour of a single fine-tuning run with specific random factors remains an open question. We address this by investigating the concentration of the typical LoRA generalization gap around its mean. Our main upper bound reveals a sample complexity of $\tilde{\mathcal{O}}\left(\frac{\sqrt{r}}{\sqrt{N}}\right)$ with high probability for rank $r$ LoRAs trained on $N$ samples. Additionally, we also determine the fundamental limits in terms of sample efficiency, establishing a matching lower bound of $\mathcal{O}\left(\frac{1}{\sqrt{N}}\right)$. By more closely reflecting the practical scenario of a single fine-tuning run, our findings offer crucial insights into the reliability and practicality of asymmetric LoRA. 

**Abstract (ZH)**: Asymmetric Low-Rank Adaptation (LoRA) with Frozen Random Factors: A Comprehensive Theoretical Characterization 

---
# Leveraging External Factors in Household-Level Electrical Consumption Forecasting using Hypernetworks 

**Title (ZH)**: 基于超网络在户级电气消费预测中利用外部因素的研究 

**Authors**: Fabien Bernier, Maxime Cordy, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2506.14472)  

**Abstract**: Accurate electrical consumption forecasting is crucial for efficient energy management and resource allocation. While traditional time series forecasting relies on historical patterns and temporal dependencies, incorporating external factors -- such as weather indicators -- has shown significant potential for improving prediction accuracy in complex real-world applications. However, the inclusion of these additional features often degrades the performance of global predictive models trained on entire populations, despite improving individual household-level models. To address this challenge, we found that a hypernetwork architecture can effectively leverage external factors to enhance the accuracy of global electrical consumption forecasting models, by specifically adjusting the model weights to each consumer.
We collected a comprehensive dataset spanning two years, comprising consumption data from over 6000 luxembourgish households and corresponding external factors such as weather indicators, holidays, and major local events. By comparing various forecasting models, we demonstrate that a hypernetwork approach outperforms existing methods when associated to external factors, reducing forecasting errors and achieving the best accuracy while maintaining the benefits of a global model. 

**Abstract (ZH)**: 准确的电能消耗forecasting对于高效的能源管理和资源配置至关重要。虽然传统的时序forecasting依赖历史模式和时间依赖性，但在复杂的实际应用中，将外部因素（如天气指标）纳入其中已显示出显著提高预测精度的潜力。然而，尽管外部因素的加入可以提升单个家庭级别的模型表现，但通常会降低全局预测模型的表现。为应对这一挑战，我们发现，超网络架构能够通过针对性地调整模型权重来有效利用外部因素，从而增强全局电能消耗 forecasting 模型的准确性。通过对比各种 forecasting 模型，我们证明，结合外部因素的超网络方法优于现有方法，能够减少 forecasting 错误，并在保持全局模型优势的同时实现最佳精度。 

---
# A Scalable Hybrid Training Approach for Recurrent Spiking Neural Networks 

**Title (ZH)**: 可扩展的混合训练方法用于递归神经脉冲网络 

**Authors**: Maximilian Baronig, Yeganeh Bahariasl, Ozan Özdenizci, Robert Legenstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.14464)  

**Abstract**: Recurrent spiking neural networks (RSNNs) can be implemented very efficiently in neuromorphic systems. Nevertheless, training of these models with powerful gradient-based learning algorithms is mostly performed on standard digital hardware using Backpropagation through time (BPTT). However, BPTT has substantial limitations. It does not permit online training and its memory consumption scales linearly with the number of computation steps. In contrast, learning methods using forward propagation of gradients operate in an online manner with a memory consumption independent of the number of time steps. These methods enable SNNs to learn from continuous, infinite-length input sequences. Yet, slow execution speed on conventional hardware as well as inferior performance has hindered their widespread application. In this work, we introduce HYbrid PRopagation (HYPR) that combines the efficiency of parallelization with approximate online forward learning. Our algorithm yields high-throughput online learning through parallelization, paired with constant, i.e., sequence length independent, memory demands. HYPR enables parallelization of parameter update computation over the sub sequences for RSNNs consisting of almost arbitrary non-linear spiking neuron models. We apply HYPR to networks of spiking neurons with oscillatory subthreshold dynamics. We find that this type of neuron model is particularly well trainable by HYPR, resulting in an unprecedentedly low task performance gap between approximate forward gradient learning and BPTT. 

**Abstract (ZH)**: 杂合前向传播（HYPR）：结合并行化效率与近似在线前向学习 

---
# Hamiltonian Formalism for Comparing Quantum and Classical Intelligence 

**Title (ZH)**: 量子与经典智能比较的哈密顿正则形式分析 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2506.14456)  

**Abstract**: The prospect of AGI instantiated on quantum substrates motivates the development of mathematical frameworks that enable direct comparison of their operation in classical and quantum environments. To this end, we introduce a Hamiltonian formalism for describing classical and quantum AGI tasks as a means of contrasting their interaction with the environment. We propose a decomposition of AGI dynamics into Hamiltonian generators for core functions such as induction, reasoning, recursion, learning, measurement, and memory. This formalism aims to contribute to the development of a precise mathematical language for how quantum and classical agents differ via environmental interaction. 

**Abstract (ZH)**: 基于量子底层实现的AGI的前景促使我们发展数学框架以直接对比其在经典和量子环境中的操作。为此，我们引入了一种哈密顿 formalism 来描述经典和量子AGI任务，以便对比它们与环境的交互。我们提议将AGI动力学分解为哈密顿生成器，用于核心功能如归纳、推理、递归、学习、测量和记忆。此 formalism 目标是为通过环境交互如何量子和经典代理的不同提供一种精确的数学语言。 

---
# Model compression using knowledge distillation with integrated gradients 

**Title (ZH)**: 使用集成梯度的知识蒸馏模型压缩 

**Authors**: David E. Hernandez, Jose Chang, Torbjörn E. M. Nordling  

**Link**: [PDF](https://arxiv.org/pdf/2506.14440)  

**Abstract**: Model compression is critical for deploying deep learning models on resource-constrained devices. We introduce a novel method enhancing knowledge distillation with integrated gradients (IG) as a data augmentation strategy. Our approach overlays IG maps onto input images during training, providing student models with deeper insights into teacher models' decision-making processes. Extensive evaluation on CIFAR-10 demonstrates that our IG-augmented knowledge distillation achieves 92.6% testing accuracy with a 4.1x compression factor-a significant 1.1 percentage point improvement ($p<0.001$) over non-distilled models (91.5%). This compression reduces inference time from 140 ms to 13 ms. Our method precomputes IG maps before training, transforming substantial runtime costs into a one-time preprocessing step. Our comprehensive experiments include: (1) comparisons with attention transfer, revealing complementary benefits when combined with our approach; (2) Monte Carlo simulations confirming statistical robustness; (3) systematic evaluation of compression factor versus accuracy trade-offs across a wide range (2.2x-1122x); and (4) validation on an ImageNet subset aligned with CIFAR-10 classes, demonstrating generalisability beyond the initial dataset. These extensive ablation studies confirm that IG-based knowledge distillation consistently outperforms conventional approaches across varied architectures and compression ratios. Our results establish this framework as a viable compression technique for real-world deployment on edge devices while maintaining competitive accuracy. 

**Abstract (ZH)**: 一种结合集成梯度的知识蒸馏方法在资源受限设备上部署深度学习模型的关键技术 

---
# sHGCN: Simplified hyperbolic graph convolutional neural networks 

**Title (ZH)**: SHGCN: 简化双曲图卷积神经网络 

**Authors**: Pol Arévalo, Alexis Molina, Álvaro Ciudad  

**Link**: [PDF](https://arxiv.org/pdf/2506.14438)  

**Abstract**: Hyperbolic geometry has emerged as a powerful tool for modeling complex, structured data, particularly where hierarchical or tree-like relationships are present. By enabling embeddings with lower distortion, hyperbolic neural networks offer promising alternatives to Euclidean-based models for capturing intricate data structures. Despite these advantages, they often face performance challenges, particularly in computational efficiency and tasks requiring high precision. In this work, we address these limitations by simplifying key operations within hyperbolic neural networks, achieving notable improvements in both runtime and performance. Our findings demonstrate that streamlined hyperbolic operations can lead to substantial gains in computational speed and predictive accuracy, making hyperbolic neural networks a more viable choice for a broader range of applications. 

**Abstract (ZH)**: 双曲几何已成为建模复杂结构数据的强大工具，尤其是在存在层次或树形关系的情况下。通过提供较低失真的嵌入，双曲神经网络为捕捉复杂数据结构提供了欧几里得模型的有前途的替代方案。尽管具有这些优势，它们通常在计算效率和需要高精度的任务方面面临性能挑战。在本文中，我们通过简化双曲神经网络中的关键操作，实现了显著的运行时和性能改进。我们的研究表明，优化的双曲操作可以带来計算速度和预测准确性上的重大提升，使双曲神经网络成为更广泛应用的可行选择。 

---
# Unifying Streaming and Non-streaming Zipformer-based ASR 

**Title (ZH)**: 基于Zipformer的统一流式与非流式ASR 

**Authors**: Bidisha Sharma, Karthik Pandia Durai, Shankar Venkatesan, Jeena J Prakash, Shashi Kumar, Malolan Chetlur, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2506.14434)  

**Abstract**: There has been increasing interest in unifying streaming and non-streaming automatic speech recognition (ASR) models to reduce development, training, and deployment costs. We present a unified framework that trains a single end-to-end ASR model for both streaming and non-streaming applications, leveraging future context information. We propose to use dynamic right-context through the chunked attention masking in the training of zipformer-based ASR models. We demonstrate that using right-context is more effective in zipformer models compared to other conformer models due to its multi-scale nature. We analyze the effect of varying the number of right-context frames on accuracy and latency of the streaming ASR models. We use Librispeech and large in-house conversational datasets to train different versions of streaming and non-streaming models and evaluate them in a production grade server-client setup across diverse testsets of different domains. The proposed strategy reduces word error by relative 7.9\% with a small degradation in user-perceived latency. By adding more right-context frames, we are able to achieve streaming performance close to that of non-streaming models. Our approach also allows flexible control of the latency-accuracy tradeoff according to customers requirements. 

**Abstract (ZH)**: 统一处理流式和非流式自动语音识别模型以降低开发、训练和部署成本的研究 

---
# Is Selection All You Need in Differential Evolution? 

**Title (ZH)**: 差分进化中仅选择是否足够？ 

**Authors**: Tomofumi Kitamura, Alex Fukunaga  

**Link**: [PDF](https://arxiv.org/pdf/2506.14425)  

**Abstract**: Differential Evolution (DE) is a widely used evolutionary algorithm for black-box optimization problems. However, in modern DE implementations, a major challenge lies in the limited population diversity caused by the fixed population size enforced by the generational replacement. Population size is a critical control parameter that significantly affects DE performance. Larger populations inherently contain a more diverse set of individuals, thereby facilitating broader exploration of the search space. Conversely, when the maximum evaluation budgets is constrained, smaller populations focusing on a limited number of promising candidates may be more suitable. Many state-of-the-art DE variants incorporate an archive mechanism, in which a subset of discarded individuals is preserved in an archive during generation replacement and reused in mutation operations. However, maintaining what is essentially a secondary population via an archive introduces additional design considerations, such as policies for insertion, deletion, and appropriate sizing. To address these limitations, we propose a novel DE framework called Unbounded Differential Evolution (UDE), which adds all generated candidates to the population without discarding any individual based on fitness. Unlike conventional DE, which removes inferior individuals during generational replacement, UDE eliminates replacement altogether, along with the associated complexities of archive management and dynamic population sizing. UDE represents a fundamentally new approach to DE, relying solely on selection mechanisms and enabling a more straightforward yet powerful search algorithm. 

**Abstract (ZH)**: 无界差分进化（UDE）：一种新的差分进化框架 

---
# ImpliRet: Benchmarking the Implicit Fact Retrieval Challenge 

**Title (ZH)**: 隐含事实检索挑战的基准测试 

**Authors**: Zeinab Sadat Taghavi, Ali Modarressi, Yunpu Ma, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2506.14407)  

**Abstract**: Retrieval systems are central to many NLP pipelines, but often rely on surface-level cues such as keyword overlap and lexical semantic similarity. To evaluate retrieval beyond these shallow signals, recent benchmarks introduce reasoning-heavy queries; however, they primarily shift the burden to query-side processing techniques -- like prompting or multi-hop retrieval -- that can help resolve complexity. In contrast, we present ImpliRet, a benchmark that shifts the reasoning challenge to document-side processing: The queries are simple, but relevance depends on facts stated implicitly in documents through temporal (e.g., resolving "two days ago"), arithmetic, and world knowledge relationships. We evaluate a range of sparse and dense retrievers, all of which struggle in this setting: the best nDCG@10 is only 15.07%. We also test whether long-context models can overcome this limitation. But even with a short context of only ten documents, including the positive document, GPT-4.1 scores only 35.06%, showing that document-side reasoning remains a challenge. Our codes are available at this http URL. 

**Abstract (ZH)**: 检索系统是许多NLP管道的核心，但often依赖于关键词重叠和词义表面相似性等表面级线索。为了超越这些浅层信号评估检索，近期基准引入了重 reasoning 的查询；然而，它们主要将负担转移到查询端处理技术上——如提示或多跳检索——这些技术可以帮助解决复杂性。相比之下，我们提出了ImpliRet基准，将 reasoning 挑战转移到文档端处理：查询简单，但相关性取决于文档中通过时间（例如，“两天前”）、算术和世界知识关系隐含陈述的事实。我们评估了稀疏和密集检索器的各种方法，所有方法在这项任务中都表现不佳：最佳nDCG@10仅为15.07%。我们还测试了长语境模型是否可以克服这一限制。即使只有包括正文档在内的十个文档的短语境，GPT-4.1的得分也只有35.06%，表明文档端 reasoning 仍然是一项挑战。我们的代码可在以下网址获取。 

---
# ResNets Are Deeper Than You Think 

**Title (ZH)**: ResNets 深于你所想象 

**Authors**: Christian H.X. Ali Mehmeti-Göpel, Michael Wand  

**Link**: [PDF](https://arxiv.org/pdf/2506.14386)  

**Abstract**: Residual connections remain ubiquitous in modern neural network architectures nearly a decade after their introduction. Their widespread adoption is often credited to their dramatically improved trainability: residual networks train faster, more stably, and achieve higher accuracy than their feedforward counterparts. While numerous techniques, ranging from improved initialization to advanced learning rate schedules, have been proposed to close the performance gap between residual and feedforward networks, this gap has persisted. In this work, we propose an alternative explanation: residual networks do not merely reparameterize feedforward networks, but instead inhabit a different function space. We design a controlled post-training comparison to isolate generalization performance from trainability; we find that variable-depth architectures, similar to ResNets, consistently outperform fixed-depth networks, even when optimization is unlikely to make a difference. These results suggest that residual connections confer performance advantages beyond optimization, pointing instead to a deeper inductive bias aligned with the structure of natural data. 

**Abstract (ZH)**: 残差连接在近十年后仍然在现代神经网络架构中无处不在。它们的广泛采用通常归功于显著提高的可训练性：残差网络训练更快、更稳定，并且准确率高于其前向传播 counterparts。虽然已经提出了许多技术，从改进的初始化到先进的学习率调度，以缩小残差网络和前向传播网络之间的性能差距，但这一差距一直存在。在本工作中，我们提出了一种替代解释：残差连接不仅重新参数化前向传播网络，而是存在于不同的函数空间中。我们设计了一个受控的后训练比较以隔离泛化性能与可训练性；我们发现，类似ResNets的可变深度架构始终优于固定深度网络，即使优化不太可能产生影响。这些结果表明，残差连接提供了超越优化的性能优势，而是指向与自然数据结构一致的更深层次的归纳偏置。 

---
# IntelliLung: Advancing Safe Mechanical Ventilation using Offline RL with Hybrid Actions and Clinically Aligned Rewards 

**Title (ZH)**: IntelliLung: 利用混合动作和临床对齐奖励的离线强化学习提升安全机械通气 

**Authors**: Muhammad Hamza Yousuf, Jason Li, Sahar Vahdati, Raphael Theilen, Jakob Wittenstein, Jens Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.14375)  

**Abstract**: Invasive mechanical ventilation (MV) is a life-sustaining therapy for critically ill patients in the intensive care unit (ICU). However, optimizing its settings remains a complex and error-prone process due to patient-specific variability. While Offline Reinforcement Learning (RL) shows promise for MV control, current stateof-the-art (SOTA) methods struggle with the hybrid (continuous and discrete) nature of MV actions. Discretizing the action space limits available actions due to exponential growth in combinations and introduces distribution shifts that can compromise safety. In this paper, we propose optimizations that build upon prior work in action space reduction to address the challenges of discrete action spaces. We also adapt SOTA offline RL algorithms (IQL and EDAC) to operate directly on hybrid action spaces, thereby avoiding the pitfalls of discretization. Additionally, we introduce a clinically grounded reward function based on ventilator-free days and physiological targets, which provides a more meaningful optimization objective compared to traditional sparse mortality-based rewards. Our findings demonstrate that AI-assisted MV optimization may enhance patient safety and enable individualized lung support, representing a significant advancement toward intelligent, data-driven critical care solutions. 

**Abstract (ZH)**: 侵入性机械通气(MV)是重症监护单元(ICU)中重症患者的生命维持疗法。然而，优化其设置仍然是一个复杂且容易出错的过程，由于患者特定的变异性所致。尽管离线强化学习(OFFLINE RL)在MV控制方面显示出潜力，但当前最先进的(SOTA)方法难以应对MV动作的混合（连续和离散）特性。离散化动作空间会限制可用动作，由于组合数的指数增长而产生，并且引入可能损害安全性的分布偏移。在本文中，我们提出了基于先前动作空间缩减工作的优化方法，以解决离散动作空间的挑战。我们还适应当前最先进的离线RL算法（IQL和EDAC），使其可以直接在混合动作空间上运行，从而避免离散化带来的问题。此外，我们引入了一种基于临床目标（如无通气日数和生理目标）的奖励函数，相比于传统的稀疏死亡率基线奖励，提供了更有意义的优化目标。我们的研究结果表明，人工智能辅助的MV优化可能增强患者安全，并实现个性化的肺支持，代表了智能、数据驱动的重症监护解决方案的重要进步。 

---
# Adjustment for Confounding using Pre-Trained Representations 

**Title (ZH)**: 使用预训练表示调整混杂因素 

**Authors**: Rickmer Schulte, David Rügamer, Thomas Nagler  

**Link**: [PDF](https://arxiv.org/pdf/2506.14329)  

**Abstract**: There is growing interest in extending average treatment effect (ATE) estimation to incorporate non-tabular data, such as images and text, which may act as sources of confounding. Neglecting these effects risks biased results and flawed scientific conclusions. However, incorporating non-tabular data necessitates sophisticated feature extractors, often in combination with ideas of transfer learning. In this work, we investigate how latent features from pre-trained neural networks can be leveraged to adjust for sources of confounding. We formalize conditions under which these latent features enable valid adjustment and statistical inference in ATE estimation, demonstrating results along the example of double machine learning. We discuss critical challenges inherent to latent feature learning and downstream parameter estimation arising from the high dimensionality and non-identifiability of representations. Common structural assumptions for obtaining fast convergence rates with additive or sparse linear models are shown to be unrealistic for latent features. We argue, however, that neural networks are largely insensitive to these issues. In particular, we show that neural networks can achieve fast convergence rates by adapting to intrinsic notions of sparsity and dimension of the learning problem. 

**Abstract (ZH)**: 非表格数据的潜特征在平均处理效应估计中的应用：转移学习与泛化的挑战 

---
# orGAN: A Synthetic Data Augmentation Pipeline for Simultaneous Generation of Surgical Images and Ground Truth Labels 

**Title (ZH)**: orGAN：一种同时生成手术图像和ground truth标签的合成数据增强管道 

**Authors**: Niran Nataraj, Maina Sogabe, Kenji Kawashima  

**Link**: [PDF](https://arxiv.org/pdf/2506.14303)  

**Abstract**: Deep learning in medical imaging faces obstacles: limited data diversity, ethical issues, high acquisition costs, and the need for precise annotations. Bleeding detection and localization during surgery is especially challenging due to the scarcity of high-quality datasets that reflect real surgical scenarios. We propose orGAN, a GAN-based system for generating high-fidelity, annotated surgical images of bleeding. By leveraging small "mimicking organ" datasets, synthetic models that replicate tissue properties and bleeding, our approach reduces ethical concerns and data-collection costs. orGAN builds on StyleGAN with Relational Positional Learning to simulate bleeding events realistically and mark bleeding coordinates. A LaMa-based inpainting module then restores clean, pre-bleed visuals, enabling precise pixel-level annotations. In evaluations, a balanced dataset of orGAN and mimicking-organ images achieved 90% detection accuracy in surgical settings and up to 99% frame-level accuracy. While our development data lack diverse organ morphologies and contain intraoperative artifacts, orGAN markedly advances ethical, efficient, and cost-effective creation of realistic annotated bleeding datasets, supporting broader integration of AI in surgical practice. 

**Abstract (ZH)**: 深度学习在医学影像中面临障碍：数据多样性有限、伦理问题、高获取成本以及精确标注需求。手术中出血检测与定位尤为挑战，由于缺乏能反映实际手术场景的高质量数据集。我们提出orGAN，一种基于GAN的系统，用于生成高保真度、标注过的手术出血图像。通过利用小型“模拟器官”数据集，模拟具有组织属性和出血特征的合成模型，该方法减轻了伦理问题和数据收集成本。orGAN 基于具有关系位置学习的StyleGAN模拟真实的出血事件并在出血位置进行标记。基于LaMa的修补模块随后恢复了出血前的干净视觉效果，实现精确的像素级标注。在评估中，orGAN和模拟器官图像的均衡数据集在手术设置中实现了90%的检测准确率，并在帧级达到了99%的准确率。尽管我们的开发数据缺乏多样化的器官形态并包含术中伪影，但orGAN显著促进了伦理、高效且成本效益高的真实标注出血数据集的创建，支持更广泛地将AI集成到手术实践中。 

---
# Knowledge Adaptation as Posterior Correction 

**Title (ZH)**: 知识适应作为后验修正 

**Authors**: Mohammad Emtiyaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14262)  

**Abstract**: Adaptation is the holy grail of intelligence, but even the best AI models (like GPT) lack the adaptivity of toddlers. So the question remains: how can machines adapt quickly? Despite a lot of progress on model adaptation to facilitate continual and federated learning, as well as model merging, editing, unlearning, etc., little is known about the mechanisms by which machines can naturally learn to adapt in a similar way as humans and animals. Here, we show that all such adaptation methods can be seen as different ways of `correcting' the approximate posteriors. More accurate posteriors lead to smaller corrections, which in turn imply quicker adaptation. The result is obtained by using a dual-perspective of the Bayesian Learning Rule of Khan and Rue (2023) where interference created during adaptation is characterized by the natural-gradient mismatch over the past data. We present many examples to demonstrate the use of posterior-correction as a natural mechanism for the machines to learn to adapt quickly. 

**Abstract (ZH)**: 适应是智能的圣杯，但即使是最优秀的AI模型（如GPT）也缺乏学龄前儿童的适应性。因此，问题依然存在：机器如何能够快速适应？尽管在模型适应以促进连续学习和联邦学习、模型合并、编辑和遗忘等方面取得了很大进展，但仍不清楚机器是如何自然学习以类似人类和动物的方式适应的机制。在这里，我们显示所有这些适应方法都可以被视为纠正近似后验概率的不同方式。更准确的后验概率导致更小的纠正，进而意味着更快的适应。该结果通过使用Khan和Rue（2023）的贝叶斯学习规则的双重视角获得，其中适应过程中产生的干扰通过过去数据的自然梯度不匹配来表征。我们通过许多例子展示了后验纠正作为一种自然机制，使机器能够快速学习适应。 

---
# TriGuard: Testing Model Safety with Attribution Entropy, Verification, and Drift 

**Title (ZH)**: TriGuard: 使用归因熵、验证和漂移测试模型安全性 

**Authors**: Dipesh Tharu Mahato, Rohan Poudel, Pramod Dhungana  

**Link**: [PDF](https://arxiv.org/pdf/2506.14217)  

**Abstract**: Deep neural networks often achieve high accuracy, but ensuring their reliability under adversarial and distributional shifts remains a pressing challenge. We propose TriGuard, a unified safety evaluation framework that combines (1) formal robustness verification, (2) attribution entropy to quantify saliency concentration, and (3) a novel Attribution Drift Score measuring explanation stability. TriGuard reveals critical mismatches between model accuracy and interpretability: verified models can still exhibit unstable reasoning, and attribution-based signals provide complementary safety insights beyond adversarial accuracy. Extensive experiments across three datasets and five architectures show how TriGuard uncovers subtle fragilities in neural reasoning. We further demonstrate that entropy-regularized training reduces explanation drift without sacrificing performance. TriGuard advances the frontier in robust, interpretable model evaluation. 

**Abstract (ZH)**: 深度神经网络往往能够实现高 accuracy，但确保其在对抗性和分布移变情况下的可靠性仍然是一个迫切的挑战。我们提出了 TriGuard，一个统一的安全评估框架，该框架结合了（1）形式化的鲁棒性验证，（2）归因熵以量化显着性集中度，以及（3）一种新型的归因漂移评分以衡量解释稳定性。TriGuard 暴露了模型 accuracy 和可解释性之间的关键不匹配：经过验证的模型仍然可能出现不稳定的推理，而基于归因的信号提供了超越对抗准确性的补充安全见解。广泛的实验证实在三个数据集中和五个架构上如何借助 TriGuard 揭示神经推理的细微脆弱性。我们进一步证明，使用熵正则化的训练可以减少解释漂移而不牺牲性能。TriGuard 推动了鲁棒性和可解释性模型评估的边界。 

---
# DiffusionBlocks: Blockwise Training for Generative Models via Score-Based Diffusion 

**Title (ZH)**: 基于分数扩散的块级训练生成模型方法 

**Authors**: Makoto Shing, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2506.14202)  

**Abstract**: Training large neural networks with end-to-end backpropagation creates significant memory bottlenecks, limiting accessibility to state-of-the-art AI research. We propose $\textit{DiffusionBlocks}$, a novel training framework that interprets neural network blocks as performing denoising operations in a continuous-time diffusion process. By partitioning the network into independently trainable blocks and optimizing noise level assignments based on equal cumulative probability mass, our approach achieves significant memory efficiency while maintaining competitive performance compared to traditional backpropagation in generative tasks. Experiments on image generation and language modeling tasks demonstrate memory reduction proportional to the number of blocks while achieving superior performance. DiffusionBlocks provides a promising pathway for democratizing access to large-scale neural network training with limited computational resources. 

**Abstract (ZH)**: 使用端到端反向传播训练大型神经网络会导致显著的内存瓶颈，限制了对先进人工智能研究的访问。我们提出了一种新型训练框架 $\textit{DiffusionBlocks}$，该框架将神经网络块解释为在连续时间扩散过程中执行去噪操作。通过将网络划分为独立训练的块，并基于等累计概率质量优化噪声水平分配，我们的方法在保持与传统反向传播竞争性能的同时实现了显著的内存效率。图像生成和语言建模任务的实验显示，内存减少量与块数成正比，并且性能更优。DiffusionBlocks 为在有限计算资源下 democratize 对大规模神经网络训练的访问提供了一条有前景的道路。 

---
# Balancing Caregiving and Self-Care: Exploring Mental Health Needs of Alzheimer's and Dementia Caregivers 

**Title (ZH)**: 平衡照护与自我照护：探索阿尔茨海默病和痴呆症照护者的心理健康需求 

**Authors**: Jiayue Melissa Shi, Keran Wang, Dong Whi Yoo, Ravi Karkar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.14196)  

**Abstract**: Alzheimer's Disease and Related Dementias (AD/ADRD) are progressive neurodegenerative conditions that impair memory, thought processes, and functioning. Family caregivers of individuals with AD/ADRD face significant mental health challenges due to long-term caregiving responsibilities. Yet, current support systems often overlook the evolving nature of their mental wellbeing needs. Our study examines caregivers' mental wellbeing concerns, focusing on the practices they adopt to manage the burden of caregiving and the technologies they use for support. Through semi-structured interviews with 25 family caregivers of individuals with AD/ADRD, we identified the key causes and effects of mental health challenges, and developed a temporal mapping of how caregivers' mental wellbeing evolves across three distinct stages of the caregiving journey. Additionally, our participants shared insights into improvements for existing mental health technologies, emphasizing the need for accessible, scalable, and personalized solutions that adapt to caregivers' changing needs over time. These findings offer a foundation for designing dynamic, stage-sensitive interventions that holistically support caregivers' mental wellbeing, benefiting both caregivers and care recipients. 

**Abstract (ZH)**: 阿尔茨海默病及相关痴呆症的家庭照护者面临长期照护带来的心理健康挑战，现有支持系统往往未能关注他们不断变化的心理健康需求。本研究通过半结构化访谈25名阿尔茨海默病及相关痴呆症患者的家庭照护者，探讨了他们的心理健康问题，分析了他们管理照护负担的实践及使用支持技术的情况，识别了心理健康挑战的主要原因及其影响，并绘制了家庭照护者在照护旅程三个不同阶段心理健康变化的跨时间映射图谱。此外，参与者还提供了关于现有心理健康技术改进建议，强调需要灵活、可扩展且个性化解决方案的迫切需求，以适应照护者不断变化的需求。研究结果为设计动态、阶段敏感的干预措施提供了基础，以全面支持照护者的心理健康，惠及照护者和接受照护者。 

---
# Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages 

**Title (ZH)**: 我们能否在没有实际混杂数据的情况下训练ASR系统以处理混杂语言？以新加坡语言为案例的研究 

**Authors**: Tuan Nguyen, Huy-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.14177)  

**Abstract**: Code-switching (CS), common in multilingual settings, presents challenges for ASR due to scarce and costly transcribed data caused by linguistic complexity. This study investigates building CS-ASR using synthetic CS data. We propose a phrase-level mixing method to generate synthetic CS data that mimics natural patterns. Utilizing monolingual augmented with synthetic phrase-mixed CS data to fine-tune large pretrained ASR models (Whisper, MMS, SeamlessM4T). This paper focuses on three under-resourced Southeast Asian language pairs: Malay-English (BM-EN), Mandarin-Malay (ZH-BM), and Tamil-English (TA-EN), establishing a new comprehensive benchmark for CS-ASR to evaluate the performance of leading ASR models. Experimental results show that the proposed training strategy enhances ASR performance on monolingual and CS tests, with BM-EN showing highest gains, then TA-EN and ZH-BM. This finding offers a cost-effective approach for CS-ASR development, benefiting research and industry. 

**Abstract (ZH)**: 代码转换（CS）在多语言环境中广泛存在，由于语言复杂性导致标注数据稀缺且成本高，这对ASR构成了挑战。本研究探讨了使用合成CS数据构建CS-ASR的方法。我们提出了一种短语级混合方法来生成模拟自然模式的合成CS数据。利用单语言数据与合成短语混合的CS数据对大规模预训练ASR模型（Whisper、MMS、SeamlessM4T）进行微调。本文重点关注三种资源不足的东南亚语言对：马来-英语（BM-EN）、 Mandarin-马来（ZH-BM）和泰米尔-英语（TA-EN），建立了新的综合性CS-ASR基准，以评估顶级ASR模型的性能。实验结果表明，提出的训练策略在单语言和CS测试中提升了ASR性能，BM-EN表现出最高提升，随后是TA-EN和ZH-BM。这一发现为CS-ASR的发展提供了成本效益高的方法，有利于研究和产业。 

---
# StorySage: Conversational Autobiography Writing Powered by a Multi-Agent Framework 

**Title (ZH)**: StorySage：基于多Agent框架的对话式自传写作 

**Authors**: Shayan Talaei, Meijin Li, Kanu Grover, James Kent Hippler, Diyi Yang, Amin Saberi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14159)  

**Abstract**: Every individual carries a unique and personal life story shaped by their memories and experiences. However, these memories are often scattered and difficult to organize into a coherent narrative, a challenge that defines the task of autobiography writing. Existing conversational writing assistants tend to rely on generic user interactions and pre-defined guidelines, making it difficult for these systems to capture personal memories and develop a complete biography over time. We introduce StorySage, a user-driven software system designed to meet the needs of a diverse group of users that supports a flexible conversation and a structured approach to autobiography writing. Powered by a multi-agent framework composed of an Interviewer, Session Scribe, Planner, Section Writer, and Session Coordinator, our system iteratively collects user memories, updates their autobiography, and plans for future conversations. In experimental simulations, StorySage demonstrates its ability to navigate multiple sessions and capture user memories across many conversations. User studies (N=28) highlight how StorySage maintains improved conversational flow, narrative completeness, and higher user satisfaction when compared to a baseline. In summary, StorySage contributes both a novel architecture for autobiography writing and insights into how multi-agent systems can enhance human-AI creative partnerships. 

**Abstract (ZH)**: 每一个个体都拥有一份独特且个人化的生活故事，由他们的记忆和经历共同塑造。然而，这些记忆往往零散且难以组织成一个连贯的叙事，这正是自传写作任务的一大挑战。现有的对话式写作助手往往依赖于通用的用户交互和预定义的指南，使得这些系统难以捕捉个人记忆并随着时间的发展构建一个完整的人生历程。我们介绍了StorySage，一个用户驱动的软件系统，旨在满足不同用户群体的需求，支持灵活的对话和结构化的自传写作方法。借助由访谈员、会话记录员、规划师、段落作家和会话协调组成的多智能体框架，我们的系统能够迭代收集用户的记忆，更新其自传，并规划未来的对话。在实验模拟中，StorySage展示了其在多个会话中导航并捕捉用户记忆的能力。用户研究（N=28）表明，相比于基线系统，StorySage能够维持更好的对话流畅性、叙事完整性，并提高用户满意度。总体而言，StorySage既贡献了一种新的自传写作架构，也为多智能体系统如何增强人机创意合作提供了见解。 

---
# NeuroCoreX: An Open-Source FPGA-Based Spiking Neural Network Emulator with On-Chip Learning 

**Title (ZH)**: NeuroCoreX：一个基于FPGA的内置学习功能的脉冲神经网络模拟器 

**Authors**: Ashish Gautam, Prasanna Date, Shruti Kulkarni, Robert Patton, Thomas Potok  

**Link**: [PDF](https://arxiv.org/pdf/2506.14138)  

**Abstract**: Spiking Neural Networks (SNNs) are computational models inspired by the structure and dynamics of biological neuronal networks. Their event-driven nature enables them to achieve high energy efficiency, particularly when deployed on neuromorphic hardware platforms. Unlike conventional Artificial Neural Networks (ANNs), which primarily rely on layered architectures, SNNs naturally support a wide range of connectivity patterns, from traditional layered structures to small-world graphs characterized by locally dense and globally sparse connections. In this work, we introduce NeuroCoreX, an FPGA-based emulator designed for the flexible co-design and testing of SNNs. NeuroCoreX supports all-to-all connectivity, providing the capability to implement diverse network topologies without architectural restrictions. It features a biologically motivated local learning mechanism based on Spike-Timing-Dependent Plasticity (STDP). The neuron model implemented within NeuroCoreX is the Leaky Integrate-and-Fire (LIF) model, with current-based synapses facilitating spike integration and transmission . A Universal Asynchronous Receiver-Transmitter (UART) interface is provided for programming and configuring the network parameters, including neuron, synapse, and learning rule settings. Users interact with the emulator through a simple Python-based interface, streamlining SNN deployment from model design to hardware execution. NeuroCoreX is released as an open-source framework, aiming to accelerate research and development in energy-efficient, biologically inspired computing. 

**Abstract (ZH)**: 基于FPGA的NeuroCoreX：面向SNNs的灵活设计与测试平台 

---
# Less is More: Undertraining Experts Improves Model Upcycling 

**Title (ZH)**: 少即是多：过度训练专家可以提高模型再利用效率 

**Authors**: Stefan Horoi, Guy Wolf, Eugene Belilovsky, Gintare Karolina Dziugaite  

**Link**: [PDF](https://arxiv.org/pdf/2506.14126)  

**Abstract**: Modern deep learning is increasingly characterized by the use of open-weight foundation models that can be fine-tuned on specialized datasets. This has led to a proliferation of expert models and adapters, often shared via platforms like HuggingFace and AdapterHub. To leverage these resources, numerous model upcycling methods have emerged, enabling the reuse of fine-tuned models in multi-task systems. A natural pipeline has thus formed to harness the benefits of transfer learning and amortize sunk training costs: models are pre-trained on general data, fine-tuned on specific tasks, and then upcycled into more general-purpose systems. A prevailing assumption is that improvements at one stage of this pipeline propagate downstream, leading to gains at subsequent steps. In this work, we challenge that assumption by examining how expert fine-tuning affects model upcycling. We show that long fine-tuning of experts that optimizes for their individual performance leads to degraded merging performance, both for fully fine-tuned and LoRA-adapted models, and to worse downstream results when LoRA adapters are upcycled into MoE layers. We trace this degradation to the memorization of a small set of difficult examples that dominate late fine-tuning steps and are subsequently forgotten during merging. Finally, we demonstrate that a task-dependent aggressive early stopping strategy can significantly improve upcycling performance. 

**Abstract (ZH)**: 现代深度学习日益依赖于可针对专门数据集进行微调的开放权重基础模型。这导致了专家模型和适配器的大量涌现，常通过HuggingFace和AdapterHub等平台共享。为了利用这些资源，已经出现了多种模型再利用方法，使微调后的模型能够在多任务系统中重用。因此形成了一条自然的管线，以利用迁移学习的优势并摊薄前期训练成本：模型在通用数据上预训练，在特定任务上进行微调，然后被再利用到更通用的系统中。普遍的假设是，这一管线各阶段的改进会逐渐传递到后续步骤，带来更好的结果。然而，在本工作中，我们通过研究专家微调如何影响模型再利用，挑战了这一假设。我们表明，针对专家个体性能优化进行长时间微调会导致合并性能下降，无论是完全微调的模型还是通过LoRA适配的模型，当LoRA适配器被插入到MoE层时，后续结果也会变差。我们将这种下降归因于难以学习的例子在后期微调中占据主导地位，随后在合并过程中被遗忘。最后，我们证明了依赖任务的激进式早期停止策略能够显著提高模型再利用性能。 

---
# CLGNN: A Contrastive Learning-based GNN Model for Betweenness Centrality Prediction on Temporal Graphs 

**Title (ZH)**: CLGNN：基于对比学习的用于时间图中介中心性预测的图神经网络模型 

**Authors**: Tianming Zhang, Renbo Zhang, Zhengyi Yang, Yunjun Gao, Bin Cao, Jing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14122)  

**Abstract**: Temporal Betweenness Centrality (TBC) measures how often a node appears on optimal temporal paths, reflecting its importance in temporal networks. However, exact computation is highly expensive, and real-world TBC distributions are extremely imbalanced. The severe imbalance leads learning-based models to overfit to zero-centrality nodes, resulting in inaccurate TBC predictions and failure to identify truly central nodes. Existing graph neural network (GNN) methods either fail to handle such imbalance or ignore temporal dependencies altogether. To address these issues, we propose a scalable and inductive contrastive learning-based GNN (CLGNN) for accurate and efficient TBC prediction. CLGNN builds an instance graph to preserve path validity and temporal order, then encodes structural and temporal features using dual aggregation, i.e., mean and edge-to-node multi-head attention mechanisms, enhanced by temporal path count and time encodings. A stability-based clustering-guided contrastive module (KContrastNet) is introduced to separate high-, median-, and low-centrality nodes in representation space, mitigating class imbalance, while a regression module (ValueNet) estimates TBC values. CLGNN also supports multiple optimal path definitions to accommodate diverse temporal semantics. Extensive experiments demonstrate the effectiveness and efficiency of CLGNN across diverse benchmarks. CLGNN achieves up to a 663.7~$\times$ speedup compared to state-of-the-art exact TBC computation methods. It outperforms leading static GNN baselines with up to 31.4~$\times$ lower MAE and 16.7~$\times$ higher Spearman correlation, and surpasses state-of-the-art temporal GNNs with up to 5.7~$\times$ lower MAE and 3.9~$\times$ higher Spearman correlation. 

**Abstract (ZH)**: 基于对比学习的可扩展图神经网络（CLGNN）用于精确高效的Temporal Betweenness Centrality预测 

---
# SKOLR: Structured Koopman Operator Linear RNN for Time-Series Forecasting 

**Title (ZH)**: SKOLR：结构化库默曼算子线性RNN时间序列预测 

**Authors**: Yitian Zhang, Liheng Ma, Antonios Valkanas, Boris N. Oreshkin, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2506.14113)  

**Abstract**: Koopman operator theory provides a framework for nonlinear dynamical system analysis and time-series forecasting by mapping dynamics to a space of real-valued measurement functions, enabling a linear operator representation. Despite the advantage of linearity, the operator is generally infinite-dimensional. Therefore, the objective is to learn measurement functions that yield a tractable finite-dimensional Koopman operator approximation. In this work, we establish a connection between Koopman operator approximation and linear Recurrent Neural Networks (RNNs), which have recently demonstrated remarkable success in sequence modeling. We show that by considering an extended state consisting of lagged observations, we can establish an equivalence between a structured Koopman operator and linear RNN updates. Building on this connection, we present SKOLR, which integrates a learnable spectral decomposition of the input signal with a multilayer perceptron (MLP) as the measurement functions and implements a structured Koopman operator via a highly parallel linear RNN stack. Numerical experiments on various forecasting benchmarks and dynamical systems show that this streamlined, Koopman-theory-based design delivers exceptional performance. 

**Abstract (ZH)**: Koopman算子理论提供了将非线性动力系统映射到实值测量函数空间的框架，从而实现线性算子表示，用于动力学分析和时间序列预测。尽管线性化有优势，但算子通常是无限维的。因此，目标是学习一类可以产生可处理的有限维Koopman算子近似的测量函数。在本工作中，我们建立了Koopman算子逼近与线性递归神经网络（RNNs）之间的联系，后者在序列建模中取得了显著的成功。我们展示了通过考虑延后观测组成的扩展状态，可以建立结构化Koopman算子与线性RNN更新之间的等价性。基于这一联系，我们提出了SKOLR，它将可学习的输入信号频谱分解与多层感知机（MLP）结合作为测量函数，并通过高度并行的线性RNN堆栈实现结构化Koopman算子。数值实验在各种预测基准和动力学系统上表明，这种基于Koopman理论的简化设计提供了出色的性能。 

---
# Essential-Web v1.0: 24T tokens of organized web data 

**Title (ZH)**: Essential-Web v1.0: 组织化的网络数据24Ttokens 

**Authors**: Essential AI, Andrew Hojel, Michael Pust, Tim Romanski, Yash Vanjani, Ritvik Kapila, Mohit Parmar, Adarsh Chaluvaraju, Alok Tripathy, Anil Thomas, Ashish Tanwer, Darsh J Shah, Ishaan Shah, Karl Stratos, Khoi Nguyen, Kurt Smith, Michael Callahan, Peter Rushton, Philip Monk, Platon Mazarakis, Saad Jamal, Saurabh Srivastava, Somanshu Singla, Ashish Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2506.14111)  

**Abstract**: Data plays the most prominent role in how language models acquire skills and knowledge. The lack of massive, well-organized pre-training datasets results in costly and inaccessible data pipelines. We present Essential-Web v1.0, a 24-trillion-token dataset in which every document is annotated with a twelve-category taxonomy covering topic, format, content complexity, and quality. Taxonomy labels are produced by EAI-Distill-0.5b, a fine-tuned 0.5b-parameter model that achieves an annotator agreement within 3% of Qwen2.5-32B-Instruct. With nothing more than SQL-style filters, we obtain competitive web-curated datasets in math (-8.0% relative to SOTA), web code (+14.3%), STEM (+24.5%) and medical (+8.6%). Essential-Web v1.0 is available on HuggingFace: this https URL 

**Abstract (ZH)**: Essential-Web v1.0: 一个包含24万亿标记的多类别标注网页数据集 

---
# Toward a Graph Foundation Model: Pre-Training Transformers With Random Walks 

**Title (ZH)**: 基于图基础模型的研究：使用随机游走预训练Transformer 

**Authors**: Ziyuan Tang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14098)  

**Abstract**: A foundation model like GPT elicits many emergent abilities, owing to the pre-training with broad inclusion of data and the use of the powerful Transformer architecture. While foundation models in natural languages are prevalent, can we build similar models for graphs? This paper describes an approach toward a graph foundation model that is pre-trained with diverse graph datasets by adapting the Transformer backbone. A central challenge toward this end is how a sequence model encodes graphs of varying sizes and from different domains. We propose representing a node as multiple random walks, such that the Transformer can extract node representations from sequences, which in turn form edge and graph representations. We develop a novel context prediction loss for these random walks and theoretically analyze their expressive power in distinguishing neighborhoods and graphs. We also demonstrate the pre-training of our model and its adaptation to downstream tasks, showcasing its potential as a foundation for processing and reasoning with graph-structured data. 

**Abstract (ZH)**: 像GPT这样的基础模型由于广域数据的预训练和强大力量的Transformer架构的应用，激发了多种 emergent 能力。虽然自然语言领域中的基础模型十分普遍，我们能否构建类似的图模型？本文描述了一种通过适应Transformer骨干来使用多样化的图数据集进行预训练的方法。这一目标的核心挑战是如何将序列模型编码适用于不同大小和不同领域的图。我们提出将节点表示为多个随机游走，从而使Transformer可以从序列中提取节点表示，进而形成边和图的表示。我们开发了一种新的上下文预测损失函数，并理论分析了这些随机游走在区分邻域和图上的表达能力。我们还展示了该模型的预训练及其对下游任务的适应性，并展示了其作为处理和推理图结构数据的基础的潜力。 

---
# Scientifically-Interpretable Reasoning Network (ScIReN): Uncovering the Black-Box of Nature 

**Title (ZH)**: 科学可解释推理网络（ScIReN）：揭开自然界的黑箱 

**Authors**: Joshua Fan, Haodi Xu, Feng Tao, Md Nasim, Marc Grimson, Yiqi Luo, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2506.14054)  

**Abstract**: Neural networks are a powerful tool for learning patterns from data. However, they do not respect known scientific laws, nor can they reveal novel scientific insights due to their black-box nature. In contrast, scientific reasoning distills biological or physical principles from observations and controlled experiments, and quantitatively interprets them with process-based models made of mathematical equations. Yet, process-based models rely on numerous free parameters that must be set in an ad-hoc manner, and thus often fit observations poorly in cross-scale predictions. While prior work has embedded process-based models in conventional neural networks, discovering interpretable relationships between parameters in process-based models and input features is still a grand challenge for scientific discovery. We thus propose Scientifically-Interpretable Reasoning Network (ScIReN), a fully-transparent framework that combines interpretable neural and process-based reasoning. An interpretable encoder predicts scientifically-meaningful latent parameters, which are then passed through a differentiable process-based decoder to predict labeled output variables. ScIReN also uses a novel hard-sigmoid constraint layer to restrict latent parameters to meaningful ranges defined by scientific prior knowledge, further enhancing its interpretability. While the embedded process-based model enforces established scientific knowledge, the encoder reveals new scientific mechanisms and relationships hidden in conventional black-box models. We apply ScIReN on two tasks: simulating the flow of organic carbon through soils, and modeling ecosystem respiration from plants. In both tasks, ScIReN outperforms black-box networks in predictive accuracy while providing substantial scientific interpretability -- it can infer latent scientific mechanisms and their relationships with input features. 

**Abstract (ZH)**: 科学可解释推理网络：结合可解释神经网络与过程模型的完全透明框架 

---
# Asymptotically Smaller Encodings for Graph Problems and Scheduling 

**Title (ZH)**: 渐进更小的图问题和调度编码大小 

**Authors**: Bernardo Subercaseaux  

**Link**: [PDF](https://arxiv.org/pdf/2506.14042)  

**Abstract**: We show how several graph problems (e.g., vertex-cover, independent-set, $k$-coloring) can be encoded into CNF using only $O(|V|^2 / \lg |V|)$ many clauses, as opposed to the $\Omega(|V|^2)$ constraints used by standard encodings. This somewhat surprising result is a simple consequence of a result of Erdős, Chung, and Spencer (1983) about biclique coverings of graphs, and opens theoretical avenues to understand the success of "Bounded Variable Addition'' (Manthey, Heule, and Biere, 2012) as a preprocessing tool. Finally, we show a novel encoding for independent sets in some dense interval graphs using only $O(|V| \lg |V|)$ clauses (the direct encoding uses $\Omega(|V|^2)$), which we have successfully applied to a string-compression encoding posed by Bannai et al. (2022). As a direct byproduct, we obtain a reduction in the encoding size of a scheduling problem posed by Mayank and Modal (2020) from $O(NMT^2)$ to $O(NMT + M T^2 \lg T)$, where $N$ is the number of tasks, $T$ the total timespan, and $M$ the number of machines. 

**Abstract (ZH)**: 我们展示了如何使用只有 $O(|V|^2 / \lg |V|)$ 个子句将几个图问题（例如，顶点覆盖、独立集、$k$-着色）编码到CNF中，而不是标准编码使用的 $\Omega(|V|^2)$ 个约束。这一令人惊讶的结果是Erdős、Chung和Spencer（1983）关于图的biclique覆盖的一个结果的简单推论，并为理解“有界变量增加”（Manthey、Heule和Biere，2012）作为预处理工具的成功开辟了理论途径。最后，我们展示了一种新的编码方法，使用只有 $O(|V| \lg |V|)$ 个子句为某些稠密区间图编码独立集（直接编码使用 $\Omega(|V|^2)$ 个子句），并成功应用于Bannai等人（2022）提出的字符串压缩编码。作为直接的结果，我们得到了Mayank和Modal（2020）提出的调度问题编码大小从 $O(NMT^2)$ 减少到 $O(NMT + M T^2 \lg T)$，其中 $N$ 是任务的数量，$T$ 是总时间段，$M$ 是机器的数量。 

---
# Bures-Wasserstein Flow Matching for Graph Generation 

**Title (ZH)**: Bures-Wasserstein流匹配图生成 

**Authors**: Keyue Jiang, Jiahao Cui, Xiaowen Dong, Laura Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.14020)  

**Abstract**: Graph generation has emerged as a critical task in fields ranging from molecule design to drug discovery. Contemporary approaches, notably diffusion and flow-based models, have achieved solid graph generative performance through constructing a probability path that interpolates between a reference distribution and the data distribution. However, these methods typically model the evolution of individual nodes and edges independently and use linear interpolations to build the path assuming that the data lie in Euclidean space. We show that this is suboptimal given the intrinsic non-Euclidean structure and interconnected patterns of graphs, and it poses risks to the sampling convergence. To build a better probability path, we model the joint evolution of the nodes and edges by representing graphs as connected systems parameterized by Markov random fields (MRF). We then leverage the optimal transport displacement between MRF objects to design the probability path for graph generation. Based on this, we introduce BWFlow, a flow-matching framework for graph generation that respects the underlying geometry of graphs and provides smooth velocities in the probability path. The novel framework can be adapted to both continuous and discrete flow-matching algorithms. Experimental evaluations in plain graph generation and 2D/3D molecule generation validate the effectiveness of BWFlow in graph generation with competitive performance, stable training, and guaranteed sampling convergence. 

**Abstract (ZH)**: 图生成已在分子设计、药物发现等领域中 emerged 作为一项关键任务。现代方法，特别是扩散和流动模型，通过构建一个在参考分布和数据分布之间进行插值的概率路径，实现了坚实的图生成性能。然而，这些方法通常独立地建模节点和边的演变，并假设数据位于欧几里得空间中，使用线性插值来构建路径。我们证明，在图的内在非欧几里得结构和相互关联模式的情况下，这种建模方法是次优的，并且可能会对采样收敛性带来风险。为了构建更好的概率路径，我们将节点和边的联合演变建模为由马尔可夫随机场（MRF）参数化的连接系统。然后，我们利用MRF对象之间的最优传输位移来设计图生成的概率路径。在此基础上，我们引入了BWFlow，这是一种尊重图底层几何结构的流动匹配框架，并在概率路径中提供平滑的速度。该新型框架可以适应连续和离散的流动匹配算法。在单纯的图生成和2D/3D分子生成实验评估中，BWFlow证明了其在图生成中的有效性，具有竞争力的性能、稳定的训练和有保证的采样收敛性。 

---
# AMLgentex: Mobilizing Data-Driven Research to Combat Money Laundering 

**Title (ZH)**: AMLgentex: 利用数据驱动研究打击洗钱活动 

**Authors**: Johan Östman, Edvin Callisen, Anton Chen, Kristiina Ausmees, Emanuel Gårdh, Jovan Zamac, Jolanta Goldsteine, Hugo Wefer, Simon Whelan, Markus Reimegård  

**Link**: [PDF](https://arxiv.org/pdf/2506.13989)  

**Abstract**: Money laundering enables organized crime by allowing illicit funds to enter the legitimate economy. Although trillions of dollars are laundered each year, only a small fraction is ever uncovered. This stems from a range of factors, including deliberate evasion by launderers, the rarity of confirmed cases, and the limited visibility each financial institution has into the global transaction network. While several synthetic datasets are available, they fail to model the structural and behavioral complexity of real-world money laundering. In particular, they often overlook partial observability, sparse and uncertain labels, strategic behavior, temporal dynamics, class imbalance, and network-level dependencies. To address these limitations, we present AMLGentex, an open-source suite for generating realistic, configurable transaction data and benchmarking detection methods. It enables systematic evaluation of anti-money laundering (AML) systems in a controlled environment that captures key real-world challenges. We demonstrate how the framework can be used to rigorously evaluate methods under conditions that reflect the complexity of practical AML scenarios. 

**Abstract (ZH)**: 洗钱通过使非法资金进入合法经济从而助长有组织犯罪。尽管每年有数万亿美元被洗钱，但其中只有小部分能够被发现。这归因于多个因素，包括洗钱者故意规避、确认案例的稀少以及金融机构对全球交易网络的有限可见性。虽然有一些合成数据集可用，但它们无法模拟现实世界洗钱的结构和行为复杂性。特别是，它们通常忽略了部分可观测性、稀疏和不确定的标签、战略行为、时间动态、类别不平衡以及网络层面的依赖性。为解决这些局限性，我们提出AMLGentex，一个开源工具套件，用于生成现实且可配置的交易数据并比较检测方法。它使得在能够捕捉到关键现实挑战的受控环境中系统地评估反洗钱（AML）系统成为可能。我们展示了该框架如何在反映实际AML场景复杂性的条件下严格评估方法。 

---
# Mirror Descent Using the Tempesta Generalized Multi-parametric Logarithms 

**Title (ZH)**: 镜像下降法使用Tempesta广义多参数对数函数 

**Authors**: Andrzej Cichocki  

**Link**: [PDF](https://arxiv.org/pdf/2506.13984)  

**Abstract**: In this paper, we develop a wide class Mirror Descent (MD) algorithms, which play a key role in machine learning. For this purpose we formulated the constrained optimization problem, in which we exploits the Bregman divergence with the Tempesta multi-parametric deformation logarithm as a link function. This link function called also mirror function defines the mapping between the primal and dual spaces and is associated with a very-wide (in fact, theoretically infinite) class of generalized trace-form entropies. In order to derive novel MD updates, we estimate generalized exponential function, which closely approximates the inverse of the multi-parametric Tempesta generalized logarithm. The shape and properties of the Tempesta logarithm and its inverse-deformed exponential functions can be tuned by several hyperparameters. By learning these hyperparameters, we can adapt to distribution or geometry of training data, and we can adjust them to achieve desired properties of MD algorithms. The concept of applying multi-parametric logarithms allow us to generate a new wide and flexible family of MD and mirror-less MD updates. 

**Abstract (ZH)**: 在这篇论文中，我们开发了一类广泛的_mirror descent_ (MD)算法，这些算法在机器学习中扮演着关键角色。为此，我们形式化了一个受约束的优化问题，在该问题中利用Tempesta多元参数变形对数作为链接函数，该链接函数也被称作mirror函数，定义了原始空间和对偶空间之间的映射，并与极其广泛（实际上，理论上是无限广泛）的一类广义迹形式熵相关。为了推导新的MD更新，我们估计了广义指数函数，该函数几乎可以 approximates 多元参数Tempesta广义对数的逆。Tempesta 对数及其逆变形指数函数的形状和属性可以通过多个超参数进行调整。通过学习这些超参数，我们可以适应训练数据的分布或几何结构，并据此调整以实现MD算法所需的各种属性。应用多元参数对数的概念使我们能够生成一个新的广泛而灵活的MD及其无mirror的MD更新族。 

---
# HAELT: A Hybrid Attentive Ensemble Learning Transformer Framework for High-Frequency Stock Price Forecasting 

**Title (ZH)**: HAELT：一种用于高频率股票价格预测的混合注意集成学习变换器框架 

**Authors**: Thanh Dan Bui  

**Link**: [PDF](https://arxiv.org/pdf/2506.13981)  

**Abstract**: High-frequency stock price prediction is challenging due to non-stationarity, noise, and volatility. To tackle these issues, we propose the Hybrid Attentive Ensemble Learning Transformer (HAELT), a deep learning framework combining a ResNet-based noise-mitigation module, temporal self-attention for dynamic focus on relevant history, and a hybrid LSTM-Transformer core that captures both local and long-range dependencies. These components are adaptively ensembled based on recent performance. Evaluated on hourly Apple Inc. (AAPL) data from Jan 2024 to May 2025, HAELT achieves the highest F1-Score on the test set, effectively identifying both upward and downward price movements. This demonstrates HAELT's potential for robust, practical financial forecasting and algorithmic trading. 

**Abstract (ZH)**: 高频股票价格预测因非平稳性、噪声和波动性而具有挑战性。为应对这些挑战，我们提出了混合注意力集成学习变压器（HAELT）框架，该框架结合了基于ResNet的噪声减轻模块、时间自注意力以动态聚焦相关历史，以及同时捕捉局部和长程依赖性的混合LSTM-Transformer核心。这些组件基于近期表现进行适应性集成。实验结果表明，HAELT在从2024年1月到2025年5月的每小时Apple Inc. (AAPL)数据上测试集中的F1分数最高，有效地识别了价格的上下行变动。这展示了HAELT在稳健且实际的金融预测和算法交易中的潜力。 

---
# Making deep neural networks work for medical audio: representation, compression and domain adaptation 

**Title (ZH)**: 在医疗音频中使深度神经网络发挥作用：表示、压缩和领域适应 

**Authors**: Charles C Onu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13970)  

**Abstract**: This thesis addresses the technical challenges of applying machine learning to understand and interpret medical audio signals. The sounds of our lungs, heart, and voice convey vital information about our health. Yet, in contemporary medicine, these sounds are primarily analyzed through auditory interpretation by experts using devices like stethoscopes. Automated analysis offers the potential to standardize the processing of medical sounds, enable screening in low-resource settings where physicians are scarce, and detect subtle patterns that may elude human perception, thereby facilitating early diagnosis and treatment.
Focusing on the analysis of infant cry sounds to predict medical conditions, this thesis contributes on four key fronts. First, in low-data settings, we demonstrate that large databases of adult speech can be harnessed through neural transfer learning to develop more accurate and robust models for infant cry analysis. Second, in cost-effective modeling, we introduce an end-to-end model compression approach for recurrent networks using tensor decomposition. Our method requires no post-hoc processing, achieves compression rates of several hundred-fold, and delivers accurate, portable models suitable for resource-constrained devices. Third, we propose novel domain adaptation techniques tailored for audio models and adapt existing methods from computer vision. These approaches address dataset bias and enhance generalization across domains while maintaining strong performance on the original data. Finally, to advance research in this domain, we release a unique, open-source dataset of infant cry sounds, developed in collaboration with clinicians worldwide.
This work lays the foundation for recognizing the infant cry as a vital sign and highlights the transformative potential of AI-driven audio monitoring in shaping the future of accessible and affordable healthcare. 

**Abstract (ZH)**: 本论文探讨了将机器学习应用于理解和解释医疗音频信号所面临的technical挑战。我们的肺音、心音和语音传达着关于我们健康的重要信息。然而，在当代医学中，这些声音主要通过专家使用听诊器等设备进行听觉分析。自动化分析有可能标准化医疗声音的处理过程，在医生稀缺的低资源环境中实现筛查，并检测人眼可能察觉不到的细微模式，从而促进早期诊断和治疗。

本论文集中于婴儿啼哭声音的分析以预测医疗状况，主要从四个方面进行了贡献。首先，在数据稀缺的环境下，我们通过神经迁移学习利用大量成人语音数据集开发更准确和鲁棒的婴儿啼哭分析模型。其次，在经济高效的建模方面，我们引入了一种用于递归网络的端到端模型压缩方法，使用张量分解。我们的方法无需后处理，实现了几百倍的压缩率，并提供了准确且便于携带的模型，适合资源受限的设备。第三，我们提出了适用于音频模型的新方法，并对计算机视觉中的现有方法进行改编。这些方法解决了数据集偏差问题，增强了跨领域的一般化能力，同时在原始数据上仍保持了强大的性能。最后，为了推动该领域的研究，我们发布了一个独特的开放源代码婴儿啼哭声音数据集，该数据集是与全球临床医生合作开发的。

本研究为识别婴儿啼哭作为生命体征奠定了基础，并突显了基于AI的音频监控在推动可及性和负担得起的医疗保健未来方面的变革潜力。 

---
# Safe Domains of Attraction for Discrete-Time Nonlinear Systems: Characterization and Verifiable Neural Network Estimation 

**Title (ZH)**: 离散时间非线性系统的安全吸引域：表征及可验证神经网络估计 

**Authors**: Mohamed Serry, Haoyu Li, Ruikun Zhou, Huan Zhang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13961)  

**Abstract**: Analysis of nonlinear autonomous systems typically involves estimating domains of attraction, which have been a topic of extensive research interest for decades. Despite that, accurately estimating domains of attraction for nonlinear systems remains a challenging task, where existing methods are conservative or limited to low-dimensional systems. The estimation becomes even more challenging when accounting for state constraints. In this work, we propose a framework to accurately estimate safe (state-constrained) domains of attraction for discrete-time autonomous nonlinear systems. In establishing this framework, we first derive a new Zubov equation, whose solution corresponds to the exact safe domain of attraction. The solution to the aforementioned Zubov equation is shown to be unique and continuous over the whole state space. We then present a physics-informed approach to approximating the solution of the Zubov equation using neural networks. To obtain certifiable estimates of the domain of attraction from the neural network approximate solutions, we propose a verification framework that can be implemented using standard verification tools (e.g., $\alpha,\!\beta$-CROWN and dReal). To illustrate its effectiveness, we demonstrate our approach through numerical examples concerning nonlinear systems with state constraints. 

**Abstract (ZH)**: 离散自治非线性系统的安全（状态约束）吸引域精确估计方法 

---
# Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models 

**Title (ZH)**: 自适应引导加速推理模型的强化学习 

**Authors**: Vaskar Nath, Elaine Lau, Anisha Gunjal, Manasi Sharma, Nikhil Baharte, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.13923)  

**Abstract**: We study the process through which reasoning models trained with reinforcement learning on verifiable rewards (RLVR) can learn to solve new problems. We find that RLVR drives performance through two main means: (1) by compressing pass@$k$ into pass@1 and (2) via "capability gain" in which models learn to solve new problems that they previously could not solve even at high $k$. We find that while capability gain exists across model scales, learning to solve new problems is primarily driven through self-distillation. We demonstrate these findings across model scales ranging from 0.5B to 72B on >500,000 reasoning problems with prompts and verifiable final answers across math, science, and code domains. We further show that we can significantly improve pass@$k$ rates by leveraging natural language guidance for the model to consider within context while still requiring the model to derive a solution chain from scratch. Based of these insights, we derive $\text{Guide}$ - a new class of online training algorithms. $\text{Guide}$ adaptively incorporates hints into the model's context on problems for which all rollouts were initially incorrect and adjusts the importance sampling ratio for the "off-policy" trajectories in order to optimize the policy for contexts in which the hints are no longer present. We describe variants of $\text{Guide}$ for GRPO and PPO and empirically show that Guide-GRPO on 7B and 32B parameter models improves generalization over its vanilla counterpart with up to 4$\%$ macro-average improvement across math benchmarks. We include careful ablations to analyze $\text{Guide}$'s components and theoretically analyze Guide's learning efficiency. 

**Abstract (ZH)**: 我们研究了通过验证奖励强化学习（RLVR）训练的推理模型如何学习解决新问题的过程。我们发现RLVR通过两种主要方式提升性能：（1）将pass@$k$压缩为pass@1，（2）通过“能力提升”，模型学习解决以前即使在高$k$值下也无法解决的新问题。我们发现虽然能力提升跨越了不同的模型规模，但学习解决新问题主要通过自我精炼驱动。我们在这篇论文中展示了在数学、科学和代码领域的超过50万道推理问题上，从0.5B到72B规模的模型中这些发现。我们进一步表明，可以通过利用自然语言指导来显著提高pass@$k$率，同时仍然要求模型从头推导完整的问题解决链。基于上述洞见，我们推导出了$\text{Guide}$——一种新的在线训练算法。$\text{Guide}$自适应地将提示整合到问题的模型上下文中，对于所有展开均为错误的问题，调整“离策”轨迹的重要性采样比，以便在提示不再存在的上下文中优化策略。我们为GRPO和PPO提供了$\text{Guide}$的变体，并实验证明，对于7B和32B参数模型的Guide-GRPO相较于其原版版本，在数学基准测试中可实现高达4%的宏平均性能提升。我们还包括了精细的消融实验来分析$\text{Guide}$的各个组件，并对其学习效率进行了理论分析。 

---
# Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization 

**Title (ZH)**: 层次节点个体化下的图神经网络逻辑表达能力 

**Authors**: Arie Soeteman, Balder ten Cate  

**Link**: [PDF](https://arxiv.org/pdf/2506.13911)  

**Abstract**: We propose and study Hierarchical Ego Graph Neural Networks (HEGNNs), an expressive extension of graph neural networks (GNNs) with hierarchical node individualization, inspired by the Individualization-Refinement paradigm for graph isomorphism testing. HEGNNs generalize subgraph-GNNs and form a hierarchy of increasingly expressive models that, in the limit, can distinguish graphs up to isomorphism. We provide a logical characterization of HEGNN node classifiers, with and without subgraph restrictions, using graded hybrid logic. This characterization enables us to relate the separating power of HEGNNs to that of higher-order GNNs, GNNs enriched with local homomorphism count features, and color refinement algorithms based on Individualization-Refinement. Our experimental results confirm the practical feasibility of HEGNNs and show benefits in comparison with traditional GNN architectures, both with and without local homomorphism count features. 

**Abstract (ZH)**: 我们提出并研究了层次自中心图神经网络（HEGNNs），这是一种受图同构检验中的个体化- refinment范式启发，具有层次节点个体化扩展的图神经网络（GNNs）的表达性扩展。HEGNNs 推广了子图-GNNs，并形成了一种逐步表达性增强的模型层次，在极限情况下，能够区分同构的图。通过使用层次混合逻辑提供HEGNN节点分类器的逻辑特征化，使得我们可以将HEGNN的区分能力与其高阶GNNs、局部同构计数特征增强的GNNs以及基于个体化-refinement的颜色细化算法联系起来。实验结果证实了HEGNNs的实际可行性，并展示了与传统GNN架构及其是否包含局部同构计数特征时相比的优势。 

---
# Few-Shot Learning for Industrial Time Series: A Comparative Analysis Using the Example of Screw-Fastening Process Monitoring 

**Title (ZH)**: 工业时间序列的少量样本学习：以螺栓紧固过程监控为例的比较分析 

**Authors**: Xinyuan Tu, Haocheng Zhang, Tao Chengxu, Zuyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13909)  

**Abstract**: Few-shot learning (FSL) has shown promise in vision but remains largely unexplored for \emph{industrial} time-series data, where annotating every new defect is prohibitively expensive. We present a systematic FSL study on screw-fastening process monitoring, using a 2\,300-sample multivariate torque dataset that covers 16 uni- and multi-factorial defect types. Beyond benchmarking, we introduce a \textbf{label-aware episodic sampler} that collapses multi-label sequences into multiple single-label tasks, keeping the output dimensionality fixed while preserving combinatorial label information.
Two FSL paradigms are investigated: the metric-based \emph{Prototypical Network} and the gradient-based \emph{Model-Agnostic Meta-Learning} (MAML), each paired with three backbones: 1D CNN, InceptionTime and the 341 M-parameter transformer \emph{Moment}. On 10-shot, 3-way evaluation, the InceptionTime + Prototypical Network combination achieves a \textbf{0.944 weighted F1} in the multi-class regime and \textbf{0.935} in the multi-label regime, outperforming finetuned Moment by up to 5.3\% while requiring two orders of magnitude fewer parameters and training time. Across all backbones, metric learning consistently surpasses MAML, and our label-aware sampling yields an additional 1.7\% F1 over traditional class-based sampling.
These findings challenge the assumption that large foundation models are always superior: when data are scarce, lightweight CNN architectures augmented with simple metric learning not only converge faster but also generalize better. We release code, data splits and pre-trained weights to foster reproducible research and to catalyze the adoption of FSL in high-value manufacturing inspection. 

**Abstract (ZH)**: Few-Shot Learning for Industrial Time-Series Data in Screw-Fastening Process Monitoring: A Systematic Study with Label-Aware Episodic Sampling 

---
# A Systematic Review of User-Centred Evaluation of Explainable AI in Healthcare 

**Title (ZH)**: 面向用户的解释型人工智能在医疗保健领域中的评价系统的综述 

**Authors**: Ivania Donoso-Guzmán, Kristýna Sirka Kacafírková, Maxwell Szymanski, An Jacobs, Denis Parra, Katrien Verbert  

**Link**: [PDF](https://arxiv.org/pdf/2506.13904)  

**Abstract**: Despite promising developments in Explainable Artificial Intelligence, the practical value of XAI methods remains under-explored and insufficiently validated in real-world settings. Robust and context-aware evaluation is essential, not only to produce understandable explanations but also to ensure their trustworthiness and usability for intended users, but tends to be overlooked because of no clear guidelines on how to design an evaluation with users.
This study addresses this gap with two main goals: (1) to develop a framework of well-defined, atomic properties that characterise the user experience of XAI in healthcare; and (2) to provide clear, context-sensitive guidelines for defining evaluation strategies based on system characteristics.
We conducted a systematic review of 82 user studies, sourced from five databases, all situated within healthcare settings and focused on evaluating AI-generated explanations. The analysis was guided by a predefined coding scheme informed by an existing evaluation framework, complemented by inductive codes developed iteratively.
The review yields three key contributions: (1) a synthesis of current evaluation practices, highlighting a growing focus on human-centred approaches in healthcare XAI; (2) insights into the interrelations among explanation properties; and (3) an updated framework and a set of actionable guidelines to support interdisciplinary teams in designing and implementing effective evaluation strategies for XAI systems tailored to specific application contexts. 

**Abstract (ZH)**: 尽管可解释人工智能取得了令人鼓舞的发展，但在实际应用中，XAI方法的实际价值仍然被低估并且缺乏充分验证。 robust且情境感知的评估至关重要，不仅为了生成可理解的解释，也为了确保这些解释对于预期用户具有可信性和实用性，但由于缺乏明确的设计评估指南，这一问题往往会被人忽视。

本研究旨在填补这一空白，主要目标有两个：（1）开发一套定义明确、基本属性的框架，以描述健康 care 中的 XAI 用户体验；（2）提供基于系统特征的情境敏感指南，以定义评估策略。

我们对82篇用户研究进行了系统回顾，这些研究来自五个数据库，均位于健康 care 设置中，专注于评估 AI 生成的解释。分析基于预定义的编码方案进行，该方案由现有的评估框架启发，并由迭代开发的归纳代码补充。

该回顾研究提供了三个方面的重要贡献：（1）当前评估实践的综合分析，突显了健康 care XAI 中日益关注以人为本的方法；（2）解释属性之间关系的洞察；（3）更新的框架和一套可操作的指南，以支持跨学科团队设计和实施针对特定应用场景的 XAI 系统评估策略。 

---
# Enhancing interpretability of rule-based classifiers through feature graphs 

**Title (ZH)**: 基于特征图增强规则分类器的可解释性 

**Authors**: Christel Sirocchi, Damiano Verda  

**Link**: [PDF](https://arxiv.org/pdf/2506.13903)  

**Abstract**: In domains where transparency and trustworthiness are crucial, such as healthcare, rule-based systems are widely used and often preferred over black-box models for decision support systems due to their inherent interpretability. However, as rule-based models grow complex, discerning crucial features, understanding their interactions, and comparing feature contributions across different rule sets becomes challenging. To address this, we propose a comprehensive framework for estimating feature contributions in rule-based systems, introducing a graph-based feature visualisation strategy, a novel feature importance metric agnostic to rule-based predictors, and a distance metric for comparing rule sets based on feature contributions. By experimenting on two clinical datasets and four rule-based methods (decision trees, logic learning machines, association rules, and neural networks with rule extraction), we showcase our method's capability to uncover novel insights on the combined predictive value of clinical features, both at the dataset and class-specific levels. These insights can aid in identifying new risk factors, signature genes, and potential biomarkers, and determining the subset of patient information that should be prioritised to enhance diagnostic accuracy. Comparative analysis of the proposed feature importance score with state-of-the-art methods on 15 public benchmarks demonstrates competitive performance and superior robustness. The method implementation is available on GitHub: this https URL. 

**Abstract (ZH)**: 在医疗健康等透明性和可信度至关重要的领域，基于规则的系统广泛使用，并常用于决策支持系统，因其固有的可解释性。然而，随着基于规则的模型变得复杂，区分关键特征、理解它们的交互以及在不同规则集之间比较特征贡献变得挑战重重。为解决这一问题，我们提出了一种全面的框架来估计基于规则系统的特征贡献，引入了一种基于图的特征可视化策略、一种与基于规则的预测器无关的新颖特征重要性度量方法，以及一种基于特征贡献比较规则集的距离度量方法。通过在两个临床数据集和四种基于规则的方法（决策树、逻辑学习机、关联规则、具有规则提取的神经网络）上进行实验，展示本方法可以在数据集级别和类特定级别上揭示临床特征联合预测价值的新洞见。这些洞见有助于识别新的风险因素、特征基因和潜在生物标志物，并确定应优先考虑的患者信息子集以提高诊断准确性。与15个公开基准上的先进方法的比较分析表明，本方法具有竞争力的性能和更高的稳健性。该方法的实现可在GitHub上获得：this https URL。 

---
# Beyond Shapley Values: Cooperative Games for the Interpretation of Machine Learning Models 

**Title (ZH)**: 超越Shapley值：合作博弈在机器学习模型解释中的应用 

**Authors**: Marouane Il Idrissi, Agathe Fernandes Machado, Arthur Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2506.13900)  

**Abstract**: Cooperative game theory has become a cornerstone of post-hoc interpretability in machine learning, largely through the use of Shapley values. Yet, despite their widespread adoption, Shapley-based methods often rest on axiomatic justifications whose relevance to feature attribution remains debatable. In this paper, we revisit cooperative game theory from an interpretability perspective and argue for a broader and more principled use of its tools. We highlight two general families of efficient allocations, the Weber and Harsanyi sets, that extend beyond Shapley values and offer richer interpretative flexibility. We present an accessible overview of these allocation schemes, clarify the distinction between value functions and aggregation rules, and introduce a three-step blueprint for constructing reliable and theoretically-grounded feature attributions. Our goal is to move beyond fixed axioms and provide the XAI community with a coherent framework to design attribution methods that are both meaningful and robust to shifting methodological trends. 

**Abstract (ZH)**: 合作博弈论已成为机器学习事后解释的基石，很大程度上得益于舍甫琴值的应用。尽管如此，Shapley基方法往往依赖于对解释特征相关性存疑的公理化论据。在本文中，我们从解释性的角度重新审视合作博弈论，并倡导更广泛和更为原则地使用其工具。我们强调了两种高效的分配方案，韦伯集和哈萨尼集，这些方案超越了Shapley值，提供了更丰富的解释灵活性。我们提供了这些分配方案的可访问概述，阐明了价值函数与聚合规则之间的区别，并引入了三个步骤的蓝图，用于构建可靠且理论依据充分的特征 Attribution方法。我们的目标是超越固定的公理，为可解释人工智能社区提供一个连贯的框架，以设计既具有意义又能够抵抗方法学趋势变化的方法。 

---
# Fake it till You Make it: Reward Modeling as Discriminative Prediction 

**Title (ZH)**: 假作真时真亦假：奖励模型作为辨别性预测 

**Authors**: Runtao Liu, Jiahao Zhan, Yingqing He, Chen Wei, Alan Yuille, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13846)  

**Abstract**: An effective reward model plays a pivotal role in reinforcement learning for post-training enhancement of visual generative models. However, current approaches of reward modeling suffer from implementation complexity due to their reliance on extensive human-annotated preference data or meticulously engineered quality dimensions that are often incomplete and engineering-intensive. Inspired by adversarial training in generative adversarial networks (GANs), this paper proposes GAN-RM, an efficient reward modeling framework that eliminates manual preference annotation and explicit quality dimension engineering. Our method trains the reward model through discrimination between a small set of representative, unpaired target samples(denoted as Preference Proxy Data) and model-generated ordinary outputs, requiring only a few hundred target samples. Comprehensive experiments demonstrate our GAN-RM's effectiveness across multiple key applications including test-time scaling implemented as Best-of-N sample filtering, post-training approaches like Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). 

**Abstract (ZH)**: 一种有效的奖励模型在视觉生成模型的后训练增强中发挥着关键作用。然而，当前的奖励建模方法由于依赖广泛的用户标注偏好数据或精心构建的质量维度而面临着实施复杂性的问题，这些质量维度往往不完整且工程强度高。受生成对抗网络（GANs）中对抗训练的启发，本文提出了一种高效的奖励建模框架GAN-RM，该框架消除了手动偏好标注和显式质量维度的工程。我们的方法通过辨别一小组代表性且未配对的目标样本（称为偏好代理数据）与模型生成的普通输出来进行奖励模型的训练，仅需几百个目标样本。全面的实验结果表明，GAN-RM 在多个关键应用中均表现出有效性，包括测试时缩放（作为 Best-of-N 样本筛选实现）、后训练方法如监督微调（SFT）和直接偏好优化（DPO）。 

---
# Students' Reliance on AI in Higher Education: Identifying Contributing Factors 

**Title (ZH)**: 高等教育中学生对AI的依赖：探究影响因素 

**Authors**: Griffin Pitts, Neha Rani, Weedguet Mildort, Eva-Marie Cook  

**Link**: [PDF](https://arxiv.org/pdf/2506.13845)  

**Abstract**: The increasing availability and use of artificial intelligence (AI) tools in educational settings has raised concerns about students' overreliance on these technologies. Overreliance occurs when individuals accept incorrect AI-generated recommendations, often without critical evaluation, leading to flawed problem solutions and undermining learning outcomes. This study investigates potential factors contributing to patterns of AI reliance among undergraduate students, examining not only overreliance but also appropriate reliance (correctly accepting helpful and rejecting harmful recommendations) and underreliance (incorrectly rejecting helpful recommendations). Our approach combined pre- and post-surveys with a controlled experimental task where participants solved programming problems with an AI assistant that provided both accurate and deliberately incorrect suggestions, allowing direct observation of students' reliance patterns when faced with varying AI reliability. We find that appropriate reliance is significantly related to students' programming self-efficacy, programming literacy, and need for cognition, while showing negative correlations with post-task trust and satisfaction. Overreliance showed significant correlations with post-task trust and satisfaction with the AI assistant. Underreliance was negatively correlated with programming literacy, programming self-efficacy, and need for cognition. Overall, the findings provide insights for developing targeted interventions that promote appropriate reliance on AI tools, with implications for the integration of AI in curriculum and educational technologies. 

**Abstract (ZH)**: 人工智能工具在教育领域的日益可用性和应用增加了学生过度依赖这些技术的担忧。过度依赖发生在个人接受错误的人工智能生成建议的情况下，通常未经批判性评价，导致问题解决方案缺陷并损害学习成果。本研究调查了本科生中人工智能依赖模式的潜在因素，不仅考察了过度依赖，还考察了适当的依赖（正确接受有益建议并拒绝有害建议）和不足的依赖（错误拒绝有益建议），并结合前测和后测问卷以及一项控制实验任务，其中参与者在人工智能助手提供的准确和故意错误建议的帮助下解决编程问题，从而直接观察学生在面对不同的人工智能可靠性时的依赖模式。研究发现，适当的依赖与学生的编程自我效能感、编程素养和探究需要显著相关，而与后任务信任和满意度呈负相关。过度依赖与后任务信任和对人工智能助手的满意度显著相关。不足的依赖与编程素养、编程自我效能感和探究需要呈负相关。总体而言，研究结果提供了有关开发旨在促进适当依赖人工智能工具的干预措施的见解，这些干预措施对人工智能在课程和教育技术中的整合具有重要意义。 

---
# Sustainable Machine Learning Retraining: Optimizing Energy Efficiency Without Compromising Accuracy 

**Title (ZH)**: 可持续机器学习重训练：优化能源效率而不牺牲准确性 

**Authors**: Lorena Poenaru-Olaru, June Sallou, Luis Cruz, Jan Rellermeyer, Arie van Deursen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13838)  

**Abstract**: The reliability of machine learning (ML) software systems is heavily influenced by changes in data over time. For that reason, ML systems require regular maintenance, typically based on model retraining. However, retraining requires significant computational demand, which makes it energy-intensive and raises concerns about its environmental impact. To understand which retraining techniques should be considered when designing sustainable ML applications, in this work, we study the energy consumption of common retraining techniques. Since the accuracy of ML systems is also essential, we compare retraining techniques in terms of both energy efficiency and accuracy. We showcase that retraining with only the most recent data, compared to all available data, reduces energy consumption by up to 25\%, being a sustainable alternative to the status quo. Furthermore, our findings show that retraining a model only when there is evidence that updates are necessary, rather than on a fixed schedule, can reduce energy consumption by up to 40\%, provided a reliable data change detector is in place. Our findings pave the way for better recommendations for ML practitioners, guiding them toward more energy-efficient retraining techniques when designing sustainable ML software systems. 

**Abstract (ZH)**: 机器学习软件系统的可靠性受随时间变化的数据影响显著。因此，机器学习系统需要定期维护，通常基于模型重训练。然而，重新训练需要大量的计算资源，使其能源消耗密集并引发环境影响的担忧。为了理解在设计可持续机器学习应用时应考虑哪些重训练技术，本研究研究了常见重训练技术的能源消耗。鉴于机器学习系统准确性的重要性，我们在能源效率和准确性两方面比较了重训练技术的效果。研究表明，与使用全部可用数据相比，仅使用最新数据进行重训练可将能源消耗减少高达25%，是现状的一种可持续替代方案。此外，我们的研究发现，在有证据表明更新必要时才重新训练模型，而不是按照固定的时间表进行，可以将能源消耗减少高达40%，前提是有可靠的 数据变化检测器。我们的研究为机器学习从业者提供了更好的建议，引导他们在设计可持续机器学习软件系统时采用更节能的重训练技术。 

---
# Robustness of Reinforcement Learning-Based Traffic Signal Control under Incidents: A Comparative Study 

**Title (ZH)**: 基于强化学习的交通信号控制在遭遇事件情况下的鲁棒性：一项 comparative study 

**Authors**: Dang Viet Anh Nguyen, Carlos Lima Azevedo, Tomer Toledo, Filipe Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2506.13836)  

**Abstract**: Reinforcement learning-based traffic signal control (RL-TSC) has emerged as a promising approach for improving urban mobility. However, its robustness under real-world disruptions such as traffic incidents remains largely underexplored. In this study, we introduce T-REX, an open-source, SUMO-based simulation framework for training and evaluating RL-TSC methods under dynamic, incident scenarios. T-REX models realistic network-level performance considering drivers' probabilistic rerouting, speed adaptation, and contextual lane-changing, enabling the simulation of congestion propagation under incidents. To assess robustness, we propose a suite of metrics that extend beyond conventional traffic efficiency measures. Through extensive experiments across synthetic and real-world networks, we showcase T-REX for the evaluation of several state-of-the-art RL-TSC methods under multiple real-world deployment paradigms. Our findings show that while independent value-based and decentralized pressure-based methods offer fast convergence and generalization in stable traffic conditions and homogeneous networks, their performance degrades sharply under incident-driven distribution shifts. In contrast, hierarchical coordination methods tend to offer more stable and adaptable performance in large-scale, irregular networks, benefiting from their structured decision-making architecture. However, this comes with the trade-off of slower convergence and higher training complexity. These findings highlight the need for robustness-aware design and evaluation in RL-TSC research. T-REX contributes to this effort by providing an open, standardized and reproducible platform for benchmarking RL methods under dynamic and disruptive traffic scenarios. 

**Abstract (ZH)**: 基于强化学习的交通信号控制（RL-TSC）已成为提高城市交通流动性的有前途的方法。然而，其在如交通事件等现实世界中断情况下的鲁棒性研究仍处于起步阶段。在本研究中，我们引入了T-REX，一个开放源代码的基于SUMO的仿真框架，用于在动态和事件场景下训练和评估RL-TSC方法。T-REX考虑驾驶者的概率 rerouting、速度适应和情境车道变换，建模网络层面的真实性能，从而在事件情况下模拟拥堵传播。为了评估鲁棒性，我们提出了一套超越传统交通效率度量的指标。通过在合成和真实世界网络上的广泛实验，我们展示了T-REX在多种真实世界部署范式下评估多项先进RL-TSC方法的能力。我们的研究结果表明，虽然独立的价值基方法和分散的压力基方法在稳定交通条件和同质网络中表现出快速收敛和泛化能力，但在事件驱动的分布变化下其性能急剧下降。相比之下，分层协调方法在大型不规则网络中更能提供稳定和适应性更强的性能，得益于它们的结构化决策架构。然而，这也伴随着收敛速度较慢和更高的训练复杂性。这些发现凸显了在RL-TSC研究中鲁棒性意识设计和评估的必要性。T-REX通过提供一个开放、标准化和可重复的平台来基准测试在动态和扰动交通场景下的RL方法，为此做出了贡献。 

---
# Evolvable Conditional Diffusion 

**Title (ZH)**: 可演化条件扩散 

**Authors**: Zhao Wei, Chin Chun Ooi, Abhishek Gupta, Jian Cheng Wong, Pao-Hsiung Chiu, Sheares Xue Wen Toh, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13834)  

**Abstract**: This paper presents an evolvable conditional diffusion method such that black-box, non-differentiable multi-physics models, as are common in domains like computational fluid dynamics and electromagnetics, can be effectively used for guiding the generative process to facilitate autonomous scientific discovery. We formulate the guidance as an optimization problem where one optimizes for a desired fitness function through updates to the descriptive statistic for the denoising distribution, and derive an evolution-guided approach from first principles through the lens of probabilistic evolution. Interestingly, the final derived update algorithm is analogous to the update as per common gradient-based guided diffusion models, but without ever having to compute any derivatives. We validate our proposed evolvable diffusion algorithm in two AI for Science scenarios: the automated design of fluidic topology and meta-surface. Results demonstrate that this method effectively generates designs that better satisfy specific optimization objectives without reliance on differentiable proxies, providing an effective means of guidance-based diffusion that can capitalize on the wealth of black-box, non-differentiable multi-physics numerical models common across Science. 

**Abstract (ZH)**: 本文提出了一种可进化条件扩散方法，使得像计算流体力学和电磁学这样的领域中常见的黑盒非可微多物理模型能够有效地引导生成过程，以促进自主科学发现。我们将引导问题形式化为一个优化问题，通过更新去噪分布的描述统计量来优化期望的适应度函数，并从概率进化的视角，通过基本原则导出了一个进化引导的方法。有趣的是，最终导出的更新算法类似于常见的基于梯度引导扩散模型的更新，但无需计算任何导数。我们在两个AI for Science场景中验证了我们提出的可进化扩散算法：流体拓扑自动化设计和元表面。实验结果表明，该方法能够有效地生成能更好地满足特定优化目标的设计，而无需依赖可微近似，从而提供了一种有效的基于引导的扩散方法，能够利用跨科学领域常见的黑盒非可微多物理数值模型。 

---
# Quantifying Structure in CLIP Embeddings: A Statistical Framework for Concept Interpretation 

**Title (ZH)**: CLIP嵌入结构的量化：概念解释的统计框架 

**Authors**: Jitian Zhao, Chenghui Li, Frederic Sala, Karl Rohe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13831)  

**Abstract**: Concept-based approaches, which aim to identify human-understandable concepts within a model's internal representations, are a promising method for interpreting embeddings from deep neural network models, such as CLIP. While these approaches help explain model behavior, current methods lack statistical rigor, making it challenging to validate identified concepts and compare different techniques. To address this challenge, we introduce a hypothesis testing framework that quantifies rotation-sensitive structures within the CLIP embedding space. Once such structures are identified, we propose a post-hoc concept decomposition method. Unlike existing approaches, it offers theoretical guarantees that discovered concepts represent robust, reproducible patterns (rather than method-specific artifacts) and outperforms other techniques in terms of reconstruction error. Empirically, we demonstrate that our concept-based decomposition algorithm effectively balances reconstruction accuracy with concept interpretability and helps mitigate spurious cues in data. Applied to a popular spurious correlation dataset, our method yields a 22.6% increase in worst-group accuracy after removing spurious background concepts. 

**Abstract (ZH)**: 基于概念的方法通过识别模型内部表示中的人类可理解概念，是一种阐释深度神经网络模型（如CLIP）嵌入的有前景方法。尽管这些方法有助于解释模型行为，但现有方法缺乏统计严谨性，使得验证识别的概念和比较不同技术变得困难。为应对这一挑战，我们提出了一种假设检验框架，用于量化CLIP嵌入空间中的旋转敏感结构。一旦识别出这些结构，我们提出了一种事后概念分解方法。与现有方法不同，该方法提供了理论保证，表明发现的概念代表了稳健且可重复的模式（而非方法特定的伪影），在重构误差方面也优于其他技术。实证研究显示，我们的基于概念的分解算法在提升重构准确性和概念可解释性方面取得了平衡，并有助于减轻数据中的伪线索。应用于一个流行的伪相关数据集后，我们的方法在移除伪背景概念后，最差组的准确性提高了22.6%。 

---
# MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios 

**Title (ZH)**: MLDebugging: 面向跨多库场景的代码调试基准测试 

**Authors**: Jinyang Huang, Xiachong Feng, Qiguang Chen, Hanjie Zhao, Zihui Cheng, Jiesong Bai, Jingxuan Zhou, Min Li, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13824)  

**Abstract**: Code debugging is a crucial task in software engineering, which attracts increasing attention. While remarkable success has been made in the era of large language models (LLMs), current research still focuses on the simple no-library or single-library setting, ignoring the complex multi-library scenario in real-world applications. To address this limitation, we make the first attempt to introduce MLDebugging (Multi-Library Debugging), a comprehensive benchmark designed to assess debugging challenges within multi-library Python code. Specifically, MLDebugging encompasses 126 distinct Python libraries, covering a wide range of multi-library code issues, categorized into seven distinct types. Furthermore, we conduct a thorough evaluation of MLDebugging using both mainstream open-source and closed-source LLMs and highlight that current LLMs still struggle to correctly perform code debugging across multi-library scenarios. We hope this work can uncover the potential of LLMs in multi-library debugging scenario and offer insights for future research. 

**Abstract (ZH)**: 多库调试是软件工程中的关键任务，吸引了越来越多的关注。尽管在大规模语言模型（LLMs）时代取得了显著进展，当前研究仍主要集中在无库或单库设置上，忽视了真实世界应用中的复杂多库场景。为解决这一局限，我们首次尝试引入MLDebugging（多库调试）这一综合基准，用于评估多库Python代码中的调试挑战。具体而言，MLDebugging 包含126个不同的Python库，涵盖了广泛的多库代码问题，并按七大类进行分类。此外，我们使用主流的开源和闭源LLM对MLDebugging进行了全面评估，指出当前的LLM仍然难以在多库场景中正确执行代码调试。我们希望这项工作能揭示LLMs在多库调试场景中的潜力，并为未来的研究提供见解。 

---
# DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models 

**Title (ZH)**: DeepSeq：通过网络搜索增强的代理生成人工智能基础模型单细胞RNA测序数据标注高通量方法 

**Authors**: Saleem A. Al Dajani, Abel Sanchez, John R. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.13817)  

**Abstract**: Generative AI foundation models offer transformative potential for processing structured biological data, particularly in single-cell RNA sequencing, where datasets are rapidly scaling toward billions of cells. We propose the use of agentic foundation models with real-time web search to automate the labeling of experimental data, achieving up to 82.5% accuracy. This addresses a key bottleneck in supervised learning for structured omics data by increasing annotation throughput without manual curation and human error. Our approach enables the development of virtual cell foundation models capable of downstream tasks such as cell-typing and perturbation prediction. As data volume grows, these models may surpass human performance in labeling, paving the way for reliable inference in large-scale perturbation screens. This application demonstrates domain-specific innovation in health monitoring and diagnostics, aligned with efforts like the Human Cell Atlas and Human Tumor Atlas Network. 

**Abstract (ZH)**: 生成式AI基础模型为处理结构化生物数据提供了变革性的潜力，特别是在单细胞RNA测序领域，其中数据集正迅速扩大至数十亿细胞。我们提出了结合自主基础模型和实时网络搜索自动标注实验数据的方法，实现了高达82.5%的准确率。这通过提高注释 throughput，消除了手动编目和人为错误的关键瓶颈，从而在监督学习中处理结构化组学数据。我们的方法使开发虚拟细胞基础模型成为可能，这些模型能够执行下游任务，如细胞分类和扰动预测。随着数据量的增长，这些模型可能在注释方面超越人类表现，为大规模扰动筛选提供可靠推理。该应用展示了健康监测和诊断领域的特定领域创新，与人类细胞图谱和人类肿瘤图谱网络等努力相一致。 

---
# Analysis and Optimization of Probabilities of Beneficial Mutation and Crossover Recombination in a Hamming Space 

**Title (ZH)**: Hamming空间中有利突变和交叉重组概率的分析与优化 

**Authors**: Roman V. Belavkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13809)  

**Abstract**: Inspired by Fisher's geometric approach to study beneficial mutations, we analyse probabilities of beneficial mutation and crossover recombination of strings in a general Hamming space with arbitrary finite alphabet. Mutations and recombinations that reduce the distance to an optimum are considered as beneficial. Geometric and combinatorial analysis is used to derive closed-form expressions for transition probabilities between spheres around an optimum giving a complete description of Markov evolution of distances from an optimum over multiple generations. This paves the way for optimization of parameters of mutation and recombination operators. Here we derive optimality conditions for mutation and recombination radii maximizing the probabilities of mutation and crossover into the optimum. The analysis highlights important differences between these evolutionary operators. While mutation can potentially reach any part of the search space, the probability of beneficial mutation decreases with distance to an optimum, and the optimal mutation radius or rate should also decrease resulting in a slow-down of evolution near the optimum. Crossover recombination, on the other hand, acts in a subspace of the search space defined by the current population of strings. However, probabilities of beneficial and deleterious crossover are balanced, and their characteristics, such as variance, are translation invariant in a Hamming space, suggesting that recombination may complement mutation and boost the rate of evolution near the optimum. 

**Abstract (ZH)**: 受Fishers几何方法研究有益突变的启发，我们分析了一般哈明空间中任意有限字母字符串的有益突变和交叉重组的概率。减少到最优值距离的突变和重组被视为有益。通过几何和组合分析，我们推导出最优值周围球体内转换概率的闭式表达式，完整描述了多个世代中距离最优值的马尔可夫演化过程。这为优化突变和重组操作的参数铺平了道路。我们推导了使突变和交叉重组到最优值的概率最大化的突变和重组半径的最优性条件。分析突出了这些进化操作之间的重要差异。虽然突变有可能到达搜索空间的任何部分，但接近最优值时有益突变的概率会降低，因此最优突变半径或速率也应减少，导致进化在接近最优值时减慢。另一方面，交叉重组作用于由当前字符串群体定义的搜索空间的一个子空间。然而，有益和有害交叉重组的概率是平衡的，它们在哈明空间中的特征，如方差，是平移不变的，这表明重组可能补充突变，并促进在接近最优值时的进化速率。 

---
# BraTS orchestrator : Democratizing and Disseminating state-of-the-art brain tumor image analysis 

**Title (ZH)**: BraTS协调者：普及和传播最先进的脑肿瘤图像分析技术 

**Authors**: Florian Kofler, Marcel Rosier, Mehdi Astaraki, Ujjwal Baid, Hendrik Möller, Josef A. Buchner, Felix Steinbauer, Eva Oswald, Ezequiel de la Rosa, Ivan Ezhov, Constantin von See, Jan Kirschke, Anton Schmick, Sarthak Pati, Akis Linardos, Carla Pitarch, Sanyukta Adap, Jeffrey Rudie, Maria Correia de Verdier, Rachit Saluja, Evan Calabrese, Dominic LaBella, Mariam Aboian, Ahmed W. Moawad, Nazanin Maleki, Udunna Anazodo, Maruf Adewole, Marius George Linguraru, Anahita Fathi Kazerooni, Zhifan Jiang, Gian Marco Conte, Hongwei Li, Juan Eugenio Iglesias, Spyridon Bakas, Benedikt Wiestler, Marie Piraud, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2506.13807)  

**Abstract**: The Brain Tumor Segmentation (BraTS) cluster of challenges has significantly advanced brain tumor image analysis by providing large, curated datasets and addressing clinically relevant tasks. However, despite its success and popularity, algorithms and models developed through BraTS have seen limited adoption in both scientific and clinical communities. To accelerate their dissemination, we introduce BraTS orchestrator, an open-source Python package that provides seamless access to state-of-the-art segmentation and synthesis algorithms for diverse brain tumors from the BraTS challenge ecosystem. Available on GitHub (this https URL), the package features intuitive tutorials designed for users with minimal programming experience, enabling both researchers and clinicians to easily deploy winning BraTS algorithms for inference. By abstracting the complexities of modern deep learning, BraTS orchestrator democratizes access to the specialized knowledge developed within the BraTS community, making these advances readily available to broader neuro-radiology and neuro-oncology audiences. 

**Abstract (ZH)**: BraTS挑战集群在脑肿瘤分割方面显著推动了脑肿瘤图像分析的进步，通过提供大规模的数据集和解决临床相关任务。尽管BraTS在成功和受欢迎程度方面取得了这些成就，但通过BraTS开发的算法和模型在科学和临床社区中的应用仍然有限。为了加速其推广，我们介绍了BraTS orchestrator，这是一个开源的Python包，提供了无缝访问BraTS挑战生态系统中多种脑肿瘤先进分割和合成算法的途径。该包可在GitHub（此链接请点击：https://github.com/）上获得，包含针对编程经验有限的用户设计的直观教程，使研究人员和临床医生能够轻松部署BraTS比赛中的获胜算法进行推断。通过抽象现代深度学习的复杂性，BraTS orchestrator使BraTS社区开发的专业知识普及化，使这些进展能够被更广泛的神经放射学和神经肿瘤学受众方便地使用。 

---
# Instruction and Solution Probabilities as Heuristics for Inductive Programming 

**Title (ZH)**: 指令和解决方案概率作为归纳编程的启发式方法 

**Authors**: Edward McDaid, Sarah McDaid  

**Link**: [PDF](https://arxiv.org/pdf/2506.13804)  

**Abstract**: Instruction subsets (ISs) are heuristics that can shrink the size of the inductive programming (IP) search space by tens of orders of magnitude. Here, we extend the IS approach by introducing instruction and solution probabilities as additional heuristics. Instruction probability reflects the expectation of an instruction occurring in a solution, based on the frequency of instruction occurrence in a large code sample. The solution probability for a partial or complete program is simply the product of all constituent instruction probabilities, including duplicates. We treat the minimum solution probabilities observed in code sample program units of different sizes as solution probability thresholds. These thresholds are used to prune the search space as partial solutions are constructed, thereby eliminating any branches containing unlikely combinations of instructions. The new approach has been evaluated using a large sample of human code. We tested two formulations of instruction probability: one based on instruction occurrence across the entire code sample and another that measured the distribution separately for each IS. Our results show that both variants produce substantial further reductions in the IP search space size of up to tens of orders of magnitude, depending on solution size. In combination with IS, reductions of over 100 orders of magnitude can be achieved. We also carried out cross-validation testing to show that the heuristics should work effectively with unseen code. The approach is described and the results and some ideas for future work are discussed. 

**Abstract (ZH)**: 启发式指令子集（IS）可以通过减少归纳编程（IP）搜索空间的规模来缩小数十个数量级。在此基础上，我们通过引入指令和解决方案概率作为额外的启发式方法来扩展IS方法。指令概率反映了指令在大型代码样本中出现频率的期望值，作为其在解决方案中出现的预期。解决方案概率是所有组成部分指令概率的乘积，包括重复的指令。我们将不同大小的代码样本程序单元中观察到的最小解决方案概率视为解决方案概率阈值。这些阈值用于在构建部分解决方案时修剪搜索空间，从而排除任何包含不太可能出现的指令组合的分支。我们使用大量的人类代码样本进行了新方法的评估。我们测试了两种指令概率的公式：一种基于代码样本中指令的总体出现频率，另一种则分别测量每个指令子集的分布。结果显示，这两种变体都能进一步减少高达数十个数量级的IP搜索空间大小，具体取决于解决方案的大小。与指令子集结合使用时，可以实现超过100个数量级的减少。我们还进行了交叉验证测试，以表明这些启发式方法应能有效应用于未见过的代码。我们将描述该方法并讨论实验结果和一些未来研究的想法。 

---
# Contemporary AI foundation models increase biological weapons risk 

**Title (ZH)**: 当代AI基础模型增加了生物武器风险 

**Authors**: Roger Brent, T. Greg McKelvey Jr  

**Link**: [PDF](https://arxiv.org/pdf/2506.13798)  

**Abstract**: The rapid advancement of artificial intelligence has raised concerns about its potential to facilitate biological weapons development. We argue existing safety assessments of contemporary foundation AI models underestimate this risk, largely due to flawed assumptions and inadequate evaluation methods. First, assessments mistakenly assume biological weapons development requires tacit knowledge, or skills gained through hands-on experience that cannot be easily verbalized. Second, they rely on imperfect benchmarks that overlook how AI can uplift both nonexperts and already-skilled individuals. To challenge the tacit knowledge assumption, we examine cases where individuals without formal expertise, including a 2011 Norwegian ultranationalist who synthesized explosives, successfully carried out complex technical tasks. We also review efforts to document pathogen construction processes, highlighting how such tasks can be conveyed in text. We identify "elements of success" for biological weapons development that large language models can describe in words, including steps such as acquiring materials and performing technical procedures. Applying this framework, we find that advanced AI models Llama 3.1 405B, ChatGPT-4o, and Claude 3.5 Sonnet can accurately guide users through the recovery of live poliovirus from commercially obtained synthetic DNA, challenging recent claims that current models pose minimal biosecurity risk. We advocate for improved benchmarks, while acknowledging the window for meaningful implementation may have already closed. 

**Abstract (ZH)**: 人工智能的迅猛发展引发了对其可能促进生物武器开发潜在风险的关注。我们argue现有对当代基础人工智能模型的安全评估低估了这一风险，主要是由于错误的假设和不充分的评估方法。首先，评估错误地假设生物武器开发需要默会知识，即通过实际操作获得但难以口头表达的技能。其次，它们依赖于不完善的基准，忽视了人工智能如何提升非专家和已有技能者的水平。为挑战默会知识的假设，我们考察了没有正式专业知识的个体，如2011年挪威极右分子合成爆炸物的案例，分析他们成功完成复杂技术任务的情况。我们还审查了病原体构建过程的记录，强调这些任务可以通过文本传达。我们发现在文字中能够描述生物武器开发成功的“要素”，包括获取材料和技术操作步骤等。应用这一框架，我们发现最先进的AI模型Llama 3.1 405B、ChatGPT-4o和Claude 3.5 Sonnet能够准确引导用户从商业合成DNA中恢复活病毒，挑战了近期关于当前模型对生物安全风险较小的观点。我们呼吁改进基准，同时承认实施有意义的改进的机会可能已不再存在。 

---
# Analysis of Anonymous User Interaction Relationships and Prediction of Advertising Feedback Based on Graph Neural Network 

**Title (ZH)**: 基于图神经网络的匿名用户交互关系分析及广告反馈预测 

**Authors**: Yanjun Dai, Haoyang Feng, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13787)  

**Abstract**: While online advertising is highly dependent on implicit interaction networks of anonymous users for engagement inference, and for the selection and optimization of delivery strategies, existing graph models seldom can capture the multi-scale temporal, semantic and higher-order dependency features of these interaction networks, thus it's hard to describe the complicated patterns of the anonymous behavior. In this paper, we propose Decoupled Temporal-Hierarchical Graph Neural Network (DTH-GNN), which achieves three main contributions. Above all, we introduce temporal edge decomposition, which divides each interaction into three types of channels: short-term burst, diurnal cycle and long-range memory, and conducts feature extraction using the convolution kernel of parallel dilated residuals; Furthermore, our model builds a hierarchical heterogeneous aggregation, where user-user, user-advertisement, advertisement-advertisement subgraphs are combined through the meta-path conditional Transformer encoder, where the noise structure is dynamically tamped down via the synergy of cross-channel self-attention and gating relationship selector. Thirdly, the contrast regularity of feedback perception is formulated, the consistency of various time slices is maximized, the entropy of control exposure information with dual-view target is maximized, the global prototype of dual-momentum queue distillation is presented, and the strategy gradient layer with light weight is combined with delaying transformation signal to fine-tune the node representation for benefit-oriented. The AUC of DTH-GNN improved by 8.2% and the logarithmic loss improved by 5.7% in comparison with the best baseline model. 

**Abstract (ZH)**: 解耦时序层次图神经网络（DTH-GNN）及其在匿名用户行为模式识别中的应用 

---
# Enhancing Bagging Ensemble Regression with Data Integration for Time Series-Based Diabetes Prediction 

**Title (ZH)**: 基于数据集成的袋装聚类回归增强方法在时间序列糖尿病预测中的应用 

**Authors**: Vuong M. Ngo, Tran Quang Vinh, Patricia Kearney, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2506.13786)  

**Abstract**: Diabetes is a chronic metabolic disease characterized by elevated blood glucose levels, leading to complications like heart disease, kidney failure, and nerve damage. Accurate state-level predictions are vital for effective healthcare planning and targeted interventions, but in many cases, data for necessary analyses are incomplete. This study begins with a data engineering process to integrate diabetes-related datasets from 2011 to 2021 to create a comprehensive feature set. We then introduce an enhanced bagging ensemble regression model (EBMBag+) for time series forecasting to predict diabetes prevalence across U.S. cities. Several baseline models, including SVMReg, BDTree, LSBoost, NN, LSTM, and ERMBag, were evaluated for comparison with our EBMBag+ algorithm. The experimental results demonstrate that EBMBag+ achieved the best performance, with an MAE of 0.41, RMSE of 0.53, MAPE of 4.01, and an R2 of 0.9. 

**Abstract (ZH)**: 糖尿病是一种以高血糖为特征的慢性代谢疾病，会导致心脏病、肾衰竭和神经损伤等并发症。准确的州级预测对于有效的医疗规划和有针对性的干预至关重要，但在许多情况下，用于必要分析的数据是不完整的。本研究首先进行数据工程，整合2011年至2021年的糖尿病相关数据集，创建综合特征集。然后，我们引入了增强袋装集成回归模型(EBMBag+)进行时间序列预测，以预测美国城市糖尿病患病率。我们将EBMBag+算法与多个基线模型进行比较，包括SVMReg、BDTree、LSBoost、NN、LSTM和ERMBag。实验结果表明，EBMBag+取得了最佳性能，平均绝对误差（MAE）为0.41，均方误差（RMSE）为0.53，均绝对百分比误差（MAPE）为4.01，R²值为0.9。 

---
# Solving the Job Shop Scheduling Problem with Graph Neural Networks: A Customizable Reinforcement Learning Environment 

**Title (ZH)**: 使用图神经网络解决作业车间调度问题：可定制的强化学习环境 

**Authors**: Pablo Ariño Fernández, Carlos Quesada González  

**Link**: [PDF](https://arxiv.org/pdf/2506.13781)  

**Abstract**: The job shop scheduling problem is an NP-hard combinatorial optimization problem relevant to manufacturing and timetabling. Traditional approaches use priority dispatching rules based on simple heuristics. Recent work has attempted to replace these with deep learning models, particularly graph neural networks (GNNs), that learn to assign priorities from data. However, training such models requires customizing numerous factors: graph representation, node features, action space, and reward functions. The lack of modular libraries for experimentation makes this research time-consuming. This work introduces JobShopLib, a modular library that allows customizing these factors and creating new components with its reinforcement learning environment. We trained several dispatchers through imitation learning to demonstrate the environment's utility. One model outperformed various graph-based dispatchers using only individual operation features, highlighting the importance of feature customization. Our GNN model achieved near state-of-the-art results on large-scale problems. These results suggest significant room for improvement in developing such models. JobShopLib provides the necessary tools for future experimentation. 

**Abstract (ZH)**: 作业车间调度问题是与制造和排程相关的NP难组合优化问题。传统的方法基于简单的启发式规则使用优先级调度规则。最近的研究尝试用深度学习模型，尤其是图神经网络（GNNs），来替代这些方法，让模型从数据中学习优先级分配。然而，训练这样的模型需要定制众多因素：图表示、节点特征、操作空间和奖励函数。缺乏模块化库使得这一研究耗时。本工作介绍了一个模块化库JobShopLib，它允许定制这些因素，并使用其强化学习环境创建新组件。通过模仿学习训练了几种调度器以展示该环境的实用性。一个模型仅使用单个操作特征就优于各种基于图的调度器，强调了特征定制的重要性。我们的GNN模型在大规模问题上达到了接近最先进的结果。这些结果表明在开发此类模型方面有巨大的改进空间。JobShopLib为未来的实验提供了必要的工具。 

---
# Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning 

**Title (ZH)**: 基于问题生成的知识压缩：无需微调增强多跳文档检索 

**Authors**: Anvi Alex Eponon, Moein Shahiki-Tash, Ildar Batyrshin, Christian E. Maldonado-Sifuentes, Grigori Sidorov, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13778)  

**Abstract**: This study presents a question-based knowledge encoding approach that improves retrieval-augmented generation (RAG) systems without requiring fine-tuning or traditional chunking. We encode textual content using generated questions that span the lexical and semantic space, creating targeted retrieval cues combined with a custom syntactic reranking method.
In single-hop retrieval over 109 scientific papers, our approach achieves a Recall@3 of 0.84, outperforming traditional chunking methods by 60 percent. We also introduce "paper-cards", concise paper summaries under 300 characters, which enhance BM25 retrieval, increasing MRR@3 from 0.56 to 0.85 on simplified technical queries.
For multihop tasks, our reranking method reaches an F1 score of 0.52 with LLaMA2-Chat-7B on the LongBench 2WikiMultihopQA dataset, surpassing chunking and fine-tuned baselines which score 0.328 and 0.412 respectively.
This method eliminates fine-tuning requirements, reduces retrieval latency, enables intuitive question-driven knowledge access, and decreases vector storage demands by 80%, positioning it as a scalable and efficient RAG alternative. 

**Abstract (ZH)**: 基于问题的知識編碼方法：改進 Retrieval-Augmented Generation 系統而不需微調或傳統分塊 

---
# A Survey of Physics-Informed AI for Complex Urban Systems 

**Title (ZH)**: 物理学知情的人工智能在复杂城市系统中的调研 

**Authors**: En Xu, Huandong Wang, Yunke Zhang, Sibo Li, Yinzhou Tang, Zhilun Zhou, Yuming Lin, Yuan Yuan, Xiaochen Fan, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13777)  

**Abstract**: Urban systems are typical examples of complex systems, where the integration of physics-based modeling with artificial intelligence (AI) presents a promising paradigm for enhancing predictive accuracy, interpretability, and decision-making. In this context, AI excels at capturing complex, nonlinear relationships, while physics-based models ensure consistency with real-world laws and provide interpretable insights. We provide a comprehensive review of physics-informed AI methods in urban applications. The proposed taxonomy categorizes existing approaches into three paradigms - Physics-Integrated AI, Physics-AI Hybrid Ensemble, and AI-Integrated Physics - and further details seven representative methods. This classification clarifies the varying degrees and directions of physics-AI integration, guiding the selection and development of appropriate methods based on application needs and data availability. We systematically examine their applications across eight key urban domains: energy, environment, economy, transportation, information, public services, emergency management, and the urban system as a whole. Our analysis highlights how these methodologies leverage physical laws and data-driven models to address urban challenges, enhancing system reliability, efficiency, and adaptability. By synthesizing existing methodologies and their urban applications, we identify critical gaps and outline future research directions, paving the way toward next-generation intelligent urban system modeling. 

**Abstract (ZH)**: 城市系统是典型的复杂系统，在此基础上将基于物理的建模与人工智能相结合，为提高预测准确性、可解释性和决策制定提供了有前景的范式。在这种背景下，人工智能在捕捉复杂非线性关系方面表现出色，而基于物理的模型则确保与现实世界的规律保持一致并提供可解释的洞察。我们对城市应用中的物理信息人工智能方法进行了全面综述。提出的分类方案将现有方法分为三种范式——物理集成AI、物理-AI混合集成、AI集成物理，并进一步详细介绍了七种代表性方法。这种分类明确了物理-AI集成的差异程度和方向，指导根据应用需求和数据可用性选择和开发适当的方法。我们系统地考察了这些方法在八个关键城市领域的应用：能源、环境、经济、交通、信息、公共服务、应急管理以及城市系统整体。我们的分析突显了这些方法如何利用物理定律和数据驱动的模型来解决城市挑战，增强系统的可靠性和适应性。通过对现有方法及其城市应用的综合研究，我们识别出关键的缺口，并勾画出未来研究方向，为下一代智能城市系统建模指明了方向。 

---
