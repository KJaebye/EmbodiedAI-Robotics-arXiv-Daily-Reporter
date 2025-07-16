# ILCL: Inverse Logic-Constraint Learning from Temporally Constrained Demonstrations 

**Title (ZH)**: ILCL: 基于时间约束演示的逆逻辑约束学习 

**Authors**: Minwoo Cho, Jaehwi Jang, Daehyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11000)  

**Abstract**: We aim to solve the problem of temporal-constraint learning from demonstrations to reproduce demonstration-like logic-constrained behaviors. Learning logic constraints is challenging due to the combinatorially large space of possible specifications and the ill-posed nature of non-Markovian constraints. To figure it out, we introduce a novel temporal-constraint learning method, which we call inverse logic-constraint learning (ILCL). Our method frames ICL as a two-player zero-sum game between 1) a genetic algorithm-based temporal-logic mining (GA-TL-Mining) and 2) logic-constrained reinforcement learning (Logic-CRL). GA-TL-Mining efficiently constructs syntax trees for parameterized truncated linear temporal logic (TLTL) without predefined templates. Subsequently, Logic-CRL finds a policy that maximizes task rewards under the constructed TLTL constraints via a novel constraint redistribution scheme. Our evaluations show ILCL outperforms state-of-the-art baselines in learning and transferring TL constraints on four temporally constrained tasks. We also demonstrate successful transfer to real-world peg-in-shallow-hole tasks. 

**Abstract (ZH)**: 我们旨在通过演示解决时间约束学习问题，以重现类似演示的逻辑约束行为。学习逻辑约束具有挑战性，因为可能的规范空间极为庞大且非马尔可夫约束问题描述不明确。为此，我们提出了一种新颖的时间约束学习方法，称为逆向逻辑约束学习（ILCL）。该方法将ILCL视为遗传算法基于时间逻辑挖掘（GA-TL-Mining）与逻辑约束强化学习（Logic-CRL）之间的零和博弈。GA-TL-Mining高效地构建了参数化截断线性时序逻辑（TLTL）的语法树，而无需预定义模板。随后，Logic-CRL通过一种新颖的约束重分布方案，在构建的TLTL约束下寻找最大化任务奖励的策略。我们的评估展示了ILCL在四个时间约束任务上学习和转移TL约束方面优于现有基准。我们还展示了其在实际应用中的成功迁移，特别是固定销入浅孔任务。 

---
# RCG: Safety-Critical Scenario Generation for Robust Autonomous Driving via Real-World Crash Grounding 

**Title (ZH)**: RCG：通过现实碰撞接地实现稳健自动驾驶的安全关键场景生成 

**Authors**: Benjamin Stoler, Juliet Yang, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10749)  

**Abstract**: Safety-critical scenarios are essential for training and evaluating autonomous driving (AD) systems, yet remain extremely rare in real-world driving datasets. To address this, we propose Real-world Crash Grounding (RCG), a scenario generation framework that integrates crash-informed semantics into adversarial perturbation pipelines. We construct a safety-aware behavior representation through contrastive pre-training on large-scale driving logs, followed by fine-tuning on a small, crash-rich dataset with approximate trajectory annotations extracted from video. This embedding captures semantic structure aligned with real-world accident behaviors and supports selection of adversary trajectories that are both high-risk and behaviorally realistic. We incorporate the resulting selection mechanism into two prior scenario generation pipelines, replacing their handcrafted scoring objectives with an embedding-based criterion. Experimental results show that ego agents trained against these generated scenarios achieve consistently higher downstream success rates, with an average improvement of 9.2% across seven evaluation settings. Qualitative and quantitative analyses further demonstrate that our approach produces more plausible and nuanced adversary behaviors, enabling more effective and realistic stress testing of AD systems. Code and tools will be released publicly. 

**Abstract (ZH)**: 基于现实碰撞事件的自动驾驶安全场景生成框架（Real-world Crash Grounding for Autonomous Driving Scenario Generation） 

---
# Perspective-Aware AI in Extended Reality 

**Title (ZH)**: 视角感知AI在扩展现实中的应用 

**Authors**: Daniel Platnick, Matti Gruener, Marjan Alirezaie, Kent Larson, Dava J. Newman, Hossein Rahnama  

**Link**: [PDF](https://arxiv.org/pdf/2507.11479)  

**Abstract**: AI-enhanced Extended Reality (XR) aims to deliver adaptive, immersive experiences-yet current systems fall short due to shallow user modeling and limited cognitive context. We introduce Perspective-Aware AI in Extended Reality (PAiR), a foundational framework for integrating Perspective-Aware AI (PAi) with XR to enable interpretable, context-aware experiences grounded in user identity. PAi is built on Chronicles: reasoning-ready identity models learned from multimodal digital footprints that capture users' cognitive and experiential evolution. PAiR employs these models in a closed-loop system linking dynamic user states with immersive environments. We present PAiR's architecture, detailing its modules and system flow, and demonstrate its utility through two proof-of-concept scenarios implemented in the Unity-based OpenDome engine. PAiR opens a new direction for human-AI interaction by embedding perspective-based identity models into immersive systems. 

**Abstract (ZH)**: AI增强扩展现实（XR）旨在提供适应性和沉浸式体验—但由于浅层用户建模和有限的认知上下文，当前系统尚存不足。我们提出一种基于视角意识AI在扩展现实中的框架（PAiR），以整合视角意识AI（PAi）并使体验基于用户的身份具有可解释性和上下文意识。PAi基于Chronicles构建：从多模态数字足迹中学习的认知准备就绪身份模型，捕捉用户的认知和体验演变。PAiR通过闭环系统将动态用户状态与沉浸式环境联系起来应用这些模型。我们展示了PAiR的架构，详细说明其模块和系统流程，并通过在基于Unity的OpenDome引擎中实现的两个概念验证场景展示了其实用性。PAiR为基于视角身份模型的人机交互开辟了新方向。 

---
# Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety 

**Title (ZH)**: Chain of Thought Monitorability: 一种新的脆弱性机会，关乎AI安全 

**Authors**: Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mądry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, Vlad Mikulik  

**Link**: [PDF](https://arxiv.org/pdf/2507.11473)  

**Abstract**: AI systems that "think" in human language offer a unique opportunity for AI safety: we can monitor their chains of thought (CoT) for the intent to misbehave. Like all other known AI oversight methods, CoT monitoring is imperfect and allows some misbehavior to go unnoticed. Nevertheless, it shows promise and we recommend further research into CoT monitorability and investment in CoT monitoring alongside existing safety methods. Because CoT monitorability may be fragile, we recommend that frontier model developers consider the impact of development decisions on CoT monitorability. 

**Abstract (ZH)**: 基于人类语言“思考”的AI系统为AI安全提供了独特机会：我们可以通过监控其思维链（CoT）来检查其是否有违规意图。尽管CoT监控方法并不完善，并且可能会漏掉一些违规行为，但这种方法显示出了潜力，我们建议进一步研究CoT可监控性，并在现有安全方法之外投资于CoT监控。由于CoT可监控性可能较为脆弱，我们建议前沿模型开发者考虑其开发决策对CoT可监控性的影响。 

---
# Contestability in Quantitative Argumentation 

**Title (ZH)**: 量化论证中的可议性 

**Authors**: Xiang Yin, Nico Potyka, Antonio Rago, Timotheus Kampik, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.11323)  

**Abstract**: Contestable AI requires that AI-driven decisions align with human preferences. While various forms of argumentation have been shown to support contestability, Edge-Weighted Quantitative Bipolar Argumentation Frameworks (EW-QBAFs) have received little attention. In this work, we show how EW-QBAFs can be deployed for this purpose. Specifically, we introduce the contestability problem for EW-QBAFs, which asks how to modify edge weights (e.g., preferences) to achieve a desired strength for a specific argument of interest (i.e., a topic argument). To address this problem, we propose gradient-based relation attribution explanations (G-RAEs), which quantify the sensitivity of the topic argument's strength to changes in individual edge weights, thus providing interpretable guidance for weight adjustments towards contestability. Building on G-RAEs, we develop an iterative algorithm that progressively adjusts the edge weights to attain the desired strength. We evaluate our approach experimentally on synthetic EW-QBAFs that simulate the structural characteristics of personalised recommender systems and multi-layer perceptrons, and demonstrate that it can solve the problem effectively. 

**Abstract (ZH)**: 可争议AI要求AI驱动的决策与人类偏好保持一致。虽然各种形式的论证已被证明可以支持可争议性，但边缘加权定量 bipolar 论证框架（EW-QBAFs）尚未引起广泛关注。在本文中，我们展示了如何使用EW-QBAFs实现这一目标。具体来说，我们引入了EW-QBAFs的可争议性问题，该问题探讨如何通过修改边权重（例如，偏好）来实现特定论题的期望强度。为了解决这一问题，我们提出了基于梯度的关系归因解释（G-RAEs），以量化论题论点强度对个体边权重变化的敏感性，从而提供具有可解释性的指导，用于权重调整以促进可争议性。基于G-RAEs，我们开发了一种迭代算法，以渐进方式调整边权重以达到期望的强度。我们通过合成EW-QBAFs进行实验性评估，这些合成框架模拟了个性化推荐系统和多层感知机的结构特性，并证明了该方法可以有效解决该问题。 

---
# DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion 

**Title (ZH)**: DuetGraph: 从粗到细的知识图谱推理与双重路径全局-局部融合 

**Authors**: Jin Li, Zezhong Ding, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.11229)  

**Abstract**: Knowledge graphs (KGs) are vital for enabling knowledge reasoning across various domains. Recent KG reasoning methods that integrate both global and local information have achieved promising results. However, existing methods often suffer from score over-smoothing, which blurs the distinction between correct and incorrect answers and hinders reasoning effectiveness. To address this, we propose DuetGraph, a coarse-to-fine KG reasoning mechanism with dual-pathway global-local fusion. DuetGraph tackles over-smoothing by segregating -- rather than stacking -- the processing of local (via message passing) and global (via attention) information into two distinct pathways, preventing mutual interference and preserving representational discrimination. In addition, DuetGraph introduces a coarse-to-fine optimization, which partitions entities into high- and low-score subsets. This strategy narrows the candidate space and sharpens the score gap between the two subsets, which alleviates over-smoothing and enhances inference quality. Extensive experiments on various datasets demonstrate that DuetGraph achieves state-of-the-art (SOTA) performance, with up to an 8.7% improvement in reasoning quality and a 1.8$\times$ acceleration in training efficiency. 

**Abstract (ZH)**: 基于粗细路径全局-局部融合的知识图谱推理机制DuetGraph 

---
# Fine-grained Timing Analysis of Digital Integrated Circuits in Answer Set Programming 

**Title (ZH)**: 数字集成电路的细粒度时序分析在回答集程序设计中 

**Authors**: Alessandro Bertagnon, Marcello Dalpasso, Michele Favalli, Marco Gavanelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.11150)  

**Abstract**: In the design of integrated circuits, one critical metric is the maximum delay introduced by combinational modules within the circuit. This delay is crucial because it represents the time required to perform a computation: in an Arithmetic-Logic Unit it represents the maximum time taken by the circuit to perform an arithmetic operation. When such a circuit is part of a larger, synchronous system, like a CPU, the maximum delay directly impacts the maximum clock frequency of the entire system. Typically, hardware designers use Static Timing Analysis to compute an upper bound of the maximum delay because it can be determined in polynomial time. However, relying on this upper bound can lead to suboptimal processor speeds, thereby missing performance opportunities. In this work, we tackle the challenging task of computing the actual maximum delay, rather than an approximate value. Since the problem is computationally hard, we model it in Answer Set Programming (ASP), a logic language featuring extremely efficient solvers. We propose non-trivial encodings of the problem into ASP. Experimental results show that ASP is a viable solution to address complex problems in hardware design. 

**Abstract (ZH)**: 在集成电路设计中，一个关键指标是电路中组合模块引入的最大延迟。这一延迟至关重要，因为它代表了完成计算所需的时间：在算术逻辑单元中，它代表了电路执行算术操作所花费的最大时间。当这样一个电路是更大规模同步系统（如CPU）的一部分时，最大延迟直接关系到系统的最大时钟频率。通常，硬件设计师使用静态时序分析来计算最大延迟的一个上限，因为这个上限可以在多项式时间内确定。然而，依赖这一上限可能导致处理器速度不足，从而错失性能机会。在本工作中，我们致力于计算实际的最大延迟，而非一个近似值。由于该问题是计算上困难的，我们将其建模为回答集编程（ASP），这是一种包含极其高效求解器的逻辑语言。我们提出了问题的非平凡编码方法。实验结果表明，ASP 是解决硬件设计中复杂问题的一个可行方案。 

---
# Collaborative Trustworthiness for Good Decision Making in Autonomous Systems 

**Title (ZH)**: 自主系统中协同可信性对良好决策的影响 

**Authors**: Selma Saidi, Omar Laimona, Christoph Schmickler, Dirk Ziegenbein  

**Link**: [PDF](https://arxiv.org/pdf/2507.11135)  

**Abstract**: Autonomous systems are becoming an integral part of many application domains, like in the mobility sector. However, ensuring their safe and correct behaviour in dynamic and complex environments remains a significant challenge, where systems should autonomously make decisions e.g., about manoeuvring. We propose in this paper a general collaborative approach for increasing the level of trustworthiness in the environment of operation and improve reliability and good decision making in autonomous system. In the presence of conflicting information, aggregation becomes a major issue for trustworthy decision making based on collaborative data sharing. Unlike classical approaches in the literature that rely on consensus or majority as aggregation rule, we exploit the fact that autonomous systems have different quality attributes like perception quality. We use this criteria to determine which autonomous systems are trustworthy and borrow concepts from social epistemology to define aggregation and propagation rules, used for automated decision making. We use Binary Decision Diagrams (BDDs) as formal models for beliefs aggregation and propagation, and formulate reduction rules to reduce the size of the BDDs and allow efficient computation structures for collaborative automated reasoning. 

**Abstract (ZH)**: 自主系统正逐渐成为许多应用领域的核心组成部分，如在移动领域。然而，在动态和复杂环境中确保其安全正确的行为仍然是一个重大挑战，系统需要自主做出决策，例如关于机动行为。本文提出了一种通用的协作方法，以提高运行环境的信任度，并改善自主系统的可靠性和良好的决策能力。在存在冲突信息的情况下，聚合成为基于协作数据共享进行可信决策的主要问题。与文献中依赖一致意见或多数投票的古典方法不同，我们利用自主系统具有不同的质量属性（如感知质量）这一事实。我们使用这一标准来确定哪些自主系统是可信的，并借鉴社会认识论的概念来定义聚合和传播规则，用于自动化决策。我们使用二值决策图（BDDs）作为信念聚合和传播的形式化模型，并提出了简化规则以减小BDDs的规模，从而实现协作自动推理的有效计算结构。 

---
# Defining neurosymbolic AI 

**Title (ZH)**: 定义神经符号人工智能 

**Authors**: Lennert De Smet, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2507.11127)  

**Abstract**: Neurosymbolic AI focuses on integrating learning and reasoning, in particular, on unifying logical and neural representations. Despite the existence of an alphabet soup of neurosymbolic AI systems, the field is lacking a generally accepted formal definition of what neurosymbolic models and inference really are. We introduce a formal definition for neurosymbolic AI that makes abstraction of its key ingredients. More specifically, we define neurosymbolic inference as the computation of an integral over a product of a logical and a belief function. We show that our neurosymbolic AI definition makes abstraction of key representative neurosymbolic AI systems. 

**Abstract (ZH)**: 神经符号AI聚焦于学习与推理的整合，特别是逻辑表示与神经表示的统一。尽管存在各种神经符号AI系统，但该领域缺乏对其所指的神经符号模型和推理的一般接受的正式定义。我们提出了一种正式定义神经符号AI，抽象了其关键成分。具体而言，我们定义神经符号推理为逻辑函数与信念函数乘积的积分计算。我们展示了我们的神经符号AI定义抽象了关键的代表性神经符号AI系统。 

---
# AI Agent Architecture for Decentralized Trading of Alternative Assets 

**Title (ZH)**: AI代理架构 for 分布式交易替代资产 

**Authors**: Ailiya Borjigin, Cong He, Charles CC Lee, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.11117)  

**Abstract**: Decentralized trading of real-world alternative assets (e.g., gold) requires bridging physical asset custody with blockchain systems while meeting strict requirements for compliance, liquidity, and risk management. We present GoldMine OS, a research oriented architecture that employs multiple specialized AI agents to automate and secure the tokenization and exchange of physical gold into a blockchain based stablecoin ("OZ"). Our approach combines on chain smart contracts for critical risk controls with off chain AI agents for decision making, blending the transparency and reliability of blockchains with the flexibility of AI driven automation. We describe four cooperative agents (Compliance, Token Issuance, Market Making, and Risk Control) and a coordinating core, and evaluate the system through simulation and a controlled pilot deployment. In experiments the prototype delivers on demand token issuance in under 1.2 s, more than 100 times faster than manual workflows. The Market Making agent maintains tight liquidity with spreads often below 0.5 percent even under volatile conditions. Fault injection tests show resilience: an oracle price spoofing attack is detected and mitigated within 10 s, and a simulated vault mis reporting halts issuance immediately with minimal user impact. The architecture scales to 5000 transactions per second with 10000 concurrent users in benchmarks. These results indicate that an AI agent based decentralized exchange for alternative assets can satisfy rigorous performance and safety requirements. We discuss broader implications for democratizing access to traditionally illiquid assets and explain how our governance model -- multi signature agent updates and on chain community voting on risk parameters -- provides ongoing transparency, adaptability, and formal assurance of system integrity. 

**Abstract (ZH)**: 去中心化的实物替代资产（如黄金）交易需要将物理资产保管与区块链系统相结合，同时满足严格的合规、流动性及风险管理要求。我们提出了GoldMine OS，一种研究导向的架构，采用多个专门的AI代理自动化并安全地将实物黄金Token化为基于区块链的稳定币（OZ）。我们的方法结合了链上智能合约进行关键的风险控制与链下AI代理进行决策，将区块链的透明性和可靠性与AI驱动自动化的优势结合起来。我们描述了四个协作代理（合规性、Token发行、做市与风险管理）和一个协调核心，并通过仿真和受控试点部署评估了系统。实验结果表明，原型可在1.2秒内响应需求发行Token，比手工流程快100多倍。做市代理在波动条件下维持紧密的流动性，价差通常低于0.5％。故障注入测试表明系统的弹性：模拟预言机价格欺骗攻击在10秒内被检测并缓解，模拟金库报告错误立即停止发行，对用户的影响最小。架构在基准测试中可实现每秒5000笔交易、同时支持10000个并发用户。这些结果表明，基于AI代理的去中心化交易平台可以满足严格的性能和安全要求。我们讨论了这种架构对传统流动性差的资产实现民主化访问的更广泛影响，并解释了我们的治理模型——多签名代理更新和链上社区对风险参数的投票——如何提供持续的透明性、适应性和系统完整性的正式验证。 

---
# Personalized Exercise Recommendation with Semantically-Grounded Knowledge Tracing 

**Title (ZH)**: 基于语义引导的知识追踪的个性化锻炼推荐 

**Authors**: Yilmazcan Ozyurt, Tunaberk Almaci, Stefan Feuerriegel, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11060)  

**Abstract**: We introduce ExRec, a general framework for personalized exercise recommendation with semantically-grounded knowledge tracing. Our method builds on the observation that existing exercise recommendation approaches simulate student performance via knowledge tracing (KT) but they often overlook two key aspects: (a) the semantic content of questions and (b) the sequential, structured progression of student learning. To address this, our ExRec presents an end-to-end pipeline, from annotating the KCs of questions and learning their semantic representations to training KT models and optimizing several reinforcement learning (RL) methods. Moreover, we improve standard Q-learning-based continuous RL methods via a tailored model-based value estimation (MVE) approach that directly leverages the components of KT model in estimating cumulative knowledge improvement. We validate the effectiveness of our ExRec using various RL methods across four real-world tasks with different educational goals in online math learning. We further show that ExRec generalizes robustly to new, unseen questions and that it produces interpretable student learning trajectories. Together, our findings highlight the promise of KT-guided RL for effective personalization in education. 

**Abstract (ZH)**: 基于语义指导的知识追踪的个性化锻炼推荐框架：ExRec 

---
# Modeling Habitat Shifts: Integrating Convolutional Neural Networks and Tabular Data for Species Migration Prediction 

**Title (ZH)**: modeling 生态位转变：结合卷积神经网络和表格数据进行物种迁移预测 

**Authors**: Emir Durakovic, Min-Hong Shih  

**Link**: [PDF](https://arxiv.org/pdf/2507.10993)  

**Abstract**: Due to climate-induced changes, many habitats are experiencing range shifts away from their traditional geographic locations (Piguet, 2011). We propose a solution to accurately model whether bird species are present in a specific habitat through the combination of Convolutional Neural Networks (CNNs) (O'Shea, 2015) and tabular data. Our approach makes use of satellite imagery and environmental features (e.g., temperature, precipitation, elevation) to predict bird presence across various climates. The CNN model captures spatial characteristics of landscapes such as forestation, water bodies, and urbanization, whereas the tabular method uses ecological and geographic data. Both systems predict the distribution of birds with an average accuracy of 85%, offering a scalable but reliable method to understand bird migration. 

**Abstract (ZH)**: 由于气候诱导的变化，许多栖息地正经历着范围移迁，远离其传统的地理位置（Piguet, 2011）。我们提出通过卷积神经网络（CNNs）与表格数据的结合来准确建模特定栖息地中鸟类物种的存在性。我们的方法利用卫星图像和环境特征（如温度、降水、海拔）来预测各种气候下的鸟类分布。CNN模型捕获了景观的空间特征，如森林布局、水体和城市化，而表格方法则使用生态和地理数据。两种系统在鸟类分布的预测上平均准确率为85%，提供了一种可扩展且可靠的了解鸟类迁徙的方法。 

---
# Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization 

**Title (ZH)**: 通过知识偏好优化增强安全可控的蛋白质生成 

**Authors**: Yuhao Wang, Keyan Ding, Kehua Feng, Zeyuan Wang, Ming Qin, Xiaotong Li, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10923)  

**Abstract**: Protein language models have emerged as powerful tools for sequence generation, offering substantial advantages in functional optimization and denovo design. However, these models also present significant risks of generating harmful protein sequences, such as those that enhance viral transmissibility or evade immune responses. These concerns underscore critical biosafety and ethical challenges. To address these issues, we propose a Knowledge-guided Preference Optimization (KPO) framework that integrates prior knowledge via a Protein Safety Knowledge Graph. This framework utilizes an efficient graph pruning strategy to identify preferred sequences and employs reinforcement learning to minimize the risk of generating harmful proteins. Experimental results demonstrate that KPO effectively reduces the likelihood of producing hazardous sequences while maintaining high functionality, offering a robust safety assurance framework for applying generative models in biotechnology. 

**Abstract (ZH)**: 蛋白质语言模型已成为序列生成的强大工具，为功能优化和从头设计提供了显著优势。然而，这些模型也存在生成有害蛋白质序列的重大风险，如增强病毒传染性或逃避免疫响应等。这些问题突显了关键的生物安全和伦理挑战。为应对这些挑战，我们提出了一种知识导向的偏好优化（KPO）框架，该框架通过蛋白质安全知识图谱整合先验知识。该框架利用高效的图修剪策略来识别优选序列，并运用强化学习来最小化生成有害蛋白质的风险。实验结果表明，KPO 有效降低了产生危险序列的可能性，同时保持了高功能，为在生物技术中应用生成模型提供了稳健的安全保障框架。 

---
# WhisperKit: On-device Real-time ASR with Billion-Scale Transformers 

**Title (ZH)**: WhisperKit：基于设备的实时ASR与十亿规模变压器 

**Authors**: Atila Orhon, Arda Okan, Berkin Durmus, Zach Nagengast, Eduardo Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2507.10860)  

**Abstract**: Real-time Automatic Speech Recognition (ASR) is a fundamental building block for many commercial applications of ML, including live captioning, dictation, meeting transcriptions, and medical scribes. Accuracy and latency are the most important factors when companies select a system to deploy. We present WhisperKit, an optimized on-device inference system for real-time ASR that significantly outperforms leading cloud-based systems. We benchmark against server-side systems that deploy a diverse set of models, including a frontier model (OpenAI gpt-4o-transcribe), a proprietary model (Deepgram nova-3), and an open-source model (Fireworks large-v3-turbo).Our results show that WhisperKit matches the lowest latency at 0.46s while achieving the highest accuracy 2.2% WER. The optimizations behind the WhisperKit system are described in detail in this paper. 

**Abstract (ZH)**: 实时自动语音识别(ASR)优化系统WhisperKit：显著超越领先云服务系统的装置端推理解决方案 

---
# AF-XRAY: Visual Explanation and Resolution of Ambiguity in Legal Argumentation Frameworks 

**Title (ZH)**: AF-XRAY：法律论证框架中歧义的视觉解释与解析 

**Authors**: Yilin Xia, Heng Zheng, Shawn Bowers, Bertram Ludäscher  

**Link**: [PDF](https://arxiv.org/pdf/2507.10831)  

**Abstract**: Argumentation frameworks (AFs) provide formal approaches for legal reasoning, but identifying sources of ambiguity and explaining argument acceptance remains challenging for non-experts. We present AF-XRAY, an open-source toolkit for exploring, analyzing, and visualizing abstract AFs in legal reasoning. AF-XRAY introduces: (i) layered visualizations based on game-theoretic argument length revealing well-founded derivation structures; (ii) classification of attack edges by semantic roles (primary, secondary, blunders); (iii) overlay visualizations of alternative 2-valued solutions on ambiguous 3-valued grounded semantics; and (iv) identification of critical attack sets whose suspension resolves undecided arguments. Through systematic generation of critical attack sets, AF-XRAY transforms ambiguous scenarios into grounded solutions, enabling users to pinpoint specific causes of ambiguity and explore alternative resolutions. We use real-world legal cases (e.g., Wild Animals as modeled by Bench-Capon) to show that our tool supports teleological legal reasoning by revealing how different assumptions lead to different justified conclusions. 

**Abstract (ZH)**: Argumentation frameworks (AFs)为法律推理提供了形式化的研究方法，但识别模糊源和解释论据接受仍具有挑战性，尤其对于非专家而言。我们提出了AF-XRAY，一个用于探索、分析和可视化抽象法律推理AF的开源工具包。AF-XRAY引入了基于博弈论论据长度的分层可视化，揭示了稳固的演绎结构；通过语义角色（主要攻击、次要攻击、失误）分类攻击边；在模棱两可的三值基础语义上叠加可视化不同的二值解决方案；并识别关键攻击集，其吊销可解决未决论点。通过系统生成关键攻击集，AF-XRAY将模棱两可的情景转化为可靠解决方案，使用户能够明确具体导致模糊的原因并探索替代解决方案。我们使用实际案例（如Bench-Capon模型的野生动植物）来表明，该工具支持目的论法律推理，展示了不同假设如何导致不同的正当结论。 

---
# Uncertainty-Informed Scheduling of Decision Points for Intelligent Mobile Health Interventions 

**Title (ZH)**: 基于不确定性指导的决策点调度方法在智能移动健康干预中的应用 

**Authors**: Asim H. Gazi, Bhanu T. Gullapalli, Daiqi Gao, Benjamin M. Marlin, Vivek Shetty, Susan A. Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2507.10798)  

**Abstract**: Timely decision making is critical to the effectiveness of mobile health (mHealth) interventions. At predefined timepoints called "decision points," intelligent mHealth systems such as just-in-time adaptive interventions (JITAIs) estimate an individual's biobehavioral context from sensor or survey data and determine whether and how to intervene. For interventions targeting habitual behavior (e.g., oral hygiene), effectiveness often hinges on delivering support shortly before the target behavior is likely to occur. Current practice schedules decision points at a fixed interval (e.g., one hour) before user-provided behavior times, and the fixed interval is kept the same for all individuals. However, this one-size-fits-all approach performs poorly for individuals with irregular routines, often scheduling decision points after the target behavior has already occurred, rendering interventions ineffective. In this paper, we propose SigmaScheduling, a method to dynamically schedule decision points based on uncertainty in predicted behavior times. When behavior timing is more predictable, SigmaScheduling schedules decision points closer to the predicted behavior time; when timing is less certain, SigmaScheduling schedules decision points earlier, increasing the likelihood of timely intervention. We evaluated SigmaScheduling using real-world data from 68 participants in a 10-week trial of Oralytics, a JITAI designed to improve daily toothbrushing. SigmaScheduling increased the likelihood that decision points preceded brushing events in at least 70% of cases, preserving opportunities to intervene and impact behavior. Our results indicate that SigmaScheduling can advance precision mHealth, particularly for JITAIs targeting time-sensitive, habitual behaviors such as oral hygiene or dietary habits. 

**Abstract (ZH)**: 及时决策对于移动健康(mHealth)干预的有效性至关重要。基于预定义的时间点称为“决策点”，智能mHealth系统如及时自适应干预(JITAIs)会从传感器数据或调查数据中估算个体的生理行为上下文，并决定是否以及如何干预。对于旨在改变习惯性行为（例如口腔卫生）的干预措施，功效往往取决于在目标行为很可能发生之前不久提供支持。当前的做法是在用户提供的行为时间前固定间隔（例如一小时）安排决策点，并且该固定间隔对所有个体保持一致。然而，这种一刀切的方法对于具有不规则日常活动的个体性能较差，经常导致决策点安排在目标行为已经发生之后，使得干预无效。本文中，我们提出了SigmaScheduling方法，该方法基于预测行为时间的不确定性动态安排决策点。当行为时间可预测性较高时，SigmaScheduling将决策点更接近预测行为时间安排；当时间不确定性较大时，SigmaScheduling则更早地安排决策点，以增加及时干预的可能性。我们使用Oralytics在10周时间内的68名参与者中进行的实际数据评估了SigmaScheduling，Oralytics是一种旨在改善日常刷牙行为的JITAI工具。SigmaScheduling提高了决策点在至少70%的情况下发生在刷牙之前的概率，从而保留了干预和影响行为的机会。我们的结果表明，SigmaScheduling可以推进精准mHealth，特别适用于如口腔卫生或饮食习惯等时间敏感的习惯性行为的JITAIs。 

---
# Detecting AI Assistance in Abstract Complex Tasks 

**Title (ZH)**: 检测AI在复杂任务摘要中的辅助作用 

**Authors**: Tyler King, Nikolos Gurney, John H. Miller, Volkan Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2507.10761)  

**Abstract**: Detecting assistance from artificial intelligence is increasingly important as they become ubiquitous across complex tasks such as text generation, medical diagnosis, and autonomous driving. Aid detection is challenging for humans, especially when looking at abstract task data. Artificial neural networks excel at classification thanks to their ability to quickly learn from and process large amounts of data -- assuming appropriate preprocessing. We posit detecting help from AI as a classification task for such models. Much of the research in this space examines the classification of complex but concrete data classes, such as images. Many AI assistance detection scenarios, however, result in data that is not machine learning-friendly. We demonstrate that common models can effectively classify such data when it is appropriately preprocessed. To do so, we construct four distinct neural network-friendly image formulations along with an additional time-series formulation that explicitly encodes the exploration/exploitation of users, which allows for generalizability to other abstract tasks. We benchmark the quality of each image formulation across three classical deep learning architectures, along with a parallel CNN-RNN architecture that leverages the additional time series to maximize testing performance, showcasing the importance of encoding temporal and spatial quantities for detecting AI aid in abstract tasks. 

**Abstract (ZH)**: 随着人工智能在复杂任务如文本生成、医疗诊断和自主驾驶中变得无处不在，检测人工智能提供的帮助变得日益重要。在面对抽象任务数据时，人工神经网络因其能够快速学习和处理大量数据而适用于分类任务——前提是适当的预处理。我们将检测来自AI的帮助视为这种模型的分类任务。该领域中的许多研究都关注复杂但具体的数据类别的分类，如图像。然而，许多人工智能辅助检测场景会产生对机器学习不友好的数据。我们证明，在适当预处理的情况下，常用模型可以有效分类此类数据。为此，我们构建了四种不同的神经网络友好型图像表示，并额外构造了一个明确编码用户探索/开发过程的时间序列表示，这使模型具有泛化到其他抽象任务的能力。我们使用三种经典深度学习架构以及一个利用额外时间序列以最大化测试性能的并行CNN-RNN架构，对每种图像表示的质量进行了基准测试，强调了在检测抽象任务中的人工智能辅助时编码时间和空间量的重要性。 

---
# IoT Malware Network Traffic Detection using Deep Learning and GraphSAGE Models 

**Title (ZH)**: 基于深度学习和GraphSAGE模型的物联网恶意软件网络流量检测 

**Authors**: Nikesh Prajapati, Bimal Karki, Saroj Gopali, Akbar Siami Namin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10758)  

**Abstract**: This paper intends to detect IoT malicious attacks through deep learning models and demonstrates a comprehensive evaluation of the deep learning and graph-based models regarding malicious network traffic detection. The models particularly are based on GraphSAGE, Bidirectional encoder representations from transformers (BERT), Temporal Convolutional Network (TCN) as well as Multi-Head Attention, together with Bidirectional Long Short-Term Memory (BI-LSTM) Multi-Head Attention and BI-LSTM and LSTM models. The chosen models demonstrated great performance to model temporal patterns and detect feature significance. The observed performance are mainly due to the fact that IoT system traffic patterns are both sequential and diverse, leaving a rich set of temporal patterns for the models to learn. Experimental results showed that BERT maintained the best performance. It achieved 99.94% accuracy rate alongside high precision and recall, F1-score and AUC-ROC score of 99.99% which demonstrates its capabilities through temporal dependency capture. The Multi-Head Attention offered promising results by providing good detection capabilities with interpretable results. On the other side, the Multi-Head Attention model required significant processing time like BI-LSTM variants. The GraphSAGE model achieved good accuracy while requiring the shortest training time but yielded the lowest accuracy, precision, and F1 score compared to the other models 

**Abstract (ZH)**: 通过深度学习模型检测物联网恶意攻击：基于图的模型和序列模型的综合评估 

---
# AI and the Net-Zero Journey: Energy Demand, Emissions, and the Potential for Transition 

**Title (ZH)**: AI与净零转型：能源需求、排放及过渡潜力 

**Authors**: Pandu Devarakota, Nicolas Tsesmetzis, Faruk O. Alpak, Apurva Gala, Detlef Hohl  

**Link**: [PDF](https://arxiv.org/pdf/2507.10750)  

**Abstract**: Thanks to the availability of massive amounts of data, computing resources, and advanced algorithms, AI has entered nearly every sector. This has sparked significant investment and interest, particularly in building data centers with the necessary hardware and software to develop and operate AI models and AI-based workflows. In this technical review article, we present energy consumption scenarios of data centers and impact on GHG emissions, considering both near-term projections (up to 2030) and long-term outlook (2035 and beyond). We address the quintessential question of whether AI will have a net positive, neutral, or negative impact on CO2 emissions by 2035. Additionally, we discuss AI's potential to automate, create efficient and disruptive workflows across various fields related to energy production, supply and consumption. In the near-term scenario, the growing demand for AI will likely strain computing resources, lead to increase in electricity consumption and therefore associated CO2 emissions. This is due to the power-hungry nature of big data centers and the requirements for training and running of large and complex AI models, as well as the penetration of AI assistant search and applications for public use. However, the long-term outlook could be more promising. AI has the potential to be a game-changer in CO2 reduction. Its ability to further automate and optimize processes across industries, from energy production to logistics, could significantly decrease our carbon footprint. This positive impact is anticipated to outweigh the initial emissions bump, creating value for businesses and society in areas where traditional solutions have fallen short. In essence, AI might cause some initial growing pains for the environment, but it has the potential to support climate mitigation efforts. 

**Abstract (ZH)**: 得益于大量数据、计算资源和先进算法的 availability，AI 已几乎进入每一个领域。这引发了显著的投资和兴趣，尤其是在建设必要的硬件和软件以开发和运行 AI 模型及 AI 基础工作流方面。在本文中，我们综合了短期预测（至2030年）和长期展望（2035年及以后），讨论了数据中心的能量消耗情景及其对温室气体排放的影响。我们探讨了到2035年，AI 是否会对二氧化碳排放产生净积极、中性和消极影响。此外，我们还讨论了 AI 在能源生产、供应和消费相关领域的自动化和创造高效颠覆性工作流的潜力。在短期内，AI 需求的增长可能会给计算资源带来压力，导致电力消耗和相关二氧化碳排放的增加。这主要是由于大数据中心耗电量大以及训练和运行大型复杂 AI 模型的需求，以及 AI 辅助搜索和公共应用的普及。然而，长期来看，情况可能更有前景。AI 有可能成为减少二氧化碳排放的决定性因素。其在工业各领域进一步自动化和优化过程的能力，从能源生产到物流，有可能显著降低我们的碳足迹。这一积极影响预计会超过初始排放增加的影响，为在传统解决方案失败的领域创造业务和社会价值。总的来说，AI 可能会短期内对环境造成一些痛苦，但它有潜力支持气候变化缓解努力。 

---
# Parsing Musical Structure to Enable Meaningful Variations 

**Title (ZH)**: 解析音乐结构以实现有意义的变化 

**Authors**: Maziar Kanani, Sean O Leary, James McDermott  

**Link**: [PDF](https://arxiv.org/pdf/2507.10740)  

**Abstract**: This paper presents a novel rule-based approach for generating music by varying existing tunes. We parse each tune to find the Pathway Assembly (PA) [ 1], that is a structure representing all repetitions in the tune. The Sequitur algorithm [2 ] is used for this. The result is a grammar. We then carry out mutation on the grammar, rather than on a tune directly. There are potentially 19 types of mutations such as adding, removing, swapping or reversing parts of the grammar that can be applied to the grammars. The system employs one of the mutations randomly in this step to automatically manipulate the grammar. Following the mutation, we need to expand the grammar which returns a new tune. The output after 1 or more mutations will be a new tune related to the original tune. Our study examines how tunes change gradually over the course of multiple mutations. Edit distances, structural complexity and length of the tunes are used to show how a tune is changed after multiple mutations. In addition, the size of effect of each mutation type is analyzed. As a final point, we review the musical aspect of the output tunes. It should be noted that the study only focused on generating new pitch sequences. The study is based on an Irish traditional tune dataset and a list of integers has been used to represent each tune's pitch values. 

**Abstract (ZH)**: 基于规则的方法生成音乐的新颖途径：通过变化现有曲调实现音乐生成 

---
# SAMEP: A Secure Protocol for Persistent Context Sharing Across AI Agents 

**Title (ZH)**: SAMEP：跨AI代理的持久上下文共享安全协议 

**Authors**: Hari Masoor  

**Link**: [PDF](https://arxiv.org/pdf/2507.10562)  

**Abstract**: Current AI agent architectures suffer from ephemeral memory limitations, preventing effective collaboration and knowledge sharing across sessions and agent boundaries. We introduce SAMEP (Secure Agent Memory Exchange Protocol), a novel framework that enables persistent, secure, and semantically searchable memory sharing among AI agents. Our protocol addresses three critical challenges: (1) persistent context preservation across agent sessions, (2) secure multi-agent collaboration with fine-grained access control, and (3) efficient semantic discovery of relevant historical context. SAMEP implements a distributed memory repository with vector-based semantic search, cryptographic access controls (AES-256-GCM), and standardized APIs compatible with existing agent communication protocols (MCP, A2A). We demonstrate SAMEP's effectiveness across diverse domains including multi-agent software development, healthcare AI with HIPAA compliance, and multi-modal processing pipelines. Experimental results show 73% reduction in redundant computations, 89% improvement in context relevance scores, and complete compliance with regulatory requirements including audit trail generation. SAMEP enables a new paradigm of persistent, collaborative AI agent ecosystems while maintaining security and privacy guarantees. 

**Abstract (ZH)**: 当前的AI代理架构面临着临时内存限制的问题，这妨碍了跨会话和代理边界的有效协作和知识共享。我们提出了SAMEP（Secure Agent Memory Exchange Protocol）这一新颖框架，以实现AI代理之间持久、安全且语义可搜索的记忆共享。我们的协议解决了三个关键挑战：（1）跨代理会话的持久上下文保存，（2）细粒度访问控制下的安全多代理协作，以及（3）高效的语义发现相关历史上下文。SAMEP实现了一个分布式的记忆仓库，支持向量基的语义搜索，加密访问控制（AES-256-GCM）以及与现有代理通信协议（MCP, A2A）兼容的标准API。我们展示了SAMEP在多代理软件开发、符合HIPAA规范的医疗AI以及多模态处理流水线等多个领域的有效性。实验结果显示，冗余计算减少73%，上下文相关性提高89%，并且完全符合审计追踪生成等监管要求。SAMEP开启了持久化协作AI代理生态系统的新型范式，同时保持了安全和隐私保障。 

---
# Recursive Bound-Constrained AdaGrad with Applications to Multilevel and Domain Decomposition Minimization 

**Title (ZH)**: 递归界约束AdaGrad及其在多尺度和域分解最小化中的应用 

**Authors**: Serge Gratton, Alena Kopaničáková, Philippe Toint  

**Link**: [PDF](https://arxiv.org/pdf/2507.11513)  

**Abstract**: Two OFFO (Objective-Function Free Optimization) noise tolerant algorithms are presented that handle bound constraints, inexact gradients and use second-order information when this http URL first is a multi-level method exploiting a hierarchical description of the problem and the second is a domain-decomposition method covering the standard addditive Schwarz decompositions. Both are generalizations of the first-order AdaGrad algorithm for unconstrained optimization. Because these algorithms share a common theoretical framework, a single convergence/complexity theory is provided which covers them both. Its main result is that, with high probability, both methods need at most $O(\epsilon^{-2})$ iterations and noisy gradient evaluations to compute an $\epsilon$-approximate first-order critical point of the bound-constrained problem. Extensive numerical experiments are discussed on applications ranging from PDE-based problems to deep neural network training, illustrating their remarkable computational efficiency. 

**Abstract (ZH)**: 两种适用于边界约束、不精确梯度且利用二阶信息的OBJ-Free优化算法：一种是基于问题分层描述的多级方法，另一种是域分解方法，涵盖标准的加性 Schwarz 分解。这两种方法是无约束优化中一阶 AdaGrad 算法的一般化。由于这些算法共享相同的理论框架，提供了一个同时涵盖两者的一致收敛性和复杂度理论。其主要结果是，这两种方法在高概率下最多需要 $O(\epsilon^{-2})$ 次迭代和梯度评估来计算边界约束问题的 $\epsilon$-近似一阶临界点。讨论了从PDE问题到深度神经网络训练的应用实例，展示了它们卓越的计算效率。 

---
# COLIBRI Fuzzy Model: Color Linguistic-Based Representation and Interpretation 

**Title (ZH)**: COLIBRI 模糊模型：基于颜色语言的表示与解释 

**Authors**: Pakizar Shamoi, Nuray Toganas, Muragul Muratbekova, Elnara Kadyrgali, Adilet Yerkin, Ayan Igali, Malika Ziyada, Ayana Adilova, Aron Karatayev, Yerdauit Torekhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11488)  

**Abstract**: Colors are omnipresent in today's world and play a vital role in how humans perceive and interact with their surroundings. However, it is challenging for computers to imitate human color perception. This paper introduces the Human Perception-Based Fuzzy Color Model, COLIBRI (Color Linguistic-Based Representation and Interpretation), designed to bridge the gap between computational color representations and human visual perception. The proposed model uses fuzzy sets and logic to create a framework for color categorization. Using a three-phase experimental approach, the study first identifies distinguishable color stimuli for hue, saturation, and intensity through preliminary experiments, followed by a large-scale human categorization survey involving more than 1000 human subjects. The resulting data are used to extract fuzzy partitions and generate membership functions that reflect real-world perceptual uncertainty. The model incorporates a mechanism for adaptation that allows refinement based on feedback and contextual changes. Comparative evaluations demonstrate the model's alignment with human perception compared to traditional color models, such as RGB, HSV, and LAB. To the best of our knowledge, no previous research has documented the construction of a model for color attribute specification based on a sample of this size or a comparable sample of the human population (n = 2496). Our findings are significant for fields such as design, artificial intelligence, marketing, and human-computer interaction, where perceptually relevant color representation is critical. 

**Abstract (ZH)**: 基于人类感知的模糊颜色模型COLIBRI：颜色语言学的表征与解释 

---
# COLI: A Hierarchical Efficient Compressor for Large Images 

**Title (ZH)**: COLI：一种高效的分层大型图像压缩器 

**Authors**: Haoran Wang, Hanyu Pei, Yang Lyu, Kai Zhang, Li Li, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11443)  

**Abstract**: The escalating adoption of high-resolution, large-field-of-view imagery amplifies the need for efficient compression methodologies. Conventional techniques frequently fail to preserve critical image details, while data-driven approaches exhibit limited generalizability. Implicit Neural Representations (INRs) present a promising alternative by learning continuous mappings from spatial coordinates to pixel intensities for individual images, thereby storing network weights rather than raw pixels and avoiding the generalization problem. However, INR-based compression of large images faces challenges including slow compression speed and suboptimal compression ratios. To address these limitations, we introduce COLI (Compressor for Large Images), a novel framework leveraging Neural Representations for Videos (NeRV). First, recognizing that INR-based compression constitutes a training process, we accelerate its convergence through a pretraining-finetuning paradigm, mixed-precision training, and reformulation of the sequential loss into a parallelizable objective. Second, capitalizing on INRs' transformation of image storage constraints into weight storage, we implement Hyper-Compression, a novel post-training technique to substantially enhance compression ratios while maintaining minimal output distortion. Evaluations across two medical imaging datasets demonstrate that COLI consistently achieves competitive or superior PSNR and SSIM metrics at significantly reduced bits per pixel (bpp), while accelerating NeRV training by up to 4 times. 

**Abstract (ZH)**: 高分辨率大视野图像采用的激增放大了高效压缩方法的需求。传统的技术往往无法保留关键图像细节，而数据驱动的方法则表现出有限的泛化能力。隐式神经表示（INRs）通过从空间坐标到像素强度学习连续映射为单个图像提供了一种有前景的替代方案，从而存储网络权重而非原始像素，从而避免了泛化问题。然而，基于INR的大图像压缩面临挑战，包括压缩速度慢和压缩比不佳。为了克服这些限制，我们提出了COLI（Compressor for Large Images），一种利用视频神经表示（NeRV）的新型框架。首先，鉴于INR基压缩是一个训练过程，我们通过预训练-微调范式、混合精度训练和将顺序损失重新表述为可并行的目标来加速其收敛。其次，利用INRs将图像存储约束转化为权重存储的特点，我们实现了Hyper-Compression，这是一种新型的后训练技术，能够大幅提高压缩比同时保持最小的输出失真。针对两个医学成像数据集的评估表明，COLI在显著降低每像素比特数（bpp）的同时，实现了竞争力或优越的PSNR和SSIM指标，同时可使NeRV训练加速多达4倍。 

---
# Toward Improving fNIRS Classification: A Study on Activation Functions in Deep Neural Architectures 

**Title (ZH)**: 提高fNIRS分类性能：深度神经架构中激活函数研究 

**Authors**: Behtom Adeli, John McLinden, Pankaj Pandey, Ming Shao, Yalda Shahriari  

**Link**: [PDF](https://arxiv.org/pdf/2507.11436)  

**Abstract**: Activation functions are critical to the performance of deep neural networks, particularly in domains such as functional near-infrared spectroscopy (fNIRS), where nonlinearity, low signal-to-noise ratio (SNR), and signal variability poses significant challenges to model accuracy. However, the impact of activation functions on deep learning (DL) performance in the fNIRS domain remains underexplored and lacks systematic investigation in the current literature. This study evaluates a range of conventional and field-specific activation functions for fNIRS classification tasks using multiple deep learning architectures, including the domain-specific fNIRSNet, AbsoluteNet, MDNN, and shallowConvNet (as the baseline), all tested on a single dataset recorded during an auditory task. To ensure fair a comparison, all networks were trained and tested using standardized preprocessing and consistent training parameters. The results show that symmetrical activation functions such as Tanh and the Absolute value function Abs(x) can outperform commonly used functions like the Rectified Linear Unit (ReLU), depending on the architecture. Additionally, a focused analysis of the role of symmetry was conducted using a Modified Absolute Function (MAF), with results further supporting the effectiveness of symmetrical activation functions on performance gains. These findings underscore the importance of selecting proper activation functions that align with the signal characteristics of fNIRS data. 

**Abstract (ZH)**: 激活函数对功能性近红外光谱成像领域深度学习性能的影响研究 

---
# From Kinetic Theory to AI: a Rediscovery of High-Dimensional Divergences and Their Properties 

**Title (ZH)**: 从动力学理论到人工智能：高维 divergences 的再发现及其性质 

**Authors**: Gennaro Auricchio, Giovanni Brigati, Paolo Giudici, Giuseppe Toscani  

**Link**: [PDF](https://arxiv.org/pdf/2507.11387)  

**Abstract**: Selecting an appropriate divergence measure is a critical aspect of machine learning, as it directly impacts model performance. Among the most widely used, we find the Kullback-Leibler (KL) divergence, originally introduced in kinetic theory as a measure of relative entropy between probability distributions. Just as in machine learning, the ability to quantify the proximity of probability distributions plays a central role in kinetic theory. In this paper, we present a comparative review of divergence measures rooted in kinetic theory, highlighting their theoretical foundations and exploring their potential applications in machine learning and artificial intelligence. 

**Abstract (ZH)**: 在机器学习中选择适当的离散度测度是一项关键任务，因为它直接影响模型性能。在最为常用的离散度测度中，有克里格-莱布利（KL）离散度，最初在动能理论中作为概率分布相对熵的度量被引入。就像在机器学习中一样，量化概率分布的接近程度在动能理论中起着核心作用。在本文中，我们对源自动能理论的离散度测度进行了比较性的综述，强调其理论基础并探索其在机器学习和人工智能中的潜在应用。 

---
# Local Pairwise Distance Matching for Backpropagation-Free Reinforcement Learning 

**Title (ZH)**: 局部成对距离匹配的无需反向传播强化学习 

**Authors**: Daniel Tanneberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.11367)  

**Abstract**: Training neural networks with reinforcement learning (RL) typically relies on backpropagation (BP), necessitating storage of activations from the forward pass for subsequent backward updates. Furthermore, backpropagating error signals through multiple layers often leads to vanishing or exploding gradients, which can degrade learning performance and stability. We propose a novel approach that trains each layer of the neural network using local signals during the forward pass in RL settings. Our approach introduces local, layer-wise losses leveraging the principle of matching pairwise distances from multi-dimensional scaling, enhanced with optional reward-driven guidance. This method allows each hidden layer to be trained using local signals computed during forward propagation, thus eliminating the need for backward passes and storing intermediate activations. Our experiments, conducted with policy gradient methods across common RL benchmarks, demonstrate that this backpropagation-free method achieves competitive performance compared to their classical BP-based counterpart. Additionally, the proposed method enhances stability and consistency within and across runs, and improves performance especially in challenging environments. 

**Abstract (ZH)**: 使用强化学习训练神经网络：无需反向传播的方法 

---
# SystolicAttention: Fusing FlashAttention within a Single Systolic Array 

**Title (ZH)**: systolicAttention: 将FlashAttention融合到单个 systolic阵列中 

**Authors**: Jiawei Lin, Guokai Chen, Yuanlong Li, Thomas Bourgeat  

**Link**: [PDF](https://arxiv.org/pdf/2507.11331)  

**Abstract**: Transformer models rely heavily on scaled dot-product attention (SDPA), typically implemented using the FlashAttention algorithm. However, current systolic-array-based accelerators face significant challenges when executing FlashAttention. Systolic arrays can only achieve high utilization for consecutive and large matrix multiplications. In contrast, FlashAttention requires frequently interleaved matrix multiplications and softmax operations.
The frequent data swaps between the systolic array and external vector units result in low systolic array utilization. This is further exacerbated by the fact that softmax involves numerous non-matrix operations, which are not well-suited for systolic arrays. Moreover, the concurrent execution of matrix multiplication on systolic arrays and softmax on vector units leads to register file and SRAM port contention, further degrading performance.
To overcome these limitations, we propose FSA, an enhanced systolic array architecture that enables the entire FlashAttention algorithm to run entirely within a single systolic array, eliminating the need for external vector units. At the core of FSA is SystolicAttention, a novel scheduling algorithm that maps FlashAttention operations onto systolic arrays with fine-grained, element-wise overlap. This significantly improves array utilization while preserving the original floating-point operation order to maintain numerical stability.
We implement FSA in synthesizable RTL and evaluate its performance against state-of-the-art commercial accelerators. Our results show that FSA achieves 1.77x and 4.83x higher attention FLOPs/s utilization compared to AWS NeuronCore-v2 and Google TPUv5e, respectively, with only about 10% area overhead. 

**Abstract (ZH)**: Transformers模型 heavily依赖标量点积注意力（SDPA），通常使用FlashAttention算法实现。然而，当前的 systolic-array 基础加速器在执行FlashAttention时面临重大挑战。systolic阵列只能在连续和大规模矩阵乘法中达到高利用率。相比之下，FlashAttention需要频繁地交替进行矩阵乘法和softmax操作。systolic阵列与外部向量单元之间的频繁数据交换导致阵列利用率低下。此外，softmax操作包含大量的非矩阵运算，这并不适合systolic阵列。进一步地，systolic阵列中并行执行矩阵乘法和向量单元中的softmax操作会导致寄存器文件和SRAM端口争用，进一步降低性能。

为了克服这些限制，我们提出了一种增强的systolic阵列架构FSA，使得整个FlashAttention算法能够在单个systolic阵列中完全运行，从而省去了外部向量单元的需求。FSA的核心是SystolicAttention，这是一种新颖的调度算法，能够以细粒度、元素级的方式将FlashAttention操作映射到systolic阵列上。这显著提高了阵列利用率，同时保持原始的浮点运算顺序以确保数值稳定性。

我们在综合RTL中实现了FSA，并将其性能与最新的商业加速器进行了评估。结果显示，与AWS NeuronCore-v2和Google TPUv5e相比，FSA分别实现了约1.77倍和4.83倍更高的注意力FLOPs/s利用率，仅增加了约10%的面积开销。 

---
# Quantitative multi-metabolite imaging of Parkinson's disease using AI boosted molecular MRI 

**Title (ZH)**: 使用AI增强分子磁共振成像的帕金森病多代谢物定量成像 

**Authors**: Hagar Shmuely, Michal Rivlin, Or Perlman  

**Link**: [PDF](https://arxiv.org/pdf/2507.11329)  

**Abstract**: Traditional approaches for molecular imaging of Parkinson's disease (PD) in vivo require radioactive isotopes, lengthy scan times, or deliver only low spatial resolution. Recent advances in saturation transfer-based PD magnetic resonance imaging (MRI) have provided biochemical insights, although the image contrast is semi-quantitative and nonspecific. Here, we combined a rapid molecular MRI acquisition paradigm with deep learning based reconstruction for multi-metabolite quantification of glutamate, mobile proteins, semisolid, and mobile macromolecules in an acute MPTP (1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine) mouse model. The quantitative parameter maps are in general agreement with the histology and MR spectroscopy, and demonstrate that semisolid magnetization transfer (MT), amide, and aliphatic relayed nuclear Overhauser effect (rNOE) proton volume fractions may serve as PD biomarkers. 

**Abstract (ZH)**: 传统体内帕金森病分子成像方法需要放射性同位素、长扫描时间或仅提供低空间分辨率。最近基于饱和转移的帕金森病磁共振成像（MRI）取得了进展，提供了生化见解，尽管图像对比度是半定量且非特异性的。我们结合了快速分子MRI采集范式并与基于深度学习的重建相结合，对MPTP急性小鼠模型中的谷氨酸、移动蛋白、半固体和移动大分子进行了多代谢物定量分析。定量参数图与组织学和磁共振波谱分析基本一致，并表明半固体交换转移（MT）、酰胺和支链脂肪酸相关核Overhauser效应（rNOE）氢体积分数可能作为帕金森病生物标志物。 

---
# Turning Sand to Gold: Recycling Data to Bridge On-Policy and Off-Policy Learning via Causal Bound 

**Title (ZH)**: 将沙变 gold：通过因果界线回收数据以桥接策略内学习与策略外学习 

**Authors**: Tal Fiskus, Uri Shaham  

**Link**: [PDF](https://arxiv.org/pdf/2507.11269)  

**Abstract**: Deep reinforcement learning (DRL) agents excel in solving complex decision-making tasks across various domains. However, they often require a substantial number of training steps and a vast experience replay buffer, leading to significant computational and resource demands. To address these challenges, we introduce a novel theoretical result that leverages the Neyman-Rubin potential outcomes framework into DRL. Unlike most methods that focus on bounding the counterfactual loss, we establish a causal bound on the factual loss, which is analogous to the on-policy loss in DRL. This bound is computed by storing past value network outputs in the experience replay buffer, effectively utilizing data that is usually discarded. Extensive experiments across the Atari 2600 and MuJoCo domains on various agents, such as DQN and SAC, achieve up to 2,427% higher reward ratio, outperforming the same agents without our proposed term, and reducing the experience replay buffer size by up to 96%, significantly improving sample efficiency at negligible cost. 

**Abstract (ZH)**: 深度强化学习（DRL）智能体在解决各类复杂决策任务方面表现出色。然而，它们常常需要大量的训练步骤和庞大的经验重放缓冲区，导致显著的计算和资源需求。为应对这些挑战，我们引入了一种新的理论成果，将Neyman-Rubin潜在结果框架应用于DRL。与大多数方法专注于界定制外事实损失不同，我们建立了界定制内事实损失，这类似于DRL中的随策策略损失。该界通过在经验重放缓冲区中存储过去的值网络输出来计算，有效地利用了通常会被丢弃的数据。在Atari 2600和MuJoCo领域的广泛实验中，对各种智能体（如DQN和SAC）进行的实验显示出高达2,427%的奖励比率提升，优于未引入我们提议项的相同智能体，并将经验重放缓冲区大小减少高达96%，显著提高了样本效率，成本几乎可以忽略不计。 

---
# An Explainable AI-Enhanced Machine Learning Approach for Cardiovascular Disease Detection and Risk Assessment 

**Title (ZH)**: 可解释的人工智能增强机器学习方法在心血管疾病检测与风险评估中的应用 

**Authors**: Md. Emon Akter Sourov, Md. Sabbir Hossen, Pabon Shaha, Mohammad Minoar Hossain, Md Sadiq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2507.11185)  

**Abstract**: Heart disease remains a major global health concern, particularly in regions with limited access to medical resources and diagnostic facilities. Traditional diagnostic methods often fail to accurately identify and manage heart disease risks, leading to adverse outcomes. Machine learning has the potential to significantly enhance the accuracy, efficiency, and speed of heart disease diagnosis. In this study, we proposed a comprehensive framework that combines classification models for heart disease detection and regression models for risk prediction. We employed the Heart Disease dataset, which comprises 1,035 cases. To address the issue of class imbalance, the Synthetic Minority Oversampling Technique (SMOTE) was applied, resulting in the generation of an additional 100,000 synthetic data points. Performance metrics, including accuracy, precision, recall, F1-score, R2, MSE, RMSE, and MAE, were used to evaluate the model's effectiveness. Among the classification models, Random Forest emerged as the standout performer, achieving an accuracy of 97.2% on real data and 97.6% on synthetic data. For regression tasks, Linear Regression demonstrated the highest R2 values of 0.992 and 0.984 on real and synthetic datasets, respectively, with the lowest error metrics. Additionally, Explainable AI techniques were employed to enhance the interpretability of the models. This study highlights the potential of machine learning to revolutionize heart disease diagnosis and risk prediction, thereby facilitating early intervention and enhancing clinical decision-making. 

**Abstract (ZH)**: 心脏疾病仍然是全球健康的重大关切，特别是在医疗资源和诊断设施有限的地区。传统诊断方法往往无法准确识别和管理心脏疾病风险，导致不良结果。机器学习有可能显著提高心脏疾病诊断的准确性、效率和速度。在本研究中，我们提出了一种综合框架，结合分类模型进行心脏疾病检测和回归模型进行风险预测。我们采用了包含1,035例病例的心脏疾病数据集，并通过合成少数类过采样技术（SMOTE）解决了类别不平衡问题，生成了额外的100,000个合成数据点。使用准确率、精确率、召回率、F1分数、R2、均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）等性能指标来评估模型的有效性。在分类模型中，随机森林表现突出，在实际数据和合成数据上的准确率分别为97.2%和97.6%。对于回归任务，线性回归在实际和合成数据集上的R2值分别为0.992和0.984，具有最低的误差指标。此外，我们还采用了可解释的AI技术以增强模型的可解释性。本研究强调了机器学习在革新心脏疾病诊断和风险预测方面的潜力，从而促进早期干预并增强临床决策。 

---
# Gradient Regularization-based Neural Granger Causality 

**Title (ZH)**: 基于梯度正则化的神经格朗日因果关系 

**Authors**: Meiliang Liu, Huiwen Dong, Xiaoxiao Yang, Yunfang Xu, Zijin Li, Zhengye Si, Xinyue Yang, Zhiwen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11178)  

**Abstract**: With the advancement of deep learning technologies, various neural network-based Granger causality models have been proposed. Although these models have demonstrated notable improvements, several limitations remain. Most existing approaches adopt the component-wise architecture, necessitating the construction of a separate model for each time series, which results in substantial computational costs. In addition, imposing the sparsity-inducing penalty on the first-layer weights of the neural network to extract causal relationships weakens the model's ability to capture complex interactions. To address these limitations, we propose Gradient Regularization-based Neural Granger Causality (GRNGC), which requires only one time series prediction model and applies $L_{1}$ regularization to the gradient between model's input and output to infer Granger causality. Moreover, GRNGC is not tied to a specific time series forecasting model and can be implemented with diverse architectures such as KAN, MLP, and LSTM, offering enhanced flexibility. Numerical simulations on DREAM, Lorenz-96, fMRI BOLD, and CausalTime show that GRNGC outperforms existing baselines and significantly reduces computational overhead. Meanwhile, experiments on real-world DNA, Yeast, HeLa, and bladder urothelial carcinoma datasets further validate the model's effectiveness in reconstructing gene regulatory networks. 

**Abstract (ZH)**: 基于梯度正则化的神经格兰杰因果模型（GRNGC） 

---
# Improving Wi-Fi Network Performance Prediction with Deep Learning Models 

**Title (ZH)**: 使用深度学习模型提高Wi-Fi网络性能预测 

**Authors**: Gabriele Formis, Amanda Ericson, Stefan Forsstrom, Kyi Thar, Gianluca Cena, Stefano Scanzio  

**Link**: [PDF](https://arxiv.org/pdf/2507.11168)  

**Abstract**: The increasing need for robustness, reliability, and determinism in wireless networks for industrial and mission-critical applications is the driver for the growth of new innovative methods. The study presented in this work makes use of machine learning techniques to predict channel quality in a Wi-Fi network in terms of the frame delivery ratio. Predictions can be used proactively to adjust communication parameters at runtime and optimize network operations for industrial applications. Methods including convolutional neural networks and long short-term memory were analyzed on datasets acquired from a real Wi-Fi setup across multiple channels. The models were compared in terms of prediction accuracy and computational complexity. Results show that the frame delivery ratio can be reliably predicted, and convolutional neural networks, although slightly less effective than other models, are more efficient in terms of CPU usage and memory consumption. This enhances the model's usability on embedded and industrial systems. 

**Abstract (ZH)**: 无线网络中工业和关键任务应用对可靠性和确定性的日益增长需求推动了新创新方法的发展：基于卷积神经网络和长短期记忆的Wi-Fi信道质量预测研究 

---
# Latent Space Consistency for Sparse-View CT Reconstruction 

**Title (ZH)**: 潜在空间一致性在稀少量子CT重建中的应用 

**Authors**: Duoyou Chen, Yunqing Chen, Can Zhang, Zhou Wang, Cheng Chen, Ruoxiu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11152)  

**Abstract**: Computed Tomography (CT) is a widely utilized imaging modality in clinical settings. Using densely acquired rotational X-ray arrays, CT can capture 3D spatial features. However, it is confronted with challenged such as significant time consumption and high radiation exposure. CT reconstruction methods based on sparse-view X-ray images have garnered substantial attention from researchers as they present a means to mitigate costs and risks. In recent years, diffusion models, particularly the Latent Diffusion Model (LDM), have demonstrated promising potential in the domain of 3D CT reconstruction. Nonetheless, due to the substantial differences between the 2D latent representation of X-ray modalities and the 3D latent representation of CT modalities, the vanilla LDM is incapable of achieving effective alignment within the latent space. To address this issue, we propose the Consistent Latent Space Diffusion Model (CLS-DM), which incorporates cross-modal feature contrastive learning to efficiently extract latent 3D information from 2D X-ray images and achieve latent space alignment between modalities. Experimental results indicate that CLS-DM outperforms classical and state-of-the-art generative models in terms of standard voxel-level metrics (PSNR, SSIM) on the LIDC-IDRI and CTSpine1K datasets. This methodology not only aids in enhancing the effectiveness and economic viability of sparse X-ray reconstructed CT but can also be generalized to other cross-modal transformation tasks, such as text-to-image synthesis. We have made our code publicly available at this https URL to facilitate further research and applications in other domains. 

**Abstract (ZH)**: 基于一致性潜空间扩散模型的稀视角CT重建 

---
# EditGen: Harnessing Cross-Attention Control for Instruction-Based Auto-Regressive Audio Editing 

**Title (ZH)**: EditGen: 利用跨注意力控制实现基于指令的自回归音频编辑 

**Authors**: Vassilis Sioros, Alexandros Potamianos, Giorgos Paraskevopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.11096)  

**Abstract**: In this study, we investigate leveraging cross-attention control for efficient audio editing within auto-regressive models. Inspired by image editing methodologies, we develop a Prompt-to-Prompt-like approach that guides edits through cross and self-attention mechanisms. Integrating a diffusion-based strategy, influenced by Auffusion, we extend the model's functionality to support refinement edits, establishing a baseline for prompt-guided audio editing. Additionally, we introduce an alternative approach by incorporating MUSICGEN, a pre-trained frozen auto-regressive model, and propose three editing mechanisms, based on Replacement, Reweighting, and Refinement of the attention scores. We employ commonly-used music-specific evaluation metrics and a human study, to gauge time-varying controllability, adherence to global text cues, and overall audio realism. The automatic and human evaluations indicate that the proposed combination of prompt-to-prompt guidance with autoregressive generation models significantly outperforms the diffusion-based baseline in terms of melody, dynamics, and tempo of the generated audio. Our code is available at this https URL 

**Abstract (ZH)**: 本研究探讨了在自回归模型中利用跨注意力控制进行高效音频编辑的方法。受图像编辑方法的启发，我们开发了一种类似于Prompt-to-Prompt的方法，通过跨注意力和自我注意力机制引导编辑。结合Auffusion的扩散策略，我们扩展了模型的功能，支持细化编辑，并建立了基于提示引导音频编辑的基本模型。此外，我们引入了一种替代方法，通过引入MUSICGEN（一个预先训练并冻结的自回归模型），并提出了基于替换、重赋权重和注意力分数细化三种编辑机制。我们使用常用的音乐特定评估指标和人类研究，来评估时间变化的可控性、对全局文本提示的遵守程度以及整体音频的真实性。自动和人工评估表明，所提出的提示到提示引导与自回归生成模型的结合，显著优于基于扩散的基本模型，在生成音频的旋律、动态和节拍方面表现更优。代码可在以下链接获取。 

---
# Standards-Compliant DM-RS Allocation via Temporal Channel Prediction for Massive MIMO Systems 

**Title (ZH)**: 符合标准的DM-RS分配方法：基于时域信道预测的大规模MIMO系统 

**Authors**: Sehyun Ryu, Hyun Jong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11064)  

**Abstract**: Reducing feedback overhead in beyond 5G networks is a critical challenge, as the growing number of antennas in modern massive MIMO systems substantially increases the channel state information (CSI) feedback demand in frequency division duplex (FDD) systems. To address this, extensive research has focused on CSI compression and prediction, with neural network-based approaches gaining momentum and being considered for integration into the 3GPP 5G-Advanced standards. While deep learning has been effectively applied to CSI-limited beamforming and handover optimization, reference signal allocation under such constraints remains surprisingly underexplored. To fill this gap, we introduce the concept of channel prediction-based reference signal allocation (CPRS), which jointly optimizes channel prediction and DM-RS allocation to improve data throughput without requiring CSI feedback. We further propose a standards-compliant ViViT/CNN-based architecture that implements CPRS by treating evolving CSI matrices as sequential image-like data, enabling efficient and adaptive transmission in dynamic environments. Simulation results using ray-tracing channel data generated in NVIDIA Sionna validate the proposed method, showing up to 36.60% throughput improvement over benchmark strategies. 

**Abstract (ZH)**: 减小超5G网络中的反馈开销是一个关键挑战，随着现代大规模MIMO系统中天线数量的增加，频率分割双工（FDD）系统中的信道状态信息（CSI）反馈需求显著增加。为解决这一问题，大量研究集中于CSI压缩和预测，基于神经网络的方法日益受到关注，并被认为有望整合到3GPP 5G-Advanced标准中。尽管深度学习已被有效应用于CSI受限的波束形成和切换优化，但在此类约束下的参考信号分配仍然惊讶地未被充分探索。为填补这一空白，我们提出了基于信道预测的参考信号分配（CPRS）的概念，该方法联合优化信道预测和DM-RS分配，以提高数据吞吐量，而无需CSI反馈。我们进一步提出了一种符合标准的基于ViViT/CNN的架构，通过将 evolving CSI矩阵视为序贯图像-like数据来实现CPRS，从而在动态环境中实现高效且自适应的传输。使用NVIDIA Sionna生成的射线跟踪通道数据进行的仿真结果验证了所提出的方法，显示吞吐量提高了36.60%，优于基准策略。 

---
# GATE: Graph Attention Neural Networks with Real-Time Edge Construction for Robust Indoor Localization using Mobile Embedded Devices 

**Title (ZH)**: GATE：基于实时边构建的图注意力神经网络在移动嵌入式设备上实现 robust 室内定位 

**Authors**: Danish Gufran, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2507.11053)  

**Abstract**: Accurate indoor localization is crucial for enabling spatial context in smart environments and navigation systems. Wi-Fi Received Signal Strength (RSS) fingerprinting is a widely used indoor localization approach due to its compatibility with mobile embedded devices. Deep Learning (DL) models improve accuracy in localization tasks by learning RSS variations across locations, but they assume fingerprint vectors exist in a Euclidean space, failing to incorporate spatial relationships and the non-uniform distribution of real-world RSS noise. This results in poor generalization across heterogeneous mobile devices, where variations in hardware and signal processing distort RSS readings. Graph Neural Networks (GNNs) can improve upon conventional DL models by encoding indoor locations as nodes and modeling their spatial and signal relationships as edges. However, GNNs struggle with non-Euclidean noise distributions and suffer from the GNN blind spot problem, leading to degraded accuracy in environments with dense access points (APs). To address these challenges, we propose GATE, a novel framework that constructs an adaptive graph representation of fingerprint vectors while preserving an indoor state-space topology, modeling the non-Euclidean structure of RSS noise to mitigate environmental noise and address device heterogeneity. GATE introduces 1) a novel Attention Hyperspace Vector (AHV) for enhanced message passing, 2) a novel Multi-Dimensional Hyperspace Vector (MDHV) to mitigate the GNN blind spot, and 3) an new Real-Time Edge Construction (RTEC) approach for dynamic graph adaptation. Extensive real-world evaluations across multiple indoor spaces with varying path lengths, AP densities, and heterogeneous devices demonstrate that GATE achieves 1.6x to 4.72x lower mean localization errors and 1.85x to 4.57x lower worst-case errors compared to state-of-the-art indoor localization frameworks. 

**Abstract (ZH)**: 准确的室内定位对于在智能环境和导航系统中启用空间上下文至关重要。基于Wi-Fi接收信号强度（RSS）指纹识别的室内定位方法因其与移动嵌入式设备的兼容性而广泛使用。深度学习（DL）模型通过学习不同位置的RSS变化来提高定位精度，但它们假设指纹向量存在于欧几里得空间中，未能纳入空间关系和现实世界RSS噪声的非均匀分布。这导致在不同种类的移动设备上泛化能力较差，因为硬件和信号处理的变化会扭曲RSS读数。图神经网络（GNNs）可以通过将室内位置编码为节点，并建模它们的空间和信号关系来增强传统的DL模型。然而，GNNs在处理非欧几里得噪声分布时存在困难，并且遭受GNN盲点问题的影响，在密集访问点（APs）的环境中会导致精度下降。为了解决这些挑战，我们提出了一种名为GATE的新框架，该框架构建了指纹向量的自适应图表示，同时保留了室内的状态空间拓扑，并建模RSS噪声的非欧几里得结构以减轻环境噪声并解决设备异质性问题。GATE引入了1）一种新的注意超空间向量（AHV）以增强信息传递，2）一种新的多维超空间向量（MDHV）以减轻GNN盲点问题，并3）一种新的实时边构造（RTEC）方法以实现动态图适应。在多个室内空间中，通过对不同路径长度、AP密度和异质设备的广泛现实世界评估表明，与现有的室内部定位框架相比，GATE实现了1.6到4.72倍更低的平均定位误差和1.85到4.57倍更低的最坏情况误差。 

---
# Semantically Informed Salient Regions Guided Radiology Report Generation 

**Title (ZH)**: 语义驱动的突出区域指导的放射学报告生成 

**Authors**: Zeyi Hou, Zeqiang Wei, Ruixin Yan, Ning Lang, Xiuzhuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.11015)  

**Abstract**: Recent advances in automated radiology report generation from chest X-rays using deep learning algorithms have the potential to significantly reduce the arduous workload of radiologists. However, due to the inherent massive data bias in radiology images, where abnormalities are typically subtle and sparsely distributed, existing methods often produce fluent yet medically inaccurate reports, limiting their applicability in clinical practice. To address this issue effectively, we propose a Semantically Informed Salient Regions-guided (SISRNet) report generation method. Specifically, our approach explicitly identifies salient regions with medically critical characteristics using fine-grained cross-modal semantics. Then, SISRNet systematically focuses on these high-information regions during both image modeling and report generation, effectively capturing subtle abnormal findings, mitigating the negative impact of data bias, and ultimately generating clinically accurate reports. Compared to its peers, SISRNet demonstrates superior performance on widely used IU-Xray and MIMIC-CXR datasets. 

**Abstract (ZH)**: 近年来，利用深度学习算法从胸部X光片自动生成放射学报告的进展，有可能显著减轻放射科医生的繁重工作负担。然而，由于放射学图像中固有的大量数据偏差，其中异常通常极为细微且分布稀疏，现有方法往往会生成流畅但医学上不准确的报告，限制了其在临床实践中的应用。为有效解决这一问题，我们提出了一种基于语义引导的重要区域（SISRNet）报告生成方法。具体而言，我们的方法通过细粒度的跨模态语义明确识别具有医学关键特征的重要区域。然后，SISRNet在图像建模和报告生成过程中系统地关注这些信息丰富区域，有效捕捉细微的异常发现，减轻数据偏差的负面影响，并最终生成临床准确的报告。与同类方法相比，SISRNet在广泛使用的IU-Xray和MIMIC-CXR数据集中展示了更优的性能。 

---
# Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data 

**Title (ZH)**: 在流形上的不可感知 adversarial 攻击构建方法：针对表格数据 

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2507.10998)  

**Abstract**: Adversarial attacks on tabular data present fundamental challenges distinct from image or text domains due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions, making them detectable. We propose a latent space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate imperceptible adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We specify In-Distribution Success Rate (IDSR) to measure the proportion of adversarial examples that remain statistically indistinguishable from the input distribution. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches. Our comprehensive analysis includes hyperparameter sensitivity, sparsity control mechanisms, and generative architectural comparisons, revealing that VAE-based attacks depend critically on reconstruction quality but offer superior practical utility when sufficient training data is available. This work highlights the importance of on-manifold perturbations for realistic adversarial attacks on tabular data, offering a robust approach for practical deployment. The source code can be accessed through this https URL. 

**Abstract (ZH)**: 针对表格数据的对抗攻击由于混合的类别和数值特征的异质性质，提出了不同于图像或文本域的基本挑战。我们提出了一种使用混合输入变分自编码器（VAE）的潜在空间扰动框架，以生成不可感知的对抗样本。所提出的VAE将类别嵌入和数值特征整合到统一的潜在流形中，使扰动能够保持统计一致性。我们通过In-Distribution Success Rate（IDSR）衡量对抗样本在统计上与输入分布差异不大的比例。在六个公开数据集和三种模型架构上的评估表明，与传统输入空间攻击和其他源自图像域方法的VAE基方法相比，我们的方法实现了显著更低的异常值率和更一致的性能。我们的全面分析包括超参数敏感性、稀疏性控制机制和生成架构比较，揭示了基于VAE的攻击在有足够的训练数据时对重构质量的依赖性，但提供了优于其他方法的实用优势。这项工作强调了在表格数据上进行现实对抗攻击时沿流形扰动的重要性，提供了一种适用于实际部署的稳健方法。源代码可通过以下链接访问：这个https URL。 

---
# Misalignment from Treating Means as Ends 

**Title (ZH)**: 将均值视为最终目标的偏移 

**Authors**: Henrik Marklund, Alex Infanger, Benjamin Van Roy  

**Link**: [PDF](https://arxiv.org/pdf/2507.10995)  

**Abstract**: Reward functions, learned or manually specified, are rarely perfect. Instead of accurately expressing human goals, these reward functions are often distorted by human beliefs about how best to achieve those goals. Specifically, these reward functions often express a combination of the human's terminal goals -- those which are ends in themselves -- and the human's instrumental goals -- those which are means to an end. We formulate a simple example in which even slight conflation of instrumental and terminal goals results in severe misalignment: optimizing the misspecified reward function results in poor performance when measured by the true reward function. This example distills the essential properties of environments that make reinforcement learning highly sensitive to conflation of instrumental and terminal goals. We discuss how this issue can arise with a common approach to reward learning and how it can manifest in real environments. 

**Abstract (ZH)**: 奖励函数，无论是学习得到的还是人工指定的，往往并不完美。这些奖励函数通常受到人类如何最好地实现目标的信念扭曲，而不是准确表达人类的目标。具体来说，这些奖励函数往往包含了人类的终端目标——这些目标本身即为目的，以及工具性目标——这些目标是达到其他目标的手段。我们提出一个简单的例子，即使轻微地混同工具性目标和终端目标也会导致严重的不一致性：优化这个错指定的奖励函数会导致在真正的奖励函数下表现不佳。该例子提炼了使强化学习对混同工具性目标和终端目标高度敏感的环境的基本特性。我们讨论这种问题如何通过一种常见的奖励学习方法出现，以及它在真实环境中的可能表现形式。 

---
# High-Throughput Distributed Reinforcement Learning via Adaptive Policy Synchronization 

**Title (ZH)**: 高 throughput 分布式强化学习通过自适应策略同步 

**Authors**: Rodney Lafuente-Mercado  

**Link**: [PDF](https://arxiv.org/pdf/2507.10990)  

**Abstract**: Scaling reinforcement learning (RL) workloads often requires distributing environment simulation across compute clusters. Existing frameworks entangle simulation, learning logic, and orchestration into monolithic systems, limiting modularity and reusability. We present ClusterEnv, a lightweight, learner-agnostic interface for distributed environment execution that mirrors the Gymnasium API. ClusterEnv introduces the DETACH pattern, which decouples simulation from training by offloading reset() and step() operations to remote workers while keeping learning centralized. To address policy staleness in distributed execution, we propose Adaptive Actor Policy Synchronization (AAPS), a divergence-triggered update mechanism that reduces synchronization overhead without sacrificing performance. ClusterEnv integrates cleanly into existing RL pipelines, supports both on-policy and off-policy methods, and requires minimal code changes. Experiments on discrete control tasks demonstrate that AAPS achieves high sample efficiency with significantly fewer weight updates. Source code is available at this https URL. 

**Abstract (ZH)**: 分布式执行环境的轻量级、学习器无关接口：ClusterEnv及其在强化学习工作负载扩展中的应用 

---
# Pronunciation Deviation Analysis Through Voice Cloning and Acoustic Comparison 

**Title (ZH)**: 通过声音克隆和声学对比进行发音偏差分析 

**Authors**: Andrew Valdivia, Yueming Zhang, Hailu Xu, Amir Ghasemkhani, Xin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10985)  

**Abstract**: This paper presents a novel approach for detecting mispronunciations by analyzing deviations between a user's original speech and their voice-cloned counterpart with corrected pronunciation. We hypothesize that regions with maximal acoustic deviation between the original and cloned utterances indicate potential mispronunciations. Our method leverages recent advances in voice cloning to generate a synthetic version of the user's voice with proper pronunciation, then performs frame-by-frame comparisons to identify problematic segments. Experimental results demonstrate the effectiveness of this approach in pinpointing specific pronunciation errors without requiring predefined phonetic rules or extensive training data for each target language. 

**Abstract (ZH)**: 本文提出了一种通过分析用户原始发音与其带有正确发音的克隆语音之间的偏差来检测误发音的新方法。我们假设原始发音与克隆发音之间声学偏差最大的区域可能表明潜在的误发音。该方法利用近期语音克隆技术生成用户带有正确发音的合成语音版本，然后进行帧对帧比较以识别问题段落。实验结果证明了该方法在无需预定义 Phonetic 规则或针对每个目标语言进行大量训练数据的情况下，能够精准定位特定的发音错误。 

---
# Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures 

**Title (ZH)**: 生物处理单元：利用昆虫连接组先行探索生物仿真的神经架构 

**Authors**: Siyu Yu, Zihan Qin, Tingshan Liu, Beiya Xu, R. Jacob Vogelstein, Jason Brown, Joshua T. Vogelstein  

**Link**: [PDF](https://arxiv.org/pdf/2507.10951)  

**Abstract**: The complete connectome of the Drosophila larva brain offers a unique opportunity to investigate whether biologically evolved circuits can support artificial intelligence. We convert this wiring diagram into a Biological Processing Unit (BPU), a fixed recurrent network derived directly from synaptic connectivity. Despite its modest size 3,000 neurons and 65,000 weights between them), the unmodified BPU achieves 98% accuracy on MNIST and 58% on CIFAR-10, surpassing size-matched MLPs. Scaling the BPU via structured connectome expansions further improves CIFAR-10 performance, while modality-specific ablations reveal the uneven contributions of different sensory subsystems. On the ChessBench dataset, a lightweight GNN-BPU model trained on only 10,000 games achieves 60% move accuracy, nearly 10x better than any size transformer. Moreover, CNN-BPU models with ~2M parameters outperform parameter-matched Transformers, and with a depth-6 minimax search at inference, reach 91.7% accuracy, exceeding even a 9M-parameter Transformer baseline. These results demonstrate the potential of biofidelic neural architectures to support complex cognitive tasks and motivate scaling to larger and more intelligent connectomes in future work. 

**Abstract (ZH)**: 果蝇幼虫脑的完整连接组提供了一个独特的机会，以研究生物进化电路是否能支持人工智能。我们将其连接图转换为生物处理单元（BPU），这是一种直接源自突触连接的固定循环网络。尽管其规模较小（3,000个神经元和它们之间65,000个权重），未经修改的BPU在MNIST上的准确率达到98%，CIFAR-10上的准确率为58%，超越了相同规模的多层感知机（MLP）。通过对BPU进行结构化连接组扩展进行放大规模进一步提高了CIFAR-10的性能，而模态特定的消融实验揭示了不同感觉子系统不均匀的贡献。在ChessBench数据集上，仅使用10,000场比赛训练的轻量级GNN-BPU模型实现了60%的走棋准确性，几乎是任何大小的变换器的10倍。此外，具有约200万个参数的CNN-BPU模型超过了参数匹配的变换器，并且在推理过程中进行深度为6的极小极大搜索，准确率达到了91.7%，甚至超过了900万个参数的变换器基线。这些结果展示了生物忠实神经架构支持复杂认知任务的潜力，并激励未来工作将这些架构扩展到更大、更智能的连接组。 

---
# Class-Proportional Coreset Selection for Difficulty-Separable Data 

**Title (ZH)**: 难度可分数据的类别比例核心集选择方法 

**Authors**: Elisa Tsai, Haizhong Zheng, Atul Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2507.10904)  

**Abstract**: High-quality training data is essential for building reliable and efficient machine learning systems. One-shot coreset selection addresses this by pruning the dataset while maintaining or even improving model performance, often relying on training-dynamics-based data difficulty scores. However, most existing methods implicitly assume class-wise homogeneity in data difficulty, overlooking variation in data difficulty across different classes.
In this work, we challenge this assumption by showing that, in domains such as network intrusion detection and medical imaging, data difficulty often clusters by class. We formalize this as class-difficulty separability and introduce the Class Difficulty Separability Coefficient (CDSC) as a quantitative measure. We demonstrate that high CDSC values correlate with performance degradation in class-agnostic coreset methods, which tend to overrepresent easy majority classes while neglecting rare but informative ones.
To address this, we introduce class-proportional variants of multiple sampling strategies. Evaluated on five diverse datasets spanning security and medical domains, our methods consistently achieve state-of-the-art data efficiency. For instance, on CTU-13, at an extreme 99% pruning rate, a class-proportional variant of Coverage-centric Coreset Selection (CCS-CP) shows remarkable stability, with accuracy dropping only 2.58%, precision 0.49%, and recall 0.19%. In contrast, the class-agnostic CCS baseline, the next best method, suffers sharper declines of 7.59% in accuracy, 4.57% in precision, and 4.11% in recall.
We further show that aggressive pruning enhances generalization in noisy, imbalanced, and large-scale datasets. Our results underscore that explicitly modeling class-difficulty separability leads to more effective, robust, and generalizable data pruning, particularly in high-stakes scenarios. 

**Abstract (ZH)**: 高质量训练数据对于构建可靠高效的机器学习系统至关重要。基于训练动力学的数据难度评分的一次性核心集选择通过裁剪数据集同时保持或甚至提高模型性能。然而，现有的大多数方法隐含地假设数据难度在类内均匀分布，忽略了不同类间数据难度的变异性。

在本工作中，我们挑战了这一假设，通过实验证明在网络入侵检测和医学成像等领域，数据难度往往按类聚类。我们将其形式化为类难度可分性，并引入了类难度分离系数（CDSC）作为定量度量。我们证明了高CDSC值与类无关核心集方法性能下降相关，这些方法倾向于过度代表易见的多数类而忽视稀有但信息丰富的类。

为此，我们引入了多种采样策略的类比例变体。在覆盖集中核心集选择（CCS-CP）的类比例变体等方面，在五个涵盖安全和医疗领域的大不相同的數據集中，我们的方法始终保持了最先进的数据效率。例如，在CTU-13数据集上，极端99%的裁剪率下，CCS-CP的类比例变体表现出显著的稳定性，准确率仅下降2.58%，精确率0.49%，召回率0.19%。与之相反，类无关的CCS基线方法，作为第二好的方法，在准确率、精确率和召回率方面的下降分别达到7.59%、4.57%和4.11%。

此外，我们还展示了在嘈杂、不平衡和大规模数据集上具有激进裁剪增强泛化能力。我们的结果表明，明确建模类难度可分性可导致更有效、更稳健和更具推广能力的数据裁剪，尤其是在高风险场景中。 

---
# MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning 

**Title (ZH)**: MalCodeAI: 基于语言无关代码推理的自主漏洞检测与修复 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Noha El Kachach  

**Link**: [PDF](https://arxiv.org/pdf/2507.10898)  

**Abstract**: The growing complexity of cyber threats and the limitations of traditional vulnerability detection tools necessitate novel approaches for securing software systems. We introduce MalCodeAI, a language-agnostic, multi-stage AI pipeline for autonomous code security analysis and remediation. MalCodeAI combines code decomposition and semantic reasoning using fine-tuned Qwen2.5-Coder-3B-Instruct models, optimized through Low-Rank Adaptation (LoRA) within the MLX framework, and delivers scalable, accurate results across 14 programming languages. In Phase 1, the model achieved a validation loss as low as 0.397 for functional decomposition and summarization of code segments after 200 iterations, 6 trainable layers, and a learning rate of 2 x 10^(-5). In Phase 2, for vulnerability detection and remediation, it achieved a best validation loss of 0.199 using the same number of iterations and trainable layers but with an increased learning rate of 4 x 10^(-5), effectively identifying security flaws and suggesting actionable fixes. MalCodeAI supports red-hat-style exploit tracing, CVSS-based risk scoring, and zero-shot generalization to detect complex, zero-day vulnerabilities. In a qualitative evaluation involving 15 developers, the system received high scores in usefulness (mean 8.06/10), interpretability (mean 7.40/10), and readability of outputs (mean 7.53/10), confirming its practical value in real-world development workflows. This work marks a significant advancement toward intelligent, explainable, and developer-centric software security solutions. 

**Abstract (ZH)**: 面向复杂网络威胁的软件系统新型保护方法：MalCodeAI多阶段AI代码安全分析与修复pipeline 

---
# Commuting Distance Regularization for Timescale-Dependent Label Inconsistency in EEG Emotion Recognition 

**Title (ZH)**: 时间尺度依赖的标签不一致性下基于通勤距离的正则化方法在EEG情绪识别中的应用 

**Authors**: Xiaocong Zeng, Craig Michoski, Yan Pang, Dongyang Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10895)  

**Abstract**: In this work, we address the often-overlooked issue of Timescale Dependent Label Inconsistency (TsDLI) in training neural network models for EEG-based human emotion recognition. To mitigate TsDLI and enhance model generalization and explainability, we propose two novel regularization strategies: Local Variation Loss (LVL) and Local-Global Consistency Loss (LGCL). Both methods incorporate classical mathematical principles--specifically, functions of bounded variation and commute-time distances--within a graph theoretic framework. Complementing our regularizers, we introduce a suite of new evaluation metrics that better capture the alignment between temporally local predictions and their associated global emotion labels. We validate our approach through comprehensive experiments on two widely used EEG emotion datasets, DREAMER and DEAP, across a range of neural architectures including LSTM and transformer-based models. Performance is assessed using five distinct metrics encompassing both quantitative accuracy and qualitative consistency. Results consistently show that our proposed methods outperform state-of-the-art baselines, delivering superior aggregate performance and offering a principled trade-off between interpretability and predictive power under label inconsistency. Notably, LVL achieves the best aggregate rank across all benchmarked backbones and metrics, while LGCL frequently ranks the second, highlighting the effectiveness of our framework. 

**Abstract (ZH)**: 在基于EEG的人类情绪识别中应对时间尺度依赖标签不一致性的时间尺度依赖标签不一致性问题：提出两种新型正则化策略以增强模型的泛化能力和可解释性 

---
# Modernizing CNN-based Weather Forecast Model towards Higher Computational Efficiency 

**Title (ZH)**: 基于现代计算效率提升的CNN气象预报模型现代化研究 

**Authors**: Minjong Cheon, Eunhan Goo, Su-Hyeon Shin, Muhammad Ahmed, Hyungjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.10893)  

**Abstract**: Recently, AI-based weather forecast models have achieved impressive advances. These models have reached accuracy levels comparable to traditional NWP systems, marking a significant milestone in data-driven weather prediction. However, they mostly leverage Transformer-based architectures, which often leads to high training complexity and resource demands due to the massive parameter sizes. In this study, we introduce a modernized CNN-based model for global weather forecasting that delivers competitive accuracy while significantly reducing computational requirements. To present a systematic modernization roadmap, we highlight key architectural enhancements across multiple design scales from an earlier CNN-based approach. KAI-a incorporates a scale-invariant architecture and InceptionNeXt-based blocks within a geophysically-aware design, tailored to the structure of Earth system data. Trained on the ERA5 daily dataset with 67 atmospheric variables, the model contains about 7 million parameters and completes training in just 12 hours on a single NVIDIA L40s GPU. Our evaluation shows that KAI-a matches the performance of state-of-the-art models in medium-range weather forecasting, while offering a significantly lightweight design. Furthermore, case studies on the 2018 European heatwave and the East Asian summer monsoon demonstrate KAI-a's robust skill in capturing extreme events, reinforcing its practical utility. 

**Abstract (ZH)**: 基于AI的天气预报模型最近取得了显著进展：现代CNN模型在降低计算需求的同时实现竞争性准确度 

---
# How to Protect Models against Adversarial Unlearning? 

**Title (ZH)**: 如何保护模型免受对抗性遗忘攻击？ 

**Authors**: Patryk Jasiorski, Marek Klonowski, Michał Woźniak  

**Link**: [PDF](https://arxiv.org/pdf/2507.10886)  

**Abstract**: AI models need to be unlearned to fulfill the requirements of legal acts such as the AI Act or GDPR, and also because of the need to remove toxic content, debiasing, the impact of malicious instances, or changes in the data distribution structure in which a model works. Unfortunately, removing knowledge may cause undesirable side effects, such as a deterioration in model performance. In this paper, we investigate the problem of adversarial unlearning, where a malicious party intentionally sends unlearn requests to deteriorate the model's performance maximally. We show that this phenomenon and the adversary's capabilities depend on many factors, primarily on the backbone model itself and strategy/limitations in selecting data to be unlearned. The main result of this work is a new method of protecting model performance from these side effects, both in the case of unlearned behavior resulting from spontaneous processes and adversary actions. 

**Abstract (ZH)**: 探讨对抗性遗忘问题：保护模型性能免受恶意遗忘请求的影响 

---
# Overview of the TREC 2022 deep learning track 

**Title (ZH)**: TREC 2022 深度学习赛道综述 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, Ian Soboroff  

**Link**: [PDF](https://arxiv.org/pdf/2507.10865)  

**Abstract**: This is the fourth year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we also leverage both the refreshed passage and document collections that were released last year leading to a nearly $16$ times increase in the size of the passage collection and nearly four times increase in the document collection size. Unlike previous years, in 2022 we mainly focused on constructing a more complete test collection for the passage retrieval task, which has been the primary focus of the track. The document ranking task was kept as a secondary task, where document-level labels were inferred from the passage-level labels. Our analysis shows that similar to previous years, deep neural ranking models that employ large scale pretraining continued to outperform traditional retrieval methods. Due to the focusing our judging resources on passage judging, we are more confident in the quality of this year's queries and judgments, with respect to our ability to distinguish between runs and reuse the dataset in future. We also see some surprises in overall outcomes. Some top-performing runs did not do dense retrieval. Runs that did single-stage dense retrieval were not as competitive this year as they were last year. 

**Abstract (ZH)**: TREC深度学习赛道的第四年：MS MARCO数据集的应用与测试集合的扩展 

---
# PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark 

**Title (ZH)**: PhreshPhish: 一个真实世界、高质量、大规模的网络钓鱼网站数据集及基准 

**Authors**: Thomas Dalton, Hemanth Gowda, Girish Rao, Sachin Pargi, Alireza Hadj Khodabakhshi, Joseph Rombs, Stephan Jou, Manish Marwah  

**Link**: [PDF](https://arxiv.org/pdf/2507.10854)  

**Abstract**: Phishing remains a pervasive and growing threat, inflicting heavy economic and reputational damage. While machine learning has been effective in real-time detection of phishing attacks, progress is hindered by lack of large, high-quality datasets and benchmarks. In addition to poor-quality due to challenges in data collection, existing datasets suffer from leakage and unrealistic base rates, leading to overly optimistic performance results. In this paper, we introduce PhreshPhish, a large-scale, high-quality dataset of phishing websites that addresses these limitations. Compared to existing public datasets, PhreshPhish is substantially larger and provides significantly higher quality, as measured by the estimated rate of invalid or mislabeled data points. Additionally, we propose a comprehensive suite of benchmark datasets specifically designed for realistic model evaluation by minimizing leakage, increasing task difficulty, enhancing dataset diversity, and adjustment of base rates more likely to be seen in the real world. We train and evaluate multiple solution approaches to provide baseline performance on the benchmark sets. We believe the availability of this dataset and benchmarks will enable realistic, standardized model comparison and foster further advances in phishing detection. The datasets and benchmarks are available on Hugging Face (this https URL). 

**Abstract (ZH)**: 钓鱼攻击仍然是一种普遍且不断增长的威胁，对经济和声誉造成重大损害。虽然机器学习在实时检测钓鱼攻击方面表现出效，但缺乏大规模、高质量的数据集和基准数据阻碍了进展。现有的数据集因数据收集挑战而导致质量低下，同时还存在数据泄漏和不切实际的基础率问题，导致过度乐观的性能结果。在本文中，我们介绍了PhreshPhish，这是一个大规模、高质量的钓鱼网站数据集，解决了这些限制。与现有的公共数据集相比，PhreshPhish 更大且数据质量更高，通过估计无效或误标数据点的比例来衡量。此外，我们提出了一整套专门设计的基准数据集，用于最小化数据泄漏、增加任务难度、增强数据集多样性，并调整更接近现实世界的基础率。我们对多个解决方案进行了训练和评估，提供了基准集上的基线性能。我们相信，该数据集和基准的可用性将促进钓鱼检测的现实标准化模型比较，并推动进一步的技术进步。数据集和基准可在 Hugging Face 上获取（此链接：this https URL）。 

---
# Supporting SENĆOTEN Language Documentation Efforts with Automatic Speech Recognition 

**Title (ZH)**: 使用自动语音识别支持SENĆOTEN语言记录工作 

**Authors**: Mengzhe Geng, Patrick Littell, Aidan Pine, PENÁĆ, Marc Tessier, Roland Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2507.10827)  

**Abstract**: The SENĆOTEN language, spoken on the Saanich peninsula of southern Vancouver Island, is in the midst of vigorous language revitalization efforts to turn the tide of language loss as a result of colonial language policies. To support these on-the-ground efforts, the community is turning to digital technology. Automatic Speech Recognition (ASR) technology holds great promise for accelerating language documentation and the creation of educational resources. However, developing ASR systems for SENĆOTEN is challenging due to limited data and significant vocabulary variation from its polysynthetic structure and stress-driven metathesis. To address these challenges, we propose an ASR-driven documentation pipeline that leverages augmented speech data from a text-to-speech (TTS) system and cross-lingual transfer learning with Speech Foundation Models (SFMs). An n-gram language model is also incorporated via shallow fusion or n-best restoring to maximize the use of available data. Experiments on the SENĆOTEN dataset show a word error rate (WER) of 19.34% and a character error rate (CER) of 5.09% on the test set with a 57.02% out-of-vocabulary (OOV) rate. After filtering minor cedilla-related errors, WER improves to 14.32% (26.48% on unseen words) and CER to 3.45%, demonstrating the potential of our ASR-driven pipeline to support SENĆOTEN language documentation. 

**Abstract (ZH)**: SENĆOTEN语言在南温哥华岛萨尼奇半岛上的 revitalization 努力正处于蓬勃发展的阶段，旨在扭转殖民语言政策导致的语言流失。为支持这些实地努力，该社区转向了数字技术。自动语音识别（ASR）技术有望加速语言记录和教育资源的创建。然而，由于数据有限且由于其多合词结构和音节驱动的转移现象导致的词汇变化显著，为SENĆOTEN开发ASR系统具有挑战性。为应对这些挑战，我们提出了一种基于ASR的语言记录管道，该管道利用文本转语音（TTS）系统扩增的语音数据，并通过语音基础模型（SFMs）进行跨语言迁移学习。还通过浅层融合或n-best恢复结合n-gram语言模型，以充分利用可用数据。在SENĆOTEN数据集上的实验结果显示，在测试集上的词错误率（WER）为19.34%，字符错误率（CER）为5.09%，且在词汇外率（OOV）为57.02%的情况下。过滤掉轻型cédilla相关的错误后，WER提高到14.32%（在未见过的单词上为26.48%），CER降低到3.45%，这表明我们的ASR驱动管道在支持SENĆOTEN语言记录方面的潜力。 

---
# Past, Present and Future: Exploring Adaptive AI in Software Development Bots 

**Title (ZH)**: 过去、现在和未来：探索软件开发机器人中的自适应AI 

**Authors**: Omar Elsisi, Glaucia Melo  

**Link**: [PDF](https://arxiv.org/pdf/2507.10822)  

**Abstract**: Conversational agents, such as chatbots and virtual assistants, have become essential in software development, boosting productivity, collaboration, and automating various tasks. This paper examines the role of adaptive AI-powered conversational agents in software development, highlighting their ability to offer dynamic, context-aware assistance to developers. Unlike traditional rule-based systems, adaptive AI agents use machine learning and natural language processing to learn from interactions and improve over time, providing more personalized and responsive help. We look at how these tools have evolved from simple query-based systems to advanced AI-driven solutions like GitHub Copilot and Microsoft Teams bots. We also explore the challenges of integrating adaptive AI into software development processes. The study aims to assess the benefits and limitations of these systems, address concerns like data privacy and ethical issues, and offer insights into their future use in the field. Ultimately, adaptive AI chatbots have great potential to revolutionize software development by delivering real-time, customized support and enhancing the efficiency of development cycles. 

**Abstract (ZH)**: 适应性AI驱动的对话代理在软件开发中的角色研究：从简单查询系统到GitHub Copilot和Microsoft Teams机器人的演变及其挑战 

---
# "Is it always watching? Is it always listening?" Exploring Contextual Privacy and Security Concerns Toward Domestic Social Robots 

**Title (ZH)**: “一直都在监控吗？一直都在倾听吗？”探究家庭社交机器人面临的上下文隐私与安全问题 

**Authors**: Henry Bell, Jabari Kwesi, Hiba Laabadli, Pardis Emami-Naeini  

**Link**: [PDF](https://arxiv.org/pdf/2507.10786)  

**Abstract**: Equipped with artificial intelligence (AI) and advanced sensing capabilities, social robots are gaining interest among consumers in the United States. These robots seem like a natural evolution of traditional smart home devices. However, their extensive data collection capabilities, anthropomorphic features, and capacity to interact with their environment make social robots a more significant security and privacy threat. Increased risks include data linkage, unauthorized data sharing, and the physical safety of users and their homes. It is critical to investigate U.S. users' security and privacy needs and concerns to guide the design of social robots while these devices are still in the early stages of commercialization in the U.S. market. Through 19 semi-structured interviews, we identified significant security and privacy concerns, highlighting the need for transparency, usability, and robust privacy controls to support adoption. For educational applications, participants worried most about misinformation, and in medical use cases, they worried about the reliability of these devices. Participants were also concerned with the data inference that social robots could enable. We found that participants expect tangible privacy controls, indicators of data collection, and context-appropriate functionality. 

**Abstract (ZH)**: 配备人工智能和先进感应能力的社会机器人在美国消费者中引起了关注。这些机器人似乎是传统智能家居设备的自然进化。然而，它们广泛的数据收集能力、拟人特征以及与环境互动的能力使社会机器人成为更大的安全和隐私威胁。增加的风险包括数据链接、未授权数据共享以及用户和其家庭的身体安全。在这些设备在美国市场商业化初期，亟需调查美国用户的安全和隐私需求与关切，以指导社会机器人的设计。通过19次半结构化访谈，我们确定了显著的安全和隐私关切，强调了透明度、易用性和 robust 的隐私控制以支持采用的重要性。对于教育应用，参与者最担忧的是虚假信息，而在医疗应用场景中，他们担心这些设备的可靠性。参与者还对社会机器人可能导致的数据推论表示担忧。我们发现参与者期望实际的隐私控制、数据收集的指示器以及上下文适当的功能。 

---
# Auditing Facial Emotion Recognition Datasets for Posed Expressions and Racial Bias 

**Title (ZH)**: 审计面部情感识别数据集中的摆拍表情和种族偏见 

**Authors**: Rina Khan, Catherine Stinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.10755)  

**Abstract**: Facial expression recognition (FER) algorithms classify facial expressions into emotions such as happy, sad, or angry. An evaluative challenge facing FER algorithms is the fall in performance when detecting spontaneous expressions compared to posed expressions. An ethical (and evaluative) challenge facing FER algorithms is that they tend to perform poorly for people of some races and skin colors. These challenges are linked to the data collection practices employed in the creation of FER datasets. In this study, we audit two state-of-the-art FER datasets. We take random samples from each dataset and examine whether images are spontaneous or posed. In doing so, we propose a methodology for identifying spontaneous or posed images. We discover a significant number of images that were posed in the datasets purporting to consist of in-the-wild images. Since performance of FER models vary between spontaneous and posed images, the performance of models trained on these datasets will not represent the true performance if such models were to be deployed in in-the-wild applications. We also observe the skin color of individuals in the samples, and test three models trained on each of the datasets to predict facial expressions of people from various races and skin tones. We find that the FER models audited were more likely to predict people labeled as not white or determined to have dark skin as showing a negative emotion such as anger or sadness even when they were smiling. This bias makes such models prone to perpetuate harm in real life applications. 

**Abstract (ZH)**: 面部表情识别（FER）算法将面部表情分类为快乐、悲伤或愤怒等情绪。FER算法面临的评估挑战在于，在检测自发表情时的性能低于检测摆造型表情时的性能。FER算法还面临伦理和评估方面的挑战，即它们往往在某些种族和肤色的人群中表现不佳。这些问题与构建FER数据集时采用的数据采集实践有关。在本研究中，我们审计了两个最先进的FER数据集。我们从每个数据集中随机抽取样本，并检查这些图像是否为自发表情或摆造型表情。在此过程中，我们提出了识别自发或摆造型图像的方法。我们发现，声称包含野外拍摄图像的数据集中存在大量摆造型图像。由于FER模型在自发表情和摆造型表情上的性能不同，如果这些模型用于野外应用，其性能将无法代表真实性能。我们还观察了样本中个体的肤色，并在每个数据集上训练了三种模型，以预测来自不同种族和肤色的人的面部表情。我们发现，审计的FER模型更有可能将被标记为非白人或被确定为深肤色的人预测为显示负面情绪如愤怒或悲伤，即使他们在微笑。这种偏见使这些模型在实际应用中更容易造成危害。 

---
# A Group Theoretic Analysis of the Symmetries Underlying Base Addition and Their Learnability by Neural Networks 

**Title (ZH)**: 基于群论分析的基础进位运算的对称性及其可学会性研究 

**Authors**: Cutter Dawes, Simon Segert, Kamesh Krishnamurthy, Jonathan D. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10678)  

**Abstract**: A major challenge in the use of neural networks both for modeling human cognitive function and for artificial intelligence is the design of systems with the capacity to efficiently learn functions that support radical generalization. At the roots of this is the capacity to discover and implement symmetry functions. In this paper, we investigate a paradigmatic example of radical generalization through the use of symmetry: base addition. We present a group theoretic analysis of base addition, a fundamental and defining characteristic of which is the carry function -- the transfer of the remainder, when a sum exceeds the base modulus, to the next significant place. Our analysis exposes a range of alternative carry functions for a given base, and we introduce quantitative measures to characterize these. We then exploit differences in carry functions to probe the inductive biases of neural networks in symmetry learning, by training neural networks to carry out base addition using different carries, and comparing efficacy and rate of learning as a function of their structure. We find that even simple neural networks can achieve radical generalization with the right input format and carry function, and that learning speed is closely correlated with carry function structure. We then discuss the relevance this has for cognitive science and machine learning. 

**Abstract (ZH)**: 神经网络中用于建模人类认知功能和人工智能的重大挑战之一是在高效学习支持深刻泛化的函数方面设计系统的能力。这一挑战的根源在于发现和实现对称函数的能力。本文通过使用对称性探讨了一个深刻的泛化范例：基数进位。我们对基数进位进行了群论分析，其基本特征是进位函数——当和数超过基数模时，将余数转移到下一位显著位置。我们的分析揭示了给定基数下的多种替代进位函数，并引入了量化指标来表征这些函数。然后，我们利用进位函数的差异来探究神经网络在对称学习中的归纳偏见，通过使用不同的进位函数训练神经网络进行基数进位，并比较其效率和学习速率与结构的关系。我们发现，即使简单的神经网络在合适的输入格式和进位函数下也能实现深刻的泛化，且学习速度与进位函数结构密切相关。最后，我们讨论了这一发现对认知科学和机器学习的相关性。 

---
# CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance 

**Title (ZH)**: CodeAssistBench (CAB): 数据集与基于多轮对话的代码辅助基准测试 

**Authors**: Myeongsoo Kim, Shweta Garg, Baishakhi Ray, Varun Kumar, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2507.10646)  

**Abstract**: Programming assistants powered by large language models have transformed software development, yet most benchmarks focus narrowly on code generation tasks. Recent efforts like InfiBench and StackEval attempt to address this gap using Stack Overflow data but remain limited to single-turn interactions in isolated contexts, require significant manual curation, and fail to represent complete project environments. We introduce CodeAssistBench (CAB), the first benchmark framework for evaluating multi-turn programming assistance in realistic settings that address real-world questions about actual codebases. Unlike existing programming Q&A benchmarks, CAB automatically generates scalable datasets from question-related GitHub issues using configurable parameters (e.g., repository creation date, star count, programming languages), and includes automatic containerization of codebases for evaluation. It then evaluates models through simulated users in these containerized environments with full codebase access. Using this framework, we constructed a test set of 3,286 real-world programming questions across 231 repositories, spanning seven programming languages and diverse problem domains. Our evaluation of leading LLMs reveals a substantial capability gap: while models perform well on Stack Overflow questions with success rates of 70-83%, they resolve only up to 16.49% of CAB's recent issues. This discrepancy highlights the challenges of providing assistance in complex, project-specific contexts versus answering standalone questions. 

**Abstract (ZH)**: 由大规模语言模型驱动的编程助手已经革新了软件开发，但大多数基准测试主要集中在代码生成任务上。InfiBench和StackEval等最近的努力尝试利用Stack Overflow数据来解决这一问题，但仍局限于孤立情境中的单轮交互，需要大量人工整理，并未能代表完整的项目环境。我们提出了CodeAssistBench (CAB)，这是第一个用于评估多轮编程辅助的基准框架，可在现实情境中评估关于实际代码库的现实问题。与现有的编程问答基准不同，CAB通过可配置参数（如仓库创建日期、星标数、编程语言）自动生成可扩展的数据集，并包括对代码库进行自动容器化以进行评估。随后，在这些容器化环境中通过模拟用户进行评估，提供完整的代码库访问权限。通过该框架，我们构建了一个包含3,286个实际编程问题的测试集，跨越231个仓库，涉及七种编程语言和多种问题领域。我们的评估结果揭示了领头的语言模型之间存在显著的能力差距：尽管模型在Stack Overflow问题上的成功率在70-83%之间表现良好，但它们仅解决CAB的最近问题的16.49%。这一差异突显了在复杂、项目特定的背景下提供帮助与回答独立问题的挑战。 

---
# TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models 

**Title (ZH)**: 基于泰勒展开的后验归因改进方法：TaylorPODA 

**Authors**: Yuchi Tang, Iñaki Esnaola, Suzanne Mason, George Panoutsos  

**Link**: [PDF](https://arxiv.org/pdf/2507.10643)  

**Abstract**: Existing post-hoc model-agnostic methods generate external explanations for opaque models, primarily by locally attributing the model output to its input features. However, they often lack an explicit and systematic framework for quantifying the contribution of individual features. Building on the Taylor expansion framework introduced by Deng et al. (2024) to unify existing local attribution methods, we propose a rigorous set of postulates -- "precision", "federation", and "zero-discrepancy" -- to govern Taylor term-specific attribution. Guided by these postulates, we introduce TaylorPODA (Taylor expansion-derived imPortance-Order aDapted Attribution), which incorporates an additional "adaptation" property. This property enables alignment with task-specific goals, especially in post-hoc settings lacking ground-truth explanations. Empirical evaluations demonstrate that TaylorPODA achieves competitive results against baseline methods, providing principled and visualization-friendly explanations. This work represents a step toward the trustworthy deployment of opaque models by offering explanations with stronger theoretical grounding. 

**Abstract (ZH)**: 现有的后验模型无导方法通过局部归因模型输出到输入特征来为不透明模型生成外部解释，但往往缺乏明确且系统的框架来量化单个特征的贡献。基于Deng等（2024）引入的用于统一现有局部归因方法的泰勒展开框架，我们提出了一套严格的公理——“精确性”、“联邦性”和“零偏差”——来管理泰勒项特定的归因。遵循这些公理，我们提出了TaylorPODA（泰勒展开衍生的重要性和适应性归因），并引入了额外的“适应性”性质。该性质使得解释能够与特定任务目标对齐，特别是在缺乏ground-truth解释的后验 scenarios 中。实证评估表明，TaylorPODA 在与基准方法的对比中取得了竞争力的结果，提供了具有更强理论支撑且易于可视化解释。这项工作代表了在增强不透明模型可信赖部署方面迈出的一步，通过提供具有更强理论基础的解释。 

---
# First-of-its-kind AI model for bioacoustic detection using a lightweight associative memory Hopfield neural network 

**Title (ZH)**: 首创的基于轻量级关联记忆霍皮菲尔德神经网络的生物声学检测AI模型 

**Authors**: Andrew Gascoyne, Wendy Lomas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10642)  

**Abstract**: A growing issue within conservation bioacoustics is the task of analysing the vast amount of data generated from the use of passive acoustic monitoring devices. In this paper, we present an alternative AI model which has the potential to help alleviate this problem. Our model formulation addresses the key issues encountered when using current AI models for bioacoustic analysis, namely the: limited training data available; environmental impact, particularly in energy consumption and carbon footprint of training and implementing these models; and associated hardware requirements. The model developed in this work uses associative memory via a transparent, explainable Hopfield neural network to store signals and detect similar signals which can then be used to classify species. Training is rapid ($3$\,ms), as only one representative signal is required for each target sound within a dataset. The model is fast, taking only $5.4$\,s to pre-process and classify all $10384$ publicly available bat recordings, on a standard Apple MacBook Air. The model is also lightweight with a small memory footprint of $144.09$\,MB of RAM usage. Hence, the low computational demands make the model ideal for use on a variety of standard personal devices with potential for deployment in the field via edge-processing devices. It is also competitively accurate, with up to $86\%$ precision on the dataset used to evaluate the model. In fact, we could not find a single case of disagreement between model and manual identification via expert field guides. Although a dataset of bat echolocation calls was chosen to demo this first-of-its-kind AI model, trained on only two representative calls, the model is not species specific. In conclusion, we propose an equitable AI model that has the potential to be a game changer for fast, lightweight, sustainable, transparent, explainable and accurate bioacoustic analysis. 

**Abstract (ZH)**: 生物声学中一个日益突出的问题是如何处理由被动声学监测设备生成的大规模数据。本文提出了一种替代的人工智能模型，旨在缓解这一问题。该模型解决了当前用于生物声学分析的人工智能模型面临的几个关键问题，包括有限的训练数据、环境影响（尤其是训练和实施这些模型的能耗和碳足迹）以及相关的硬件需求。本文开发的模型利用透明可解释的霍普菲尔德神经网络的关联记忆来存储信号并检测相似信号，从而进行物种分类。训练快速（3毫秒），只需要每个目标声音数据集中的一组代表性信号。该模型速度快，仅需5.4秒即可预处理并分类全部10384个公开的蝙蝠录音，运行于标准的苹果MacBook Air上。该模型也具有轻量化特点，内存占用仅为144.09兆字节。因此，低计算需求使得该模型适用于各种标准个人设备，并具有通过边缘处理设备在现场部署的潜力。此外，该模型还具有竞争力的准确度，精度可达86%。实际上，在利用该模型进行评估的数据集中，模型与专家实地指南的手动识别之间没有发现任何分歧。尽管该模型最初用于演示目的的数据集选择了蝙蝠回声定位叫声，并且仅基于两个代表性叫声进行训练，但该模型并非针对特定物种。综上所述，我们提出了一种公平的人工智能模型，该模型有望成为快速、轻量化、可持续、透明和准确的生物声学分析的变革者。 

---
# A Simple Baseline for Stable and Plastic Neural Networks 

**Title (ZH)**: 稳定且可塑的神经网络的一个简单基线 

**Authors**: É. Künzel, A. Jaziri, V. Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10637)  

**Abstract**: Continual learning in computer vision requires that models adapt to a continuous stream of tasks without forgetting prior knowledge, yet existing approaches often tip the balance heavily toward either plasticity or stability. We introduce RDBP, a simple, low-overhead baseline that unites two complementary mechanisms: ReLUDown, a lightweight activation modification that preserves feature sensitivity while preventing neuron dormancy, and Decreasing Backpropagation, a biologically inspired gradient-scheduling scheme that progressively shields early layers from catastrophic updates. Evaluated on the Continual ImageNet benchmark, RDBP matches or exceeds the plasticity and stability of state-of-the-art methods while reducing computational cost. RDBP thus provides both a practical solution for real-world continual learning and a clear benchmark against which future continual learning strategies can be measured. 

**Abstract (ZH)**: 计算机视觉中的持续学习要求模型能够适应连续的任务流并保留先前的知识，但现有方法往往在塑性和稳定性之间偏向一方。我们提出了RDBP，一种简单且低开销的基线方法，结合了两种互补机制：ReLUDown，一种轻量级的激活修改，保留特征敏感性同时防止神经元休眠，以及递减反向传播，一种受生物启发的梯度调度方案，逐步屏蔽早期层免遭灾难性更新。在持续ImageNet基准上评估，RDBP在塑性和稳定性方面与最先进的方法相当或超越，同时降低了计算成本。因此，RDBP不仅提供了一种实用的现实世界持续学习解决方案，还提供了一个明确的基准，未来持续学习策略可据此进行衡量。 

---
# Scalable Unsupervised Segmentation via Random Fourier Feature-based Gaussian Process 

**Title (ZH)**: 基于随机傅里叶特征的高斯过程的可扩展无监督分割 

**Authors**: Issei Saito, Masatoshi Nagano, Tomoaki Nakamura, Daichi Mochihashi, Koki Mimura  

**Link**: [PDF](https://arxiv.org/pdf/2507.10632)  

**Abstract**: In this paper, we propose RFF-GP-HSMM, a fast unsupervised time-series segmentation method that incorporates random Fourier features (RFF) to address the high computational cost of the Gaussian process hidden semi-Markov model (GP-HSMM). GP-HSMM models time-series data using Gaussian processes, requiring inversion of an N times N kernel matrix during training, where N is the number of data points. As the scale of the data increases, matrix inversion incurs a significant computational cost. To address this, the proposed method approximates the Gaussian process with linear regression using RFF, preserving expressive power while eliminating the need for inversion of the kernel matrix. Experiments on the Carnegie Mellon University (CMU) motion-capture dataset demonstrate that the proposed method achieves segmentation performance comparable to that of conventional methods, with approximately 278 times faster segmentation on time-series data comprising 39,200 frames. 

**Abstract (ZH)**: 本研究提出了一种快速无监督时间序列分割方法RFF-GP-HSMM，该方法通过引入随机傅里叶特征（RFF）来解决高斯过程隐半马尔可夫模型（GP-HSMM）的高计算成本问题。 

---
# SQLord: A Robust Enterprise Text-to-SQL Solution via Reverse Data Generation and Workflow Decomposition 

**Title (ZH)**: SQLord：一种基于逆向数据生成和工作流分解的企业级文本到SQL解决方案 

**Authors**: Song Cheng, Qiannan Cheng, Linbo Jin, Lei Yi, Guannan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10629)  

**Abstract**: Transforming natural language into SQL queries (NL2SQL) is crucial for data-driven business applications. Existing frameworks, trained on open-source datasets, struggle with complex business logic and lack domain-specific data for fine-tuning. Additionally, evaluation methods often require annotated data and executable database environments, which are scarce in real-world scenarios. To address these challenges, we propose SQLord, an enterprise-level NL2SQL framework. First, SQLord introduces a data reverse generation approach to convert raw SQL statements into annotated data for supervised fine-tuning (SFT). Second, it proposes a decomposition method for complex queries using an automated workflow generator. Additionally, SQLord features a comprehensive GPT-Judge evaluation framework, including Execution Evaluation (EXE), Query-SQL Evaluation (QSE), and SQL-SQL Evaluation (SSE), tailored to diverse scenarios. Offline tests significantly outperform state of the art baselines, and online accuracy consistently exceeds 90, highlighting SQLord's advantages and effectiveness in complex real world scenarios. SQLord has been successfully applied across multiple scenarios on the world's largest B2B e-commerce platform. 

**Abstract (ZH)**: 将自然语言转换为SQL查询（NL2SQL）对于数据驱动的企业应用至关重要。现有的框架在开源数据集上训练，难以处理复杂的业务逻辑，并缺乏领域特定的数据进行微调。此外，评估方法通常需要标注数据和可执行的数据库环境，而在实际场景中这些资源稀缺。为解决这些问题，我们提出了SQLord，一个面向企业的NL2SQL框架。首先，SQLord引入了数据逆向生成方法，将原始SQL语句转换为标注数据以进行监督微调（SFT）。其次，它提出了复杂查询的分解方法，并使用自动化工作流生成器。此外，SQLord还具备全面的GPT-Judge评估框架，包括执行评估（EXE）、查询-SQL评估（QSE）和SQL-SQL评估（SSE），以适应不同的场景。离线测试显著优于最新的基线方法，在线准确率始终超过90%，突显了SQLord在复杂实际场景中的优势和有效性。SQLord已在世界上最大的B2B电子商务平台上成功应用于多个场景。 

---
# Player-Team Heterogeneous Interaction Graph Transformer for Soccer Outcome Prediction 

**Title (ZH)**: 基于球员-团队异质交互图变换器的足球比赛结果预测 

**Authors**: Lintao Wang, Shiwen Xu, Michael Horton, Joachim Gudmundsson, Zhiyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10626)  

**Abstract**: Predicting soccer match outcomes is a challenging task due to the inherently unpredictable nature of the game and the numerous dynamic factors influencing results. While it conventionally relies on meticulous feature engineering, deep learning techniques have recently shown a great promise in learning effective player and team representations directly for soccer outcome prediction. However, existing methods often overlook the heterogeneous nature of interactions among players and teams, which is crucial for accurately modeling match dynamics. To address this gap, we propose HIGFormer (Heterogeneous Interaction Graph Transformer), a novel graph-augmented transformer-based deep learning model for soccer outcome prediction. HIGFormer introduces a multi-level interaction framework that captures both fine-grained player dynamics and high-level team interactions. Specifically, it comprises (1) a Player Interaction Network, which encodes player performance through heterogeneous interaction graphs, combining local graph convolutions with a global graph-augmented transformer; (2) a Team Interaction Network, which constructs interaction graphs from a team-to-team perspective to model historical match relationships; and (3) a Match Comparison Transformer, which jointly analyzes both team and player-level information to predict match outcomes. Extensive experiments on the WyScout Open Access Dataset, a large-scale real-world soccer dataset, demonstrate that HIGFormer significantly outperforms existing methods in prediction accuracy. Furthermore, we provide valuable insights into leveraging our model for player performance evaluation, offering a new perspective on talent scouting and team strategy analysis. 

**Abstract (ZH)**: 基于异质交互图变换器的足球比赛结果预测 

---
# Spectral Feature Extraction for Robust Network Intrusion Detection Using MFCCs 

**Title (ZH)**: 基于MFCC的谱特征提取在鲁棒网络入侵检测中的应用 

**Authors**: HyeYoung Lee, Muhammad Nadeem, Pavel Tsoi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10622)  

**Abstract**: The rapid expansion of Internet of Things (IoT) networks has led to a surge in security vulnerabilities, emphasizing the critical need for robust anomaly detection and classification techniques. In this work, we propose a novel approach for identifying anomalies in IoT network traffic by leveraging the Mel-frequency cepstral coefficients (MFCC) and ResNet-18, a deep learning model known for its effectiveness in feature extraction and image-based tasks. Learnable MFCCs enable adaptive spectral feature representation, capturing the temporal patterns inherent in network traffic more effectively than traditional fixed MFCCs. We demonstrate that transforming raw signals into MFCCs maps the data into a higher-dimensional space, enhancing class separability and enabling more effective multiclass classification. Our approach combines the strengths of MFCCs with the robust feature extraction capabilities of ResNet-18, offering a powerful framework for anomaly detection. The proposed model is evaluated on three widely used IoT intrusion detection datasets: CICIoT2023, NSL-KDD, and IoTID20. The experimental results highlight the potential of integrating adaptive signal processing techniques with deep learning architectures to achieve robust and scalable anomaly detection in heterogeneous IoT network landscapes. 

**Abstract (ZH)**: 物联网网络的迅速扩展引发了安全漏洞的激增，强调了需要强大的异常检测和分类技术的重要性。在此项工作中，我们提出了一种通过利用梅尔频率倒谱系数（MFCC）和ResNet-18（一种在特征提取和图像任务中 effectiveness 优异的深度学习模型）来识别物联网网络流量中异常的新方法。可学习的MFCC能够实现自适应频谱特征表示，相比传统的固定MFCC更能有效捕捉网络流量中的时序模式。我们证明，将原始信号转换为MFCC能够将数据映射到更高维度的空间，增强类间的可分性，并使多类分类更加有效。该方法结合了MFCC的优势与ResNet-18强大的特征提取能力，提供了一种强大的异常检测框架。所提出的模型在CICIoT2023、NSL-KDD和IoTID20三个广泛使用的物联网入侵检测数据集上进行了评估。实验结果表明，将自适应信号处理技术与深度学习架构相结合，能够在异构的物联网网络环境中实现稳健且可扩展的异常检测。 

---
# Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats 

**Title (ZH)**: 博弈论邂逅大语言模型与自主AI：重塑智能威胁时代的网络安全 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10621)  

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.
The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.
This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems. 

**Abstract (ZH)**: 游戏理论、自主AI与网络安全的交集 

---
# Compute Requirements for Algorithmic Innovation in Frontier AI Models 

**Title (ZH)**: 算法创新在前沿AI模型中的计算需求 

**Authors**: Peter Barnett  

**Link**: [PDF](https://arxiv.org/pdf/2507.10618)  

**Abstract**: Algorithmic innovation in the pretraining of large language models has driven a massive reduction in the total compute required to reach a given level of capability. In this paper we empirically investigate the compute requirements for developing algorithmic innovations. We catalog 36 pre-training algorithmic innovations used in Llama 3 and DeepSeek-V3. For each innovation we estimate both the total FLOP used in development and the FLOP/s of the hardware utilized. Innovations using significant resources double in their requirements each year. We then use this dataset to investigate the effect of compute caps on innovation. Our analysis suggests that compute caps alone are unlikely to dramatically slow AI algorithmic progress. Even stringent compute caps -- such as capping total operations to the compute used to train GPT-2 or capping hardware capacity to 8 H100 GPUs -- could still have allowed for half of the cataloged innovations. 

**Abstract (ZH)**: 算法创新在大型语言模型的预训练中的进步推动了达到特定能力所需总计算量的大幅减少。本文通过实证研究探索算法创新所需的计算需求。我们列出了在Llama 3和DeepSeek-V3中使用的36项预训练算法创新，并为每项创新估算了开发过程中使用的总FLOP以及所用硬件的FLOP/s。使用大量资源的创新每年其需求翻倍。我们利用此数据集研究计算限制对创新的影响。我们的分析表明，仅靠计算限制不大可能显著减慢AI算法的进步。即使是非常严格的计算限制——如将总操作量限制为训练GPT-2所用的计算量或将硬件容量限制为8个H100 GPU——也可能仍允许实现所列创新的一半。 

---
# FedGSCA: Medical Federated Learning with Global Sample Selector and Client Adaptive Adjuster under Label Noise 

**Title (ZH)**: FedGSCA: 面向标签噪声的医疗联邦学习方法，包含全局样本选择器和客户端自适应调整器 

**Authors**: Mengwen Ye, Yingzi Huangfu, Shujian Gao, Wei Ren, Weifan Liu, Zekuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10611)  

**Abstract**: Federated Learning (FL) emerged as a solution for collaborative medical image classification while preserving data privacy. However, label noise, which arises from inter-institutional data variability, can cause training instability and degrade model performance. Existing FL methods struggle with noise heterogeneity and the imbalance in medical data. Motivated by these challenges, we propose FedGSCA, a novel framework for enhancing robustness in noisy medical FL. FedGSCA introduces a Global Sample Selector that aggregates noise knowledge from all clients, effectively addressing noise heterogeneity and improving global model stability. Furthermore, we develop a Client Adaptive Adjustment (CAA) mechanism that combines adaptive threshold pseudo-label generation and Robust Credal Labeling Loss. CAA dynamically adjusts to class distributions, ensuring the inclusion of minority samples and carefully managing noisy labels by considering multiple plausible labels. This dual approach mitigates the impact of noisy data and prevents overfitting during local training, which improves the generalizability of the model. We evaluate FedGSCA on one real-world colon slides dataset and two synthetic medical datasets under various noise conditions, including symmetric, asymmetric, extreme, and heterogeneous types. The results show that FedGSCA outperforms the state-of-the-art methods, excelling in extreme and heterogeneous noise scenarios. Moreover, FedGSCA demonstrates significant advantages in improving model stability and handling complex noise, making it well-suited for real-world medical federated learning scenarios. 

**Abstract (ZH)**: 联邦学习（FL）作为一种在保护数据隐私的同时进行协作医学图像分类的解决方案而出现。然而，由机构间数据变异引起的标签噪声会导致训练不稳定并降低模型性能。现有的FL方法难以应对噪声异质性和医学数据的不平衡。为应对这些挑战，我们提出了一种增强鲁棒性的新型框架FedGSCA，用于嘈杂的医学联邦学习。FedGSCA引入了全局样本选择器，该选择器从所有客户端聚合噪声知识，有效解决了噪声异质性并提高了全局模型稳定性。此外，我们开发了一种客户端自适应调整（CAA）机制，该机制结合了自适应阈值伪标签生成和鲁棒区间标签损失。CAA机制动态适应类别分布，确保少数样本的包含，并通过考虑多个可能标签仔细管理噪声标签。这种双重方法减轻了嘈杂数据的影响，并防止了本地训练中的过拟合，从而提高了模型的泛化能力。我们在一个真实的结肠切片数据集和两个合成的医学数据集上对FedGSCA进行了评估，涵盖对称、不对称、极端和异质性等多种噪声条件。实验结果表明，FedGSCA在极端和异质噪声场景中超越了最先进的方法。此外，FedGSCA在提高模型稳定性和处理复杂噪声方面表现出显著优势，使其非常适合实际的医学联邦学习场景。 

---
# Neural Expectation Operators 

**Title (ZH)**: 神经期望算子 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10607)  

**Abstract**: This paper introduces \textbf{Measure Learning}, a paradigm for modeling ambiguity via non-linear expectations. We define Neural Expectation Operators as solutions to Backward Stochastic Differential Equations (BSDEs) whose drivers are parameterized by neural networks. The main mathematical contribution is a rigorous well-posedness theorem for BSDEs whose drivers satisfy a local Lipschitz condition in the state variable $y$ and quadratic growth in its martingale component $z$. This result circumvents the classical global Lipschitz assumption, is applicable to common neural network architectures (e.g., with ReLU activations), and holds for exponentially integrable terminal data, which is the sharp condition for this setting. Our primary innovation is to build a constructive bridge between the abstract, and often restrictive, assumptions of the deep theory of quadratic BSDEs and the world of machine learning, demonstrating that these conditions can be met by concrete, verifiable neural network designs. We provide constructive methods for enforcing key axiomatic properties, such as convexity, by architectural design. The theory is extended to the analysis of fully coupled Forward-Backward SDE systems and to the asymptotic analysis of large interacting particle systems, for which we establish both a Law of Large Numbers (propagation of chaos) and a Central Limit Theorem. This work provides the foundational mathematical framework for data-driven modeling under ambiguity. 

**Abstract (ZH)**: 这篇论文介绍了通过非线性期望建模不确定性的一种范式——度量学习。我们定义了神经期望算子，它是满足特定条件的后向随机微分方程（BSDE）的解，其驱动项由神经网络参数化。主要的数学贡献是一道严谨的存在唯一性定理，该定理适用于满足状态变量局部Lipschitz条件和鞅分量二次增长条件的BSDE，这绕过了经典的全局Lipschitz假设，适用于常见的神经网络结构（例如使用ReLU激活函数），并且适用于指数可积的终端数据，这是该设置下的尖锐条件。我们的主要创新在于，在抽象且通常具有限制性的二次BSDE深理论假设与机器学习的世界之间建立了一个建设性的桥梁，证明了这些条件可以通过具体的验证性神经网络设计实现。我们提供了通过结构设计来强制执行关键公理性质的方法，如凸性。该理论扩展到了完全耦合的前向-后向SDE系统及大规模相互作用粒子系统的渐近分析，在后者中我们建立了大数定律（混乱的传播）以及中心极限定理。本工作提供了在不确定性条件下数据驱动建模的理论基础。 

---
# DALI-PD: Diffusion-based Synthetic Layout Heatmap Generation for ML in Physical Design 

**Title (ZH)**: DALI-PD：基于扩散的合成布局热图生成方法在物理设计中的应用 

**Authors**: Bing-Yue Wu, Vidya A. Chhabria  

**Link**: [PDF](https://arxiv.org/pdf/2507.10606)  

**Abstract**: Machine learning (ML) has demonstrated significant promise in various physical design (PD) tasks. However, model generalizability remains limited by the availability of high-quality, large-scale training datasets. Creating such datasets is often computationally expensive and constrained by IP. While very few public datasets are available, they are typically static, slow to generate, and require frequent updates. To address these limitations, we present DALI-PD, a scalable framework for generating synthetic layout heatmaps to accelerate ML in PD research. DALI-PD uses a diffusion model to generate diverse layout heatmaps via fast inference in seconds. The heatmaps include power, IR drop, congestion, macro placement, and cell density maps. Using DALI-PD, we created a dataset comprising over 20,000 layout configurations with varying macro counts and placements. These heatmaps closely resemble real layouts and improve ML accuracy on downstream ML tasks such as IR drop or congestion prediction. 

**Abstract (ZH)**: 基于扩散模型的DALI-PD：规模化生成合成布局热图以加速物理设计中的机器学习研究 

---
# Divide-Then-Rule: A Cluster-Driven Hierarchical Interpolator for Attribute-Missing Graphs 

**Title (ZH)**: 分而治之：一种基于聚类的分层插值器，用于属性缺失图 

**Authors**: Yaowen Hu, Wenxuan Tu, Yue Liu, Miaomiao Li, Wenpeng Lu, Zhigang Luo, Xinwang Liu, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10595)  

**Abstract**: Deep graph clustering (DGC) for attribute-missing graphs is an unsupervised task aimed at partitioning nodes with incomplete attributes into distinct clusters. Addressing this challenging issue is vital for practical applications. However, research in this area remains underexplored. Existing imputation methods for attribute-missing graphs often fail to account for the varying amounts of information available across node neighborhoods, leading to unreliable results, especially for nodes with insufficient known neighborhood. To address this issue, we propose a novel method named Divide-Then-Rule Graph Completion (DTRGC). This method first addresses nodes with sufficient known neighborhood information and treats the imputed results as new knowledge to iteratively impute more challenging nodes, while leveraging clustering information to correct imputation errors. Specifically, Dynamic Cluster-Aware Feature Propagation (DCFP) initializes missing node attributes by adjusting propagation weights based on the clustering structure. Subsequently, Hierarchical Neighborhood-aware Imputation (HNAI) categorizes attribute-missing nodes into three groups based on the completeness of their neighborhood attributes. The imputation is performed hierarchically, prioritizing the groups with nodes that have the most available neighborhood information. The cluster structure is then used to refine the imputation and correct potential errors. Finally, Hop-wise Representation Enhancement (HRE) integrates information across multiple hops, thereby enriching the expressiveness of node representations. Experimental results on six widely used graph datasets show that DTRGC significantly improves the clustering performance of various DGC methods under attribute-missing graphs. 

**Abstract (ZH)**: 基于属性缺失图的分层图补全（DTRGC）：一种新颖的方法 

---
# Extension OL-MDISF: Online Learning from Mix-Typed, Drifted, and Incomplete Streaming Features 

**Title (ZH)**: 扩展OL-MDISF：面向混合类型、漂移和不完全流式特征的在线学习 

**Authors**: Shengda Zhuo, Di Wu, Yi He, Shuqiang Huang, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10594)  

**Abstract**: Online learning, where feature spaces can change over time, offers a flexible learning paradigm that has attracted considerable attention. However, it still faces three significant challenges. First, the heterogeneity of real-world data streams with mixed feature types presents challenges for traditional parametric modeling. Second, data stream distributions can shift over time, causing an abrupt and substantial decline in model performance. Third, it is often infeasible to label every data instance due to time and cost constraints. To address these issues, we proposed OL-MDISF (Online Learning from Mix-typed, Drifted, and Incomplete Streaming Features), which constructs a latent copula-based representation for heterogeneous features, detects drifts via ensemble entropy and latent mismatch, and performs structure-aware pseudo-labeling.
This companion paper serves as a standalone technical reference to OL-MDISF. It provides a contextual discussion of related work in mixed-type modeling, drift adaptation, and weak supervision, as well as a comprehensive set of experiments across 14 real-world datasets under two types of drift scenarios. These include CER trends, ablation studies, sensitivity analyses, and temporal ensemble dynamics. We hope this document offers a reproducible benchmark for online learning on complex, weakly supervised streaming data. 

**Abstract (ZH)**: 在线学习中的特征空间随时间变化提供了一种灵活的学习范式，吸引了广泛关注。然而，它仍面临三个重大挑战。首先，包含混合特征类型的现实世界数据流的异质性给传统参数建模带来了挑战。其次，数据流分布会随时间发生变化，导致模型性能突然且显著下降。第三，由于时间成本限制，标记每条数据实例往往是不现实的。为了解决这些问题，我们提出了OL-MDISF（在线学习中的混合类型、漂移和不完整流特征），该方法构建了一个基于潜在copula的表示，通过集成熵和潜在不匹配来检测漂移，并进行结构感知的伪标签生成。这篇同伴论文作为OL-MDISF的独立技术参考，提供了关于混合类型建模、漂移适应和弱监督的相关工作讨论，并在两种类型的漂移场景下对14个真实世界数据集进行了全面实验，包括CER趋势分析、消融研究、敏感性分析和时间集成动态。我们希望这份文档为复杂、弱监督流数据的在线学习提供可重复的基准。 

---
# MH-FSF: A Unified Framework for Overcoming Benchmarking and Reproducibility Limitations in Feature Selection Evaluation 

**Title (ZH)**: MH-FSF：克服特征选择评估中基准测试和可重复性限制的统一框架 

**Authors**: Vanderson Rocha, Diego Kreutz, Gabriel Canto, Hendrio Bragança, Eduardo Feitosa  

**Link**: [PDF](https://arxiv.org/pdf/2507.10591)  

**Abstract**: Feature selection is vital for building effective predictive models, as it reduces dimensionality and emphasizes key features. However, current research often suffers from limited benchmarking and reliance on proprietary datasets. This severely hinders reproducibility and can negatively impact overall performance. To address these limitations, we introduce the MH-FSF framework, a comprehensive, modular, and extensible platform designed to facilitate the reproduction and implementation of feature selection methods. Developed through collaborative research, MH-FSF provides implementations of 17 methods (11 classical, 6 domain-specific) and enables systematic evaluation on 10 publicly available Android malware datasets. Our results reveal performance variations across both balanced and imbalanced datasets, highlighting the critical need for data preprocessing and selection criteria that account for these asymmetries. We demonstrate the importance of a unified platform for comparing diverse feature selection techniques, fostering methodological consistency and rigor. By providing this framework, we aim to significantly broaden the existing literature and pave the way for new research directions in feature selection, particularly within the context of Android malware detection. 

**Abstract (ZH)**: 特征选择对于构建有效的预测模型至关重要，因为它可以降低维度并强调关键特征。然而，当前的研究往往受限于有限的基准测试和对专有数据集的依赖。这严重阻碍了可重复性并可能负面影响整体性能。为解决这些局限性，我们提出了MH-FSF框架，这是一个全面、模块化和可扩展的平台，旨在促进特征选择方法的重现和实现。MH-FSF通过合作研究提供了17种方法（11种经典方法、6种领域特定方法）的实现，并在10个公开可用的Android恶意软件数据集上进行系统的评估。我们的结果显示，在平衡和不平衡数据集上存在性能差异，突显了在这些不对称性中考虑数据预处理和选择标准的迫切需求。我们证明了提供统一平台以比较多样化的特征选择技术的重要性，促进方法论的一致性和严谨性。通过提供这一框架，我们希望能够显著拓宽现有文献并为特征选择领域的新兴研究方向铺平道路，特别是在Android恶意软件检测的背景下。 

---
# $\texttt{Droid}$: A Resource Suite for AI-Generated Code Detection 

**Title (ZH)**: Droid: 一种用于AI生成代码检测的资源套件 

**Authors**: Daniil Orel, Indraneil Paul, Iryna Gurevych, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10583)  

**Abstract**: In this work, we compile $\textbf{$\texttt{DroidCollection}$}$, the most extensive open data suite for training and evaluating machine-generated code detectors, comprising over a million code samples, seven programming languages, outputs from 43 coding models, and over three real-world coding domains. Alongside fully AI-generated samples, our collection includes human-AI co-authored code, as well as adversarial samples explicitly crafted to evade detection. Subsequently, we develop $\textbf{$\texttt{DroidDetect}$}$, a suite of encoder-only detectors trained using a multi-task objective over $\texttt{DroidCollection}$. Our experiments show that existing detectors' performance fails to generalise to diverse coding domains and programming languages outside of their narrow training data. Additionally, we demonstrate that while most detectors are easily compromised by humanising the output distributions using superficial prompting and alignment approaches, this problem can be easily amended by training on a small amount of adversarial data. Finally, we demonstrate the effectiveness of metric learning and uncertainty-based resampling as means to enhance detector training on possibly noisy distributions. 

**Abstract (ZH)**: 本工作中，我们编纂了规模最大、最全面的开放数据集库$\textbf{$\texttt{DroidCollection}$}$，用于训练和评估机器生成代码检测器，包含超过一百万条代码样本、七种编程语言、43种编程模型的输出以及超过三个实际编程领域。除了完全由AI生成的样本外，我们的数据集还包括人机合著代码以及故意设计以规避检测的对抗样本。随后，我们开发了$\textbf{$\texttt{DroidDetect}$}$检测套件，该套件的检测器仅基于$\texttt{DroidCollection}$上的多任务目标训练。实验结果表明，现有检测器的性能无法泛化到其狭窄训练数据集以外的多样编程领域和编程语言。此外，我们证明了通过使用表面性的提示和对齐方法使人机化输出分布，大多数检测器容易被篡改，这一问题可以通过少量对抗数据的训练得以轻易解决。最后，我们展示了使用度量学习和基于不确定性重采样作为提高检测器训练效率的有效方法。 

---
# Universal Approximation Theorem for a Single-Layer Transformer 

**Title (ZH)**: 单层变换器的通用逼近定理 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10581)  

**Abstract**: Deep learning employs multi-layer neural networks trained via the backpropagation algorithm. This approach has achieved success across many domains and relies on adaptive gradient methods such as the Adam optimizer. Sequence modeling evolved from recurrent neural networks to attention-based models, culminating in the Transformer architecture. Transformers have achieved state-of-the-art performance in natural language processing (for example, BERT and GPT-3) and have been applied in computer vision and computational biology. However, theoretical understanding of these models remains limited. In this paper, we examine the mathematical foundations of deep learning and Transformers and present a novel theoretical result. We review key concepts from linear algebra, probability, and optimization that underpin deep learning, and we analyze the multi-head self-attention mechanism and the backpropagation algorithm in detail. Our main contribution is a universal approximation theorem for Transformers: we prove that a single-layer Transformer, comprising one self-attention layer followed by a position-wise feed-forward network with ReLU activation, can approximate any continuous sequence-to-sequence mapping on a compact domain to arbitrary precision. We provide a formal statement and a complete proof. Finally, we present case studies that demonstrate the practical implications of this result. Our findings advance the theoretical understanding of Transformer models and help bridge the gap between theory and practice. 

**Abstract (ZH)**: 深度学习采用多层神经网络并通过反向传播算法训练。这种方法已在许多领域取得成功，并依赖于自适应梯度方法，如Adam优化器。序列建模从递归神经网络发展到基于注意力的模型，最终形成了Transformer架构。Transformer在自然语言处理（例如BERT和GPT-3）中达到了最先进的性能，并被应用于计算机视觉和计算生物学。然而，对这些模型的理解仍缺乏理论上的认识。在本文中，我们探讨了深度学习和Transformer的数学基础，并提出了一个新的理论成果。我们回顾了线性代数、概率和优化中的核心概念，这些概念支撑了深度学习，并详细分析了多头自注意力机制和反向传播算法。我们的主要贡献是为Transformer提出一个普遍逼近定理：证明一个包含一个自注意力层后跟一个具有ReLU激活的位置-wise前馈网络的单层Transformer，可以任意精度地逼近紧致领域上的任意连续序列到序列映射。我们提供了正式陈述并给出了完整证明。最后，我们展示了案例研究以说明该结果的实际影响。我们的发现推进了对Transformer模型理论的理解，并有助于理论与实践之间的鸿沟。 

---
# When and Where do Data Poisons Attack Textual Inversion? 

**Title (ZH)**: 何时何地数据毒药攻击文本反转？ 

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong  

**Link**: [PDF](https://arxiv.org/pdf/2507.10578)  

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: this http URL Data: this http URL 

**Abstract (ZH)**: 中毒攻击对扩散模型（DMs）的鲁棒性构成重大挑战。在本文中，我们系统分析了中毒攻击如何影响文本反转（TI），这是一种广泛使用的DM个性化技术。我们首先介绍了语义敏感图，这是一种新型的可视化方法，用于展示中毒对文本嵌入的影响。其次，我们发现并实验证明了DMs在时间步具有非均匀的学习行为，重点关注低噪声样本。中毒攻击继承了这一偏差，并在较早的时间步注入敌对信号。最后，我们观察到敌对信号会将学习过程引导至与训练数据中相关概念区域无关的方向，破坏TI过程。基于这些见解，我们提出了一种新的防御机制——安全区训练（SZT），其包含三个关键组件：（1）JPEG压缩以削弱高频毒信号，（2）在TI训练期间限制使用高时间步以避免低时间步的敌对信号，（3）损失屏蔽以限制学习到相关区域。广泛的实验表明，SZT显著增强了TI在所有中毒攻击下的鲁棒性，并且改善了生成质量，超越了先前发布的防御措施。代码：详见链接。数据：详见链接。 

---
# Enhancing Cross Entropy with a Linearly Adaptive Loss Function for Optimized Classification Performance 

**Title (ZH)**: 改进交叉熵通过线性自适应损失函数优化分类性能 

**Authors**: Jae Wan Shim  

**Link**: [PDF](https://arxiv.org/pdf/2507.10574)  

**Abstract**: We propose the Linearly Adaptive Cross Entropy Loss function. This is a novel measure derived from the information theory. In comparison to the standard cross entropy loss function, the proposed one has an additional term that depends on the predicted probability of the true class. This feature serves to enhance the optimization process in classification tasks involving one-hot encoded class labels. The proposed one has been evaluated on a ResNet-based model using the CIFAR-100 dataset. Preliminary results show that the proposed one consistently outperforms the standard cross entropy loss function in terms of classification accuracy. Moreover, the proposed one maintains simplicity, achieving practically the same efficiency to the traditional cross entropy loss. These findings suggest that our approach could broaden the scope for future research into loss function design. 

**Abstract (ZH)**: 我们提出线性自适应交叉熵损失函数。这是一种源自信息理论的新型度量。与标准交叉熵损失函数相比，所提出的方法包含一个额外项，该项依赖于真实类别的预测概率。该特性有助于在使用独热编码类标签的分类任务中优化过程。所提出的方法在基于ResNet的模型和CIFAR-100数据集上进行了评估。初步结果显示，所提出的方法在分类准确率上始终优于标准交叉熵损失函数。此外，所提出的方法保持了简洁性，达到与传统交叉熵损失相似的效率。这些发现表明，我们的方法可能为未来关于损失函数设计的研究拓宽视野。 

---
# Tool-to-Tool Matching Analysis Based Difference Score Computation Methods for Semiconductor Manufacturing 

**Title (ZH)**: 基于工具到工具匹配的差异得分计算方法研究（半导体制造） 

**Authors**: Sameera Bharadwaja H., Siddhrath Jandial, Shashank S. Agashe, Rajesh Kumar Reddy Moore, Youngkwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.10564)  

**Abstract**: We consider the problem of tool-to-tool matching (TTTM), also called, chamber matching in the context of a semiconductor manufacturing equipment. Traditional TTTM approaches utilize static configuration data or depend on a golden reference which are difficult to obtain in a commercial manufacturing line. Further, existing methods do not extend very well to a heterogeneous setting, where equipment are of different make-and-model, sourced from different equipment vendors. We propose novel TTTM analysis pipelines to overcome these issues. We hypothesize that a mismatched equipment would have higher variance and/or higher number of modes in the data. Our best univariate method achieves a correlation coefficient >0.95 and >0.5 with the variance and number of modes, respectively showing that the proposed methods are effective. Also, the best multivariate method achieves a correlation coefficient >0.75 with the top-performing univariate methods, showing its effectiveness. Finally, we analyze the sensitivity of the multivariate algorithms to the algorithm hyper-parameters. 

**Abstract (ZH)**: 半导体制造设备中工具到工具匹配问题分析 

---
# A Biomimetic Way for Coral-Reef-Inspired Swarm Intelligence for Carbon-Neutral Wastewater Treatment 

**Title (ZH)**: 珊瑚礁启发的 swarm 智能的生物模拟方法及其在碳中和废水处理中的应用 

**Authors**: Antonis Messinis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10563)  

**Abstract**: With increasing wastewater rates, achieving energy-neutral purification is challenging. We introduce a coral-reef-inspired Swarm Interaction Network for carbon-neutral wastewater treatment, combining morphogenetic abstraction with multi-task carbon awareness. Scalability stems from linear token complexity, mitigating the energy-removal problem. Compared with seven baselines, our approach achieves 96.7\% removal efficiency, 0.31~kWh~m$^{-3}$ energy consumption, and 14.2~g~m$^{-3}$ CO$_2$ emissions. Variance analysis demonstrates robustness under sensor drift. Field scenarios--insular lagoons, brewery spikes, and desert greenhouses--show potential diesel savings of up to 22\%. However, data-science staffing remains an impediment. Future work will integrate AutoML wrappers within the project scope, although governance restrictions pose interpretability challenges that require further visual analytics. 

**Abstract (ZH)**: 基于珊瑚礁启发的 Swarm Interaction Network 促进碳中和废水处理：结合形态发生抽象与多任务碳意识 

---
# Collaboration Promotes Group Resilience in Multi-Agent AI 

**Title (ZH)**: 多智能体AI中的合作促进群体韧性 

**Authors**: Sarah Keren, Matthias Gerstgrasser, Ofir Abu, Jeffrey Rosenschein  

**Link**: [PDF](https://arxiv.org/pdf/2111.06614)  

**Abstract**: To effectively operate in various dynamic scenarios, RL agents must be resilient to unexpected changes in their environment. Previous work on this form of resilience has focused on single-agent settings. In this work, we introduce and formalize a multi-agent variant of resilience, which we term group resilience. We further hypothesize that collaboration with other agents is key to achieving group resilience; collaborating agents adapt better to environmental perturbations in multi-agent reinforcement learning (MARL) settings. We test our hypothesis empirically by evaluating different collaboration protocols and examining their effect on group resilience. Our experiments show that all the examined collaborative approaches achieve higher group resilience than their non-collaborative counterparts. 

**Abstract (ZH)**: 为了在各种动态场景中有效运作，RL代理必须对其环境中的意外变化具有弹性。先前对此类弹性的研究主要集中在单代理设置上。在此项工作中，我们引入并形式化了一个多代理弹性变体，我们称之为群体弹性。我们进一步假设与其他代理的合作对于实现群体弹性至关重要；在多代理强化学习（MARL）设置中，合作代理更能适应环境扰动。我们通过评估不同的合作协议并检查它们对群体弹性的影響来实证测试我们的假设。我们的实验表明，所有检查的合作方法在群体弹性方面都优于其非合作对应方法。 

---
