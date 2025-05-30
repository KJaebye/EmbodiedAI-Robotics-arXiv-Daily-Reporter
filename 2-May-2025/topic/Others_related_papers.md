# Forward kinematics of a general Stewart-Gough platform by elimination templates 

**Title (ZH)**: 一般Stewart-Gough平台的前向运动学求解方法——消元模板 Approach to Forward Kinematics of a General Stewart-Gough Platform Using Elimination Templates 

**Authors**: Evgeniy Martyushev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00634)  

**Abstract**: The paper proposes an efficient algebraic solution to the problem of forward kinematics for a general Stewart-Gough platform. The problem involves determining all possible postures of a mobile platform connected to a fixed base by six legs, given the leg lengths and the internal geometries of the platform and base. The problem is known to have 40 solutions (whether real or complex). The proposed algorithm consists of three main steps: (i) a specific sparse matrix of size 293x362 (the elimination template) is constructed from the coefficients of the polynomial system describing the platform's kinematics; (ii) the PLU decomposition of this matrix is used to construct a pair of 69x69 matrices; (iii) all 40 solutions (including complex ones) are obtained by computing the generalized eigenvectors of this matrix pair. The proposed algorithm is numerically robust, computationally efficient, and straightforward to implement - requiring only standard linear algebra decompositions. MATLAB, Julia, and Python implementations of the algorithm will be made publicly available. 

**Abstract (ZH)**: 本文提出了一种有效地求解通用Stewart-Gough平台前向运动学问题的代数解决方案。该问题涉及在给出腿长以及平台和基座的内部几何结构的情况下，确定连接在固定基座上的移动平台的所有可能姿态。该问题已知有40个解（无论是实数解还是复数解）。所提出的算法包括三个主要步骤：（i）从描述平台运动学的多项式系统系数中构建一个具体的稀疏矩阵（大小为293x362，称为消元模板）；（ii）使用该矩阵的PLU分解构造一对69x69矩阵；（iii）通过计算这对矩阵的广义特征向量来获得所有40个解（包括复数解）。所提出的算法在数值上稳健、计算上高效且易于实现，仅需标准线性代数分解。算法的MATLAB、Julia和Python实现将被公开发布。 

---
# TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching 

**Title (ZH)**: TeLoGraF: 基于图编码流匹配的时间逻辑规划 

**Authors**: Yue Meng, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00562)  

**Abstract**: Learning to solve complex tasks with signal temporal logic (STL) specifications is crucial to many real-world applications. However, most previous works only consider fixed or parametrized STL specifications due to the lack of a diverse STL dataset and encoders to effectively extract temporal logic information for downstream tasks. In this paper, we propose TeLoGraF, Temporal Logic Graph-encoded Flow, which utilizes Graph Neural Networks (GNN) encoder and flow-matching to learn solutions for general STL specifications. We identify four commonly used STL templates and collect a total of 200K specifications with paired demonstrations. We conduct extensive experiments in five simulation environments ranging from simple dynamical models in the 2D space to high-dimensional 7DoF Franka Panda robot arm and Ant quadruped navigation. Results show that our method outperforms other baselines in the STL satisfaction rate. Compared to classical STL planning algorithms, our approach is 10-100X faster in inference and can work on any system dynamics. Besides, we show our graph-encoding method's capability to solve complex STLs and robustness to out-distribution STL specifications. Code is available at this https URL 

**Abstract (ZH)**: 使用信号时序逻辑（STL）规范学习解决复杂任务：TeLoGraF，时序逻辑图编码流 

---
# Safety-Critical Traffic Simulation with Guided Latent Diffusion Model 

**Title (ZH)**: 受指导的潜在扩散模型在交通安全临界交通模拟中的应用 

**Authors**: Mingxing Peng, Ruoyu Yao, Xusen Guo, Yuting Xie, Xianda Chen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00515)  

**Abstract**: Safety-critical traffic simulation plays a crucial role in evaluating autonomous driving systems under rare and challenging scenarios. However, existing approaches often generate unrealistic scenarios due to insufficient consideration of physical plausibility and suffer from low generation efficiency. To address these limitations, we propose a guided latent diffusion model (LDM) capable of generating physically realistic and adversarial safety-critical traffic scenarios. Specifically, our model employs a graph-based variational autoencoder (VAE) to learn a compact latent space that captures complex multi-agent interactions while improving computational efficiency. Within this latent space, the diffusion model performs the denoising process to produce realistic trajectories. To enable controllable and adversarial scenario generation, we introduce novel guidance objectives that drive the diffusion process toward producing adversarial and behaviorally realistic driving behaviors. Furthermore, we develop a sample selection module based on physical feasibility checks to further enhance the physical plausibility of the generated scenarios. Extensive experiments on the nuScenes dataset demonstrate that our method achieves superior adversarial effectiveness and generation efficiency compared to existing baselines while maintaining a high level of realism. Our work provides an effective tool for realistic safety-critical scenario simulation, paving the way for more robust evaluation of autonomous driving systems. 

**Abstract (ZH)**: 安全性关键交通模拟在评估无人驾驶系统在罕见和挑战性场景下的性能中发挥着关键作用。然而，现有方法往往由于物理合理性考虑不足而生成不现实的场景，并且生成效率较低。为解决这些限制，我们提出了一种指导下的潜变量扩散模型（LDM），能够生成物理上现实且具有对抗性的安全关键交通场景。具体而言，我们的模型通过图基变分自编码器（VAE）学习一个紧凑的潜在空间，以捕捉复杂的多智能体交互并提高计算效率。在该潜在空间中，扩散模型执行去噪过程以生成现实的轨迹。为了实现可控和对抗性场景生成，我们引入了新的指导目标，以驱动扩散过程生成对抗性和行为上现实的驾驶行为。此外，我们开发了一种基于物理可行性检查的样本选择模块，以进一步增强生成场景的物理合理性。在nuScenes数据集上的 extensive 实验表明，我们的方法在对抗性效果和生成效率上均优于现有基线，同时保持了高水平的现实性。我们的工作提供了一种有效的工具，用于现实的安全关键场景模拟，为无人驾驶系统的更稳健评估铺平了道路。 

---
# Optimal Interactive Learning on the Job via Facility Location Planning 

**Title (ZH)**: 基于设施位置规划的最优在职互动学习 

**Authors**: Shivam Vats, Michelle Zhao, Patrick Callaghan, Mingxi Jia, Maxim Likhachev, Oliver Kroemer, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.00490)  

**Abstract**: Collaborative robots must continually adapt to novel tasks and user preferences without overburdening the user. While prior interactive robot learning methods aim to reduce human effort, they are typically limited to single-task scenarios and are not well-suited for sustained, multi-task collaboration. We propose COIL (Cost-Optimal Interactive Learning) -- a multi-task interaction planner that minimizes human effort across a sequence of tasks by strategically selecting among three query types (skill, preference, and help). When user preferences are known, we formulate COIL as an uncapacitated facility location (UFL) problem, which enables bounded-suboptimal planning in polynomial time using off-the-shelf approximation algorithms. We extend our formulation to handle uncertainty in user preferences by incorporating one-step belief space planning, which uses these approximation algorithms as subroutines to maintain polynomial-time performance. Simulated and physical experiments on manipulation tasks show that our framework significantly reduces the amount of work allocated to the human while maintaining successful task completion. 

**Abstract (ZH)**: 协作机器人必须不断适应新颖任务和用户偏好，同时避免过度负担用户。虽然之前的交互式机器人学习方法旨在减少人类努力，但它们通常局限于单一任务场景，并不适用于持续的多任务协作。我们提出COIL（成本最优交互学习）——一种多任务交互规划器，通过战略性地选择三种查询类型（技能、偏好和帮助），在一系列任务中最小化人类努力。当用户偏好已知时，我们将COIL建模为未容量限制的设施定位（UFL）问题，这使得使用现成的近似算法可以在多项式时间内进行有界次优规划。我们通过引入一步信念空间规划扩展了我们的建模方式，利用这些近似算法作为子例行程序来保持多项式时间性能以处理用户偏好不确定性。在操作任务上的模拟和物理实验表明，我们的框架在保持任务成功完成的同时显著减少了分配给人类的工作量。 

---
# AI2-Active Safety: AI-enabled Interaction-aware Active Safety Analysis with Vehicle Dynamics 

**Title (ZH)**: AI2-主动安全：基于车辆动力学的智能交互感知主动安全分析 

**Authors**: Keshu Wu, Zihao Li, Sixu Li, Xinyue Ye, Dominique Lord, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00322)  

**Abstract**: This paper introduces an AI-enabled, interaction-aware active safety analysis framework that accounts for groupwise vehicle interactions. Specifically, the framework employs a bicycle model-augmented with road gradient considerations-to accurately capture vehicle dynamics. In parallel, a hypergraph-based AI model is developed to predict probabilistic trajectories of ambient traffic. By integrating these two components, the framework derives vehicle intra-spacing over a 3D road surface as the solution of a stochastic ordinary differential equation, yielding high-fidelity surrogate safety measures such as time-to-collision (TTC). To demonstrate its effectiveness, the framework is analyzed using stochastic numerical methods comprising 4th-order Runge-Kutta integration and AI inference, generating probability-weighted high-fidelity TTC (HF-TTC) distributions that reflect complex multi-agent maneuvers and behavioral uncertainties. Evaluated with HF-TTC against traditional constant-velocity TTC and non-interaction-aware approaches on highway datasets, the proposed framework offers a systematic methodology for active safety analysis with enhanced potential for improving safety perception in complex traffic environments. 

**Abstract (ZH)**: 基于AI赋能和交互感知的群体车辆交互智能安全分析框架 

---
# FedEMA: Federated Exponential Moving Averaging with Negative Entropy Regularizer in Autonomous Driving 

**Title (ZH)**: FedEMA：自主驾驶中的负熵正则化指数移动平均 federated learning 方法 

**Authors**: Wei-Bin Kou, Guangxu Zhu, Bingyang Cheng, Shuai Wang, Ming Tang, Yik-Chung Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00318)  

**Abstract**: Street Scene Semantic Understanding (denoted as S3U) is a crucial but complex task for autonomous driving (AD) vehicles. Their inference models typically face poor generalization due to domain-shift. Federated Learning (FL) has emerged as a promising paradigm for enhancing the generalization of AD models through privacy-preserving distributed learning. However, these FL AD models face significant temporal catastrophic forgetting when deployed in dynamically evolving environments, where continuous adaptation causes abrupt erosion of historical knowledge. This paper proposes Federated Exponential Moving Average (FedEMA), a novel framework that addresses this challenge through two integral innovations: (I) Server-side model's historical fitting capability preservation via fusing current FL round's aggregation model and a proposed previous FL round's exponential moving average (EMA) model; (II) Vehicle-side negative entropy regularization to prevent FL models' possible overfitting to EMA-introduced temporal patterns. Above two strategies empower FedEMA a dual-objective optimization that balances model generalization and adaptability. In addition, we conduct theoretical convergence analysis for the proposed FedEMA. Extensive experiments both on Cityscapes dataset and Camvid dataset demonstrate FedEMA's superiority over existing approaches, showing 7.12% higher mean Intersection-over-Union (mIoU). 

**Abstract (ZH)**: 基于街道场景语义理解的联邦指数移动平均 Federated Exponential Moving Average for Street Scene Semantic Understanding 

---
# PSN Game: Game-theoretic Planning via a Player Selection Network 

**Title (ZH)**: PSN游戏：基于玩家选择网络的游戏理论规划 

**Authors**: Tianyu Qiu, Eric Ouano, Fernando Palafox, Christian Ellis, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2505.00213)  

**Abstract**: While game-theoretic planning frameworks are effective at modeling multi-agent interactions, they require solving optimization problems with hundreds or thousands of variables, resulting in long computation times that limit their use in large-scale, real-time systems. To address this issue, we propose PSN Game: a novel game-theoretic planning framework that reduces runtime by learning a Player Selection Network (PSN). A PSN outputs a player selection mask that distinguishes influential players from less relevant ones, enabling the ego player to solve a smaller, masked game involving only selected players. By reducing the number of variables in the optimization problem, PSN directly lowers computation time. The PSN Game framework is more flexible than existing player selection methods as it i) relies solely on observations of players' past trajectories, without requiring full state, control, or other game-specific information; and ii) requires no online parameter tuning. We train PSNs in an unsupervised manner using a differentiable dynamic game solver, with reference trajectories from full-player games guiding the learning. Experiments in both simulated scenarios and human trajectory datasets demonstrate that i) PSNs outperform baseline selection methods in trajectory smoothness and length, while maintaining comparable safety and achieving a 10x speedup in runtime; and ii) PSNs generalize effectively to real-world scenarios without fine-tuning. By selecting only the most relevant players for decision-making, PSNs offer a general mechanism for reducing planning complexity that can be seamlessly integrated into existing multi-agent planning frameworks. 

**Abstract (ZH)**: PSN Game:一种通过学习玩家选择网络减少运行时的游戏理论规划框架 

---
# Position: AI Competitions Provide the Gold Standard for Empirical Rigor in GenAI Evaluation 

**Title (ZH)**: 位置：AI竞赛提供了生成式AI评估的黄金标准 empirical rigor。 

**Authors**: D. Sculley, Will Cukierski, Phil Culliton, Sohier Dane, Maggie Demkin, Ryan Holbrook, Addison Howard, Paul Mooney, Walter Reade, Megan Risdal, Nate Keating  

**Link**: [PDF](https://arxiv.org/pdf/2505.00612)  

**Abstract**: In this position paper, we observe that empirical evaluation in Generative AI is at a crisis point since traditional ML evaluation and benchmarking strategies are insufficient to meet the needs of evaluating modern GenAI models and systems. There are many reasons for this, including the fact that these models typically have nearly unbounded input and output spaces, typically do not have a well defined ground truth target, and typically exhibit strong feedback loops and prediction dependence based on context of previous model outputs. On top of these critical issues, we argue that the problems of {\em leakage} and {\em contamination} are in fact the most important and difficult issues to address for GenAI evaluations. Interestingly, the field of AI Competitions has developed effective measures and practices to combat leakage for the purpose of counteracting cheating by bad actors within a competition setting. This makes AI Competitions an especially valuable (but underutilized) resource. Now is time for the field to view AI Competitions as the gold standard for empirical rigor in GenAI evaluation, and to harness and harvest their results with according value. 

**Abstract (ZH)**: 基于生成型AI的实证评估处于危机点：泄漏和污染问题的挑战与机遇 

---
# Rule-based Classifier Models 

**Title (ZH)**: 基于规则的分类器模型 

**Authors**: Cecilia Di Florio, Huimin Dong, Antonino Rotolo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00474)  

**Abstract**: We extend the formal framework of classifier models used in the legal domain. While the existing classifier framework characterises cases solely through the facts involved, legal reasoning fundamentally relies on both facts and rules, particularly the ratio decidendi. This paper presents an initial approach to incorporating sets of rules within a classifier. Our work is built on the work of Canavotto et al. (2023), which has developed the rule-based reason model of precedential constraint within a hierarchy of factors. We demonstrate how decisions for new cases can be inferred using this enriched rule-based classifier framework. Additionally, we provide an example of how the time element and the hierarchy of courts can be used in the new classifier framework. 

**Abstract (ZH)**: 我们扩展了在法律领域使用的分类模型的形式框架。现有的分类框架仅通过案件事实来表征案例，而法律推理本质上依赖于事实和规则，特别是判例中的判决理由。本文提出了一种初步方法，在分类模型中纳入规则集。我们的工作基于Canavotto等人（2023）的工作，该工作在因素层次结构中发展了基于规则的判例约束推理模型。我们展示了如何使用这种增强的基于规则的分类框架来推断新案件的判决。此外，我们提供了如何在新分类框架中使用时间元素和法院层级结构的例子。 

---
# ScaleTrack: Scaling and back-tracking Automated GUI Agents 

**Title (ZH)**: ScaleTrack: 自动化GUI代理的缩放与反向追踪 

**Authors**: Jing Huang, Zhixiong Zeng, Wenkang Han, Yufeng Zhong, Liming Zheng, Shuai Fu, Jingyuan Chen, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00416)  

**Abstract**: Automated GUI agents aims to facilitate user interaction by automatically performing complex tasks in digital environments, such as web, mobile, desktop devices. It receives textual task instruction and GUI description to generate executable actions (\emph{e.g.}, click) and operation boxes step by step. Training a GUI agent mainly involves grounding and planning stages, in which the GUI grounding focuses on finding the execution coordinates according to the task, while the planning stage aims to predict the next action based on historical actions. However, previous work suffers from the limitations of insufficient training data for GUI grounding, as well as the ignorance of backtracking historical behaviors for GUI planning. To handle the above challenges, we propose ScaleTrack, a training framework by scaling grounding and backtracking planning for automated GUI agents. We carefully collected GUI samples of different synthesis criterions from a wide range of sources, and unified them into the same template for training GUI grounding models. Moreover, we design a novel training strategy that predicts the next action from the current GUI image, while also backtracking the historical actions that led to the GUI image. In this way, ScaleTrack explains the correspondence between GUI images and actions, which effectively describes the evolution rules of the GUI environment. Extensive experimental results demonstrate the effectiveness of ScaleTrack. Data and code will be available at url. 

**Abstract (ZH)**: 自动GUI代理旨在通过在数字环境中（如网络、移动设备、桌面设备）自动执行复杂任务来简化用户交互。它接收文本任务指令和GUI描述，逐步生成可执行动作（例如点击）和操作框。训练GUI代理主要涉及语义接地和规划阶段，其中的GUI语义接地专注于根据任务找到执行坐标，而规划阶段旨在基于历史动作预测下一个动作。然而，先前的工作受限于GUI语义接地不足的训练数据，以及在GUI规划中忽视了回溯历史行为。为了应对上述挑战，我们提出ScaleTrack，一种通过扩展语义接地和回溯规划来训练自动GUI代理的框架。我们从广泛的数据源中精心收集了不同合成标准的GUI样本，并将它们统一到同一个模板以训练GUI语义接地模型。此外，我们设计了一种新的训练策略，从当前的GUI图像预测下一个动作，同时也回溯导致该GUI图像的历史动作。通过这种方式，ScaleTrack解释了GUI图像与动作之间的对应关系，有效地描述了GUI环境的演变规则。广泛的实验结果证明了ScaleTrack的有效性。数据和代码将在此URL上提供。 

---
# CognitionNet: A Collaborative Neural Network for Play Style Discovery in Online Skill Gaming Platform 

**Title (ZH)**: 认知网络：在线技能游戏平台上玩法风格发现的协作神经网络 

**Authors**: Rukma Talwadker, Surajit Chakrabarty, Aditya Pareek, Tridib Mukherjee, Deepak Saini  

**Link**: [PDF](https://arxiv.org/pdf/2505.00325)  

**Abstract**: Games are one of the safest source of realizing self-esteem and relaxation at the same time. An online gaming platform typically has massive data coming in, e.g., in-game actions, player moves, clickstreams, transactions etc. It is rather interesting, as something as simple as data on gaming moves can help create a psychological imprint of the user at that moment, based on her impulsive reactions and response to a situation in the game. Mining this knowledge can: (a) immediately help better explain observed and predicted player behavior; and (b) consequently propel deeper understanding towards players' experience, growth and protection. To this effect, we focus on discovery of the "game behaviours" as micro-patterns formed by continuous sequence of games and the persistent "play styles" of the players' as a sequence of such sequences on an online skill gaming platform for Rummy. We propose a two stage deep neural network, CognitionNet. The first stage focuses on mining game behaviours as cluster representations in a latent space while the second aggregates over these micro patterns to discover play styles via a supervised classification objective around player engagement. The dual objective allows CognitionNet to reveal several player psychology inspired decision making and tactics. To our knowledge, this is the first and one-of-its-kind research to fully automate the discovery of: (i) player psychology and game tactics from telemetry data; and (ii) relevant diagnostic explanations to players' engagement predictions. The collaborative training of the two networks with differential input dimensions is enabled using a novel formulation of "bridge loss". The network plays pivotal role in obtaining homogeneous and consistent play style definitions and significantly outperforms the SOTA baselines wherever applicable. 

**Abstract (ZH)**: 在线游戏平台中通过深度神经网络发现玩家心理与游戏策略自动化的研究 

---
# DeCo: Defect-Aware Modeling with Contrasting Matching for Optimizing Task Assignment in Online IC Testing 

**Title (ZH)**: DeCo: 基于对比匹配的缺陷意识建模方法以优化在线IC测试任务分配 

**Authors**: Lo Pang-Yun Ting, Yu-Hao Chiang, Yi-Tung Tsai, Hsu-Chao Lai, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00278)  

**Abstract**: In the semiconductor industry, integrated circuit (IC) processes play a vital role, as the rising complexity and market expectations necessitate improvements in yield. Identifying IC defects and assigning IC testing tasks to the right engineers improves efficiency and reduces losses. While current studies emphasize fault localization or defect classification, they overlook the integration of defect characteristics, historical failures, and the insights from engineer expertise, which restrains their effectiveness in improving IC handling. To leverage AI for these challenges, we propose DeCo, an innovative approach for optimizing task assignment in IC testing. DeCo constructs a novel defect-aware graph from IC testing reports, capturing co-failure relationships to enhance defect differentiation, even with scarce defect data. Additionally, it formulates defect-aware representations for engineers and tasks, reinforced by local and global structure modeling on the defect-aware graph. Finally, a contrasting-based assignment mechanism pairs testing tasks with QA engineers by considering their skill level and current workload, thus promoting an equitable and efficient job dispatch. Experiments on a real-world dataset demonstrate that DeCo achieves the highest task-handling success rates in different scenarios, exceeding 80\%, while also maintaining balanced workloads on both scarce or expanded defect data. Moreover, case studies reveal that DeCo can assign tasks to potentially capable engineers, even for their unfamiliar defects, highlighting its potential as an AI-driven solution for the real-world IC failure analysis and task handling. 

**Abstract (ZH)**: 半导体行业中，集成电路（IC）工艺发挥着关键作用，随着复杂性的提高和市场预期的增强，提高良率变得尤为重要。通过识别IC缺陷并将IC测试任务分配给合适的工程师可以提高效率并减少损失。当前的研究侧重于故障定位或缺陷分类，但忽视了缺陷特征、历史失效及工程师 expertise 的整合，这限制了它们在改善IC处理方面的有效性。为了应对这些挑战，我们提出了DeCo，这是一种优化IC测试任务分配的新颖方法。DeCo从IC测试报告中构建了一个新颖的缺陷感知图，捕获共失效关系以增强缺陷区分能力，即使在缺陷数据稀缺的情况下也是如此。此外，它为工程师和任务制定了缺陷感知表示，并通过缺陷感知图上的局部和全局结构建模增强了这些表示。最后，通过考虑工程师的技能水平和当前工作负荷来实现对比驱动的任务分配机制，从而促进公平且高效的职责分配。实验结果表明，DeCo在不同场景下的任务处理成功率最高，超过80%，同时在稀缺或扩展的缺陷数据下也能保持均衡的工作负荷。此外，案例研究显示，DeCo能够将任务分配给潜在有能力但不熟悉该缺陷的工程师，突显了其作为AI驱动解决方案在实际IC失效分析和任务处理中的潜力。 

---
# Real-World Gaps in AI Governance Research 

**Title (ZH)**: AI治理研究中的现实差距 

**Authors**: Ilan Strauss, Isobel Moure, Tim O'Reilly, Sruly Rosenblat  

**Link**: [PDF](https://arxiv.org/pdf/2505.00174)  

**Abstract**: Drawing on 1,178 safety and reliability papers from 9,439 generative AI papers (January 2020 - March 2025), we compare research outputs of leading AI companies (Anthropic, Google DeepMind, Meta, Microsoft, and OpenAI) and AI universities (CMU, MIT, NYU, Stanford, UC Berkeley, and University of Washington). We find that corporate AI research increasingly concentrates on pre-deployment areas -- model alignment and testing & evaluation -- while attention to deployment-stage issues such as model bias has waned. Significant research gaps exist in high-risk deployment domains, including healthcare, finance, misinformation, persuasive and addictive features, hallucinations, and copyright. Without improved observability into deployed AI, growing corporate concentration could deepen knowledge deficits. We recommend expanding external researcher access to deployment data and systematic observability of in-market AI behaviors. 

**Abstract (ZH)**: 基于2020年1月至2025年3月的9439篇生成式AI论文中的1178篇安全与可靠性论文，我们比较了Anthropic、Google DeepMind、Meta、Microsoft和OpenAI等领先AI公司以及CMU、MIT、NYU、Stanford、UC Berkeley和University of Washington等AI大学的研究成果。研究发现，企业AI研究越来越集中在部署前领域（如模型对齐和测试与评估），而对部署阶段问题（如模型偏差）的关注度下降。在包括医疗、金融、 misinformation、说服性和成瘾性功能、幻觉和版权在内的高风险部署领域，存在显著的研究缺口。缺乏对部署中AI的增强可观察性可能导致知识差距加剧。我们建议扩大外部研究人员对部署数据的访问权限，并加强对市场中AI行为的系统观察。 

---
# First Order Logic with Fuzzy Semantics for Describing and Recognizing Nerves in Medical Images 

**Title (ZH)**: 具有模糊语义的一阶逻辑描述与识别医疗图像中的神经结构 

**Authors**: Isabelle Bloch, Enzo Bonnot, Pietro Gori, Giammarco La Barbera, Sabine Sarnacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.00173)  

**Abstract**: This article deals with the description and recognition of fiber bundles, in particular nerves, in medical images, based on the anatomical description of the fiber trajectories. To this end, we propose a logical formalization of this anatomical knowledge. The intrinsically imprecise description of nerves, as found in anatomical textbooks, leads us to propose fuzzy semantics combined with first-order logic. We define a language representing spatial entities, relations between these entities and quantifiers. A formula in this language is then a formalization of the natural language description. The semantics are given by fuzzy representations in a concrete domain and satisfaction degrees of relations. Based on this formalization, a spatial reasoning algorithm is proposed for segmentation and recognition of nerves from anatomical and diffusion magnetic resonance images, which is illustrated on pelvic nerves in pediatric imaging, enabling surgeons to plan surgery. 

**Abstract (ZH)**: 基于解剖学轨迹描述的纤维束（特别是神经）在医学图像中的描述与识别：一种层次化逻辑形式化方法及其在儿科盆腔神经影像中的应用 

---
# Wasserstein Policy Optimization 

**Title (ZH)**: Wasserstein 政策优化 

**Authors**: David Pfau, Ian Davies, Diana Borsa, Joao G. M. Araujo, Brendan Tracey, Hado van Hasselt  

**Link**: [PDF](https://arxiv.org/pdf/2505.00663)  

**Abstract**: We introduce Wasserstein Policy Optimization (WPO), an actor-critic algorithm for reinforcement learning in continuous action spaces. WPO can be derived as an approximation to Wasserstein gradient flow over the space of all policies projected into a finite-dimensional parameter space (e.g., the weights of a neural network), leading to a simple and completely general closed-form update. The resulting algorithm combines many properties of deterministic and classic policy gradient methods. Like deterministic policy gradients, it exploits knowledge of the gradient of the action-value function with respect to the action. Like classic policy gradients, it can be applied to stochastic policies with arbitrary distributions over actions -- without using the reparameterization trick. We show results on the DeepMind Control Suite and a magnetic confinement fusion task which compare favorably with state-of-the-art continuous control methods. 

**Abstract (ZH)**: Wasserstein策略优化：连续动作空间中的演员-评论家算法 

---
# OmicsCL: Unsupervised Contrastive Learning for Cancer Subtype Discovery and Survival Stratification 

**Title (ZH)**: OmicsCL：无监督对比学习在癌症亚型发现和生存分层中的应用 

**Authors**: Atahan Karagoz  

**Link**: [PDF](https://arxiv.org/pdf/2505.00650)  

**Abstract**: Unsupervised learning of disease subtypes from multi-omics data presents a significant opportunity for advancing personalized medicine. We introduce OmicsCL, a modular contrastive learning framework that jointly embeds heterogeneous omics modalities-such as gene expression, DNA methylation, and miRNA expression-into a unified latent space. Our method incorporates a survival-aware contrastive loss that encourages the model to learn representations aligned with survival-related patterns, without relying on labeled outcomes. Evaluated on the TCGA BRCA dataset, OmicsCL uncovers clinically meaningful clusters and achieves strong unsupervised concordance with patient survival. The framework demonstrates robustness across hyperparameter configurations and can be tuned to prioritize either subtype coherence or survival stratification. Ablation studies confirm that integrating survival-aware loss significantly enhances the predictive power of learned embeddings. These results highlight the promise of contrastive objectives for biological insight discovery in high-dimensional, heterogeneous omics data. 

**Abstract (ZH)**: 无监督学习多组学数据中的疾病亚型具有推进个性化医学的重大潜力：OmicsCL，一个联合嵌入多组学模态的模块化对比学习框架 

---
# Deep Learning Assisted Outer Volume Removal for Highly-Accelerated Real-Time Dynamic MRI 

**Title (ZH)**: 深度学习辅助快速实时动态MRI外边缘体积去除 

**Authors**: Merve Gülle, Sebastian Weingärtner, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.00643)  

**Abstract**: Real-time (RT) dynamic MRI plays a vital role in capturing rapid physiological processes, offering unique insights into organ motion and function. Among these applications, RT cine MRI is particularly important for functional assessment of the heart with high temporal resolution. RT imaging enables free-breathing, ungated imaging of cardiac motion, making it a crucial alternative for patients who cannot tolerate conventional breath-hold, ECG-gated acquisitions. However, achieving high acceleration rates in RT cine MRI is challenging due to aliasing artifacts from extra-cardiac tissues, particularly at high undersampling factors. In this study, we propose a novel outer volume removal (OVR) method to address this challenge by eliminating aliasing contributions from non-cardiac regions in a post-processing framework. Our approach estimates the outer volume signal for each timeframe using composite temporal images from time-interleaved undersampling patterns, which inherently contain pseudo-periodic ghosting artifacts. A deep learning (DL) model is trained to identify and remove these artifacts, producing a clean outer volume estimate that is subsequently subtracted from the corresponding k-space data. The final reconstruction is performed with a physics-driven DL (PD-DL) method trained using an OVR-specific loss function to restore high spatio-temporal resolution images. Experimental results show that the proposed method at high accelerations achieves image quality that is visually comparable to clinical baseline images, while outperforming conventional reconstruction techniques, both qualitatively and quantitatively. The proposed approach provides a practical and effective solution for artifact reduction in RT cine MRI without requiring acquisition modifications, offering a pathway to higher acceleration rates while preserving diagnostic quality. 

**Abstract (ZH)**: 实时（RT）动态MRI在捕捉快速生理过程方面发挥着重要作用，提供了对器官运动和功能的独特见解。在这些应用中，RT cine MRI特别重要，因为它可以提供高质量的时间分辨率，用于心脏的功能评估。RT成像允许自由呼吸、非门控的心脏运动成像，使其成为不能耐受常规屏气、心电门控采集的患者的重要替代方法。然而，由于额外心脏组织的伪影，在高欠采样因子下实现高加速率在RT cine MRI中具有挑战性。在这项研究中，我们提出了一种新颖的外部体积去除（OVR）方法，通过消除非心脏区域的伪影贡献来解决这一挑战，该方法在后处理框架中实现。我们的方法使用时间交错欠采样模式中的复合时间图像来估计每个时间帧的外部体积信号，这些图像本质上包含伪周期性鬼影伪影。一个深度学习（DL）模型被训练来识别并去除这些伪影，产生一个干净的外部体积估计，并随后从相应的k空间数据中减去。最终重建使用特定于OVR的损失函数进行的物理驱动深度学习（PD-DL）方法来进行，以恢复高空间-时间分辨率图像。实验结果表明，所提出的方法在高加速率下实现了视觉上与临床基线图像相当的图像质量，并在定性和定量上均优于传统重建技术。所提出的方法提供了一种实用且有效的解决方案，可以在无需更改采集的情况下减少RT cine MRI中的伪影，为提高加速率并保持诊断质量提供了途径。 

---
# Fast and Low-Cost Genomic Foundation Models via Outlier Removal 

**Title (ZH)**: 基于离群值去除的快速低成本基因组基础模型 

**Authors**: Haozheng Luo, Chenghao Qiu, Maojiang Su, Zhihan Zhou, Zoe Mehta, Guo Ye, Jerry Yao-Chieh Hu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00598)  

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GERM. Unlike existing GFM benchmarks, GERM offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Empirically, transformer-based models exhibit greater robustness to adversarial perturbations compared to HyenaDNA, highlighting the impact of architectural design on vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. 

**Abstract (ZH)**: 我们提出了第一个针对基因组基础模型（GFMs）的统一对抗性攻击基准——GERM。GERM提供了第一个全面的评估框架，系统性地评估GFMs对对抗性攻击的脆弱性。从方法论上，我们使用四种广泛采用的攻击算法和三种防御策略，评估了五种最先进的GFMs的对抗性鲁棒性。重要的是，我们的基准提供了一个易于访问和全面的框架，用于分析GFMs在模型架构、量化方案和训练数据集方面的脆弱性。实验证据表明，基于变换器的模型在对抗性扰动方面比HyenaDNA表现出更大的鲁棒性，这突显了架构设计对脆弱性的影响。此外，对抗性攻击频繁针对生物学上意义重大的基因组区域，表明这些模型有效地捕获了有意义的序列特征。 

---
# Synthesizing and Identifying Noise Levels in Autonomous Vehicle Camera Radar Datasets 

**Title (ZH)**: 合成和识别自主车辆摄像头雷达数据中的噪声水平 

**Authors**: Mathis Morales, Golnaz Habibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00584)  

**Abstract**: Detecting and tracking objects is a crucial component of any autonomous navigation method. For the past decades, object detection has yielded promising results using neural networks on various datasets. While many methods focus on performance metrics, few projects focus on improving the robustness of these detection and tracking pipelines, notably to sensor failures. In this paper we attempt to address this issue by creating a realistic synthetic data augmentation pipeline for camera-radar Autonomous Vehicle (AV) datasets. Our goal is to accurately simulate sensor failures and data deterioration due to real-world interferences. We also present our results of a baseline lightweight Noise Recognition neural network trained and tested on our augmented dataset, reaching an overall recognition accuracy of 54.4\% on 11 categories across 10086 images and 2145 radar point-clouds. 

**Abstract (ZH)**: 检测与跟踪对象是任何自主导航方法的关键组成部分。在过去几十年中，利用神经网络在各类数据集上进行对象检测取得了令人鼓舞的结果。尽管许多方法关注性能指标，但很少有项目致力于提高这些检测和跟踪管道的鲁棒性，特别是提高其对传感器故障的鲁棒性。在本文中，我们试图通过为基于摄像头-雷达自主车辆（AV）数据集创建一种现实的合成数据增强管道来解决这个问题。我们的目标是准确模拟传感器故障和由于实际干扰引起的数据衰减。我们还介绍了在我们的增强数据集上训练和测试的基础轻量级噪声识别神经网络的结果，该网络在10086张图像和2145个雷达点云的11个类别上达到了整体识别准确率54.4%。 

---
# Voice Cloning: Comprehensive Survey 

**Title (ZH)**: 语音克隆：综述研究 

**Authors**: Hussam Azzuni, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2505.00579)  

**Abstract**: Voice Cloning has rapidly advanced in today's digital world, with many researchers and corporations working to improve these algorithms for various applications. This article aims to establish a standardized terminology for voice cloning and explore its different variations. It will cover speaker adaptation as the fundamental concept and then delve deeper into topics such as few-shot, zero-shot, and multilingual TTS within that context. Finally, we will explore the evaluation metrics commonly used in voice cloning research and related datasets. This survey compiles the available voice cloning algorithms to encourage research toward its generation and detection to limit its misuse. 

**Abstract (ZH)**: 语音克隆在当今数字世界中迅速发展，许多研究者和公司致力于改进这些算法以应用于多种场景。本文旨在建立语音克隆的标准术语，并探索其不同的变体。文章将涵盖说话人适应作为基本概念，进而深入探讨此类情境下的少量学习、零样本学习和多语言TTS等领域。最后，本文将探讨语音克隆研究中常用的评估指标及相关数据集。本文综述了可用的语音克隆算法，以促进其生成和检测研究，限制其不当使用。 

---
# Learning to Learn with Quantum Optimization via Quantum Neural Networks 

**Title (ZH)**: 使用量子神经网络通过量子优化进行学习 

**Authors**: Kuan-Cheng Chen, Hiromichi Matsuyama, Wei-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00561)  

**Abstract**: Quantum Approximate Optimization Algorithms (QAOA) promise efficient solutions to classically intractable combinatorial optimization problems by harnessing shallow-depth quantum circuits. Yet, their performance and scalability often hinge on effective parameter optimization, which remains nontrivial due to rugged energy landscapes and hardware noise. In this work, we introduce a quantum meta-learning framework that combines quantum neural networks, specifically Quantum Long Short-Term Memory (QLSTM) architectures, with QAOA. By training the QLSTM optimizer on smaller graph instances, our approach rapidly generalizes to larger, more complex problems, substantially reducing the number of iterations required for convergence. Through comprehensive benchmarks on Max-Cut and Sherrington-Kirkpatrick model instances, we demonstrate that QLSTM-based optimizers converge faster and achieve higher approximation ratios compared to classical baselines, thereby offering a robust pathway toward scalable quantum optimization in the NISQ era. 

**Abstract (ZH)**: 量子近似优化算法(QAOA)通过利用浅层量子电路有望高效解决经典上难以处理的组合优化问题。然而，其性能和扩展性往往依赖于有效的参数优化，由于复杂的能量景观和硬件噪声，这一过程仍然颇具挑战性。本文提出了一种结合量子神经网络，特别是量子长短期记忆(Quantum Long Short-Term Memory, QLSTM)架构与QAOA的量子元学习框架。通过在较小的图实例上训练QLSTM优化器，我们的方法能够快速泛化到更大、更复杂的优化问题，显著减少收敛所需的迭代次数。通过对最大割(Max-Cut)和施拉热廷格-基克帕克(Sherrington-Kirkpatrick)模型实例进行全面 benchmark，我们展示了基于QLSTM的优化器收敛速度更快，达到更高的近似比，提供了在量子有限采样(NISQ)时代实现可扩展量子优化的稳健途径。 

---
# On the Mechanistic Interpretability of Neural Networks for Causality in Bio-statistics 

**Title (ZH)**: 神经网络在生物统计学因果性中的机理可解释性 

**Authors**: Jean-Baptiste A. Conan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00555)  

**Abstract**: Interpretable insights from predictive models remain critical in bio-statistics, particularly when assessing causality, where classical statistical and machine learning methods often provide inherent clarity. While Neural Networks (NNs) offer powerful capabilities for modeling complex biological data, their traditional "black-box" nature presents challenges for validation and trust in high-stakes health applications. Recent advances in Mechanistic Interpretability (MI) aim to decipher the internal computations learned by these networks. This work investigates the application of MI techniques to NNs within the context of causal inference for bio-statistics.
We demonstrate that MI tools can be leveraged to: (1) probe and validate the internal representations learned by NNs, such as those estimating nuisance functions in frameworks like Targeted Minimum Loss-based Estimation (TMLE); (2) discover and visualize the distinct computational pathways employed by the network to process different types of inputs, potentially revealing how confounders and treatments are handled; and (3) provide methodologies for comparing the learned mechanisms and extracted insights across statistical, machine learning, and NN models, fostering a deeper understanding of their respective strengths and weaknesses for causal bio-statistical analysis. 

**Abstract (ZH)**: 可解释的见解对于生物统计中的预测模型依然至关重要，特别是在评估因果性时，传统统计和机器学习方法往往提供内在的清晰性。虽然神经网络（NNs）能够建模复杂的生物数据，但它们传统的“黑箱”性质在高风险健康应用中的验证和信任方面提出了挑战。最近在机制可解释性（MI）方面的进展旨在解析这些网络所学习的内部计算。本研究探讨了在因果推断的生物统计背景下将MI技术应用于NNs的应用。我们展示了MI工具可以：（1）探究和验证NNs学习的内部表示，如在目标最小损失基于估计（TMLE）框架中估计的害处函数；（2）发现并可视化网络处理不同类型输入时所使用的独特计算路径，可能揭示协变量和治疗措施的处理方式；（3）提供统计、机器学习和NN模型中学习机制和提取见解的比较方法学，促进对其各自优势和劣势的更深层次理解，以支持因果生物统计分析。 

---
# Test-time Correlation Alignment 

**Title (ZH)**: 测试时 correlations 对齐 

**Authors**: Linjing You, Jiabao Lu, Xiayuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00533)  

**Abstract**: Deep neural networks often experience performance drops due to distribution shifts between training and test data. Although domain adaptation offers a solution, privacy concerns restrict access to training data in many real-world scenarios. This restriction has spurred interest in Test-Time Adaptation (TTA), which adapts models using only unlabeled test data. However, current TTA methods still face practical challenges: (1) a primary focus on instance-wise alignment, overlooking CORrelation ALignment (CORAL) due to missing source correlations; (2) complex backpropagation operations for model updating, resulting in overhead computation and (3) domain forgetting.
To address these challenges, we provide a theoretical analysis to investigate the feasibility of Test-time Correlation Alignment (TCA), demonstrating that correlation alignment between high-certainty instances and test instances can enhance test performances with a theoretical guarantee. Based on this, we propose two simple yet effective algorithms: LinearTCA and LinearTCA+. LinearTCA applies a simple linear transformation to achieve both instance and correlation alignment without additional model updates, while LinearTCA+ serves as a plug-and-play module that can easily boost existing TTA methods. Extensive experiments validate our theoretical insights and show that TCA methods significantly outperforms baselines across various tasks, benchmarks and backbones. Notably, LinearTCA improves adaptation accuracy by 5.88% on OfficeHome dataset, while using only 4% maximum GPU memory usage and 0.6% computation time compared to the best baseline TTA method. 

**Abstract (ZH)**: 深层神经网络经常由于训练数据与测试数据分布的变化而遭受性能下降。尽管领域适应提供了一种解决方案，但在许多现实场景中，隐私问题限制了对训练数据的访问。这种限制激发了对测试时适应（TTA）的兴趣，即仅使用未标记的测试数据来适应模型。然而，当前的TTA方法仍然面临一些实际挑战：（1）主要集中在实例级别的对齐，忽略了由于缺少源数据相关性的CORAL方法；（2）模型更新涉及复杂的反向传播操作，导致计算开销增加；（3）领域遗忘。

为解决这些挑战，我们提供了理论分析来探讨测试时相关性对齐（TCA）的可行性，证明了高可信度实例与测试实例之间相关性的对齐可以在理论上保证测试性能的提升。基于此，我们提出了两个简单而有效的方法：LinearTCA和LinearTCA+。LinearTCA通过简单的线性变换实现实例和相关性的对齐，而无需额外的模型更新，LinearTCA+则作为即插即用模块，可以轻松增强现有的TTA方法。广泛的实验证明了我们的理论见解，并表明TCA方法在各种任务、基准和骨干网络上显著优于基线方法。值得注意的是，LinearTCA在OfficeHome数据集上的适应准确性提高了5.88%，同时仅使用最大GPU内存的4%和计算时间的0.6%与最佳基线TTA方法相比。 

---
# Analysis of the vulnerability of machine learning regression models to adversarial attacks using data from 5G wireless networks 

**Title (ZH)**: 基于5G无线网络数据的机器学习回归模型对对抗攻击的脆弱性分析 

**Authors**: Leonid Legashev, Artur Zhigalov, Denis Parfenov  

**Link**: [PDF](https://arxiv.org/pdf/2505.00487)  

**Abstract**: This article describes the process of creating a script and conducting an analytical study of a dataset using the DeepMIMO emulator. An advertorial attack was carried out using the FGSM method to maximize the gradient. A comparison is made of the effectiveness of binary classifiers in the task of detecting distorted data. The dynamics of changes in the quality indicators of the regression model were analyzed in conditions without adversarial attacks, during an adversarial attack and when the distorted data was isolated. It is shown that an adversarial FGSM attack with gradient maximization leads to an increase in the value of the MSE metric by 33% and a decrease in the R2 indicator by 10% on average. The LightGBM binary classifier effectively identifies data with adversarial anomalies with 98% accuracy. Regression machine learning models are susceptible to adversarial attacks, but rapid analysis of network traffic and data transmitted over the network makes it possible to identify malicious activity 

**Abstract (ZH)**: 本文描述了使用DeepMIMO模拟器创建脚本并进行数据集分析性研究的过程。采用FGSM方法执行推销广告攻击以最大化梯度。比较了二元分类器在检测篡改数据任务中的有效性。分析了在无对抗攻击、对抗攻击期间以及隔离篡改数据条件下的回归模型质量指标变化动态。结果显示，具有梯度最大化特征的对抗性FGSM攻击使得MSE指标值平均增加了33%，R2指标降低了10%。LightGBM二元分类器能够以98%的准确率识别对抗性异常数据。回归机器学习模型容易受到对抗攻击的影响，但快速分析网络流量和网络上传输的数据可以识别出恶意活动。 

---
# Per-Domain Generalizing Policies: On Validation Instances and Scaling Behavior 

**Title (ZH)**: 基于域的泛化策略：关于验证实例和扩展行为的研究 

**Authors**: Timo P. Gros, Nicola J. Müller, Daniel Fiser, Isabel Valera, Verena Wolf, Jörg Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.00439)  

**Abstract**: Recent work has shown that successful per-domain generalizing action policies can be learned. Scaling behavior, from small training instances to large test instances, is the key objective; and the use of validation instances larger than training instances is one key to achieve it. Prior work has used fixed validation sets. Here, we introduce a method generating the validation set dynamically, on the fly, increasing instance size so long as informative and this http URL also introduce refined methodology for evaluating scaling behavior, generating test instances systematically to guarantee a given confidence in coverage performance for each instance size. In experiments, dynamic validation improves scaling behavior of GNN policies in all 9 domains used. 

**Abstract (ZH)**: 近期的工作表明，可以学习到在各个领域中表现成功的动作策略。从少量的训练样本到大量的测试样本，扩展行为是关键目标；在训练样本少于测试样本的情况下使用动态生成的验证集是实现这一目标的关键。以往的工作使用固定的验证集。我们在此引入了一种动态生成验证集的方法，在运行过程中根据样本信息不断增加样本大小。同时，我们还引入了一种改进的评估扩展行为的方法，系统地生成测试样本以确保每个样本大小下的覆盖率性能具有给定的置信度。在实验中，动态验证集提高了所有9个领域中GNN策略的扩展行为。 

---
# Perceptual Implications of Automatic Anonymization in Pathological Speech 

**Title (ZH)**: 病理语音的自动匿名化感知影响 

**Authors**: Soroosh Tayebi Arasteh, Saba Afza, Tri-Thien Nguyen, Lukas Buess, Maryam Parvin, Tomas Arias-Vergara, Paula Andrea Perez-Toro, Hiu Ching Hung, Mahshad Lotfinia, Thomas Gorges, Elmar Noeth, Maria Schuster, Seung Hee Yang, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.00409)  

**Abstract**: Automatic anonymization techniques are essential for ethical sharing of pathological speech data, yet their perceptual consequences remain understudied. This study presents the first comprehensive human-centered analysis of anonymized pathological speech, using a structured perceptual protocol involving ten native and non-native German listeners with diverse linguistic, clinical, and technical backgrounds. Listeners evaluated anonymized-original utterance pairs from 180 speakers spanning Cleft Lip and Palate, Dysarthria, Dysglossia, Dysphonia, and age-matched healthy controls. Speech was anonymized using state-of-the-art automatic methods (equal error rates in the range of 30-40%). Listeners completed Turing-style discrimination and quality rating tasks under zero-shot (single-exposure) and few-shot (repeated-exposure) conditions. Discrimination accuracy was high overall (91% zero-shot; 93% few-shot), but varied by disorder (repeated-measures ANOVA: p=0.007), ranging from 96% (Dysarthria) to 86% (Dysphonia). Anonymization consistently reduced perceived quality (from 83% to 59%, p<0.001), with pathology-specific degradation patterns (one-way ANOVA: p=0.005). Native listeners rated original speech slightly higher than non-native listeners (Delta=4%, p=0.199), but this difference nearly disappeared after anonymization (Delta=1%, p=0.724). No significant gender-based bias was observed. Critically, human perceptual outcomes did not correlate with automatic privacy or clinical utility metrics. These results underscore the need for listener-informed, disorder- and context-specific anonymization strategies that preserve privacy while maintaining interpretability, communicative functions, and diagnostic utility, especially for vulnerable populations such as children. 

**Abstract (ZH)**: 自动匿名化技术对于病理语音数据的伦理共享至关重要，但其感知后果仍研究不足。本研究首次进行了综合的人本中心分析，使用了包含十名语言背景、临床背景和技术背景多样化的德语母语者和非母语者的结构化感知协议。参与者评估了180名讲者的匿名化原始表述对，涵盖唇裂和腭裂、构音障碍、构词障碍、声带障碍以及年龄匹配的健康对照组。语音使用最先进的自动匿名化方法进行了匿名处理（等错误率范围在30-40%）。参与者在单次接触和多次接触条件下完成了图灵风格的辨别和质量评分任务。总体上，辨别准确率较高（单次接触91%，多次接触93%），但不同疾病之间的准确率存在差异（重复测量ANOVA：p=0.007），范围从96%（构音障碍）到86%（声带障碍）。匿名处理一致降低了感知质量（从83%下降到59%，p<0.001），显示出疾病特异性的降质模式（单因素ANOVA：p=0.005）。母语者对原始语音的评价略高于非母语者（Delta=4%，p=0.199），但在匿名处理后，这种差异几乎消失（Delta=1%，p=0.724）。未观察到显著的性别偏见。关键的是，人类感知结果与自动隐私或临床效用指标无相关性。这些结果强调了需要基于听众输入、针对特定疾病和情境定制的匿名化策略的重要性，以同时保护隐私和保持可解释性、沟通功能和诊断效用，尤其是在易损人群如儿童中尤为重要。 

---
# DeepSTA: A Spatial-Temporal Attention Network for Logistics Delivery Timely Rate Prediction in Anomaly Conditions 

**Title (ZH)**: DeepSTA：在异常条件下用于物流配送准时率预测的空间-时间注意力网络 

**Authors**: Jinhui Yi, Huan Yan, Haotian Wang, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00402)  

**Abstract**: Prediction of couriers' delivery timely rates in advance is essential to the logistics industry, enabling companies to take preemptive measures to ensure the normal operation of delivery services. This becomes even more critical during anomaly conditions like the epidemic outbreak, during which couriers' delivery timely rate will decline markedly and fluctuates significantly. Existing studies pay less attention to the logistics scenario. Moreover, many works focusing on prediction tasks in anomaly scenarios fail to explicitly model abnormal events, e.g., treating external factors equally with other features, resulting in great information loss. Further, since some anomalous events occur infrequently, traditional data-driven methods perform poorly in these scenarios. To deal with them, we propose a deep spatial-temporal attention model, named DeepSTA. To be specific, to avoid information loss, we design an anomaly spatio-temporal learning module that employs a recurrent neural network to model incident information. Additionally, we utilize Node2vec to model correlations between road districts, and adopt graph neural networks and long short-term memory to capture the spatial-temporal dependencies of couriers. To tackle the issue of insufficient training data in abnormal circumstances, we propose an anomaly pattern attention module that adopts a memory network for couriers' anomaly feature patterns storage via attention mechanisms. The experiments on real-world logistics datasets during the COVID-19 outbreak in 2022 show the model outperforms the best baselines by 12.11% in MAE and 13.71% in MSE, demonstrating its superior performance over multiple competitive baselines. 

**Abstract (ZH)**: 预测快递员在异常情况下的准时率对于物流行业至关重要，有助于企业在发生预兆时采取预防措施，确保配送服务的正常运行。在疫情期间等异常情况下，快递员的准时率会显著下降并波动较大。现有研究较少关注物流场景。此外，许多专注于异常场景下的预测任务的工作未能明确建模异常事件，例如将外部因素与其他特征同等对待，导致大量信息损失。由于一些异常事件发生的频率很低，传统数据驱动方法在这种情况下表现不佳。为应对这种情况，我们提出了一种深度空间-时间注意力模型，名为DeepSTA。具体而言，为避免信息丢失，我们设计了一种异常时空学习模块，利用递归神经网络建模事件信息。此外，我们利用Node2vec建模道路区域之间的关联，并采用图神经网络和长短期记忆网络捕捉快递员的空间-时间依赖关系。为了应对异常情况下训练数据不足的问题，我们提出了一种异常模式注意力模块，利用记忆网络通过注意力机制存储快递员的异常特征模式。在2022年COVID-19疫情期间的真实物流数据集上的实验结果显示，该模型在MAE上优于最佳基线12.11%，在MSE上优于13.71%，展示了其在多个竞争性基线中的优越性能。 

---
# Learning to Estimate Package Delivery Time in Mixed Imbalanced Delivery and Pickup Logistics Services 

**Title (ZH)**: 学习估计混合不平衡配送与取货物流服务的包裹配送时间 

**Authors**: Jinhui Yi, Huan Yan, Haotian Wang, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00375)  

**Abstract**: Accurately estimating package delivery time is essential to the logistics industry, which enables reasonable work allocation and on-time service guarantee. This becomes even more necessary in mixed logistics scenarios where couriers handle a high volume of delivery and a smaller number of pickup simultaneously. However, most of the related works treat the pickup and delivery patterns on couriers' decision behavior equally, neglecting that the pickup has a greater impact on couriers' decision-making compared to the delivery due to its tighter time constraints. In such context, we have three main challenges: 1) multiple spatiotemporal factors are intricately interconnected, significantly affecting couriers' delivery behavior; 2) pickups have stricter time requirements but are limited in number, making it challenging to model their effects on couriers' delivery process; 3) couriers' spatial mobility patterns are critical determinants of their delivery behavior, but have been insufficiently explored. To deal with these, we propose TransPDT, a Transformer-based multi-task package delivery time prediction model. We first employ the Transformer encoder architecture to capture the spatio-temporal dependencies of couriers' historical travel routes and pending package sets. Then we design the pattern memory to learn the patterns of pickup in the imbalanced dataset via attention mechanism. We also set the route prediction as an auxiliary task of delivery time prediction, and incorporate the prior courier spatial movement regularities in prediction. Extensive experiments on real industry-scale datasets demonstrate the superiority of our method. A system based on TransPDT is deployed internally in JD Logistics to track more than 2000 couriers handling hundreds of thousands of packages per day in Beijing. 

**Abstract (ZH)**: 基于 Transformer 的多任务快递配送时间预测模型 TransPDT 

---
# SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices 

**Title (ZH)**: SacFL: 自适应联邦连续学习roach for 资源受限的边缘设备 

**Authors**: Zhengyi Zhong, Weidong Bao, Ji Wang, Jianguo Chen, Lingjuan Lyu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00365)  

**Abstract**: The proliferation of end devices has led to a distributed computing paradigm, wherein on-device machine learning models continuously process diverse data generated by these devices. The dynamic nature of this data, characterized by continuous changes or data drift, poses significant challenges for on-device models. To address this issue, continual learning (CL) is proposed, enabling machine learning models to incrementally update their knowledge and mitigate catastrophic forgetting. However, the traditional centralized approach to CL is unsuitable for end devices due to privacy and data volume concerns. In this context, federated continual learning (FCL) emerges as a promising solution, preserving user data locally while enhancing models through collaborative updates. Aiming at the challenges of limited storage resources for CL, poor autonomy in task shift detection, and difficulty in coping with new adversarial tasks in FCL scenario, we propose a novel FCL framework named SacFL. SacFL employs an Encoder-Decoder architecture to separate task-robust and task-sensitive components, significantly reducing storage demands by retaining lightweight task-sensitive components for resource-constrained end devices. Moreover, $\rm{SacFL}$ leverages contrastive learning to introduce an autonomous data shift detection mechanism, enabling it to discern whether a new task has emerged and whether it is a benign task. This capability ultimately allows the device to autonomously trigger CL or attack defense strategy without additional information, which is more practical for end devices. Comprehensive experiments conducted on multiple text and image datasets, such as Cifar100 and THUCNews, have validated the effectiveness of $\rm{SacFL}$ in both class-incremental and domain-incremental scenarios. Furthermore, a demo system has been developed to verify its practicality. 

**Abstract (ZH)**: 联邦持续学习框架SacFL：适应有限存储资源和自主数据转移检测 

---
# TNStream: Applying Tightest Neighbors to Micro-Clusters to Define Multi-Density Clusters in Streaming Data 

**Title (ZH)**: TNStream：利用最紧邻微聚类定义流数据中的多密度聚类 

**Authors**: Qifen Zeng, Haomin Bao, Yuanzhuo Hu, Zirui Zhang, Yuheng Zheng, Luosheng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00359)  

**Abstract**: In data stream clustering, systematic theory of stream clustering algorithms remains relatively scarce. Recently, density-based methods have gained attention. However, existing algorithms struggle to simultaneously handle arbitrarily shaped, multi-density, high-dimensional data while maintaining strong outlier resistance. Clustering quality significantly deteriorates when data density varies complexly. This paper proposes a clustering algorithm based on the novel concept of Tightest Neighbors and introduces a data stream clustering theory based on the Skeleton Set. Based on these theories, this paper develops a new method, TNStream, a fully online algorithm. The algorithm adaptively determines the clustering radius based on local similarity, summarizing the evolution of multi-density data streams in micro-clusters. It then applies a Tightest Neighbors-based clustering algorithm to form final clusters. To improve efficiency in high-dimensional cases, Locality-Sensitive Hashing (LSH) is employed to structure micro-clusters, addressing the challenge of storing k-nearest neighbors. TNStream is evaluated on various synthetic and real-world datasets using different clustering metrics. Experimental results demonstrate its effectiveness in improving clustering quality for multi-density data and validate the proposed data stream clustering theory. 

**Abstract (ZH)**: 基于最邻近点的流数据聚类算法及其理论 

---
# Pushing the Limits of Low-Bit Optimizers: A Focus on EMA Dynamics 

**Title (ZH)**: 低比特优化器能力的极限探索：聚焦EMA动力学 

**Authors**: Cong Xu, Wenbin Liang, Mo Yu, Anan Liu, Ke-Yue Zhang, Lizhuang Ma, Jianyong Wang, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00347)  

**Abstract**: The explosion in model sizes leads to continued growth in prohibitive training/fine-tuning costs, particularly for stateful optimizers which maintain auxiliary information of even 2x the model size to achieve optimal convergence. We therefore present in this work a novel type of optimizer that carries with extremely lightweight state overloads, achieved through ultra-low-precision quantization. While previous efforts have achieved certain success with 8-bit or 4-bit quantization, our approach enables optimizers to operate at precision as low as 3 bits, or even 2 bits per state element. This is accomplished by identifying and addressing two critical challenges: the signal swamping problem in unsigned quantization that results in unchanged state dynamics, and the rapidly increased gradient variance in signed quantization that leads to incorrect descent directions. The theoretical analysis suggests a tailored logarithmic quantization for the former and a precision-specific momentum value for the latter. Consequently, the proposed SOLO achieves substantial memory savings (approximately 45 GB when training a 7B model) with minimal accuracy loss. We hope that SOLO can contribute to overcoming the bottleneck in computational resources, thereby promoting greater accessibility in fundamental research. 

**Abstract (ZH)**: 模型规模的爆炸式增长导致训练/微调成本持续飙升，特别是对于维护甚至达到模型大小两倍的辅助信息以实现最优收敛的状态型优化器。因此，在本文中，我们提出了一种新型优化器，该优化器通过超低精度量化携带极其轻量级的状态。虽然之前的努力在8位或4位量化方面取得了一定的成功，但我们的方法使优化器能够在每个状态元素低至3位，甚至2位的精度下运行。通过识别并解决两种关键挑战——无符号量化中的信号淹没问题导致状态动力学不变，以及有符号量化中梯度方差迅速增加导致错误的下降方向——实现了这一目标。理论分析表明，前者应采用定制的对数量化，后者应采用特定精度的动量值。因此，所提出的SOLO在保持最小精度损失的情况下实现了显著的内存节省（例如，训练一个7B模型时约节省45 GB）。我们希望SOLO能够有助于克服计算资源瓶颈，从而促进基础研究的更大普及。 

---
# Enhancing AI-Driven Education: Integrating Cognitive Frameworks, Linguistic Feedback Analysis, and Ethical Considerations for Improved Content Generation 

**Title (ZH)**: 增强AI驱动的教育：结合认知框架、语言反馈分析和伦理考虑以改进内容生成 

**Authors**: Antoun Yaacoub, Sansiri Tarnpradab, Phattara Khumprom, Zainab Assaghir, Lionel Prevost, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2505.00339)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming education, presenting unprecedented opportunities for personalized learning and streamlined content creation. However, realizing the full potential of AI in educational settings necessitates careful consideration of the quality, cognitive depth, and ethical implications of AI-generated materials. This paper synthesizes insights from four related studies to propose a comprehensive framework for enhancing AI-driven educational tools. We integrate cognitive assessment frameworks (Bloom's Taxonomy and SOLO Taxonomy), linguistic analysis of AI-generated feedback, and ethical design principles to guide the development of effective and responsible AI tools. We outline a structured three-phase approach encompassing cognitive alignment, linguistic feedback integration, and ethical safeguards. The practical application of this framework is demonstrated through its integration into OneClickQuiz, an AI-powered Moodle plugin for quiz generation. This work contributes a comprehensive and actionable guide for educators, researchers, and developers aiming to harness AI's potential while upholding pedagogical and ethical standards in educational content generation. 

**Abstract (ZH)**: 人工智能（AI）正迅速变革教育，为个性化学习和内容创建提供前所未有的机遇。然而，在教育环境中实现AI的全部潜能需要仔细考虑AI生成材料的质量、认知深度和伦理影响。本文综合四项相关研究的见解，提出一个全面框架以提升AI驱动的教育工具。我们结合认知评估框架（布卢姆分类法和SOLO分类法）、AI生成反馈的语言分析以及伦理设计原则，指导有效负责任的AI工具的开发。我们概述了一个结构化的三阶段方法，包括认知对齐、语言反馈集成和伦理保障。通过将其整合到OneClickQuiz（一个基于AI的Moodle插件以生成测验）中，展示了该框架的实际应用。本研究为致力于利用AI潜力、同时在教育内容生成中维护教学和伦理标准的教育者、研究人员和开发人员提供了一个全面且可操作的指南。 

---
# Surrogate modeling of Cellular-Potts Agent-Based Models as a segmentation task using the U-Net neural network architecture 

**Title (ZH)**: 基于U-Net神经网络架构的细胞-质点代理模型的代理建模作为分割任务 

**Authors**: Tien Comlekoglu, J. Quetzalcóatl Toledo-Marín, Tina Comlekoglu, Douglas W. DeSimone, Shayn M. Peirce, Geoffrey Fox, James A. Glazier  

**Link**: [PDF](https://arxiv.org/pdf/2505.00316)  

**Abstract**: The Cellular-Potts model is a powerful and ubiquitous framework for developing computational models for simulating complex multicellular biological systems. Cellular-Potts models (CPMs) are often computationally expensive due to the explicit modeling of interactions among large numbers of individual model agents and diffusive fields described by partial differential equations (PDEs). In this work, we develop a convolutional neural network (CNN) surrogate model using a U-Net architecture that accounts for periodic boundary conditions. We use this model to accelerate the evaluation of a mechanistic CPM previously used to investigate \textit{in vitro} vasculogenesis. The surrogate model was trained to predict 100 computational steps ahead (Monte-Carlo steps, MCS), accelerating simulation evaluations by a factor of 590 times compared to CPM code execution. Over multiple recursive evaluations, our model effectively captures the emergent behaviors demonstrated by the original Cellular-Potts model of such as vessel sprouting, extension and anastomosis, and contraction of vascular lacunae. This approach demonstrates the potential for deep learning to serve as efficient surrogate models for CPM simulations, enabling faster evaluation of computationally expensive CPM of biological processes at greater spatial and temporal scales. 

**Abstract (ZH)**: 基于卷积神经网络的U-Net架构周期边界条件模型用于加速细胞-_Potts_模型的评价 

---
# AI-Assisted Decision-Making for Clinical Assessment of Auto-Segmented Contour Quality 

**Title (ZH)**: AI辅助决策在自动分割轮廓质量临床评估中的应用 

**Authors**: Biling Wang, Austen Maniscalco, Ti Bai, Siqiu Wang, Michael Dohopolski, Mu-Han Lin, Chenyang Shen, Dan Nguyen, Junzhou Huang, Steve Jiang, Xinlei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00308)  

**Abstract**: Purpose: This study presents a Deep Learning (DL)-based quality assessment (QA) approach for evaluating auto-generated contours (auto-contours) in radiotherapy, with emphasis on Online Adaptive Radiotherapy (OART). Leveraging Bayesian Ordinal Classification (BOC) and calibrated uncertainty thresholds, the method enables confident QA predictions without relying on ground truth contours or extensive manual labeling. Methods: We developed a BOC model to classify auto-contour quality and quantify prediction uncertainty. A calibration step was used to optimize uncertainty thresholds that meet clinical accuracy needs. The method was validated under three data scenarios: no manual labels, limited labels, and extensive labels. For rectum contours in prostate cancer, we applied geometric surrogate labels when manual labels were absent, transfer learning when limited, and direct supervision when ample labels were available. Results: The BOC model delivered robust performance across all scenarios. Fine-tuning with just 30 manual labels and calibrating with 34 subjects yielded over 90% accuracy on test data. Using the calibrated threshold, over 93% of the auto-contours' qualities were accurately predicted in over 98% of cases, reducing unnecessary manual reviews and highlighting cases needing correction. Conclusion: The proposed QA model enhances contouring efficiency in OART by reducing manual workload and enabling fast, informed clinical decisions. Through uncertainty quantification, it ensures safer, more reliable radiotherapy workflows. 

**Abstract (ZH)**: 目的：本文提出了一种基于深度学习（DL）的质量评估（QA）方法，用于评估放射治疗中的自动生成轮廓（auto-contours），特别是在在线自适应放射治疗（OART）中。该方法利用贝叶斯序贯分类（BOC）和校准的不确定性阈值，能够在无需参考标准轮廓或大量人工标注的情况下进行自信的质量评估。方法：我们开发了一个BOC模型来分类自动轮廓的质量并量化预测的不确定性。通过校准步骤，优化了满足临床准确性的不确定性阈值。该方法在三种数据场景下进行了验证：无人工标注、少量人工标注和大量人工标注。对于前列腺癌患者的直肠轮廓，当缺乏人工标注时，我们使用几何代理标签；当标注有限时，我们采用迁移学习；当标注充足时，我们直接进行监督学习。结果：BOC模型在所有场景中均表现出稳健的性能。仅使用30个人工标注进行微调并校准34个受试者后，测试数据的准确率超过90%。使用校准的阈值，超过93%的自动轮廓的质量预测准确，且在超过98%的情况下减少了不必要的手动审查，并识别了需要纠正的病例。结论：提出的质量评估模型通过减少人工工作量并促进快速、有根据的临床决策，增强了OART中的轮廓绘制效率。通过不确定性量化，确保了更安全、更可靠的放射治疗工作流程。 

---
# Multi-Hierarchical Fine-Grained Feature Mapping Driven by Feature Contribution for Molecular Odor Prediction 

**Title (ZH)**: 基于特征贡献的多层级细粒度特征映射分子气味预测 

**Authors**: Hong Xin Xie, Jian De Sun, Fan Fu Xue, Zi Fei Han, Shan Shan Feng, Qi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00290)  

**Abstract**: Molecular odor prediction is the process of using a molecule's structure to predict its smell. While accurate prediction remains challenging, AI models can suggest potential odors. Existing methods, however, often rely on basic descriptors or handcrafted fingerprints, which lack expressive power and hinder effective learning. Furthermore, these methods suffer from severe class imbalance, limiting the training effectiveness of AI models. To address these challenges, we propose a Feature Contribution-driven Hierarchical Multi-Feature Mapping Network (HMFNet). Specifically, we introduce a fine-grained, Local Multi-Hierarchy Feature Extraction module (LMFE) that performs deep feature extraction at the atomic level, capturing detailed features crucial for odor prediction. To enhance the extraction of discriminative atomic features, we integrate a Harmonic Modulated Feature Mapping (HMFM). This module dynamically learns feature importance and frequency modulation, improving the model's capability to capture relevant patterns. Additionally, a Global Multi-Hierarchy Feature Extraction module (GMFE) is designed to learn global features from the molecular graph topology, enabling the model to fully leverage global information and enhance its discriminative power for odor prediction. To further mitigate the issue of class imbalance, we propose a Chemically-Informed Loss (CIL). Experimental results demonstrate that our approach significantly improves performance across various deep learning models, highlighting its potential to advance molecular structure representation and accelerate the development of AI-driven technologies. 

**Abstract (ZH)**: 分子气味预测是使用分子结构预测其气味的过程。虽然准确预测仍然具有挑战性，但AI模型可以建议潜在的气味。现有方法通常依赖于基本描述符或手工地指纹，这些方法缺乏表达能力，阻碍了有效的学习。此外，这些方法还受到严重类别不平衡的影响，限制了AI模型的训练效果。为了解决这些挑战，我们提出了一种特征贡献驱动的分层多特征映射网络（HMFNet）。具体而言，我们引入了一种细粒度的局部多层次特征提取模块（LMFE），在原子级别进行深度特征提取，捕获对气味预测至关重要的详细特征。为了增强原子特征的提取，我们整合了一种谐波调制特征映射（HMFM）模块，该模块动态学习特征的重要性并进行频率调制，提高模型捕捉相关模式的能力。此外，我们设计了一种全局多层次特征提取模块（GMFE），从分子图拓扑中学习全局特征，使模型能够充分利用全局信息并增强其对气味预测的区分能力。为了进一步缓解类别不平衡问题，我们提出了一种化学信息损失函数（CIL）。实验结果表明，我们的方法显著提高了各种深度学习模型的性能，展示了其在分子结构表示和加速AI驱动技术开发方面的潜在价值。 

---
# Pack-PTQ: Advancing Post-training Quantization of Neural Networks by Pack-wise Reconstruction 

**Title (ZH)**: Pack-PTQ: 通过包级重建促进神经网络的后训练量化 

**Authors**: Changjun Li, Runqing Jiang, Zhuo Song, Pengpeng Yu, Ye Zhang, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00259)  

**Abstract**: Post-training quantization (PTQ) has evolved as a prominent solution for compressing complex models, which advocates a small calibration dataset and avoids end-to-end retraining. However, most existing PTQ methods employ block-wise reconstruction, which neglects cross-block dependency and exhibits a notable accuracy drop in low-bit cases. To address these limitations, this paper presents a novel PTQ method, dubbed Pack-PTQ. First, we design a Hessian-guided adaptive packing mechanism to partition blocks into non-overlapping packs, which serve as the base unit for reconstruction, thereby preserving the cross-block dependency and enabling accurate quantization parameters estimation. Second, based on the pack configuration, we propose a mixed-precision quantization approach to assign varied bit-widths to packs according to their distinct sensitivities, thereby further enhancing performance. Extensive experiments on 2D image and 3D point cloud classification tasks, using various network architectures, demonstrate the superiority of our method over the state-of-the-art PTQ methods. 

**Abstract (ZH)**: Post-训练量化(Pack-PTQ)：一种考虑跨块依赖性的新型后训练量化方法 

---
# Predicting Estimated Times of Restoration for Electrical Outages Using Longitudinal Tabular Transformers 

**Title (ZH)**: 使用纵向表格变换器预测电力中断的预计恢复时间 

**Authors**: Bogireddy Sai Prasanna Teja, Valliappan Muthukaruppan, Carls Benjamin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00225)  

**Abstract**: As climate variability increases, the ability of utility providers to deliver precise Estimated Times of Restoration (ETR) during natural disasters has become increasingly critical. Accurate and timely ETRs are essential for enabling customer preparedness during extended power outages, where informed decision-making can be crucial, particularly in severe weather conditions. Nonetheless, prevailing utility practices predominantly depend on manual assessments or traditional statistical methods, which often fail to achieve the level of precision required for reliable and actionable predictions. To address these limitations, we propose a Longitudinal Tabular Transformer (LTT) model that leverages historical outage event data along with sequential updates of these events to improve the accuracy of ETR predictions. The model's performance was evaluated over 34,000 storm-related outage events from three major utility companies, collectively serving over 3 million customers over a 2-year period. Results demonstrate that the LTT model improves the Customer Satisfaction Impact (CSI) metric by an average of 19.08% (p > 0.001) compared to existing methods. Additionally, we introduce customer-informed regression metrics that align model evaluation with real-world satisfaction, ensuring the outcomes resonate with customer expectations. Furthermore, we employ interpretability techniques to analyze the temporal significance of incorporating sequential updates in modeling outage events and to identify the contributions of predictive features to a given ETR. This comprehensive approach not only improves predictive accuracy but also enhances transparency, fostering greater trust in the model's capabilities. 

**Abstract (ZH)**: 随着气候变异性增加，utility提供商在自然灾害期间提供精确恢复时间估计(ETR)的能力变得越来越关键。准确及时的ETR对于帮助客户在长时间断电期间做好准备至关重要，特别是在恶劣天气条件下，基于信息的决策可以起到关键作用。然而，现有的utility做法主要依赖手工评估或传统统计方法，往往无法实现可靠且可操作的预测所需的精确度。为了解决这些限制，我们提出了一种纵向表格变换器(LTT)模型，该模型利用历史停电事件数据以及这些事件的序列更新来提高ETR预测的准确性。该模型在来自三家主要utility公司的超过34,000个风暴相关停电事件中进行了评估，这些公司共同服务了超过300万客户，评估期为两年。结果显示，与现有方法相比，LTT模型将客户满意度影响(CSI)指标平均提高了19.08%（p > 0.001）。此外，我们引入了基于客户的回归指标，确保模型评估与实际满意度相一致，从而使结果能够反映客户期望。此外，我们使用可解释性技术来分析在建模停电事件中序列更新的时间重要性，并确定预测特征对特定ETR的贡献。这种全面的方法不仅提高了预测准确性，还增强了透明度，促进了对模型能力的信任。 

---
# Online Federation For Mixtures of Proprietary Agents with Black-Box Encoders 

**Title (ZH)**: 在线联邦学习框架下混合私有代理的黑盒编码 

**Authors**: Xuwei Yang, Fatemeh Tavakoli, David B. Emerson, Anastasis Kratsios  

**Link**: [PDF](https://arxiv.org/pdf/2505.00216)  

**Abstract**: Most industry-standard generative AIs and feature encoders are proprietary, offering only black-box access: their outputs are observable, but their internal parameters and architectures remain hidden from the end-user. This black-box access is especially limiting when constructing mixture-of-expert type ensemble models since the user cannot optimize each proprietary AI's internal parameters. Our problem naturally lends itself to a non-competitive game-theoretic lens where each proprietary AI (agent) is inherently competing against the other AI agents, with this competition arising naturally due to their obliviousness of the AI's to their internal structure. In contrast, the user acts as a central planner trying to synchronize the ensemble of competing AIs.
We show the existence of the unique Nash equilibrium in the online setting, which we even compute in closed-form by eliciting a feedback mechanism between any given time series and the sequence generated by each (proprietary) AI agent. Our solution is implemented as a decentralized, federated-learning algorithm in which each agent optimizes their structure locally on their machine without ever releasing any internal structure to the others. We obtain refined expressions for pre-trained models such as transformers, random feature models, and echo-state networks. Our ``proprietary federated learning'' algorithm is implemented on a range of real-world and synthetic time-series benchmarks. It achieves orders-of-magnitude improvements in predictive accuracy over natural benchmarks, of which there are surprisingly few due to this natural problem still being largely unexplored. 

**Abstract (ZH)**: 大多数工业标准的生成AI和特征编码器是专有的，只提供黑盒访问：它们的输出是可观测的，但其内部参数和架构对最终用户仍然是隐藏的。这种黑盒访问在构建专家混合类型集成模型时尤为受限，因为用户无法优化每个专有AI的内部参数。我们的问题自然适用于非竞争性的博弈论视角，在这种视角下，每个专有AI（代理）本质上是在与其他AI代理竞争，这种竞争自然地产生于它们对其内部结构的无知。相比之下，用户作为中央规划者，试图同步竞争中的AI代理群。 

---
# Empirical Evaluation of Progressive Coding for Sparse Autoencoders 

**Title (ZH)**: 渐进编码在稀疏自编码器中的实证评价 

**Authors**: Hans Peter, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2505.00190)  

**Abstract**: Sparse autoencoders (SAEs) \citep{bricken2023monosemanticity,gao2024scalingevaluatingsparseautoencoders} rely on dictionary learning to extract interpretable features from neural networks at scale in an unsupervised manner, with applications to representation engineering and information retrieval. SAEs are, however, computationally expensive \citep{lieberum2024gemmascopeopensparse}, especially when multiple SAEs of different sizes are needed. We show that dictionary importance in vanilla SAEs follows a power law. We compare progressive coding based on subset pruning of SAEs -- to jointly training nested SAEs, or so-called {\em Matryoshka} SAEs \citep{bussmann2024learning,nabeshima2024Matryoshka} -- on a language modeling task. We show Matryoshka SAEs exhibit lower reconstruction loss and recaptured language modeling loss, as well as higher representational similarity. Pruned vanilla SAEs are more interpretable, however. We discuss the origins and implications of this trade-off. 

**Abstract (ZH)**: 稀疏自编码器（SAEs）依赖字典学习从神经网络中以无监督方式大规模提取可解释特征，应用于表示工程和信息检索。然而，SAEs计算成本较高，尤其是当需要多个不同规模的SAEs时。我们发现，vanilla SAEs中的字典重要性遵循幂律分布。我们将基于SAEs子集剪枝的渐进编码与联合训练嵌套SAEs或所谓的“木头娃娃”SAEs进行比较，展示了在语言建模任务中的表现。我们发现，“木头娃娃”SAEs具有更低的重建损失和捕获语言建模损失，以及更高的表示相似性。然而，剪枝的vanilla SAEs更具可解释性。我们讨论这种权衡的来源及其影响。 

---
# Neuroevolution of Self-Attention Over Proto-Objects 

**Title (ZH)**: 自注意力在原型对象上的神经进化 

**Authors**: Rafael C. Pinto, Anderson R. Tavares  

**Link**: [PDF](https://arxiv.org/pdf/2505.00186)  

**Abstract**: Proto-objects - image regions that share common visual properties - offer a promising alternative to traditional attention mechanisms based on rectangular-shaped image patches in neural networks. Although previous work demonstrated that evolving a patch-based hard-attention module alongside a controller network could achieve state-of-the-art performance in visual reinforcement learning tasks, our approach leverages image segmentation to work with higher-level features. By operating on proto-objects rather than fixed patches, we significantly reduce the representational complexity: each image decomposes into fewer proto-objects than regular patches, and each proto-object can be efficiently encoded as a compact feature vector. This enables a substantially smaller self-attention module that processes richer semantic information. Our experiments demonstrate that this proto-object-based approach matches or exceeds the state-of-the-art performance of patch-based implementations with 62% less parameters and 2.6 times less training time. 

**Abstract (ZH)**: 基于原型对象的图像区域——具有共同视觉属性的图像区域——为神经网络中传统基于矩形图像块的注意力机制提供了有前景的替代方案。尽管之前的工作证明，在控制器网络的同时演化一个基于块的硬注意力模块可以实现视觉强化学习任务中的前沿性能，我们的方法利用图像分割来处理更高层次的特征。通过操作原型对象而非固定块，我们显著降低了表示复杂性：每个图像分解为较少的原型对象，每个原型对象可以高效地编码为紧凑的特征向量。这使得自注意力模块更小且能够处理更丰富的语义信息。我们的实验表明，这种基于原型对象的方法在参数量减少62%且训练时间减少2.6倍的情况下，匹配或超越了基于块实现的前沿性能。 

---
# Attention-enabled Explainable AI for Bladder Cancer Recurrence Prediction 

**Title (ZH)**: 基于注意力机制的可解释AI在膀胱癌复发预测中的应用 

**Authors**: Saram Abbas, Naeem Soomro, Rishad Shafik, Rakesh Heer, Kabita Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2505.00171)  

**Abstract**: Non-muscle-invasive bladder cancer (NMIBC) is a relentless challenge in oncology, with recurrence rates soaring as high as 70-80%. Each recurrence triggers a cascade of invasive procedures, lifelong surveillance, and escalating healthcare costs - affecting 460,000 individuals worldwide. However, existing clinical prediction tools remain fundamentally flawed, often overestimating recurrence risk and failing to provide personalized insights for patient management. In this work, we propose an interpretable deep learning framework that integrates vector embeddings and attention mechanisms to improve NMIBC recurrence prediction performance. We incorporate vector embeddings for categorical variables such as smoking status and intravesical treatments, allowing the model to capture complex relationships between patient attributes and recurrence risk. These embeddings provide a richer representation of the data, enabling improved feature interactions and enhancing prediction performance. Our approach not only enhances performance but also provides clinicians with patient-specific insights by highlighting the most influential features contributing to recurrence risk for each patient. Our model achieves accuracy of 70% with tabular data, outperforming conventional statistical methods while providing clinician-friendly patient-level explanations through feature attention. Unlike previous studies, our approach identifies new important factors influencing recurrence, such as surgical duration and hospital stay, which had not been considered in existing NMIBC prediction models. 

**Abstract (ZH)**: 非肌肉浸润性膀胱癌（NMIBC）的持续挑战：复发率高达70-80%，每次复发都会引发一系列侵入性程序、终生监测和医疗费用急剧上升，影响着全球460,000名患者。然而，现有的临床预测工具仍存在根本性缺陷，往往会高估复发风险，无法为患者的管理提供个性化见解。在这项工作中，我们提出了一种可解释的深度学习框架，结合向量嵌入和注意力机制以提高NMIBC复发预测性能。我们将向量嵌入应用于如吸烟状态和膀胱内治疗等分类变量，使模型能够捕捉患者属性与复发风险之间的复杂关系。这些嵌入提供了数据的 richer 表示，有助于改善特征交互，从而提升预测性能。我们的方法不仅提高了性能，还通过突出显示每位患者最能影响复发风险的特征，为临床医生提供了患者特定的见解。我们的模型在表格数据上的准确率达到70%，同时通过特征注意力为临床医生提供患者层面的解释，优于传统的统计方法。与以往研究不同，我们的方法识别了新的重要影响因素，如手术时间和住院时间，这些因素在现有的NMIBC预测模型中未被考虑。 

---
# GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation 

**Title (ZH)**: GEOM-Drugs 重访：朝向更化学准确的3D分子生成基准方向 

**Authors**: Filipp Nikitin, Ian Dunn, David Ryan Koes, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00169)  

**Abstract**: Deep generative models have shown significant promise in generating valid 3D molecular structures, with the GEOM-Drugs dataset serving as a key benchmark. However, current evaluation protocols suffer from critical flaws, including incorrect valency definitions, bugs in bond order calculations, and reliance on force fields inconsistent with the reference data. In this work, we revisit GEOM-Drugs and propose a corrected evaluation framework: we identify and fix issues in data preprocessing, construct chemically accurate valency tables, and introduce a GFN2-xTB-based geometry and energy benchmark. We retrain and re-evaluate several leading models under this framework, providing updated performance metrics and practical recommendations for future benchmarking. Our results underscore the need for chemically rigorous evaluation practices in 3D molecular generation. Our recommended evaluation methods and GEOM-Drugs processing scripts are available at this https URL. 

**Abstract (ZH)**: 深度生成模型在生成有效3D分子结构方面展现了显著的潜力，GEOM-Drugs数据集是关键基准之一。然而，当前的评估协议存在严重缺陷，包括错误的价键定义、键级计算中的bug以及参考数据不一致的力场依赖。在本文中，我们重新审视了GEOM-Drugs数据集，并提出了一种修正的评估框架：我们识别并修复了数据预处理中的问题，构建了化学准确的价键表，并引入了基于GFN2-xTB的几何和能量基准。我们在该框架下重新训练和评估了多个领先模型，提供了更新的性能指标和未来基准测试的实用建议。我们的结果强调了在3D分子生成中采用化学严谨的评估实践的必要性。我们推荐的评估方法和GEOM-Drugs处理脚本可在以下链接获取：this https URL。 

---
# GPRat: Gaussian Process Regression with Asynchronous Tasks 

**Title (ZH)**: GPRat: 异步任务的高斯过程回归 

**Authors**: Maksim Helmann, Alexander Strack, Dirk Pflüger  

**Link**: [PDF](https://arxiv.org/pdf/2505.00136)  

**Abstract**: Python is the de-facto language for software development in artificial intelligence (AI). Commonly used libraries, such as PyTorch and TensorFlow, rely on parallelization built into their BLAS backends to achieve speedup on CPUs. However, only applying parallelization in a low-level backend can lead to performance and scaling degradation. In this work, we present a novel way of binding task-based C++ code built on the asynchronous runtime model HPX to a high-level Python API using pybind11. We develop a parallel Gaussian process (GP) li- brary as an application. The resulting Python library GPRat combines the ease of use of commonly available GP libraries with the performance and scalability of asynchronous runtime systems. We evaluate the per- formance on a mass-spring-damper system, a standard benchmark from control theory, for varying numbers of regressors (features). The results show almost no binding overhead when binding the asynchronous HPX code using pybind11. Compared to GPyTorch and GPflow, GPRat shows superior scaling on up to 64 cores on an AMD EPYC 7742 CPU for train- ing. Furthermore, our library achieves a prediction speedup of 7.63 over GPyTorch and 25.25 over GPflow. If we increase the number of features from eight to 128, we observe speedups of 29.62 and 21.19, respectively. These results showcase the potential of using asynchronous tasks within Python-based AI applications. 

**Abstract (ZH)**: 基于异步运行时模型的C++任务绑定到高级Python API的新型方法：异步HPX与GPRat库在人工智能应用中的性能与扩展性探索 

---
# Evaluating the AI-Lab Intervention: Impact on Student Perception and Use of Generative AI in Early Undergraduate Computer Science Courses 

**Title (ZH)**: 评估AI实验室干预措施：对学生对生成式AI在早期本科计算机科学课程中认知和使用影响的研究 

**Authors**: Ethan Dickey, Andres Bejarano, Rhianna Kuperus, Bárbara Fagundes  

**Link**: [PDF](https://arxiv.org/pdf/2505.00100)  

**Abstract**: Generative AI (GenAI) is rapidly entering computer science education, yet its effects on student learning, skill development, and perceptions remain underexplored. Concerns about overreliance coexist with a gap in research on structured scaffolding to guide tool use in formal courses. This study examines the impact of a dedicated "AI-Lab" intervention -- emphasizing guided scaffolding and mindful engagement -- on undergraduate students in Data Structures and Algorithms, Competitive Programming, and first-year engineering courses at Purdue University.
Over three semesters, we integrated AI-Lab modules into four mandatory and elective courses, yielding 831 matched pre- and post-intervention survey responses, alongside focus group discussions. Employing a mixed-methods approach, we analyzed quantitative shifts in usage patterns and attitudes as well as qualitative narratives of student experiences.
While the overall frequency of GenAI usage for homework or programming projects remained largely stable, we observed large effect sizes in comfort and openness across conceptual, debugging, and homework problems. Notably, usage patterns for debugging also shifted statistically significantly, reflecting students' more mindful and deliberate approach. Focus group discussions corroborated these results, suggesting that the intervention "bridged the gap" between naive GenAI usage and more nuanced, reflective integration of AI tools into coursework, ultimately heightening students' awareness of their own skill development.
These findings suggest that structured, scaffolded interventions can enable students to harness GenAI's benefits without undermining essential competencies. We offer evidence-based recommendations for educators seeking to integrate GenAI responsibly into computing curricula and identify avenues for future research on GenAI-supported pedagogy. 

**Abstract (ZH)**: 生成式人工智能（GenAI）正迅速融入计算机科学教育，但其对学生学习、技能发展和认知影响的研究仍不足。关于过度依赖的担忧与正式课程中结构化支架引导工具使用研究不足并存。本研究考察了“AI-Lab”专门干预措施——强调引导式支架和自觉参与——对学生在普渡大学数据结构与算法、编程竞赛以及大一工程课程中的影响。

在三个学期中，我们将AI-Lab模块整合到四门必修和选修课程中，获得了831份前后测问卷响应，以及焦点小组讨论的数据。采用混合方法，我们分析了使用模式和态度的定量变化以及学生的质性叙述。

虽然GenAI在家作业或编程项目中的总体使用频率保持相对稳定，但我们在概念性问题、调试问题和家庭作业问题上观察到了显著的影响大小。值得注意的是，调试阶段的使用模式也实现了统计显著的变化，反映了学生更加自觉和审慎的方法。焦点小组讨论也证实了这些结果，表明干预措施填补了简陋使用GenAI与更加细致、反思性地将AI工具整合到学业之间的差距，最终提升了学生对自己技能发展的意识。

研究结果表明，结构化的支架式干预措施能够使学生充分利用GenAI的优势，而不削弱其基本能力。我们提供了教育者在计算课程中负责任地整合GenAI方面的证据基于的建议，并指出了GenAI支持的教学方法未来研究的途径。 

---
# Emotional Analysis of Fashion Trends Using Social Media and AI: Sentiment Analysis on Twitter for Fashion Trend Forecasting 

**Title (ZH)**: 基于社交媒体和AI的情感分析：微博上的情感分析在时尚趋势预测中的应用 

**Authors**: Aayam Bansal, Agneya Tharun  

**Link**: [PDF](https://arxiv.org/pdf/2505.00050)  

**Abstract**: This study explores the intersection of fashion trends and social media sentiment through computational analysis of Twitter data using the T4SA (Twitter for Sentiment Analysis) dataset. By applying natural language processing and machine learning techniques, we examine how sentiment patterns in fashion-related social media conversations can serve as predictors for emerging fashion trends. Our analysis involves the identification and categorization of fashion-related content, sentiment classification with improved normalization techniques, time series decomposition, statistically validated causal relationship modeling, cross-platform sentiment comparison, and brand-specific sentiment analysis. Results indicate correlations between sentiment patterns and fashion theme popularity, with accessories and streetwear themes showing statistically significant rising trends. The Granger causality analysis establishes sustainability and streetwear as primary trend drivers, showing bidirectional relationships with several other themes. The findings demonstrate that social media sentiment analysis can serve as an effective early indicator of fashion trend trajectories when proper statistical validation is applied. Our improved predictive model achieved 78.35% balanced accuracy in sentiment classification, establishing a reliable foundation for trend prediction across positive, neutral, and negative sentiment categories. 

**Abstract (ZH)**: 本研究通过计算分析Twitter数据（使用T4SA（Twitter for Sentiment Analysis）数据集）探索时尚趋势与社交媒体情感的交集。通过对自然语言处理和机器学习技术的应用，我们研究了与时尚相关的社交媒体对话中的情感模式如何成为新兴时尚趋势的预测指标。分析涉及时尚相关内容的识别和分类、改进规范化技术的情感分类、时间序列分解、经过统计验证的因果关系建模、跨平台情感比较以及品牌特定的情感分析。结果表明，情感模式与时尚主题流行度之间存在相关性，配饰和街头潮流主题显示出统计上显著的增长趋势。遍历因果分析确立了可持续性和街头潮流为主要趋势驱动因素，并与多种其他主题之间存在双向关系。研究结果表明，在应用适当的统计验证时，社交媒体情感分析可作为时尚趋势轨迹的有效早期指标。改进的预测模型在情感分类中的平衡准确率达到78.35%，为在正面、中性和负面情感类别中预测趋势奠定了可靠的基础。 

---
# Convolutional Autoencoders for Data Compression and Anomaly Detection in Small Satellite Technologies 

**Title (ZH)**: 卷积自编码器在小卫星技术中的数据压缩与异常检测 

**Authors**: Dishanand Jayeprokash, Julia Gonski  

**Link**: [PDF](https://arxiv.org/pdf/2505.00040)  

**Abstract**: Small satellite technologies have enhanced the potential and feasibility of geodesic missions, through simplification of design and decreased costs allowing for more frequent launches. On-satellite data acquisition systems can benefit from the implementation of machine learning (ML), for better performance and greater efficiency on tasks such as image processing or feature extraction. This work presents convolutional autoencoders for implementation on the payload of small satellites, designed to achieve dual functionality of data compression for more efficient off-satellite transmission, and at-source anomaly detection to inform satellite data-taking. This capability is demonstrated for a use case of disaster monitoring using aerial image datasets of the African continent, offering avenues for both novel ML-based approaches in small satellite applications along with the expansion of space technology and artificial intelligence in Africa. 

**Abstract (ZH)**: 小卫星技术通过简化设计和降低发射成本，增强了地理测量任务的潜力和可行性，车上数据获取系统可通过实施机器学习实现更好的性能和更高的效率，用于图像处理或特征提取任务。本文提出在小卫星载荷中实现卷积自编码器，旨在实现数据压缩以提高离轨传输效率，并在源头实现异常检测以指导卫星数据采集。这一能力通过使用非洲大陆航空图像数据集进行灾害监测的应用案例得以展示，也为小型卫星应用中新型机器学习方法以及非洲的空间技术和人工智能扩展提供了契机。 

---
# Linguistic Complexity and Socio-cultural Patterns in Hip-Hop Lyrics 

**Title (ZH)**: 汉语语言复杂性与嘻哈歌词中的社会文化模式 

**Authors**: Aayam Bansal, Raghav Agarwal, Kaashvi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2505.00035)  

**Abstract**: This paper presents a comprehensive computational framework for analyzing linguistic complexity and socio-cultural trends in hip-hop lyrics. Using a dataset of 3,814 songs from 146 influential artists spanning four decades (1980-2020), we employ natural language processing techniques to quantify multiple dimensions of lyrical complexity. Our analysis reveals a 23.7% increase in vocabulary diversity over the study period, with East Coast artists demonstrating 17.3% higher lexical variation than other regions. Rhyme density increased by 34.2% across all regions, with Midwest artists exhibiting the highest technical complexity (3.04 rhymes per line). Topic modeling identified significant shifts in thematic content, with social justice themes decreasing from 28.5% to 13.8% of content while introspective themes increased from 7.6% to 26.3%. Sentiment analysis demon- strated that lyrics became significantly more negative during sociopolitical crises, with polarity decreasing by 0.31 following major social unrest. Multi-dimensional analysis revealed four dis- tinct stylistic approaches that correlate strongly with geographic origin (r=0.68, p!0.001) and time period (r=0.59, p<0.001). These findings establish quantitative evidence for the evolution of hip- hop as both an art form and a reflection of societal dynamics, providing insights into the interplay between linguistic innovation and cultural context in popular music. 

**Abstract (ZH)**: 本文 Presents 一个全面的计算框架，用于分析嘻哈歌词中的语言复杂性和社会文化趋势。利用1980-2020年四个年代、146位有影响力的艺术家的3,814首歌曲数据集，我们运用自然语言处理技术量化歌词复杂性的多个维度。分析结果显示，研究期间词汇多样性增加了23.7%，而东海岸艺术家的词汇变化率比其他地区高17.3%。韵律密度在所有地区增加了34.2%，中西部艺术家表现出最高的技术复杂度（每行3.04个韵脚）。主题建模揭示了主题内容的显著变化，社会正义主题从内容的28.5%下降到13.8%，而 introspective 主题从7.6%增加到26.3%。情感分析表明，在社会政治危机期间，歌词变得显著更具负面性，主要社会动荡后极性下降了0.31。多维分析揭示了四种与地理起源（r=0.68，p<0.001）和时间时期（r=0.59，p<0.001）高度相关的独特风格方法。这些发现为嘻哈作为一种艺术形式及其对社会动态的反映的演变提供了定量证据，提供了关于语言创新与文化语境在流行音乐中相互作用的洞见。 

---
# Keep the General, Inject the Specific: Structured Dialogue Fine-Tuning for Knowledge Injection without Catastrophic Forgetting 

**Title (ZH)**: 保持通用性，注入特定性：结构化对话微调以实现知识注入且无灾难性遗忘 

**Authors**: Yijie Hong, Xiaofei Yin, Xinzhong Wang, Yi Tu, Ya Guo, Sufeng Duan, Weiqiang Wang, Lingyong Fang, Depeng Wang, Huijia Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00029)  

**Abstract**: Large Vision Language Models have demonstrated impressive versatile capabilities through extensive multimodal pre-training, but face significant limitations when incorporating specialized knowledge domains beyond their training distribution. These models struggle with a fundamental dilemma: direct adaptation approaches that inject domain-specific knowledge often trigger catastrophic forgetting of foundational visual-linguistic abilities. We introduce Structured Dialogue Fine-Tuning (SDFT), an effective approach that effectively injects domain-specific knowledge while minimizing catastrophic forgetting. Drawing inspiration from supervised fine-tuning in LLMs and subject-driven personalization in text-to-image diffusion models, our method employs a three-phase dialogue structure: Foundation Preservation reinforces pre-trained visual-linguistic alignment through caption tasks; Contrastive Disambiguation introduces carefully designed counterfactual examples to maintain semantic boundaries; and Knowledge Specialization embeds specialized information through chain-of-thought reasoning. Experimental results across multiple domains confirm SDFT's effectiveness in balancing specialized knowledge acquisition with general capability retention. Our key contributions include a data-centric dialogue template that balances foundational alignment with targeted knowledge integration, a weighted multi-turn supervision framework, and comprehensive evaluation across diverse knowledge types. 

**Abstract (ZH)**: 大规模视觉语言模型通过广泛的多模态预训练展示了令人印象深刻的多功能能力，但在融入超出其训练分布的专门知识领域时面临显著限制。这些模型在根本上面临一个难题：直接适应方法虽然可以注入领域特定知识，但往往会引发对基础视觉-语言能力的灾难性遗忘。我们引入了结构化对话微调（SDFT），这是一种有效的方法，它能够有效注入专门知识，同时最大限度地减少灾难性遗忘。该方法受到大规模语言模型的监督微调和文本到图像扩散模型的主题驱动个性化启发，采用三个阶段的对话结构：基础保护通过字幕任务强化预训练的视觉-语言对齐；对比去模糊通过引入精心设计的反事实示例维持语义边界；知识专业化通过链式推理嵌入专门信息。实验结果在多个领域证实了SDFT在专业知识学习与保持一般能力之间的平衡效果。我们的主要贡献包括以数据为中心的对话模板，平衡基础对齐与目标知识整合，加权多轮监督框架，以及跨多种知识类型进行全面评估。 

---
# Extracting Abstraction Dimensions by Identifying Syntax Pattern from Texts 

**Title (ZH)**: 从文本中识别语法模式提取抽象维度 

**Authors**: Jian Zhou, Jiazheng Li, Sirui Zhuge, Hai Zhuge  

**Link**: [PDF](https://arxiv.org/pdf/2505.00027)  

**Abstract**: This paper proposed an approach to automatically discovering subject dimension, action dimension, object dimension and adverbial dimension from texts to efficiently operate texts and support query in natural language. The high quality of trees guarantees that all subjects, actions, objects and adverbials and their subclass relations within texts can be represented. The independency of trees ensures that there is no redundant representation between trees. The expressiveness of trees ensures that the majority of sentences can be accessed from each tree and the rest of sentences can be accessed from at least one tree so that the tree-based search mechanism can support querying in natural language. Experiments show that the average precision, recall and F1-score of the abstraction trees constructed by the subclass relations of subject, action, object and adverbial are all greater than 80%. The application of the proposed approach to supporting query in natural language demonstrates that different types of question patterns for querying subject or object have high coverage of texts, and searching multiple trees on subject, action, object and adverbial according to the question pattern can quickly reduce search space to locate target sentences, which can support precise operation on texts. 

**Abstract (ZH)**: 本文提出了一种自动发现主题维度、动作维度、对象维度和状语维度的方法，以高效地操作文本并支持自然语言查询。高质量的树结构保证了文本中所有主题、动作、对象和状语及其子类关系能够被表示。树的独立性保证了树之间没有冗余表示。树的表达能力确保大多数句子可以从每棵树中访问到，剩余的句子可以从至少一棵树中访问到，从而支持基于树的查询机制的自然语言查询。实验结果表明，由主题、动作、对象和状语的子类关系构建的抽象树的平均精度、召回率和F1分数均大于80%。所提出的方法应用于支持自然语言查询中，展示了不同类型的查询模式在查询主题或对象时对文本的高覆盖度，并且根据查询模式在主题、动作、对象和状语上的多棵树搜索可以迅速减少搜索空间，定位目标句子，从而支持对文本的精确操作。 

---
# Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning 

**Title (ZH)**: Nemotron-研究工具N1：配备强化推理的工具使用语言模型 

**Authors**: Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00024)  

**Abstract**: Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text generation tasks. Prior work typically enhances tool-use abilities by either applying supervised fine-tuning (SFT) to enforce tool-call correctness or distilling reasoning traces from stronger models for SFT. However, both approaches fall short, either omitting reasoning entirely or producing imitative reasoning that limits generalization. Inspired by the success of DeepSeek-R1 in eliciting reasoning through rule-based reinforcement learning, we develop the Nemotron-Research-Tool-N1 series of tool-using language models using a similar training paradigm. Instead of restrictively supervising intermediate reasoning traces distilled from stronger models, Nemotron-Research-Tool-N1 is optimized with a binary reward that evaluates only the structural validity and functional correctness of tool invocations. This lightweight supervision allows the model to autonomously internalize reasoning strategies, without the need for annotated reasoning trajectories. Experiments on the BFCL and API-Bank benchmarks show that Nemotron-Research-Tool-N1-7B and Nemotron-Research-Tool-N1-14B, built on Qwen-2.5-7B/14B-Instruct, achieve state-of-the-art results, outperforming GPT-4o on both evaluations. 

**Abstract (ZH)**: 外部工具赋能的大语言模型已成为扩展其功能超越文本生成任务的关键策略。Nemotron-Research-Tool-N1系列工具使用语言模型通过类似训练范式借鉴DeepSeek-R1的成功经验，以二元奖励优化工具调用的结构有效性和功能正确性，实现轻量级监督，促进模型自主内化推理策略。实验表明，Nemotron-Research-Tool-N1-7B和Nemotron-Research-Tool-N1-14B在BFCL和API-Bank基准上的表现卓越，优于GPT-4o。 

---
# CORG: Generating Answers from Complex, Interrelated Contexts 

**Title (ZH)**: CORG: 从复杂相关背景中生成答案 

**Authors**: Hyunji Lee, Franck Dernoncourt, Trung Bui, Seunghyun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2505.00023)  

**Abstract**: In a real-world corpus, knowledge frequently recurs across documents but often contains inconsistencies due to ambiguous naming, outdated information, or errors, leading to complex interrelationships between contexts. Previous research has shown that language models struggle with these complexities, typically focusing on single factors in isolation. We classify these relationships into four types: distracting, ambiguous, counterfactual, and duplicated. Our analysis reveals that no single approach effectively addresses all these interrelationships simultaneously. Therefore, we introduce Context Organizer (CORG), a framework that organizes multiple contexts into independently processed groups. This design allows the model to efficiently find all relevant answers while ensuring disambiguation. CORG consists of three key components: a graph constructor, a reranker, and an aggregator. Our results demonstrate that CORG balances performance and efficiency effectively, outperforming existing grouping methods and achieving comparable results to more computationally intensive, single-context approaches. 

**Abstract (ZH)**: 在真实世界的语料库中，知识跨越多个文档频繁出现，但由于命名模糊、信息过时或错误，常常包含不一致性，导致上下文之间关系复杂。先前的研究表明，语言模型在处理这些复杂性时表现出色，通常侧重于孤立地考虑单一因素。我们将这些关系分类为四种类型：干扰性、模糊性、假设性逆反和重复。我们的分析揭示，没有单一的方法能够同时有效解决所有这些关系。因此，我们提出了Context Organizer (CORG)框架，该框架将多个上下文组织成独立处理的组。该设计允许模型高效地找到所有相关答案并确保去模糊化。CORG由三个关键组件组成：图构建器、重排序器和聚合器。我们的结果显示，CORG能够在性能和效率之间取得有效平衡，优于现有分组方法，并达到与计算密集型单一上下文方法可比的结果。 

---
# Ustnlp16 at SemEval-2025 Task 9: Improving Model Performance through Imbalance Handling and Focal Loss 

**Title (ZH)**: Ustnlp16 在 SemEval-2025 任务 9 中通过不平衡处理和焦 LOSS 提升模型性能 

**Authors**: Zhuoang Cai, Zhenghao Li, Yang Liu, Liyuan Guo, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.00021)  

**Abstract**: Classification tasks often suffer from imbal- anced data distribution, which presents chal- lenges in food hazard detection due to severe class imbalances, short and unstructured text, and overlapping semantic categories. In this paper, we present our system for SemEval- 2025 Task 9: Food Hazard Detection, which ad- dresses these issues by applying data augmenta- tion techniques to improve classification perfor- mance. We utilize transformer-based models, BERT and RoBERTa, as backbone classifiers and explore various data balancing strategies, including random oversampling, Easy Data Augmentation (EDA), and focal loss. Our ex- periments show that EDA effectively mitigates class imbalance, leading to significant improve- ments in accuracy and F1 scores. Furthermore, combining focal loss with oversampling and EDA further enhances model robustness, par- ticularly for hard-to-classify examples. These findings contribute to the development of more effective NLP-based classification models for food hazard detection. 

**Abstract (ZH)**: 食品危害检测任务往往受到不平衡数据分布的影响，这在严重类别不平衡、短且无结构的文本以及重叠的语义类别下尤其具有挑战性。在本文中，我们提出了我们的系统以应对SemEval-2025 Task 9：食品危害检测任务，通过应用数据增强技术来提高分类性能。我们利用基于Transformer的模型BERT和RoBERTa作为基础分类器，并探索了随机过采样、Easy Data Augmentation (EDA)和焦点损失等多种数据平衡策略。实验结果显示，EDA有效缓解了类别不平衡问题，显著提高了准确率和F1分数。此外，结合焦点损失与过采样和EDA进一步增强了模型的稳健性，特别是在难以分类的例子上。这些发现为开发更有效的基于NLP的食品危害检测分类模型做出了贡献。 

---
# The AI Co-Ethnographer: How Far Can Automation Take Qualitative Research? 

**Title (ZH)**: AI 共同民族志研究者：自动化能将定性研究推进多远？ 

**Authors**: Fabian Retkowski, Andreas Sudmann, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.00012)  

**Abstract**: Qualitative research often involves labor-intensive processes that are difficult to scale while preserving analytical depth. This paper introduces The AI Co-Ethnographer (AICoE), a novel end-to-end pipeline developed for qualitative research and designed to move beyond the limitations of simply automating code assignments, offering a more integrated approach. AICoE organizes the entire process, encompassing open coding, code consolidation, code application, and even pattern discovery, leading to a comprehensive analysis of qualitative data. 

**Abstract (ZH)**: 定性研究往往涉及劳动密集型过程，难以在保持分析深度的同时 scalability。本文介绍了《AI 共同民族志学者（AICoE）》，这是一种针对定性研究开发的端到端管道，旨在超越简单自动化代码分配的局限性，提供一种更集成的方法。AICoE 整合了整个过程，包括开放编码、代码整合、代码应用，甚至模式发现，从而实现定性数据的全面分析。 

---
# Toward a digital twin of U.S. Congress 

**Title (ZH)**: 向美国国会的数字孪生体迈进 

**Authors**: Hayden Helm, Tianyi Chen, Harvey McGuinness, Paige Lee, Brandon Duderstadt, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2505.00006)  

**Abstract**: In this paper we provide evidence that a virtual model of U.S. congresspersons based on a collection of language models satisfies the definition of a digital twin. In particular, we introduce and provide high-level descriptions of a daily-updated dataset that contains every Tweet from every U.S. congressperson during their respective terms. We demonstrate that a modern language model equipped with congressperson-specific subsets of this data are capable of producing Tweets that are largely indistinguishable from actual Tweets posted by their physical counterparts. We illustrate how generated Tweets can be used to predict roll-call vote behaviors and to quantify the likelihood of congresspersons crossing party lines, thereby assisting stakeholders in allocating resources and potentially impacting real-world legislative dynamics. We conclude with a discussion of the limitations and important extensions of our analysis. 

**Abstract (ZH)**: 本文提供了证据，证明基于语言模型集合构建的美国国会成员虚拟模型符合数字孪生的定义。特别是，我们引入并提供了包含每位美国国会成员在其任期内发布的每条推特的每日更新数据集。我们演示了装备有特定于国会成员数据子集的现代语言模型能够生成与物理国会成员实际发布的推特几乎无法区别的推特。我们展示了生成的推特如何用于预测投票行为，并量化国会成员跨越党派线的可能性，从而帮助利益相关者分配资源，并可能影响实际立法动态。最后，我们讨论了分析的局限性和重要扩展。 

---
