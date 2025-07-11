# Context Matters: Learning Generalizable Rewards via Calibrated Features 

**Title (ZH)**: 背景重要：通过校准特征学习可泛化的奖励 

**Authors**: Alexandra Forsey-Smerek, Julie Shah, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15012)  

**Abstract**: A key challenge in reward learning from human input is that desired agent behavior often changes based on context. Traditional methods typically treat each new context as a separate task with its own reward function. For example, if a previously ignored stove becomes too hot to be around, the robot must learn a new reward from scratch, even though the underlying preference for prioritizing safety over efficiency remains unchanged. We observe that context influences not the underlying preference itself, but rather the $\textit{saliency}$--or importance--of reward features. For instance, stove heat affects the importance of the robot's proximity, yet the human's safety preference stays the same. Existing multi-task and meta IRL methods learn context-dependent representations $\textit{implicitly}$--without distinguishing between preferences and feature importance--resulting in substantial data requirements. Instead, we propose $\textit{explicitly}$ modeling context-invariant preferences separately from context-dependent feature saliency, creating modular reward representations that adapt to new contexts. To achieve this, we introduce $\textit{calibrated features}$--representations that capture contextual effects on feature saliency--and present specialized paired comparison queries that isolate saliency from preference for efficient learning. Experiments with simulated users show our method significantly improves sample efficiency, requiring 10x fewer preference queries than baselines to achieve equivalent reward accuracy, with up to 15% better performance in low-data regimes (5-10 queries). An in-person user study (N=12) demonstrates that participants can effectively teach their unique personal contextual preferences using our method, enabling more adaptable and personalized reward learning. 

**Abstract (ZH)**: 从人类输入中学习奖励的关键挑战在于期望的代理行为往往基于上下文而变化。传统方法通常将每个新的上下文视为具有自己奖励函数的独立任务。例如，如果之前未被注意的烤炉变得过热而无法靠近时，机器人必须从头开始学习新的奖励，尽管其优先安全性而非效率的底层偏好并未改变。我们观察到，上下文影响的不是底层偏好本身，而是奖励特征的$\textit{显著性}$——即重要性。例如，烤炉的温度影响机器人靠近的显著性，但人类的安全偏好保持不变。现有的多任务和元IRL方法在不了解偏好的同时学习上下文依赖的表征，导致需要大量数据。相反，我们提出分别明确建模上下文不变的偏好和上下文依赖的特征显著性，创建模块化的奖励表示以适应新上下文。为此，我们引入了$\textit{校准特征}$——能够捕捉特征显著性受上下文影响的表示，并且提出了专门的配对比较查询以有效隔离显著性与偏好，从而使学习更加高效。仿真人实验显示，我们的方法显著提高了样本效率，只需基线方法的十分之一的偏好查询即可达到同等的奖励准确性，特别是在数据稀缺的条件下（5-10个查询时）可以提高多达15%的性能。面对面用户研究（N=12）表明，参与者能够有效使用我们的方法教授其独特的个人上下文偏好，从而实现更具适应性和个性化的奖励学习。 

---
# Minimizing Structural Vibrations via Guided Flow Matching Design Optimization 

**Title (ZH)**: 通过引导流匹配设计优化来最小化结构振动 

**Authors**: Jan van Delden, Julius Schultz, Sebastian Rothe, Christian Libner, Sabine C. Langer, Timo Lüddecke  

**Link**: [PDF](https://arxiv.org/pdf/2506.15263)  

**Abstract**: Structural vibrations are a source of unwanted noise in engineering systems like cars, trains or airplanes. Minimizing these vibrations is crucial for improving passenger comfort. This work presents a novel design optimization approach based on guided flow matching for reducing vibrations by placing beadings (indentations) in plate-like structures. Our method integrates a generative flow matching model and a surrogate model trained to predict structural vibrations. During the generation process, the flow matching model pushes towards manufacturability while the surrogate model pushes to low-vibration solutions. The flow matching model and its training data implicitly define the design space, enabling a broader exploration of potential solutions as no optimization of manually-defined design parameters is required. We apply our method to a range of differentiable optimization objectives, including direct optimization of specific eigenfrequencies through careful construction of the objective function. Results demonstrate that our method generates diverse and manufacturable plate designs with reduced structural vibrations compared to designs from random search, a criterion-based design heuristic and genetic optimization. The code and data are available from this https URL. 

**Abstract (ZH)**: 结构振动是汽车、列车或飞机等工程系统中的一个不必要的噪声来源。减少这些振动对于提高乘客舒适度至关重要。本文提出了一种基于引导流匹配的新型设计优化方法，通过在板状结构中放置凸台（凹陷）来减少振动。该方法结合了生成流匹配模型和一个用于预测结构振动的代理模型。在生成过程中，流匹配模型推动模型更具可制造性，而代理模型则推动低振动解决方案。流匹配模型及其训练数据隐式定义了设计空间，使得无需优化手动定义的设计参数即可进行更广泛的设计探索。我们将该方法应用于各种可微优化目标，包括通过精心构建目标函数直接优化特定的固有频率。结果表明，与随机搜索、基于准则的设计启发式方法和遗传优化生成的设计相比，我们的方法能够生成具有减少结构振动的多样化且可制造的板状设计。代码和数据可从该网址获得。 

---
# Probabilistic Trajectory GOSPA: A Metric for Uncertainty-Aware Multi-Object Tracking Performance Evaluation 

**Title (ZH)**: 概率轨迹GOSPA：一种面向不确定性多目标跟踪性能评估的度量标准 

**Authors**: Yuxuan Xia, Ángel F. García-Fernández, Johan Karlsson, Yu Ge, Lennart Svensson, Ting Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15148)  

**Abstract**: This paper presents a generalization of the trajectory general optimal sub-pattern assignment (GOSPA) metric for evaluating multi-object tracking algorithms that provide trajectory estimates with track-level uncertainties. This metric builds on the recently introduced probabilistic GOSPA metric to account for both the existence and state estimation uncertainties of individual object states. Similar to trajectory GOSPA (TGOSPA), it can be formulated as a multidimensional assignment problem, and its linear programming relaxation--also a valid metric--is computable in polynomial time. Additionally, this metric retains the interpretability of TGOSPA, and we show that its decomposition yields intuitive costs terms associated to expected localization error and existence probability mismatch error for properly detected objects, expected missed and false detection error, and track switch error. The effectiveness of the proposed metric is demonstrated through a simulation study. 

**Abstract (ZH)**: 本文提出了一种轨迹广义最优子模式分配（GOSPA）度量的一般化方法，用于评估提供具有轨迹级不确定性轨迹估计的多目标跟踪算法。该度量基于最近引入的概率GOSPA度量，以计算个体对象状态的存在性和状态估计不确定性。类似于轨迹GOSPA（TGOSPA），它可以形式化为一个多维分配问题，其线性规划 relaxation 也是一个有效的度量，并且可以在多项式时间内计算。此外，该度量保留了TGOSPA的可解释性，我们展示了其分解产生了与正确检测对象的期望定位误差和存在概率不匹配误差、预期的遗漏和假检测误差以及轨迹切换误差相关的直观成本项。通过仿真研究证明了所提出度量的有效性。 

---
# Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review 

**Title (ZH)**: 近期多Agent人类轨迹预测的研究进展：一篇综合Review 

**Authors**: Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea  

**Link**: [PDF](https://arxiv.org/pdf/2506.14831)  

**Abstract**: With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as autonomous navigation and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2024. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP. 

**Abstract (ZH)**: 基于深度学习的多agent轨迹预测 Recent advancements and future research directions in human trajectory prediction 

---
# The AI Policy Module: Developing Computer Science Student Competency in AI Ethics and Policy 

**Title (ZH)**: AI政策模块：培养计算机科学学生在AI伦理与政策方面的能力 

**Authors**: James Weichert, Daniel Dunlap, Mohammed Farghally, Hoda Eldardiry  

**Link**: [PDF](https://arxiv.org/pdf/2506.15639)  

**Abstract**: As artificial intelligence (AI) further embeds itself into many settings across personal and professional contexts, increasing attention must be paid not only to AI ethics, but also to the governance and regulation of AI technologies through AI policy. However, the prevailing post-secondary computing curriculum is currently ill-equipped to prepare future AI practitioners to confront increasing demands to implement abstract ethical principles and normative policy preferences into the design and development of AI systems. We believe that familiarity with the 'AI policy landscape' and the ability to translate ethical principles to practices will in the future constitute an important responsibility for even the most technically-focused AI engineers.
Toward preparing current computer science (CS) students for these new expectations, we developed an AI Policy Module to introduce discussions of AI policy into the CS curriculum. Building on a successful pilot in fall 2024, in this innovative practice full paper we present an updated and expanded version of the module, including a technical assignment on "AI regulation". We present the findings from our pilot of the AI Policy Module 2.0, evaluating student attitudes towards AI ethics and policy through pre- and post-module surveys. Following the module, students reported increased concern about the ethical impacts of AI technologies while also expressing greater confidence in their abilities to engage in discussions about AI regulation. Finally, we highlight the AI Regulation Assignment as an effective and engaging tool for exploring the limits of AI alignment and emphasizing the role of 'policy' in addressing ethical challenges. 

**Abstract (ZH)**: 随着人工智能（AI）进一步渗透到个人和专业环境中的众多领域，不仅需要关注AI伦理，还需要通过AI政策来治理和监管AI技术。当前的大学计算机课程尚不足以使未来的AI从业者准备好应对将抽象的伦理原则和规范性政策偏好融入AI系统设计和开发中的日益增长的要求。我们认为，熟悉“AI政策 landscape”并能够将伦理原则转化为实践将成为未来技术重点的AI工程师的重要责任。

为准备当前计算机科学（CS）学生应对这些新期望，我们开发了一个AI政策模块，将AI政策讨论引入CS课程中。基于2024年秋季成功的试点项目，本文介绍了更新和扩展后的模块版本，并包括了一个“AI监管”技术作业。本文报告了AI政策模块2.0试点项目的成果，通过模块前后的调查评估了学生对AI伦理和政策的态度变化。学生在完成模块后报告称，他们对AI技术的伦理影响更加关注，并表达了在讨论AI监管方面更强的信心。最后，本文强调AI监管作业是一个有效且有趣的工具，用于探讨AI对齐的局限性，并强调“政策”在解决伦理挑战中的作用。 

---
# Joint Computation Offloading and Resource Allocation for Uncertain Maritime MEC via Cooperation of UAVs and Vessels 

**Title (ZH)**: 基于UAV和船舶协同的不确定性海洋MEC的联合计算卸载与资源分配 

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.15225)  

**Abstract**: The computation demands from the maritime Internet of Things (MIoT) increase rapidly in recent years, and the unmanned aerial vehicles (UAVs) and vessels based multi-access edge computing (MEC) can fulfill these MIoT requirements. However, the uncertain maritime tasks present significant challenges of inefficient computation offloading and resource allocation. In this paper, we focus on the maritime computation offloading and resource allocation through the cooperation of UAVs and vessels, with consideration of uncertain tasks. Specifically, we propose a cooperative MEC framework for computation offloading and resource allocation, including MIoT devices, UAVs and vessels. Then, we formulate the optimization problem to minimize the total execution time. As for the uncertain MIoT tasks, we leverage Lyapunov optimization to tackle the unpredictable task arrivals and varying computational resource availability. 
By converting the long-term constraints into short-term constraints, we obtain a set of small-scale optimization problems. Further, considering the heterogeneity of actions and resources of UAVs and vessels, we reformulate the small-scale optimization problem into a Markov game (MG). Moreover, a heterogeneous-agent soft actor-critic is proposed to sequentially update various neural networks and effectively solve the MG problem. 
Finally, simulations are conducted to verify the effectiveness in addressing computational offloading and resource allocation. 

**Abstract (ZH)**: 海洋物联网计算卸载与资源分配中的无人飞行器和船只合作框架研究 

---
# Truncated Proximal Policy Optimization 

**Title (ZH)**: 截断 proximity 策略优化 

**Authors**: Tiantian Fan, Lingjun Liu, Yu Yue, Jiaze Chen, Chengyi Wang, Qiying Yu, Chi Zhang, Zhiqi Lin, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Bole Ma, Mofan Zhang, Gaohong Liu, Ru Zhang, Haotian Zhou, Cong Xie, Ruidong Zhu, Zhi Zhang, Xin Liu, Mingxuan Wang, Lin Yan, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15050)  

**Abstract**: Recently, test-time scaling Large Language Models (LLMs) have demonstrated exceptional reasoning capabilities across scientific and professional tasks by generating long chains-of-thought (CoT). As a crucial component for developing these reasoning models, reinforcement learning (RL), exemplified by Proximal Policy Optimization (PPO) and its variants, allows models to learn through trial and error. However, PPO can be time-consuming due to its inherent on-policy nature, which is further exacerbated by increasing response lengths. In this work, we propose Truncated Proximal Policy Optimization (T-PPO), a novel extension to PPO that improves training efficiency by streamlining policy update and length-restricted response generation. T-PPO mitigates the issue of low hardware utilization, an inherent drawback of fully synchronized long-generation procedures, where resources often sit idle during the waiting periods for complete rollouts. Our contributions are two-folds. First, we propose Extended Generalized Advantage Estimation (EGAE) for advantage estimation derived from incomplete responses while maintaining the integrity of policy learning. Second, we devise a computationally optimized mechanism that allows for the independent optimization of the policy and value models. By selectively filtering prompt and truncated tokens, this mechanism reduces redundant computations and accelerates the training process without sacrificing convergence performance. We demonstrate the effectiveness and efficacy of T-PPO on AIME 2024 with a 32B base model. The experimental results show that T-PPO improves the training efficiency of reasoning LLMs by up to 2.5x and outperforms its existing competitors. 

**Abstract (ZH)**: Recently, 测试时缩放大型语言模型 (LLMs) 通过生成长链式思考 (CoT) 在科学和专业任务中展现了卓越的推理能力。通过代理强化学习 (RL)，以近端策略优化 (PPO) 及其变种为代表，模型能够通过尝试和错误来学习。然而，PPO 由于其固有的在线策略性质，响应长度增加时，训练过程可能会变得耗时。在本工作中，我们提出了截断近端策略优化 (T-PPO)，这是一种对 PPO 的新颖扩展，通过简化策略更新和长度受限的响应生成来提高训练效率。T-PPO 缓解了全同步长生成过程中固有的硬件利用率低的问题，即在完整采样期间，资源往往处于闲置状态。我们的贡献包括两方面。首先，我们提出了扩展的一般优势估计 (EGAE)，用于从不完整响应中进行优势估计，同时保持策略学习的完整性。其次，我们设计了一种计算优化机制，允许策略模型和价值模型独立优化。通过选择性过滤提示和截断的令牌，该机制减少冗余计算，加速训练过程而不牺牲收敛性能。我们使用 AIME 2024 和 32B 基模型验证了 T-PPO 的有效性和效率。实验结果表明，T-PPO 将推理 LLMs 的训练效率提高了 2.5 倍，并且优于其现有竞争对手。 

---
# MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning 

**Title (ZH)**: MEAL：持续多智能体强化学习的标准基准 

**Authors**: Tristan Tomilin, Luka van den Boogaard, Samuel Garcin, Bram Grooten, Meng Fang, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2506.14990)  

**Abstract**: Benchmarks play a crucial role in the development and analysis of reinforcement learning (RL) algorithms, with environment availability strongly impacting research. One particularly underexplored intersection is continual learning (CL) in cooperative multi-agent settings. To remedy this, we introduce MEAL (Multi-agent Environments for Adaptive Learning), the first benchmark tailored for continual multi-agent reinforcement learning (CMARL). Existing CL benchmarks run environments on the CPU, leading to computational bottlenecks and limiting the length of task sequences. MEAL leverages JAX for GPU acceleration, enabling continual learning across sequences of 100 tasks on a standard desktop PC in a few hours. We show that naively combining popular CL and MARL methods yields strong performance on simple environments, but fails to scale to more complex settings requiring sustained coordination and adaptation. Our ablation study identifies architectural and algorithmic features critical for CMARL on MEAL. 

**Abstract (ZH)**: 基于MEAL的持续多代理强化学习基准 

---
# AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning 

**Title (ZH)**: AutoRule: 基于推理链的规则提取奖励改进偏好学习 

**Authors**: Tevin Wang, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.15651)  

**Abstract**: Rule-based rewards offer a promising strategy for improving reinforcement learning from human feedback (RLHF), but current approaches often rely on manual rule engineering. We present AutoRule, a fully automated method for extracting rules from preference feedback and formulating them into rule-based rewards. AutoRule extraction operates in three stages: it leverages a reasoning model to interpret user preferences, identifies candidate rules from the reasoning chain of these interpretations, and synthesizes them into a unified rule set. Leveraging the finalized rule set, we employ language-model verifiers to compute the fraction of rules satisfied by each output, using this metric as an auxiliary reward alongside the learned reward model during policy optimization. Training a Llama-3-8B model with AutoRule results in a 28.6\% relative improvement in length-controlled win rate on AlpacaEval2.0, and a 6.1\% relative gain in second-turn performance on a held-out MT-Bench subset, compared to a GRPO baseline trained with the same learned reward model but without the rule-based auxiliary reward. Our analysis confirms that the extracted rules exhibit good agreement with dataset preference. We find that AutoRule demonstrates reduced reward hacking compared to a learned reward model when run over two episodes. Finally, our case study suggests that the extracted rules capture unique qualities valued in different datasets. The extracted rules are provided in the appendix, and the code is open-sourced at this https URL. 

**Abstract (ZH)**: 基于规则的奖励规则自动化提取为人类反馈强化学习（RLHF）提供了一种有希望的策略，但现有方法往往依赖于手动规则工程。我们提出AutoRule，这是一种完全自动化的从偏好反馈中提取规则并将其格式化为基于规则的奖励的方法。AutoRule 提取过程分为三个阶段：利用推理模型解释用户偏好、从这些解释的推理链中识别候选规则，并将它们合成统一的规则集。利用最终确定的规则集，我们使用语言模型验证器计算每个输出满足规则的比例，并将此指标作为策略优化期间辅助奖励与学习奖励模型的一致奖励。使用AutoRule训练Llama-3-8B模型在AlpacaEval2.0上的长度控制获胜率相对提高28.6%，在保留的MT-Bench子集上第二轮性能相对提升6.1%，相较于使用相同学习奖励模型但没有基于规则的辅助奖励的GRPO基线训练模型。我们的分析确认提取的规则与数据集偏好具有良好的一致性。我们发现，当在两个回合中运行时，AutoRule与学习奖励模型相比表现出较低的奖励作弊倾向。最后，我们的案例研究表明，提取的规则捕捉到了不同数据集中独特的价值属性。提取的规则附在附录中，代码开源于此网址。 

---
# Federated Learning for MRI-based BrainAGE: a multicenter study on post-stroke functional outcome prediction 

**Title (ZH)**: 基于MRI的脑年龄联邦学习：多中心Stroke后功能结局预测研究 

**Authors**: Vincent Roca, Marc Tommasi, Paul Andrey, Aurélien Bellet, Markus D. Schirmer, Hilde Henon, Laurent Puy, Julien Ramon, Grégory Kuchcinski, Martin Bretzner, Renaud Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2506.15626)  

**Abstract**: $\textbf{Objective:}$ Brain-predicted age difference (BrainAGE) is a neuroimaging biomarker reflecting brain health. However, training robust BrainAGE models requires large datasets, often restricted by privacy concerns. This study evaluates the performance of federated learning (FL) for BrainAGE estimation in ischemic stroke patients treated with mechanical thrombectomy, and investigates its association with clinical phenotypes and functional outcomes.
$\textbf{Methods:}$ We used FLAIR brain images from 1674 stroke patients across 16 hospital centers. We implemented standard machine learning and deep learning models for BrainAGE estimates under three data management strategies: centralized learning (pooled data), FL (local training at each site), and single-site learning. We reported prediction errors and examined associations between BrainAGE and vascular risk factors (e.g., diabetes mellitus, hypertension, smoking), as well as functional outcomes at three months post-stroke. Logistic regression evaluated BrainAGE's predictive value for these outcomes, adjusting for age, sex, vascular risk factors, stroke severity, time between MRI and arterial puncture, prior intravenous thrombolysis, and recanalisation outcome.
$\textbf{Results:}$ While centralized learning yielded the most accurate predictions, FL consistently outperformed single-site models. BrainAGE was significantly higher in patients with diabetes mellitus across all models. Comparisons between patients with good and poor functional outcomes, and multivariate predictions of these outcomes showed the significance of the association between BrainAGE and post-stroke recovery.
$\textbf{Conclusion:}$ FL enables accurate age predictions without data centralization. The strong association between BrainAGE, vascular risk factors, and post-stroke recovery highlights its potential for prognostic modeling in stroke care. 

**Abstract (ZH)**: objectives: Brain-predicted年龄difference (BrainAGE) 是一个反映大脑健康的神经影像学生物标志物。然而，训练稳健的BrainAGE模型需要大量数据，通常受到隐私问题的限制。本研究评估了联邦学习（FL）在机械取栓治疗的缺血性中风患者中预测BrainAGE的表现，并探讨了其与临床表型和功能预后的关系。 

---
# GFLC: Graph-based Fairness-aware Label Correction for Fair Classification 

**Title (ZH)**: 基于图的公平感知标签修正以实现公平分类 

**Authors**: Modar Sulaiman, Kallol Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.15620)  

**Abstract**: Fairness in machine learning (ML) has a critical importance for building trustworthy machine learning system as artificial intelligence (AI) systems increasingly impact various aspects of society, including healthcare decisions and legal judgments. Moreover, numerous studies demonstrate evidence of unfair outcomes in ML and the need for more robust fairness-aware methods. However, the data we use to train and develop debiasing techniques often contains biased and noisy labels. As a result, the label bias in the training data affects model performance and misrepresents the fairness of classifiers during testing. To tackle this problem, our paper presents Graph-based Fairness-aware Label Correction (GFLC), an efficient method for correcting label noise while preserving demographic parity in datasets. In particular, our approach combines three key components: prediction confidence measure, graph-based regularization through Ricci-flow-optimized graph Laplacians, and explicit demographic parity incentives. Our experimental findings show the effectiveness of our proposed approach and show significant improvements in the trade-off between performance and fairness metrics compared to the baseline. 

**Abstract (ZH)**: 基于图的公平性aware标签矫正（GFLC）：在保持人口统计学平等的情况下纠正标签噪声 

---
# From Model to Classroom: Evaluating Generated MCQs for Portuguese with Narrative and Difficulty Concerns 

**Title (ZH)**: 从模型到课堂：基于叙事和难度考量评估生成的葡萄牙语选择题 

**Authors**: Bernardo Leite, Henrique Lopes Cardoso, Pedro Pinto, Abel Ferreira, Luís Abreu, Isabel Rangel, Sandra Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.15598)  

**Abstract**: While MCQs are valuable for learning and evaluation, manually creating them with varying difficulty levels and targeted reading skills remains a time-consuming and costly task. Recent advances in generative AI provide an opportunity to automate MCQ generation efficiently. However, assessing the actual quality and reliability of generated MCQs has received limited attention -- particularly regarding cases where generation fails. This aspect becomes particularly important when the generated MCQs are meant to be applied in real-world settings. Additionally, most MCQ generation studies focus on English, leaving other languages underexplored. This paper investigates the capabilities of current generative models in producing MCQs for reading comprehension in Portuguese, a morphologically rich language. Our study focuses on generating MCQs that align with curriculum-relevant narrative elements and span different difficulty levels. We evaluate these MCQs through expert review and by analyzing the psychometric properties extracted from student responses to assess their suitability for elementary school students. Our results show that current models can generate MCQs of comparable quality to human-authored ones. However, we identify issues related to semantic clarity and answerability. Also, challenges remain in generating distractors that engage students and meet established criteria for high-quality MCQ option design. 

**Abstract (ZH)**: 生成式AI在自动生成葡萄牙语阅读理解选择题中的能力研究 

---
# Towards Explainable Indoor Localization: Interpreting Neural Network Learning on Wi-Fi Fingerprints Using Logic Gates 

**Title (ZH)**: 基于逻辑门解释的室内定位可解释性研究：解析Wi-Fi指纹学习過程 

**Authors**: Danish Gufran, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15559)  

**Abstract**: Indoor localization using deep learning (DL) has demonstrated strong accuracy in mapping Wi-Fi RSS fingerprints to physical locations; however, most existing DL frameworks function as black-box models, offering limited insight into how predictions are made or how models respond to real-world noise over time. This lack of interpretability hampers our ability to understand the impact of temporal variations - caused by environmental dynamics - and to adapt models for long-term reliability. To address this, we introduce LogNet, a novel logic gate-based framework designed to interpret and enhance DL-based indoor localization. LogNet enables transparent reasoning by identifying which access points (APs) are most influential for each reference point (RP) and reveals how environmental noise disrupts DL-driven localization decisions. This interpretability allows us to trace and diagnose model failures and adapt DL systems for more stable long-term deployments. Evaluations across multiple real-world building floorplans and over two years of temporal variation show that LogNet not only interprets the internal behavior of DL models but also improves performance-achieving up to 1.1x to 2.8x lower localization error, 3.4x to 43.3x smaller model size, and 1.5x to 3.6x lower latency compared to prior DL-based models. 

**Abstract (ZH)**: 基于逻辑门的室内定位解释框架LogNet：提高DL模型的可解释性和性能 

---
# DAILOC: Domain-Incremental Learning for Indoor Localization using Smartphones 

**Title (ZH)**: DAILOC：基于智能手机的领域增量学习室内定位方法 

**Authors**: Akhil Singampalli, Danish Gufran, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15554)  

**Abstract**: Wi-Fi fingerprinting-based indoor localization faces significant challenges in real-world deployments due to domain shifts arising from device heterogeneity and temporal variations within indoor environments. Existing approaches often address these issues independently, resulting in poor generalization and susceptibility to catastrophic forgetting over time. In this work, we propose DAILOC, a novel domain-incremental learning framework that jointly addresses both temporal and device-induced domain shifts. DAILOC introduces a novel disentanglement strategy that separates domain shifts from location-relevant features using a multi-level variational autoencoder. Additionally, we introduce a novel memory-guided class latent alignment mechanism to address the effects of catastrophic forgetting over time. Experiments across multiple smartphones, buildings, and time instances demonstrate that DAILOC significantly outperforms state-of-the-art methods, achieving up to 2.74x lower average error and 4.6x lower worst-case error. 

**Abstract (ZH)**: 基于Wi-Fi指纹的室内定位在实际部署中面临着因设备异质性和室内环境内的时域变化引起的大域转移问题，现有方法往往独立解决这些问题，导致泛化能力差且容易在长时间内发生灾难性遗忘。本文提出DAILOC，一种新的大域增量学习框架，同时解决时域和设备引起的领域转移问题。DAILOC引入了一种新颖的分离策略，利用多级变异自编码器将领域转移与位置相关的特征分离，并引入了一种新颖的基于记忆的类别潜在对齐机制来应对时间内的灾难性遗忘。实验表明，DAILOC在多个智能手机、建筑物和时间实例上显著优于现有方法，平均误差降低了2.74倍，最坏情况误差降低了4.6倍。 

---
# Learning Algorithms in the Limit 

**Title (ZH)**: 限届学习算法 

**Authors**: Hristo Papazov, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2506.15543)  

**Abstract**: This paper studies the problem of learning computable functions in the limit by extending Gold's inductive inference framework to incorporate \textit{computational observations} and \textit{restricted input sources}. Complimentary to the traditional Input-Output Observations, we introduce Time-Bound Observations, and Policy-Trajectory Observations to study the learnability of general recursive functions under more realistic constraints. While input-output observations do not suffice for learning the class of general recursive functions in the limit, we overcome this learning barrier by imposing computational complexity constraints or supplementing with approximate time-bound observations. Further, we build a formal framework around observations of \textit{computational agents} and show that learning computable functions from policy trajectories reduces to learning rational functions from input and output, thereby revealing interesting connections to finite-state transducer inference. On the negative side, we show that computable or polynomial-mass characteristic sets cannot exist for the class of linear-time computable functions even for policy-trajectory observations. 

**Abstract (ZH)**: 本文通过将Gold的归纳推理框架扩展以融入计算观察和受限输入源，研究在极限学习可计算函数的问题。作为传统输入-输出观察的补充，我们引入了时间限定观察和策略-轨迹观察，以在更现实的约束条件下研究通用递归函数的学习能力。尽管输入-输出观察不足以在极限学习中覆盖所有通用递归函数类，但通过施加计算复杂性约束或补充近似时间限定观察，我们克服了这一学习障碍。我们围绕计算代理的观察建立了一套正式框架，表明从策略轨迹学习可计算函数等价于从输入和输出学习有理函数，从而揭示了有限状态转换推测的有趣联系。在负面方面，我们证明即使在策略-轨迹观察下，线性时间可计算函数或多项式质量特征集也不存在。 

---
# Intrinsic and Extrinsic Organized Attention: Softmax Invariance and Network Sparsity 

**Title (ZH)**: 内在与外在组织的注意力：Softmax 不变性与网络稀疏性 

**Authors**: Oluwadamilola Fasina, Ruben V.C. Pohle, Pei-Chun Su, Ronald R. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2506.15541)  

**Abstract**: We examine the intrinsic (within the attention head) and extrinsic (amongst the attention heads) structure of the self-attention mechanism in transformers. Theoretical evidence for invariance of the self-attention mechanism to softmax activation is obtained by appealing to paradifferential calculus, (and is supported by computational examples), which relies on the intrinsic organization of the attention heads. Furthermore, we use an existing methodology for hierarchical organization of tensors to examine network structure by constructing hierarchal partition trees with respect to the query, key, and head axes of network 3-tensors. Such an organization is consequential since it allows one to profitably execute common signal processing tasks on a geometry where the organized network 3-tensors exhibit regularity. We exemplify this qualitatively, by visualizing the hierarchical organization of the tree comprised of attention heads and the diffusion map embeddings, and quantitatively by investigating network sparsity with the expansion coefficients of individual attention heads and the entire network with respect to the bi and tri-haar bases (respectively) on the space of queries, keys, and heads of the network. To showcase the utility of our theoretical and methodological findings, we provide computational examples using vision and language transformers. The ramifications of these findings are two-fold: (1) a subsequent step in interpretability analysis is theoretically admitted, and can be exploited empirically for downstream interpretability tasks (2) one can use the network 3-tensor organization for empirical network applications such as model pruning (by virtue of network sparsity) and network architecture comparison. 

**Abstract (ZH)**: 我们探讨了变压器中自注意力机制的内在（_within each attention head_）与外在（_among the attention heads_）结构。通过调和差分微积分，获得了自注意力机制对softmax激活不变性的理论证据（并由计算例子支持），这一证据依赖于注意力头的内在组织结构。此外，我们利用现有的张量分层组织方法，通过构建与网络3张量的query、key和head轴相关的分层分区树，来研究网络结构。这种组织方式是有重要意义的，因为它允许我们在3张量有序性表现出规律性的几何上执行诸如信号处理等任务。我们从定性的角度通过可视化由注意力头和扩散映射嵌入组成的树的分层组织来举例说明这一点，从定量的角度通过检查各个注意力头以及整个网络在查询、键和头空间中相对于双haar和三haar基的展开系数来研究网络稀疏性。为了展示我们理论和方法论发现的应用价值，我们提供了视觉和语言变压器的计算例子。这些发现的影响是两方面的：（1）随后的可解释性分析步骤具有理论上的可行性，可以为下游可解释性任务提供经验支持；（2）可以利用3张量的网络组织来进行如模型剪枝（通过网络稀疏性）和网络架构比较等经验性网络应用。 

---
# Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework 

**Title (ZH)**: 用PRISM捕捉多义性：一种多概念特征描述框架 

**Authors**: Laura Kopf, Nils Feldhus, Kirill Bykov, Philine Lou Bommer, Anna Hedström, Marina M.-C. Höhne, Oliver Eberle  

**Link**: [PDF](https://arxiv.org/pdf/2506.15538)  

**Abstract**: Automated interpretability research aims to identify concepts encoded in neural network features to enhance human understanding of model behavior. Current feature description methods face two critical challenges: limited robustness and the flawed assumption that each neuron encodes only a single concept (monosemanticity), despite growing evidence that neurons are often polysemantic. This assumption restricts the expressiveness of feature descriptions and limits their ability to capture the full range of behaviors encoded in model internals. To address this, we introduce Polysemantic FeatuRe Identification and Scoring Method (PRISM), a novel framework that captures the inherent complexity of neural network features. Unlike prior approaches that assign a single description per feature, PRISM provides more nuanced descriptions for both polysemantic and monosemantic features. We apply PRISM to language models and, through extensive benchmarking against existing methods, demonstrate that our approach produces more accurate and faithful feature descriptions, improving both overall description quality (via a description score) and the ability to capture distinct concepts when polysemanticity is present (via a polysemanticity score). 

**Abstract (ZH)**: 基于多义性特征识别和评分方法的自动可解释性研究 

---
# Over-squashing in Spatiotemporal Graph Neural Networks 

**Title (ZH)**: 时空图神经网络中的过度压缩 

**Authors**: Ivan Marisca, Jacob Bamberger, Cesare Alippi, Michael M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.15507)  

**Abstract**: Graph Neural Networks (GNNs) have achieved remarkable success across various domains. However, recent theoretical advances have identified fundamental limitations in their information propagation capabilities, such as over-squashing, where distant nodes fail to effectively exchange information. While extensively studied in static contexts, this issue remains unexplored in Spatiotemporal GNNs (STGNNs), which process sequences associated with graph nodes. Nonetheless, the temporal dimension amplifies this challenge by increasing the information that must be propagated. In this work, we formalize the spatiotemporal over-squashing problem and demonstrate its distinct characteristics compared to the static case. Our analysis reveals that counterintuitively, convolutional STGNNs favor information propagation from points temporally distant rather than close in time. Moreover, we prove that architectures that follow either time-and-space or time-then-space processing paradigms are equally affected by this phenomenon, providing theoretical justification for computationally efficient implementations. We validate our findings on synthetic and real-world datasets, providing deeper insights into their operational dynamics and principled guidance for more effective designs. 

**Abstract (ZH)**: Graph神经网络（GNNs）在各个领域取得了显著的成功。然而，近年来的理论进展揭示了它们在信息传播能力上的根本限制，例如过榨干现象，即远距离节点无法有效交换信息。虽然在静态环境中这项问题得到了广泛研究，但在处理与图节点关联的时间空间序列的时空GNNs（STGNNs）中，这一问题仍未被探索。而时间维度的增加通过提高需要传播的信息量来加剧这一挑战。在本文中，我们正式化了时空过榨干问题，并展示了它与静态情况的区别特征。我们的分析表明，出乎意料的是，卷积时空GNNs更倾向于从时间上远离的节点传播信息，而不是时间上接近的节点。此外，我们证明了遵循时间和空间或时间后处理空间处理范式的架构都会受到这种现象的影响，从而为高效实现提供了理论依据。我们在合成数据集和真实世界数据集上验证了我们的发现，提供了对其操作动态的更深入见解，并为更有效的设计提供了原则性指导。 

---
# Pixel-level Certified Explanations via Randomized Smoothing 

**Title (ZH)**: 像素级认证解释方法：随机平滑 

**Authors**: Alaa Anani, Tobias Lorenz, Mario Fritz, Bernt Schiele  

**Link**: [PDF](https://arxiv.org/pdf/2506.15499)  

**Abstract**: Post-hoc attribution methods aim to explain deep learning predictions by highlighting influential input pixels. However, these explanations are highly non-robust: small, imperceptible input perturbations can drastically alter the attribution map while maintaining the same prediction. This vulnerability undermines their trustworthiness and calls for rigorous robustness guarantees of pixel-level attribution scores. We introduce the first certification framework that guarantees pixel-level robustness for any black-box attribution method using randomized smoothing. By sparsifying and smoothing attribution maps, we reformulate the task as a segmentation problem and certify each pixel's importance against $\ell_2$-bounded perturbations. We further propose three evaluation metrics to assess certified robustness, localization, and faithfulness. An extensive evaluation of 12 attribution methods across 5 ImageNet models shows that our certified attributions are robust, interpretable, and faithful, enabling reliable use in downstream tasks. Our code is at this https URL. 

**Abstract (ZH)**: 后验归因方法通过突出显示有影响力的输入像素来解释深度学习预测，但这些解释高度不 robust：即使是不可感知的输入扰动也可能大幅改变归因图，同时保持相同的预测。这种脆弱性损害了其可靠性，并要求像素级归因得分的严格 robustness 保证。我们引入了第一个使用随机化平滑技术为任何黑盒归因方法提供像素级 robustness 保证的认证框架。通过稀疏化和平滑化归因图，我们将任务重新形式化为分割问题，并对每个像素的重要性在 $\ell_2$ 界定的扰动下进行认证。我们进一步提出了三种评估认证 robustness、定位和忠实地度量标准。在对 5 种 ImageNet 模型的 12 种归因方法进行广泛评估后，结果显示我们认证的归因是 robust、可解释且忠实的，可在下游任务中可靠使用。我们的代码位于 <https://>。 

---
# Co-Creative Learning via Metropolis-Hastings Interaction between Humans and AI 

**Title (ZH)**: 基于人类与AI之间梅尔索-哈斯廷斯交互的协同创作学习 

**Authors**: Ryota Okumura, Tadahiro Taniguchi, Akira Taniguchi, Yoshinobu Hagiwara  

**Link**: [PDF](https://arxiv.org/pdf/2506.15468)  

**Abstract**: We propose co-creative learning as a novel paradigm where humans and AI, i.e., biological and artificial agents, mutually integrate their partial perceptual information and knowledge to construct shared external representations, a process we interpret as symbol emergence. Unlike traditional AI teaching based on unilateral knowledge transfer, this addresses the challenge of integrating information from inherently different modalities. We empirically test this framework using a human-AI interaction model based on the Metropolis-Hastings naming game (MHNG), a decentralized Bayesian inference mechanism. In an online experiment, 69 participants played a joint attention naming game (JA-NG) with one of three computer agent types (MH-based, always-accept, or always-reject) under partial observability. Results show that human-AI pairs with an MH-based agent significantly improved categorization accuracy through interaction and achieved stronger convergence toward a shared sign system. Furthermore, human acceptance behavior aligned closely with the MH-derived acceptance probability. These findings provide the first empirical evidence for co-creative learning emerging in human-AI dyads via MHNG-based interaction. This suggests a promising path toward symbiotic AI systems that learn with humans, rather than from them, by dynamically aligning perceptual experiences, opening a new venue for symbiotic AI alignment. 

**Abstract (ZH)**: 我们提出共创学习作为一种新型范式，人类和AI，即生物和人工代理，相互整合其部分感知信息和知识以构建共享的外部表示，我们认为这一过程是符号 emergence。不同于传统的单向知识转移的AI教学，这种方法解决了来自固有不同模态的信息整合挑战。我们通过基于Metropolis-Hastings命名游戏（MHNG）的人机交互模型，利用分散的贝叶斯推断机制，进行了实证测试。在线实验中，69名参与者在部分可观测性条件下，与三种计算机代理类型之一（基于MH、总是接受或总是拒绝）共同参与注意力共享命名游戏（JA-NG）。结果显示，与基于MH的代理交互的人机配对在交互中显著提高了分类准确性，并且更趋于达成共享符号系统的更强一致性。此外，人类的接受行为与MH推断出的接受概率高度一致。这些发现提供了共创学习在基于MHNG的人机交互中首次实证证据。这表明，通过动态对齐感知体验，朝向共生AI系统的潜在路径，这种系统与人类共同学习而非仅从人类学习，从而为共生AI对齐开辟新的前景。 

---
# Warping and Matching Subsequences Between Time Series 

**Title (ZH)**: 时间序列之间的时间 warp 和子序列匹配 

**Authors**: Simiao Lin, Wannes Meert, Pieter Robberechts, Hendrik Blockeel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15452)  

**Abstract**: Comparing time series is essential in various tasks such as clustering and classification. While elastic distance measures that allow warping provide a robust quantitative comparison, a qualitative comparison on top of them is missing. Traditional visualizations focus on point-to-point alignment and do not convey the broader structural relationships at the level of subsequences. This limitation makes it difficult to understand how and where one time series shifts, speeds up or slows down with respect to another. To address this, we propose a novel technique that simplifies the warping path to highlight, quantify and visualize key transformations (shift, compression, difference in amplitude). By offering a clearer representation of how subsequences match between time series, our method enhances interpretability in time series comparison. 

**Abstract (ZH)**: 在聚类和分类等多种任务中，比较时间序列是必不可少的。虽然允许拉伸的弹性距离度量可以提供稳健的定量比较，但其上的定性比较却缺失。传统的可视化方法侧重于点对点对齐，无法传达子序列层面的 broader 结构关系。这一限制使得理解一个时间序列相对于另一个如何、何时以及在何处发生位移、加速或减速变得困难。为了解决这个问题，我们提出了一种新颖的技术，简化拉伸路径以突出、量化和可视化关键转换（位移、压缩、振幅差异）。通过提供时间序列之间子序列匹配的更清晰表示，我们的方法增强了时间序列比较的可解释性。 

---
# Zero-Shot Reinforcement Learning Under Partial Observability 

**Title (ZH)**: 零样本部分可观测性的强化学习 

**Authors**: Scott Jeen, Tom Bewley, Jonathan M. Cullen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15446)  

**Abstract**: Recent work has shown that, under certain assumptions, zero-shot reinforcement learning (RL) methods can generalise to any unseen task in an environment after reward-free pre-training. Access to Markov states is one such assumption, yet, in many real-world applications, the Markov state is only partially observable. Here, we explore how the performance of standard zero-shot RL methods degrades when subjected to partially observability, and show that, as in single-task RL, memory-based architectures are an effective remedy. We evaluate our memory-based zero-shot RL methods in domains where the states, rewards and a change in dynamics are partially observed, and show improved performance over memory-free baselines. Our code is open-sourced via: this https URL. 

**Abstract (ZH)**: 近期研究表明，在某些假设下，零样本强化学习（RL）方法在奖励免费预训练之后，可以在环境中任何未见过的任务上泛化。当Markov状态部分可观测时，标准的零样本RL方法的性能会下降，类似单任务RL，基于记忆的架构是一种有效的解决方案。我们在部分可观测状态、奖励和动力学变化的领域评估了我们的基于记忆的零样本RL方法，并展示了相对于无记忆基线的性能提升。我们的代码在以下地址开源：this https URL。 

---
# Reward Models in Deep Reinforcement Learning: A Survey 

**Title (ZH)**: 深度强化学习中的奖励模型：一个综述 

**Authors**: Rui Yu, Shenghua Wan, Yucen Wang, Chen-Xiao Gao, Le Gan, Zongzhang Zhang, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15421)  

**Abstract**: In reinforcement learning (RL), agents continually interact with the environment and use the feedback to refine their behavior. To guide policy optimization, reward models are introduced as proxies of the desired objectives, such that when the agent maximizes the accumulated reward, it also fulfills the task designer's intentions. Recently, significant attention from both academic and industrial researchers has focused on developing reward models that not only align closely with the true objectives but also facilitate policy optimization. In this survey, we provide a comprehensive review of reward modeling techniques within the deep RL literature. We begin by outlining the background and preliminaries in reward modeling. Next, we present an overview of recent reward modeling approaches, categorizing them based on the source, the mechanism, and the learning paradigm. Building on this understanding, we discuss various applications of these reward modeling techniques and review methods for evaluating reward models. Finally, we conclude by highlighting promising research directions in reward modeling. Altogether, this survey includes both established and emerging methods, filling the vacancy of a systematic review of reward models in current literature. 

**Abstract (ZH)**: 在强化学习中的reward模型技术综述：从背景到评估与展望 

---
# Unifying VXAI: A Systematic Review and Framework for the Evaluation of Explainable AI 

**Title (ZH)**: 统一VXAI：可解释人工智能的系统回顾与评估框架 

**Authors**: David Dembinsky, Adriano Lucieri, Stanislav Frolov, Hiba Najjar, Ko Watanabe, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15408)  

**Abstract**: Modern AI systems frequently rely on opaque black-box models, most notably Deep Neural Networks, whose performance stems from complex architectures with millions of learned parameters. While powerful, their complexity poses a major challenge to trustworthiness, particularly due to a lack of transparency. Explainable AI (XAI) addresses this issue by providing human-understandable explanations of model behavior. However, to ensure their usefulness and trustworthiness, such explanations must be rigorously evaluated. Despite the growing number of XAI methods, the field lacks standardized evaluation protocols and consensus on appropriate metrics. To address this gap, we conduct a systematic literature review following the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines and introduce a unified framework for the eValuation of XAI (VXAI). We identify 362 relevant publications and aggregate their contributions into 41 functionally similar metric groups. In addition, we propose a three-dimensional categorization scheme spanning explanation type, evaluation contextuality, and explanation quality desiderata. Our framework provides the most comprehensive and structured overview of VXAI to date. It supports systematic metric selection, promotes comparability across methods, and offers a flexible foundation for future extensions. 

**Abstract (ZH)**: 现代AI系统通常依赖于不透明的黑盒模型，尤其是深度神经网络，其性能源自具有数百万个学习参数的复杂架构。尽管这些模型非常强大，但其复杂性对可信度构成了重大挑战，特别是在透明度缺乏的情况下。可解释AI（XAI）通过提供易于人类理解的模型行为解释来解决这一问题。然而，为了确保其有用性和可信度，这些解释必须经过严格的评估。尽管可解释AI方法的数量在不断增加，但该领域缺乏标准化的评估协议和适当度量标准的共识。为了填补这一空白，我们遵循系统评价和荟萃分析的首选报告项目（PRISMA）指南，进行了系统文献综述，并引入了统一的XAI评估框架（VXAI）。我们识别出362篇相关文献，并将其贡献整合为41组功能上相似的度量标准组。此外，我们提出了一种涵盖解释类型、评估背景和解释质量要求的三维分类方案。该框架提供了迄今为止最全面和结构化的VXAI综述。它支持系统的度量标准选择、促进方法之间的可比性，并为未来扩展提供了灵活的基础。 

---
# Evaluation Pipeline for systematically searching for Anomaly Detection Systems 

**Title (ZH)**: 系统性搜索异常检测系统的评估管道 

**Authors**: Florian Rokohl, Alexander Lehnert, Marc Reichenbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.15388)  

**Abstract**: Digitalization in the medical world provides major benefits while making it a target for attackers and thus hard to secure. To deal with network intruders we propose an anomaly detection system on hardware to detect malicious clients in real-time. We meet real-time and power restrictions using FPGAs. Overall system performance is achieved via the presented holistic system evaluation. 

**Abstract (ZH)**: 数字化在医疗领域的应用提供了巨大好处，但也使其成为攻击目标，难以确保安全。为应对网络入侵者，我们提出一种基于硬件的异常检测系统，以实时检测恶意客户端。我们通过使用FPGAs满足实时性和能耗限制。整体系统性能通过呈现的综合评估实现。 

---
# When and How Unlabeled Data Provably Improve In-Context Learning 

**Title (ZH)**: 何时以及如何无标签数据可以证明改进上下文学习 

**Authors**: Yingcong Li, Xiangyu Chang, Muti Kara, Xiaofeng Liu, Amit Roy-Chowdhury, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2506.15329)  

**Abstract**: Recent research shows that in-context learning (ICL) can be effective even when demonstrations have missing or incorrect labels. To shed light on this capability, we examine a canonical setting where the demonstrations are drawn according to a binary Gaussian mixture model (GMM) and a certain fraction of the demonstrations have missing labels. We provide a comprehensive theoretical study to show that: (1) The loss landscape of one-layer linear attention models recover the optimal fully-supervised estimator but completely fail to exploit unlabeled data; (2) In contrast, multilayer or looped transformers can effectively leverage unlabeled data by implicitly constructing estimators of the form $\sum_{i\ge 0} a_i (X^\top X)^iX^\top y$ with $X$ and $y$ denoting features and partially-observed labels (with missing entries set to zero). We characterize the class of polynomials that can be expressed as a function of depth and draw connections to Expectation Maximization, an iterative pseudo-labeling algorithm commonly used in semi-supervised learning. Importantly, the leading polynomial power is exponential in depth, so mild amount of depth/looping suffices. As an application of theory, we propose looping off-the-shelf tabular foundation models to enhance their semi-supervision capabilities. Extensive evaluations on real-world datasets show that our method significantly improves the semisupervised tabular learning performance over the standard single pass inference. 

**Abstract (ZH)**: 近期研究表明，在上下文学习（ICL）中，即使示例具有缺失或错误的标签，也能取得有效成果。为深入探讨这一能力，我们考察了这样一种经典设置：示例根据二元高斯混合模型（GMM）抽取，其中一部分示例具有缺失标签。我们进行了一项全面的理论研究，表明：（1）单层线性注意力模型的损失景观能够恢复最优的全监督估计器，但完全无法利用未标注数据；（2）相比之下，多层或循环的变压器能够通过隐式构造形如$\sum_{i\ge 0} a_i (X^\top X)^iX^\top y$的估计器有效利用未标注数据，其中$X$和$y$分别表示特征和部分可观测的标签（缺失项设为零）。我们刻画了能够表示为深度函数的多项式类，并将其与期望最大化的迭代伪标签算法联系起来，该算法是半监督学习中常见的算法。重要的是，主要的多项式幂是深度的指数函数，因此轻微的深度/循环即可。作为理论应用，我们提议循环现成的表格基础模型以增强其半监督能力。实证研究显示，我们的方法在实际数据集上的半监督表格学习性能显著优于标准的一次性推理。 

---
# J3DAI: A tiny DNN-Based Edge AI Accelerator for 3D-Stacked CMOS Image Sensor 

**Title (ZH)**: J3DAI：基于3D堆叠CMOS图像传感器的超小型DNN加速器 

**Authors**: Benoit Tain, Raphael Millet, Romain Lemaire, Michal Szczepanski, Laurent Alacoque, Emmanuel Pluchart, Sylvain Choisnet, Rohit Prasad, Jerome Chossat, Pascal Pierunek, Pascal Vivet, Sebastien Thuries  

**Link**: [PDF](https://arxiv.org/pdf/2506.15316)  

**Abstract**: This paper presents J3DAI, a tiny deep neural network-based hardware accelerator for a 3-layer 3D-stacked CMOS image sensor featuring an artificial intelligence (AI) chip integrating a Deep Neural Network (DNN)-based accelerator. The DNN accelerator is designed to efficiently perform neural network tasks such as image classification and segmentation. This paper focuses on the digital system of J3DAI, highlighting its Performance-Power-Area (PPA) characteristics and showcasing advanced edge AI capabilities on a CMOS image sensor. To support hardware, we utilized the Aidge comprehensive software framework, which enables the programming of both the host processor and the DNN accelerator. Aidge supports post-training quantization, significantly reducing memory footprint and computational complexity, making it crucial for deploying models on resource-constrained hardware like J3DAI. Our experimental results demonstrate the versatility and efficiency of this innovative design in the field of edge AI, showcasing its potential to handle both simple and computationally intensive tasks. Future work will focus on further optimizing the architecture and exploring new applications to fully leverage the capabilities of J3DAI. As edge AI continues to grow in importance, innovations like J3DAI will play a crucial role in enabling real-time, low-latency, and energy-efficient AI processing at the edge. 

**Abstract (ZH)**: 基于深度神经网络的J3DAI小型硬件加速器：一种适用于3层3D堆叠CMOS图像传感器的人工智能芯片设计及其先进边缘AI能力展示 

---
# Active Learning-Guided Seq2Seq Variational Autoencoder for Multi-target Inhibitor Generation 

**Title (ZH)**: 基于主动学习的Seq2Seq变分自编码器多目标抑制剂生成 

**Authors**: Júlia Vilalta-Mor, Alexis Molina, Laura Ortega Varga, Isaac Filella-Merce, Victor Guallar  

**Link**: [PDF](https://arxiv.org/pdf/2506.15309)  

**Abstract**: Simultaneously optimizing molecules against multiple therapeutic targets remains a profound challenge in drug discovery, particularly due to sparse rewards and conflicting design constraints. We propose a structured active learning (AL) paradigm integrating a sequence-to-sequence (Seq2Seq) variational autoencoder (VAE) into iterative loops designed to balance chemical diversity, molecular quality, and multi-target affinity. Our method alternates between expanding chemically feasible regions of latent space and progressively constraining molecules based on increasingly stringent multi-target docking thresholds. In a proof-of-concept study targeting three related coronavirus main proteases (SARS-CoV-2, SARS-CoV, MERS-CoV), our approach efficiently generated a structurally diverse set of pan-inhibitor candidates. We demonstrate that careful timing and strategic placement of chemical filters within this active learning pipeline markedly enhance exploration of beneficial chemical space, transforming the sparse-reward, multi-objective drug design problem into an accessible computational task. Our framework thus provides a generalizable roadmap for efficiently navigating complex polypharmacological landscapes. 

**Abstract (ZH)**: 同时优化多种治疗靶点的分子仍然被认为是药物发现中的一个深刻挑战，尤其是在稀疏奖励和冲突的设计约束方面。我们提出了一种结构化的主动学习（AL）范式，将序列到序列（Seq2Seq）变分自编码器（VAE）集成到迭代循环中，以平衡化学多样性、分子质量和多靶点亲和力。我们的方法交替进行潜在空间中化学可行区域的扩展和基于越来越严格的多靶点对接阈值逐步限制分子。在针对三种相关冠状病毒主蛋白酶（SARS-CoV-2、SARS-CoV、MERS-CoV）的目标概念验证研究中，我们的方式高效地生成了一组结构多样性的广谱抑制剂候选物。我们证明，在此主动学习管道中仔细安排化学滤镜的时间和位置显著增强了有利化学空间的探索，将稀疏奖励、多目标药物设计问题转化为可访问的计算任务。因此，我们的框架提供了一种通用的路线图，以有效地导航复杂的多药理学景观。 

---
# Accessible Gesture-Driven Augmented Reality Interaction System 

**Title (ZH)**: 可访问的手势驱动增强现实交互系统 

**Authors**: Yikan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15189)  

**Abstract**: Augmented reality (AR) offers immersive interaction but remains inaccessible for users with motor impairments or limited dexterity due to reliance on precise input methods. This study proposes a gesture-based interaction system for AR environments, leveraging deep learning to recognize hand and body gestures from wearable sensors and cameras, adapting interfaces to user capabilities. The system employs vision transformers (ViTs), temporal convolutional networks (TCNs), and graph attention networks (GATs) for gesture processing, with federated learning ensuring privacy-preserving model training across diverse users. Reinforcement learning optimizes interface elements like menu layouts and interaction modes. Experiments demonstrate a 20% improvement in task completion efficiency and a 25% increase in user satisfaction for motor-impaired users compared to baseline AR systems. This approach enhances AR accessibility and scalability. Keywords: Deep learning, Federated learning, Gesture recognition, Augmented reality, Accessibility, Human-computer interaction 

**Abstract (ZH)**: 基于手势的增强现实交互系统：通过深度学习提升肢体障碍用户 accessibility 和可扩展性 

---
# Classification of Multi-Parametric Body MRI Series Using Deep Learning 

**Title (ZH)**: 多参数身体MRI系列的深度学习分类 

**Authors**: Boah Kim, Tejas Sudharshan Mathai, Kimberly Helm, Peter A. Pinto, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2506.15182)  

**Abstract**: Multi-parametric magnetic resonance imaging (mpMRI) exams have various series types acquired with different imaging protocols. The DICOM headers of these series often have incorrect information due to the sheer diversity of protocols and occasional technologist errors. To address this, we present a deep learning-based classification model to classify 8 different body mpMRI series types so that radiologists read the exams efficiently. Using mpMRI data from various institutions, multiple deep learning-based classifiers of ResNet, EfficientNet, and DenseNet are trained to classify 8 different MRI series, and their performance is compared. Then, the best-performing classifier is identified, and its classification capability under the setting of different training data quantities is studied. Also, the model is evaluated on the out-of-training-distribution datasets. Moreover, the model is trained using mpMRI exams obtained from different scanners in two training strategies, and its performance is tested. Experimental results show that the DenseNet-121 model achieves the highest F1-score and accuracy of 0.966 and 0.972 over the other classification models with p-value$<$0.05. The model shows greater than 0.95 accuracy when trained with over 729 studies of the training data, whose performance improves as the training data quantities grew larger. On the external data with the DLDS and CPTAC-UCEC datasets, the model yields 0.872 and 0.810 accuracy for each. These results indicate that in both the internal and external datasets, the DenseNet-121 model attains high accuracy for the task of classifying 8 body MRI series types. 

**Abstract (ZH)**: 基于深度学习的多参数磁共振成像序列分类模型 

---
# Thunder-Tok: Minimizing Tokens per Word in Tokenizing Korean Texts for Generative Language Models 

**Title (ZH)**: Thunder-Tok: 最小化韩文分词中每个词的令牌数量以适应生成型语言模型 

**Authors**: Gyeongje Cho, Yeonkyoun So, Chanwoo Park, Sangmin Lee, Sungmok Jung, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.15138)  

**Abstract**: This paper introduces Thunder-Tok, a new Korean tokenizer designed to reduce token fertility without compromising model performance. Our approach uses a rule-based pre-tokenization method that aligns with the linguistic structure of the Korean language. We also create a seed vocabulary containing tokens that resemble linguistic units and employ a branching entropy-based selection algorithm. These techniques increase the average token length, thus lowering fertility while preserving linguistic information. Experimental results indicate that Thunder-Tok reduces fertility by approximately 10% (i.e., reduces the number of tokens by 10%, improving the inference speed by 10%) compared to BPE without compromising performance across various downstream tasks. These findings demonstrate that our linguistically informed approach is effective and practical for designing efficient tokenizers for language models. 

**Abstract (ZH)**: Thunder-Tok：一种通过减少标记丰度同时不牺牲模型性能的新韩语分词器 

---
# Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution 

**Title (ZH)**: 基于Rényi距离的解决方案：推荐系统中损失函数进展的比较研究 

**Authors**: Shengjia Zhang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15120)  

**Abstract**: Loss functions play a pivotal role in optimizing recommendation models. Among various loss functions, Softmax Loss (SL) and Cosine Contrastive Loss (CCL) are particularly effective. Their theoretical connections and differences warrant in-depth exploration. This work conducts comprehensive analyses of these losses, yielding significant insights: 1) Common strengths -- both can be viewed as augmentations of traditional losses with Distributional Robust Optimization (DRO), enhancing robustness to distributional shifts; 2) Respective limitations -- stemming from their use of different distribution distance metrics in DRO optimization, SL exhibits high sensitivity to false negative instances, whereas CCL suffers from low data utilization. To address these limitations, this work proposes a new loss function, DrRL, which generalizes SL and CCL by leveraging Rényi-divergence in DRO optimization. DrRL incorporates the advantageous structures of both SL and CCL, and can be demonstrated to effectively mitigate their limitations. Extensive experiments have been conducted to validate the superiority of DrRL on both recommendation accuracy and robustness. 

**Abstract (ZH)**: Softmax Loss 和 Cosine Contrastive Loss 的理论联系与差异及其改进研究：基于 Rényi 散度的 DrRL 方法 

---
# Transit for All: Mapping Equitable Bike2Subway Connection using Region Representation Learning 

**Title (ZH)**: 全民通勤：基于区域表示学习的公平自行车至地铁接驳 Mapping 

**Authors**: Min Namgung, JangHyeon Lee, Fangyi Ding, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15113)  

**Abstract**: Ensuring equitable public transit access remains challenging, particularly in densely populated cities like New York City (NYC), where low-income and minority communities often face limited transit accessibility. Bike-sharing systems (BSS) can bridge these equity gaps by providing affordable first- and last-mile connections. However, strategically expanding BSS into underserved neighborhoods is difficult due to uncertain bike-sharing demand at newly planned ("cold-start") station locations and limitations in traditional accessibility metrics that may overlook realistic bike usage potential. We introduce Transit for All (TFA), a spatial computing framework designed to guide the equitable expansion of BSS through three components: (1) spatially-informed bike-sharing demand prediction at cold-start stations using region representation learning that integrates multimodal geospatial data, (2) comprehensive transit accessibility assessment leveraging our novel weighted Public Transport Accessibility Level (wPTAL) by combining predicted bike-sharing demand with conventional transit accessibility metrics, and (3) strategic recommendations for new bike station placements that consider potential ridership and equity enhancement. Using NYC as a case study, we identify transit accessibility gaps that disproportionately impact low-income and minority communities in historically underserved neighborhoods. Our results show that strategically placing new stations guided by wPTAL notably reduces disparities in transit access related to economic and demographic factors. From our study, we demonstrate that TFA provides practical guidance for urban planners to promote equitable transit and enhance the quality of life in underserved urban communities. 

**Abstract (ZH)**: 确保公共 Transit 访问的公平性在 densely populated 城市如纽约市仍然具有挑战性，特别是低收入和少数族裔社区常常面临 Transit 访问受限的情况。共享单车系统（BSS）可以通过提供经济实惠的首末公里接驳来弥补这些公平性差距。然而，由于新规划站点（“冷启动”站点）位置的骑行需求不确定性和传统可达性评估指标的局限性（可能忽视实际的骑行使用潜力），在欠服务社区战略性扩张 BSS 比较困难。为此，我们引入了 Transit for All（TFA）空间计算框架，旨在通过三个组成部分指导 BSS 的公平扩展：（1）使用区域表示学习融合多模式地理空间数据的空间导向型共享单车需求预测；（2）综合评估 Transit 访问性，通过将预测的共享单车需求与传统 Transit 访可达性指标结合来利用我们新颖的加权公共交通可达性水平（wPTAL）；（3）在考虑潜在骑行量和公平性提升的前提下，提出新的自行车站布局战略建议。以纽约市为例，我们识别出历史上欠服务社区中低收入和少数族裔社区在 Transit 访问性方面的不成比例差距。我们的结果显示，根据 wPTAL 指导战略性设置新站点显著减少了与经济和社会因素相关的 Transit 访问性不平等。从我们的研究来看，TFA 为城市规划者提供了实用指南，以促进公平的 Transit 并提升欠服务城市社区的生活质量。 

---
# Improving Dialogue Discourse Parsing through Discourse-aware Utterance Clarification 

**Title (ZH)**: 通过话语意识的语句澄清改进对话话语解析 

**Authors**: Yaxin Fan, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15081)  

**Abstract**: Dialogue discourse parsing aims to identify and analyze discourse relations between the utterances within dialogues. However, linguistic features in dialogues, such as omission and idiom, frequently introduce ambiguities that obscure the intended discourse relations, posing significant challenges for parsers. To address this issue, we propose a Discourse-aware Clarification Module (DCM) to enhance the performance of the dialogue discourse parser. DCM employs two distinct reasoning processes: clarification type reasoning and discourse goal reasoning. The former analyzes linguistic features, while the latter distinguishes the intended relation from the ambiguous one. Furthermore, we introduce Contribution-aware Preference Optimization (CPO) to mitigate the risk of erroneous clarifications, thereby reducing cascading errors. CPO enables the parser to assess the contributions of the clarifications from DCM and provide feedback to optimize the DCM, enhancing its adaptability and alignment with the parser's requirements. Extensive experiments on the STAC and Molweni datasets demonstrate that our approach effectively resolves ambiguities and significantly outperforms the state-of-the-art (SOTA) baselines. 

**Abstract (ZH)**: 对话话语解析旨在识别和分析对话中话语单元之间的关系。但由于对话中的语言特征，如省略和成语，经常引入歧义，模糊了本意的话语关系，给解析器带来了重大挑战。为解决这一问题，我们提出了一种话语感知澄清模块（DCM）以提高话语解析器的性能。DCM运用了两种不同的推理过程：澄清类型推理和话语目标推理。前者分析语言特征，后者区分意图关系与歧义关系。此外，我们引入了贡献感知偏好优化（CPO）来减少错误澄清的风险，从而减少级联错误。CPO使解析器能够评估DCM澄清的贡献，并提供反馈以优化DCM，增强其适应性和与解析器需求的一致性。在STAC和Molweni数据集上的广泛实验表明，我们的方法有效解决了歧义问题，并显著优于当前最先进的（SOTA）基线。 

---
# Sequential Policy Gradient for Adaptive Hyperparameter Optimization 

**Title (ZH)**: 随序列策略梯度适应性超参数优化 

**Authors**: Zheng Li, Jerry Cheng, Huanying Helen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15051)  

**Abstract**: Reinforcement learning is essential for neural architecture search and hyperparameter optimization, but the conventional approaches impede widespread use due to prohibitive time and computational costs. Inspired by DeepSeek-V3 multi-token prediction architecture, we propose Sequential Policy Gradient modeling (SPG), a novel trajectory generation paradigm for lightweight online hyperparameter optimization. In contrast to conventional policy gradient methods, SPG extends the base model with temporary modules, enabling it to generate state-action (padded) trajectories in a single forward pass. Our experiments demonstrate that models gain performance when retrained with SPG on their original datasets and also outperform standard transfer fine-tuning. We evaluate on five datasets spanning computer vision (ImageNet, COCO), natural language processing (GLUE, SQuAD), and audio (SUPERB) to assess the industrial applicability of SPG. The proposed method demonstrates consistent improvements across widely adopted models, achieving performance gains of $+0.2\sim7\%$, with significantly low computational costs. Fully reproducible code and pre-trained models: this https URL. 

**Abstract (ZH)**: 强化学习对于神经架构搜索和超参数优化至关重要，但传统方法因其高昂的时间和计算成本而阻碍了其广泛应用。受DeepSeek-V3多令牌预测架构启发，我们提出了一种轻量级在线超参数优化的序贯策略梯度建模（SPG），这是一种新颖的轨迹生成范式。与传统的策略梯度方法不同，SPG通过在基础模型中引入临时模块，使其能够在单次前向传播中生成状态动作（填充）轨迹。我们的实验表明，使用SPG重新训练模型后可在原始数据集上获得性能提升，并且优于标准的迁移微调方法。我们在计算机视觉（ImageNet、COCO）、自然语言处理（GLUE、SQuAD）和音频（SUPERB）的五个数据集上进行了评估，以评估SPG的工业应用潜力。该方法在广泛采用的模型中显示出一致的性能提升，性能提升范围为2%至7%，同时计算成本显著降低。可完全复现的代码和预训练模型：this https URL。 

---
# Advanced Prediction of Hypersonic Missile Trajectories with CNN-LSTM-GRU Architectures 

**Title (ZH)**: 基于CNN-LSTM-GRU架构的高超声速导弹 trajectories 高级预测 

**Authors**: Amir Hossein Baradaran  

**Link**: [PDF](https://arxiv.org/pdf/2506.15043)  

**Abstract**: Advancements in the defense industry are paramount for ensuring the safety and security of nations, providing robust protection against emerging threats. Among these threats, hypersonic missiles pose a significant challenge due to their extreme speeds and maneuverability, making accurate trajectory prediction a critical necessity for effective countermeasures. This paper addresses this challenge by employing a novel hybrid deep learning approach, integrating Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs). By leveraging the strengths of these architectures, the proposed method successfully predicts the complex trajectories of hypersonic missiles with high accuracy, offering a significant contribution to defense strategies and missile interception technologies. This research demonstrates the potential of advanced machine learning techniques in enhancing the predictive capabilities of defense systems. 

**Abstract (ZH)**: 国防工业的进步对于确保国家的安全与安全至关重要，提供对新兴威胁的 robust 保护。其中，由于其极高速度和机动性，高超声速导弹构成了重大挑战，因此准确的轨迹预测是有效反制措施的必要条件。本文通过采用一种新颖的混合深度学习方法来应对这一挑战，该方法结合了卷积神经网络（CNNs）、长短期记忆（LSTM）网络和门控循环单元（GRUs）。通过利用这些架构的优势，所提出的方法成功地以高精度预测了高超声速导弹的复杂轨迹，为防御策略和导弹拦截技术做出了重要贡献。这项研究展示了先进机器学习技术在增强防御系统预测能力方面的潜力。 

---
# Fair Algorithms with Probing for Multi-Agent Multi-Armed Bandits 

**Title (ZH)**: 公平多代理多臂-bandits算法探查方法 

**Authors**: Tianyi Xu, Jiaxin Liu, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.14988)  

**Abstract**: We propose a multi-agent multi-armed bandit (MA-MAB) framework aimed at ensuring fair outcomes across agents while maximizing overall system performance. A key challenge in this setting is decision-making under limited information about arm rewards. To address this, we introduce a novel probing framework that strategically gathers information about selected arms before allocation. In the offline setting, where reward distributions are known, we leverage submodular properties to design a greedy probing algorithm with a provable performance bound. For the more complex online setting, we develop an algorithm that achieves sublinear regret while maintaining fairness. Extensive experiments on synthetic and real-world datasets show that our approach outperforms baseline methods, achieving better fairness and efficiency. 

**Abstract (ZH)**: 我们提出一种多代理多臂 bandit (MA-MAB) 框架，旨在确保代理间的公平性同时最大化整体系统性能。在仅有限了解臂奖励信息的情况下作出决策是一个关键挑战。为此，我们引入了一个新颖的信息探查框架，该框架有选择地收集信息后再进行分配。在奖励分布已知的离线场景中，我们利用亚模性质设计了一个具有可证明性能边界的贪婪探查算法。在更加复杂的在线场景中，我们开发了一种算法，该算法在保持公平性的同时实现了次线性遗憾。我们在合成数据集和现实世界数据集上的 extensive 实验表明，我们的方法优于基准方法，实现了更好的公平性和效率。 

---
# Flat Channels to Infinity in Neural Loss Landscapes 

**Title (ZH)**: 无限延伸的平坦通道在神经网络损失景观中 

**Authors**: Flavio Martinelli, Alexander Van Meegen, Berfin Şimşek, Wulfram Gerstner, Johanni Brea  

**Link**: [PDF](https://arxiv.org/pdf/2506.14951)  

**Abstract**: The loss landscapes of neural networks contain minima and saddle points that may be connected in flat regions or appear in isolation. We identify and characterize a special structure in the loss landscape: channels along which the loss decreases extremely slowly, while the output weights of at least two neurons, $a_i$ and $a_j$, diverge to $\pm$infinity, and their input weight vectors, $\mathbf{w_i}$ and $\mathbf{w_j}$, become equal to each other. At convergence, the two neurons implement a gated linear unit: $a_i\sigma(\mathbf{w_i} \cdot \mathbf{x}) + a_j\sigma(\mathbf{w_j} \cdot \mathbf{x}) \rightarrow \sigma(\mathbf{w} \cdot \mathbf{x}) + (\mathbf{v} \cdot \mathbf{x}) \sigma'(\mathbf{w} \cdot \mathbf{x})$. Geometrically, these channels to infinity are asymptotically parallel to symmetry-induced lines of critical points. Gradient flow solvers, and related optimization methods like SGD or ADAM, reach the channels with high probability in diverse regression settings, but without careful inspection they look like flat local minima with finite parameter values. Our characterization provides a comprehensive picture of these quasi-flat regions in terms of gradient dynamics, geometry, and functional interpretation. The emergence of gated linear units at the end of the channels highlights a surprising aspect of the computational capabilities of fully connected layers. 

**Abstract (ZH)**: 神经网络的损失景观包含连接在平坦区域的极小值和鞍点，或孤立出现的极小值和鞍点。我们识别并characterized一种特殊的损失景观结构：在这些结构中，沿某些通道损失变化极慢，同时至少有两个神经元$a_i$和$a_j$的输出权重发散至无穷大正负方向，且它们输入权重向量$\mathbf{w_i}$和$\mathbf{w_j}$变得相等。在收敛时，这两个神经元实现了一个门控线性单元：$a_i\sigma(\mathbf{w_i} \cdot \mathbf{x}) + a_j\sigma(\mathbf{w_j} \cdot \mathbf{x}) \rightarrow \sigma(\mathbf{w} \cdot \mathbf{x}) + (\mathbf{v} \cdot \mathbf{x}) \sigma'(\mathbf{w} \cdot \mathbf{x})$。几何上，这些通往无穷的通道在经由对称性诱导的临界点线方面渐近平行。梯度流求解器以及相关优化方法如SGD或ADAM在多种回归设置中以高概率到达这些通道，但如果没有仔细检查，这些通道看起来像是具有有限参数值的平坦局部极小值。我们的characterization为这些准平坦区域提供了从梯度动力学、几何和功能解释的全面图景。通道末端出现门控线性单元的现象揭示了全连接层计算能力的一种令人惊讶的方面。 

---
# Determinação Automática de Limiar de Detecção de Ataques em Redes de Computadores Utilizando Autoencoders 

**Title (ZH)**: 利用自编码器自动确定计算机网络攻击检测阈值 

**Authors**: Luan Gonçalves Miranda, Pedro Ivo da Cruz, Murilo Bellezoni Loiola  

**Link**: [PDF](https://arxiv.org/pdf/2506.14937)  

**Abstract**: Currently, digital security mechanisms like Anomaly Detection Systems using Autoencoders (AE) show great potential for bypassing problems intrinsic to the data, such as data imbalance. Because AE use a non-trivial and nonstandardized separation threshold to classify the extracted reconstruction error, the definition of this threshold directly impacts the performance of the detection process. Thus, this work proposes the automatic definition of this threshold using some machine learning algorithms. For this, three algorithms were evaluated: the K-Nearst Neighbors, the K-Means and the Support Vector Machine. 

**Abstract (ZH)**: 当前，使用自动编码器（AE）的异常检测系统在克服数据固有难题（如数据不平衡）方面展现出巨大潜力。由于AE使用一个非平凡且非标准化的分离阈值来分类重构误差，因此该阈值的定义直接影响检测过程的性能。为此，本研究提出了一种使用某些机器学习算法自动定义该阈值的方法。为此，评估了三种算法：K-最近邻、K-均值和支撑向量机。 

---
# Forecasting the spatiotemporal evolution of fluid-induced microearthquakes with deep learning 

**Title (ZH)**: 基于深度学习的流体诱发微地震的时空演化预测 

**Authors**: Jaehong Chung, Michael Manga, Timothy Kneafsey, Tapan Mukerji, Mengsu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14923)  

**Abstract**: Microearthquakes (MEQs) generated by subsurface fluid injection record the evolving stress state and permeability of reservoirs. Forecasting their full spatiotemporal evolution is therefore critical for applications such as enhanced geothermal systems (EGS), CO$_2$ sequestration and other geo-engineering applications. We present a transformer-based deep learning model that ingests hydraulic stimulation history and prior MEQ observations to forecast four key quantities: cumulative MEQ count, cumulative logarithmic seismic moment, and the 50th- and 95th-percentile extents ($P_{50}, P_{95}$) of the MEQ cloud. Applied to the EGS Collab Experiment 1 dataset, the model achieves $R^2 >0.98$ for the 1-second forecast horizon and $R^2 >0.88$ for the 15-second forecast horizon across all targets, and supplies uncertainty estimates through a learned standard deviation term. These accurate, uncertainty-quantified forecasts enable real-time inference of fracture propagation and permeability evolution, demonstrating the strong potential of deep-learning approaches to improve seismic-risk assessment and guide mitigation strategies in future fluid-injection operations. 

**Abstract (ZH)**: 微震事件（MEQs）由地下流体注入产生，记录了储层应力状态和渗透率的变化。预测其全空间-时间演化对于增强地热系统（EGS）、CO₂密封和其它地质工程应用至关重要。我们提出了一种基于变压器的深度学习模型，该模型整合了液压刺激历史和前期微震观测数据，预测四个关键量：累计微震事件计数、累计对数地震矩以及微震云的第50百分位（P₅₀）和第95百分位（P₉₅）范围。该模型应用于EGS Collab Experiment 1数据集，对于1秒和15秒的预测窗口，在所有目标上的决定系数（$R^2$）分别大于0.98和0.88，并通过学习得到的标准差项提供不确定性估计。这些准确且具有不确定量化测的预测能够实时推断裂缝扩展和渗透率演化，展示了深度学习方法在提高地震风险评估并指导未来流体注入操作中的缓解策略方面的强大潜力。 

---
# Foundation Artificial Intelligence Models for Health Recognition Using Face Photographs (FAHR-Face) 

**Title (ZH)**: 基于面部照片的健康识别基础人工智能模型（FAHR-Face） 

**Authors**: Fridolin Haugg, Grace Lee, John He, Leonard Nürnberg, Dennis Bontempi, Danielle S. Bitterman, Paul Catalano, Vasco Prudente, Dmitrii Glubokov, Andrew Warrington, Suraj Pai, Dirk De Ruysscher, Christian Guthier, Benjamin H. Kann, Vadim N. Gladyshev, Hugo JWL Aerts, Raymond H. Mak  

**Link**: [PDF](https://arxiv.org/pdf/2506.14909)  

**Abstract**: Background: Facial appearance offers a noninvasive window into health. We built FAHR-Face, a foundation model trained on >40 million facial images and fine-tuned it for two distinct tasks: biological age estimation (FAHR-FaceAge) and survival risk prediction (FAHR-FaceSurvival).
Methods: FAHR-FaceAge underwent a two-stage, age-balanced fine-tuning on 749,935 public images; FAHR-FaceSurvival was fine-tuned on 34,389 photos of cancer patients. Model robustness (cosmetic surgery, makeup, pose, lighting) and independence (saliency mapping) was tested extensively. Both models were clinically tested in two independent cancer patient datasets with survival analyzed by multivariable Cox models and adjusted for clinical prognostic factors.
Findings: For age estimation, FAHR-FaceAge had the lowest mean absolute error of 5.1 years on public datasets, outperforming benchmark models and maintaining accuracy across the full human lifespan. In cancer patients, FAHR-FaceAge outperformed a prior facial age estimation model in survival prognostication. FAHR-FaceSurvival demonstrated robust prediction of mortality, and the highest-risk quartile had more than triple the mortality of the lowest (adjusted hazard ratio 3.22; P<0.001). These findings were validated in the independent cohort and both models showed generalizability across age, sex, race and cancer subgroups. The two algorithms provided distinct, complementary prognostic information; saliency mapping revealed each model relied on distinct facial regions. The combination of FAHR-FaceAge and FAHR-FaceSurvival improved prognostic accuracy.
Interpretation: A single foundation model can generate inexpensive, scalable facial biomarkers that capture both biological ageing and disease-related mortality risk. The foundation model enabled effective training using relatively small clinical datasets. 

**Abstract (ZH)**: 背景：面部外观提供了健康状况的非侵入性窗口。我们构建了FAHR-Face基础模型，该模型训练于超过4000万张面部图像，并针对两种不同的任务进行了微调：生物学年龄估计（FAHR-FaceAge）和生存风险预测（FAHR-FaceSurvival）。 

---
# Preparing for the Intelligence Explosion 

**Title (ZH)**: 准备迎接智能爆炸 

**Authors**: William MacAskill, Fin Moorhouse  

**Link**: [PDF](https://arxiv.org/pdf/2506.14863)  

**Abstract**: AI that can accelerate research could drive a century of technological progress over just a few years. During such a period, new technological or political developments will raise consequential and hard-to-reverse decisions, in rapid succession. We call these developments grand challenges. These challenges include new weapons of mass destruction, AI-enabled autocracies, races to grab offworld resources, and digital beings worthy of moral consideration, as well as opportunities to dramatically improve quality of life and collective decision-making. We argue that these challenges cannot always be delegated to future AI systems, and suggest things we can do today to meaningfully improve our prospects. AGI preparedness is therefore not just about ensuring that advanced AI systems are aligned: we should be preparing, now, for the disorienting range of developments an intelligence explosion would bring. 

**Abstract (ZH)**: AI能够加速研发可能在几年内推动百年科技进步。在这段时间内，新的技术或政治发展将迅速提出重要且难以逆转的决策。我们称之为宏伟挑战。这些挑战包括新型大规模毁灭性武器、AI驱动的獨裁政权、争夺外太空资源的竞争，以及值得道德关怀的数字生命体，同时也有大幅提高生活质量及集体决策的机会。我们主张这些挑战不能总是留给未来的AI系统处理，建议我们现在可以采取措施以实质性地改善我们的前景。因此，超人工智能准备不仅关乎确保先进AI系统的对齐：我们应当为智力爆炸带来的各种错综复杂的发展做好准备。 

---
# Identifiability by common backdoor in summary causal graphs of time series 

**Title (ZH)**: 时间序列汇总因果图中的公共后门可识别性 

**Authors**: Clément Yvernes, Charles K. Assaad, Emilie Devijver, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2506.14862)  

**Abstract**: The identifiability problem for interventions aims at assessing whether the total effect of some given interventions can be written with a do-free formula, and thus be computed from observational data only. We study this problem, considering multiple interventions and multiple effects, in the context of time series when only abstractions of the true causal graph in the form of summary causal graphs are available. We focus in this study on identifiability by a common backdoor set, and establish, for time series with and without consistency throughout time, conditions under which such a set exists. We also provide algorithms of limited complexity to decide whether the problem is identifiable or not. 

**Abstract (ZH)**: 干预的可识别性问题旨在评估某些给定干预的总效果是否可以用 do-自由公式表示，并从而仅通过观测数据进行计算。我们在仅可获取真实因果图的抽象形式即汇总因果图的时间序列情境下，研究此问题，考虑多个干预和多个效果。本研究聚焦于由共同后门集的可识别性，并为具有和不具时间一致性的时间序列建立了这样的集存在条件。我们还提供了复杂度有限的算法来决定该问题是否可识别。 

---
# BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models 

**Title (ZH)**: BMFM-RNA：一个构建和评估转录组基础模型的开源框架 

**Authors**: Bharath Dandala, Michael M. Danziger, Ella Barkan, Tanwi Biswas, Viatcheslav Gurev, Jianying Hu, Matthew Madgwick, Akira Koseki, Tal Kozlovski, Michal Rosen-Zvi, Yishai Shimoni, Ching-Huei Tsou  

**Link**: [PDF](https://arxiv.org/pdf/2506.14861)  

**Abstract**: Transcriptomic foundation models (TFMs) have recently emerged as powerful tools for analyzing gene expression in cells and tissues, supporting key tasks such as cell-type annotation, batch correction, and perturbation prediction. However, the diversity of model implementations and training strategies across recent TFMs, though promising, makes it challenging to isolate the contribution of individual design choices or evaluate their potential synergies. This hinders the field's ability to converge on best practices and limits the reproducibility of insights across studies. We present BMFM-RNA, an open-source, modular software package that unifies diverse TFM pretraining and fine-tuning objectives within a single framework. Leveraging this capability, we introduce a novel training objective, whole cell expression decoder (WCED), which captures global expression patterns using an autoencoder-like CLS bottleneck representation. In this paper, we describe the framework, supported input representations, and training objectives. We evaluated four model checkpoints pretrained on CELLxGENE using combinations of masked language modeling (MLM), WCED and multitask learning. Using the benchmarking capabilities of BMFM-RNA, we show that WCED-based models achieve performance that matches or exceeds state-of-the-art approaches like scGPT across more than a dozen datasets in both zero-shot and fine-tuning tasks. BMFM-RNA, available as part of the biomed-multi-omics project ( this https URL ), offers a reproducible foundation for systematic benchmarking and community-driven exploration of optimal TFM training strategies, enabling the development of more effective tools to leverage the latest advances in AI for understanding cell biology. 

**Abstract (ZH)**: 转录组基石模型（TFMs） recently emerged as powerful tools for analyzing gene expression in cells and tissues, supporting key tasks such as cell-type annotation, batch correction, and perturbation prediction. However, the diversity of model implementations and training strategies across recent TFMs, though promising, makes it challenging to isolate the contribution of individual design choices or evaluate their potential synergies. This hinders the field's ability to converge on best practices and limits the reproducibility of insights across studies. We present BMFM-RNA, an open-source, modular software package that unifies diverse TFM pretraining and fine-tuning objectives within a single framework. Leveraging this capability, we introduce a novel training objective, whole cell expression decoder (WCED), which captures global expression patterns using an autoencoder-like CLS bottleneck representation. In this paper, we describe the framework, supported input representations, and training objectives. We evaluated four model checkpoints pretrained on CELLxGENE using combinations of masked language modeling (MLM), WCED and multitask learning. Using the benchmarking capabilities of BMFM-RNA, we show that WCED-based models achieve performance that matches or exceeds state-of-the-art approaches like scGPT across more than a dozen datasets in both zero-shot and fine-tuning tasks. BMFM-RNA, available as part of the biomed-multi-omics project (<https://github.com/biomed-multi-omics/bmfm-rna>), offers a reproducible foundation for systematic benchmarking and community-driven exploration of optimal TFM training strategies, enabling the development of more effective tools to leverage the latest advances in AI for understanding cell biology. 

---
# Real-Time, Low-Latency Surveillance Using Entropy-Based Adaptive Buffering and MobileNetV2 on Edge Devices 

**Title (ZH)**: 基于熵自适应缓冲和MobileNetV2的边缘设备实时低延时 surveillance 技术 

**Authors**: Poojashree Chandrashekar Pankaj M Sajjanar  

**Link**: [PDF](https://arxiv.org/pdf/2506.14833)  

**Abstract**: This paper describes a high-performance, low-latency video surveillance system designed for resource-constrained environments. We have proposed a formal entropy-based adaptive frame buffering algorithm and integrated that with MobileNetV2 to achieve high throughput with low latency. The system is capable of processing live streams of video with sub-50ms end-to-end inference latency on resource-constrained devices (embedding platforms) such as Raspberry Pi, Amazon, and NVIDIA Jetson Nano. Our method maintains over 92% detection accuracy on standard datasets focused on video surveillance and exhibits robustness to varying lighting, backgrounds, and speeds. A number of comparative and ablation experiments validate the effectiveness of our design. Finally, our architecture is scalable, inexpensive, and compliant with stricter data privacy regulations than common surveillance systems, so that the system could coexist in a smart city or embedded security architecture. 

**Abstract (ZH)**: 一种资源受限环境下高性能低延迟的视频 surveillance 系统设计与实现 

---
# ArchShapeNet:An Interpretable 3D-CNN Framework for Evaluating Architectural Shapes 

**Title (ZH)**: ArchShapeNet：一个可解释的3D-CNN框架，用于评估建筑形态 

**Authors**: Jun Yin, Jing Zhong, Pengyu Zeng, Peilin Li, Zixuan Dai, Miao Zhang, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14832)  

**Abstract**: In contemporary architectural design, the growing complexity and diversity of design demands have made generative plugin tools essential for quickly producing initial concepts and exploring novel 3D forms. However, objectively analyzing the differences between human-designed and machine-generated 3D forms remains a challenge, limiting our understanding of their respective strengths and hindering the advancement of generative tools.
To address this, we built ArchForms-4000, a dataset containing 2,000 architect-designed and 2,000 Evomass-generated 3D forms; Proposed ArchShapeNet, a 3D convolutional neural network tailored for classifying and analyzing architectural forms, incorporating a saliency module to highlight key spatial features aligned with architectural reasoning; And conducted comparative experiments showing our model outperforms human experts in distinguishing form origins, achieving 94.29% accuracy, 96.2% precision, and 98.51% recall.
This study not only highlights the distinctive advantages of human-designed forms in spatial organization, proportional harmony, and detail refinement but also provides valuable insights for enhancing generative design tools in the future. 

**Abstract (ZH)**: 当前建筑设计中日益增长的复杂性和多样性需求使得生成式插件工具成为快速产生初步概念和探索新型3D形式的必要工具。然而，客观分析人类设计与机器生成的3D形式之间的差异仍然颇具挑战性，这限制了我们对其各自优势的理解，并阻碍了生成工具的发展。

为此，我们构建了ArchForms-4000数据集，包含2000个人类设计的和2000个由Evomass生成的3D形式；提出了ArchShapeNet，一种专为分类和分析建筑形式设计的3D卷积神经网络，该网络包含一个显著性模块，用于突出与建筑推理相一致的关键空间特征；并通过比较实验展示了我们的模型在区分形式起源方面优于人类专家，准确率达94.29%，精确率达96.2%，召回率达98.51%。

该研究不仅突显了人类设计形式在空间组织、比例和谐和细节精炼方面的独特优势，还为未来增强生成设计工具提供了宝贵的见解。 

---
# Optimization of bi-directional gated loop cell based on multi-head attention mechanism for SSD health state classification model 

**Title (ZH)**: 基于多头注意力机制的双向门控环形单元优化在SSD健康状态分类模型中 

**Authors**: Zhizhao Wen, Ruoxin Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14830)  

**Abstract**: Aiming at the critical role of SSD health state prediction in data reliability assurance, this study proposes a hybrid BiGRU-MHA model that incorporates a multi-head attention mechanism to enhance the accuracy and stability of storage device health classification. The model innovatively integrates temporal feature extraction and key information focusing capabilities. Specifically, it leverages the bidirectional timing modeling advantages of the BiGRU network to capture both forward and backward dependencies of SSD degradation features. Simultaneously, the multi-head attention mechanism dynamically assigns feature weights, improving the model's sensitivity to critical health indicators. Experimental results show that the proposed model achieves classification accuracies of 92.70% on the training set and 92.44% on the test set, with a minimal performance gap of only 0.26%, demonstrating excellent generalization ability. Further analysis using the receiver operating characteristic (ROC) curve shows an area under the curve (AUC) of 0.94 on the test set, confirming the model's robust binary classification performance. This work not only presents a new technical approach for SSD health prediction but also addresses the generalization bottleneck of traditional models, offering a verifiable method with practical value for preventive maintenance of industrial-grade storage systems. The results show the model can significantly reduce data loss risks by providing early failure warnings and help optimize maintenance costs, supporting intelligent decision-making in building reliable storage systems for cloud computing data centers and edge storage environments. 

**Abstract (ZH)**: 针对SSD健康状态预测在数据可靠性保障中的关键作用，本研究提出了一种结合多头注意力机制的混合BiGRU-MHA模型，以提高存储设备健康分类的准确性和稳定性。该模型创新性地整合了时间特征提取和关键信息聚焦能力。具体而言，它利用BiGRU网络的双向时间建模优势，捕捉SSD退化特征的前后依赖关系。同时，多头注意力机制动态分配特征权重，提高模型对关键健康指标的敏感性。实验结果表明，所提出模型在训练集上的分类准确率为92.70%，在测试集上的分类准确率为92.44%，性能差距仅为0.26%，展现出优秀的泛化能力。进一步使用接收者操作特征（ROC）曲线分析，测试集下的曲线下面积（AUC）为0.94，证实了模型的稳健二分类性能。本工作不仅提出了一种新的SSD健康预测技术方法，还解决了传统模型的泛化瓶颈问题，提供了一种具有实际价值的验证方法，用于工业级存储系统的预防性维护。研究结果表明，该模型可以显著降低数据丢失风险并提供早期故障预警，有助于优化维护成本，支持在云计算数据中心和边缘存储环境中构建可靠存储系统的智能决策。 

---
# The Hardness of Achieving Impact in AI for Social Impact Research: A Ground-Level View of Challenges & Opportunities 

**Title (ZH)**: 在人工智能社会影响研究中实现影响的挑战与机遇：地面视角下的难度encent
note
纠正了最后的错误，输出为：
在人工智能社会影响研究中实现影响的挑战与机遇：地面视角下的 hardness 

**Authors**: Aditya Majumdar, Wenbo Zhang, Kashvi Prawal, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.14829)  

**Abstract**: In an attempt to tackle the UN SDGs, AI for Social Impact (AI4SI) projects focus on harnessing AI to address societal issues in areas such as healthcare, social justice, etc. Unfortunately, despite growing interest in AI4SI, achieving tangible, on-the-ground impact remains a significant challenge. For example, identifying and engaging motivated collaborators who are willing to co-design and deploy AI based solutions in real-world settings is often difficult. Even when such partnerships are established, many AI4SI projects "fail" to progress beyond the proof-of-concept stage, and hence, are unable to transition to at-scale production-level solutions. Furthermore, the unique challenges faced by AI4SI researchers are not always fully recognized within the broader AI community, where such work is sometimes viewed as primarily applied and not aligning with the traditional criteria for novelty emphasized in core AI venues. This paper attempts to shine a light on the diverse challenges faced in AI4SI research by diagnosing a multitude of factors that prevent AI4SI partnerships from achieving real-world impact on the ground. Drawing on semi-structured interviews with six leading AI4SI researchers - complemented by the authors' own lived experiences in conducting AI4SI research - this paper attempts to understand the day-to-day difficulties faced in developing and deploying socially impactful AI solutions. Through thematic analysis, we identify structural and organizational, communication, collaboration, and operational challenges as key barriers to deployment. While there are no easy fixes, we synthesize best practices and actionable strategies drawn from these interviews and our own work in this space. In doing so, we hope this paper serves as a practical reference guide for AI4SI researchers and partner organizations seeking to engage more effectively in socially impactful AI collaborations. 

**Abstract (ZH)**: 利用人工智能促进社会影响以应对联合国可持续发展目标的挑战与策略：识别并解决实际应用中的关键障碍 

---
# Collaborative Interest-aware Graph Learning for Group Identification 

**Title (ZH)**: 协作式兴趣感知图学习在群体识别中的应用 

**Authors**: Rui Zhao, Beihong Jin, Beibei Li, Yiyuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.14826)  

**Abstract**: With the popularity of social media, an increasing number of users are joining group activities on online social platforms. This elicits the requirement of group identification (GI), which is to recommend groups to users. We reveal that users are influenced by both group-level and item-level interests, and these dual-level interests have a collaborative evolution relationship: joining a group expands the user's item interests, further prompting the user to join new groups. Ultimately, the two interests tend to align dynamically. However, existing GI methods fail to fully model this collaborative evolution relationship, ignoring the enhancement of group-level interests on item-level interests, and suffering from false-negative samples when aligning cross-level interests. In order to fully model the collaborative evolution relationship between dual-level user interests, we propose CI4GI, a Collaborative Interest-aware model for Group Identification. Specifically, we design an interest enhancement strategy that identifies additional interests of users from the items interacted with by the groups they have joined as a supplement to item-level interests. In addition, we adopt the distance between interest distributions of two users to optimize the identification of negative samples for a user, mitigating the interference of false-negative samples during cross-level interests alignment. The results of experiments on three real-world datasets demonstrate that CI4GI significantly outperforms state-of-the-art models. 

**Abstract (ZH)**: 基于协作兴趣的认知群体识别模型 CI4GI 

---
# Next-Generation Conflict Forecasting: Unleashing Predictive Patterns through Spatiotemporal Learning 

**Title (ZH)**: 下一代冲突预测：通过空间时间学习释放预测模式 

**Authors**: Simon P. von der Maase  

**Link**: [PDF](https://arxiv.org/pdf/2506.14817)  

**Abstract**: Forecasting violent conflict at high spatial and temporal resolution remains a central challenge for both researchers and policymakers. This study presents a novel neural network architecture for forecasting three distinct types of violence -- state-based, non-state, and one-sided -- at the subnational (priogrid-month) level, up to 36 months in advance. The model jointly performs classification and regression tasks, producing both probabilistic estimates and expected magnitudes of future events. It achieves state-of-the-art performance across all tasks and generates approximate predictive posterior distributions to quantify forecast uncertainty.
The architecture is built on a Monte Carlo Dropout Long Short-Term Memory (LSTM) U-Net, integrating convolutional layers to capture spatial dependencies with recurrent structures to model temporal dynamics. Unlike many existing approaches, it requires no manual feature engineering and relies solely on historical conflict data. This design enables the model to autonomously learn complex spatiotemporal patterns underlying violent conflict.
Beyond achieving state-of-the-art predictive performance, the model is also highly extensible: it can readily integrate additional data sources and jointly forecast auxiliary variables. These capabilities make it a promising tool for early warning systems, humanitarian response planning, and evidence-based peacebuilding initiatives. 

**Abstract (ZH)**: 高空间和时间分辨率下预测暴力冲突仍是一项重大挑战：一种新型神经网络架构及其在非国家、单方面和国家间暴力冲突预测中的应用及其不确定性量化 

---
# Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks 

**Title (ZH)**: 自信训练：通过自动化主动检查捕获深度学习训练中的隐形错误 

**Authors**: Yuxuan Jiang, Ziming Zhou, Boyu Xu, Beijie Liu, Runhui Xu, Peng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14813)  

**Abstract**: Training deep learning (DL) models is a complex process, making it prone to silent errors that are challenging to detect and diagnose. This paper presents TRAINCHECK, a framework that takes a proactive checking approach to address silent training errors. TRAINCHECK automatically infers invariants tailored for DL training. It uses these invariants to proactively detect silent errors during the training process while providing debugging help. To evaluate TRAINCHECK, we reproduce 20 real-world silent training errors with diverse root causes. TRAINCHECK successfully detects 18 errors within a single training iteration. It also uncovers 6 unknown bugs in popular training libraries that lead to silent errors. 

**Abstract (ZH)**: 训练深度学习模型是一个复杂的过程，容易产生难以检测和诊断的隐形错误。本文提出了一种名为TRAINCHECK的框架，采取主动检查的方法来应对隐形训练错误。TRAINCHECK自动推断适用于深度学习训练的不变式，并在训练过程中主动检测隐形错误，同时提供调试帮助。为了评估TRAINCHECK，我们重现了20个具有不同根本原因的实世界隐形训练错误。TRAINCHECK成功在单个训练迭代中检测到18个错误，并发现了6个在流行训练库中导致隐形错误的未知 bug。 

---
# ss-Mamba: Semantic-Spline Selective State-Space Model 

**Title (ZH)**: 基于语义样条的选择性状态空间模型:ss-Mamba 

**Authors**: Zuochen Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.14802)  

**Abstract**: We propose ss-Mamba, a novel foundation model that enhances time series forecasting by integrating semantic-aware embeddings and adaptive spline-based temporal encoding within a selective state-space modeling framework. Building upon the recent success of Transformer architectures, ss-Mamba adopts the Mamba selective state space model as an efficient alternative that achieves comparable performance while significantly reducing computational complexity from quadratic to linear time. Semantic index embeddings, initialized from pretrained language models, allow effective generalization to previously unseen series through meaningful semantic priors. Additionally, spline-based Kolmogorov-Arnold Networks (KAN) dynamically and interpretably capture complex seasonalities and non-stationary temporal effects, providing a powerful enhancement over conventional temporal feature encodings. Extensive experimental evaluations confirm that ss-Mamba delivers superior accuracy, robustness, and interpretability, demonstrating its capability as a versatile and computationally efficient alternative to traditional Transformer-based models in time-series forecasting. 

**Abstract (ZH)**: 我们提出ss-Mamba，一种新型基础模型，通过在选择性状态空间建模框架内整合语义意识嵌入和自适应样条时间编码来增强时间序列预测。基于Transformer架构的 recent 成功，ss-Mamba 采用 Mamba 选择性状态空间模型作为高效的替代方案，既能实现类似性能，又大大减少了从二次到线性的时间计算复杂度。预训练语言模型初始化的语义索引嵌入，通过有意义的语义先验有效泛化到以前未见过的时间序列。此外，基于样条的柯尔莫哥洛夫-阿诺尔德网络 (KAN) 动态且可解释地捕获复杂季节性和非 stationary 时间效果，提供了对传统时间特征编码的强大增强。大量实验证明，ss-Mamba 在准确性、鲁棒性和可解释性方面表现出优越性，证明了其作为传统基于Transformer的模型在时间序列预测中的一种灵活且计算高效的替代方案的能力。 

---
# MODS: Multi-source Observations Conditional Diffusion Model for Meteorological State Downscaling 

**Title (ZH)**: MODS：多源观测条件扩散模型气象状态精细化模拟能力 

**Authors**: Siwei Tu, Jingyi Xu, Weidong Yang, Lei Bai, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.14798)  

**Abstract**: Accurate acquisition of high-resolution surface meteorological conditions is critical for forecasting and simulating meteorological variables. Directly applying spatial interpolation methods to derive meteorological values at specific locations from low-resolution grid fields often yields results that deviate significantly from the actual conditions. Existing downscaling methods primarily rely on the coupling relationship between geostationary satellites and ERA5 variables as a condition. However, using brightness temperature data from geostationary satellites alone fails to comprehensively capture all the changes in meteorological variables in ERA5 maps. To address this limitation, we can use a wider range of satellite data to make more full use of its inversion effects on various meteorological variables, thus producing more realistic results across different meteorological variables. To further improve the accuracy of downscaling meteorological variables at any location, we propose the Multi-source Observation Down-Scaling Model (MODS). It is a conditional diffusion model that fuses data from multiple geostationary satellites GridSat, polar-orbiting satellites (AMSU-A, HIRS, and MHS), and topographic data (GEBCO), as conditions, and is pre-trained on the ERA5 reanalysis dataset. During training, latent features from diverse conditional inputs are extracted separately and fused into ERA5 maps via a multi-source cross-attention module. By exploiting the inversion relationships between reanalysis data and multi-source atmospheric variables, MODS generates atmospheric states that align more closely with real-world conditions. During sampling, MODS enhances downscaling consistency by incorporating low-resolution ERA5 maps and station-level meteorological data as guidance. Experimental results demonstrate that MODS achieves higher fidelity when downscaling ERA5 maps to a 6.25 km resolution. 

**Abstract (ZH)**: 准确获取高分辨率地表气象条件对于气象变量的预报和模拟至关重要。直接将低分辨率网格场中的气象值通过空间插值方法推导至特定位置往往会导致与实际条件显著偏差的结果。现有的降尺度方法主要依赖于静止卫星和ERA5变量之间的耦合关系。然而，仅使用静止卫星的亮度温度数据无法全面捕捉ERA5地图中气象变量的所有变化。为解决这一局限性，可以使用更广泛的卫星数据来充分发挥其对各种气象变量的反演效果，从而在不同气象变量中生成更加符合实际情况的结果。为进一步提高在任何地点降尺度气象变量的准确性，我们提出了多源观测降尺度模型（MODS）。该模型是一个条件扩散模型，结合来自多个静止卫星（GridSat）、极轨卫星（AMSU-A、HIRS和MHS）以及地形数据（GEBCO）的条件输入数据，并在ERA5再分析数据集上预先训练。在训练过程中，多种条件输入的潜在特征被分别提取并通过多源交叉注意力模块融合到ERA5地图中。通过利用再分析数据和多源大气变量之间的反演关系，MODS生成的气象状态更贴近实际情况。在采样过程中，MODS通过纳入低分辨率ERA5地图和站级气象数据来增强降尺度一致性。实验结果表明，MODS在降尺度ERA5地图至6.25公里分辨率时获得了更高的保真度。 

---
# Bound by semanticity: universal laws governing the generalization-identification tradeoff 

**Title (ZH)**: 受语义性约束：泛化-识别权衡的基本规律 

**Authors**: Marco Nurisso, Jesseba Fernando, Raj Deshpande, Alan Perotti, Raja Marjieh, Steven M. Frankland, Richard L. Lewis, Taylor W. Webb, Declan Campbell, Francesco Vaccarino, Jonathan D. Cohen, Giovanni Petri  

**Link**: [PDF](https://arxiv.org/pdf/2506.14797)  

**Abstract**: Intelligent systems must deploy internal representations that are simultaneously structured -- to support broad generalization -- and selective -- to preserve input identity. We expose a fundamental limit on this tradeoff. For any model whose representational similarity between inputs decays with finite semantic resolution $\varepsilon$, we derive closed-form expressions that pin its probability of correct generalization $p_S$ and identification $p_I$ to a universal Pareto front independent of input space geometry. Extending the analysis to noisy, heterogeneous spaces and to $n>2$ inputs predicts a sharp $1/n$ collapse of multi-input processing capacity and a non-monotonic optimum for $p_S$. A minimal ReLU network trained end-to-end reproduces these laws: during learning a resolution boundary self-organizes and empirical $(p_S,p_I)$ trajectories closely follow theoretical curves for linearly decaying similarity. Finally, we demonstrate that the same limits persist in two markedly more complex settings -- a convolutional neural network and state-of-the-art vision-language models -- confirming that finite-resolution similarity is a fundamental emergent informational constraint, not merely a toy-model artifact. Together, these results provide an exact theory of the generalization-identification trade-off and clarify how semantic resolution shapes the representational capacity of deep networks and brains alike. 

**Abstract (ZH)**: 智能系统必须部署同时具备结构化以支持广泛泛化和选择性以保留输入身份的内部表示。我们揭示了这一权衡的基本限制。对于任何代表相似性随有限语义分辨率ε衰减的模型，我们推导出闭合形式的表达式，将其正确泛化概率$p_S$和识别概率$p_I$限制在一个与输入空间几何形状无关的万能帕累托前沿。将分析扩展到噪声和异质空间以及$n>2$个输入预测了多输入处理能力的锐利$1/n$崩溃以及$p_S$的非单调最优值。端到端训练的极小ReLU网络再现了这些定律：在学习过程中，一个分辨率边界自行组织，并且经验$(p_S,p_I)$轨迹紧密遵循线性衰减相似性理论曲线。最后，我们在两个更为复杂的环境中展示了相同限制的存在——卷积神经网络和最先进的视觉-语言模型，证实有限分辨率相似性是一种基本 emergent 信息系统约束，而不仅仅是玩具模型的特征。这些结果共同提供了泛化与识别权衡的确切理论，并阐明了语义分辨率如何塑造深度网络乃至大脑的表示能力。 

---
# PFMBench: Protein Foundation Model Benchmark 

**Title (ZH)**: PFMBench: 蛋白质基础模型基准 

**Authors**: Zhangyang Gao, Hao Wang, Cheng Tan, Chenrui Xu, Mengdi Liu, Bozhen Hu, Linlin Chao, Xiaoming Zhang, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14796)  

**Abstract**: This study investigates the current landscape and future directions of protein foundation model research. While recent advancements have transformed protein science and engineering, the field lacks a comprehensive benchmark for fair evaluation and in-depth understanding. Since ESM-1B, numerous protein foundation models have emerged, each with unique datasets and methodologies. However, evaluations often focus on limited tasks tailored to specific models, hindering insights into broader generalization and limitations. Specifically, researchers struggle to understand the relationships between tasks, assess how well current models perform across them, and determine the criteria in developing new foundation models. To fill this gap, we present PFMBench, a comprehensive benchmark evaluating protein foundation models across 38 tasks spanning 8 key areas of protein science. Through hundreds of experiments on 17 state-of-the-art models across 38 tasks, PFMBench reveals the inherent correlations between tasks, identifies top-performing models, and provides a streamlined evaluation protocol. Code is available at \href{this https URL}{\textcolor{blue}{GitHub}}. 

**Abstract (ZH)**: 本研究探讨了蛋白质基础模型研究的当前状况和未来方向。尽管近期进展已改变蛋白质科学与工程的面貌，该领域仍缺乏一个全面的基准来公平评价和深入理解。自ESM-1B以来，涌现出了众多蛋白质基础模型，每个模型都有独特的数据集和方法。然而，评估往往集中在针对特定模型的特定任务上，阻碍了对更广泛泛化能力和限制的理解。具体而言，研究人员难以理解任务之间的关系，评估当前模型在这些任务上的表现情况，并确定开发新基础模型的标准。为了填补这一空白，我们提出了PFMBench基准，该基准涵盖38项任务，涉及蛋白质科学八大关键领域，对17个前沿模型进行了数百次实验，揭示了任务之间的固有关联，确定了表现最佳的模型，并提供了一套简化评估协议。代码可在GitHub（\href{this https URL}{GitHub}）获取。 

---
# Comparative Analysis of QNN Architectures for Wind Power Prediction: Feature Maps and Ansatz Configurations 

**Title (ZH)**: 基于量子神经网络架构的风功率预测比较分析：特征图和Ansatz配置 

**Authors**: Batuhan Hangun, Emine Akpinar, Oguz Altun, Onder Eyecioglu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14795)  

**Abstract**: Quantum Machine Learning (QML) is an emerging field at the intersection of quantum computing and machine learning, aiming to enhance classical machine learning methods by leveraging quantum mechanics principles such as entanglement and superposition. However, skepticism persists regarding the practical advantages of QML, mainly due to the current limitations of noisy intermediate-scale quantum (NISQ) devices. This study addresses these concerns by extensively assessing Quantum Neural Networks (QNNs)-quantum-inspired counterparts of Artificial Neural Networks (ANNs), demonstrating their effectiveness compared to classical methods. We systematically construct and evaluate twelve distinct QNN configurations, utilizing two unique quantum feature maps combined with six different entanglement strategies for ansatz design. Experiments conducted on a wind energy dataset reveal that QNNs employing the Z feature map achieve up to 93% prediction accuracy when forecasting wind power output using only four input parameters. Our findings show that QNNs outperform classical methods in predictive tasks, underscoring the potential of QML in real-world applications. 

**Abstract (ZH)**: 量子机器学习（QML）是量子计算与机器学习交叉领域的新兴领域，旨在通过利用量子力学原理如纠缠和叠加来增强经典机器学习方法。然而，关于QML实用优势的疑虑仍然存在，主要原因是目前_noisy intermediate-scale quantum (NISQ)_设备的局限性。本研究通过广泛评估量子神经网络（QNNs）-受人工神经网络（ANNs）启发的量子对应物，展示了它们在预测任务中的有效性，相比经典方法有所提升。我们系统地构建和评估了十二种不同的QNN配置，结合了两种独特的量子特征映射和六种不同的纠缠策略用于Ansatz设计。在风能数据集上的实验结果显示，使用Z特征映射的QNN在仅使用四个输入参数预测风功率输出时，可以达到93%的预测精度。我们的研究发现表明，QNN在预测任务中优于经典方法，强调了量子机器学习在实际应用中的潜力。 

---
# PIPE: Physics-Informed Position Encoding for Alignment of Satellite Images and Time Series 

**Title (ZH)**: PIPE：结合物理信息的位置编码在卫星图像与时序数据对齐中的应用 

**Authors**: Haobo Li, Eunseo Jung, Zixin Chen, Zhaowei Wang, Yueya Wang, Huamin Qu, Alexis Kai Hon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.14786)  

**Abstract**: Multimodal time series forecasting is foundational in various fields, such as utilizing satellite imagery and numerical data for predicting typhoons in climate science. However, existing multimodal approaches primarily focus on utilizing text data to help time series forecasting, leaving the visual data in existing time series datasets untouched. Furthermore, it is challenging for models to effectively capture the physical information embedded in visual data, such as satellite imagery's temporal and geospatial context, which extends beyond images themselves. To address this gap, we propose physics-informed positional encoding (PIPE), a lightweight method that embeds physical information into vision language models (VLMs). PIPE introduces two key innovations: (1) a physics-informed positional indexing scheme for mapping physics to positional IDs, and (2) a variant-frequency positional encoding mechanism for encoding frequency information of physical variables and sequential order of tokens within the embedding space. By preserving both the physical information and sequential order information, PIPE significantly improves multimodal alignment and forecasting accuracy. Through the experiments on the most representative and the largest open-sourced satellite image dataset, PIPE achieves state-of-the-art performance in both deep learning forecasting and climate domain methods, demonstrating superiority across benchmarks, including a 12% improvement in typhoon intensity forecasting over prior works. Our code is provided in the supplementary material. 

**Abstract (ZH)**: 多模态时间序列预测在利用卫星图像和数值数据预测台风等领域基础扎实，然而现有的多模态方法主要侧重于利用文本数据辅助时间序列预测，忽视了现有时间序列数据集中现有的视觉数据。同时，模型难以有效捕捉嵌入在视觉数据中的物理信息，如卫星图像的时间和地理空间上下文，这些信息超越了图像本身。为解决这一问题，我们提出了物理信息嵌入位置编码（PIPE），这是一种轻量级方法，将物理信息嵌入到视觉语言模型中。PIPE引入了两项关键创新：（1）一种物理信息导向的位置索引方案，用于将物理信息映射到位置ID，（2）一种变频位置编码机制，用于编码物理变量的频率信息以及嵌入空间中标记的顺序信息。通过保留物理信息和顺序信息，PIPE显著提高了多模态对齐和预测准确性。通过在最具代表性和最大的开源卫星图像数据集上进行实验，PIPE在深度学习预测和气候领域方法中均取得了最先进的性能，优于基准方法，台风强度预测性能提升了12%。我们在补充材料中提供了代码。 

---
# WebXAII: an open-source web framework to study human-XAI interaction 

**Title (ZH)**: WebXAII: 一个用于研究人类-XAI交互的开源网络框架 

**Authors**: Jules Leguy, Pierre-Antoine Jean, Felipe Torres Figueroa, Sébastien Harispe  

**Link**: [PDF](https://arxiv.org/pdf/2506.14777)  

**Abstract**: This article introduces WebXAII, an open-source web framework designed to facilitate research on human interaction with eXplainable Artificial Intelligence (XAI) systems. The field of XAI is rapidly expanding, driven by the growing societal implications of the widespread adoption of AI (and in particular machine learning) across diverse applications. Researchers who study the interaction between humans and XAI techniques typically develop ad hoc interfaces in order to conduct their studies. These interfaces are usually not shared alongside the results of the studies, which limits their reusability and the reproducibility of experiments. In response, we design and implement WebXAII, a web-based platform that can embody full experimental protocols, meaning that it can present all aspects of the experiment to human participants and record their responses. The experimental protocols are translated into a composite architecture of generic views and modules, which offers a lot of flexibility. The architecture is defined in a structured configuration file, so that protocols can be implemented with minimal programming skills. We demonstrate that WebXAII can effectively embody relevant protocols, by reproducing the protocol of a state-of-the-art study of the literature. The framework is available at this https URL. 

**Abstract (ZH)**: 本文介绍了WebXAII，一个开源框架，旨在促进对可解释人工智能(XAI)系统的人机交互研究。随着AI（特别是机器学习）在各类应用中广泛应用所带来的社会影响日益显著，XAI领域正在迅速扩展。研究人机交互的研究人员通常会开发临时界面来进行研究，但这些界面通常不会与研究结果一并分享，这限制了它们的可重用性和实验的可再现性。为此，我们设计并实现了WebXAII，这是一个基于Web的平台，能够涵盖完整的实验流程，即它可以向人类参与者呈现实验的所有方面并记录其反应。实验流程被翻译成一个由通用视图和模块组成的综合架构，提供了很大的灵活性。架构定义在一个结构化的配置文件中，使得协议可以用最少的编程技能来实现。我们通过重现文献中一项前沿研究的实验流程，证明了WebXAII的有效性。该框架可在以下链接获取：这是一个HTTPS链接。 

---
# See What I Mean? CUE: A Cognitive Model of Understanding Explanations 

**Title (ZH)**: 见我所言？CUE：一种理解解释的心智模型 

**Authors**: Tobias Labarta, Nhi Hoang, Katharina Weitz, Wojciech Samek, Sebastian Lapuschkin, Leander Weber  

**Link**: [PDF](https://arxiv.org/pdf/2506.14775)  

**Abstract**: As machine learning systems increasingly inform critical decisions, the need for human-understandable explanations grows. Current evaluations of Explainable AI (XAI) often prioritize technical fidelity over cognitive accessibility which critically affects users, in particular those with visual impairments. We propose CUE, a model for Cognitive Understanding of Explanations, linking explanation properties to cognitive sub-processes: legibility (perception), readability (comprehension), and interpretability (interpretation). In a study (N=455) testing heatmaps with varying colormaps (BWR, Cividis, Coolwarm), we found comparable task performance but lower confidence/effort for visually impaired users. Unlike expected, these gaps were not mitigated and sometimes worsened by accessibility-focused color maps like Cividis. These results challenge assumptions about perceptual optimization and support the need for adaptive XAI interfaces. They also validate CUE by demonstrating that altering explanation legibility affects understandability. We contribute: (1) a formalized cognitive model for explanation understanding, (2) an integrated definition of human-centered explanation properties, and (3) empirical evidence motivating accessible, user-tailored XAI. 

**Abstract (ZH)**: 随着机器学习系统在越来越多的关键决策中发挥作用，对人类可理解解释的需要也在增长。当前对可解释人工智能（XAI）的评估往往优先考虑技术准确度而忽视了认知易用性，这严重影响了用户，特别是视觉障碍用户。我们提出了CUE模型，链接了解释属性与认知子过程：可读性（感知）、可读性（理解）和解释性（解释）。在一项包含455名受试者的研究中，我们发现不同色板（BWR、Cividis、Coolwarm）的热图在任务性能上表现出色，但视觉障碍用户的信心和努力程度较低。出乎意料的是，这些差距并未被聚焦于无障碍性的色板如Cividis减轻，有时甚至加剧。这些结果挑战了关于感知优化的假设，并支持需要适应性XAI界面。它们还通过实证证明改变了解释的可读性会影响理解性，从而验证了CUE。我们贡献了：（1）一种形式化的认知模型以解释理解，（2）一个联系人类中心解释属性的综合定义，以及（3）支持可访问性和用户定制的XAI的实验证据。 

---
# MixRep: Hidden Representation Mixup for Low-Resource Speech Recognition 

**Title (ZH)**: MixRep: 隐藏表示混叠方法在低资源语音识别中的应用 

**Authors**: Jiamin Xie, John H.L. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2310.18450)  

**Abstract**: In this paper, we present MixRep, a simple and effective data augmentation strategy based on mixup for low-resource ASR. MixRep interpolates the feature dimensions of hidden representations in the neural network that can be applied to both the acoustic feature input and the output of each layer, which generalizes the previous MixSpeech method. Further, we propose to combine the mixup with a regularization along the time axis of the input, which is shown as complementary. We apply MixRep to a Conformer encoder of an E2E LAS architecture trained with a joint CTC loss. We experiment on the WSJ dataset and subsets of the SWB dataset, covering reading and telephony conversational speech. Experimental results show that MixRep consistently outperforms other regularization methods for low-resource ASR. Compared to a strong SpecAugment baseline, MixRep achieves a +6.5\% and a +6.7\% relative WER reduction on the eval92 set and the Callhome part of the eval'2000 set. 

**Abstract (ZH)**: 基于mixup的MixRep：一种适用于低资源ASR的简单有效数据增强策略 

---
# Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model 

**Title (ZH)**: 动态ASR路径：面向多语言ASR模型高效剪枝的自适应掩蔽方法 

**Authors**: Jiamin Xie, Ke Li, Jinxi Guo, Andros Tjandra, Yuan Shangguan, Leda Sari, Chunyang Wu, Junteng Jia, Jay Mahadeokar, Ozlem Kalinli  

**Link**: [PDF](https://arxiv.org/pdf/2309.13018)  

**Abstract**: Neural network pruning offers an effective method for compressing a multilingual automatic speech recognition (ASR) model with minimal performance loss. However, it entails several rounds of pruning and re-training needed to be run for each language. In this work, we propose the use of an adaptive masking approach in two scenarios for pruning a multilingual ASR model efficiently, each resulting in sparse monolingual models or a sparse multilingual model (named as Dynamic ASR Pathways). Our approach dynamically adapts the sub-network, avoiding premature decisions about a fixed sub-network structure. We show that our approach outperforms existing pruning methods when targeting sparse monolingual models. Further, we illustrate that Dynamic ASR Pathways jointly discovers and trains better sub-networks (pathways) of a single multilingual model by adapting from different sub-network initializations, thereby reducing the need for language-specific pruning. 

**Abstract (ZH)**: 神经网络剪枝提供了一种有效的方法来压缩多语言自动语音识别（ASR）模型，同时将性能损失降至最低。然而，这需要为每种语言运行多轮剪枝和重新训练。在本文中，我们提出了一种自适应掩码方法，在两种场景下高效地修剪多语言ASR模型，分别生成稀疏的单语言模型或稀疏的多语言模型（名为动态ASR路径）。我们的方法动态地适应子网络，避免了过早决定固定子网络结构。我们显示，当目标是稀疏的单语言模型时，我们的方法优于现有的剪枝方法。进一步地，我们展示了动态ASR路径通过从不同的子网络初始化进行适应，共同发现和训练单多语言模型的更好子网络（路径），从而减少了语言特定剪枝的需要。 

---
# DEFORMER: Coupling Deformed Localized Patterns with Global Context for Robust End-to-end Speech Recognition 

**Title (ZH)**: DEFORMER: 结合变形局部模式与全局上下文以实现稳健的端到端语音识别 

**Authors**: Jiamin Xie, John H.L. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2207.01732)  

**Abstract**: Convolutional neural networks (CNN) have improved speech recognition performance greatly by exploiting localized time-frequency patterns. But these patterns are assumed to appear in symmetric and rigid kernels by the conventional CNN operation. It motivates the question: What about asymmetric kernels? In this study, we illustrate adaptive views can discover local features which couple better with attention than fixed views of the input. We replace depthwise CNNs in the Conformer architecture with a deformable counterpart, dubbed this "Deformer". By analyzing our best-performing model, we visualize both local receptive fields and global attention maps learned by the Deformer and show increased feature associations on the utterance level. The statistical analysis of learned kernel offsets provides an insight into the change of information in features with the network depth. Finally, replacing only half of the layers in the encoder, the Deformer improves +5.6% relative WER without a LM and +6.4% relative WER with a LM over the Conformer baseline on the WSJ eval92 set. 

**Abstract (ZH)**: 卷积神经网络通过利用局部时频模式大幅提高了语音识别性能，但这些模式假定以传统的卷积操作对称和刚性的核出现。这引发了这样的问题：非对称核会怎样？在此研究中，我们展示了自适应视角可以发现与注意力耦合更好的局部特征，优于固定视角。我们将Conformer架构中的深度卷积神经网络替换为一个可变形的对应物，称为“Deformer”。通过分析表现最佳的模型，我们可视化了Deformer学习到的局部感受野和全局注意力图，并展示了在短语水平上特征关联的增强。学习到的核偏移的统计分析揭示了网络深度对特征中信息变化的洞察。最后，在编码器中仅替换一半层，Deformer在WSJ eval92集上相对于Conformer基线，无语言模型获得了相对WER提高5.6%，有语言模型提高了6.4%。 

---
