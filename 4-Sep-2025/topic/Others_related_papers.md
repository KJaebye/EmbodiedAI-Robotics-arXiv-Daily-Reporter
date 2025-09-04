# Can the Waymo Open Motion Dataset Support Realistic Behavioral Modeling? A Validation Study with Naturalistic Trajectories 

**Title (ZH)**: Waymo开放运动数据集能否支持现实行为建模？基于自然轨迹的验证研究 

**Authors**: Yanlin Zhang, Sungyong Chung, Nachuan Li, Dana Monzer, Hani S. Mahmassani, Samer H. Hamdar, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2509.03515)  

**Abstract**: The Waymo Open Motion Dataset (WOMD) has become a popular resource for data-driven modeling of autonomous vehicles (AVs) behavior. However, its validity for behavioral analysis remains uncertain due to proprietary post-processing, the absence of error quantification, and the segmentation of trajectories into 20-second clips. This study examines whether WOMD accurately captures the dynamics and interactions observed in real-world AV operations. Leveraging an independently collected naturalistic dataset from Level 4 AV operations in Phoenix, Arizona (PHX), we perform comparative analyses across three representative urban driving scenarios: discharging at signalized intersections, car-following, and lane-changing behaviors. For the discharging analysis, headways are manually extracted from aerial video to ensure negligible measurement error. For the car-following and lane-changing cases, we apply the Simulation-Extrapolation (SIMEX) method to account for empirically estimated error in the PHX data and use Dynamic Time Warping (DTW) distances to quantify behavioral differences. Results across all scenarios consistently show that behavior in PHX falls outside the behavioral envelope of WOMD. Notably, WOMD underrepresents short headways and abrupt decelerations. These findings suggest that behavioral models calibrated solely on WOMD may systematically underestimate the variability, risk, and complexity of naturalistic driving. Caution is therefore warranted when using WOMD for behavior modeling without proper validation against independently collected data. 

**Abstract (ZH)**: Waymo开放运动数据集（WOMD）在自主车辆行为建模中的适用性分析：基于亚利桑那州凤凰城（PHX）Level 4自主车辆自然驾驶数据的比较研究 

---
# Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression 

**Title (ZH)**: 面向高效实时域自适应密集回归的不确定性感知测试时训练（UT$^3$） 

**Authors**: Uddeshya Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2509.03012)  

**Abstract**: Deep neural networks (DNNs) are increasingly being used in autonomous systems. However, DNNs do not generalize well to domain shift. Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous systems deployed to the real world. Recent work on test-time training proposes methods that adapt to a new test distribution on the fly by optimizing the DNN model for each test input using self-supervision. However, these techniques result in a sharp increase in inference time as multiple forward and backward passes are required for a single test sample (for test-time training) before finally making the prediction based on the fine-tuned features. This is undesirable for real-world robotics applications where these models may be deployed to resource constraint hardware with strong latency requirements. In this work, we propose a new framework (called UT$^3$) that leverages test-time training for improved performance in the presence of continuous domain shift while also decreasing the inference time, making it suitable for real-world applications. Our method proposes an uncertainty-aware self-supervision task for efficient test-time training that leverages the quantified uncertainty to selectively apply the training leading to sharp improvements in the inference time while performing comparably to standard test-time training protocol. Our proposed protocol offers a continuous setting to identify the selected keyframes, allowing the end-user to control how often to apply test-time training. We demonstrate the efficacy of our method on a dense regression task - monocular depth estimation. 

**Abstract (ZH)**: 基于连续领域转移的高效测试时训练框架UT$^3$ 

---
# IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments 

**Title (ZH)**: 基于特征awareness的智能线协助SLAM用于动态环境 

**Authors**: Haolan Zhang, Thanh Nguyen Canh, Chenghao Li, Ruidong Yang, Yonghoon Ji, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02972)  

**Abstract**: Visual Simultaneous Localization and Mapping (SLAM) plays a crucial role in autonomous systems. Traditional SLAM methods, based on static environment assumptions, struggle to handle complex dynamic environments. Recent dynamic SLAM systems employ geometric constraints and deep learning to remove dynamic features, yet this creates a new challenge: insufficient remaining point features for subsequent SLAM processes. Existing solutions address this by continuously introducing additional line and plane features to supplement point features, achieving robust tracking and pose estimation. However, current methods continuously introduce additional features regardless of necessity, causing two problems: unnecessary computational overhead and potential performance degradation from accumulated low-quality additional features and noise. To address these issues, this paper proposes a feature-aware mechanism that evaluates whether current features are adequate to determine if line feature support should be activated. This decision mechanism enables the system to introduce line features only when necessary, significantly reducing computational complexity of additional features while minimizing the introduction of low-quality features and noise. In subsequent processing, the introduced line features assist in obtaining better initial camera poses through tracking, local mapping, and loop closure, but are excluded from global optimization to avoid potential negative impacts from low-quality additional features in long-term process. Extensive experiments on TUM datasets demonstrate substantial improvements in both ATE and RPE metrics compared to ORB-SLAM3 baseline and superior performance over other dynamic SLAM and multi-feature methods. 

**Abstract (ZH)**: 视觉 simultaneous localization and mapping (SLAM) 在自主系统中发挥着关键作用。传统的 SLAM 方法基于静态环境假设，难以处理复杂的动态环境。最近的动态 SLAM 系统采用几何约束和深度学习来移除动态特征，但这也带来了一个新的挑战：剩余的点特征不足，不足以支持后续的 SLAM 过程。现有的解决方案通过不断引入额外的线性和平面特征来补充点特征，实现稳健的跟踪和姿态估计。然而，当前的方法在不需要时也不断引入额外特征，这导致了两个问题：不必要的计算开销和由于低质量的额外特征和噪声累积而导致的潜在性能下降。为了应对这些问题，本文提出了一种特征感知机制，该机制评估当前特征是否足够，以确定是否激活线特征支持。这种决策机制能使系统仅在必要时引入线特征，显著减少了额外特征的计算复杂性，同时减少了低质量特征和噪声的引入。在后续处理中，引入的线特征通过跟踪、局部地图构建和环视闭合辅助获得更好的初始相机姿态，但在全局优化中被排除，以避免长期过程中低质量额外特征的潜在负面影响。在 TUM 数据集上的 extensive 实验表明，与 ORB-SLAM3 基准相比，该方法在 ATE 和 RPE 指标上取得了显著的改进，并且在与其他动态 SLAM 和多特征方法的性能比较中表现出优越性。 

---
# SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data 

**Title (ZH)**: SmartPoser：基于UWB和IMU数据的智能手机和智能手表臂部姿态估计 

**Authors**: Nathan DeVrio, Vimal Mollyn, Chris Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2509.03451)  

**Abstract**: The ability to track a user's arm pose could be valuable in a wide range of applications, including fitness, rehabilitation, augmented reality input, life logging, and context-aware assistants. Unfortunately, this capability is not readily available to consumers. Systems either require cameras, which carry privacy issues, or utilize multiple worn IMUs or markers. In this work, we describe how an off-the-shelf smartphone and smartwatch can work together to accurately estimate arm pose. Moving beyond prior work, we take advantage of more recent ultra-wideband (UWB) functionality on these devices to capture absolute distance between the two devices. This measurement is the perfect complement to inertial data, which is relative and suffers from drift. We quantify the performance of our software-only approach using off-the-shelf devices, showing it can estimate the wrist and elbow joints with a \hl{median positional error of 11.0~cm}, without the user having to provide training data. 

**Abstract (ZH)**: 一款智能手机和智能手表协同工作的方法可以准确估计手臂姿态 

---
# AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation 

**Title (ZH)**: 电动汽车中的人工智能安全保障：基于人工智能驱动的SOC估算案例研究 

**Authors**: Martin Skoglund, Fredrik Warg, Aria Mirzai, Anders Thorsen, Karl Lundgren, Peter Folkesson, Bastian Havers-zulka  

**Link**: [PDF](https://arxiv.org/pdf/2509.03270)  

**Abstract**: Integrating Artificial Intelligence (AI) technology in electric vehicles (EV) introduces unique challenges for safety assurance, particularly within the framework of ISO 26262, which governs functional safety in the automotive domain. Traditional assessment methodologies are not geared toward evaluating AI-based functions and require evolving standards and practices. This paper explores how an independent assessment of an AI component in an EV can be achieved when combining ISO 26262 with the recently released ISO/PAS 8800, whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC) battery estimation exemplifies the process. Key features relevant to the independent assessment of this extended evaluation approach are identified. As part of the evaluation, robustness testing of the AI component is conducted using fault injection experiments, wherein perturbed sensor inputs are systematically introduced to assess the component's resilience to input variance. 

**Abstract (ZH)**: 将人工智能技术集成到电动汽车中，为ISO 26262框架下的功能安全保证带来了独特的挑战。本文探讨了如何结合ISO 26262和近期发布的ISO/PAS 8800来独立评估电动汽车中的人工智能组件，后者专注于道路车辆的人工智能安全。以基于人工智能的电池荷电状态（SOC）估算为例，识别了这种扩展评估方法中关键的相关特性。作为评估的一部分，通过故障注入实验，对人工智能组件的鲁棒性进行了测试，系统地引入扰动传感器输入以评估其对输入变化的抗扰性。 

---
# Population-aware Online Mirror Descent for Mean-Field Games with Common Noise by Deep Reinforcement Learning 

**Title (ZH)**: 基于种群感知的在线镜像下降方法用于具有公共噪声的大规模动态博弈的深度强化学习 

**Authors**: Zida Wu, Mathieu Lauriere, Matthieu Geist, Olivier Pietquin, Ankur Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.03030)  

**Abstract**: Mean Field Games (MFGs) offer a powerful framework for studying large-scale multi-agent systems. Yet, learning Nash equilibria in MFGs remains a challenging problem, particularly when the initial distribution is unknown or when the population is subject to common noise. In this paper, we introduce an efficient deep reinforcement learning (DRL) algorithm designed to achieve population-dependent Nash equilibria without relying on averaging or historical sampling, inspired by Munchausen RL and Online Mirror Descent. The resulting policy is adaptable to various initial distributions and sources of common noise. Through numerical experiments on seven canonical examples, we demonstrate that our algorithm exhibits superior convergence properties compared to state-of-the-art algorithms, particularly a DRL version of Fictitious Play for population-dependent policies. The performance in the presence of common noise underscores the robustness and adaptability of our approach. 

**Abstract (ZH)**: MFGs基于Munchausen RL和在线镜像下降的自适应深度 reinforcement学习算法及应用 

---
# Approximate constrained stochastic optimal control via parameterized input inference 

**Title (ZH)**: 参数化输入推断实现近似约束随机最优控制 

**Authors**: Shahbaz P Qadri Syed, He Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.02922)  

**Abstract**: Approximate methods to solve stochastic optimal control (SOC) problems have received significant interest from researchers in the past decade. Probabilistic inference approaches to SOC have been developed to solve nonlinear quadratic Gaussian problems. In this work, we propose an Expectation-Maximization (EM) based inference procedure to generate state-feedback controls for constrained SOC problems. We consider the inequality constraints for the state and controls and also the structural constraints for the controls. We employ barrier functions to address state and control constraints. We show that the expectation step leads to smoothing of the state-control pair while the the maximization step on the non-zero subsets of the control parameters allows inference of structured stochastic optimal controllers. We demonstrate the effectiveness of the algorithm on unicycle obstacle avoidance, four-unicycle formation control, and quadcopter navigation in windy environment examples. In these examples, we perform an empirical study on the parametric effect of barrier functions on the state constraint satisfaction. We also present a comparative study of smoothing algorithms on the performance of the proposed approach. 

**Abstract (ZH)**: 基于EM方法求解约束随机最优控制问题的近似方法 

---
# Who Owns The Robot?: Four Ethical and Socio-technical Questions about Wellbeing Robots in the Real World through Community Engagement 

**Title (ZH)**: 谁拥有机器人？通过社区参与探索现实世界中福祉机器人的人文与社会技术问题四问 

**Authors**: Minja Axelsson, Jiaee Cheong, Rune Nyrup, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2509.02624)  

**Abstract**: Recent studies indicate that robotic coaches can play a crucial role in promoting wellbeing. However, the real-world deployment of wellbeing robots raises numerous ethical and socio-technical questions and concerns. To explore these questions, we undertake a community-centered investigation to examine three different communities' perspectives on using robotic wellbeing coaches in real-world environments. We frame our work as an anticipatory ethical investigation, which we undertake to better inform the development of robotic technologies with communities' opinions, with the ultimate goal of aligning robot development with public interest. We conducted workshops with three communities who are under-represented in robotics development: 1) members of the public at a science festival, 2) women computer scientists at a conference, and 3) humanities researchers interested in history and philosophy of science. In the workshops, we collected qualitative data using the Social Robot Co-Design Canvas on Ethics. We analysed the collected qualitative data with Thematic Analysis, informed by notes taken during workshops. Through our analysis, we identify four themes regarding key ethical and socio-technical questions about the real-world use of wellbeing robots. We group participants' insights and discussions around these broad thematic questions, discuss them in light of state-of-the-art literature, and highlight areas for future investigation. Finally, we provide the four questions as a broad framework that roboticists can and should use during robotic development and deployment, in order to reflect on the ethics and socio-technical dimensions of their robotic applications, and to engage in dialogue with communities of robot users. The four questions are: 1) Is the robot safe and how can we know that?, 2) Who is the robot built for and with?, 3) Who owns the robot and the data?, and 4) Why a robot?. 

**Abstract (ZH)**: 近期研究表明，机器人教练在促进福祉方面起着关键作用。然而，福祉机器人的实际部署引发了众多伦理和社会技术方面的问题与担忧。为了探索这些问题，我们开展了一项以社区为中心的调查，考察了三个不同社区对在实际环境使用机器人福祉教练的看法。我们将我们的工作视为一种预见性伦理调查，旨在通过社区意见更好地指导机器人技术的发展，最终目标是使机器人开发与公众利益相一致。我们与在机器人开发中代表性不足的三个社区进行了工作坊：1）科学节上的公众成员，2）女性计算机科学家，以及3）对科学史和哲学感兴趣的 humanities 研究者。在工作坊中，我们使用社会机器人共设计伦理画布收集定性数据。我们通过工作坊期间记录的笔记进行主题分析，识别出了四个关键的主题关于福祉机器人在实际应用中的伦理和社会技术问题。我们总结了参与者关于这些问题的见解和讨论，结合当前先进文献进行了讨论，并指出了未来研究的重点领域。最后，我们提出了这四个问题作为机器人研究者在机器人开发和部署过程中应使用和关注的广泛框架，以反思其机器人应用的伦理和社会技术维度，并与机器人用户社区进行对话。这四个问题是：1）机器人是否安全，我们如何知道？2）机器人是为谁设计和构建的？3）机器人及其数据的所有权是谁？4）为什么要使用机器人？ 

---
# Accountability Framework for Healthcare AI Systems: Towards Joint Accountability in Decision Making 

**Title (ZH)**: 医疗健康领域AI系统问责框架：走向决策中的共同问责 

**Authors**: Prachi Bagave, Marcus Westberg, Marijn Janssen, Aaron Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.03286)  

**Abstract**: AI is transforming the healthcare domain and is increasingly helping practitioners to make health-related decisions. Therefore, accountability becomes a crucial concern for critical AI-driven decisions. Although regulatory bodies, such as the EU commission, provide guidelines, they are highlevel and focus on the ''what'' that should be done and less on the ''how'', creating a knowledge gap for actors. Through an extensive analysis, we found that the term accountability is perceived and dealt with in many different ways, depending on the actor's expertise and domain of work. With increasing concerns about AI accountability issues and the ambiguity around this term, this paper bridges the gap between the ''what'' and ''how'' of AI accountability, specifically for AI systems in healthcare. We do this by analysing the concept of accountability, formulating an accountability framework, and providing a three-tier structure for handling various accountability mechanisms. Our accountability framework positions the regulations of healthcare AI systems and the mechanisms adopted by the actors under a consistent accountability regime. Moreover, the three-tier structure guides the actors of the healthcare AI system to categorise the mechanisms based on their conduct. Through our framework, we advocate that decision-making in healthcare AI holds shared dependencies, where accountability should be dealt with jointly and should foster collaborations. We highlight the role of explainability in instigating communication and information sharing between the actors to further facilitate the collaborative process. 

**Abstract (ZH)**: AI在医疗健康领域的应用正在不断改变医疗领域，越来越多地帮助医疗从业者做出健康相关的决策。因此，对于关键的AI驱动决策，问责制成为了一个重要的关注点。尽管欧盟委员会等监管机构提供了指导方针，但这些指导方针往往是高层次的，更多关注“应该做什么”，而对“如何做”则关注较少，造成了行动者之间的知识空白。通过广泛分析，我们发现问责制这一概念在不同行动者之间被理解和处理的方式多种多样，这取决于他们的专业知识和工作领域。鉴于对AI问责制问题日益增长的担忧以及此概念的模糊性，本文旨在弥合“应该做什么”与“如何做”的问责制差距，特别是针对医疗健康领域的AI系统。我们通过分析问责制概念、构建问责制框架，并提供一种多层次结构来处理各种问责机制，来实现这一目标。我们的问责制框架将医疗AI系统的法规与行动者所采用的机制置于一个一致的问责体系之下。此外，三层结构指导医疗AI系统的行动者根据其行为对机制进行分类。通过我们的框架，我们提倡医疗AI决策应共享依赖关系，问责制应当共同处理，并促进合作。我们强调可解释性在促进行动者之间的沟通和信息共享方面的作用，进一步促进协作过程。 

---
# Uncertainty-driven Adaptive Exploration 

**Title (ZH)**: 不确定性驱动的自适应探索 

**Authors**: Leonidas Bakopoulos, Georgios Chalkiadakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.03219)  

**Abstract**: Adaptive exploration methods propose ways to learn complex policies via alternating between exploration and exploitation. An important question for such methods is to determine the appropriate moment to switch between exploration and exploitation and vice versa. This is critical in domains that require the learning of long and complex sequences of actions. In this work, we present a generic adaptive exploration framework that employs uncertainty to address this important issue in a principled manner. Our framework includes previous adaptive exploration approaches as special cases. Moreover, we can incorporate in our framework any uncertainty-measuring mechanism of choice, for instance mechanisms used in intrinsic motivation or epistemic uncertainty-based exploration methods. We experimentally demonstrate that our framework gives rise to adaptive exploration strategies that outperform standard ones across several MuJoCo environments. 

**Abstract (ZH)**: 自适应探索方法通过在探索和利用之间交替来学习复杂策略。这类方法的一个重要问题是确定在何时切换探索和利用以及反之亦然的适当时刻。在需要学习长且复杂的动作序列的领域中，这一点至关重要。在这项工作中，我们提出了一种基于不确定性的一般自适应探索框架，以基本原则的方式解决这一重要问题。我们的框架包含了先前的自适应探索方法作为特殊情况。此外，我们可以将任何自选的不确定性测量机制纳入我们的框架，例如内在动机或基于认识不确定性探索方法中使用的机制。实验结果表明，我们的框架产生了在多个MuJoCo环境中优于标准方法的自适应探索策略。 

---
# Learning General Policies From Examples 

**Title (ZH)**: 从示例中学习通用策略 

**Authors**: Blai Bonet, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02794)  

**Abstract**: Combinatorial methods for learning general policies that solve large collections of planning problems have been recently developed. One of their strengths, in relation to deep learning approaches, is that the resulting policies can be understood and shown to be correct. A weakness is that the methods do not scale up and learn only from small training instances and feature pools that contain a few hundreds of states and features at most. In this work, we propose a new symbolic method for learning policies based on the generalization of sampled plans that ensures structural termination and hence acyclicity. The proposed learning approach is not based on SAT/ASP, as previous symbolic methods, but on a hitting set algorithm that can effectively handle problems with millions of states, and pools with hundreds of thousands of features. The formal properties of the approach are analyzed, and its scalability is tested on a number of benchmarks. 

**Abstract (ZH)**: 基于采样计划泛化的符号学习方法：确保结构终止并有效处理大规模状态和特征池的问题 

---
# Key Principles in Cross-Domain Hyper-Heuristic Performance 

**Title (ZH)**: 跨领域超元启发式性能的关键原则 

**Authors**: Václav Sobotka, Lucas Kletzander, Nysret Musliu, Hana Rudová  

**Link**: [PDF](https://arxiv.org/pdf/2509.02782)  

**Abstract**: Cross-domain selection hyper-heuristics aim to distill decades of research on problem-specific heuristic search algorithms into adaptable general-purpose search strategies. In this respect, existing selection hyper-heuristics primarily focus on an adaptive selection of low-level heuristics (LLHs) from a predefined set. In contrast, we concentrate on the composition of this set and its strategic transformations. We systematically analyze transformations based on three key principles: solution acceptance, LLH repetitions, and perturbation intensity, i.e., the proportion of a solution affected by a perturbative LLH. We demonstrate the raw effects of our transformations on a trivial unbiased random selection mechanism. With an appropriately constructed transformation, this trivial method outperforms all available state-of-the-art hyper-heuristics on three challenging real-world domains and finds 11 new best-known solutions. The same method is competitive with the winner of the CHeSC competition, commonly used as the standard cross-domain benchmark. Moreover, we accompany several recent hyper-heuristics with such strategic transformations. Using this approach, we outperform the current state-of-the-art methods on both the CHeSC benchmark and real-world domains while often simplifying their designs. 

**Abstract (ZH)**: 跨域选择超启发式方法旨在将数十年针对具体问题启发式搜索算法的研究成果提炼为可适应的一般搜索策略。在这方面，现有的选择超启发式方法主要集中在从预定义集合中适配选择低级启发式算法（LLHs）。相比之下，我们专注于集合的构成及其战略变换。我们系统地分析了基于三个关键原则的变换：解接受、LLH重复以及扰动强度，即扰动性LLH影响解的比例。我们展示了我们的变换对一个简单无偏随机选择机制的基础效果。通过适当构造的变换，该简单方法在三个具有挑战性的实际领域中均优于所有现有的最先进的超启发式方法，并找到11个新的最优解。该方法在CHeSC比赛的冠军和常用的跨域基准测试中具有竞争力。此外，我们还为几种近期的超启发式方法配备了此类战略变换。采用这种方法，我们在CHeSC基准和实际领域中均优于当前最先进的方法，同时往往简化了它们的设计。 

---
# The Future of Artificial Intelligence and the Mathematical and Physical Sciences (AI+MPS) 

**Title (ZH)**: 人工智能的未来与数学及物理科学（AI+MPS） 

**Authors**: Andrew Ferguson, Marisa LaFleur, Lars Ruthotto, Jesse Thaler, Yuan-Sen Ting, Pratyush Tiwary, Soledad Villar, E. Paulo Alves, Jeremy Avigad, Simon Billinge, Camille Bilodeau, Keith Brown, Emmanuel Candes, Arghya Chattopadhyay, Bingqing Cheng, Jonathan Clausen, Connor Coley, Andrew Connolly, Fred Daum, Sijia Dong, Chrisy Xiyu Du, Cora Dvorkin, Cristiano Fanelli, Eric B. Ford, Luis Manuel Frutos, Nicolás García Trillos, Cecilia Garraffo, Robert Ghrist, Rafael Gomez-Bombarelli, Gianluca Guadagni, Sreelekha Guggilam, Sergei Gukov, Juan B. Gutiérrez, Salman Habib, Johannes Hachmann, Boris Hanin, Philip Harris, Murray Holland, Elizabeth Holm, Hsin-Yuan Huang, Shih-Chieh Hsu, Nick Jackson, Olexandr Isayev, Heng Ji, Aggelos Katsaggelos, Jeremy Kepner, Yannis Kevrekidis, Michelle Kuchera, J. Nathan Kutz, Branislava Lalic, Ann Lee, Matt LeBlanc, Josiah Lim, Rebecca Lindsey, Yongmin Liu, Peter Y. Lu, Sudhir Malik, Vuk Mandic, Vidya Manian, Emeka P. Mazi, Pankaj Mehta, Peter Melchior, Brice Ménard, Jennifer Ngadiuba, Stella Offner, Elsa Olivetti, Shyue Ping Ong, Christopher Rackauckas, Philippe Rigollet, Chad Risko, Philip Romero, Grant Rotskoff, Brett Savoie, Uros Seljak, David Shih, Gary Shiu, Dima Shlyakhtenko, Eva Silverstein, Taylor Sparks, Thomas Strohmer, Christopher Stubbs, Stephen Thomas, Suriyanarayanan Vaikuntanathan, Rene Vidal, Francisco Villaescusa-Navarro, Gregory Voth, Benjamin Wandelt, Rachel Ward, Melanie Weber, Risa Wechsler, Stephen Whitelam, Olaf Wiest, Mike Williams, Zhuoran Yang, Yaroslava G. Yingling, Bin Yu, Shuwen Yue, Ann Zabludoff, Huimin Zhao, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02661)  

**Abstract**: This community paper developed out of the NSF Workshop on the Future of Artificial Intelligence (AI) and the Mathematical and Physics Sciences (MPS), which was held in March 2025 with the goal of understanding how the MPS domains (Astronomy, Chemistry, Materials Research, Mathematical Sciences, and Physics) can best capitalize on, and contribute to, the future of AI. We present here a summary and snapshot of the MPS community's perspective, as of Spring/Summer 2025, in a rapidly developing field. The link between AI and MPS is becoming increasingly inextricable; now is a crucial moment to strengthen the link between AI and Science by pursuing a strategy that proactively and thoughtfully leverages the potential of AI for scientific discovery and optimizes opportunities to impact the development of AI by applying concepts from fundamental science. To achieve this, we propose activities and strategic priorities that: (1) enable AI+MPS research in both directions; (2) build up an interdisciplinary community of AI+MPS researchers; and (3) foster education and workforce development in AI for MPS researchers and students. We conclude with a summary of suggested priorities for funding agencies, educational institutions, and individual researchers to help position the MPS community to be a leader in, and take full advantage of, the transformative potential of AI+MPS. 

**Abstract (ZH)**: 本研究社区论文源自2025年3月举行的NSF人工智能（AI）与数学及物理科学（MPS）未来研讨会，旨在了解数学及物理科学领域（天文学、化学、材料研究、数学科学和物理学）如何最有效地利用AI，并为AI的未来发展做出贡献。本文提供了截至2025年春季/夏季数学及物理科学社区观点的总结和快照，展示了该领域的快速发展。AI与数学及物理科学之间的联系越来越密不可分；现在是加强AI与科学之间联系的关键时刻，通过积极和深思熟虑地利用AI的潜力促进科学发现，并通过应用基础科学的概念优化影响AI发展的机会。为此，我们提出了以下活动和战略优先事项：（1）实现双向的AI+MPS研究；（2）建立跨学科的AI+MPS研究人员社区；（3）促进AI教育和工作队伍发展，为MPS研究人员和学生。最后，我们总结了对资助机构、教育机构和个人研究人员的建议优先事项，以帮助数学及物理科学社区在AI+MPS的变革潜力方面处于领先地位并充分利用其潜力。 

---
# Can Media Act as a Soft Regulator of Safe AI Development? A Game Theoretical Analysis 

**Title (ZH)**: 媒体能在安全人工智能发展软监管中发挥作用吗？一种博弈理论分析 

**Authors**: Henrique Correia da Fonseca, António Fernandes, Zhao Song, Theodor Cimpeanu, Nataliya Balabanova, Adeela Bashir, Paolo Bova, Alessio Buscemi, Alessandro Di Stefano, Manh Hong Duong, Elias Fernandez Domingos, Ndidi Bianca Ogbo, Simon T. Powers, Daniele Proverbio, Zia Ush Shamszaman, Fernando P. Santos, Anh Han, Marcus Krellner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02650)  

**Abstract**: When developers of artificial intelligence (AI) products need to decide between profit and safety for the users, they likely choose profit. Untrustworthy AI technology must come packaged with tangible negative consequences. Here, we envisage those consequences as the loss of reputation caused by media coverage of their misdeeds, disseminated to the public. We explore whether media coverage has the potential to push AI creators into the production of safe products, enabling widespread adoption of AI technology. We created artificial populations of self-interested creators and users and studied them through the lens of evolutionary game theory. Our results reveal that media is indeed able to foster cooperation between creators and users, but not always. Cooperation does not evolve if the quality of the information provided by the media is not reliable enough, or if the costs of either accessing media or ensuring safety are too high. By shaping public perception and holding developers accountable, media emerges as a powerful soft regulator -- guiding AI safety even in the absence of formal government oversight. 

**Abstract (ZH)**: 当人工智能产品的开发者在利润与用户安全之间作出选择时，他们倾向于选择利润。不可信赖的人工智能技术必须伴随着实际的负面后果。在此，我们设想这些后果是由于媒体对其不当行为的报道而造成的声誉损失，并传播给公众。我们探讨媒体覆盖是否有可能促使人工智能创造者生产安全产品，从而促进人工智能技术的广泛应用。我们构建了自私的创造者和用户的人工群体，并通过进化博弈理论的角度研究了它们。我们的结果表明，媒体确实能够促进创造者和用户之间的合作，但这并不总能实现。如果媒体提供的信息质量不够可靠，或者访问媒体或确保安全的成本过高，合作就不会演化。通过塑造公众认知并促使开发者承担责任，媒体作为一种强大的软监管工具在缺乏正式政府监管的情况下，仍然能够引导人工智能的安全发展。 

---
# Warming Up for Zeroth-Order Federated Pre-Training with Low Resource Clients 

**Title (ZH)**: 零资源客户端 warming 超预训练的联邦预训练预热 

**Authors**: Gwen Legate, Irina Rish, Eugene Belilovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.03503)  

**Abstract**: Federated learning enables collaborative model training across numerous edge devices without requiring participants to share data; however, memory and communication constraints on these edge devices may preclude their participation in training. We consider a setting in which a subset of edge devices are below a critical memory or communication threshold required to conduct model updates. Under typical federated optimization algorithms, these devices are excluded from training which renders their data inaccessible and increases system induced bias. We are inspired by MeZO, a zeroth-order method used for memory-efficient fine-tuning. The increased variance inherent to zeroth-order gradient approximations has relegated previous zeroth-order optimizers exclusively to the domain of fine tuning; a limitation we seek to correct. We devise a federated, memory-efficient zeroth-order optimizer, ZOWarmUp that permits zeroth-order training from a random initialization. ZOWarmUp leverages differing client capabilities and careful variance reduction techniques to facilitate participation of under-represented, low-resource clients in model training. Like other federated zeroth-order methods, ZOWarmUp eliminates the need for edge devices to transmit their full gradients to the server and instead relies on only a small set of random seeds, rendering the up-link communication cost negligible. We present experiments using various datasets and model architectures to show that ZOWarmUp is a robust algorithm that can can be applied under a wide variety of circumstances. For systems with a high proportion of edge devices that would otherwise be excluded from training, this algorithm provides access to a greater volume and diversity of data, thus improving training outcomes. 

**Abstract (ZH)**: 联邦学习使得多个边缘设备能够在不共享数据的情况下进行协作模型训练；然而，这些边缘设备的内存和通信约束可能使其无法参与训练。我们考虑一种情况下，部分边缘设备低于进行模型更新所需的关键内存或通信阈值。在典型的联邦优化算法中，这些设备被排除在训练之外，这使得它们的数据不可访问并增加了系统引入的偏差。我们受到了MeZO的启发，这是一种用于高效微调的记忆高效零阶方法。零阶梯度近似的固有方差限制了先前的零阶优化器仅限于微调领域；这是一个我们希望纠正的局限。我们设计了一个联邦记忆高效零阶优化器ZOWarmUp，它允许从随机初始化进行零阶训练。ZOWarmUp利用客户端差异能力和精心设计的方差减少技术，促进资源不足的客户端参与模型训练。与其他联邦零阶方法类似，ZOWarmUp消除了边缘设备向服务器传输完整梯度的需求，而是依赖于少量随机种子，使得上行通信成本可以忽略不计。我们使用各种数据集和模型架构进行了实验，展示了ZOWarmUp是一个稳健的算法，可以在多种情境下应用。对于具有大量本应被排除在训练之外的边缘设备的系统，该算法提供了更大的数据量和多样性，从而提升了训练效果。 

---
# DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling 

**Title (ZH)**: DPQuant: 通过动态量化调度实现高效差分隐私模型训练 

**Authors**: Yubo Gao, Renbo Tu, Gennady Pekhimenko, Nandita Vijaykumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.03472)  

**Abstract**: Differentially-Private SGD (DP-SGD) is a powerful technique to protect user privacy when using sensitive data to train neural networks. During training, converting model weights and activations into low-precision formats, i.e., quantization, can drastically reduce training times, energy consumption, and cost, and is thus a widely used technique. In this work, we demonstrate that quantization causes significantly higher accuracy degradation in DP-SGD compared to regular SGD. We observe that this is caused by noise injection in DP-SGD, which amplifies quantization variance, leading to disproportionately large accuracy degradation. To address this challenge, we present QPQuant, a dynamic quantization framework that adaptively selects a changing subset of layers to quantize at each epoch. Our method combines two key ideas that effectively reduce quantization variance: (i) probabilistic sampling of the layers that rotates which layers are quantized every epoch, and (ii) loss-aware layer prioritization, which uses a differentially private loss sensitivity estimator to identify layers that can be quantized with minimal impact on model quality. This estimator consumes a negligible fraction of the overall privacy budget, preserving DP guarantees. Empirical evaluations on ResNet18, ResNet50, and DenseNet121 across a range of datasets demonstrate that DPQuant consistently outperforms static quantization baselines, achieving near Pareto-optimal accuracy-compute trade-offs and up to 2.21x theoretical throughput improvements on low-precision hardware, with less than 2% drop in validation accuracy. 

**Abstract (ZH)**: 差分隐私SGD（DP-SGD）中的量化影响研究：QPQuant动态量化框架 

---
# Multi-level SSL Feature Gating for Audio Deepfake Detection 

**Title (ZH)**: 多级SSL特征门控音频合成换音检测 

**Authors**: Hoan My Tran, Damien Lolive, Aghilas Sini, Arnaud Delhay, Pierre-François Marteau, David Guennec  

**Link**: [PDF](https://arxiv.org/pdf/2509.03409)  

**Abstract**: Recent advancements in generative AI, particularly in speech synthesis, have enabled the generation of highly natural-sounding synthetic speech that closely mimics human voices. While these innovations hold promise for applications like assistive technologies, they also pose significant risks, including misuse for fraudulent activities, identity theft, and security threats. Current research on spoofing detection countermeasures remains limited by generalization to unseen deepfake attacks and languages. To address this, we propose a gating mechanism extracting relevant feature from the speech foundation XLS-R model as a front-end feature extractor. For downstream back-end classifier, we employ Multi-kernel gated Convolution (MultiConv) to capture both local and global speech artifacts. Additionally, we introduce Centered Kernel Alignment (CKA) as a similarity metric to enforce diversity in learned features across different MultiConv layers. By integrating CKA with our gating mechanism, we hypothesize that each component helps improving the learning of distinct synthetic speech patterns. Experimental results demonstrate that our approach achieves state-of-the-art performance on in-domain benchmarks while generalizing robustly to out-of-domain datasets, including multilingual speech samples. This underscores its potential as a versatile solution for detecting evolving speech deepfake threats. 

**Abstract (ZH)**: 近年来，在生成AI，特别是语音合成方面的进展，使生成高自然度合成语音成为可能，这种语音接近真人声音。虽然这些创新在辅助技术等领域充满潜力，但也带来了严重的滥用风险，包括欺诈活动、身份盗窃和安全威胁。当前的伪造检测对策研究主要局限于对未见的深度伪造攻击和语言的一般化。为应对这一挑战，我们提出了一种门控机制，从语音基础XLS-R模型中提取相关特征作为前端特征提取器。对于下游后端分类器，我们采用多核门控卷积（MultiConv）来捕获局部和全局语音特征。此外，我们引入了中心核对齐（CKA）作为相似性度量，以确保不同多核卷积层中学习到的特征具有多样性。通过将CKA与我们的门控机制结合，我们假设每一部分都有助于学习不同的合成语音模式。实验结果显示，我们的方法在领域内基准测试中达到最佳性能，并且在领域外数据集（包括多语言语音样本）上具有稳健的一般化能力。这表明它可能成为一个多功能的伪造检测解决方案，应对不断演变的语音深度伪造威胁。 

---
# Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training 

**Title (ZH)**: 超越正确性：通过RL训练谐调过程和结果奖励 

**Authors**: Chenlu Ye, Zhou Yu, Ziji Zhang, Hao Chen, Narayanan Sadagopan, Jing Huang, Tong Zhang, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.03403)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged to be a predominant paradigm for mathematical reasoning tasks, offering stable improvements in reasoning ability. However, Outcome Reward Models (ORMs) in RLVR are too coarse-grained to distinguish flawed reasoning within correct answers or valid reasoning within incorrect answers. This lack of granularity introduces noisy and misleading gradients significantly and hinders further progress in reasoning process quality. While Process Reward Models (PRMs) offer fine-grained guidance for intermediate steps, they frequently suffer from inaccuracies and are susceptible to reward hacking.
To resolve this dilemma, we introduce PRocess cOnsistency Filter (PROF), an effective data process curation method that harmonizes noisy, fine-grained process rewards with accurate, coarse-grained outcome rewards. Rather than naively blending PRM and ORM in the objective function (arXiv:archive/2506.18896), PROF leverages their complementary strengths through consistency-driven sample selection. Our approach retains correct responses with higher averaged process values and incorrect responses with lower averaged process values, while maintaining positive/negative training sample balance. Extensive experiments demonstrate that our method not only consistently improves the final accuracy over $4\%$ compared to the blending approaches, but also strengthens the quality of intermediate reasoning steps. Codes and training recipes are available at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已在数学推理任务中崭露头角，提供了稳定改进的推理能力。然而，RLVR中的结果奖励模型（ORMs）过于粗粒度，难以区分正确答案中的错误推理或错误答案中的合理推理。这种缺乏粒度的区分引入了噪声和误导性的梯度，阻碍了推理过程质量的进一步提升。虽然过程奖励模型（PRMs）为中间步骤提供了细粒度的指导，但它们经常存在准确性问题且容易受到奖励劫持的影响。

为了解决这一困境，我们引入了过程一致性过滤器（PROF），这是一种有效的数据处理方法，能够将噪声的细粒度过程奖励与准确的粗粒度结果奖励和谐统一。我们的方法通过一致性驱动的样本选择，结合利用PRM和ORM的互补优势，而不是单纯地在目标函数中混融合并（arXiv:archive/2506.18896）。PROF在保持答案正确性和错误性训练样本平衡的同时，提升了中间推理步骤的质量。广泛的实验表明，我们的方法不仅在最终准确率上比融合方法提高了超过4%，还增强了中间推理步骤的质量。相关代码和训练配方可在以下链接获取。 

---
# Neural Field Turing Machine: A Differentiable Spatial Computer 

**Title (ZH)**: 神经场图灵机：一个可微时空计算机 

**Authors**: Akash Malhotra, Nacéra Seghouani  

**Link**: [PDF](https://arxiv.org/pdf/2509.03370)  

**Abstract**: We introduce the Neural Field Turing Machine (NFTM), a differentiable architecture that unifies symbolic computation, physical simulation, and perceptual inference within continuous spatial fields. NFTM combines a neural controller, continuous memory field, and movable read/write heads that perform local updates. At each timestep, the controller reads local patches, computes updates via learned rules, and writes them back while updating head positions. This design achieves linear O(N) scaling through fixed-radius neighborhoods while maintaining Turing completeness under bounded error. We demonstrate three example instantiations of NFTM: cellular automata simulation (Rule 110), physics-informed PDE solvers (2D heat equation), and iterative image refinement (CIFAR-10 inpainting). These instantiations learn local update rules that compose into global dynamics, exhibit stable long-horizon rollouts, and generalize beyond training horizons. NFTM provides a unified computational substrate bridging discrete algorithms and continuous field dynamics within a single differentiable framework. 

**Abstract (ZH)**: neural场图灵机 (NFTM): 统一符号计算、物理仿真和感知推理的可微架构 

---
# Fair Resource Allocation for Fleet Intelligence 

**Title (ZH)**: 公平资源分配以实现车队智能 

**Authors**: Oguzhan Baser, Kaan Kale, Po-han Li, Sandeep Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2509.03353)  

**Abstract**: Resource allocation is crucial for the performance optimization of cloud-assisted multi-agent intelligence. Traditional methods often overlook agents' diverse computational capabilities and complex operating environments, leading to inefficient and unfair resource distribution. To address this, we open-sourced Fair-Synergy, an algorithmic framework that utilizes the concave relationship between the agents' accuracy and the system resources to ensure fair resource allocation across fleet intelligence. We extend traditional allocation approaches to encompass a multidimensional machine learning utility landscape defined by model parameters, training data volume, and task complexity. We evaluate Fair-Synergy with advanced vision and language models such as BERT, VGG16, MobileNet, and ResNets on datasets including MNIST, CIFAR-10, CIFAR-100, BDD, and GLUE. We demonstrate that Fair-Synergy outperforms standard benchmarks by up to 25% in multi-agent inference and 11% in multi-agent learning settings. Also, we explore how the level of fairness affects the least advantaged, most advantaged, and average agents, providing insights for equitable fleet intelligence. 

**Abstract (ZH)**: 云辅助多智能体系统中资源分配对于性能优化至关重要。传统方法往往忽视了智能体多样化的计算能力和复杂的运行环境，导致资源配置效率低下且不公平。为此，我们开源了Fair-Synergy这一算法框架，利用智能体准确率与系统资源之间的凹关系，确保在整个智能舰队中实现公平资源配置。我们扩展了传统的分配方法，将其包含在由模型参数、训练数据量和任务复杂度定义的多维机器学习效用景观中。我们使用包括BERT、VGG16、MobileNet和ResNets在内的高级视觉和语言模型，在MNIST、CIFAR-10、CIFAR-100、BDD和GLUE等数据集上评估了Fair-Synergy。结果显示，与标准基准相比，Fair-Synergy在多智能体推理中最高可提升25%，在多智能体学习环境中可提升11%。我们还探讨了公平性水平对最不利、最有利和平均智能体的影响，为公平智能舰队提供了见解。 

---
# epiGPTope: A machine learning-based epitope generator and classifier 

**Title (ZH)**: epiGPTope：一种基于机器学习的表位生成器和分类器 

**Authors**: Natalia Flechas Manrique, Alberto Martínez, Elena López-Martínez, Luc Andrea, Román Orus, Aitor Manteca, Aitziber L. Cortajarena, Llorenç Espinosa-Portalés  

**Link**: [PDF](https://arxiv.org/pdf/2509.03351)  

**Abstract**: Epitopes are short antigenic peptide sequences which are recognized by antibodies or immune cell receptors. These are central to the development of immunotherapies, vaccines, and diagnostics. However, the rational design of synthetic epitope libraries is challenging due to the large combinatorial sequence space, $20^n$ combinations for linear epitopes of n amino acids, making screening and testing unfeasible, even with high throughput experimental techniques. In this study, we present a large language model, epiGPTope, pre-trained on protein data and specifically fine-tuned on linear epitopes, which for the first time can directly generate novel epitope-like sequences, which are found to possess statistical properties analogous to the ones of known epitopes. This generative approach can be used to prepare libraries of epitope candidate sequences. We further train statistical classifiers to predict whether an epitope sequence is of bacterial or viral origin, thus narrowing the candidate library and increasing the likelihood of identifying specific epitopes. We propose that such combination of generative and predictive models can be of assistance in epitope discovery. The approach uses only primary amino acid sequences of linear epitopes, bypassing the need for a geometric framework or hand-crafted features of the sequences. By developing a method to create biologically feasible sequences, we anticipate faster and more cost-effective generation and screening of synthetic epitopes, with relevant applications in the development of new biotechnologies. 

**Abstract (ZH)**: Epitope生成与预测模型epiGPTope在合成表位库设计中的应用 

---
# On the MIA Vulnerability Gap Between Private GANs and Diffusion Models 

**Title (ZH)**: private GANs与扩散模型之间的MIA漏洞差距 

**Authors**: Ilana Sebag, Jean-Yves Franceschi, Alain Rakotomamonjy, Alexandre Allauzen, Jamal Atif  

**Link**: [PDF](https://arxiv.org/pdf/2509.03341)  

**Abstract**: Generative Adversarial Networks (GANs) and diffusion models have emerged as leading approaches for high-quality image synthesis. While both can be trained under differential privacy (DP) to protect sensitive data, their sensitivity to membership inference attacks (MIAs), a key threat to data confidentiality, remains poorly understood. In this work, we present the first unified theoretical and empirical analysis of the privacy risks faced by differentially private generative models. We begin by showing, through a stability-based analysis, that GANs exhibit fundamentally lower sensitivity to data perturbations than diffusion models, suggesting a structural advantage in resisting MIAs. We then validate this insight with a comprehensive empirical study using a standardized MIA pipeline to evaluate privacy leakage across datasets and privacy budgets. Our results consistently reveal a marked privacy robustness gap in favor of GANs, even in strong DP regimes, highlighting that model type alone can critically shape privacy leakage. 

**Abstract (ZH)**: 生成对抗网络（GANs）和扩散模型已成为高质量图像合成的领先方法。虽然两者都可以在差异隐私（DP）下进行训练以保护敏感数据，但它们在成员推理攻击（MIAs）方面的敏感性，这一数据保密性的重要威胁，仍然了解不足。在本工作中，我们首次对不同差分隐私生成模型面临的隐私风险进行了统一的理论和实证分析。我们首先通过稳定性分析显示，GANs在数据扰动下的敏感性明显低于扩散模型，这表明GANs在抵抗MIAs方面具有结构上的优势。然后，我们通过一个标准化的MIAs管道进行全面的实证研究，评估在不同数据集和隐私预算下的隐私泄露情况。我们的结果一致表明，即使在强DP条件下，GANs也显示出明显的隐私鲁棒性差距，强调了模型类型本身对隐私泄露的严重影响。 

---
# Automatic Differentiation of Agent-Based Models 

**Title (ZH)**: 基于代理的模型的自动微分 

**Authors**: Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu, Doyne Farmer, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2509.03303)  

**Abstract**: Agent-based models (ABMs) simulate complex systems by capturing the bottom-up interactions of individual agents comprising the system. Many complex systems of interest, such as epidemics or financial markets, involve thousands or even millions of agents. Consequently, ABMs often become computationally demanding and rely on the calibration of numerous free parameters, which has significantly hindered their widespread adoption. In this paper, we demonstrate that automatic differentiation (AD) techniques can effectively alleviate these computational burdens. By applying AD to ABMs, the gradients of the simulator become readily available, greatly facilitating essential tasks such as calibration and sensitivity analysis. Specifically, we show how AD enables variational inference (VI) techniques for efficient parameter calibration. Our experiments demonstrate substantial performance improvements and computational savings using VI on three prominent ABMs: Axtell's model of firms; Sugarscape; and the SIR epidemiological model. Our approach thus significantly enhances the practicality and scalability of ABMs for studying complex systems. 

**Abstract (ZH)**: 基于代理模型（ABMs）通过捕捉系统中个体代理的自底向上的交互来模拟复杂系统。许多感兴趣的复杂系统，如流行病或金融市场，涉及数千甚至数百万个代理。因此，ABMs往往变得计算密集，并依赖于大量自由参数的校准，这极大地阻碍了它们的广泛应用。在本文中，我们证明自动微分（AD）技术可以有效减轻这些计算负担。通过将AD应用于ABMs，模拟器的梯度变得易得，极大地促进了诸如校准和灵敏度分析等关键任务的进行。具体而言，我们展示了AD如何使变量推理（VI）技术能够用于高效参数校准。我们的实验表明，在Axtell的企业的模型、Sugarscape和SIR流行病学模型这三个著名ABMs上使用VI方法实现了显著的性能改进和计算节省。因此，我们的方法显著提高了ABMs在研究复杂系统方面的实用性和可扩展性。 

---
# A Comprehensive Guide to Differential Privacy: From Theory to User Expectations 

**Title (ZH)**: 差分隐私全面指南：从理论到用户期望 

**Authors**: Napsu Karmitsa, Antti Airola, Tapio Pahikkala, Tinja Pitkämäki  

**Link**: [PDF](https://arxiv.org/pdf/2509.03294)  

**Abstract**: The increasing availability of personal data has enabled significant advances in fields such as machine learning, healthcare, and cybersecurity. However, this data abundance also raises serious privacy concerns, especially in light of powerful re-identification attacks and growing legal and ethical demands for responsible data use. Differential privacy (DP) has emerged as a principled, mathematically grounded framework for mitigating these risks. This review provides a comprehensive survey of DP, covering its theoretical foundations, practical mechanisms, and real-world applications. It explores key algorithmic tools and domain-specific challenges - particularly in privacy-preserving machine learning and synthetic data generation. The report also highlights usability issues and the need for improved communication and transparency in DP systems. Overall, the goal is to support informed adoption of DP by researchers and practitioners navigating the evolving landscape of data privacy. 

**Abstract (ZH)**: 个人数据的日益可用性促进了机器学习、医疗保健和网络安全等领域的重要进展。然而，这种数据 abundance 也引发了严重的隐私担忧，特别是在强大的重新识别攻击和日益增长的法律和伦理要求背景下。差分隐私（DP）已成为缓解这些风险的一种有原则且数学上坚实的方法论框架。本文综述了差分隐私，涵盖了其理论基础、实用机制及其实际应用。报告探讨了关键的算法工具和特定领域挑战，特别是在隐私保护机器学习和合成数据生成方面的挑战。报告还强调了易用性问题，并指出了需要提高差分隐私系统中的沟通与透明度。总体目标是为研究人员和从业者在不断演变的数据隐私 landscape 中作出知情采纳提供支持。 

---
# Estudio de la eficiencia en la escalabilidad de GPUs para el entrenamiento de Inteligencia Artificial 

**Title (ZH)**: GPU在人工智能训练中可扩展性效率研究 

**Authors**: David Cortes, Carlos Juiz, Belen Bermejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.03263)  

**Abstract**: Training large-scale deep learning models has become a key challenge for the scientific community and industry. While the massive use of GPUs can significantly speed up training times, this approach has a negative impact on efficiency. In this article, we present a detailed analysis of the times reported by MLPerf Training v4.1 on four workloads: BERT, Llama2 LoRA, RetinaNet, and Stable Diffusion, showing that there are configurations that optimise the relationship between performance, GPU usage, and efficiency. The results point to a break-even point that allows training times to be reduced while maximising efficiency. 

**Abstract (ZH)**: 大规模深度学习模型的训练已成为科学界和工业界的key挑战。虽然大量使用GPU可以显著加快训练时间，但这种方法会负面影响效率。在本文中，我们详细分析了MLPerf Training v4.1在四个工作负载（BERT、Llama2 LoRA、RetinaNet、Stable Diffusion）上报告的时间，表明存在优化性能、GPU使用和效率之间关系的配置。结果指出了一个临界点，允许在最大化效率的同时减少训练时间。 

---
# HyPV-LEAD: Proactive Early-Warning of Cryptocurrency Anomalies through Data-Driven Structural-Temporal Modeling 

**Title (ZH)**: HyPV-LEAD: 基于数据驱动结构时序建模的加密货币异常前瞻性早期预警 

**Authors**: Minjung Park, Gyuyeon Na, Soyoun Kim, Sunyoung Moon, HyeonJeong Cha, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.03260)  

**Abstract**: Abnormal cryptocurrency transactions - such as mixing services, fraudulent transfers, and pump-and-dump operations -- pose escalating risks to financial integrity but remain notoriously difficult to detect due to class imbalance, temporal volatility, and complex network dependencies. Existing approaches are predominantly model-centric and post hoc, flagging anomalies only after they occur and thus offering limited preventive value. This paper introduces HyPV-LEAD (Hyperbolic Peak-Valley Lead-time Enabled Anomaly Detection), a data-driven early-warning framework that explicitly incorporates lead time into anomaly detection. Unlike prior methods, HyPV-LEAD integrates three innovations: (1) window-horizon modeling to guarantee actionable lead-time alerts, (2) Peak-Valley (PV) sampling to mitigate class imbalance while preserving temporal continuity, and (3) hyperbolic embedding to capture the hierarchical and scale-free properties of blockchain transaction networks. Empirical evaluation on large-scale Bitcoin transaction data demonstrates that HyPV-LEAD consistently outperforms state-of-the-art baselines, achieving a PR-AUC of 0.9624 with significant gains in precision and recall. Ablation studies further confirm that each component - PV sampling, hyperbolic embedding, and structural-temporal modeling - provides complementary benefits, with the full framework delivering the highest performance. By shifting anomaly detection from reactive classification to proactive early-warning, HyPV-LEAD establishes a robust foundation for real-time risk management, anti-money laundering (AML) compliance, and financial security in dynamic blockchain environments. 

**Abstract (ZH)**: 异常加密货币交易检测：一种引入领先时间的异常检测框架 

---
# Structure Transfer: an Inference-Based Calculus for the Transformation of Representations 

**Title (ZH)**: 结构转移：一种基于推理的表示转换计算规则 

**Authors**: Daniel Raggi, Gem Stapleton, Mateja Jamnik, Aaron Stockdill, Grecia Garcia Garcia, Peter C-H. Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03249)  

**Abstract**: Representation choice is of fundamental importance to our ability to communicate and reason effectively. A major unsolved problem, addressed in this paper, is how to devise \textit{representational-system (RS) agnostic} techniques that drive representation transformation and choice. We present a novel calculus, called \textit{structure transfer}, that enables representation transformation across diverse RSs. Specifically, given a \textit{source} representation drawn from a source RS, the rules of structure transfer allow us to generate a \textit{target} representation for a target RS. The generality of structure transfer comes in part from its ability to ensure that the source representation and the generated target representation satisfy \textit{any} specified relation (such as semantic equivalence). This is done by exploiting \textit{schemas}, which encode knowledge about RSs. Specifically, schemas can express \textit{preservation of information} across relations between any pair of RSs, and this knowledge is used by structure transfer to derive a structure for the target representation which ensures that the desired relation holds. We formalise this using Representational Systems Theory~\cite{raggi2022rst}, building on the key concept of a \textit{construction space}. The abstract nature of construction spaces grants them the generality to model RSs of diverse kinds, including formal languages, geometric figures and diagrams, as well as informal notations. Consequently, structure transfer is a system-agnostic calculus that can be used to identify alternative representations in a wide range of practical settings. 

**Abstract (ZH)**: 代表性的选择对于有效沟通和推理至关重要。本文解决的一个主要未解决问题是如何设计代表系统（RS）无关的技术，以驱动代表转换和选择。我们提出了一种新的计算法则，称为结构转移，它能够在不同的RS之间实现代表转换。具体来说，给定一个源自源RS的源代表，结构转移的规则允许我们生成一个目标RS的目标代表。结构转移的通用性部分来自于它能够确保源代表和生成的目标代表满足任何指定的关系（如语义等价）。这一过程通过利用编码了关于RS知识的规范来实现。具体而言，规范可以表达任何一对RS之间关系下的信息保全，并且这些知识被结构转移所利用，以推导出目标代表的结构，确保所需的关系成立。我们使用Representational Systems Theory（Raggi et al., 2022）对此进行形式化，基于构造空间的关键概念。构造空间的抽象性质赋予了它们广泛的通用性，可以模型化各种类型的RS，包括形式语言、几何图形和图表，以及非正式的符号系统。因此，结构转移是一种系统无关的计算法则，可以广泛应用于多种实际场景中来识别替代表示。 

---
# FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization 

**Title (ZH)**: FoMEMO: 朝着昂贵多目标优化的基石模型方向 

**Authors**: Yiming Yao, Fei Liu, Liang Zhao, Xi Lin, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03244)  

**Abstract**: Expensive multi-objective optimization is a prevalent and crucial concern in many real-world scenarios, where sample-efficiency is vital due to the limited evaluations to recover the true Pareto front for decision making. Existing works either involve rebuilding Gaussian process surrogates from scratch for each objective in each new problem encountered, or rely on extensive past domain experiments for pre-training deep learning models, making them hard to generalize and impractical to cope with various emerging applications in the real world. To address this issue, we propose a new paradigm named FoMEMO (Foundation Models for Expensive Multi-objective Optimization), which enables the establishment of a foundation model conditioned on any domain trajectory and user preference, and facilitates fast in-context optimization based on the predicted preference-wise aggregation posteriors. Rather than accessing extensive domain experiments in the real world, we demonstrate that pre-training the foundation model with a diverse set of hundreds of millions of synthetic data can lead to superior adaptability to unknown problems, without necessitating any subsequent model training or updates in the optimization process. We evaluate our method across a variety of synthetic benchmarks and real-word applications, and demonstrate its superior generality and competitive performance compared to existing methods. 

**Abstract (ZH)**: 昂贵的多目标优化是许多实际场景中普遍而关键的问题，由于评估有限，样本效率成为恢复真实帕累托前沿决策的关键。现有方法要么需要为每个新问题从头重建高斯过程代理模型，要么依赖于大型过去的领域实验进行深度学习模型的预训练，这使得它们难以泛化并在现实世界的各种新兴应用中变得不切实际。为了解决这一问题，我们提出了一种名为FoMEMO（基于基础模型的昂贵多目标优化）的新范式，该范式能够在任何领域轨迹和用户偏好的条件下建立基础模型，并基于预测的偏好感知聚合后验实现快速上下文优化。我们证明，使用数百亿个合成数据进行基础模型的预训练，可以在不需要后续模型训练或优化过程中的更新的情况下，实现对未知问题的优秀适应性。我们跨多种合成基准和实际应用评估了该方法，并展示了其在通用性和竞争性能方面优于现有方法。 

---
# Evaluation of Stress Detection as Time Series Events -- A Novel Window-Based F1-Metric 

**Title (ZH)**: 时间序列事件中压力检测的评估——一种新型窗口基F1度量 

**Authors**: Harald Vilhelm Skat-Rørdam, Sneha Das, Kathrine Sofie Rasmussen, Nicole Nadine Lønfeldt, Line Clemmensen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03240)  

**Abstract**: Accurate evaluation of event detection in time series is essential for applications such as stress monitoring with wearable devices, where ground truth is typically annotated as single-point events, even though the underlying phenomena are gradual and temporally diffused. Standard metrics like F1 and point-adjusted F1 (F1$_{pa}$) often misrepresent model performance in such real-world, imbalanced datasets. We introduce a window-based F1 metric (F1$_w$) that incorporates temporal tolerance, enabling a more robust assessment of event detection when exact alignment is unrealistic. Empirical analysis in three physiological datasets, two in-the-wild (ADARP, Wrist Angel) and one experimental (ROAD), indicates that F1$_w$ reveals meaningful model performance patterns invisible to conventional metrics, while its window size can be adapted to domain knowledge to avoid overestimation. We show that the choice of evaluation metric strongly influences the interpretation of model performance: using predictions from TimesFM, only our temporally tolerant metrics reveal statistically significant improvements over random and null baselines in the two in-the-wild use cases. This work addresses key gaps in time series evaluation and provides practical guidance for healthcare applications where requirements for temporal precision vary by context. 

**Abstract (ZH)**: 时间序列中事件检测的准确评估对于可穿戴设备压力监测等应用至关重要，即使底层现象是逐渐且时间上分布的，ground truth通常被标记为单点事件。标准的评价指标如F1和调整后的点F1（F1$_{pa}$）在现实世界中不平衡的数据集上往往不能真实反映模型性能。我们提出了一个基于窗口的F1指标（F1$_w$），该指标考虑了时间容差，使得在精确对齐不现实的情况下，事件检测的评估更加稳健。在三个生理数据集中的实证分析（两个野外实验集ADARP和Wrist Angel，一个实验集ROAD）表明，F1$_w$揭示了传统指标无法观察到的有意义的模型性能模式，其窗口大小可以根据领域知识进行调整以避免高估。我们证明评价指标的选择对模型性能的解释有重大影响：使用TimesFM的预测，在两个野外实验场景中，只有我们的时间容忍性指标能显著优于随机和零基准。本工作填补了时间序列评估的关键空白，并为不同情况下对时间精确度有不同的要求的健康医疗应用提供了实用指导。 

---
# Rashomon in the Streets: Explanation Ambiguity in Scene Understanding 

**Title (ZH)**: 街上的 Rashomon 效应：场景理解中的解释歧义 

**Authors**: Helge Spieker, Jørn Eirik Betten, Arnaud Gotlieb, Nadjib Lazaar, Nassim Belmecheri  

**Link**: [PDF](https://arxiv.org/pdf/2509.03169)  

**Abstract**: Explainable AI (XAI) is essential for validating and trusting models in safety-critical applications like autonomous driving. However, the reliability of XAI is challenged by the Rashomon effect, where multiple, equally accurate models can offer divergent explanations for the same prediction. This paper provides the first empirical quantification of this effect for the task of action prediction in real-world driving scenes. Using Qualitative Explainable Graphs (QXGs) as a symbolic scene representation, we train Rashomon sets of two distinct model classes: interpretable, pair-based gradient boosting models and complex, graph-based Graph Neural Networks (GNNs). Using feature attribution methods, we measure the agreement of explanations both within and between these classes. Our results reveal significant explanation disagreement. Our findings suggest that explanation ambiguity is an inherent property of the problem, not just a modeling artifact. 

**Abstract (ZH)**: 可解释AI（XAI）在自动驾驶等安全关键应用中验证和信任模型至关重要。然而， Rashomon效应对XAI的可靠性构成了挑战，该效应导致多个同等准确的模型可以获得对于同一预测截然不同的解释。本文首次通过实证量化方法研究了该效应在真实驾驶场景中动作预测任务中的表现。利用定性可解释图（QXGs）作为符号场景表示，我们训练了两类不同的模型集合：可解释的成对梯度提升模型和复杂的图基图神经网络（GNNs）。通过特征归因方法，我们测量了这些类内部和跨类解释的一致性。我们的结果揭示了显著的解释分歧。我们的发现表明，解释模糊性是问题本身的固有属性，而不仅仅是一种建模结果。 

---
# A Neural Network Approach to Multi-radionuclide TDCR Beta Spectroscopy 

**Title (ZH)**: 神经网络方法在多放射性核素TDCR_beta能谱分析中的应用 

**Authors**: Li Yi, Qian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03137)  

**Abstract**: Liquid scintillation triple-to-doubly coincident ratio (TDCR) spectroscopy is widely adopted as a standard method for radionuclide quantification because of its inherent advantages such as high precision, self-calibrating capability, and independence from radioactive reference sources. However, multiradionuclide analysis via TDCR faces the challenges of limited automation and reliance on mixture-specific standards, which may not be easily available. Here, we present an Artificial Intelligence (AI) framework that combines numerical spectral simulation and deep learning for standard-free automated analysis. $\beta$ spectra for model training were generated using Geant4 simulations coupled with statistically modeled detector response sampling. A tailored neural network architecture, trained on this dataset covering various nuclei mix ratio and quenching scenarios, enables autonomous resolution of individual radionuclide activities and detecting efficiency through end-to-end learning paradigms. The model delivers consistent high accuracy across tasks: activity proportions (mean absolute error = 0.009), detection efficiencies (mean absolute error = 0.002), and spectral reconstruction (Structural Similarity Index = 0.9998), validating its physical plausibility for quenched $\beta$ spectroscopy. This AI-driven methodology exhibits significant potential for automated safety-compliant multiradionuclide analysis with robust generalization, real-time processing capabilities, and engineering feasibility, particularly in scenarios where reference materials are unavailable or rapid field analysis is required. 

**Abstract (ZH)**: 基于人工智能的数值谱拟合与深度学习结合的自由标定自动化多放射性核素分析方法 

---
# Information transmission: Inferring change area from change moment in time series remote sensing images 

**Title (ZH)**: 时间序列遥感图像中从变化时刻推断变化区域的信息传递 

**Authors**: Jialu Li, Chen Wu, Meiqi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03112)  

**Abstract**: Time series change detection is a critical task for exploring ecosystem dynamics using time series remote sensing images, because it can simultaneously indicate where and when change occur. While deep learning has shown excellent performance in this domain, it continues to approach change area detection and change moment identification as distinct tasks. Given that change area can be inferred from change moment, we propose a time series change detection network, named CAIM-Net (Change Area Inference from Moment Network), to ensure consistency between change area and change moment results. CAIM-Net infers change area from change moment based on the intrinsic relationship between time series analysis and spatial change detection. The CAIM-Net comprises three key steps: Difference Extraction and Enhancement, Coarse Change Moment Extraction, and Fine Change Moment Extraction and Change Area Inference. In the Difference Extraction and Enhancement, a lightweight encoder with batch dimension stacking is designed to rapidly extract difference features. Subsequently, boundary enhancement convolution is applied to amplify these difference features. In the Coarse Change Moment Extraction, the enhanced difference features from the first step are used to spatiotemporal correlation analysis, and then two distinct methods are employed to determine coarse change moments. In the Fine Change Moment Extraction and Change Area Inference, a multiscale temporal Class Activation Mapping (CAM) module first increases the weight of the change-occurring moment from coarse change moments. Then the weighted change moment is used to infer change area based on the fact that pixels with the change moment must have undergone a change. 

**Abstract (ZH)**: 基于时刻的时序变化检测网络：CAIM-Net 

---
# Efficient Privacy-Preserving Recommendation on Sparse Data using Fully Homomorphic Encryption 

**Title (ZH)**: 基于全同态加密的稀疏数据高效隐私保护推荐 

**Authors**: Moontaha Nishat Chowdhury, André Bauer, Minxuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.03024)  

**Abstract**: In today's data-driven world, recommendation systems personalize user experiences across industries but rely on sensitive data, raising privacy concerns. Fully homomorphic encryption (FHE) can secure these systems, but a significant challenge in applying FHE to recommendation systems is efficiently handling the inherently large and sparse user-item rating matrices. FHE operations are computationally intensive, and naively processing various sparse matrices in recommendation systems would be prohibitively expensive. Additionally, the communication overhead between parties remains a critical concern in encrypted domains. We propose a novel approach combining Compressed Sparse Row (CSR) representation with FHE-based matrix factorization that efficiently handles matrix sparsity in the encrypted domain while minimizing communication costs. Our experimental results demonstrate high recommendation accuracy with encrypted data while achieving the lowest communication costs, effectively preserving user privacy. 

**Abstract (ZH)**: 在数据驱动的世界中，推荐系统跨行业个性化用户体验，但依赖敏感数据，引发隐私关切。全同态加密(FHE)可以确保这些系统的安全性，但在推荐系统中应用FHE的一个主要挑战是高效处理固有的大型和稀疏的用户-项评分矩阵。FHE操作计算密集，如果未经优化地处理推荐系统中的各种稀疏矩阵，将变得极其昂贵。此外，加密域中的通信开销仍然是一个关键问题。我们提出了一种新颖的方法，结合压缩稀疏行(CSR)表示与基于FHE的矩阵分解，以在加密域中高效处理矩阵稀疏性并最小化通信成本。实验结果表明，即使在加密数据下也能实现高推荐准确率，同时实现最低的通信成本，从而有效保护用户隐私。 

---
# StableSleep: Source-Free Test-Time Adaptation for Sleep Staging with Lightweight Safety Rails 

**Title (ZH)**: StableSleep：基于轻量级安全引导的无源测试时自适应睡眠分期方法 

**Authors**: Hritik Arasu, Faisal R Jahangiri  

**Link**: [PDF](https://arxiv.org/pdf/2509.02982)  

**Abstract**: Sleep staging models often degrade when deployed on patients with unseen physiology or recording conditions. We propose a streaming, source-free test-time adaptation (TTA) recipe that combines entropy minimization (Tent) with Batch-Norm statistic refresh and two safety rails: an entropy gate to pause adaptation on uncertain windows and an EMA-based reset to reel back drift. On Sleep-EDF Expanded, using single-lead EEG (Fpz-Cz, 100 Hz, 30s epochs; R&K to AASM mapping), we show consistent gains over a frozen baseline at seconds-level latency and minimal memory, reporting per-stage metrics and Cohen's k. The method is model-agnostic, requires no source data or patient calibration, and is practical for on-device or bedside use. 

**Abstract (ZH)**: 睡眠阶段模型在未见过的生理状态或记录条件下部署时往往会退化。我们提出了一种基于熵最小化（Tent）结合Batch-Norm统计更新，并伴有两种安全机制的在线测试时自适应（TTA）方法：熵门限以暂停不确定窗口上的自适应过程，以及基于EMA的重置以回退漂移。在Sleep-EDF Expanded数据集上，使用单导联EEG（Fpz-Cz，100 Hz，30秒 epoch；R&K到AASM映射），我们显示该方法在毫秒级延迟和minimal内存下相对于冻结基线的一致性改进，并报告了阶段内指标和Cohen's k。该方法对任何模型都是通用的，无需源数据或患者校准，并适用于设备端或床边使用。 

---
# AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov-Arnold Network for Time Series Forecasting 

**Title (ZH)**: AR-KAN：自回归权增强的柯尔莫哥洛夫-阿诺尔德网络时间序列预测 

**Authors**: Chen Zeng, Tiehang Xu, Qiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02967)  

**Abstract**: Conventional neural networks frequently face challenges in spectral analysis of signals. To address this challenge, Fourier neural networks (FNNs) and similar approaches integrate components of Fourier series into the structure of neural networks. Nonetheless, a significant hurdle is often overlooked: the superposition of periodic signals does not necessarily result in a periodic signal. For example, when forecasting almost periodic functions composed of signals with incommensurate frequencies, traditional models such as Autoregressive Integrated Moving Average (ARIMA) frequently outperform most neural networks including large language models (LLMs). To tackle this goal, we propose Autoregressive-Weight-Enhanced AR-KAN, a hybrid model that combines the benefits of both methods. Using the Universal Myopic Mapping Theorem, we apply a Kolmogorov-Arnold Network (KAN) for the static nonlinear part and include memory through a pre-trained AR component, which can be explained to retain the most useful information while eliminating redundancy. Experimental data indicates that AR-KAN delivers superior results on $72\%$ of real-world datasets. 

**Abstract (ZH)**: 传统的神经网络在信号频谱分析中经常面临挑战。为了应对这一挑战，Fourier神经网络（FNNs）和其他类似方法将Fourier级数的成分整合到神经网络结构中。然而，一个重要的障碍经常被忽视：周期信号的叠加并不一定产生周期信号。例如，在预测由非通约频率信号组成的几乎周期函数时，传统的模型如自回归整定移动平均模型（ARIMA）通常能比大多数神经网络，包括大型语言模型（LLMs），表现得更好。为了应对这一目标，我们提出了一种混合模型Autoregressive-Weight-Enhanced AR-KAN，该模型结合了两种方法的优点。利用宇宙近视映射定理，我们使用Kolmogorov-Arnold网络（KAN）处理静态非线性部分，并通过预训练的AR组件引入记忆，可以解释为保留最有用的信息并消除冗余。实验数据表明，AR-KAN在72%的实际数据集中表现更优。 

---
# Lattice Annotated Temporal (LAT) Logic for Non-Markovian Reasoning 

**Title (ZH)**: 非马尔可夫推理的格注释时序（LAT）逻辑 

**Authors**: Kaustuv Mukherji, Jaikrishna Manojkumar Patil, Dyuman Aditya, Paulo Shakarian, Devendra Parkar, Lahari Pokala, Clark Dorman, Gerardo I. Simari  

**Link**: [PDF](https://arxiv.org/pdf/2509.02958)  

**Abstract**: We introduce Lattice Annotated Temporal (LAT) Logic, an extension of Generalized Annotated Logic Programs (GAPs) that incorporates temporal reasoning and supports open-world semantics through the use of a lower lattice structure. This logic combines an efficient deduction process with temporal logic programming to support non-Markovian relationships and open-world reasoning capabilities. The open-world aspect, a by-product of the use of the lower-lattice annotation structure, allows for efficient grounding through a Skolemization process, even in domains with infinite or highly diverse constants.
We provide a suite of theoretical results that bound the computational complexity of the grounding process, in addition to showing that many of the results on GAPs (using an upper lattice) still hold with the lower lattice and temporal extensions (though different proof techniques are required). Our open-source implementation, PyReason, features modular design, machine-level optimizations, and direct integration with reinforcement learning environments. Empirical evaluations across multi-agent simulations and knowledge graph tasks demonstrate up to three orders of magnitude speedup and up to five orders of magnitude memory reduction while maintaining or improving task performance. Additionally, we evaluate LAT Logic's value in reinforcement learning environments as a non-Markovian simulator, achieving up to three orders of magnitude faster simulation with improved agent performance, including a 26% increase in win rate due to capturing richer temporal dependencies. These results highlight LAT Logic's potential as a unified, extensible framework for open-world temporal reasoning in dynamic and uncertain environments. Our implementation is available at: this http URL. 

**Abstract (ZH)**: 晶格注释时序逻辑：一种结合时序推理和支持开放式语义的广义注释逻辑程序扩展 

---
# Simulacra Naturae: Generative Ecosystem driven by Agent-Based Simulations and Brain Organoid Collective Intelligence 

**Title (ZH)**: 自然 simulacra：基于基于代理的模拟和脑类器官集体智能的生成生态系统 

**Authors**: Nefeli Manoudaki, Mert Toka, Iason Paterakis, Diarmid Flatley  

**Link**: [PDF](https://arxiv.org/pdf/2509.02924)  

**Abstract**: Simulacra Naturae is a data-driven media installation that explores collective care through the entanglement of biological computation, material ecologies, and generative systems. The work translates pre-recorded neural activity from brain organoids, lab-grown three-dimensional clusters of neurons, into a multi-sensory environment composed of generative visuals, spatial audio, living plants, and fabricated clay artifacts. These biosignals, streamed through a real-time system, modulate emergent agent behaviors inspired by natural systems such as termite colonies and slime molds. Rather than using biosignals as direct control inputs, Simulacra Naturae treats organoid activity as a co-creative force, allowing neural rhythms to guide the growth, form, and atmosphere of a generative ecosystem. The installation features computationally fabricated clay prints embedded with solenoids, adding physical sound resonances to the generative surround composition. The spatial environment, filled with live tropical plants and a floor-level projection layer featuring real-time generative AI visuals, invites participants into a sensory field shaped by nonhuman cognition. By grounding abstract data in living materials and embodied experience, Simulacra Naturae reimagines visualization as a practice of care, one that decentralizes human agency and opens new spaces for ethics, empathy, and ecological attunement within hybrid computational systems. 

**Abstract (ZH)**: Simulacra Naturae是一部数据驱动的媒体装置，通过生物计算、物质生态和生成系统交织探索集体关怀。作品将从脑类器官中预录的神经活动转化为由生成性视觉、空间音频、活植物和手工 clay 工艺品组成的多感官环境。这些生物信号通过实时系统流式传输，驱动受到自然系统如白蚁 colony 和黏菌启发的自动生成代理行为。Simulacra Naturae 不将生物信号作为直接控制输入，而是将脑类器官活动视为一种共创造性力量，允许神经节律引导生成生态系统中生物体的生长、形态和氛围。装置中嵌有 solenoid 的计算生成 clay 模具增加了生成性环绕音效的物理共振。充满活热带植物的空间环境和地面上层的实时生成 AI 视觉投影，邀请参与者进入由非人类认知塑造的感官场域。通过将抽象数据根植于活材料和体验性经验中，Simulacra Naturae 重新构想可视化作为一种关怀实践，分散人类Agency，为混合计算系统中的伦理、共鸣和生态调谐开辟新空间。 

---
# Single Domain Generalization in Diabetic Retinopathy: A Neuro-Symbolic Learning Approach 

**Title (ZH)**: 糖尿病视网膜病变单域泛化：一种神经符号学习方法 

**Authors**: Midhat Urooj, Ayan Banerjee, Farhat Shaikh, Kuntal Thakur, Sandeep Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.02918)  

**Abstract**: Domain generalization remains a critical challenge in medical imaging, where models trained on single sources often fail under real-world distribution shifts. We propose KG-DG, a neuro-symbolic framework for diabetic retinopathy (DR) classification that integrates vision transformers with expert-guided symbolic reasoning to enable robust generalization across unseen domains. Our approach leverages clinical lesion ontologies through structured, rule-based features and retinal vessel segmentation, fusing them with deep visual representations via a confidence-weighted integration strategy. The framework addresses both single-domain generalization (SDG) and multi-domain generalization (MDG) by minimizing the KL divergence between domain embeddings, thereby enforcing alignment of high-level clinical semantics. Extensive experiments across four public datasets (APTOS, EyePACS, Messidor-1, Messidor-2) demonstrate significant improvements: up to a 5.2% accuracy gain in cross-domain settings and a 6% improvement over baseline ViT models. Notably, our symbolic-only model achieves a 63.67% average accuracy in MDG, while the complete neuro-symbolic integration achieves the highest accuracy compared to existing published baselines and benchmarks in challenging SDG scenarios. Ablation studies reveal that lesion-based features (84.65% accuracy) substantially outperform purely neural approaches, confirming that symbolic components act as effective regularizers beyond merely enhancing interpretability. Our findings establish neuro-symbolic integration as a promising paradigm for building clinically robust, and domain-invariant medical AI systems. 

**Abstract (ZH)**: 神经符号框架KG-DG在糖尿病视网膜病变分类中的跨域泛化研究 

---
# A-SEA3L-QA: A Fully Automated Self-Evolving, Adversarial Workflow for Arabic Long-Context Question-Answer Generation 

**Title (ZH)**: A-SEA3L-QA：一种全自动自演化、对抗性 workflows 的阿拉伯语长上下文问答生成方法 

**Authors**: Kesen Wang, Daulet Toibazar, Pedro J. Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2509.02864)  

**Abstract**: We present an end-to-end, self-evolving adversarial workflow for long-context Question-Answer (QA) Generation in Arabic. By orchestrating multiple specialized LVLMs: a question generator, an evaluator, and a swarm of answer generators, our system iteratively refines its own performance without any human intervention. Starting from raw, multi-page Arabic documents across diverse domains, the question generator produces fine-grained, context-aware queries to be tackled by the answer generator swarm, and the evaluator assesses and feeds back quality metrics. This closed-loop cycle enables continuous learning: low-confidence outputs trigger automated re-generation and model updates, progressively enhancing question difficulty and relevance. Moreover, we set the quality metrics as a tunable hyperparameter, enabling question generation at controllable and customizable difficulty levels. We release AraLongBench, a large-scale Arabic benchmark of single- and multi-page challenges spanning hundreds of pages, and demonstrate that our self-evolving workflow substantially outperform static pipelines, markedly boosting the long-context comprehension capabilities of leading Arabic Large Vision Language Models (LVLMs). Lastly, we also meticulously architect a fully automated agentic workflow for long-context Arabic document collection. 

**Abstract (ZH)**: 我们提出了一种端到端的自进化对抗工作流，用于阿拉伯语长上下文问答生成。该系统通过协调多个专门的LVLM：问题生成器、评估器和一群答案生成器，迭代地提升自身的性能，无需任何人工干预。从来自不同领域的多页阿拉伯文档开始，问题生成器生成细粒度、上下文感知的问题供答案生成器群组处理，评估器评估并反馈质量指标。这种闭环循环实现了持续学习：低置信度输出触发自动重新生成和模型更新，逐步提升问题的难度和相关性。此外，我们将质量指标设置为可调节的超参数，使问题生成能够在可控和可定制的难度下进行。我们发布了AraLongBench，这是一个包含单页和多页挑战的大规模阿拉伯基准，跨越数百页，证明了我们的自进化工作流显著优于静态管道，极大地提升了领先阿拉伯大型视觉语言模型（LVLM）的长上下文理解能力。最后，我们还精心设计了一种完全自动的代理工作流，用于长上下文阿拉伯文档集合的收集。 

---
# Enhancing Machine Learning for Imbalanced Medical Data: A Quantum-Inspired Approach to Synthetic Oversampling (QI-SMOTE) 

**Title (ZH)**: 基于量子启发的合成过抽样（QI-SMOTE）方法：提升不平衡医疗数据的机器学习算法 

**Authors**: Vikas Kashtriya, Pardeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.02863)  

**Abstract**: Class imbalance remains a critical challenge in machine learning (ML), particularly in the medical domain, where underrepresented minority classes lead to biased models and reduced predictive performance. This study introduces Quantum-Inspired SMOTE (QI-SMOTE), a novel data augmentation technique that enhances the performance of ML classifiers, including Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), k-Nearest Neighbors (KNN), Gradient Boosting (GB), and Neural Networks, by leveraging quantum principles such as quantum evolution and layered entanglement. Unlike conventional oversampling methods, QI-SMOTE generates synthetic instances that preserve complex data structures, improving model generalization and classification accuracy. We validate QI-SMOTE on the MIMIC-III and MIMIC-IV datasets, using mortality detection as a benchmark task due to their clinical significance and inherent class imbalance. We compare our method against traditional oversampling techniques, including Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-TOMEK, and SVM-SMOTE, using key performance metrics such as Accuracy, F1-score, G-Mean, and AUC-ROC. The results demonstrate that QI-SMOTE significantly improves the effectiveness of ensemble methods (RF, GB, ADA), kernel-based models (SVM), and deep learning approaches by producing more informative and balanced training data. By integrating quantum-inspired transformations into the ML pipeline, QI-SMOTE not only mitigates class imbalance but also enhances the robustness and reliability of predictive models in medical diagnostics and decision-making. This study highlights the potential of quantum-inspired resampling techniques in advancing state-of-the-art ML methodologies. 

**Abstract (ZH)**: 量子启发式SMOTE (QI-SMOTE)在机器学习中的应用：缓解医疗领域类别不平衡问题 

---
# The Architecture of AI Transformation: Four Strategic Patterns and an Emerging Frontier 

**Title (ZH)**: 人工智能转型的架构：四种战略模式与新兴前沿 

**Authors**: Diana A. Wolfe, Alice Choe, Fergus Kidd  

**Link**: [PDF](https://arxiv.org/pdf/2509.02853)  

**Abstract**: Despite extensive investment in artificial intelligence, 95% of enterprises report no measurable profit impact from AI deployments (MIT, 2025). We argue that this gap reflects paradigmatic lock-in that channels AI into incremental optimization rather than structural transformation. Using a cross-case analysis, we propose a 2x2 framework that reconceptualizes AI strategy along two independent dimensions: the degree of transformation achieved (incremental to transformational) and the treatment of human contribution (reduced to amplified). The framework surfaces four patterns now dominant in practice: individual augmentation, process automation, workforce substitution, and a less deployed frontier of collaborative intelligence. Evidence shows that the first three reinforce legacy work models and yield localized gains without durable value capture. Realizing collaborative intelligence requires three mechanisms: complementarity (pairing distinct human and machine strengths), co-evolution (mutual adaptation through interaction), and boundary-setting (human determination of ethical and strategic parameters). Complementarity and boundary-setting are observable in regulated and high-stakes domains; co-evolution is largely absent, which helps explain limited system-level impact. A case study analysis illustrates that advancing toward collaborative intelligence requires material restructuring of roles, governance, and data architecture rather than additional tools. The framework reframes AI transformation as an organizational design challenge: moving from optimizing the division of labor between humans and machines to architecting their convergence, with implications for operating models, workforce development, and the future of work. 

**Abstract (ZH)**: 尽管在人工智能方面进行了大量投资，但95%的企业报告称其AI部署未产生可量化的利润影响（麻省理工学院，2025年）。我们认为这一差距反映了范式锁定，使得AI局限于增量优化而非结构性变革。通过跨案例分析，我们提出了一种2x2框架，重新构想了AI战略的两个独立维度：实现的变革程度（从增量到结构性变革）以及对人类贡献的处理方式（从减少到放大）。该框架揭示了目前实践中占主导地位的四种模式：个体增强、流程自动化、劳动力替代以及尚未广泛应用的合作智能前沿。证据表明，前三种模式强化了传统工作模式，带来了局部收益但缺乏持久的价值捕获。实现合作智能需要三种机制：互补性（配对人类与机器的独特优势）、共生演化（通过交互实现相互适应）以及边界设定（人类确定伦理和战略参数）。互补性和边界设定在受监管和高风险领域中可见；而共生演化几乎不存在，这也解释了系统层面影响有限的原因。案例研究分析表明，向合作智能迈进需要对角色、治理和数据架构进行实质性重构，而不仅仅是提供额外的工具。该框架将AI变革重述为组织设计挑战：从优化人类与机器之间的劳动分工转向设计其融合，这具有对运营模式、劳动力发展和未来工作的影响。 

---
# Conformal Prediction for Time-series Forecasting with Change Points 

**Title (ZH)**: 变点时间序列预测的齐性预测方法 

**Authors**: Sophia Sun, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02844)  

**Abstract**: Conformal prediction has been explored as a general and efficient way to provide uncertainty quantification for time series. However, current methods struggle to handle time series data with change points - sudden shifts in the underlying data-generating process. In this paper, we propose a novel Conformal Prediction for Time-series with Change points (CPTC) algorithm, addressing this gap by integrating a model to predict the underlying state with online conformal prediction to model uncertainties in non-stationary time series. We prove CPTC's validity and improved adaptivity in the time series setting under minimum assumptions, and demonstrate CPTC's practical effectiveness on 6 synthetic and real-world datasets, showing improved validity and adaptivity compared to state-of-the-art baselines. 

**Abstract (ZH)**: 基于变化点的时序自适应一致性预测（CPTC） 

---
# HF-RAG: Hierarchical Fusion-based RAG with Multiple Sources and Rankers 

**Title (ZH)**: 多源与排序器基于层级融合的RAG（检索增强生成） 

**Authors**: Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2509.02837)  

**Abstract**: Leveraging both labeled (input-output associations) and unlabeled data (wider contextual grounding) may provide complementary benefits in retrieval augmented generation (RAG). However, effectively combining evidence from these heterogeneous sources is challenging as the respective similarity scores are not inter-comparable. Additionally, aggregating beliefs from the outputs of multiple rankers can improve the effectiveness of RAG. Our proposed method first aggregates the top-documents from a number of IR models using a standard rank fusion technique for each source (labeled and unlabeled). Next, we standardize the retrieval score distributions within each source by applying z-score transformation before merging the top-retrieved documents from the two sources. We evaluate our approach on the fact verification task, demonstrating that it consistently improves over the best-performing individual ranker or source and also shows better out-of-domain generalization. 

**Abstract (ZH)**: 利用标记数据（输入-输出关联）和未标记数据（更广泛的情境关联）相结合的方式可能在检索增强生成（RAG）中提供互补益处。然而，有效地结合这些异构来源的证据具有挑战性，因为各自的相似度评分无法相互比较。此外，从多个排名器的输出中聚合信念可以提高RAG的有效性。我们提出的方法首先使用标准的排名融合技术从每种来源（标记和未标记）聚合顶级文档。接下来，我们通过应用z-score标准化在每种来源内部的检索分数分布，然后将两种来源中检索到的顶级文档合并。我们在事实核查任务上评估了该方法，结果显示它始终优于表现最好的单一排名器或来源，并且在跨域泛化方面表现更好。 

---
# Ensemble Learning for Healthcare: A Comparative Analysis of Hybrid Voting and Ensemble Stacking in Obesity Risk Prediction 

**Title (ZH)**: 健康管理中的集成学习：肥胖风险预测中混合投票和集成堆叠的比较分析 

**Authors**: Towhidul Islam, Md Sumon Ali  

**Link**: [PDF](https://arxiv.org/pdf/2509.02826)  

**Abstract**: Obesity is a critical global health issue driven by dietary, physiological, and environmental factors, and is strongly associated with chronic diseases such as diabetes, cardiovascular disorders, and cancer. Machine learning has emerged as a promising approach for early obesity risk prediction, yet a comparative evaluation of ensemble techniques -- particularly hybrid majority voting and ensemble stacking -- remains limited. This study aims to compare hybrid majority voting and ensemble stacking methods for obesity risk prediction, identifying which approach delivers higher accuracy and efficiency. The analysis seeks to highlight the complementary strengths of these ensemble techniques in guiding better predictive model selection for healthcare applications. Two datasets were utilized to evaluate three ensemble models: Majority Hard Voting, Weighted Hard Voting, and Stacking (with a Multi-Layer Perceptron as meta-classifier). A pool of nine Machine Learning (ML) algorithms, evaluated across a total of 50 hyperparameter configurations, was analyzed to identify the top three models to serve as base learners for the ensemble methods. Preprocessing steps involved dataset balancing, and outlier detection, and model performance was evaluated using Accuracy and F1-Score. On Dataset-1, weighted hard voting and stacking achieved nearly identical performance (Accuracy: 0.920304, F1: 0.920070), outperforming majority hard voting. On Dataset-2, stacking demonstrated superior results (Accuracy: 0.989837, F1: 0.989825) compared to majority hard voting (Accuracy: 0.981707, F1: 0.981675) and weighted hard voting, which showed the lowest performance. The findings confirm that ensemble stacking provides stronger predictive capability, particularly for complex data distributions, while hybrid majority voting remains a robust alternative. 

**Abstract (ZH)**: 肥胖是由饮食、生理和环境因素驱动的关键全球健康问题，并与糖尿病、心血管疾病和癌症等慢性疾病密切相关。机器学习已成为早期肥胖风险预测的有希望的方法，但关于集成技术（尤其是混合多数投票和集成堆叠）的比较评估仍然有限。本研究旨在比较混合多数投票和集成堆叠方法在肥胖风险预测中的效果，确定哪种方法能提供更高的准确性和效率。分析旨在突出这些集成技术的互补优势，以指导更有效的预测模型选择，适用于医疗健康应用。本研究使用了两个数据集来评估三种集成模型：多数硬投票、加权硬投票和堆叠（使用多层感知器作为元分类器）。分析了九种机器学习算法共计50种超参数配置，以确定最佳的基学习器。预处理步骤包括数据集平衡和异常值检测，模型性能通过准确率和F1分数进行评估。在Dataset-1上，加权硬投票和堆叠几乎取得了相同的性能（准确率：0.920304，F1分数：0.920070），优于多数硬投票。在Dataset-2上，堆叠方法的性能优于多数硬投票和加权硬投票，后者表现出最低的性能（准确率：0.981707，F1分数：0.981675，准确率：0.989837，F1分数：0.989825）。研究结果表明，集成堆叠在复杂数据分布情况下提供了更强的预测能力，而混合多数投票仍然是一个稳健的替代方案。 

---
# DrDiff: Dynamic Routing Diffusion with Hierarchical Attention for Breaking the Efficiency-Quality Trade-off 

**Title (ZH)**: DrDiff: 动态路由扩散与层次注意力机制以打破效率与质量的trade-off 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Zimeng Huang, Xiaofei Sun, Jian Wang, Chengpei Tang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02785)  

**Abstract**: This paper introduces DrDiff, a novel framework for long-text generation that overcomes the efficiency-quality trade-off through three core technologies. First, we design a dynamic expert scheduling mechanism that intelligently allocates computational resources during the diffusion process based on text complexity, enabling more efficient handling of text generation tasks of varying difficulty. Second, we introduce a Hierarchical Sparse Attention (HSA) mechanism that adaptively adjusts attention patterns according to a variety of input lengths, reducing computational complexity from O($n^2$) to O($n$) while maintaining model performance. Finally, we propose a soft absorption guidance optimization strategy that combines with DPM-solver++ to reduce diffusion steps, significantly improving generation speed. Comprehensive experiments on various long-text generation benchmarks demonstrate the superiority of our DrDiff over the existing SOTA methods. 

**Abstract (ZH)**: 本文介绍了DrDiff，这是一种通过三种核心技术突破效率与质量权衡的新颖长文本生成框架。首先，我们设计了一种动态专家调度机制，在扩散过程中根据文本复杂度智能分配计算资源，实现对不同难度文本生成任务的更高效处理。其次，我们引入了一种层次稀疏注意（HSA）机制，根据不同长度的输入自适应调整注意模式，将计算复杂度从O($n^2$)降低到O($n$)，同时保持模型性能。最后，我们提出了与DPM-solver++结合的软吸收指导优化策略，显著提高生成速度。一系列综合实验表明，DrDiff在各种长文本生成基准测试中优于现有最优方法。 

---
# The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface 

**Title (ZH)**: 透明地球：地球地下空间的多模态基础模型 

**Authors**: Arnab Mazumder, Javier E. Santos, Noah Hobbs, Mohamed Mehana, Daniel O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2509.02783)  

**Abstract**: We present the Transparent Earth, a transformer-based architecture for reconstructing subsurface properties from heterogeneous datasets that vary in sparsity, resolution, and modality, where each modality represents a distinct type of observation (e.g., stress angle, mantle temperature, tectonic plate type). The model incorporates positional encodings of observations together with modality encodings, derived from a text embedding model applied to a description of each modality. This design enables the model to scale to an arbitrary number of modalities, making it straightforward to add new ones not considered in the initial design. We currently include eight modalities spanning directional angles, categorical classes, and continuous properties such as temperature and thickness. These capabilities support in-context learning, enabling the model to generate predictions either with no inputs or with an arbitrary number of additional observations from any subset of modalities. On validation data, this reduces errors in predicting stress angle by more than a factor of three. The proposed architecture is scalable and demonstrates improved performance with increased parameters. Together, these advances make the Transparent Earth an initial foundation model for the Earth's subsurface that ultimately aims to predict any subsurface property anywhere on Earth. 

**Abstract (ZH)**: 透明地球：基于变换器的 architecture 用于从异构数据集中重建地下属性 

---
# Optimizing Geometry Problem Sets for Skill Development 

**Title (ZH)**: 优化几何问题集以促进技能发展 

**Authors**: Michael Bouzinier, Sergey Trifonov  

**Link**: [PDF](https://arxiv.org/pdf/2509.02758)  

**Abstract**: This article describes an ontology and methodology for annotating and organizing Euclidean Geometry problems, developed in the early 1990s and implemented as a software tool. While the majority of this work -- including the ontology and solution graph paradigm -- was completed over thirty years ago, we argue that it has renewed relevance in the context of modern artificial intelligence. In particular, we explore the hypothesis that this established framework can facilitate automated solution validation and feedback when paired with contemporary large language models, thereby supporting teachers and self-learners in geometry education. We document the original architecture and its enduring value, and outline pathways for bridging historical educational resources with next-generation AI techniques. 

**Abstract (ZH)**: 本文描述了在20世纪90年代初开发的一种本体论和方法论，用于标注和组织欧几里得几何问题，并以软件工具的形式实现。尽管这项工作的大部分——包括本体论和解决方案图范式——已完成逾三十年，我们认为它在现代人工智能的背景下具有新的相关性。特别是，我们探讨了这种现有框架与当代大型语言模型结合时，如何促进自动解题验证和反馈，从而支持几何教育中的教师和自我学习者。我们记录了原始架构及其持久价值，并概述了将历史教育资源与下一代AI技术相连接的途径。 

---
# Mentality: A Mamba-based Approach towards Foundation Models for EEG 

**Title (ZH)**: 心态：基于Mamba的面向脑电图基础模型的方法 

**Authors**: Saarang Panchavati, Corey Arnold, William Speier  

**Link**: [PDF](https://arxiv.org/pdf/2509.02746)  

**Abstract**: This work explores the potential of foundation models, specifically a Mamba-based selective state space model, for enhancing EEG analysis in neurological disorder diagnosis. EEG, crucial for diagnosing conditions like epilepsy, presents significant challenges due to its noisy, high-dimensional, and nonlinear nature. Traditional machine learning methods have made advances in automating EEG analysis but often fail to capture its complex spatio-temporal dynamics. Recent advances in deep learning, particularly in sequence modeling, offer new avenues for creating more generalized and expressive models capable of handling such complexities. By training a Mamba-based model on a large dataset containing seizure and non-seizure EEG recordings through a self-supervised reconstruction task followed by a seizure detection task, we demonstrate the model's effectiveness, achieving an AUROC of 0.72 on a held-out test set. This approach marks a significant step toward developing large-scale, clinically applicable foundation models for EEG data analysis. 

**Abstract (ZH)**: 基于Mamba的可选状态空间模型在神经障碍诊断中增强EEG分析的潜力 

---
# BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation 

**Title (ZH)**: BioMD：生物分子动力学模拟的原子级生成模型 

**Authors**: Bin Feng, Jiying Zhang, Xinni Zhang, Zijing Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02642)  

**Abstract**: Molecular dynamics (MD) simulations are essential tools in computational chemistry and drug discovery, offering crucial insights into dynamic molecular behavior. However, their utility is significantly limited by substantial computational costs, which severely restrict accessible timescales for many biologically relevant processes. Despite the encouraging performance of existing machine learning (ML) methods, they struggle to generate extended biomolecular system trajectories, primarily due to the lack of MD datasets and the large computational demands of modeling long historical trajectories. Here, we introduce BioMD, the first all-atom generative model to simulate long-timescale protein-ligand dynamics using a hierarchical framework of forecasting and interpolation. We demonstrate the effectiveness and versatility of BioMD on the DD-13M (ligand unbinding) and MISATO datasets. For both datasets, BioMD generates highly realistic conformations, showing high physical plausibility and low reconstruction errors. Besides, BioMD successfully generates ligand unbinding paths for 97.1% of the protein-ligand systems within ten attempts, demonstrating its ability to explore critical unbinding pathways. Collectively, these results establish BioMD as a tool for simulating complex biomolecular processes, offering broad applicability for computational chemistry and drug discovery. 

**Abstract (ZH)**: 分子动力学（MD）模拟是计算化学和药物发现中的关键工具，提供了对动态分子行为的宝贵见解。然而，它们的应用受到巨大计算成本的限制，严重限制了许多生物相关过程的可访问时间尺度。尽管现有的机器学习（ML）方法表现出色，但在生成扩展的生物分子系统轨迹方面仍面临挑战，主要原因是缺乏MD数据集和模拟长期历史轨迹的高计算需求。在这里，我们介绍BioMD，这是首个使用分级预测和内插框架来模拟蛋白质-配体长时间尺度动力学的原子级生成模型。我们在DD-13M（配体解离）和MISATO数据集上展示了BioMD的有效性和灵活性。对于两个数据集，BioMD生成了高度真实的构象，显示了高的物理可信度和低的重构误差。此外，BioMD在十次尝试内成功生成了97.1%的蛋白质-配体系统中的配体解离路径，证明了其探索关键解离路径的能力。综上所述，这些结果确立了BioMD作为模拟复杂生物分子过程的工具，具有广泛的计算化学和药物发现应用前景。 

---
# Adaptive Learning Strategies for Mitotic Figure Classification in MIDOG2025 Challenge 

**Title (ZH)**: MIDOG2025挑战中有丝分裂图分类的自适应学习策略 

**Authors**: Biwen Meng, Xi Long, Jingxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02640)  

**Abstract**: Atypical mitotic figures (AMFs) are clinically relevant indicators of abnormal cell division, yet their reliable detection remains challenging due to morphological ambiguity and scanner variability. In this work, we investigated three variants of adapting the pathology foundation model UNI2-h for the MIDOG2025 Track 2 challenge. Starting from a LoRA-based baseline, we found that visual prompt tuning (VPT) substantially improved generalization, and that further integrating test-time augmentation (TTA) with Vahadane and Macenko stain normalization provided the best robustness. Our final submission achieved a balanced accuracy of 0.8837 and an ROC-AUC of 0.9513 on the preliminary leaderboard, ranking within the top 10 teams. These results demonstrate that prompt-based adaptation combined with stain-normalization TTA offers an effective strategy for atypical mitosis classification under diverse imaging conditions. 

**Abstract (ZH)**: 异常有丝分裂图象（AMFs）是临床相关性指标，用于反映异常细胞分裂，但由于形态学的模糊性和扫描仪的差异，其可靠的检测依然具有挑战性。在本研究中，我们探讨了三种适应病理学基础模型UNI2-h的方法，用于参与MIDOG2025赛道2的挑战。从一个基于LoRA的基本模型出发，我们发现视觉提示调整（VPT）显著提高了泛化能力，并且进一步整合测试时增强（TTA）与Vahadane和Macenko染色规范化技术提供了最佳的鲁棒性。最终提交结果在预览排行榜上达到了0.8837的平衡准确率和0.9513的ROC-AUC，排名前10位。这些结果表明，在不同成像条件下，基于提示的适应结合染色规范化TTA是一种有效的异型有丝分裂分类策略。 

---
# Enhanced Single-Cell RNA-seq Embedding through Gene Expression and Data-Driven Gene-Gene Interaction Integration 

**Title (ZH)**: 通过基因表达和数据驱动的基因-基因相互作用集成增强的单细胞RNA-seq嵌入 

**Authors**: Hojjat Torabi Goudarzi, Maziyar Baran Pouyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.02639)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) provides unprecedented insights into cellular heterogeneity, enabling detailed analysis of complex biological systems at single-cell resolution. However, the high dimensionality and technical noise inherent in scRNA-seq data pose significant analytical challenges. While current embedding methods focus primarily on gene expression levels, they often overlook crucial gene-gene interactions that govern cellular identity and function. To address this limitation, we present a novel embedding approach that integrates both gene expression profiles and data-driven gene-gene interactions. Our method first constructs a Cell-Leaf Graph (CLG) using random forest models to capture regulatory relationships between genes, while simultaneously building a K-Nearest Neighbor Graph (KNNG) to represent expression similarities between cells. These graphs are then combined into an Enriched Cell-Leaf Graph (ECLG), which serves as input for a graph neural network to compute cell embeddings. By incorporating both expression levels and gene-gene interactions, our approach provides a more comprehensive representation of cellular states. Extensive evaluation across multiple datasets demonstrates that our method enhances the detection of rare cell populations and improves downstream analyses such as visualization, clustering, and trajectory inference. This integrated approach represents a significant advance in single-cell data analysis, offering a more complete framework for understanding cellular diversity and dynamics. 

**Abstract (ZH)**: 单细胞RNA测序（scRNA-seq）提供了前所未有的细胞异质性见解，使在单细胞分辨率下对复杂生物系统进行详细分析成为可能。然而，scRNA-seq数据内置的高维度性和技术噪声带来了显著的分析挑战。尽管当前的嵌入方法主要关注基因表达水平，但往往会忽视调控细胞身份和功能的关键基因-基因相互作用。为解决这一限制，我们提出了一种新的嵌入方法，将基因表达谱和数据驱动的基因-基因相互作用相结合。该方法首先使用随机森林模型构建细胞-叶图（CLG），以捕获基因之间的调控关系，在此同时构建最近邻图（KNNG）以表示细胞之间的表达相似性。然后将这些图合并为增强的细胞-叶图（ECLG），作为图神经网络的输入以计算细胞嵌入。通过结合表达水平和基因-基因相互作用，我们的方法提供了更全面的细胞状态表示。广泛的数据集评估表明，我们的方法能够更好地检测稀有种群细胞，并提高下游分析如可视化、聚类和轨迹推断的效果。该集成方法在单细胞数据分析中代表了重要进展，提供了理解细胞多样性和动态的更完整框架。 

---
# IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering 

**Title (ZH)**: IS${}^3$：在声场景中基于深度滤波的通用冲击-稳态声分离 

**Authors**: Berger Clémentine, Stamadiatis Paraskevas, Badeau Roland, Essid Slim  

**Link**: [PDF](https://arxiv.org/pdf/2509.02622)  

**Abstract**: We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics. To this end, we introduce IS${}^3$, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics. 

**Abstract (ZH)**: 基于冲动-稳态声音分离的IS${}^3$神经网络及其应用 

---
# Radio Astronomy in the Era of Vision-Language Models: Prompt Sensitivity and Adaptation 

**Title (ZH)**: 视觉-语言模型时代射电天文：提示敏感性与适应性探究 

**Authors**: Mariia Drozdova, Erica Lastufka, Vitaliy Kinakh, Taras Holotyak, Daniel Schaerer, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2509.02615)  

**Abstract**: Vision-Language Models (VLMs), such as recent Qwen and Gemini models, are positioned as general-purpose AI systems capable of reasoning across domains. Yet their capabilities in scientific imaging, especially on unfamiliar and potentially previously unseen data distributions, remain poorly understood. In this work, we assess whether generic VLMs, presumed to lack exposure to astronomical corpora, can perform morphology-based classification of radio galaxies using the MiraBest FR-I/FR-II dataset. We explore prompting strategies using natural language and schematic diagrams, and, to the best of our knowledge, we are the first to introduce visual in-context examples within prompts in astronomy. Additionally, we evaluate lightweight supervised adaptation via LoRA fine-tuning. Our findings reveal three trends: (i) even prompt-based approaches can achieve good performance, suggesting that VLMs encode useful priors for unfamiliar scientific domains; (ii) however, outputs are highly unstable, i.e. varying sharply with superficial prompt changes such as layout, ordering, or decoding temperature, even when semantic content is held constant; and (iii) with just 15M trainable parameters and no astronomy-specific pretraining, fine-tuned Qwen-VL achieves near state-of-the-art performance (3% Error rate), rivaling domain-specific models. These results suggest that the apparent "reasoning" of VLMs often reflects prompt sensitivity rather than genuine inference, raising caution for their use in scientific domains. At the same time, with minimal adaptation, generic VLMs can rival specialized models, offering a promising but fragile tool for scientific discovery. 

**Abstract (ZH)**: 视觉-语言模型（VLMs），如近期的Qwen和Gemini模型，被定位为通用型AI系统，能够跨领域进行推理。然而，它们在科学成像领域的能力，特别是在不熟悉的且可能之前未见过的数据分布上的能力，仍然知之甚少。在本文中，我们评估未经天文语料库训练的通用VLMs是否能使用MiraBest FR-I/FR-II数据集对射电星系进行基于形态的分类。我们探讨了使用自然语言和示意图的提示策略，并据我们所知，我们是首次在天文学中引入视觉上下文示例作为提示。此外，我们评估了通过LoRA微调的轻量级监督适应。我们的发现揭示了三个趋势：（i）即使基于提示的方法也能实现良好的性能，表明VLMs编码了对于不熟悉的科学领域有用的前提；（ii）然而，输出非常不稳定，即轻微的提示变化（如布局、排序或解码温度）都会导致急剧变化，即使语义内容保持不变；（iii）在仅有1500万个可训练参数且没有特定于天文领域的预训练的情况下，微调的Qwen-VL实现了接近最先进的性能（3%的错误率），媲美专门模型。这些结果表明，VLMs表面上的“推理”往往反映了对提示的敏感性而非真正的推断，这对它们在科学领域的应用提出了警示。同时，通过最少的适配，通用VLMs可以媲美专门模型，为科学发现提供了一个有前景但易受限制的工具。 

---
# Resilient Biosecurity in the Era of AI-Enabled Bioweapons 

**Title (ZH)**: 人工智能赋能生物武器时代的韧性生物安全 

**Authors**: Jonathan Feldman, Tal Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2509.02610)  

**Abstract**: Recent advances in generative biology have enabled the design of novel proteins, creating significant opportunities for drug discovery while also introducing new risks, including the potential development of synthetic bioweapons. Existing biosafety measures primarily rely on inference-time filters such as sequence alignment and protein-protein interaction (PPI) prediction to detect dangerous outputs. In this study, we evaluate the performance of three leading PPI prediction tools: AlphaFold 3, AF3Complex, and SpatialPPIv2. These models were tested on well-characterized viral-host interactions, such as those involving Hepatitis B and SARS-CoV-2. Despite being trained on many of the same viruses, the models fail to detect a substantial number of known interactions. Strikingly, none of the tools successfully identify any of the four experimentally validated SARS-CoV-2 mutants with confirmed binding. These findings suggest that current predictive filters are inadequate for reliably flagging even known biological threats and are even more unlikely to detect novel ones. We argue for a shift toward response-oriented infrastructure, including rapid experimental validation, adaptable biomanufacturing, and regulatory frameworks capable of operating at the speed of AI-driven developments. 

**Abstract (ZH)**: 近期生成生物学的进展使得新型蛋白质的设计成为可能，为药物发现带来了重大机遇，同时也引入了新的风险，包括合成生物武器的可能性。现有的生物安全措施主要依赖于比对滤波以及蛋白质-蛋白质相互作用（PPI）预测等推理时滤波器来检测危险输出。本研究评估了三种领先的PPI预测工具：AlphaFold 3、AF3Complex和SpatialPPIv2。这些模型在已充分表征的病毒-宿主相互作用上进行了测试，例如乙型肝炎和SARS-CoV-2。尽管这些模型是基于相同病毒的大量数据进行训练，但它们未能检测到大量已知的相互作用。令人惊讶的是，这些工具均未能识别任何经过实验证实与SARS-CoV-2结合的四个突变体。这些发现表明，当前的预测滤波器不足以可靠地标记已知的生物威胁，更不用说检测新型生物威胁了。我们呼吁转向以响应为导向的基础设施，包括快速实验验证、灵活的生物制造以及能够与AI驱动的开发速度相匹配的监管框架。 

---
# Contrastive clustering based on regular equivalence for influential node identification in complex networks 

**Title (ZH)**: 基于规则等价性的对比聚类在复杂网络中关键节点识别 

**Authors**: Yanmei Hu, Yihang Wu, Bing Sun, Xue Yue, Biao Cai, Xiangtao Li, Yang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02609)  

**Abstract**: Identifying influential nodes in complex networks is a fundamental task in network analysis with wide-ranging applications across domains. While deep learning has advanced node influence detection, existing supervised approaches remain constrained by their reliance on labeled data, limiting their applicability in real-world scenarios where labels are scarce or unavailable. While contrastive learning demonstrates significant potential for performance enhancement, existing approaches predominantly rely on multiple-embedding generation to construct positive/negative sample pairs. To overcome these limitations, we propose ReCC (\textit{r}egular \textit{e}quivalence-based \textit{c}ontrastive \textit{c}lustering), a novel deep unsupervised framework for influential node identification. We first reformalize influential node identification as a label-free deep clustering problem, then develop a contrastive learning mechanism that leverages regular equivalence-based similarity, which captures structural similarities between nodes beyond local neighborhoods, to generate positive and negative samples. This mechanism is integrated into a graph convolutional network to learn node embeddings that are used to differentiate influential from non-influential nodes. ReCC is pre-trained using network reconstruction loss and fine-tuned with a combined contrastive and clustering loss, with both phases being independent of labeled data. Additionally, ReCC enhances node representations by combining structural metrics with regular equivalence-based similarities. Extensive experiments demonstrate that ReCC outperforms state-of-the-art approaches across several benchmarks. 

**Abstract (ZH)**: 基于正则等价的对比聚类：一种无监督的节点影响力识别深度学习框架 

---
# Towards Digital Twins for Optimal Radioembolization 

**Title (ZH)**: 面向最佳放射性栓塞的数字孪生技术研究 

**Authors**: Nisanth Kumar Panneerselvam, Guneet Mummaneni, Emilie Roncali  

**Link**: [PDF](https://arxiv.org/pdf/2509.02607)  

**Abstract**: Radioembolization is a localized liver cancer treatment that delivers radioactive microspheres (30 micron) to tumors via a catheter inserted in the hepatic arterial tree. The goal is to maximize therapeutic efficacy while minimizing damage to healthy liver tissue. However, optimization is challenging due to complex hepatic artery anatomy, variable blood flow, and uncertainty in microsphere transport. The creation of dynamic, patient-specific digital twins may provide a transformative solution to these challenges. This work outlines a framework for a liver radioembolization digital twin using high-fidelity computational fluid dynamics (CFD) and/or recent physics-informed machine learning approaches. The CFD approach involves microsphere transport calculations in the hepatic arterial tree with individual patient data, which enables personalized treatment planning. Although accurate, traditional CFD is computationally expensive and limits clinical applicability.
To accelerate simulations, physics-informed neural networks (PINNs) and their generative extensions play an increasingly important role. PINNs integrate governing equations, such as the Navier-Stokes equations, directly into the neural network training process, enabling mesh-free, data-efficient approximation of blood flow and microsphere transport. Physics-informed generative adversarial networks (PI-GANs), diffusion models (PI-DMs), and transformer-based architectures further enable uncertainty-aware, temporally resolved predictions with reduced computational cost. These AI surrogates not only maintain physical fidelity but also support rapid sampling of diverse flow scenarios, facilitating real-time decision support.
Together, CFD and physics-informed AI methods form the foundation of dynamic, patient-specific digital twin to optimize radioembolization planning and ultimately improve clinical outcomes. 

**Abstract (ZH)**: 基于物理学约束的神经网络和计算流体动力学在肝癌放射性栓塞数字双胞胎中的应用 

---
# MIDOG 2025: Mitotic Figure Detection with Attention-Guided False Positive Correction 

**Title (ZH)**: MIDOG 2025: 注意力引导的假阳性修正有丝分裂图检测 

**Authors**: Andrew Broad, Jason Keighley, Lucy Godson, Alex Wright  

**Link**: [PDF](https://arxiv.org/pdf/2509.02598)  

**Abstract**: We present a novel approach which extends the existing Fully Convolutional One-Stage Object Detector (FCOS) for mitotic figure detection. Our composite model adds a Feedback Attention Ladder CNN (FAL-CNN) model for classification of normal versus abnormal mitotic figures, feeding into a fusion network that is trained to generate adjustments to bounding boxes predicted by FCOS. Our network aims to reduce the false positive rate of the FCOS object detector, to improve the accuracy of object detection and enhance the generalisability of the network. Our model achieved an F1 score of 0.655 for mitosis detection on the preliminary evaluation dataset. 

**Abstract (ZH)**: 我们提出了一种新颖的方法，将现有的全卷积一阶段物体检测器（FCOS）扩展应用于有丝分裂图鉴识。我们的复合模型增加了反馈注意力梯形CNN（FAL-CNN）模型，用于正常与异常有丝分裂图的分类，并将其输入到一个融合网络中，该网络经过训练以生成对FCOS预测边界框的调整。我们的网络旨在降低FCOS物体检测器的假阳性率，提高物体检测的准确性，并增强网络的一般化能力。在初步评价数据集上，我们的模型实现了有丝分裂检测的F1分数为0.655。 

---
# Beyond Synthetic Augmentation: Group-Aware Threshold Calibration for Robust Balanced Accuracy in Imbalanced Learning 

**Title (ZH)**: 超越合成增强：群体意识阈值校准以实现稳健的类别平衡准确率在不平衡学习中的应用 

**Authors**: Hunter Gittlin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02592)  

**Abstract**: Class imbalance remains a fundamental challenge in machine learning, with traditional solutions often creating as many problems as they solve. We demonstrate that group-aware threshold calibration--setting different decision thresholds for different demographic groups--provides superior robustness compared to synthetic data generation methods. Through extensive experiments, we show that group-specific thresholds achieve 1.5-4% higher balanced accuracy than SMOTE and CT-GAN augmented models while improving worst-group balanced accuracy. Unlike single-threshold approaches that apply one cutoff across all groups, our group-aware method optimizes the Pareto frontier between balanced accuracy and worst-group balanced accuracy, enabling fine-grained control over group-level performance. Critically, we find that applying group thresholds to synthetically augmented data yields minimal additional benefit, suggesting these approaches are fundamentally redundant. Our results span seven model families including linear, tree-based, instance-based, and boosting methods, confirming that group-aware threshold calibration offers a simpler, more interpretable, and more effective solution to class imbalance. 

**Abstract (ZH)**: 群体意识阈值校准在机器学习中的不平衡类别问题上提供了 superior 的稳健性，相比合成数据生成方法，群体特定阈值在保持均衡准确率方面高出 1.5-4%，并在最不利群体的均衡准确率上有所改进。与在所有群体中应用单一阈值的方法不同，我们的群体意识方法优化了均衡准确率与最不利群体均衡准确率之间的帕累托前沿，从而实现对群体级性能的细粒度控制。关键的是，我们发现将群体阈值应用于合成增强数据几乎没有额外的好处，这表明这些方法本质上是冗余的。我们的结果涵盖了七类模型，包括线性模型、树基模型、实例基模型和提升方法，证实群体意识阈值校准提供了一个更简单、更具可解释性且更有效的解决类别不平衡问题的方法。 

---
# Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: Atypical Mitosis Classification 

**Title (ZH)**: Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: 非典型有丝分裂分类 

**Authors**: Mieko Ochi, Bae Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.02591)  

**Abstract**: Mitotic figures are classified into typical and atypical variants, with atypical counts correlating strongly with tumor aggressiveness. Accurate differentiation is therefore essential for patient prognostication and resource allocation, yet remains challenging even for expert pathologists. Here, we leveraged Pathology Foundation Models (PFMs) pre-trained on large histopathology datasets and applied parameter-efficient fine-tuning via low-rank adaptation. During training, we employ a fisheye transform to emphasize mitoses and Fourier Domain Adaptation using ImageNet target images. Finally, we ensembled multiple PFMs to integrate complementary morphological insights, achieving a high balanced accuracy on the Preliminary Evaluation Phase dataset. 

**Abstract (ZH)**: 有丝分裂图型被分类为典型和非典型变异，非典型计数与肿瘤 aggressiveness 强烈相关。因此，准确区分对于患者预后和资源分配至关重要，但即使是专家病理学家也面临挑战。在这里，我们利用预训练在大规模组织病理学数据集上的病理学基础模型（PFMs），并通过低秩适应进行参数高效的微调。在训练期间，我们采用鱼眼变换强调有丝分裂，并使用 ImageNet 目标图像进行频率域适应。最终，我们将多个 PFMs 進行集成，以结合互补的形态学见解，在初步评估阶段数据集上实现了高平衡准确率。 

---
# Charting the Future of Scholarly Knowledge with AI: A Community Perspective 

**Title (ZH)**: 用AI绘制学术知识的未来图景：一种社区视角 

**Authors**: Azanzi Jiomekong, Hande Küçük McGinty, Keith G. Mills, Allard Oelen, Enayat Rajabi, Harry McElroy, Antrea Christou, Anmol Saini, Janice Anta Zebaze, Hannah Kim, Anna M. Jacyszyn, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2509.02581)  

**Abstract**: Despite the growing availability of tools designed to support scholarly knowledge extraction and organization, many researchers still rely on manual methods, sometimes due to unfamiliarity with existing technologies or limited access to domain-adapted solutions. Meanwhile, the rapid increase in scholarly publications across disciplines has made it increasingly difficult to stay current, further underscoring the need for scalable, AI-enabled approaches to structuring and synthesizing scholarly knowledge. Various research communities have begun addressing this challenge independently, developing tools and frameworks aimed at building reliable, dynamic, and queryable scholarly knowledge bases. However, limited interaction across these communities has hindered the exchange of methods, models, and best practices, slowing progress toward more integrated solutions. This manuscript identifies ways to foster cross-disciplinary dialogue, identify shared challenges, categorize new collaboration and shape future research directions in scholarly knowledge and organization. 

**Abstract (ZH)**: 尽管设计用于支持学术知识提取和组织的工具日益增多，许多研究人员仍然依赖手动方法，有时是因为不熟悉现有技术或无法访问专业化的解决方案。同时，跨学科的学术出版物激增使得保持最新变得更加困难，进一步突显了需要可扩展的、基于AI的方法来结构化和综合学术知识的迫切性。各个研究社区已经开始独立应对这一挑战，开发工具和框架以构建可靠、动态和可查询的学术知识库。然而，这些社区之间的有限互动阻碍了方法、模型和最佳实践的交流，减缓了向更集成解决方案发展的进程。本文识别促进跨学科对话、确定共性挑战、分类新合作并塑造未来学术知识和组织研究方向的方式。 

---
# The Lifecycle Principle: Stabilizing Dynamic Neural Networks with State Memory 

**Title (ZH)**: 生命周期原理：通过状态记忆稳定动态神经网络 

**Authors**: Zichuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02575)  

**Abstract**: I investigate a stronger form of regularization by deactivating neurons for extended periods, a departure from the temporary changes of methods like Dropout. However, this long-term dynamism introduces a critical challenge: severe training instability when neurons are revived with random weights. To solve this, I propose the Lifecycle (LC) principle, a regularization mechanism centered on a key innovation: state memory. Instead of re-initializing a revived neuron, my method restores its parameters to their last known effective state. This process preserves learned knowledge and avoids destructive optimization shocks. My theoretical analysis reveals that the LC principle smooths the loss landscape, guiding optimization towards flatter minima associated with better generalization. Experiments on image classification benchmarks demonstrate that my method improves generalization and robustness. Crucially, ablation studies confirm that state memory is essential for achieving these gains. 

**Abstract (ZH)**: 一种新的长期去激活神经元正则化方法及其应用 

---
