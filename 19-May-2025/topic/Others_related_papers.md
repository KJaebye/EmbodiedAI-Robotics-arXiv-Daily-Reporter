# Conditioning Matters: Training Diffusion Policies is Faster Than You Think 

**Title (ZH)**: 条件决定一切：预训练扩散策略比你想象的更快 

**Authors**: Zibin Dong, Yicheng Liu, Yinchuan Li, Hang Zhao, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11123)  

**Abstract**: Diffusion policies have emerged as a mainstream paradigm for building vision-language-action (VLA) models. Although they demonstrate strong robot control capabilities, their training efficiency remains suboptimal. In this work, we identify a fundamental challenge in conditional diffusion policy training: when generative conditions are hard to distinguish, the training objective degenerates into modeling the marginal action distribution, a phenomenon we term loss collapse. To overcome this, we propose Cocos, a simple yet general solution that modifies the source distribution in the conditional flow matching to be condition-dependent. By anchoring the source distribution around semantics extracted from condition inputs, Cocos encourages stronger condition integration and prevents the loss collapse. We provide theoretical justification and extensive empirical results across simulation and real-world benchmarks. Our method achieves faster convergence and higher success rates than existing approaches, matching the performance of large-scale pre-trained VLAs using significantly fewer gradient steps and parameters. Cocos is lightweight, easy to implement, and compatible with diverse policy architectures, offering a general-purpose improvement to diffusion policy training. 

**Abstract (ZH)**: Diffusion策略已成为构建视觉-语言-动作（VLA）模型的主要范式。尽管它们展现了强大的机器人控制能力，但其训练效率仍不尽如人意。在这项工作中，我们识别出条件扩散策略训练中的一个基本挑战：当生成条件难以区分时，训练目标退化为建模边际动作分布，我们称这一现象为损失坍塌。为克服这一挑战，我们提出了Cocos，这是一种简单而通用的解决方案，通过在条件流匹配中修改源分布，使其依赖于条件。通过锚定源分布以与条件输入中提取的语义相关，Cocos 促进更强的条件整合并防止损失坍塌。我们提供了理论依据并在模拟和现实世界基准测试中进行了广泛的经验验证。该方法实现了更快的收敛速度和更高的成功率，使用显著较少的梯度步骤和参数匹配大规模预训练VLA的表现。Cocos 轻量级、易于实现，并与多种策略架构兼容，为扩散策略训练提供了一种通用改进方案。 

---
# GrowSplat: Constructing Temporal Digital Twins of Plants with Gaussian Splats 

**Title (ZH)**: GrowSplat: 基于高斯点构建植物的时空数字双胞胎 

**Authors**: Simeon Adebola, Shuangyu Xie, Chung Min Kim, Justin Kerr, Bart M. van Marrewijk, Mieke van Vlaardingen, Tim van Daalen, Robert van Loo, Jose Luis Susa Rincon, Eugen Solowjow, Rick van de Zedde, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.10923)  

**Abstract**: Accurate temporal reconstructions of plant growth are essential for plant phenotyping and breeding, yet remain challenging due to complex geometries, occlusions, and non-rigid deformations of plants. We present a novel framework for building temporal digital twins of plants by combining 3D Gaussian Splatting with a robust sample alignment pipeline. Our method begins by reconstructing Gaussian Splats from multi-view camera data, then leverages a two-stage registration approach: coarse alignment through feature-based matching and Fast Global Registration, followed by fine alignment with Iterative Closest Point. This pipeline yields a consistent 4D model of plant development in discrete time steps. We evaluate the approach on data from the Netherlands Plant Eco-phenotyping Center, demonstrating detailed temporal reconstructions of Sequoia and Quinoa species. Videos and Images can be seen at this https URL 

**Abstract (ZH)**: 植物生长精确的时间重建对于植物表型分析和育种至关重要，但由于植物几何结构的复杂性、遮挡和非刚性变形，这一任务依然具有挑战性。我们提出了一种结合3D高斯点云技术和稳健的样本对齐管道的新框架，以构建植物的时间数字孪生体。该方法首先从多视角相机数据中重建高斯点云，然后利用两阶段注册方法：基于特征的粗略对齐和快速全局注册，随后通过迭代最近点进行精细对齐。此管道生成了一致的离散时间步长的4D植物发育模型。我们在荷兰植物生态表型中心的数据上评估了该方法，展示了红杉和藜麦物种的详细时间重建结果。更多视频和图片请参见此链接：此 https URL。 

---
# Counterfactual Behavior Cloning: Offline Imitation Learning from Imperfect Human Demonstrations 

**Title (ZH)**: 反事实行为克隆：从 imperfect 人类示范中进行离线模仿学习 

**Authors**: Shahabedin Sagheb, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2505.10760)  

**Abstract**: Learning from humans is challenging because people are imperfect teachers. When everyday humans show the robot a new task they want it to perform, humans inevitably make errors (e.g., inputting noisy actions) and provide suboptimal examples (e.g., overshooting the goal). Existing methods learn by mimicking the exact behaviors the human teacher provides -- but this approach is fundamentally limited because the demonstrations themselves are imperfect. In this work we advance offline imitation learning by enabling robots to extrapolate what the human teacher meant, instead of only considering what the human actually showed. We achieve this by hypothesizing that all of the human's demonstrations are trying to convey a single, consistent policy, while the noise and sub-optimality within their behaviors obfuscates the data and introduces unintentional complexity. To recover the underlying policy and learn what the human teacher meant, we introduce Counter-BC, a generalized version of behavior cloning. Counter-BC expands the given dataset to include actions close to behaviors the human demonstrated (i.e., counterfactual actions that the human teacher could have intended, but did not actually show). During training Counter-BC autonomously modifies the human's demonstrations within this expanded region to reach a simple and consistent policy that explains the underlying trends in the human's dataset. Theoretically, we prove that Counter-BC can extract the desired policy from imperfect data, multiple users, and teachers of varying skill levels. Empirically, we compare Counter-BC to state-of-the-art alternatives in simulated and real-world settings with noisy demonstrations, standardized datasets, and real human teachers. See videos of our work here: this https URL 

**Abstract (ZH)**: 从人类那里学习具有挑战性，因为人类是不完美的教师。当日常中的人类向机器人展示他们想要机器人执行的新任务时，人类不可避免地会犯错误（例如，输入噪声动作）并提供次优示例（例如，超出目标）。现有方法通过模仿人类教师提供的 exact 行为来学习——但这种方法本质上是有限的，因为示范本身是不完美的。在本工作中，我们通过使机器人能够推断人类教师的意图，而不是仅仅考虑人类实际展示的内容，推进了离线模仿学习。我们假设所有的人类示范都在尝试传达一个单一且一致的策略，而行为中的噪声和次优性混淆了数据并引入了无意中的复杂性。为了恢复底层策略并学习人类教师的意图，我们引入了 Counter-BC，一种行为克隆的通用版本。Counter-BC 扩展了给定的数据集，包括靠近人类示范的动作（即，人类教师可能意图展示但未实际展示的反事实动作）。在训练过程中，Counter-BC 自动修改扩展现有区域中的人类示范，以达到一个简单且一致的策略，该策略解释了人类数据集中的基本趋势。理论上，我们证明 Counter-BC 可以从不完善的数据、多位用户和不同技能水平的教师中提取所需策略。实验上，我们在嘈杂示范、标准化数据集和真实人类教师的模拟和真实世界场景中，将 Counter-BC 与最先进的替代方法进行了比较。更多信息及视频请访问：this https URL 

---
# Predicting Human Behavior in Autonomous Systems: A Collaborative Machine Teaching Approach for Reducing Transfer of Control Events 

**Title (ZH)**: 自主系统中人类行为预测：一种减少控制权转移事件的协同机器教学方法 

**Authors**: Julian Wolter, Amr Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10695)  

**Abstract**: As autonomous systems become integral to various industries, effective strategies for fault handling are essential to ensure reliability and efficiency. Transfer of Control (ToC), a traditional approach for interrupting automated processes during faults, is often triggered unnecessarily in non-critical situations. To address this, we propose a data-driven method that uses human interaction data to train AI models capable of preemptively identifying and addressing issues or assisting users in resolution. Using an interactive tool simulating an industrial vacuum cleaner, we collected data and developed an LSTM-based model to predict user behavior. Our findings reveal that even data from non-experts can effectively train models to reduce unnecessary ToC events, enhancing the system's robustness. This approach highlights the potential of AI to learn directly from human problem-solving behaviors, complementing sensor data to improve industrial automation and human-AI collaboration. 

**Abstract (ZH)**: 随着自主系统在各个行业中的应用日益广泛，有效的故障处理策略对于确保可靠性和效率至关重要。转移控制（ToC），一种传统的在故障期间中断自动化流程的方法，往往在非关键情况下被不必要的触发。为了解决这一问题，我们提出了一种基于数据的方法，利用人类交互数据训练AI模型，以预先识别和解决问题，或在用户遇到问题时提供协助。我们使用一个模拟工业吸尘器的交互工具来收集数据，并开发了一个基于LSTM的模型来预测用户行为。研究结果表明，即使是非专家的数据也能有效训练模型以减少不必要的ToC事件，提高系统的鲁棒性。该方法突显了AI直接从人类问题解决行为中学习的潜力，从而补充传感器数据，提高工业自动化和人机协作的性能。 

---
# Decoupling Collision Avoidance in and for Optimal Control using Least-Squares Support Vector Machines 

**Title (ZH)**: 基于最小二乘支持向量机的碰撞避免与最优控制解耦 

**Authors**: Dries Dirckx, Wilm Decré, Jan Swevers  

**Link**: [PDF](https://arxiv.org/pdf/2505.11376)  

**Abstract**: This paper details an approach to linearise differentiable but non-convex collision avoidance constraints tailored to convex shapes. It revisits introducing differential collision avoidance constraints for convex objects into an optimal control problem (OCP) using the separating hyperplane theorem. By framing this theorem as a classification problem, the hyperplanes are eliminated as optimisation variables from the OCP. This effectively transforms non-convex constraints into linear constraints. A bi-level algorithm computes the hyperplanes between the iterations of an optimisation solver and subsequently embeds them as parameters into the OCP. Experiments demonstrate the approach's favourable scalability towards cluttered environments and its applicability to various motion planning approaches. It decreases trajectory computation times between 50\% and 90\% compared to a state-of-the-art approach that directly includes the hyperplanes as variables in the optimal control problem. 

**Abstract (ZH)**: 本文详细介绍了针对凸形物体的一种差分碰撞避免约束线性化的办法。通过重新审视使用分离超平面定理将差分碰撞避免约束引入最优控制问题（OCP）的方法，将分离超平面定理框架化为分类问题，从而在OCP中消除超平面作为优化变量，将非凸约束有效转换为线性约束。双层算法在优化求解器的迭代过程中计算超平面，并将其嵌入为参数到OCP中。实验表明，该方法在复杂环境下的可扩展性优越，并适用于多种运动规划方法，与直接将超平面作为变量纳入最优控制问题的最新方法相比，轨迹计算时间减少50%至90%。 

---
# Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions 

**Title (ZH)**: 使用广义李雅普诺夫函数认证强化学习策略的稳定性 

**Authors**: Kehan Long, Jorge Cortés, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2505.10947)  

**Abstract**: We study the problem of certifying the stability of closed-loop systems under control policies derived from optimal control or reinforcement learning (RL). Classical Lyapunov methods require a strict step-wise decrease in the Lyapunov function but such a certificate is difficult to construct for a learned control policy. The value function associated with an RL policy is a natural Lyapunov function candidate but it is not clear how it should be modified. To gain intuition, we first study the linear quadratic regulator (LQR) problem and make two key observations. First, a Lyapunov function can be obtained from the value function of an LQR policy by augmenting it with a residual term related to the system dynamics and stage cost. Second, the classical Lyapunov decrease requirement can be relaxed to a generalized Lyapunov condition requiring only decrease on average over multiple time steps. Using this intuition, we consider the nonlinear setting and formulate an approach to learn generalized Lyapunov functions by augmenting RL value functions with neural network residual terms. Our approach successfully certifies the stability of RL policies trained on Gymnasium and DeepMind Control benchmarks. We also extend our method to jointly train neural controllers and stability certificates using a multi-step Lyapunov loss, resulting in larger certified inner approximations of the region of attraction compared to the classical Lyapunov approach. Overall, our formulation enables stability certification for a broad class of systems with learned policies by making certificates easier to construct, thereby bridging classical control theory and modern learning-based methods. 

**Abstract (ZH)**: 研究基于最优控制或强化学习策略的闭环系统稳定性的认证问题 

---
# mmMirror: Device Free mmWave Indoor NLoS Localization Using Van-Atta-Array IRS 

**Title (ZH)**: mmMirror: 基于Van-Atta-阵列IRS的无设备毫米波室内非视距定位 

**Authors**: Yihe Yan, Zhenguo Shi, Yanxiang Wang, Cheng Jiang, Chun Tung Chou, Wen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10816)  

**Abstract**: Industry 4.0 is transforming manufacturing and logistics by integrating robots into shared human environments, such as factories, warehouses, and healthcare facilities. However, the risk of human-robot collisions, especially in Non-Line-of-Sight (NLoS) scenarios like around corners, remains a critical challenge. Existing solutions, such as vision-based and LiDAR systems, often fail under occlusion, lighting constraints, or privacy concerns, while RF-based systems are limited by range and accuracy.
To address these limitations, we propose mmMirror, a novel system leveraging a Van Atta Array-based millimeter-wave (mmWave) reconfigurable intelligent reflecting surface (IRS) for precise, device-free NLoS localization. mmMirror integrates seamlessly with existing frequency-modulated continuous-wave (FMCW) radars and offers: (i) robust NLoS localization with centimeter-level accuracy at ranges up to 3 m, (ii) seamless uplink and downlink communication between radar and IRS, (iii) support for multi-radar and multi-target scenarios via dynamic beam steering, and (iv) reduced scanning latency through adaptive time slot allocation. Implemented using commodity 24 GHz radars and a PCB-based IRS prototype, mmMirror demonstrates its potential in enabling safe human-robot interactions in dynamic and complex environments. 

**Abstract (ZH)**: 基于Van Atta阵列毫米波可重构智能反射表面的mmMirror：非视距场景下的精确无设备定位系统 

---
# MOSAAIC: Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation 

**Title (ZH)**: MOSAAIC: 管理优化以实现共创中的共享自主性、权威性和主动性 

**Authors**: Alayt Issak, Jeba Rezwana, Casper Harteveld  

**Link**: [PDF](https://arxiv.org/pdf/2505.11481)  

**Abstract**: Striking the appropriate balance between humans and co-creative AI is an open research question in computational creativity. Co-creativity, a form of hybrid intelligence where both humans and AI take action proactively, is a process that leads to shared creative artifacts and ideas. Achieving a balanced dynamic in co-creativity requires characterizing control and identifying strategies to distribute control between humans and AI. We define control as the power to determine, initiate, and direct the process of co-creation. Informed by a systematic literature review of 172 full-length papers, we introduce MOSAAIC (Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation), a novel framework for characterizing and balancing control in co-creation. MOSAAIC identifies three key dimensions of control: autonomy, initiative, and authority. We supplement our framework with control optimization strategies in co-creation. To demonstrate MOSAAIC's applicability, we analyze the distribution of control in six existing co-creative AI case studies and present the implications of using this framework. 

**Abstract (ZH)**: 在计算创意中实现人类与合创AI之间的适当平衡是一个开放的研究问题。合创是一种混合智能形式，其中人类和AI主动采取行动，这一过程会产生共享的创意成果和想法。实现合创中的平衡动态需要表征控制并确定在人类和AI之间分配控制的策略。我们将控制定义为确定、启动和指导合创过程的能力。基于对172篇全文论文的系统的文献综述，我们引入了MOSAAIC（管理和优化以实现合创中的共享自主权、权威和主动性）框架，这是一种表征和平衡合创中控制的新框架。MOSAAIC识别出控制的三个关键维度：自主权、主动性以及权威。我们还为合创中的控制优化提供了策略。为了展示MOSAAIC的适用性，我们分析了六个现有合创AI案例研究中的控制分配，并讨论了使用该框架的影响。 

---
# Automatic Reward Shaping from Confounded Offline Data 

**Title (ZH)**: 从混杂的离线数据中自动构建奖励函数 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2505.11478)  

**Abstract**: A key task in Artificial Intelligence is learning effective policies for controlling agents in unknown environments to optimize performance measures. Off-policy learning methods, like Q-learning, allow learners to make optimal decisions based on past experiences. This paper studies off-policy learning from biased data in complex and high-dimensional domains where \emph{unobserved confounding} cannot be ruled out a priori. Building on the well-celebrated Deep Q-Network (DQN), we propose a novel deep reinforcement learning algorithm robust to confounding biases in observed data. Specifically, our algorithm attempts to find a safe policy for the worst-case environment compatible with the observations. We apply our method to twelve confounded Atari games, and find that it consistently dominates the standard DQN in all games where the observed input to the behavioral and target policies mismatch and unobserved confounders exist. 

**Abstract (ZH)**: 人工智能中的一个关键任务是学习有效的策略以在未知环境中控制代理，以优化性能指标。离策学习方法，如Q学习，允许学习者基于过往经验做出最优决策。本文研究了在复杂和高维域中从可能存在未观察混杂因素的偏差数据中进行离策学习。基于广受赞誉的深度Q网络（DQN），我们提出了一种新型深度强化学习算法，该算法能够抵抗观测数据中的混杂偏差。具体而言，我们的算法尝试找到与观测数据相兼容的最坏情况环境下的安全策略。我们将方法应用于十二个存在混杂因素的Atari游戏，并发现它在所有行为策略和目标策略的输入匹配存在偏差的游戏场景中均优于标准DQN。 

---
# Extracting Explainable Dates From Medical Images By Reverse-Engineering UNIX Timestamps 

**Title (ZH)**: 从医学图像中通过逆向工程UNIX时间戳提取可解释日期 

**Authors**: Lee Harris, James Bentham, Philippe De Wilde  

**Link**: [PDF](https://arxiv.org/pdf/2505.11451)  

**Abstract**: Dates often contribute towards highly impactful medical decisions, but it is rarely clear how to extract this data. AI has only just begun to be used transcribe such documents, and common methods are either to trust that the output produced by a complex AI model, or to parse the text using regular expressions. Recent work has established that regular expressions are an explainable form of logic, but it is difficult to decompose these into the component parts that are required to construct precise UNIX timestamps. First, we test publicly-available regular expressions, and we found that these were unable to capture a significant number of our dates. Next, we manually created easily-decomposable regular expressions, and we found that these were able to detect the majority of real dates, but also a lot of sequences of text that look like dates. Finally, we used regular expression synthesis to automatically identify regular expressions from the reverse-engineered UNIX timestamps that we created. We find that regular expressions created by regular expression synthesis detect far fewer sequences of text that look like dates than those that were manually created, at the cost of a slight increase to the number of missed dates. Overall, our results show that regular expressions can be created through regular expression synthesis to identify complex dates and date ranges in text transcriptions. To our knowledge, our proposed way of learning deterministic logic by reverse-engineering several many-one mappings and feeding these into a regular expression synthesiser is a new approach. 

**Abstract (ZH)**: 日期信息通常对医学决策有重大影响，但很少清楚如何提取这些数据。现有的方法要么完全信任复杂AI模型的输出，要么使用正则表达式解析文本。最近的研究表明，正则表达式是一种可解释的逻辑形式，但将其分解为构建精确UNIX时间戳所需的组件部分却相当困难。首先，我们测试了公开可用的正则表达式，发现它们无法捕捉到我们大量日期中的很大一部分。接着，我们手动创建了易于分解的正则表达式，并发现这些正则表达式能够检测到大部分真实日期，但也检测到了大量看起来像日期的文本序列。最后，我们使用正则表达式合成从我们逆向工程得到的UNIX时间戳自动识别正则表达式。我们发现，由正则表达式合成生成的正则表达式检测到的看起来像日期的文本序列要少于手工创建的正则表达式，但可能会错过更多日期。总体来说，我们的结果表明，可以通过正则表达式合成识别文本转录中的复杂日期和日期范围。据我们所知，通过逆向工程多个单值映射并将其输入正则表达式合成器来学习确定性逻辑是一种新的方法。 

---
# Meta-World+: An Improved, Standardized, RL Benchmark 

**Title (ZH)**: Meta-World+: 一个改进的、标准化的RL基准 

**Authors**: Reginald McLean, Evangelos Chatzaroulas, Luc McCutcheon, Frank Röder, Tianhe Yu, Zhanpeng He, K.R. Zentner, Ryan Julian, J K Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2505.11289)  

**Abstract**: Meta-World is widely used for evaluating multi-task and meta-reinforcement learning agents, which are challenged to master diverse skills simultaneously. Since its introduction however, there have been numerous undocumented changes which inhibit a fair comparison of algorithms. This work strives to disambiguate these results from the literature, while also leveraging the past versions of Meta-World to provide insights into multi-task and meta-reinforcement learning benchmark design. Through this process we release a new open-source version of Meta-World (this https URL) that has full reproducibility of past results, is more technically ergonomic, and gives users more control over the tasks that are included in a task set. 

**Abstract (ZH)**: Meta-World：用于评估多任务和元强化学习代理的广泛使用基准，尽管自引入以来存在众多未记录的变化，阻碍了算法之间的公平比较，本工作致力于澄清文献中的结果，并利用Meta-World的过去版本提供多任务和元强化学习基准设计的见解。通过这一过程，我们发布了Meta-World的新开源版本（详见https://...），该版本具有对过去结果的完全可再现性、更高的技术便捷性和更多的任务控制权。 

---
# GLOVA: Global and Local Variation-Aware Analog Circuit Design with Risk-Sensitive Reinforcement Learning 

**Title (ZH)**: GLOVA：具有风险敏感型强化学习的全局与局部变异aware模拟电路设计 

**Authors**: Dongjun Kim, Junwoo Park, Chaehyeon Shin, Jaeheon Jung, Kyungho Shin, Seungheon Baek, Sanghyuk Heo, Woongrae Kim, Inchul Jeong, Joohwan Cho, Jongsun Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.11208)  

**Abstract**: Analog/mixed-signal circuit design encounters significant challenges due to performance degradation from process, voltage, and temperature (PVT) variations. To achieve commercial-grade reliability, iterative manual design revisions and extensive statistical simulations are required. While several studies have aimed to automate variation aware analog design to reduce time-to-market, the substantial mismatches in real-world wafers have not been thoroughly addressed. In this paper, we present GLOVA, an analog circuit sizing framework that effectively manages the impact of diverse random mismatches to improve robustness against PVT variations. In the proposed approach, risk-sensitive reinforcement learning is leveraged to account for the reliability bound affected by PVT variations, and ensemble-based critic is introduced to achieve sample-efficient learning. For design verification, we also propose $\mu$-$\sigma$ evaluation and simulation reordering method to reduce simulation costs of identifying failed designs. GLOVA supports verification through industrial-level PVT variation evaluation methods, including corner simulation as well as global and local Monte Carlo (MC) simulations. Compared to previous state-of-the-art variation-aware analog sizing frameworks, GLOVA achieves up to 80.5$\times$ improvement in sample efficiency and 76.0$\times$ reduction in time. 

**Abstract (ZH)**: 模拟/混合信号电路设计面临着严重的挑战，由于工艺、电压和温度（PVT）变化导致性能下降。为了实现商业级别的可靠性，需要进行迭代的手动设计修订和大量的统计模拟。尽管有多项研究致力于通过自动化变异感知模拟设计来缩短上市时间，但在实际晶圆上的显著不匹配问题尚未得到充分解决。在这篇论文中，我们提出了GLOVA，一种有效的模拟电路缩放框架，用于管理各种随机不匹配的影響，以提高对PVT变化的鲁棒性。在所提出的方法中，风险敏感的强化学习被用于考虑PVT变化影响的可靠性边界，引入基于集成的评论者以实现样本高效学习。在设计验证方面，我们还提出了$\mu$-$\sigma$评估和仿真排序方法，以降低识别失败设计的仿真成本。GLOVA支持通过工业级PVT变化评估方法进行验证，包括角仿真以及全局和局部蒙特卡洛（MC）仿真。与之前最先进的变异感知模拟缩放框架相比，GLOVA在样本效率上实现了最高80.5倍的改进，并在时间上实现了76.0倍的减少。 

---
# Scalability of Reinforcement Learning Methods for Dispatching in Semiconductor Frontend Fabs: A Comparison of Open-Source Models with Real Industry Datasets 

**Title (ZH)**: 半导体前端fab中调度方法的强化学习方法可扩展性研究：开源模型与实际工业数据的对比 

**Authors**: Patrick Stöckermann, Henning Südfeld, Alessandro Immordino, Thomas Altenmüller, Marc Wegmann, Martin Gebser, Konstantin Schekotihin, Georg Seidel, Chew Wye Chan, Fei Fei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11135)  

**Abstract**: Benchmark datasets are crucial for evaluating approaches to scheduling or dispatching in the semiconductor industry during the development and deployment phases. However, commonly used benchmark datasets like the Minifab or SMT2020 lack the complex details and constraints found in real-world scenarios. To mitigate this shortcoming, we compare open-source simulation models with a real industry dataset to evaluate how optimization methods scale with different levels of complexity. Specifically, we focus on Reinforcement Learning methods, performing optimization based on policy-gradient and Evolution Strategies. Our research provides insights into the effectiveness of these optimization methods and their applicability to realistic semiconductor frontend fab simulations. We show that our proposed Evolution Strategies-based method scales much better than a comparable policy-gradient-based approach. Moreover, we identify the selection and combination of relevant bottleneck tools to control by the agent as crucial for an efficient optimization. For the generalization across different loading scenarios and stochastic tool failure patterns, we achieve advantages when utilizing a diverse training dataset. While the overall approach is computationally expensive, it manages to scale well with the number of CPU cores used for training. For the real industry dataset, we achieve an improvement of up to 4% regarding tardiness and up to 1% regarding throughput. For the less complex open-source models Minifab and SMT2020, we observe double-digit percentage improvement in tardiness and single digit percentage improvement in throughput by use of Evolution Strategies. 

**Abstract (ZH)**: 基准数据集对于半导体行业调度或分派方法的开发和部署阶段评估至关重要。然而，常用的基准数据集如Minifab或SMT2020缺乏实际场景中的复杂细节和约束。为了弥补这一不足，我们将开源仿真模型与真实工业数据集进行比较，评估优化方法在不同复杂度水平下的可扩展性。具体而言，我们专注于基于策略梯度和进化策略的强化学习方法。我们的研究提供了这些优化方法有效性和适用性的见解，特别是在实际半导体前端晶圆厂仿真中的应用。我们证明基于进化策略的方法在可扩展性方面明显优于基于策略梯度的方法。此外，我们发现代理控制的相关瓶颈工具的选择和组合对于高效的优化至关重要。为了在不同装载场景和随机工具故障模式下实现泛化，使用多样化训练数据集可以获得优势。尽管总体方法计算成本较高，但它能够很好地与用于训练的CPU内核数量扩展。对于真实工业数据集，我们关于延误的改进高达4%，关于吞吐量的改进高达1%。对于较简单的开源模型Minifab和SMT2020，我们通过使用进化策略观察到延误的多位数百分比改进和吞吐量的一位数百分比改进。 

---
# Predicting Student Dropout Risk With A Dual-Modal Abrupt Behavioral Changes Approach 

**Title (ZH)**: 基于双模态突变行为变化的方法预测学生辍学风险 

**Authors**: Jiabei Cheng, Zhen-Qun Yang, Jiannong Cao, Yu Yang, Xinzhe Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11119)  

**Abstract**: Timely prediction of students at high risk of dropout is critical for early intervention and improving educational outcomes. However, in offline educational settings, poor data quality, limited scale, and high heterogeneity often hinder the application of advanced machine learning models. Furthermore, while educational theories provide valuable insights into dropout phenomena, the lack of quantifiable metrics for key indicators limits their use in data-driven modeling. Through data analysis and a review of educational literature, we identified abrupt changes in student behavior as key early signals of dropout risk. To address this, we propose the Dual-Modal Multiscale Sliding Window (DMSW) Model, which integrates academic performance and behavioral data to dynamically capture behavior patterns using minimal data. The DMSW model improves prediction accuracy by 15% compared to traditional methods, enabling educators to identify high-risk students earlier, provide timely support, and foster a more inclusive learning environment. Our analysis highlights key behavior patterns, offering practical insights for preventive strategies and tailored support. These findings bridge the gap between theory and practice in dropout prediction, giving educators an innovative tool to enhance student retention and outcomes. 

**Abstract (ZH)**: 及时预测高风险辍学学生对于早期干预和提高教育成果至关重要。然而，在线教育环境中，数据质量差、规模有限以及高度异质性常常阻碍先进机器学习模型的应用。此外，尽管教育理论为辍学现象提供了宝贵见解，但由于缺乏可量化的关键指标的度量标准，限制了其在数据驱动建模中的应用。通过数据分析和教育文献综述，我们确定了学生行为的突变是辍学风险的早期关键信号。为此，我们提出了一种双模态多尺度滑动窗口（DMSW）模型，该模型整合了学业表现和行为数据，通过最少的数据动态捕捉行为模式。与传统方法相比，DMSW模型将预测准确性提高了15%，使教育者能够更早地识别高风险学生，提供及时支持，并促进更具包容性的学习环境。我们的分析揭示了关键行为模式，提供了预防策略和个性化支持的实用见解。这些发现弥合了辍学预测理论与实践之间的差距，为教育者提供了一个创新工具，以增强学生的留存率和成果。 

---
# Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data 

**Title (ZH)**: 使用原型检测和反事实解释分析客户旅程sequential数据中的客户旅程 

**Authors**: Keita Kinjo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11086)  

**Abstract**: Recently, the proliferation of omni-channel platforms has attracted interest in customer journeys, particularly regarding their role in developing marketing strategies. However, few efforts have been taken to quantitatively study or comprehensively analyze them owing to the sequential nature of their data and the complexity involved in analysis. In this study, we propose a novel approach comprising three steps for analyzing customer journeys. First, the distance between sequential data is defined and used to identify and visualize representative sequences. Second, the likelihood of purchase is predicted based on this distance. Third, if a sequence suggests no purchase, counterfactual sequences are recommended to increase the probability of a purchase using a proposed method, which extracts counterfactual explanations for sequential data. A survey was conducted, and the data were analyzed; the results revealed that typical sequences could be extracted, and the parts of those sequences important for purchase could be detected. We believe that the proposed approach can support improvements in various marketing activities. 

**Abstract (ZH)**: 最近，全渠道平台的普及引起了对客户旅程的兴趣，特别是在制定营销策略中的作用。然而，由于其数据的序列性质和分析的复杂性，鲜有人对其进行定量研究或全面分析。在本研究中，我们提出了一种包含三个步骤的新方法来分析客户旅程。首先，定义序列数据之间的距离并用于识别和可视化代表性序列。其次，基于此距离预测购买的可能性。第三，如果序列表明无购买，我们将推荐反事实序列以提高购买概率，该方法提取了序列数据的反事实解释。我们进行了调查并分析了数据，结果显示可以提取典型的序列，并能够检测这些序列中与购买相关的重要部分。我们认为所提出的方法可以支持各类营销活动的改进。 

---
# Most General Explanations of Tree Ensembles 

**Title (ZH)**: 树ensemble的最一般解释 

**Authors**: Yacine Izza, Alexey Ignatiev, Joao Marques-Silva, Peter J. Stuckey  

**Link**: [PDF](https://arxiv.org/pdf/2505.10991)  

**Abstract**: Explainable Artificial Intelligence (XAI) is critical for attaining trust in the operation of AI systems. A key question of an AI system is ``why was this decision made this way''. Formal approaches to XAI use a formal model of the AI system to identify abductive explanations. While abductive explanations may be applicable to a large number of inputs sharing the same concrete values, more general explanations may be preferred for numeric inputs. So-called inflated abductive explanations give intervals for each feature ensuring that any input whose values fall withing these intervals is still guaranteed to make the same prediction. Inflated explanations cover a larger portion of the input space, and hence are deemed more general explanations. But there can be many (inflated) abductive explanations for an instance. Which is the best? In this paper, we show how to find a most general abductive explanation for an AI decision. This explanation covers as much of the input space as possible, while still being a correct formal explanation of the model's behaviour. Given that we only want to give a human one explanation for a decision, the most general explanation gives us the explanation with the broadest applicability, and hence the one most likely to seem sensible. (The paper has been accepted at IJCAI2025 conference.) 

**Abstract (ZH)**: 解释可理解的人工智能（XAI）对于获得对AI系统操作的信任至关重要。一个关键问题是如何从AI系统中解释“为何作出了这种决策”。形式化的XAI方法利用AI系统的形式模型来识别 abduction 解释。虽然 abduction 解释可能适用于具有相同具体值的大量输入，但对于数值输入而言，更一般的解释可能更受青睐。所谓的膨胀 abduction 解释为每个特征提供区间，确保任何值落在这些区间内的输入依然能得到相同的预测。膨胀解释覆盖了较大的输入空间，因此被认为是更一般的解释。但一个实例可能有多个（膨胀的） abduction 解释。哪个是最好的？本文展示了如何找到一个最一般的 abduction 解释，该解释覆盖尽可能多的输入空间，同时仍然是正确的形式模型行为解释。鉴于我们只想给人类提供一个关于决策的解释，最一般的解释提供了最具广泛适用性的解释，因此最有可能显得合理。（该论文已被接受参加IJCAI2025会议。） 

---
# DRL-Based Injection Molding Process Parameter Optimization for Adaptive and Profitable Production 

**Title (ZH)**: 基于DRL的注射 molding 工艺参数优化以实现自适应和盈利生产 

**Authors**: Joon-Young Kim, Jecheon Yu, Heekyu Kim, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10988)  

**Abstract**: Plastic injection molding remains essential to modern manufacturing. However, optimizing process parameters to balance product quality and profitability under dynamic environmental and economic conditions remains a persistent challenge. This study presents a novel deep reinforcement learning (DRL)-based framework for real-time process optimization in injection molding, integrating product quality and profitability into the control objective. A profit function was developed to reflect real-world manufacturing costs, incorporating resin, mold wear, and electricity prices, including time-of-use variations. Surrogate models were constructed to predict product quality and cycle time, enabling efficient offline training of DRL agents using soft actor-critic (SAC) and proximal policy optimization (PPO) algorithms. Experimental results demonstrate that the proposed DRL framework can dynamically adapt to seasonal and operational variations, consistently maintaining product quality while maximizing profit. Compared to traditional optimization methods such as genetic algorithms, the DRL models achieved comparable economic performance with up to 135x faster inference speeds, making them well-suited for real-time applications. The framework's scalability and adaptability highlight its potential as a foundation for intelligent, data-driven decision-making in modern manufacturing environments. 

**Abstract (ZH)**: 基于深度强化学习的注射模具实时优化框架：平衡产品质量与盈利能力 

---
# Facets in Argumentation: A Formal Approach to Argument Significance 

**Title (ZH)**: 论辩要素：论据显著性的形式化方法 

**Authors**: Johannes Fichte, Nicolas Fröhlich, Markus Hecher, Victor Lagerkvist, Yasir Mahmood, Arne Meier, Jonathan Persson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10982)  

**Abstract**: Argumentation is a central subarea of Artificial Intelligence (AI) for modeling and reasoning about arguments. The semantics of abstract argumentation frameworks (AFs) is given by sets of arguments (extensions) and conditions on the relationship between them, such as stable or admissible. Today's solvers implement tasks such as finding extensions, deciding credulous or skeptical acceptance, counting, or enumerating extensions. While these tasks are well charted, the area between decision, counting/enumeration and fine-grained reasoning requires expensive reasoning so far. We introduce a novel concept (facets) for reasoning between decision and enumeration. Facets are arguments that belong to some extensions (credulous) but not to all extensions (skeptical). They are most natural when a user aims to navigate, filter, or comprehend the significance of specific arguments, according to their needs. We study the complexity and show that tasks involving facets are much easier than counting extensions. Finally, we provide an implementation, and conduct experiments to demonstrate feasibility. 

**Abstract (ZH)**: 论辩是人工智能（AI）中的一个核心子领域，用于建模和推理论辩。抽象论辩框架（AFs）的语义由论辩集（扩展）及其之间的关系条件给出，例如稳定或可接受。当前的求解器实现查找扩展、决定性的或怀疑性的接受、计数或枚举扩展等功能。尽管这些任务已经很清楚，但在决策、计数/枚举与精细推理之间的区域仍需要昂贵的推理。我们引入了一个新的概念（切面），用于决策与枚举之间的推理。切面是属于某些扩展（确信的）但不属于所有扩展（怀疑的）的论辩。当用户旨在根据其需求导航、过滤或理解特定论辩的意义时，它们是最自然的。我们研究了切面任务的复杂性，表明涉及切面的任务比计数扩展要容易得多。最后，我们提供了一个实现，并进行实验以证明可行性。 

---
# MPS-Prover: Advancing Stepwise Theorem Proving by Multi-Perspective Search and Data Curation 

**Title (ZH)**: MPS-Prover: 通过多视角搜索和数据整理推进分步定理证明 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10962)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages remains a formidable challenge in AI, demanding rigorous logical deduction and navigating vast search spaces. While large language models (LLMs) have shown promising performance, existing stepwise provers often suffer from biased search guidance, leading to inefficiencies and suboptimal proof strategies. This paper introduces the Multi-Perspective Search Prover (MPS-Prover), a novel stepwise ATP system designed to overcome these limitations. MPS-Prover incorporates two key innovations: a highly effective post-training data curation strategy that prunes approximately 40% of redundant training data without sacrificing performance, and a multi-perspective tree search mechanism. This search integrates a learned critic model with strategically designed heuristic rules to diversify tactic selection, prevent getting trapped in unproductive states, and enhance search robustness. Extensive evaluations demonstrate that MPS-Prover achieves state-of-the-art performance on multiple challenging benchmarks, including miniF2F and ProofNet, outperforming prior 7B parameter models. Furthermore, our analyses reveal that MPS-Prover generates significantly shorter and more diverse proofs compared to existing stepwise and whole-proof methods, highlighting its efficiency and efficacy. Our work advances the capabilities of LLM-based formal reasoning and offers a robust framework and a comprehensive analysis for developing more powerful theorem provers. 

**Abstract (ZH)**: 形式语言中的自动定理证明（ATP）仍然是AI领域的 formidable 挑战，要求严格的逻辑推理和探索庞大的搜索空间。虽然大型语言模型（LLMs）展现了令人鼓舞的性能，但现有的逐步证明器常常遭受偏颇的搜索指导，导致效率低下和次优证明策略。本文介绍了多视角搜索证明器（MPS-Prover），这是一种新型的逐步ATP系统，旨在克服这些限制。MPS-Prover 包含两项关键创新：一种高效的后训练data curation 策略，该策略在不牺牲性能的情况下去掉了约40%的冗余训练数据，以及一种多视角树搜索机制。这种搜索机制结合了学习到的critic 模型和精心设计的启发式规则，以多样化策略选择、避免陷入无生产力状态，并增强搜索的稳健性。 extensive 评估表明，MPS-Prover 在多个具有挑战性的基准测试，包括miniF2F和ProofNet 上达到了最先进的性能，超越了此前的7B参数模型。此外，我们的分析表明，与现有的逐步证明方法和整体证明方法相比，MPS-Prover 生成的证明更短且更具多样性，突显了其高效性和有效性。本文推进了基于LLM的形式推理能力，并提供了更强大的定理证明器开发的坚固框架和全面分析。 

---
# MCU: Improving Machine Unlearning through Mode Connectivity 

**Title (ZH)**: MCU：通过模式连通性提升机器卸载性能 

**Authors**: Yingdan Shi, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10859)  

**Abstract**: Machine Unlearning (MU) aims to remove the information of specific training data from a trained model, ensuring compliance with privacy regulations and user requests. While one line of existing MU methods relies on linear parameter updates via task arithmetic, they suffer from weight entanglement. In this work, we propose a novel MU framework called Mode Connectivity Unlearning (MCU) that leverages mode connectivity to find an unlearning pathway in a nonlinear manner. To further enhance performance and efficiency, we introduce a parameter mask strategy that not only improves unlearning effectiveness but also reduces computational overhead. Moreover, we propose an adaptive adjustment strategy for our unlearning penalty coefficient to adaptively balance forgetting quality and predictive performance during training, eliminating the need for empirical hyperparameter tuning. Unlike traditional MU methods that identify only a single unlearning model, MCU uncovers a spectrum of unlearning models along the pathway. Overall, MCU serves as a plug-and-play framework that seamlessly integrates with any existing MU methods, consistently improving unlearning efficacy. Extensive experiments on the image classification task demonstrate that MCU achieves superior performance. 

**Abstract (ZH)**: 模式连接性卸学（MCU）：一种利用模式连接性进行非线性卸学的新型框架 

---
# SECRET: Semi-supervised Clinical Trial Document Similarity Search 

**Title (ZH)**: SECTOR: 半监督临床试验文档相似性搜索 

**Authors**: Trisha Das, Afrah Shafquat, Beigi Mandis, Jacob Aptekar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.10780)  

**Abstract**: Clinical trials are vital for evaluation of safety and efficacy of new treatments. However, clinical trials are resource-intensive, time-consuming and expensive to conduct, where errors in trial design, reduced efficacy, and safety events can result in significant delays, financial losses, and damage to reputation. These risks underline the importance of informed and strategic decisions in trial design to mitigate these risks and improve the chances of a successful trial. Identifying similar historical trials is critical as these trials can provide an important reference for potential pitfalls and challenges including serious adverse events, dosage inaccuracies, recruitment difficulties, patient adherence issues, etc. Addressing these challenges in trial design can lead to development of more effective study protocols with optimized patient safety and trial efficiency. In this paper, we present a novel method to identify similar historical trials by summarizing clinical trial protocols and searching for similar trials based on a query trial's protocol. Our approach significantly outperforms all baselines, achieving up to a 78% improvement in recall@1 and a 53% improvement in precision@1 over the best baseline. We also show that our method outperforms all other baselines in partial trial similarity search and zero-shot patient-trial matching, highlighting its superior utility in these tasks. 

**Abstract (ZH)**: 临床试验对于评估新治疗方法的安全性和有效性至关重要。然而，临床试验耗资巨大、耗时且费用高昂，试验设计中的错误、效力降低和安全性事件可能会导致重大延误、经济损失和声誉损害。这些风险凸显了在试验设计中做出知情和战略决策的重要性，以降低这些风险并提高试验成功的可能性。识别相似的历史试验至关重要，这些试验可以为潜在的风险和挑战（包括严重不良事件、剂量不准确、招募困难、患者依从性问题等）提供重要参考。通过在试验设计中应对这些挑战，可以开发出更有效的研究方案，优化患者安全和试验效率。在本文中，我们提出了一种新颖的方法，通过总结临床试验协议并依据查询试验的协议搜索相似试验。我们的方法显著优于所有基线，召回率@1提高了78%，精确率@1提高了53%。我们还展示了我们的方法在部分试验相似性搜索和零样本患者-试验匹配任务中均优于所有其他基线，突显了其在这些任务中的优越实用性。 

---
# On the Evaluation of Engineering Artificial General Intelligence 

**Title (ZH)**: 工程通用人工智能的评估方法 

**Authors**: Sandeep Neema, Susmit Jha, Adam Nagel, Ethan Lew, Chandrasekar Sureshkumar, Aleksa Gordic, Chase Shimmin, Hieu Nguygen, Paul Eremenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.10653)  

**Abstract**: We discuss the challenges and propose a framework for evaluating engineering artificial general intelligence (eAGI) agents. We consider eAGI as a specialization of artificial general intelligence (AGI), deemed capable of addressing a broad range of problems in the engineering of physical systems and associated controllers. We exclude software engineering for a tractable scoping of eAGI and expect dedicated software engineering AI agents to address the software implementation challenges. Similar to human engineers, eAGI agents should possess a unique blend of background knowledge (recall and retrieve) of facts and methods, demonstrate familiarity with tools and processes, exhibit deep understanding of industrial components and well-known design families, and be able to engage in creative problem solving (analyze and synthesize), transferring ideas acquired in one context to another. Given this broad mandate, evaluating and qualifying the performance of eAGI agents is a challenge in itself and, arguably, a critical enabler to developing eAGI agents. In this paper, we address this challenge by proposing an extensible evaluation framework that specializes and grounds Bloom's taxonomy - a framework for evaluating human learning that has also been recently used for evaluating LLMs - in an engineering design context. Our proposed framework advances the state of the art in benchmarking and evaluation of AI agents in terms of the following: (a) developing a rich taxonomy of evaluation questions spanning from methodological knowledge to real-world design problems; (b) motivating a pluggable evaluation framework that can evaluate not only textual responses but also evaluate structured design artifacts such as CAD models and SysML models; and (c) outlining an automatable procedure to customize the evaluation benchmark to different engineering contexts. 

**Abstract (ZH)**: 我们探讨了挑战并提出了评估工程通用人工智能（eAGI）代理的框架。 

---
# HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation 

**Title (ZH)**: HumaniBench: 以人为本的大型多模态模型评估框架 

**Authors**: Shaina Raza, Aravind Narayanan, Vahid Reza Khazaie, Ashmal Vayani, Mukund S. Chettiar, Amandeep Singh, Mubarak Shah, Deval Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2505.11454)  

**Abstract**: Large multimodal models (LMMs) now excel on many vision language benchmarks, however, they still struggle with human centered criteria such as fairness, ethics, empathy, and inclusivity, key to aligning with human values. We introduce HumaniBench, a holistic benchmark of 32K real-world image question pairs, annotated via a scalable GPT4o assisted pipeline and exhaustively verified by domain experts. HumaniBench evaluates seven Human Centered AI (HCAI) principles: fairness, ethics, understanding, reasoning, language inclusivity, empathy, and robustness, across seven diverse tasks, including open and closed ended visual question answering (VQA), multilingual QA, visual grounding, empathetic captioning, and robustness tests. Benchmarking 15 state of the art LMMs (open and closed source) reveals that proprietary models generally lead, though robustness and visual grounding remain weak points. Some open-source models also struggle to balance accuracy with adherence to human-aligned principles. HumaniBench is the first benchmark purpose built around HCAI principles. It provides a rigorous testbed for diagnosing alignment gaps and guiding LMMs toward behavior that is both accurate and socially responsible. Dataset, annotation prompts, and evaluation code are available at: this https URL 

**Abstract (ZH)**: 大规模多模态模型（LMMs）在许多视觉语言基准测试中已表现出色，但在公平性、伦理、同理心和包容性等以人为本的标准方面仍面临挑战，这些都是与人类价值观对齐的关键。我们引入了HumaniBench，这是一个由32K真实世界图像问题对组成的综合基准，通过可扩展的GPT4o辅助管道标注，并由领域专家详尽验证。HumaniBench评估了七个人本中心AI（HCAI）原则：公平性、伦理、理解、推理、语言包容性、同理心和鲁棒性，涵盖了七个不同任务，包括开放和封闭式的视觉问答（VQA）、多语言问答、视觉定位、同理心描述和鲁棒性测试。对15个最先进的LMMs（开源和封闭源）的评估显示，专有模型通常表现更好，但鲁棒性和视觉定位仍然是薄弱环节。一些开源模型也难以在准确性与遵循以人为本的原则之间找到平衡。HumaniBench是首个专门围绕HCAI原则构建的基准。它提供了一个严格的测试平台，用于诊断对齐差距，并指导LMMs向同时准确和负责任的行为发展。数据集、标注提示和评估代码可在以下链接获取：this https URL 

---
# Mergenetic: a Simple Evolutionary Model Merging Library 

**Title (ZH)**: Mergenetic：一种简化的进化模型集成功能库 

**Authors**: Adrian Robert Minut, Tommaso Mencattini, Andrea Santilli, Donato Crisostomi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2505.11427)  

**Abstract**: Model merging allows combining the capabilities of existing models into a new one - post hoc, without additional training. This has made it increasingly popular thanks to its low cost and the availability of libraries that support merging on consumer GPUs. Recent work shows that pairing merging with evolutionary algorithms can boost performance, but no framework currently supports flexible experimentation with such strategies in language models. We introduce Mergenetic, an open-source library for evolutionary model merging. Mergenetic enables easy composition of merging methods and evolutionary algorithms while incorporating lightweight fitness estimators to reduce evaluation costs. We describe its design and demonstrate that Mergenetic produces competitive results across tasks and languages using modest hardware. 

**Abstract (ZH)**: 模型合并允许将现有模型的能力合并到一个新的模型中——事后合并，无需额外训练。这种方法由于其低成本和可用的支持合并的库而在近期变得越来越流行。最近的研究表明，将合并与进化算法相结合可以提升性能，但目前还没有框架支持在语言模型中灵活地实验这类策略。我们介绍了Mergenetic，这是一个开源的进化模型合并库。Mergenetic简化了合并方法和进化算法的组合，并通过引入轻量级的适应度估算器来减少评估成本。我们描述了其设计，并证明了即使使用 modest 硬件，Mergenetic也能在任务和语言方面产生竞争力的结果。 

---
# MID-L: Matrix-Interpolated Dropout Layer with Layer-wise Neuron Selection 

**Title (ZH)**: MID-L: 基于层内神经元选择的矩阵插值dropout层 

**Authors**: Pouya Shaeri, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2505.11416)  

**Abstract**: Modern neural networks often activate all neurons for every input, leading to unnecessary computation and inefficiency. We introduce Matrix-Interpolated Dropout Layer (MID-L), a novel module that dynamically selects and activates only the most informative neurons by interpolating between two transformation paths via a learned, input-dependent gating vector. Unlike conventional dropout or static sparsity methods, MID-L employs a differentiable Top-k masking strategy, enabling per-input adaptive computation while maintaining end-to-end differentiability. MID-L is model-agnostic and integrates seamlessly into existing architectures. Extensive experiments on six benchmarks, including MNIST, CIFAR-10, CIFAR-100, SVHN, UCI Adult, and IMDB, show that MID-L achieves up to average 55\% reduction in active neurons, 1.7$\times$ FLOPs savings, and maintains or exceeds baseline accuracy. We further validate the informativeness and selectivity of the learned neurons via Sliced Mutual Information (SMI) and observe improved robustness under overfitting and noisy data conditions. Additionally, MID-L demonstrates favorable inference latency and memory usage profiles, making it suitable for both research exploration and deployment on compute-constrained systems. These results position MID-L as a general-purpose, plug-and-play dynamic computation layer, bridging the gap between dropout regularization and efficient inference. 

**Abstract (ZH)**: 现代神经网络常为每个输入激活所有神经元，导致不必要的计算和低效。我们引入了矩阵内插丢弃层（MID-L），这是一种新型模块，通过学习的输入依赖门控向量在两种变换路径之间进行内插，动态地选择并激活最具信息性的神经元。不同于传统的丢弃或静态稀疏性方法，MID-L 使用可微分的Top-k掩码策略，实现输入适配的计算同时保持端到端可微性。MID-L 兼容性强且无缝集成到现有架构中。在包括MNIST、CIFAR-10、CIFAR-100、SVHN、UCI Adult 和 IMDB在内的六个基准测试上进行的实验表明，MID-L 可以将活跃神经元减少多达平均55%，节省1.7倍的FLOPs，并且保持或超越基线准确度。此外，通过Sliced Mutual Information (SMI) 验证所学习的神经元的信息性和选择性，并观察到在过拟合和嘈杂数据条件下鲁棒性增强。另外，MID-L 展现了有利的推理延迟和内存使用性能，使其适用于计算受限系统的研究探索和部署。这些结果将MID-L 定位为一种通用型、即插即用的动态计算层，填补了丢弃正则化与高效推理之间的差距。 

---
# Visual Planning: Let's Think Only with Images 

**Title (ZH)**: 视觉规划：让我们仅凭图像思考。 

**Authors**: Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2505.11409)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations, independent of text. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising alternative to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）及其多模态扩展（MLLMs）在跨多样化任务中的机器推理方面取得了显著进展。然而，这些模型主要依赖纯文本作为表达和结构化推理的媒介，即使存在视觉信息也是如此。在本文中，我们argue认为，特别是在涉及空间和几何信息的任务中，语言可能并不是最自然或最有效的推理模态。受此启发，我们提出了一个新的范式——视觉规划，该范式通过纯视觉表示进行规划，独立于文本。在这个范式中，规划是通过序列图像来执行的，这些图像在视觉领域中编码逐步推理，类似于人类如何草图或可视化未来动作。我们介绍了一个由GRPO赋能的新强化学习框架——基于强化学习的视觉规划（VPRL），该框架极大地提高了在冰湖（FrozenLake）、迷宫（Maze）和迷你行为（MiniBehavior）等代表性视觉导航任务中的规划性能。我们的视觉规划范式在纯文本空间推理的所有其他变体中均表现出更优的性能。我们的研究结果确立了视觉规划作为语言基于推理的可行且有前景的替代方案的地位，为受益于直观、基于图像的推理的任务开辟了新的途径。 

---
# DecompileBench: A Comprehensive Benchmark for Evaluating Decompilers in Real-World Scenarios 

**Title (ZH)**: DecompileBench：一种全面的基准测试，用于评估实际场景中去编译器的性能 

**Authors**: Zeyu Gao, Yuxin Cui, Hao Wang, Siliang Qin, Yuanda Wang, Bolun Zhang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11340)  

**Abstract**: Decompilers are fundamental tools for critical security tasks, from vulnerability discovery to malware analysis, yet their evaluation remains fragmented. Existing approaches primarily focus on syntactic correctness through synthetic micro-benchmarks or subjective human ratings, failing to address real-world requirements for semantic fidelity and analyst usability. We present DecompileBench, the first comprehensive framework that enables effective evaluation of decompilers in reverse engineering workflows through three key components: \textit{real-world function extraction} (comprising 23,400 functions from 130 real-world programs), \textit{runtime-aware validation}, and \textit{automated human-centric assessment} using LLM-as-Judge to quantify the effectiveness of decompilers in reverse engineering workflows. Through a systematic comparison between six industrial-strength decompilers and six recent LLM-powered approaches, we demonstrate that LLM-based methods surpass commercial tools in code understandability despite 52.2% lower functionality correctness. These findings highlight the potential of LLM-based approaches to transform human-centric reverse engineering. We open source \href{this https URL}{DecompileBench} to provide a framework to advance research on decompilers and assist security experts in making informed tool selections based on their specific requirements. 

**Abstract (ZH)**: Decompilers在反向工程工作流中的全面评估：从功能提取到自动化人类为中心的评估 

---
# Explaining Strategic Decisions in Multi-Agent Reinforcement Learning for Aerial Combat Tactics 

**Title (ZH)**: 多智能体 reinforcement learning 在空战战术中的战略决策解释 

**Authors**: Ardian Selmonaj, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger, Matthias Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11311)  

**Abstract**: Artificial intelligence (AI) is reshaping strategic planning, with Multi-Agent Reinforcement Learning (MARL) enabling coordination among autonomous agents in complex scenarios. However, its practical deployment in sensitive military contexts is constrained by the lack of explainability, which is an essential factor for trust, safety, and alignment with human strategies. This work reviews and assesses current advances in explainability methods for MARL with a focus on simulated air combat scenarios. We proceed by adapting various explainability techniques to different aerial combat scenarios to gain explanatory insights about the model behavior. By linking AI-generated tactics with human-understandable reasoning, we emphasize the need for transparency to ensure reliable deployment and meaningful human-machine interaction. By illuminating the crucial importance of explainability in advancing MARL for operational defense, our work supports not only strategic planning but also the training of military personnel with insightful and comprehensible analyses. 

**Abstract (ZH)**: 人工智能（AI）正在重塑战略规划，多智能体强化学习（MARL）使自主智能体在复杂场景中的协作成为可能。然而，其在敏感军事环境中的实际部署受到可解释性的限制，可解释性是建立信任、保障安全和与人类策略一致的重要因素。本文回顾并评估了当前MARL可解释性方法的发展，重点关注模拟空战场景。通过将各种解释性技术应用于不同的空中作战场景，我们获得了关于模型行为的解释性洞察。通过将AI生成的战术与人类可理解的推理相结合，我们强调了透明度的重要性，以确保可靠的部署和有意义的人机交互。通过阐述可解释性在推动MARL在作战防御中的应用的重要性，我们的工作不仅支持战略规划，还通过提供深入且易懂的分析来培训军事人员。 

---
# Heterogeneity-Aware Client Sampling: A Unified Solution for Consistent Federated Learning 

**Title (ZH)**: 面向异质性的客户端采样： federated learning 的统一解决方案 

**Authors**: Shudi Weng, Chao Ren, Ming Xiao, Mikael Skoglund  

**Link**: [PDF](https://arxiv.org/pdf/2505.11304)  

**Abstract**: Federated learning (FL) commonly involves clients with diverse communication and computational capabilities. Such heterogeneity can significantly distort the optimization dynamics and lead to objective inconsistency, where the global model converges to an incorrect stationary point potentially far from the pursued optimum. Despite its critical impact, the joint effect of communication and computation heterogeneity has remained largely unexplored, due to the intrinsic complexity of their interaction. In this paper, we reveal the fundamentally distinct mechanisms through which heterogeneous communication and computation drive inconsistency in FL. To the best of our knowledge, this is the first unified theoretical analysis of general heterogeneous FL, offering a principled understanding of how these two forms of heterogeneity jointly distort the optimization trajectory under arbitrary choices of local solvers. Motivated by these insights, we propose Federated Heterogeneity-Aware Client Sampling, FedACS, a universal method to eliminate all types of objective inconsistency. We theoretically prove that FedACS converges to the correct optimum at a rate of $O(1/\sqrt{R})$, even in dynamic heterogeneous environments. Extensive experiments across multiple datasets show that FedACS outperforms state-of-the-art and category-specific baselines by 4.3%-36%, while reducing communication costs by 22%-89% and computation loads by 14%-105%, respectively. 

**Abstract (ZH)**: 联邦学习中异构通信和计算的联合效应及其理论分析：一种消除目标不一致性的联邦异构感知客户端采样方法 

---
# TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes 

**Title (ZH)**: TAIJI: 基于MCP的多模态数据湖上的数据分析 

**Authors**: Chao Zhang, Shaolei Zhang, Quehuan Liu, Sibei Chen, Tong Li, Ju Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11270)  

**Abstract**: The variety of data in data lakes presents significant challenges for data analytics, as data scientists must simultaneously analyze multi-modal data, including structured, semi-structured, and unstructured data. While Large Language Models (LLMs) have demonstrated promising capabilities, they still remain inadequate for multi-modal data analytics in terms of accuracy, efficiency, and freshness. First, current natural language (NL) or SQL-like query languages may struggle to precisely and comprehensively capture users' analytical intent. Second, relying on a single unified LLM to process diverse data modalities often leads to substantial inference overhead. Third, data stored in data lakes may be incomplete or outdated, making it essential to integrate external open-domain knowledge to generate timely and relevant analytics results.
In this paper, we envision a new multi-modal data analytics system. Specifically, we propose a novel architecture built upon the Model Context Protocol (MCP), an emerging paradigm that enables LLMs to collaborate with knowledgeable agents. First, we define a semantic operator hierarchy tailored for querying multi-modal data in data lakes and develop an AI-agent-powered NL2Operator translator to bridge user intent and analytical execution. Next, we introduce an MCP-based execution framework, in which each MCP server hosts specialized foundation models optimized for specific data modalities. This design enhances both accuracy and efficiency, while supporting high scalability through modular deployment. Finally, we propose a updating mechanism by harnessing the deep research and machine unlearning techniques to refresh the data lakes and LLM knowledges, with the goal of balancing the data freshness and inference efficiency. 

**Abstract (ZH)**: 多模态数据湖中数据的多样性为数据analytics带来了巨大挑战，数据科学家必须同时分析结构化、半结构化和非结构化等多种模态的数据。尽管大型语言模型（LLMs）显示出了潜在的能力，但在准确度、效率和新鲜度方面仍然无法满足多模态数据分析的需求。首先，当前的自然语言（NL）或SQL-like查询语言可能难以精确且全面地捕捉用户的数据分析意图。其次，依赖单一的统一LLM处理多样化模态的数据通常会导致显著的推理开销。最后，存储在数据湖中的数据可能不完整或过时，因此有必要整合开放领域知识以生成及时且相关的结果。

在本文中，我们设想了一种新的多模态数据分析系统。具体而言，我们提出了基于模型上下文协议（MCP）的新架构，这是一种新兴的范式，可以实现LLMs与知识型代理的协作。首先，我们定义了一个针对数据湖中多模态数据查询的语义操作符层次结构，并开发了一个基于AI代理的NL2Operator翻译器，以连接用户意图与分析执行。其次，我们介绍了基于MCP的执行框架，在该框架中，每个MCP服务器托管针对特定数据模态优化的基础模型。这一设计提高了准确性和效率，并通过模块化部署支持高可扩展性。最后，我们提出了一种更新机制，利用深度研究和机器遗忘技术来刷新数据湖和LLM的知识，旨在平衡数据新鲜度和推理效率。 

---
# Equal is Not Always Fair: A New Perspective on Hyperspectral Representation Non-Uniformity 

**Title (ZH)**: 公平并不总是公正：超谱表示非均匀性的新视角 

**Authors**: Wuzhou Quan, Mingqiang Wei, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11267)  

**Abstract**: Hyperspectral image (HSI) representation is fundamentally challenged by pervasive non-uniformity, where spectral dependencies, spatial continuity, and feature efficiency exhibit complex and often conflicting behaviors. Most existing models rely on a unified processing paradigm that assumes homogeneity across dimensions, leading to suboptimal performance and biased representations. To address this, we propose FairHyp, a fairness-directed framework that explicitly disentangles and resolves the threefold non-uniformity through cooperative yet specialized modules. We introduce a Runge-Kutta-inspired spatial variability adapter to restore spatial coherence under resolution discrepancies, a multi-receptive field convolution module with sparse-aware refinement to enhance discriminative features while respecting inherent sparsity, and a spectral-context state space model that captures stable and long-range spectral dependencies via bidirectional Mamba scanning and statistical aggregation. Unlike one-size-fits-all solutions, FairHyp achieves dimension-specific adaptation while preserving global consistency and mutual reinforcement. This design is grounded in the view that non-uniformity arises from the intrinsic structure of HSI representations, rather than any particular task setting. To validate this, we apply FairHyp across four representative tasks including classification, denoising, super-resolution, and inpaintin, demonstrating its effectiveness in modeling a shared structural flaw. Extensive experiments show that FairHyp consistently outperforms state-of-the-art methods under varied imaging conditions. Our findings redefine fairness as a structural necessity in HSI modeling and offer a new paradigm for balancing adaptability, efficiency, and fidelity in high-dimensional vision tasks. 

**Abstract (ZH)**: 基于公平性的超光谱图像表示框架：解决普遍存在的非均匀性问题 

---
# A Set-Sequence Model for Time Series 

**Title (ZH)**: 时间序列的集序列模型 

**Authors**: Elliot L. Epstein, Apaar Sadhwani, Kay Giesecke  

**Link**: [PDF](https://arxiv.org/pdf/2505.11243)  

**Abstract**: In many financial prediction problems, the behavior of individual units (such as loans, bonds, or stocks) is influenced by observable unit-level factors and macroeconomic variables, as well as by latent cross-sectional effects. Traditional approaches attempt to capture these latent effects via handcrafted summary features. We propose a Set-Sequence model that eliminates the need for handcrafted features. The Set model first learns a shared cross-sectional summary at each period. The Sequence model then ingests the summary-augmented time series for each unit independently to predict its outcome. Both components are learned jointly over arbitrary sets sampled during training. Our approach harnesses the set nature of the cross-section and is computationally efficient, generating set summaries in linear time relative to the number of units. It is also flexible, allowing the use of existing sequence models and accommodating a variable number of units at inference. Empirical evaluations demonstrate that our Set-Sequence model significantly outperforms benchmarks on stock return prediction and mortgage behavior tasks. Code will be released. 

**Abstract (ZH)**: 在许多金融预测问题中，个体单位（如贷款、债券或股票）的行为受到可观测的单位级因素、宏观经济学变量以及潜在的横截面效应的影响。传统方法试图通过手工设计的摘要特征来捕捉这些潜在效应。我们提出了一个Set-Sequence模型，以消除手工设计特征的需求。Set模型首先在每个时期学习一个共享的横截面摘要。Sequence模型然后独立地摄取每个单位的摘要增强时间序列，以预测其结果。两个组件在训练期间任意采样的集合中联合学习。我们的方法利用了横截面的集合并行性质，计算效率高，生成集合摘要的时间复杂度与单位数量成线性关系。它还具有灵活性，允许使用现有的序列模型，并在推断时容纳数量可变的单位。实证评估表明，我们的Set-Sequence模型在股票回报预测和抵押行为任务中显著优于基准模型。代码将开源。 

---
# Bayesian Hierarchical Invariant Prediction 

**Title (ZH)**: 贝叶斯层次不变预测 

**Authors**: Francisco Madaleno, Pernille Julie Viuff Sand, Francisco C. Pereira, Sergio Hernan Garrido Mejia  

**Link**: [PDF](https://arxiv.org/pdf/2505.11211)  

**Abstract**: We propose Bayesian Hierarchical Invariant Prediction (BHIP) reframing Invariant Causal Prediction (ICP) through the lens of Hierarchical Bayes. We leverage the hierarchical structure to explicitly test invariance of causal mechanisms under heterogeneous data, resulting in improved computational scalability for a larger number of predictors compared to ICP. Moreover, given its Bayesian nature BHIP enables the use of prior information. In this paper, we test two sparsity inducing priors: horseshoe and spike-and-slab, both of which allow us a more reliable identification of causal features. We test BHIP in synthetic and real-world data showing its potential as an alternative inference method to ICP. 

**Abstract (ZH)**: Bayesian Hierarchical Invariant Prediction: Reframing Invariant Causal Prediction Through the Lens of Hierarchical Bayes 

---
# RanDeS: Randomized Delta Superposition for Multi-Model Compression 

**Title (ZH)**: RanDeS: 随机Delta 超position 多模型压缩 

**Authors**: Hangyu Zhou, Aaron Gokaslan, Volodymyr Kuleshov, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11204)  

**Abstract**: From a multi-model compression perspective, model merging enables memory-efficient serving of multiple models fine-tuned from the same base, but suffers from degraded performance due to interference among their task-specific parameter adjustments (i.e., deltas). In this paper, we reformulate model merging as a compress-and-retrieve scheme, revealing that the task interference arises from the summation of irrelevant deltas during model retrieval. To address this issue, we use random orthogonal transformations to decorrelate these vectors into self-cancellation. We show that this approach drastically reduces interference, improving performance across both vision and language tasks. Since these transformations are fully defined by random seeds, adding new models requires no extra memory. Further, their data- and model-agnostic nature enables easy addition or removal of models with minimal compute overhead, supporting efficient and flexible multi-model serving. 

**Abstract (ZH)**: 从多模型压缩视角出发，模型合并能够在不牺牲太多性能的情况下高效服务于多个来自同一基础模型的精调模型，但任务特定参数调整（即差异）之间的相互干扰会导致性能下降。在本文中，我们将模型合并重新表述为一种压缩和检索方案，揭示了任务干扰源于模型检索过程中无关差异的叠加。为了解决这一问题，我们使用随机正交变换将这些向量去相关化，使其自我抵消。我们证明了这种方法大幅减少了干扰，提高了视觉和语言任务的性能。由于这些变换完全由随机种子定义，增加新模型无需额外内存。此外，它们的数据和模型无关性使得在最小计算开销下轻松添加或移除模型，支持高效的多模型服务。 

---
# User-centric Music Recommendations 

**Title (ZH)**: 用户中心的音乐推荐 

**Authors**: Jaime Ramirez Castillo, M. Julia Flores, Ann E. Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11198)  

**Abstract**: This work presents a user-centric recommendation framework, designed as a pipeline with four distinct, connected, and customizable phases. These phases are intended to improve explainability and boost user engagement.
We have collected the historical this http URL track playback records of a single user over approximately 15 years. The collected dataset includes more than 90,000 playbacks and approximately 14,000 unique tracks.
From track playback records, we have created a dataset of user temporal contexts (each row is a specific moment when the user listened to certain music descriptors). As music descriptors, we have used community-contributed this http URL tags and Spotify audio features. They represent the music that, throughout years, the user has been listening to.
Next, given the most relevant this http URL tags of a moment (e.g. the hour of the day), we predict the Spotify audio features that best fit the user preferences in that particular moment. Finally, we use the predicted audio features to find tracks similar to these features. The final aim is to recommend (and discover) tracks that the user may feel like listening to at a particular moment.
For our initial study case, we have chosen to predict only a single audio feature target: danceability. The framework, however, allows to include more target variables.
The ability to learn the musical habits from a single user can be quite powerful, and this framework could be extended to other users. 

**Abstract (ZH)**: 用户中心的推荐框架：基于四阶段可定制流水线的音乐推荐与解释性提升 

---
# Imputation-free and Alignment-free: Incomplete Multi-view Clustering Driven by Consensus Semantic Learning 

**Title (ZH)**: 无填充且无对齐：基于共识语义学习的不完整多视图聚类 

**Authors**: Yuzhuo Dai, Jiaqi Jin, Zhibin Dong, Siwei Wang, Xinwang Liu, En Zhu, Xihong Yang, Xinbiao Gan, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11182)  

**Abstract**: In incomplete multi-view clustering (IMVC), missing data induce prototype shifts within views and semantic inconsistencies across views. A feasible solution is to explore cross-view consistency in paired complete observations, further imputing and aligning the similarity relationships inherently shared across views. Nevertheless, existing methods are constrained by two-tiered limitations: (1) Neither instance- nor cluster-level consistency learning construct a semantic space shared across views to learn consensus semantics. The former enforces cross-view instances alignment, and wrongly regards unpaired observations with semantic consistency as negative pairs; the latter focuses on cross-view cluster counterparts while coarsely handling fine-grained intra-cluster relationships within views. (2) Excessive reliance on consistency results in unreliable imputation and alignment without incorporating view-specific cluster information. Thus, we propose an IMVC framework, imputation- and alignment-free for consensus semantics learning (FreeCSL). To bridge semantic gaps across all observations, we learn consensus prototypes from available data to discover a shared space, where semantically similar observations are pulled closer for consensus semantics learning. To capture semantic relationships within specific views, we design a heuristic graph clustering based on modularity to recover cluster structure with intra-cluster compactness and inter-cluster separation for cluster semantics enhancement. Extensive experiments demonstrate, compared to state-of-the-art competitors, FreeCSL achieves more confident and robust assignments on IMVC task. 

**Abstract (ZH)**: 不完备多视图聚类中缺失数据导致视图内原型偏移和视图间语义不一致。可行的解决方案是在配对的完整观测中探索跨视图一致性，进一步推导和对齐跨视图固有的相似关系。然而，现有方法受到两类限制：（1）实例级和聚类级的一致性学习构造未能建立跨视图共享的语义空间来学习一致语义。前者强加跨视图实例对齐，错误地将具有语义一致性的未配对观测视为负样本对；后者侧重于跨视图聚类对应物，而粗糙地处理视图内部聚类的细粒度关系。（2）过度依赖一致性导致在不结合视图特定聚类信息的情况下进行不可靠的推导和对齐。因此，我们提出了一种用于共识语义学习的无推导和对齐框架（FreeCSL）。为跨越所有观测填补语义差距，我们从可用数据中学习共识原型，发现一个共享空间，在此空间中，语义相似的观测被拉近以进行共识语义学习。为了捕捉特定视图内的语义关系，我们基于模块性的启发式图聚类设计，以恢复具有内部紧凑性和外部分离性的聚类结构，提高聚类语义。大量实验表明，与最先进的竞争对手相比，FreeCSL在不完备多视图聚类任务中实现了更自信和稳健的分配。 

---
# Low-Resource Language Processing: An OCR-Driven Summarization and Translation Pipeline 

**Title (ZH)**: 低资源语言处理：一种基于OCR的总结与翻译管道 

**Authors**: Hrishit Madhavi, Jacob Cherian, Yuvraj Khamkar, Dhananjay Bhagat  

**Link**: [PDF](https://arxiv.org/pdf/2505.11177)  

**Abstract**: This paper presents an end-to-end suite for multilingual information extraction and processing from image-based documents. The system uses Optical Character Recognition (Tesseract) to extract text in languages such as English, Hindi, and Tamil, and then a pipeline involving large language model APIs (Gemini) for cross-lingual translation, abstractive summarization, and re-translation into a target language. Additional modules add sentiment analysis (TensorFlow), topic classification (Transformers), and date extraction (Regex) for better document comprehension. Made available in an accessible Gradio interface, the current research shows a real-world application of libraries, models, and APIs to close the language gap and enhance access to information in image media across different linguistic environments 

**Abstract (ZH)**: 这篇论文提出了一套端到端的多语言信息从图像文档中提取和处理方案。该系统使用光学字符识别(Tesseract)从英语、印地语和泰米尔语等语言的图像文档中提取文本，然后通过大型语言模型APIs(Gemini)进行跨语言翻译、抽象总结以及目标语言的再次翻译。此外，还增加了情感分析（TensorFlow）、主题分类（Transformers）和日期提取（Regex）模块以改善文档理解。该系统通过可访问的Gradio界面提供，当前的研究展示了如何利用库、模型和API在不同语言环境中缩小语言差距并增强图像媒体中的信息访问。 

---
# Maximizing Asynchronicity in Event-based Neural Networks 

**Title (ZH)**: 基于事件的神经网络中最大化异步性 

**Authors**: Haiqing Hao, Nikola Zubić, Weihua He, Zhipeng Sui, Davide Scaramuzza, Wenhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11165)  

**Abstract**: Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned representations for ML pipelines, existing A2S approaches often sacrifice representation expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous representation learning), a novel A2S framework to generate highly expressive and generalizable event-by-event representations. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a remarkable 47.7 mAP on the Gen1 dataset. These results underscore EVA's transformative potential for advancing real-time event-based vision applications. 

**Abstract (ZH)**: 事件相机通过提供高时间分辨率、低延迟和最小冗余的视觉数据，虽然其异步、稀疏的序列特性挑战了标准张量基机器学习方法，但仍存在瓶颈。尽管近期提出的异步到同步（A2S）范式旨在通过异步地将事件编码为学习表示以适应机器学习管道，但现有A2S方法通常在表达性和泛化能力上逊色于密集同步方法。本文介绍了一种新的A2S框架EVA（EVent Asynchronous representation learning），用于生成高度表达性和泛化能力的逐事件表示。受事件与语言之间类比的启发，EVA独特地采用了线性注意力和自我监督学习在构建中的进步。实验展示中，EVA在识别任务（DVS128-Gesture和N-Cars）上优于先前的A2S方法，并是第一个成功掌握 demanding 检测任务的A2S框架，实现了Gen1数据集上的47.7 mAP。这些结果强调了EVA在推动实时事件驱动视觉应用方面具有颠覆性的潜力。 

---
# FairSHAP: Preprocessing for Fairness Through Attribution-Based Data Augmentation 

**Title (ZH)**: 基于归因数据增强的公平性预处理：FairSHAP 

**Authors**: Lin Zhu, Yijun Bian, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2505.11111)  

**Abstract**: Ensuring fairness in machine learning models is critical, particularly in high-stakes domains where biased decisions can lead to serious societal consequences. Existing preprocessing approaches generally lack transparent mechanisms for identifying which features or instances are responsible for unfairness. This obscures the rationale behind data modifications. We introduce FairSHAP, a novel pre-processing framework that leverages Shapley value attribution to improve both individual and group fairness. FairSHAP identifies fairness-critical instances in the training data using an interpretable measure of feature importance, and systematically modifies them through instance-level matching across sensitive groups. This process reduces discriminative risk - an individual fairness metric - while preserving data integrity and model accuracy. We demonstrate that FairSHAP significantly improves demographic parity and equality of opportunity across diverse tabular datasets, achieving fairness gains with minimal data perturbation and, in some cases, improved predictive performance. As a model-agnostic and transparent method, FairSHAP integrates seamlessly into existing machine learning pipelines and provides actionable insights into the sources of this http URL code is on this https URL. 

**Abstract (ZH)**: 确保机器学习模型的公平性至关重要，特别是在高风险领域，偏见决策可能导致严重社会后果。现有的预处理方法通常缺乏透明的机制来识别导致不公平的特征或实例，这掩盖了数据修改的原因。我们提出了FairSHAP，一种新颖的预处理框架，利用Shapley值归因来改进个体公平性和群体公平性。FairSHAP使用可解释的特征重要性度量来识别训练数据中的公平关键实例，并通过敏感群体的实例级匹配系统地对其进行修改。这一过程减少了歧视性风险（一种个体公平性指标），同时保持了数据完整性和模型准确性。我们证明，FairSHAP在多种表格数据集中显著提高了人口统计正义和平等的机会，并通过最少的数据扰动实现了公平性收益，在某些情况下还提高了预测性能。作为一种模型无关且透明的方法，FairSHAP可无缝集成到现有的机器学习管道中，并提供有关这一http URL的可操作见解。代码可在此https://github.com/fairshap-team/FairSHAP 获取。 

---
# Inferring the Most Similar Variable-length Subsequences between Multidimensional Time Series 

**Title (ZH)**: 多维时间序列中最相似变长子序列的推断 

**Authors**: Thanadej Rattanakornphan, Piyanon Charoenpoonpanich, Chainarong Amornbunchornvej  

**Link**: [PDF](https://arxiv.org/pdf/2505.11106)  

**Abstract**: Finding the most similar subsequences between two multidimensional time series has many applications: e.g. capturing dependency in stock market or discovering coordinated movement of baboons. Considering one pattern occurring in one time series, we might be wondering whether the same pattern occurs in another time series with some distortion that might have a different length. Nevertheless, to the best of our knowledge, there is no efficient framework that deals with this problem yet. In this work, we propose an algorithm that provides the exact solution of finding the most similar multidimensional subsequences between time series where there is a difference in length both between time series and between subsequences. The algorithm is built based on theoretical guarantee of correctness and efficiency. The result in simulation datasets illustrated that our approach not just only provided correct solution, but it also utilized running time only quarter of time compared against the baseline approaches. In real-world datasets, it extracted the most similar subsequences even faster (up to 20 times faster against baseline methods) and provided insights regarding the situation in stock market and following relations of multidimensional time series of baboon movement. Our approach can be used for any time series. The code and datasets of this work are provided for the public use. 

**Abstract (ZH)**: 在多维时间序列之间寻找最相似子序列的研究：从股票市场依赖性到狒狒协调运动的发现 

---
# Bidirectional Distillation: A Mixed-Play Framework for Multi-Agent Generalizable Behaviors 

**Title (ZH)**: 双向 distillation: 一种多智能体通用行为的混合博弈框架 

**Authors**: Lang Feng, Jiahao Lin, Dong Xing, Li Zhang, De Ma, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11100)  

**Abstract**: Population-population generalization is a challenging problem in multi-agent reinforcement learning (MARL), particularly when agents encounter unseen co-players. However, existing self-play-based methods are constrained by the limitation of inside-space generalization. In this study, we propose Bidirectional Distillation (BiDist), a novel mixed-play framework, to overcome this limitation in MARL. BiDist leverages knowledge distillation in two alternating directions: forward distillation, which emulates the historical policies' space and creates an implicit self-play, and reverse distillation, which systematically drives agents towards novel distributions outside the known policy space in a non-self-play manner. In addition, BiDist operates as a concise and efficient solution without the need for the complex and costly storage of past policies. We provide both theoretical analysis and empirical evidence to support BiDist's effectiveness. Our results highlight its remarkable generalization ability across a variety of cooperative, competitive, and social dilemma tasks, and reveal that BiDist significantly diversifies the policy distribution space. We also present comprehensive ablation studies to reinforce BiDist's effectiveness and key success factors. Source codes are available in the supplementary material. 

**Abstract (ZH)**: 人口-人口泛化是多智能体强化学习（MARL）中的一个挑战性问题，特别是在智能体遇到未见过的合作者时。然而，现有的基于自我对弈的方法受到内部空间泛化的限制。本研究提出了一种新颖的双向灌输（BiDist）框架，以克服MARL中的这一限制。BiDist 利用双向灌输：前向灌输模拟历史策略空间并创建隐式自我对弈，反向灌输系统地引导智能体向已知策略空间之外的新分布发展，不采用自我对弈方式。此外，BiDist 作为一个简洁高效的方法，无需复杂且成本高昂的过去策略存储。我们提供了理论分析和实验证据来支持BiDist的有效性。我们的结果强调了BiDist在其合作、竞争和社会困境任务上的显著泛化能力，并揭示了BiDist显著多样化了策略分布空间。我们还进行了全面的消融研究以增强BiDist的有效性和关键成功因素。附带代码详见补充材料。 

---
# A Fast Kernel-based Conditional Independence test with Application to Causal Discovery 

**Title (ZH)**: 一种基于核的方法的快速条件独立性检验及其在因果发现中的应用 

**Authors**: Oliver Schacht, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11085)  

**Abstract**: Kernel-based conditional independence (KCI) testing is a powerful nonparametric method commonly employed in causal discovery tasks. Despite its flexibility and statistical reliability, cubic computational complexity limits its application to large datasets. To address this computational bottleneck, we propose \textit{FastKCI}, a scalable and parallelizable kernel-based conditional independence test that utilizes a mixture-of-experts approach inspired by embarrassingly parallel inference techniques for Gaussian processes. By partitioning the dataset based on a Gaussian mixture model over the conditioning variables, FastKCI conducts local KCI tests in parallel, aggregating the results using an importance-weighted sampling scheme. Experiments on synthetic datasets and benchmarks on real-world production data validate that FastKCI maintains the statistical power of the original KCI test while achieving substantial computational speedups. FastKCI thus represents a practical and efficient solution for conditional independence testing in causal inference on large-scale data. 

**Abstract (ZH)**: 基于核的条件独立性（KCI）测试是一种在因果发现任务中广泛应用的强非参数方法。尽管其具有灵活性和统计可靠性，但三次方的计算复杂度限制了其在大规模数据集上的应用。为解决这一计算瓶颈，我们提出了一种名为FastKCI的可扩展且并行化的基于核的条件独立性测试方法，该方法受到高斯过程的尴尬并行推理技术启发，采用混合专家方法。通过基于条件变量的高斯混合模型对数据集进行分区，FastKCI在局部并行执行KCI测试，并通过重要性加权采样方案汇总结果。实验结果在合成数据集和实际生产数据集上的基准测试验证了FastKCI在保持原始KCI测试统计功效的同时，实现了显著的计算加速。FastKCI因此为大规模数据上的条件独立性测试提供了一个实用且高效的解决方案。 

---
# Fault Diagnosis across Heterogeneous Domains via Self-Adaptive Temporal-Spatial Attention and Sample Generation 

**Title (ZH)**: 跨异构域故障诊断：基于自适应时空注意力和样本生成的方法 

**Authors**: Guangqiang Li, M. Amine Atoui, Xiangshun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11083)  

**Abstract**: Deep learning methods have shown promising performance in fault diagnosis for multimode process. Most existing studies assume that the collected health state categories from different operating modes are identical. However, in real industrial scenarios, these categories typically exhibit only partial overlap. The incompleteness of the available data and the large distributional differences between the operating modes pose a significant challenge to existing fault diagnosis methods. To address this problem, a novel fault diagnosis model named self-adaptive temporal-spatial attention network (TSA-SAN) is proposed. First, inter-mode mappings are constructed using healthy category data to generate multimode samples. To enrich the diversity of the fault data, interpolation is performed between healthy and fault samples. Subsequently, the fault diagnosis model is trained using real and generated data. The self-adaptive instance normalization is established to suppress irrelevant information while retaining essential statistical features for diagnosis. In addition, a temporal-spatial attention mechanism is constructed to focus on the key features, thus enhancing the generalization ability of the model. The extensive experiments demonstrate that the proposed model significantly outperforms the state-of-the-art methods. The code will be available on Github at this https URL. 

**Abstract (ZH)**: 基于自适应时序空域注意力网络的多模式过程故障诊断方法 

---
# Assessing the Performance of Analog Training for Transfer Learning 

**Title (ZH)**: 评估模拟训练在迁移学习中的性能 

**Authors**: Omobayode Fagbohungbe, Corey Lammie, Malte J. Rasch, Takashi Ando, Tayfun Gokmen, Vijay Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11067)  

**Abstract**: Analog in-memory computing is a next-generation computing paradigm that promises fast, parallel, and energy-efficient deep learning training and transfer learning (TL). However, achieving this promise has remained elusive due to a lack of suitable training algorithms. Analog memory devices exhibit asymmetric and non-linear switching behavior in addition to device-to-device variation, meaning that most, if not all, of the current off-the-shelf training algorithms cannot achieve good training outcomes. Also, recently introduced algorithms have enjoyed limited attention, as they require bi-directionally switching devices of unrealistically high symmetry and precision and are highly sensitive. A new algorithm chopped TTv2 (c-TTv2), has been introduced, which leverages the chopped technique to address many of the challenges mentioned above. In this paper, we assess the performance of the c-TTv2 algorithm for analog TL using a Swin-ViT model on a subset of the CIFAR100 dataset. We also investigate the robustness of our algorithm to changes in some device specifications, including weight transfer noise, symmetry point skew, and symmetry point variability 

**Abstract (ZH)**: 模拟内存计算是一种下一代计算范式，有望实现快速、并行和节能的深度学习训练和迁移学习（TL）。然而，由于缺乏合适的训练算法，这一承诺尚未实现。模拟内存设备表现出非对称和非线性的开关行为，同时还存在器件间的差异，这意味着当前大多数甚至所有现成的训练算法都无法达到良好的训练效果。另外，最近引入的一些算法也受到了限制，因为它们需要具有不切实际的高对称性和精度的双向切换设备，并且非常敏感。一种新的算法——截断TTv2（c-TTv2）——已被提出，它利用截断技术来解决上述许多挑战。在本文中，我们使用Swin-ViT模型在CIFAR100数据集的部分子集上评估c-TTv2算法在模拟迁移学习中的性能。我们还研究了我们的算法对某些器件规格变化的鲁棒性，包括权重转移噪声、对称点偏斜和对称点变异性。 

---
# CUBIC: Concept Embeddings for Unsupervised Bias Identification using VLMs 

**Title (ZH)**: CUBIC: 无监督偏见识别的概念嵌入使用大语言模型 

**Authors**: David Méndez, Gianpaolo Bontempo, Elisa Ficarra, Roberto Confalonieri, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2505.11060)  

**Abstract**: Deep vision models often rely on biases learned from spurious correlations in datasets. To identify these biases, methods that interpret high-level, human-understandable concepts are more effective than those relying primarily on low-level features like heatmaps. A major challenge for these concept-based methods is the lack of image annotations indicating potentially bias-inducing concepts, since creating such annotations requires detailed labeling for each dataset and concept, which is highly labor-intensive. We present CUBIC (Concept embeddings for Unsupervised Bias IdentifiCation), a novel method that automatically discovers interpretable concepts that may bias classifier behavior. Unlike existing approaches, CUBIC does not rely on predefined bias candidates or examples of model failures tied to specific biases, as such information is not always available. Instead, it leverages image-text latent space and linear classifier probes to examine how the latent representation of a superclass label$\unicode{x2014}$shared by all instances in the dataset$\unicode{x2014}$is influenced by the presence of a given concept. By measuring these shifts against the normal vector to the classifier's decision boundary, CUBIC identifies concepts that significantly influence model predictions. Our experiments demonstrate that CUBIC effectively uncovers previously unknown biases using Vision-Language Models (VLMs) without requiring the samples in the dataset where the classifier underperforms or prior knowledge of potential biases. 

**Abstract (ZH)**: 基于概念的无监督偏见识别：CUBIC（概念嵌入用于无监督偏见识别） 

---
# Halting Recurrent GNNs and the Graded $μ$-Calculus 

**Title (ZH)**: 停止递归GNNs与分级μ-演算 

**Authors**: Jeroen Bollen, Jan Van den Bussche, Stijn Vansummeren, Jonni Virtema  

**Link**: [PDF](https://arxiv.org/pdf/2505.11050)  

**Abstract**: Graph Neural Networks (GNNs) are a class of machine-learning models that operate on graph-structured data. Their expressive power is intimately related to logics that are invariant under graded bisimilarity. Current proposals for recurrent GNNs either assume that the graph size is given to the model, or suffer from a lack of termination guarantees. In this paper, we propose a halting mechanism for recurrent GNNs. We prove that our halting model can express all node classifiers definable in graded modal mu-calculus, even for the standard GNN variant that is oblivious to the graph size. A recent breakthrough in the study of the expressivity of graded modal mu-calculus in the finite suggests that conversely, restricted to node classifiers definable in monadic second-order logic, recurrent GNNs can express only node classifiers definable in graded modal mu-calculus. To prove our main result, we develop a new approximate semantics for graded mu-calculus, which we believe to be of independent interest. We leverage this new semantics into a new model-checking algorithm, called the counting algorithm, which is oblivious to the graph size. In a final step we show that the counting algorithm can be implemented on a halting recurrent GNN. 

**Abstract (ZH)**: 循环图神经网络中止机制的研究：基于分级模μ演算的节点分类表达能力 

---
# CleanPatrick: A Benchmark for Image Data Cleaning 

**Title (ZH)**: CleanPatrick：图像数据清洗基准 

**Authors**: Fabian Gröger, Simone Lionetti, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Ludovic Amruthalingam, Elisabeth Victoria Goessinger, Hanna Lindemann, Marie Bargiela, Marie Hofbauer, Omar Badri, Philipp Tschandl, Arash Koochek, Matthew Groh, Alexander A. Navarini, Marc Pouly  

**Link**: [PDF](https://arxiv.org/pdf/2505.11034)  

**Abstract**: Robust machine learning depends on clean data, yet current image data cleaning benchmarks rely on synthetic noise or narrow human studies, limiting comparison and real-world relevance. We introduce CleanPatrick, the first large-scale benchmark for data cleaning in the image domain, built upon the publicly available Fitzpatrick17k dermatology dataset. We collect 496,377 binary annotations from 933 medical crowd workers, identify off-topic samples (4%), near-duplicates (21%), and label errors (22%), and employ an aggregation model inspired by item-response theory followed by expert review to derive high-quality ground truth. CleanPatrick formalizes issue detection as a ranking task and adopts typical ranking metrics mirroring real audit workflows. Benchmarking classical anomaly detectors, perceptual hashing, SSIM, Confident Learning, NoiseRank, and SelfClean, we find that, on CleanPatrick, self-supervised representations excel at near-duplicate detection, classical methods achieve competitive off-topic detection under constrained review budgets, and label-error detection remains an open challenge for fine-grained medical classification. By releasing both the dataset and the evaluation framework, CleanPatrick enables a systematic comparison of image-cleaning strategies and paves the way for more reliable data-centric artificial intelligence. 

**Abstract (ZH)**: Robust机器学习依赖于干净的数据，然而当前的图像数据清洁基准主要依赖于合成噪声或狭窄的人类研究，限制了比较和现实世界的相关性。我们引入了CleanPatrick，这是首个针对图像领域数据清洁的大规模基准，基于公开可用的Fitzpatrick17k皮肤病学数据集构建。我们收集了933名医学众包工作者的496,377个二元注释，识别出离题样本（4%）、近重复样本（21%）和标签错误（22%），并采用受项目反应理论启发的聚合模型结合专家审核，获得高质量的ground truth。CleanPatrick将问题检测形式化为一个排名任务，并采用常见排名指标来模拟实际审查工作流程。通过基准测试经典异常检测方法、感知哈希、SSIM、自信学习、NoiseRank和SelfClean，我们发现在CleanPatrick上，自我监督表示在近重复检测方面表现出色，经典方法在有限的审查预算下实现了竞争性的离题检测，而标签错误检测仍然是细粒度医疗分类的开放挑战。通过发布该数据集和评估框架，CleanPatrick使得图像清洁策略的系统比较成为可能，并为更可靠的数据为中心的人工智能铺平道路。 

---
# The heteronomy of algorithms: Traditional knowledge and computational knowledge 

**Title (ZH)**: 算法的非自律性：传统知识与计算知识 

**Authors**: David M. Berry  

**Link**: [PDF](https://arxiv.org/pdf/2505.11030)  

**Abstract**: If an active citizen should increasingly be a computationally enlightened one, replacing the autonomy of reason with the heteronomy of algorithms, then I argue in this article that we must begin teaching the principles of critiquing the computal through new notions of what we might call digital Bildung. Indeed, if civil society itself is mediated by computational systems and media, the public use of reason must also be complemented by skills for negotiating and using these computal forms to articulate such critique. Not only is there a need to raise the intellectual tone regarding computation and its related softwarization processes, but there is an urgent need to attend to the likely epistemic challenges from computation which, as presently constituted, tends towards justification through a philosophy of utility rather than through a philosophy of care for the territory of the intellect. We therefore need to develop an approach to this field that uses concepts and methods drawn from philosophy, politics, history, anthropology, sociology, media studies, computer science, and the humanities more generally, to try to understand these issues - particularly the way in which software and data increasingly penetrate our everyday life and the pressures and fissures that are created. We must, in other words, move to undertake a critical interdisciplinary research program to understand the way in which these systems are created, instantiated, and normatively engendered in both specific and general contexts. 

**Abstract (ZH)**: 如果活跃公民应越来越多地成为一个具备计算素养的公民，并且算法自治取代了理性自治，那么本文认为我们必须开始通过新观念教授批判计算的原则，即所谓的数字博登教育。如果公民社会本身被计算系统和媒体中介化，那么公共理性的使用也必须通过协商和使用这些计算形式来阐述这样的批判。不仅需要提高关于计算及其相关软计算化进程的思想水平，而且还需要关注计算可能带来的认识论挑战，目前的计算倾向于通过实用主义哲学而非关怀哲学来寻求正当性。因此，我们需要开发一种方法，从哲学、政治学、历史学、人类学、 sociology、媒体研究、计算机科学和更广泛的文科领域借用概念和方法，以理解这些议题——特别是软件和数据如何越来越多地渗透到我们的日常生活以及由此产生的压力和裂隙。换句话说，我们必须开展一项批判性的跨学科研究项目，以理解这些系统的创建、实现及其在特定和一般背景下的规范性起源。 

---
# StRuCom: A Novel Dataset of Structured Code Comments in Russian 

**Title (ZH)**: StRuCom: 一种新的俄语结构化代码注释数据集 

**Authors**: Maria Dziuba, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11026)  

**Abstract**: Structured code comments in docstring format are essential for code comprehension and maintenance, but existing machine learning models for their generation perform poorly for Russian compared to English. To bridge this gap, we present StRuCom - the first large-scale dataset (153K examples) specifically designed for Russian code documentation. Unlike machine-translated English datasets that distort terminology (e.g., technical loanwords vs. literal translations) and docstring structures, StRuCom combines human-written comments from Russian GitHub repositories with synthetically generated ones, ensuring compliance with Python, Java, JavaScript, C#, and Go standards through automated validation. Fine-tuning Qwen2.5-Coder models (0.5B-7B) on StRuCom shows statistically significant improvements of chrf++ and BERTScore over baseline models. 

**Abstract (ZH)**: 结构化的代码注释以docstring格式对于代码理解和维护至关重要，但现有的生成俄罗斯语代码注释的机器学习模型在效果上逊于英语。为了弥合这一差距，我们提出了StRuCom——第一个专门针对俄罗斯代码文档的大规模数据集（包含153,000个示例）。不同于通过机器翻译生成的英语数据集可能会扭曲术语（例如技术借词与直译的区别）和docstring结构，StRuCom将来自俄罗斯GitHub仓库的人工编写注释与合成生成的注释相结合，并通过自动化验证确保其符合Python、Java、JavaScript、C#和Go的标准。在StRuCom上微调Qwen2.5-Coder模型（0.5B至7B参数）显示了在chrf++和BERTScore指标上的统计显著改进。 

---
# Space Group Equivariant Crystal Diffusion 

**Title (ZH)**: 空间群共变晶体扩散 

**Authors**: Rees Chang, Angela Pak, Alex Guerra, Ni Zhan, Nick Richardson, Elif Ertekin, Ryan P. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2505.10994)  

**Abstract**: Accelerating inverse design of crystalline materials with generative models has significant implications for a range of technologies. Unlike other atomic systems, 3D crystals are invariant to discrete groups of isometries called the space groups. Crucially, these space group symmetries are known to heavily influence materials properties. We propose SGEquiDiff, a crystal generative model which naturally handles space group constraints with space group invariant likelihoods. SGEquiDiff consists of an SE(3)-invariant, telescoping discrete sampler of crystal lattices; permutation-invariant, transformer-based autoregressive sampling of Wyckoff positions, elements, and numbers of symmetrically unique atoms; and space group equivariant diffusion of atomic coordinates. We show that space group equivariant vector fields automatically live in the tangent spaces of the Wyckoff positions. SGEquiDiff achieves state-of-the-art performance on standard benchmark datasets as assessed by quantitative proxy metrics and quantum mechanical calculations. 

**Abstract (ZH)**: 使用生成模型加速晶体材料逆设计在一系列技术中具有重要意义。空间群不变性 likelihoods 的晶体生成模型 SGEquiDiff 处理空间群约束。SGEquiDiff 包含 SE(3) 不变的分层晶格采样器；Wyckoff 位置、元素及对称唯一原子数的置换不变性、基于变压器的自回归采样器；以及原子坐标的空间群可变扩散。我们证明了空间群可变向量场自然存在于 Wyckoff 位置的切空间中。SGEquiDiff 在标准基准数据集上的定量代理指标和量子力学计算中达到最佳性能。 

---
# GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models 

**Title (ZH)**: GenoArmory：面向基因组基础模型对抗攻击的统一评估框架 

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Han Liu, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10983)  

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. 

**Abstract (ZH)**: 我们提出了首个针对基因组基础模型（GFMs）的统一对抗攻击基准，名为GenoArmory。不同于现有的GFMs基准，GenoArmory提供了首个全面的评估框架，系统地评估GFMs对对抗攻击的脆弱性。从方法论上，我们使用四种广泛采用的攻击算法和三种防御策略，评估了五种最先进的GFMs的对抗鲁棒性。重要的是，我们的基准提供了一个易于访问且全面的框架，用于分析模型架构、量化方案和训练数据集对GFMs脆弱性的影响。此外，我们引入了GenoAdv，一个新设计的对抗样本数据集，旨在提高GFMs的安全性。实验结果表明，分类模型比生成模型对对抗扰动更具鲁棒性，这突显了任务类型对模型脆弱性的影响。此外，对抗攻击经常针对生物上重要的基因组区域，表明这些模型有效地捕捉了有意义的序列特征。 

---
# Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio 

**Title (ZH)**: 单声道音频端到端多说话人自动语音识别综述 

**Authors**: Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2505.10975)  

**Abstract**: Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR. 

**Abstract (ZH)**: 单声道多说话人自动语音识别（ASR）由于数据稀缺性和识别和归因给个体说话人词汇的固有难度，特别是在重叠语音中，依然具有挑战性。最近的进展推动了从级联系统向端到端（E2E）架构的转变，这减少了错误传播并更好地利用了语音内容与说话人身份之间的协同作用。尽管在E2E多说话人ASR领域取得了快速进展，但该领域缺乏对最近发展的全面综述。本文提供了E2E神经方法在多说话人ASR领域的系统分类，突出了最近的发展和比较分析。具体而言，我们分析了：（1）预分割音频的架构范式（SIMO vs. SISO），分析其各自的特点和权衡；（2）基于这两个范式的最近架构和算法改进；（3）长音频形式的扩展，包括分割策略和说话人一致假设拼接。此外，我们（4）在标准基准上评估和比较方法。最后，我们讨论了构建稳健和可扩展的多说话人ASR面临的主要挑战和未来研究方向。 

---
# Relational Graph Transformer 

**Title (ZH)**: 关系图变换器 

**Authors**: Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2505.10960)  

**Abstract**: Relational Deep Learning (RDL) is a promising approach for building state-of-the-art predictive models on multi-table relational data by representing it as a heterogeneous temporal graph. However, commonly used Graph Neural Network models suffer from fundamental limitations in capturing complex structural patterns and long-range dependencies that are inherent in relational data. While Graph Transformers have emerged as powerful alternatives to GNNs on general graphs, applying them to relational entity graphs presents unique challenges: (i) Traditional positional encodings fail to generalize to massive, heterogeneous graphs; (ii) existing architectures cannot model the temporal dynamics and schema constraints of relational data; (iii) existing tokenization schemes lose critical structural information. Here we introduce the Relational Graph Transformer (RelGT), the first graph transformer architecture designed specifically for relational tables. RelGT employs a novel multi-element tokenization strategy that decomposes each node into five components (features, type, hop distance, time, and local structure), enabling efficient encoding of heterogeneity, temporality, and topology without expensive precomputation. Our architecture combines local attention over sampled subgraphs with global attention to learnable centroids, incorporating both local and database-wide representations. Across 21 tasks from the RelBench benchmark, RelGT consistently matches or outperforms GNN baselines by up to 18%, establishing Graph Transformers as a powerful architecture for Relational Deep Learning. 

**Abstract (ZH)**: 关系图变换器（RelGT）：一种专门针对关系表的图变换器架构 

---
# Constrained Preferential Bayesian Optimization and Its Application in Banner Ad Design 

**Title (ZH)**: 受限偏好贝叶斯优化及其在Banner广告设计中的应用 

**Authors**: Koki Iwai, Yusuke Kumagae, Yuki Koyama, Masahiro Hamasaki, Masataka Goto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10954)  

**Abstract**: Preferential Bayesian optimization (PBO) is a variant of Bayesian optimization that observes relative preferences (e.g., pairwise comparisons) instead of direct objective values, making it especially suitable for human-in-the-loop scenarios. However, real-world optimization tasks often involve inequality constraints, which existing PBO methods have not yet addressed. To fill this gap, we propose constrained preferential Bayesian optimization (CPBO), an extension of PBO that incorporates inequality constraints for the first time. Specifically, we present a novel acquisition function for this purpose. Our technical evaluation shows that our CPBO method successfully identifies optimal solutions by focusing on exploring feasible regions. As a practical application, we also present a designer-in-the-loop system for banner ad design using CPBO, where the objective is the designer's subjective preference, and the constraint ensures a target predicted click-through rate. We conducted a user study with professional ad designers, demonstrating the potential benefits of our approach in guiding creative design under real-world constraints. 

**Abstract (ZH)**: 约束偏好贝叶斯优化（CPBO）：首次将不等式约束纳入偏好贝叶斯优化中 

---
# Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks 

**Title (ZH)**: Phi：利用基于模式的层次稀疏性构建高效突触神经网络 

**Authors**: Chiyue Wei, Bowen Duan, Cong Guo, Jingyang Zhang, Qingyue Song, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10909)  

**Abstract**: Spiking Neural Networks (SNNs) are gaining attention for their energy efficiency and biological plausibility, utilizing 0-1 activation sparsity through spike-driven computation. While existing SNN accelerators exploit this sparsity to skip zero computations, they often overlook the unique distribution patterns inherent in binary activations. In this work, we observe that particular patterns exist in spike activations, which we can utilize to reduce the substantial computation of SNN models. Based on these findings, we propose a novel \textbf{pattern-based hierarchical sparsity} framework, termed \textbf{\textit{Phi}}, to optimize computation.
\textit{Phi} introduces a two-level sparsity hierarchy: Level 1 exhibits vector-wise sparsity by representing activations with pre-defined patterns, allowing for offline pre-computation with weights and significantly reducing most runtime computation. Level 2 features element-wise sparsity by complementing the Level 1 matrix, using a highly sparse matrix to further reduce computation while maintaining accuracy. We present an algorithm-hardware co-design approach. Algorithmically, we employ a k-means-based pattern selection method to identify representative patterns and introduce a pattern-aware fine-tuning technique to enhance Level 2 sparsity. Architecturally, we design \textbf{\textit{Phi}}, a dedicated hardware architecture that efficiently processes the two levels of \textit{Phi} sparsity on the fly. Extensive experiments demonstrate that \textit{Phi} achieves a $3.45\times$ speedup and a $4.93\times$ improvement in energy efficiency compared to state-of-the-art SNN accelerators, showcasing the effectiveness of our framework in optimizing SNN computation. 

**Abstract (ZH)**: 基于模式的分层稀疏性框架（Phi）：优化脉冲神经网络计算 

---
# On the Security Risks of ML-based Malware Detection Systems: A Survey 

**Title (ZH)**: 基于机器学习的恶意软件检测系统安全风险综述 

**Authors**: Ping He, Yuhao Mao, Changjiang Li, Lorenzo Cavallaro, Ting Wang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.10903)  

**Abstract**: Malware presents a persistent threat to user privacy and data integrity. To combat this, machine learning-based (ML-based) malware detection (MD) systems have been developed. However, these systems have increasingly been attacked in recent years, undermining their effectiveness in practice. While the security risks associated with ML-based MD systems have garnered considerable attention, the majority of prior works is limited to adversarial malware examples, lacking a comprehensive analysis of practical security risks. This paper addresses this gap by utilizing the CIA principles to define the scope of security risks. We then deconstruct ML-based MD systems into distinct operational stages, thus developing a stage-based taxonomy. Utilizing this taxonomy, we summarize the technical progress and discuss the gaps in the attack and defense proposals related to the ML-based MD systems within each stage. Subsequently, we conduct two case studies, using both inter-stage and intra-stage analyses according to the stage-based taxonomy to provide new empirical insights. Based on these analyses and insights, we suggest potential future directions from both inter-stage and intra-stage perspectives. 

**Abstract (ZH)**: 基于机器学习的恶意软件检测系统面临持久的隐私和数据完整性威胁。尽管已经开发了基于机器学习的恶意软件检测系统（ML-based MD），但这些系统近年来不断受到攻击，影响了其实际效果。虽然与基于机器学习的恶意软件检测系统相关的安全风险已经引起了广泛关注，但大多数前期工作主要集中在对抗性恶意软件样本上，缺乏对实际安全风险的全面分析。本文通过利用CIA原则来定义安全风险的范围，并将基于机器学习的恶意软件检测系统分解为不同的操作阶段，从而构建一个阶段导向的分类体系。利用这一分类体系，我们总结了技术进步，讨论了每个阶段与基于机器学习的恶意软件检测系统相关的攻击和防御方案中的差距。随后，我们进行了两个案例研究，根据阶段导向分类体系进行跨阶段和同阶段分析，提供了新的实证见解。基于这些分析和见解，我们从跨阶段和同阶段两个视角提出了潜在的未来方向。 

---
# BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset 

**Title (ZH)**: BanglaFake: 构建与评估一个专门的孟加拉语深度假音音频数据集 

**Authors**: Istiaq Ahmed Fahad, Kamruzzaman Asif, Sifat Sikder  

**Link**: [PDF](https://arxiv.org/pdf/2505.10885)  

**Abstract**: Deepfake audio detection is challenging for low-resource languages like Bengali due to limited datasets and subtle acoustic features. To address this, we introduce BangalFake, a Bengali Deepfake Audio Dataset with 12,260 real and 13,260 deepfake utterances. Synthetic speech is generated using SOTA Text-to-Speech (TTS) models, ensuring high naturalness and quality. We evaluate the dataset through both qualitative and quantitative analyses. Mean Opinion Score (MOS) from 30 native speakers shows Robust-MOS of 3.40 (naturalness) and 4.01 (intelligibility). t-SNE visualization of MFCCs highlights real vs. fake differentiation challenges. This dataset serves as a crucial resource for advancing deepfake detection in Bengali, addressing the limitations of low-resource language research. 

**Abstract (ZH)**: Bengali Deepfake Audio Dataset: Addressing Challenges in Low-Resource Language Deepfake Detection 

---
# Graph and Simplicial Complex Prediction Gaussian Process via the Hodgelet Representations 

**Title (ZH)**: Hodgelet表示下的图与 simplicial 复杂网络预测高斯过程 

**Authors**: Mathieu Alain, So Takao, Xiaowen Dong, Bastian Rieck, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10877)  

**Abstract**: Predicting the labels of graph-structured data is crucial in scientific applications and is often achieved using graph neural networks (GNNs). However, when data is scarce, GNNs suffer from overfitting, leading to poor performance. Recently, Gaussian processes (GPs) with graph-level inputs have been proposed as an alternative. In this work, we extend the Gaussian process framework to simplicial complexes (SCs), enabling the handling of edge-level attributes and attributes supported on higher-order simplices. We further augment the resulting SC representations by considering their Hodge decompositions, allowing us to account for homological information, such as the number of holes, in the SC. We demonstrate that our framework enhances the predictions across various applications, paving the way for GPs to be more widely used for graph and SC-level predictions. 

**Abstract (ZH)**: 预测图结构数据的标签在科学应用中至关重要，通常使用图神经网络（GNNs）实现。然而，当数据稀缺时，GNNs会遭受过拟合，导致性能不佳。近年来，作为替代方案，基于图级输入的高斯过程（GPs）已被提出。在本文中，我们扩展了高斯过程框架至单纯复形（SCs），使得能够处理边级属性以及支持于高维单纯形上的属性。在此基础上，我们通过考虑单纯复形的亥姆霍兹分解进一步增强其表示，允许我们捕捉单纯复形中的同调信息，如洞的数量。我们证明，我们的框架在各种应用中增强了预测能力，为GPs在图和单纯复形级别上的广泛应用铺平了道路。 

---
# Preference Isolation Forest for Structure-based Anomaly Detection 

**Title (ZH)**: 基于结构的异常检测中的偏好隔离森林 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10876)  

**Abstract**: We address the problem of detecting anomalies as samples that do not conform to structured patterns represented by low-dimensional manifolds. To this end, we conceive a general anomaly detection framework called Preference Isolation Forest (PIF), that combines the benefits of adaptive isolation-based methods with the flexibility of preference embedding. The key intuition is to embed the data into a high-dimensional preference space by fitting low-dimensional manifolds, and to identify anomalies as isolated points. We propose three isolation approaches to identify anomalies: $i$) Voronoi-iForest, the most general solution, $ii$) RuzHash-iForest, that avoids explicit computation of distances via Local Sensitive Hashing, and $iii$) Sliding-PIF, that leverages a locality prior to improve efficiency and effectiveness. 

**Abstract (ZH)**: 基于偏好隔离森林的异常检测框架 

---
# MultiLink: Multi-class Structure Recovery via Agglomerative Clustering and Model Selection 

**Title (ZH)**: 多链接：基于凝聚聚类和模型选择的多类别结构恢复 

**Authors**: Luca Magri, Filippo Leveni, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10874)  

**Abstract**: We address the problem of recovering multiple structures of different classes in a dataset contaminated by noise and outliers. In particular, we consider geometric structures defined by a mixture of underlying parametric models (e.g. planes and cylinders, homographies and fundamental matrices), and we tackle the robust fitting problem by preference analysis and clustering. We present a new algorithm, termed MultiLink, that simultaneously deals with multiple classes of models. MultiLink combines on-the-fly model fitting and model selection in a novel linkage scheme that determines whether two clusters are to be merged. The resulting method features many practical advantages with respect to methods based on preference analysis, being faster, less sensitive to the inlier threshold, and able to compensate limitations deriving from hypotheses sampling. Experiments on several public datasets demonstrate that Multi-Link favourably compares with state of the art alternatives, both in multi-class and single-class problems. Code is publicly made available for download. 

**Abstract (ZH)**: 我们研究了在含有噪声和离群点的数据集中恢复不同类别的多个结构的问题。特别是，我们考虑由多个底层参数模型的混合定义的几何结构（例如平面和圆柱、仿射变换和基本矩阵），并通过偏好分析和聚类解决稳健拟合问题。我们提出了一种新的算法，称为MultiLink，该算法能够同时处理多种模型类别。MultiLink通过一种新颖的链接方案结合了模型的即席拟合和模型选择，该方案决定了两个聚类是否应被合并。该方法相对于基于偏好分析的方法具有许多实用优势，例如速度更快、对残余阈值的敏感性较低，并且能够补偿由于假设采样引起的限制。在多个公开数据集上的实验表明，MultiLink在多类和单类问题中都优于现有的解决方案。代码已公开提供下载。 

---
# Hashing for Structure-based Anomaly Detection 

**Title (ZH)**: 基于结构的异常检测的哈希方法 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10873)  

**Abstract**: We focus on the problem of identifying samples in a set that do not conform to structured patterns represented by low-dimensional manifolds. An effective way to solve this problem is to embed data in a high dimensional space, called Preference Space, where anomalies can be identified as the most isolated points. In this work, we employ Locality Sensitive Hashing to avoid explicit computation of distances in high dimensions and thus improve Anomaly Detection efficiency. Specifically, we present an isolation-based anomaly detection technique designed to work in the Preference Space which achieves state-of-the-art performance at a lower computational cost. Code is publicly available at this https URL. 

**Abstract (ZH)**: 我们关注识别不符合由低维流形表示的结构模式的数据样本的问题。解决这一问题的有效方法是将数据嵌入一个称为偏好空间的高维空间，在该空间中，异常点可以被认为是孤立度最大的点。在本工作中，我们采用局部敏感哈希来避免高维空间中的显式距离计算，从而提高异常检测效率。具体而言，我们提出了一种基于隔离的异常检测技术，该技术设计用于偏好空间中工作，能够在较低的计算成本下实现最先进的性能。代码可在以下网址公开获取。 

---
# Optimal Allocation of Privacy Budget on Hierarchical Data Release 

**Title (ZH)**: 层级数据发布中隐私预算的最优分配 

**Authors**: Joonhyuk Ko, Juba Ziani, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10871)  

**Abstract**: Releasing useful information from datasets with hierarchical structures while preserving individual privacy presents a significant challenge. Standard privacy-preserving mechanisms, and in particular Differential Privacy, often require careful allocation of a finite privacy budget across different levels and components of the hierarchy. Sub-optimal allocation can lead to either excessive noise, rendering the data useless, or to insufficient protections for sensitive information. This paper addresses the critical problem of optimal privacy budget allocation for hierarchical data release. It formulates this challenge as a constrained optimization problem, aiming to maximize data utility subject to a total privacy budget while considering the inherent trade-offs between data granularity and privacy loss. The proposed approach is supported by theoretical analysis and validated through comprehensive experiments on real hierarchical datasets. These experiments demonstrate that optimal privacy budget allocation significantly enhances the utility of the released data and improves the performance of downstream tasks. 

**Abstract (ZH)**: 具有层次结构的数据集在释放有用信息的同时保护个体隐私 presents a significant challenge. Optimal Privacy Budget Allocation for Hierarchical Data Release 

---
# ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data 

**Title (ZH)**: ImputeINR：通过隐式神经表示进行时间序列插补以处理缺失数据的疾病诊断 

**Authors**: Mengxuan Li, Ke Liu, Jialong Guo, Jiajun Bu, Hongwei Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10856)  

**Abstract**: Healthcare data frequently contain a substantial proportion of missing values, necessitating effective time series imputation to support downstream disease diagnosis tasks. However, existing imputation methods focus on discrete data points and are unable to effectively model sparse data, resulting in particularly poor performance for imputing substantial missing values. In this paper, we propose a novel approach, ImputeINR, for time series imputation by employing implicit neural representations (INR) to learn continuous functions for time series. ImputeINR leverages the merits of INR in that the continuous functions are not coupled to sampling frequency and have infinite sampling frequency, allowing ImputeINR to generate fine-grained imputations even on extremely sparse observed values. Extensive experiments conducted on eight datasets with five ratios of masked values show the superior imputation performance of ImputeINR, especially for high missing ratios in time series data. Furthermore, we validate that applying ImputeINR to impute missing values in healthcare data enhances the performance of downstream disease diagnosis tasks. Codes are available. 

**Abstract (ZH)**: 健康数据经常包含大量的缺失值， necessitating 有效的时间序列插补以支持下游的疾病诊断任务。然而，现有的插补方法主要针对离散数据点，无法有效地建模稀疏数据，导致在插补大量缺失值时表现特别差。本文提出了一种新的方法 ImputeINR，通过使用隐式神经表示（INR）来学习时间序列的连续函数进行时间序列插补。ImputeINR 利用了 INR 的优点，即连续函数与采样频率无关并且具有无限的采样频率，使得 ImputeINR 能够在极端稀疏的观测值上生成细粒度的插补。在八个数据集上进行的实验结果表明，ImputeINR 在时间序列数据中的高缺失比例插补中表现尤为出色。此外，我们验证了将 ImputeINR 应用于医疗健康数据的缺失值插补可以提升下游疾病诊断任务的性能。代码已开源。 

---
# Ready2Unlearn: A Learning-Time Approach for Preparing Models with Future Unlearning Readiness 

**Title (ZH)**: Ready2Unlearn: 一种为未来遗忘准备的训练时方法 

**Authors**: Hanyu Duan, Yi Yang, Ahmed Abbasi, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2505.10845)  

**Abstract**: This paper introduces Ready2Unlearn, a learning-time optimization approach designed to facilitate future unlearning processes. Unlike the majority of existing unlearning efforts that focus on designing unlearning algorithms, which are typically implemented reactively when an unlearning request is made during the model deployment phase, Ready2Unlearn shifts the focus to the training phase, adopting a "forward-looking" perspective. Building upon well-established meta-learning principles, Ready2Unlearn proactively trains machine learning models with unlearning readiness, such that they are well prepared and can handle future unlearning requests in a more efficient and principled manner. Ready2Unlearn is model-agnostic and compatible with any gradient ascent-based machine unlearning algorithms. We evaluate the method on both vision and language tasks under various unlearning settings, including class-wise unlearning and random data unlearning. Experimental results show that by incorporating such preparedness at training time, Ready2Unlearn produces an unlearning-ready model state, which offers several key advantages when future unlearning is required, including reduced unlearning time, improved retention of overall model capability, and enhanced resistance to the inadvertent recovery of forgotten data. We hope this work could inspire future efforts to explore more proactive strategies for equipping machine learning models with built-in readiness towards more reliable and principled machine unlearning. 

**Abstract (ZH)**: Ready2Unlearn：一种促进未来遗忘过程的训练时优化方法 

---
# Creating General User Models from Computer Use 

**Title (ZH)**: 从计算机使用中创建通用用户模型 

**Authors**: Omar Shaikh, Shardul Sapkota, Shan Rizvi, Eric Horvitz, Joon Sung Park, Diyi Yang, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.10831)  

**Abstract**: Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture that user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs. 

**Abstract (ZH)**: 人类计算机交互：一种通用用户模型的架构及其应用 

---
# Attention-Based Reward Shaping for Sparse and Delayed Rewards 

**Title (ZH)**: 基于注意力的奖励塑造：针对稀疏和延迟奖励 

**Authors**: Ian Holmes, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10802)  

**Abstract**: Sparse and delayed reward functions pose a significant obstacle for real-world Reinforcement Learning (RL) applications. In this work, we propose Attention-based REward Shaping (ARES), a general and robust algorithm which uses a transformer's attention mechanism to generate shaped rewards and create a dense reward function for any environment. ARES requires a set of episodes and their final returns as input. It can be trained entirely offline and is able to generate meaningful shaped rewards even when using small datasets or episodes produced by agents taking random actions. ARES is compatible with any RL algorithm and can handle any level of reward sparsity. In our experiments, we focus on the most challenging case where rewards are fully delayed until the end of each episode. We evaluate ARES across a diverse range of environments, widely used RL algorithms, and baseline methods to assess the effectiveness of the shaped rewards it produces. Our results show that ARES can significantly improve learning in delayed reward settings, enabling RL agents to train in scenarios that would otherwise require impractical amounts of data or even be unlearnable. To our knowledge, ARES is the first approach that works fully offline, remains robust to extreme reward delays and low-quality data, and is not limited to goal-based tasks. 

**Abstract (ZH)**: 基于注意力的奖励塑形（ARES）：一种通用且 robust 的算法 

---
# Analyzing Patterns and Influence of Advertising in Print Newspapers 

**Title (ZH)**: 分析印刷报纸中广告的模式与影响 

**Authors**: N Harsha Vardhan, Ponnurangam Kumaraguru, Kiran Garimella  

**Link**: [PDF](https://arxiv.org/pdf/2505.10791)  

**Abstract**: This paper investigates advertising practices in print newspapers across India using a novel data-driven approach. We develop a pipeline employing image processing and OCR techniques to extract articles and advertisements from digital versions of print newspapers with high accuracy. Applying this methodology to five popular newspapers that span multiple regions and three languages, English, Hindi, and Telugu, we assembled a dataset of more than 12,000 editions containing several hundred thousand advertisements. Collectively, these newspapers reach a readership of over 100 million people. Using this extensive dataset, we conduct a comprehensive analysis to answer key questions about print advertising: who advertises, what they advertise, when they advertise, where they place their ads, and how they advertise. Our findings reveal significant patterns, including the consistent level of print advertising over the past six years despite declining print circulation, the overrepresentation of company ads on prominent pages, and the disproportionate revenue contributed by government ads. Furthermore, we examine whether advertising in a newspaper influences the coverage an advertiser receives. Through regression analyses on coverage volume and sentiment, we find strong evidence supporting this hypothesis for corporate advertisers. The results indicate a clear trend where increased advertising correlates with more favorable and extensive media coverage, a relationship that remains robust over time and across different levels of advertiser popularity. 

**Abstract (ZH)**: 本文采用一种新颖的数据驱动方法研究印度印刷报纸的广告实践。我们开发了一条管线，利用图像处理和OCR技术从印刷报纸的数字化版本中高精度地提取文章和广告。将这种方法应用到五种流行的横跨多个区域和三种语言（英语、印地语和泰卢固语）的报纸上，我们构建了一个包含数十万条广告的超过12,000个版面的数据库。这些报纸的读者人数超过1亿。使用这个庞大的数据库，我们开展全面分析以回答关于印刷广告的关键问题：广告商是谁，他们广告的内容是什么，何时进行广告宣传，广告位置在何处，以及他们如何进行广告宣传。我们的发现揭示了显著的模式，包括过去六年印刷广告的一贯水平尽管印刷发行量下降，公司广告在显要版面的过度代表性以及政府广告对收入的不成比例贡献。此外，我们还探讨了报纸的广告是否会影响广告商获得的报道。通过回归分析报道量和情感指标，我们发现对于企业广告商而言，存在有力证据支持这一假设。结果显示，广告增加与更正面和更广泛的媒体覆盖之间存在明确趋势，这种关系在时间上和不同广告商受欢迎程度的层次上都保持稳健。 

---
# Neural-Inspired Advances in Integral Cryptanalysis 

**Title (ZH)**: 神经启发的积分攻击进展 

**Authors**: Liu Zhang, Yiran Yao, Danping Shi, Dongchen Chai, Jian Guo, Zilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10790)  

**Abstract**: The study by Gohr this http URL at CRYPTO 2019 and sunsequent related works have shown that neural networks can uncover previously unused features, offering novel insights into cryptanalysis. Motivated by these findings, we employ neural networks to learn features specifically related to integral properties and integrate the corresponding insights into optimized search frameworks. These findings validate the framework of using neural networks for feature exploration, providing researchers with novel insights that advance established cryptanalysis methods.
Neural networks have inspired the development of more precise integral search models. By comparing the integral distinguishers obtained via neural networks with those identified by classical methods, we observe that existing automated search models often fail to find optimal distinguishers. To address this issue, we develop a meet in the middle search framework that balances model accuracy and computational efficiency. As a result, we reduce the number of active plaintext bits required for an 11 rounds integral distinguisher on SKINNY64/64, and further identify a 12 rounds key dependent integral distinguisher achieving one additional round over the previous best-known result.
The integral distinguishers discovered by neural networks enable key recovery attacks on more rounds. We identify a 7 rounds key independent integral distinguisher from neural networks with even only one active plaintext cell, which is based on linear combinations of bits. This distinguisher enables a 15 rounds key recovery attack on SKINNYn/n, improving upon the previous record by one round. Additionally, we discover an 8 rounds key dependent integral distinguisher using neural network that further reduces the time complexity of key recovery attacks against SKINNY. 

**Abstract (ZH)**: 神经网络在CRYPTO 2019及后续相关工作中的研究表明，神经网络可以发掘未被使用的特征，为密码分析提供新的见解。受这些发现的启发，我们利用神经网络学习与整体性质相关的特点，并将相应的见解整合到优化的搜索框架中。这些发现验证了使用神经网络进行特征探索的框架的有效性，为研究人员提供了新的见解，推动了现有的密码分析方法的发展。 

---
# Learning Repetition-Invariant Representations for Polymer Informatics 

**Title (ZH)**: 学习不变重复聚合物表示方法 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Tengfei Luo, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10726)  

**Abstract**: Polymers are large macromolecules composed of repeating structural units known as monomers and are widely applied in fields such as energy storage, construction, medicine, and aerospace. However, existing graph neural network methods, though effective for small molecules, only model the single unit of polymers and fail to produce consistent vector representations for the true polymer structure with varying numbers of units. To address this challenge, we introduce Graph Repetition Invariance (GRIN), a novel method to learn polymer representations that are invariant to the number of repeating units in their graph representations. GRIN integrates a graph-based maximum spanning tree alignment with repeat-unit augmentation to ensure structural consistency. We provide theoretical guarantees for repetition-invariance from both model and data perspectives, demonstrating that three repeating units are the minimal augmentation required for optimal invariant representation learning. GRIN outperforms state-of-the-art baselines on both homopolymer and copolymer benchmarks, learning stable, repetition-invariant representations that generalize effectively to polymer chains of unseen sizes. 

**Abstract (ZH)**: 聚合物是由重复结构单元（单体）组成的大型高分子，广泛应用于储能、建筑、医疗和航天等领域。然而，现有的图神经网络方法虽对小分子有效，但仅能建模聚合物的单个单位，无法为具有不同单位数目的真实聚合物结构生成一致的向量表示。为解决这一挑战，我们提出了图重复不变性（GRIN）方法，该方法能够学习对图表示中重复单元数量不变的聚合物表示。GRIN 综合了基于图的最大生成树对齐和重复单元增强，以确保结构一致性。我们从模型和数据两个角度提供了重复不变性的理论保证，证明了三个重复单元是最小的增强需求，以实现最佳不变表示学习。GRIN 在同聚物和共聚物基准测试中均优于现有最佳baseline，学习到稳定且对重复不变的表示，能够有效泛化到未见过尺寸的聚合物链上。 

---
# GNN-Suite: a Graph Neural Network Benchmarking Framework for Biomedical Informatics 

**Title (ZH)**: GNN-Suite：生物医学informatics领域图神经网络基准测试框架 

**Authors**: Sebestyén Kamp, Giovanni Stracquadanio, T. Ian Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10711)  

**Abstract**: We present GNN-Suite, a robust modular framework for constructing and benchmarking Graph Neural Network (GNN) architectures in computational biology. GNN-Suite standardises experimentation and reproducibility using the Nextflow workflow to evaluate GNN performance. We demonstrate its utility in identifying cancer-driver genes by constructing molecular networks from protein-protein interaction (PPI) data from STRING and BioGRID and annotating nodes with features from the PCAWG, PID, and COSMIC-CGC repositories.
Our design enables fair comparisons among diverse GNN architectures including GAT, GAT3H, GCN, GCN2, GIN, GTN, HGCN, PHGCN, and GraphSAGE and a baseline Logistic Regression (LR) model. All GNNs were configured as standardised two-layer models and trained with uniform hyperparameters (dropout = 0.2; Adam optimiser with learning rate = 0.01; and an adjusted binary cross-entropy loss to address class imbalance) over an 80/20 train-test split for 300 epochs. Each model was evaluated over 10 independent runs with different random seeds to yield statistically robust performance metrics, with balanced accuracy (BACC) as the primary measure. Notably, GCN2 achieved the highest BACC (0.807 +/- 0.035) on a STRING-based network, although all GNN types outperformed the LR baseline, highlighting the advantage of network-based learning over feature-only approaches.
Our results show that a common framework for implementing and evaluating GNN architectures aids in identifying not only the best model but also the most effective means of incorporating complementary data. By making GNN-Suite publicly available, we aim to foster reproducible research and promote improved benchmarking standards in computational biology. Future work will explore additional omics datasets and further refine network architectures to enhance predictive accuracy and interpretability in biomedical applications. 

**Abstract (ZH)**: 一种用于计算生物学中图神经网络架构构建与基准测试的稳健模块化框架：GNN-Suite及其在识别癌症驱动基因中的应用 

---
# Predicting Risk of Pulmonary Fibrosis Formation in PASC Patients 

**Title (ZH)**: PASC患者肺纤维化形成风险的预测 

**Authors**: Wanying Dou, Gorkem Durak, Koushik Biswas, Ziliang Hong, Andrea Mia Bejar, Elif Keles, Kaan Akin, Sukru Mehmet Erturk, Alpay Medetalibeyoglu, Marc Sala, Alexander Misharin, Hatice Savas, Mary Salvatore, Sachin Jambawalikar, Drew Torigian, Jayaram K. Udupa, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2505.10691)  

**Abstract**: While the acute phase of the COVID-19 pandemic has subsided, its long-term effects persist through Post-Acute Sequelae of COVID-19 (PASC), commonly known as Long COVID. There remains substantial uncertainty regarding both its duration and optimal management strategies. PASC manifests as a diverse array of persistent or newly emerging symptoms--ranging from fatigue, dyspnea, and neurologic impairments (e.g., brain fog), to cardiovascular, pulmonary, and musculoskeletal abnormalities--that extend beyond the acute infection phase. This heterogeneous presentation poses substantial challenges for clinical assessment, diagnosis, and treatment planning. In this paper, we focus on imaging findings that may suggest fibrotic damage in the lungs, a critical manifestation characterized by scarring of lung tissue, which can potentially affect long-term respiratory function in patients with PASC. This study introduces a novel multi-center chest CT analysis framework that combines deep learning and radiomics for fibrosis prediction. Our approach leverages convolutional neural networks (CNNs) and interpretable feature extraction, achieving 82.2% accuracy and 85.5% AUC in classification tasks. We demonstrate the effectiveness of Grad-CAM visualization and radiomics-based feature analysis in providing clinically relevant insights for PASC-related lung fibrosis prediction. Our findings highlight the potential of deep learning-driven computational methods for early detection and risk assessment of PASC-related lung fibrosis--presented for the first time in the literature. 

**Abstract (ZH)**: 尽管COVID-19急性期已过去，其长期影响通过新冠后遗症（PASC）或俗称“长 COVID”持续存在。关于其持续时间和最佳管理策略仍存在大量不确定性。PASC 表现为一系列持续或新出现的症状——从疲劳、呼吸困难和神经系统损害（如脑雾）到心血管、肺部和肌肉骨骼异常——这些症状超出了急性感染期。这种异质性表现给临床评估、诊断和治疗规划带来了巨大挑战。本文聚焦于影像学发现，这些发现可能表明肺纤维化损伤，这是一种关键表现，特点是肺组织疤痕化，可能影响PASC患者的长期呼吸功能。本研究引入了一种结合深度学习和 Radiomics 的多中心胸部CT分析框架，用于纤维化预测。我们的方法利用卷积神经网络（CNNs）和可解释的特征提取，分类任务的准确率为82.2%，AUC为85.5%。我们展示了Grad-CAM可视化和基于Radiomics的特征分析在提供与PASC相关的肺纤维化预测的临床相关见解方面的有效性。我们的研究结果突出了基于深度学习的计算方法在PASC相关肺纤维化的早期检测和风险评估中的潜力——这是首次在文献中提出。 

---
# A Conformal Predictive Measure for Assessing Catastrophic Forgetting 

**Title (ZH)**: 用于评估灾难性遗忘的配准预测衡量指标 

**Authors**: Ioannis Pitsiorlas, Nour Jamoussi, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2505.10677)  

**Abstract**: This work introduces a novel methodology for assessing catastrophic forgetting (CF) in continual learning. We propose a new conformal prediction (CP)-based metric, termed the Conformal Prediction Confidence Factor (CPCF), to quantify and evaluate CF effectively. Our framework leverages adaptive CP to estimate forgetting by monitoring the model's confidence on previously learned tasks. This approach provides a dynamic and practical solution for monitoring and measuring CF of previous tasks as new ones are introduced, offering greater suitability for real-world applications. Experimental results on four benchmark datasets demonstrate a strong correlation between CPCF and the accuracy of previous tasks, validating the reliability and interpretability of the proposed metric. Our results highlight the potential of CPCF as a robust and effective tool for assessing and understanding CF in dynamic learning environments. 

**Abstract (ZH)**: 本研究提出了一种新的方法论，用于评估连续学习中的灾难性遗忘（CF）。我们提出了一种基于可信区间（CP）的新度量方法，称为可信区间信心因子（CPCF），以有效量化和评估CF。我们的框架利用自适应CP来通过监控模型对之前学习的任务的信心来估计遗忘。这种 Approach 提供了一种动态且实用的方法来监测和测量随新任务引入而来的之前的任务的CF，使其更适用于实际应用。四项基准数据集上的实验结果表明，CPCF与之前任务的准确性之间存在密切关联，验证了所提出度量的可靠性和可解释性。研究结果突显了CPCF作为评估和理解动态学习环境中CF的稳健而有效的工具的潜力。 

---
# Seasonal Forecasting of Pan-Arctic Sea Ice with State Space Model 

**Title (ZH)**: 北极Pan-Arctic区域海冰季节预报模型研究 

**Authors**: Wei Wang, Weidong Yang, Lei Wang, Guihua Wang, Ruibo Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.10665)  

**Abstract**: The rapid decline of Arctic sea ice resulting from anthropogenic climate change poses significant risks to indigenous communities, ecosystems, and the global climate system. This situation emphasizes the immediate necessity for precise seasonal sea ice forecasts. While dynamical models perform well for short-term forecasts, they encounter limitations in long-term forecasts and are computationally intensive. Deep learning models, while more computationally efficient, often have difficulty managing seasonal variations and uncertainties when dealing with complex sea ice dynamics. In this research, we introduce IceMamba, a deep learning architecture that integrates sophisticated attention mechanisms within the state space model. Through comparative analysis of 25 renowned forecast models, including dynamical, statistical, and deep learning approaches, our experimental results indicate that IceMamba delivers excellent seasonal forecasting capabilities for Pan-Arctic sea ice concentration. Specifically, IceMamba outperforms all tested models regarding average RMSE and anomaly correlation coefficient (ACC) and ranks second in Integrated Ice Edge Error (IIEE). This innovative approach enhances our ability to foresee and alleviate the effects of sea ice variability, offering essential insights for strategies aimed at climate adaptation. 

**Abstract (ZH)**: 北极海冰因人为气候变化的快速减少给土著社区、生态系统和全球气候系统带来了重大风险。这种情况强调了进行精确季节性海冰预报的迫切需求。虽然动力模型在短期预报中表现良好，但在长期预报中存在局限性且计算成本高昂。深度学习模型虽然在计算效率上更具优势，但在处理复杂海冰动力学时往往难以应对季节变化和不确定性。在此研究中，我们引入了IceMamba，这是一种在状态空间模型中集成高级注意机制的深度学习架构。通过对比分析包括动力学、统计学和深度学习方法在内的25个知名预报模型，我们的实验结果表明，IceMamba在北极地区海冰浓度季节性预报方面表现出色。具体而言，IceMamba在平均RMSE和异常相关系数（ACC）方面超过了所有测试模型，在集成冰缘误差（IIEE）方面排名第二。这一创新方法增强了我们预测和缓解海冰变异性影响的能力，为气候适应策略提供了重要见解。 

---
# Artificial Intelligence Bias on English Language Learners in Automatic Scoring 

**Title (ZH)**: 人工智能偏见对英语学习者自动评分的影响 

**Authors**: Shuchen Guo, Yun Wang, Jichao Yu, Xuansheng Wu, Bilgehan Ayik, Field M. Watts, Ehsan Latif, Ninghao Liu, Lei Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.10643)  

**Abstract**: This study investigated potential scoring biases and disparities toward English Language Learners (ELLs) when using automatic scoring systems for middle school students' written responses to science assessments. We specifically focus on examining how unbalanced training data with ELLs contributes to scoring bias and disparities. We fine-tuned BERT with four datasets: responses from (1) ELLs, (2) non-ELLs, (3) a mixed dataset reflecting the real-world proportion of ELLs and non-ELLs (unbalanced), and (4) a balanced mixed dataset with equal representation of both groups. The study analyzed 21 assessment items: 10 items with about 30,000 ELL responses, five items with about 1,000 ELL responses, and six items with about 200 ELL responses. Scoring accuracy (Acc) was calculated and compared to identify bias using Friedman tests. We measured the Mean Score Gaps (MSGs) between ELLs and non-ELLs and then calculated the differences in MSGs generated through both the human and AI models to identify the scoring disparities. We found that no AI bias and distorted disparities between ELLs and non-ELLs were found when the training dataset was large enough (ELL = 30,000 and ELL = 1,000), but concerns could exist if the sample size is limited (ELL = 200). 

**Abstract (ZH)**: 本研究调查了在使用自动评分系统对中学生科学评估书面回答时，英语语言学习者（ELLs）评分偏差和不平等现象的可能性。我们重点关注不平衡训练数据对ELLs评分偏差和不平等的影响。我们使用四个数据集微调了BERT：（1）ELLs的回答，（2）非ELLs的回答，（3）反映实际ELLs和非ELLs比例的不均衡混合数据集，（4）均衡混合数据集，其中两个群体的代表数量相等。研究分析了21项评估项目：10个项目约有30,000份ELL回答，5个项目约有1,000份ELL回答，6个项目约有200份ELL回答。计算评分准确性（Acc）并使用弗里德曼检验进行比较以识别偏差。我们测量了ELLs和非ELLs之间的平均得分差距（MSGs），并计算了通过人类和AI模型产生的MSGs差异，以鉴定评分不平等现象。研究发现，当训练数据集足够大时（ELL = 30,000和ELL = 1,000），未发现AI偏见和扭曲的不平等现象，但如果样本量有限（ELL = 200），可能存在担忧。 

---
# Agent Name Service (ANS): A Universal Directory for Secure AI Agent Discovery and Interoperability 

**Title (ZH)**: 代理名称服务（ANS）：一种安全的AI代理发现和互操作的通用目录 

**Authors**: Ken Huang, Vineeth Sai Narajala, Idan Habler, Akram Sheriff  

**Link**: [PDF](https://arxiv.org/pdf/2505.10609)  

**Abstract**: The proliferation of AI agents requires robust mechanisms for secure discovery. This paper introduces the Agent Name Service (ANS), a novel architecture based on DNS addressing the lack of a public agent discovery framework. ANS provides a protocol-agnostic registry infrastructure that leverages Public Key Infrastructure (PKI) certificates for verifiable agent identity and trust. The architecture features several key innovations: a formalized agent registration and renewal mechanism for lifecycle management; DNS-inspired naming conventions with capability-aware resolution; a modular Protocol Adapter Layer supporting diverse communication standards (A2A, MCP, ACP etc.); and precisely defined algorithms for secure resolution. We implement structured communication using JSON Schema and conduct a comprehensive threat analysis of our proposal. The result is a foundational directory service addressing the core challenges of secured discovery and interaction in multi-agent systems, paving the way for future interoperable, trustworthy, and scalable agent ecosystems. 

**Abstract (ZH)**: AI代理的 proliferations 需要稳健的机制来确保安全发现。本文介绍了代理名称服务（ANS），这是一种新型架构，基于DNS，旨在解决公共代理发现框架缺乏的问题。ANS提供了一种协议无关的注册基础设施，利用公钥基础设施（PKI）证书来验证代理身份和建立信任。该架构包含多项关键创新：正式规定的代理注册和续期机制以管理生命周期；DNS启发式的命名规范具有能力感知的解析；支持各种通信标准（A2A、MCP、ACP等）的模块化协议适配器层；以及定义精确的算法以实现安全解析。我们使用JSON Schema实现结构化通信，并对我们的方案进行全面的安全威胁分析。结果是一种基础目录服务，解决了多代理系统中安全发现和交互的核心挑战，为未来可互操作、可信赖和可扩展的代理生态系统铺平了道路。 

---
# Enhancing IoT Cyber Attack Detection in the Presence of Highly Imbalanced Data 

**Title (ZH)**: 在高度不平衡数据存在下的物联网网络攻击检测增强方法 

**Authors**: Md. Ehsanul Haque, Md. Saymon Hosen Polash, Md Al-Imran Sanjida Simla, Md Alomgir Hossain, Sarwar Jahan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10600)  

**Abstract**: Due to the rapid growth in the number of Internet of Things (IoT) networks, the cyber risk has increased exponentially, and therefore, we have to develop effective IDS that can work well with highly imbalanced datasets. A high rate of missed threats can be the result, as traditional machine learning models tend to struggle in identifying attacks when normal data volume is much higher than the volume of attacks. For example, the dataset used in this study reveals a strong class imbalance with 94,659 instances of the majority class and only 28 instances of the minority class, making it quite challenging to determine rare attacks accurately. The challenges presented in this research are addressed by hybrid sampling techniques designed to improve data imbalance detection accuracy in IoT domains. After applying these techniques, we evaluate the performance of several machine learning models such as Random Forest, Soft Voting, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Logistic Regression with respect to the classification of cyber-attacks. The obtained results indicate that the Random Forest model achieved the best performance with a Kappa score of 0.9903, test accuracy of 0.9961, and AUC of 0.9994. Strong performance is also shown by the Soft Voting model, with an accuracy of 0.9952 and AUC of 0.9997, indicating the benefits of combining model predictions. Overall, this work demonstrates the value of hybrid sampling combined with robust model and feature selection for significantly improving IoT security against cyber-attacks, especially in highly imbalanced data environments. 

**Abstract (ZH)**: 由于物联网（IoT）网络的数量快速增长，网络风险已呈指数级增加，因此我们必须开发有效的入侵检测系统（IDS），以应对高度不平衡的数据集。传统的机器学习模型在正常数据量远高于攻击数据量的情况下，难以识别攻击，容易导致误报率过高。本研究使用的数据集展示了严重的类别不平衡，主要类别有94,659个实例，而少数类别仅有28个实例，这使得准确确定罕见攻击变得非常具有挑战性。本研究通过设计的混合采样技术解决了这些挑战，这些技术旨在提高IoT领域中的数据不平衡检测准确性。应用这些技术后，我们评估了包括随机森林、软投票、支持向量分类器（SVC）、K-近邻（KNN）、多层感知器（MLP）和逻辑回归在内的多种机器学习模型在网络安全分类中的性能。结果显示，随机森林模型在κ分数、测试准确率和AUC方面表现最佳，分别为0.9903、0.9961和0.9994。软投票模型也表现出强劲的性能，其准确率为0.9952，AUC为0.9997，显示出结合模型预测的益处。总体而言，本工作展示了混合采样技术与稳健的模型和特征选择相结合的价值，这对显著提高IoT网络安全，特别是在高度不平衡数据环境中，具有重要意义。 

---
# UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech 

**Title (ZH)**: UDDETTS：统一离散和维度情感以实现可控的情感文本-to-语音转换 

**Authors**: Jiaxuan Liu, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.10599)  

**Abstract**: Recent neural codec language models have made great progress in the field of text-to-speech (TTS), but controllable emotional TTS still faces many challenges. Traditional methods rely on predefined discrete emotion labels to control emotion categories and intensities, which can't capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotion annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a neural codec language model unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotion annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along the three dimensions of ADV space, and exhibits superior end-to-end emotional speech synthesis capabilities. 

**Abstract (ZH)**: Recent Neural Codec Language Models Have Made Great Progress in Text-to-Speech (TTS), but Controllable Emotional TTS Still Faces Many Challenges: UDDETTS, a Neural Codec Language Model Unifying Discrete and Dimensional Emotions for Controllable Emotional TTS 

---
# Inclusivity of AI Speech in Healthcare: A Decade Look Back 

**Title (ZH)**: AI语音在 healthcare 中的包容性：十年回顾 

**Authors**: Retno Larasati  

**Link**: [PDF](https://arxiv.org/pdf/2505.10596)  

**Abstract**: The integration of AI speech recognition technologies into healthcare has the potential to revolutionize clinical workflows and patient-provider communication. However, this study reveals significant gaps in inclusivity, with datasets and research disproportionately favouring high-resource languages, standardized accents, and narrow demographic groups. These biases risk perpetuating healthcare disparities, as AI systems may misinterpret speech from marginalized groups. This paper highlights the urgent need for inclusive dataset design, bias mitigation research, and policy frameworks to ensure equitable access to AI speech technologies in healthcare. 

**Abstract (ZH)**: 将AI语音识别技术集成到医疗保健中有望革命化临床流程和患者-提供者沟通。然而，本研究揭示了包容性方面的显著差距，数据集和研究过度偏向高资源语言、标准化口音和狭窄的人口群体。这些偏见可能导致医疗保健不平等的加剧，因为AI系统可能误解边缘化群体的语音。本文强调了迫切需要包容性数据集设计、偏见缓解研究和政策框架，以确保在医疗保健中公平获取AI语音技术。 

---
# Anchoring AI Capabilities in Market Valuations: The Capability Realization Rate Model and Valuation Misalignment Risk 

**Title (ZH)**: 将AI能力锚定在市场估值中：能力实现率模型与估值失衡风险 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10590)  

**Abstract**: Recent breakthroughs in artificial intelligence (AI) have triggered surges in market valuations for AI-related companies, often outpacing the realization of underlying capabilities. We examine the anchoring effect of AI capabilities on equity valuations and propose a Capability Realization Rate (CRR) model to quantify the gap between AI potential and realized performance. Using data from the 2023--2025 generative AI boom, we analyze sector-level sensitivity and conduct case studies (OpenAI, Adobe, NVIDIA, Meta, Microsoft, Goldman Sachs) to illustrate patterns of valuation premium and misalignment. Our findings indicate that AI-native firms commanded outsized valuation premiums anchored to future potential, while traditional companies integrating AI experienced re-ratings subject to proof of tangible returns. We argue that CRR can help identify valuation misalignment risk-where market prices diverge from realized AI-driven value. We conclude with policy recommendations to improve transparency, mitigate speculative bubbles, and align AI innovation with sustainable market value. 

**Abstract (ZH)**: 近期人工智能领域的突破引发了与人工智能相关公司市场估值的激增，往往超过其潜在能力的实现。我们研究了人工智能能力对股权估值的锚定效应，并提出了一种能力实现率（CRR）模型，以量化人工智能潜在价值与实现性能之间的差距。利用2023-2025年生成式人工智能热潮的数据，我们分析了行业层面的敏感性，并通过案例研究（OpenAI、Adobe、NVIDIA、Meta、Microsoft、Goldman Sachs）来阐述估值溢价和错配的模式。我们的研究发现，人工智能原生企业享有基于未来潜力的巨额估值溢价，而融合人工智能的传统企业则根据实际回报经历了重新评级。我们认为CRR可以帮助识别估值错配风险，即市场价格与通过人工智能实现的价值不符。最后，我们提出了政策建议，以提高透明度、缓解泡沫化，并使人工智能创新与可持续市场价格相一致。 

---
