# CottonSim: Development of an autonomous visual-guided robotic cotton-picking system in the Gazebo 

**Title (ZH)**: CottonSim: 基于Gazebo的自主视觉引导采摘机器人系统开发 

**Authors**: Thevathayarajh Thayananthan, Xin Zhang, Yanbo Huang, Jingdao Chen, Nuwan K. Wijewardane, Vitor S. Martins, Gary D. Chesser, Christopher T. Goodin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05317)  

**Abstract**: In this study, an autonomous visual-guided robotic cotton-picking system, built on a Clearpath's Husky robot platform and the Cotton-Eye perception system, was developed in the Gazebo robotic simulator. Furthermore, a virtual cotton farm was designed and developed as a Robot Operating System (ROS 1) package to deploy the robotic cotton picker in the Gazebo environment for simulating autonomous field navigation. The navigation was assisted by the map coordinates and an RGB-depth camera, while the ROS navigation algorithm utilized a trained YOLOv8n-seg model for instance segmentation. The model achieved a desired mean Average Precision (mAP) of 85.2%, a recall of 88.9%, and a precision of 93.0% for scene segmentation. The developed ROS navigation packages enabled our robotic cotton-picking system to autonomously navigate through the cotton field using map-based and GPS-based approaches, visually aided by a deep learning-based perception system. The GPS-based navigation approach achieved a 100% completion rate (CR) with a threshold of 5 x 10^-6 degrees, while the map-based navigation approach attained a 96.7% CR with a threshold of 0.25 m. This study establishes a fundamental baseline of simulation for future agricultural robotics and autonomous vehicles in cotton farming and beyond. CottonSim code and data are released to the research community via GitHub: this https URL 

**Abstract (ZH)**: 基于Clearpath Husky平台和Cotton-Eye感知系统的自主视觉引导棉花采摘机器人系统在Gazebo仿真器中的开发及导航研究 

---
# Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation 

**Title (ZH)**: 形态对称强化学习在双臂灵巧操控中的应用 

**Authors**: Zechu Li, Yufeng Jin, Daniel Ordonez Apraez, Claudio Semini, Puze Liu, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.05287)  

**Abstract**: Humans naturally exhibit bilateral symmetry in their gross manipulation skills, effortlessly mirroring simple actions between left and right hands. Bimanual robots-which also feature bilateral symmetry-should similarly exploit this property to perform tasks with either hand. Unlike humans, who often favor a dominant hand for fine dexterous skills, robots should ideally execute ambidextrous manipulation with equal proficiency. To this end, we introduce SYMDEX (SYMmetric DEXterity), a reinforcement learning framework for ambidextrous bi-manipulation that leverages the robot's inherent bilateral symmetry as an inductive bias. SYMDEX decomposes complex bimanual manipulation tasks into per-hand subtasks and trains dedicated policies for each. By exploiting bilateral symmetry via equivariant neural networks, experience from one arm is inherently leveraged by the opposite arm. We then distill the subtask policies into a global ambidextrous policy that is independent of the hand-task assignment. We evaluate SYMDEX on six challenging simulated manipulation tasks and demonstrate successful real-world deployment on two of them. Our approach strongly outperforms baselines on complex task in which the left and right hands perform different roles. We further demonstrate SYMDEX's scalability by extending it to a four-arm manipulation setup, where our symmetry-aware policies enable effective multi-arm collaboration and coordination. Our results highlight how structural symmetry as inductive bias in policy learning enhances sample efficiency, robustness, and generalization across diverse dexterous manipulation tasks. 

**Abstract (ZH)**: 人类在粗大运动技能中自然表现出双侧对称性，能够轻松地在左右手之间镜像简单的动作。具有双侧对称性的双臂机器人也应该利用这一特性，利用每只手熟练执行任务。与人类通常偏好使用主导手进行精细灵巧动作不同，机器人理想状态下应以同等 proficiency 实现双臂灵巧操作。为此，我们引入了 SYMDEX（SYMMETRIC DEXTERITY）框架，这是一种基于机器人固有的双侧对称性的归纳偏置的强化学习框架，用于进行双臂灵巧操作。SYMDEX 将复杂的双臂灵巧操作任务分解为单手亚任务，并为每个单手亚任务训练专门的策略。通过利用对称性（使用共变神经网络），一只手臂的经验可以天然地被另一只手臂利用。然后，我们将亚任务策略提炼为一个独立于手-任务分配的全局双臂策略。在六个具有挑战性的模拟灵巧操作任务中评估了 SYMDEX，并在其中两个任务上的现实世界部署中取得了成功。在复杂的任务中，其中左、右手执行不同的角色，我们的方法在基线上表现显著更佳。我们进一步通过扩展 SYMDEX 到四臂操作设置，展示了其可扩展性，其中我们的对称性感知策略实现了有效的多臂协作和协调。我们的结果突显了结构对称性作为策略学习中的归纳偏置如何提升样本效率、稳健性和在各种灵巧操作任务中的泛化能力。 

---
# X-Driver: Explainable Autonomous Driving with Vision-Language Models 

**Title (ZH)**: X-驱动：基于视觉语言模型的可解释自动驾驶 

**Authors**: Wei Liu, Jiyuan Zhang, Binxiong Zheng, Yufeng Hu, Yingzhan Lin, Zengfeng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05098)  

**Abstract**: End-to-end autonomous driving has advanced significantly, offering benefits such as system simplicity and stronger driving performance in both open-loop and closed-loop settings than conventional pipelines. However, existing frameworks still suffer from low success rates in closed-loop evaluations, highlighting their limitations in real-world deployment. In this paper, we introduce X-Driver, a unified multi-modal large language models(MLLMs) framework designed for closed-loop autonomous driving, leveraging Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and decision-making. We validate X-Driver across multiple autonomous driving tasks using public benchmarks in CARLA simulation environment, including Bench2Drive[6]. Our experimental results demonstrate superior closed-loop performance, surpassing the current state-of-the-art(SOTA) while improving the interpretability of driving decisions. These findings underscore the importance of structured reasoning in end-to-end driving and establish X-Driver as a strong baseline for future research in closed-loop autonomous driving. 

**Abstract (ZH)**: 端到端自主驾驶已取得显著进展，提供了系统简洁性和比传统流水线在开环和闭环设置中更强的驾驶性能。然而，现有框架在闭环评估中仍面临较低的成功率，突显了其在现实世界部署中的局限性。本文介绍了X-Driver，一个用于闭环自主驾驶的统一多模态大规模语言模型框架，通过链式思考(CoT)和自回归建模提升感知和决策能力。我们使用CARLA仿真环境中的公共基准测试X-Driver在多个自主驾驶任务中的表现，包括Bench2Drive[6]。实验结果表明，X-Driver在闭环性能上优于当前最佳水平，同时提高了驾驶决策的可解释性。这些发现强调了端到端驾驶中结构化推理的重要性，并将X-Driver确立为闭环自主驾驶未来研究的强大基线。 

---
# The City that Never Settles: Simulation-based LiDAR Dataset for Long-Term Place Recognition Under Extreme Structural Changes 

**Title (ZH)**: 永不安家的城市：极端结构变化下长期场所识别的基于模拟的LiDAR数据集 

**Authors**: Hyunho Song, Dongjae Lee, Seunghun Oh, Minwoo Jung, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05076)  

**Abstract**: Large-scale construction and demolition significantly challenge long-term place recognition (PR) by drastically reshaping urban and suburban environments. Existing datasets predominantly reflect limited or indoor-focused changes, failing to adequately represent extensive outdoor transformations. To bridge this gap, we introduce the City that Never Settles (CNS) dataset, a simulation-based dataset created using the CARLA simulator, capturing major structural changes-such as building construction and demolition-across diverse maps and sequences. Additionally, we propose TCR_sym, a symmetric version of the original TCR metric, enabling consistent measurement of structural changes irrespective of source-target ordering. Quantitative comparisons demonstrate that CNS encompasses more extensive transformations than current real-world benchmarks. Evaluations of state-of-the-art LiDAR-based PR methods on CNS reveal substantial performance degradation, underscoring the need for robust algorithms capable of handling significant environmental changes. Our dataset is available at this https URL. 

**Abstract (ZH)**: 大规模的建设与拆除显著挑战了长期地点识别（PR）任务，通过剧烈重塑城市和郊区环境。现有数据集主要反映有限的或以室内为主的变化，未能充分代表广泛的户外转变。为了弥合这一差距，我们介绍了永不沉寂的城市（CNS）数据集，这是一个使用CARLA模拟器基于模拟构建的数据集，捕捉到了不同地图和序列中主要结构变化，如建筑建设与拆除。此外，我们提出了TCR_sym，这是原始TCR度量的对称版本，能够在源-目标顺序无关的情况下一致地测量结构变化。定量比较表明，CNS涵盖了比当前现实世界基准更广泛的转变。在CNS上对最先进的LiDAR基PR方法的评估揭示了显著的性能下降，强调了能够处理重大环境变化的稳健算法的需求。我们的数据集可在以下链接获取：this https URL。 

---
# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations 

**Title (ZH)**: CLAM：连续潜在动作模型用于机器人从无标签示范学习 

**Authors**: Anthony Liang, Pavel Czempin, Matthew Hong, Yutai Zhou, Erdem Biyik, Stephen Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04999)  

**Abstract**: Learning robot policies using imitation learning requires collecting large amounts of costly action-labeled expert demonstrations, which fundamentally limits the scale of training data. A promising approach to address this bottleneck is to harness the abundance of unlabeled observations-e.g., from video demonstrations-to learn latent action labels in an unsupervised way. However, we find that existing methods struggle when applied to complex robot tasks requiring fine-grained motions. We design continuous latent action models (CLAM) which incorporate two key ingredients we find necessary for learning to solve complex continuous control tasks from unlabeled observation data: (a) using continuous latent action labels instead of discrete representations, and (b) jointly training an action decoder to ensure that the latent action space can be easily grounded to real actions with relatively few labeled examples. Importantly, the labeled examples can be collected from non-optimal play data, enabling CLAM to learn performant policies without access to any action-labeled expert data. We demonstrate on continuous control benchmarks in DMControl (locomotion) and MetaWorld (manipulation), as well as on a real WidowX robot arm that CLAM significantly outperforms prior state-of-the-art methods, remarkably with a 2-3x improvement in task success rate compared to the best baseline. Videos and code can be found at this http URL. 

**Abstract (ZH)**: 使用模仿学习学习机器人策略需要收集大量昂贵的动作标签专家示范，这根本上限制了训练数据的规模。通过利用无标签观察数据（例如来自视频示范的数据）以无监督方式学习潜在动作标签来应对这一瓶颈是一种有前景的方法。然而，我们发现现有方法在应用于需要精细动作的复杂机器人任务时表现不佳。为此，我们设计了连续潜在动作模型（CLAM），该模型包含两个我们认为对于从无标签观察数据中学习解决复杂连续控制任务所必需的关键成分：(a) 使用连续的潜在动作标签而非离散表示，(b) 联合训练一个动作解码器以确保潜在动作空间可以通过少量标记示例相对容易地与真实动作对接。重要的是，标记示例可以从非最优操作数据中收集，从而使CLAM能够在无需访问任何动作标签专家数据的情况下学习出高性能策略。我们在DMControl（运动）和MetaWorld（操作）的连续控制基准测试中以及在实际的WidowX机器人臂上展示了CLAM显著优于先前的最好方法，任务成功率相比最佳基线有2-3倍的提升。相关视频和代码见此网址。 

---
# LVLM-MPC Collaboration for Autonomous Driving: A Safety-Aware and Task-Scalable Control Architecture 

**Title (ZH)**: 基于安全意识和任务可扩展性的LVLM-MPC联合控制架构实现自主驾驶 

**Authors**: Kazuki Atsuta, Kohei Honda, Hiroyuki Okuda, Tatsuya Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04980)  

**Abstract**: This paper proposes a novel Large Vision-Language Model (LVLM) and Model Predictive Control (MPC) integration framework that delivers both task scalability and safety for Autonomous Driving (AD). LVLMs excel at high-level task planning across diverse driving scenarios. However, since these foundation models are not specifically designed for driving and their reasoning is not consistent with the feasibility of low-level motion planning, concerns remain regarding safety and smooth task switching. This paper integrates LVLMs with MPC Builder, which automatically generates MPCs on demand, based on symbolic task commands generated by the LVLM, while ensuring optimality and safety. The generated MPCs can strongly assist the execution or rejection of LVLM-driven task switching by providing feedback on the feasibility of the given tasks and generating task-switching-aware MPCs. Our approach provides a safe, flexible, and adaptable control framework, bridging the gap between cutting-edge foundation models and reliable vehicle operation. We demonstrate the effectiveness of our approach through a simulation experiment, showing that our system can safely and effectively handle highway driving while maintaining the flexibility and adaptability of LVLMs. 

**Abstract (ZH)**: 一种新型大型多模态模型与模型预测控制集成框架：自主驾驶中的任务可扩展性和安全性 

---
# AI and Vision based Autonomous Navigation of Nano-Drones in Partially-Known Environments 

**Title (ZH)**: 基于视觉的纳架无人机在部分已知环境中的自主导航技术 

**Authors**: Mattia Sartori, Chetna Singhal, Neelabhro Roy, Davide Brunelli, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2505.04972)  

**Abstract**: The miniaturisation of sensors and processors, the advancements in connected edge intelligence, and the exponential interest in Artificial Intelligence are boosting the affirmation of autonomous nano-size drones in the Internet of Robotic Things ecosystem. However, achieving safe autonomous navigation and high-level tasks such as exploration and surveillance with these tiny platforms is extremely challenging due to their limited resources. This work focuses on enabling the safe and autonomous flight of a pocket-size, 30-gram platform called Crazyflie 2.1 in a partially known environment. We propose a novel AI-aided, vision-based reactive planning method for obstacle avoidance under the ambit of Integrated Sensing, Computing and Communication paradigm. We deal with the constraints of the nano-drone by splitting the navigation task into two parts: a deep learning-based object detector runs on the edge (external hardware) while the planning algorithm is executed onboard. The results show the ability to command the drone at $\sim8$ frames-per-second and a model performance reaching a COCO mean-average-precision of $60.8$. Field experiments demonstrate the feasibility of the solution with the drone flying at a top speed of $1$ m/s while steering away from an obstacle placed in an unknown position and reaching the target destination. The outcome highlights the compatibility of the communication delay and the model performance with the requirements of the real-time navigation task. We provide a feasible alternative to a fully onboard implementation that can be extended to autonomous exploration with nano-drones. 

**Abstract (ZH)**: 基于集成传感、计算与通信范式的辅助AI视觉反应规划方法：实现30克级 biết寸平台 Crazyflie 2.1 的安全自主飞行 

---
# Visual Affordances: Enabling Robots to Understand Object Functionality 

**Title (ZH)**: 视觉功能特性：使机器人理解物体功能 

**Authors**: Tommaso Apicella, Alessio Xompero, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2505.05074)  

**Abstract**: Human-robot interaction for assistive technologies relies on the prediction of affordances, which are the potential actions a robot can perform on objects. Predicting object affordances from visual perception is formulated differently for tasks such as grasping detection, affordance classification, affordance segmentation, and hand-object interaction synthesis. In this work, we highlight the reproducibility issue in these redefinitions, making comparative benchmarks unfair and unreliable. To address this problem, we propose a unified formulation for visual affordance prediction, provide a comprehensive and systematic review of previous works highlighting strengths and limitations of methods and datasets, and analyse what challenges reproducibility. To favour transparency, we introduce the Affordance Sheet, a document to detail the proposed solution, the datasets, and the validation. As the physical properties of an object influence the interaction with the robot, we present a generic framework that links visual affordance prediction to the physical world. Using the weight of an object as an example for this framework, we discuss how estimating object mass can affect the affordance prediction. Our approach bridges the gap between affordance perception and robot actuation, and accounts for the complete information about objects of interest and how the robot interacts with them to accomplish its task. 

**Abstract (ZH)**: 人类辅助技术中的机器人交互依赖于对物体可用性的预测，即机器人能够对物体执行的潜在动作。不同任务（如抓取检测、可用性分类、可用性分割和手-物相互作用合成）中从视觉感知预测物体可用性的方法不同。在本文中，我们强调了这些重新定义中的重现性问题，使得比较基准不公平且不可靠。为了解决这一问题，我们提出了统一的视觉可用性预测公式，进行了全面而系统的前期研究，指出了方法和数据集的优点与局限性，分析了重现性受到的挑战。为了增加透明度，我们引入了“可用性表”，用于详细记录提议的解决方案、数据集和验证过程。由于物体的物理属性影响与机器人交互的方式，我们展示了将视觉可用性预测与物理世界相联系的通用框架。以物体重量为例，我们讨论了估计物体质量如何影响可用性预测。我们的方法弥合了感知可用性与机器人执行动作之间的差距，并考虑了目标物体的全部信息以及机器人如何与其交互以完成任务。 

---
# ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators 

**Title (ZH)**: 基于物理的运动模仿与对抗微分判别器 

**Authors**: Ziyu Zhang, Sergey Bashkirov, Dun Yang, Michael Taylor, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04961)  

**Abstract**: Multi-objective optimization problems, which require the simultaneous optimization of multiple terms, are prevalent across numerous applications. Existing multi-objective optimization methods often rely on manually tuned aggregation functions to formulate a joint optimization target. The performance of such hand-tuned methods is heavily dependent on careful weight selection, a time-consuming and laborious process. These limitations also arise in the setting of reinforcement-learning-based motion tracking for physically simulated characters, where intricately crafted reward functions are typically used to achieve high-fidelity results. Such solutions not only require domain expertise and significant manual adjustment, but also limit the applicability of the resulting reward function across diverse skills. To bridge this gap, we present a novel adversarial multi-objective optimization technique that is broadly applicable to a range of multi-objective optimization problems, including motion tracking. The proposed adversarial differential discriminator receives a single positive sample, yet is still effective at guiding the optimization process. We demonstrate that our technique can enable characters to closely replicate a variety of acrobatic and agile behaviors, achieving comparable quality to state-of-the-art motion-tracking methods, without relying on manually tuned reward functions. Results are best visualized through this https URL. 

**Abstract (ZH)**: 多目标优化问题要求同时优化多个目标，在众多应用中普遍存在。现有的多目标优化方法通常依赖手动调参的聚合函数来制定联合优化目标。这类手动调参的方法其性能高度依赖于权重的选择，这是一个耗时且劳动密集的过程。这些限制也同样存在于基于强化学习的动作跟踪中，通常需要精心设计奖励函数来实现高保真效果。这种解决方案不仅需要领域专业知识和大量的手动调整，还限制了所得奖励函数在不同技能中的应用。为了解决这个问题，我们提出了一种新型的对抗多目标优化技术，该技术广泛适用于多种多目标优化问题，包括动作跟踪。提出的对抗差分鉴别器仅接收一个正样本，但仍能有效地指导优化过程。我们的技术能够使角色精确再现各种杂技和敏捷行为，达到与当前最优动作跟踪方法相当的品质，而不依赖手动调参的奖励函数。结果可通过此链接可视化：https://www.example.com results。 

---
# Multi-agent Embodied AI: Advances and Future Directions 

**Title (ZH)**: 多智能体具身AI：进展与未来方向 

**Authors**: Zhaohan Feng, Ruiqi Xue, Lei Yuan, Yang Yu, Ning Ding, Meiqin Liu, Bingzhao Gao, Jian Sun, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05108)  

**Abstract**: Embodied artificial intelligence (Embodied AI) plays a pivotal role in the application of advanced technologies in the intelligent era, where AI systems are integrated with physical bodies that enable them to perceive, reason, and interact with their environments. Through the use of sensors for input and actuators for action, these systems can learn and adapt based on real-world feedback, allowing them to perform tasks effectively in dynamic and unpredictable environments. As techniques such as deep learning (DL), reinforcement learning (RL), and large language models (LLMs) mature, embodied AI has become a leading field in both academia and industry, with applications spanning robotics, healthcare, transportation, and manufacturing. However, most research has focused on single-agent systems that often assume static, closed environments, whereas real-world embodied AI must navigate far more complex scenarios. In such settings, agents must not only interact with their surroundings but also collaborate with other agents, necessitating sophisticated mechanisms for adaptation, real-time learning, and collaborative problem-solving. Despite increasing interest in multi-agent systems, existing research remains narrow in scope, often relying on simplified models that fail to capture the full complexity of dynamic, open environments for multi-agent embodied AI. Moreover, no comprehensive survey has systematically reviewed the advancements in this area. As embodied AI rapidly evolves, it is crucial to deepen our understanding of multi-agent embodied AI to address the challenges presented by real-world applications. To fill this gap and foster further development in the field, this paper reviews the current state of research, analyzes key contributions, and identifies challenges and future directions, providing insights to guide innovation and progress in this field. 

**Abstract (ZH)**: 具身人工智能（具身AI）在智能时代的先进技术应用中发挥着关键作用，其中AI系统结合了物理身体，使其能够感知、推理和与环境互动。通过使用传感器获取输入和执行器执行动作，这些系统可以根据实际反馈进行学习和适应，从而在动态和不可预测的环境中有效执行任务。随着深度学习（DL）、强化学习（RL）和大型语言模型（LLMs）等技术的成熟，具身AI已成为学术界和工业界的核心领域，其应用遍及机器人技术、医疗保健、交通运输和制造业。然而，大多数研究集中在单智能体系统上，通常假定静态的封闭环境，而现实中的具身AI必须导航更为复杂的场景。在这种环境中，智能体不仅需要与其环境互动，还需要与其他智能体协作，这需要复杂的适应机制、实时学习和协作问题解决机制。尽管对多智能体系统越来越感兴趣，但现有研究仍然局限于狭隘的范围，往往依赖于简化的模型，无法捕捉多智能体具身AI在动态开放环境中的全部复杂性。此外，尚未有全面的综述系统地回顾了这一领域的进展。随着具身AI的迅速发展，深化对多智能体具身AI的理解以应对实际应用挑战变得至关重要。为填补这一空白并促进该领域的进一步发展，本文回顾了当前的研究状态，分析了关键贡献，指出了面临的挑战和未来方向，提供了指导该领域创新和进步的见解。 

---
# Position: Epistemic Artificial Intelligence is Essential for Machine Learning Models to Know When They Do Not Know 

**Title (ZH)**: 位置：知识型人工智能对于机器学习模型在不知情时至关重要 

**Authors**: Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04950)  

**Abstract**: Despite the impressive achievements of AI, including advancements in generative models and large language models, there remains a significant gap in the ability of AI to handle uncertainty and generalize beyond the training data. We argue that AI models, especially in autonomous systems, fail to make robust predictions when faced with unfamiliar or adversarial data, as evidenced by incidents with autonomous vehicles. Traditional machine learning approaches struggle to address these issues due to an overemphasis on data fitting and domain adaptation. This position paper posits a paradigm shift towards epistemic artificial intelligence, emphasizing the need for models to learn not only from what they know but also from their ignorance. This approach, which focuses on recognizing and managing uncertainty, offers a potential solution to improve the resilience and robustness of AI systems, ensuring that they can better handle unpredictable real-world environments. 

**Abstract (ZH)**: 尽管人工智能取得了令人印象深刻的成就，包括生成模型和大规模语言模型的进步，但在处理不确定性以及泛化到训练数据之外的能力方面，人工智能仍然存在显著差距。本文认为，在自主系统中，当面对不熟悉或恶意的数据时，人工智能模型难以做出稳健的预测，这在自动驾驶车辆 incidents 中得到了体现。传统的机器学习方法由于过分强调数据拟合和领域适应，在解决这些问题上面临挑战。本文提出了一种范式转变，即认识论人工智能，强调模型不仅要从已知的信息中学习，还要从无知中学习。这种专注于识别和管理不确定性的方法，可能为提高人工智能系统的韧性和鲁棒性提供解决方案，确保它们能够更好地处理不可预测的现实环境。 

---
# Belief Filtering for Epistemic Control in Linguistic State Space 

**Title (ZH)**: 信念筛选在语言状态空间中的知识控制 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.04927)  

**Abstract**: We examine belief filtering as a mechanism for the epistemic control of artificial agents, focusing on the regulation of internal cognitive states represented as linguistic expressions. This mechanism is developed within the Semantic Manifold framework, where belief states are dynamic, structured ensembles of natural language fragments. Belief filters act as content-aware operations on these fragments across various cognitive transitions. This paper illustrates how the inherent interpretability and modularity of such a linguistically-grounded cognitive architecture directly enable belief filtering, offering a principled approach to agent regulation. The study highlights the potential for enhancing AI safety and alignment through structured interventions in an agent's internal semantic space and points to new directions for architecturally embedded cognitive governance. 

**Abstract (ZH)**: 我们探讨信念过滤作为一种机制，用于人工代理的epistemic控制，重点关注以语言表达形式表现的内部认知状态的调节。该机制在语义流形框架内发展，其中信念状态是自然语言片段构成的动力学、结构化集合。信念过滤器作为内容感知的操作，在各种认知转换过程中作用于这些片段。本文说明了这种以语言为基础的认知架构固有的可解释性和模块性如何直接促进信念过滤，提出了一种原理性的代理调节方法。研究突显了通过在代理内部语义空间中进行结构化干预来增强AI安全性和对齐的潜在可能性，并指出了嵌入式认知治理的新方向。 

---
# Computational Irreducibility as the Foundation of Agency: A Formal Model Connecting Undecidability to Autonomous Behavior in Complex Systems 

**Title (ZH)**: 计算不可约性作为agency的基础：将不可判定性与复杂系统中的自主行为正式模型连接起来 

**Authors**: Poria Azadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04646)  

**Abstract**: This article explores the emergence of autonomy and agency by connecting fundamental computational limits (decidability, completeness, computational irreducibility) with physical concepts. We introduce a formal model of a "minimal agent" operating within potentially Turing-complete environments. Using algorithmic information theory, we argue that the inherent undecidability and computational irreducibility of agent-environment interaction lead to unpredictability and novel information generation, enabling agency (effective goal-directed action). Computational irreducibility prevents full external prediction, creating necessary conditions for autonomous behavior. We relate this to computational sourcehood, where an agent is the irreducible origin of its behavior, though formalizing this concept remains challenging. Our central thesis, formally proven, is that genuine autonomy necessarily implies undecidability from an external perspective, distinguishing autonomous systems from predictable ones. We propose that agency arises when agent-environment coupling complexity allows mutual information between internal states and relevant environmental variables to increase, particularly where analytical solutions are absent and operational closure is needed for persistence. This framework links agency directly to the computational properties of interaction, offering implications for understanding consciousness, designing autonomous AI, and reconceptualizing free will in a deterministic yet computationally irreducible universe. 

**Abstract (ZH)**: 本文通过将基本的计算极限（决定性、完备性、计算不可约性）与物理概念相连，探讨自主性和能动力的涌现。我们介绍了一个“最小代理”形式模型，该代理在潜在图灵完备的环境中运行。利用算法信息理论，我们argue认为代理-环境交互的固有不可决定性和计算不可约性导致不可预测性和新颖信息的生成，从而实现能动力（有效目标导向行为）。计算不可约性阻止完全外部预测，从而创造自主行为的必要条件。我们将这一概念与计算源性相关联，其中代理是其行为的不可约来源，尽管正式化这一概念仍然具有挑战性。我们的中心论点，已被正式证明，是真正的自主性必然意味着从外部视角来看的不可决定性，从而将自主系统与可预测系统区分开来。我们提出，当代理-环境耦合的复杂性允许内部状态与相关环境变量之间的互信息增加时，能动力得以产生，特别是在缺乏解析解且需要操作闭包以维持行为持续性的情况下。这一框架直接将能动力与交互的计算性质相连，为其理解意识、设计自主AI以及在确定论但计算不可约的宇宙中重新概念化自由意志提供了意义。 

---
# Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models 

**Title (ZH)**: Biomed-DPT: 双模态提示调整方法用于生物医学视觉语言模型 

**Authors**: Wei Peng, Kang Liu, Jianchen Hu, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05189)  

**Abstract**: Prompt learning is one of the most effective paradigms for adapting pre-trained vision-language models (VLMs) to the biomedical image classification tasks in few shot scenarios. However, most of the current prompt learning methods only used the text prompts and ignored the particular structures (such as the complex anatomical structures and subtle pathological features) in the biomedical images. In this work, we propose Biomed-DPT, a knowledge-enhanced dual modality prompt tuning technique. In designing the text prompt, Biomed-DPT constructs a dual prompt including the template-driven clinical prompts and the large language model (LLM)-driven domain-adapted prompts, then extracts the clinical knowledge from the domain-adapted prompts through the knowledge distillation technique. In designing the vision prompt, Biomed-DPT introduces the zero vector as a soft prompt to leverage attention re-weighting so that the focus on non-diagnostic regions and the recognition of non-critical pathological features are avoided. Biomed-DPT achieves an average classification accuracy of 66.14\% across 11 biomedical image datasets covering 9 modalities and 10 organs, with performance reaching 78.06\% in base classes and 75.97\% in novel classes, surpassing the Context Optimization (CoOp) method by 6.20\%, 3.78\%, and 8.04\%, respectively. Our code are available at \underline{this https URL}. 

**Abstract (ZH)**: 基于知识增强的双模态提示调谐技术Biomed-DPT 

---
