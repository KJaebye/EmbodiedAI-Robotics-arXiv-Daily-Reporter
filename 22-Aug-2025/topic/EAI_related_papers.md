# Neural Robot Dynamics 

**Title (ZH)**: 神经机器人动力学 

**Authors**: Jie Xu, Eric Heiden, Iretiayo Akinola, Dieter Fox, Miles Macklin, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15755)  

**Abstract**: Accurate and efficient simulation of modern robots remains challenging due to their high degrees of freedom and intricate mechanisms. Neural simulators have emerged as a promising alternative to traditional analytical simulators, capable of efficiently predicting complex dynamics and adapting to real-world data; however, existing neural simulators typically require application-specific training and fail to generalize to novel tasks and/or environments, primarily due to inadequate representations of the global state. In this work, we address the problem of learning generalizable neural simulators for robots that are structured as articulated rigid bodies. We propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. We integrate the learned NeRD models as an interchangeable backend solver within a state-of-the-art robotics simulator. We conduct extensive experiments to show that the NeRD simulators are stable and accurate over a thousand simulation steps; generalize across tasks and environment configurations; enable policy learning exclusively in a neural engine; and, unlike most classical simulators, can be fine-tuned from real-world data to bridge the gap between simulation and reality. 

**Abstract (ZH)**: 现代机器人高效且精确的模拟仍然是一个挑战，由于它们具有高自由度和复杂的机械结构。神经模拟器作为传统解析模拟器的有前途的替代方案，能够高效预测复杂动力学并适应现实世界数据；然而，现有的神经模拟器通常需要特定应用的训练，并且难以泛化到新的任务和/or环境，主要是因为对全局状态的表示不足。在这项工作中，我们解决了学习可泛化的机器人神经模拟器的问题，这些机器人由铰接刚体结构组成。我们提出NeRD（Neural Robot Dynamics），这是一种学习到的针对铰接刚体的动态模型，用于在接触约束下预测未来状态。NeRD独特地替代了解析模拟器中的低级动力学和接触求解器，并采用以机器人为中心和空间不变的模拟状态表示。我们将学习到的NeRD模型作为可互换的后端求解器集成到最先进的机器人模拟器中。我们进行了一系列实验，结果显示，NeRD模拟器在一千个模拟步骤中稳定且准确；能够在不同任务和环境配置之间泛化；使策略学习仅在神经引擎中进行；并且，与大多数经典模拟器不同，可以从现实世界数据进行微调，以弥合模拟与现实之间的差距。 

---
# Exploiting Policy Idling for Dexterous Manipulation 

**Title (ZH)**: 利用策略空闲时间进行灵巧操作 

**Authors**: Annie S. Chen, Philemon Brakel, Antonia Bronars, Annie Xie, Sandy Huang, Oliver Groth, Maria Bauza, Markus Wulfmeier, Nicolas Heess, Dushyant Rao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15669)  

**Abstract**: Learning-based methods for dexterous manipulation have made notable progress in recent years. However, learned policies often still lack reliability and exhibit limited robustness to important factors of variation. One failure pattern that can be observed across many settings is that policies idle, i.e. they cease to move beyond a small region of states when they reach certain states. This policy idling is often a reflection of the training data. For instance, it can occur when the data contains small actions in areas where the robot needs to perform high-precision motions, e.g., when preparing to grasp an object or object insertion. Prior works have tried to mitigate this phenomenon e.g. by filtering the training data or modifying the control frequency. However, these approaches can negatively impact policy performance in other ways. As an alternative, we investigate how to leverage the detectability of idling behavior to inform exploration and policy improvement. Our approach, Pause-Induced Perturbations (PIP), applies perturbations at detected idling states, thus helping it to escape problematic basins of attraction. On a range of challenging simulated dual-arm tasks, we find that this simple approach can already noticeably improve test-time performance, with no additional supervision or training. Furthermore, since the robot tends to idle at critical points in a movement, we also find that learning from the resulting episodes leads to better iterative policy improvement compared to prior approaches. Our perturbation strategy also leads to a 15-35% improvement in absolute success rate on a real-world insertion task that requires complex multi-finger manipulation. 

**Abstract (ZH)**: 基于学习的灵巧操作方法在近年来取得了显著进展。然而，学习得到的策略在可靠性和对重要变化因素的鲁棒性方面仍然存在不足。一种常见的失败模式是策略在达到某些状态时会停滞，即它们会停止在状态空间的一个小区域内移动。这种策略停滞往往是训练数据的反映。例如，当数据中包含机器人需要进行高精度动作的区域中的小动作时，可能会发生这种情况，比如准备抓取物体或插入物体。先前的工作尝试通过过滤训练数据或修改控制频率来减轻这种现象的负面影响，但这些方法可能以其他方式负面影响策略的表现。作为替代方案，我们研究如何利用检测到的停滞行为可检测性来指导探索和策略改进。我们的方法，暂停引发扰动（PIP），在检测到的停滞状态下应用扰动，从而帮助其逃离不利的吸引子。在一系列具有挑战性的模拟双臂任务中，我们发现这种简单方法可以显著提高测试时的表现，无需额外的监督或训练。此外，由于机器人倾向于在动作的关键点处停滞，我们还发现从生成的 episode 中学习导致了与先前方法相比更好的迭代策略改进。我们提出的扰动策略在一项需要复杂多指操作的现实插入任务中绝对成功率提高了15-35%。 

---
# Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation 

**Title (ZH)**: 思维与运动一致：面向移动 manipulation 中任务规划与低层级策略的 IsaacSim 基准评测 

**Authors**: Nikita Kachaev, Andrei Spiridonov, Andrey Gorodetsky, Kirill Muravyev, Nikita Oskolkov, Aditya Narendra, Vlad Shakhuro, Dmitry Makarov, Aleksandr I. Panov, Polina Fedotova, Alexey K. Kovalev  

**Link**: [PDF](https://arxiv.org/pdf/2508.15663)  

**Abstract**: Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents. 

**Abstract (ZH)**: Kitchen-R：一种统一任务规划与低层级控制的模拟厨房环境基准 

---
# LLM-Driven Self-Refinement for Embodied Drone Task Planning 

**Title (ZH)**: 基于LLM的自主完善型无人机任务规划 

**Authors**: Deyu Zhang, Xicheng Zhang, Jiahao Li, Tingting Long, Xunhua Dai, Yongjian Fu, Jinrui Zhang, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15501)  

**Abstract**: We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at this https URL. 

**Abstract (ZH)**: 我们介绍了SRDrone，一种用于工业级实体无人机自我完善任务规划的新型系统。SRDrone包含两项关键技术贡献：首先，它采用连续状态评估方法，以稳健和准确地确定任务结果并提供解释性反馈。这种方法取代了依赖单一帧最终状态评估的做法，适用于连续动态无人机操作。其次，SRDrone实现了分层行为树(BT)修改模型。该模型将多级BT计划分析与受限策略空间相结合，以实现结构化的反思性学习。实验结果显示，与基线方法相比，SRDrone将成功率(SR)提高了44.87%。此外，通过迭代自我完善优化的经验基底在其实际部署中实现了96.25%的SR。通过在工业级BT规划框架中嵌入自适应任务完善能力，SRDrone有效结合了大型语言模型（LLMs）的通用推理智能与实体无人机固有的严格物理执行约束。代码可在以下链接获取。 

---
# Lang2Lift: A Framework for Language-Guided Pallet Detection and Pose Estimation Integrated in Autonomous Outdoor Forklift Operation 

**Title (ZH)**: Lang2Lift: 一种语言引导的托盘检测与姿态估计算法框架集成在自主室外仓储叉车操作中 

**Authors**: Huy Hoang Nguyen, Johannes Huemer, Markus Murschitz, Tobias Glueck, Minh Nhat Vu, Andreas Kugi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15427)  

**Abstract**: The logistics and construction industries face persistent challenges in automating pallet handling, especially in outdoor environments with variable payloads, inconsistencies in pallet quality and dimensions, and unstructured surroundings. In this paper, we tackle automation of a critical step in pallet transport: the pallet pick-up operation. Our work is motivated by labor shortages, safety concerns, and inefficiencies in manually locating and retrieving pallets under such conditions. We present Lang2Lift, a framework that leverages foundation models for natural language-guided pallet detection and 6D pose estimation, enabling operators to specify targets through intuitive commands such as "pick up the steel beam pallet near the crane." The perception pipeline integrates Florence-2 and SAM-2 for language-grounded segmentation with FoundationPose for robust pose estimation in cluttered, multi-pallet outdoor scenes under variable lighting. The resulting poses feed into a motion planning module for fully autonomous forklift operation. We validate Lang2Lift on the ADAPT autonomous forklift platform, achieving 0.76 mIoU pallet segmentation accuracy on a real-world test dataset. Timing and error analysis demonstrate the system's robustness and confirm its feasibility for deployment in operational logistics and construction environments. Video demonstrations are available at this https URL 

**Abstract (ZH)**: 物流和建筑业在自动托盘处理方面面临持续挑战，尤其是在具有变化负荷、托盘质量和尺寸不一致以及无结构环境的户外环境中。本文探讨了托盘运输中的关键步骤——托盘拾取操作的自动化。我们的研究动机源于劳动力短缺、安全问题以及在这种条件下手动定位和获取托盘的效率低下。我们提出了一种名为Lang2Lift的框架，该框架利用基础模型实现基于自然语言的托盘检测和6D姿态估计，使得操作员可以通过直观的指令（如“使用起重机附近的钢梁托盘”）指定目标。感知管道结合使用Florence-2和SAM-2进行语言指导的分割，并利用FoundationPose实现复杂多托盘户外场景中具有鲁棒性的姿态估计。估计的姿态输入到运动规划模块，实现全自动叉车操作。我们在ADAPT全自动叉车平台上验证了Lang2Lift，实测数据集的托盘分割准确率达到0.76 mIoU。时间分析和误差分析表明该系统的鲁棒性和在运营物流和建筑环境中的部署可行性。更多信息请参阅此链接。 

---
# Sensing, Social, and Motion Intelligence in Embodied Navigation: A Comprehensive Survey 

**Title (ZH)**: 具身导航中的传感、社会与运动智能综述 

**Authors**: Chaoran Xiong, Yulong Huang, Fangwen Yu, Changhao Chen, Yue Wang, Songpengchen Xia, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15354)  

**Abstract**: Embodied navigation (EN) advances traditional navigation by enabling robots to perform complex egocentric tasks through sensing, social, and motion intelligence. In contrast to classic methodologies that rely on explicit localization and pre-defined maps, EN leverages egocentric perception and human-like interaction strategies. This survey introduces a comprehensive EN formulation structured into five stages: Transition, Observation, Fusion, Reward-policy construction, and Action (TOFRA). The TOFRA framework serves to synthesize the current state of the art, provide a critical review of relevant platforms and evaluation metrics, and identify critical open research challenges. A list of studies is available at this https URL. 

**Abstract (ZH)**: 嵌入式导航（EN）通过感知、社会交互和运动智能使机器人能够执行复杂的第一人称任务，从而推动了传统导航的进步。与依赖显式定位和预制地图的经典方法不同，EN利用第一人称感知和类人的交互策略。本文综述将嵌入式导航结构化为五个阶段：转换、观察、融合、奖励策略构建和行动（TOFRA）。TOFRA框架旨在综合当前的研究成果，提供相关平台和评价指标的批判性评审，并识别关键的开放研究挑战。相关研究列表可在以下链接获取：this https URL。 

---
# Survey of Vision-Language-Action Models for Embodied Manipulation 

**Title (ZH)**: 视觉-语言-行动模型综述：赋能实体操纵 

**Authors**: Haoran Li, Yuhui Chen, Wenbo Cui, Weiheng Liu, Kai Liu, Mingcai Zhou, Zhengtao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15201)  

**Abstract**: Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions. 

**Abstract (ZH)**: 具身智能系统中的视觉-语言-动作模型：全面回顾及其关键挑战与未来研究方向 

---
# Hardware Implementation of a Zero-Prior-Knowledge Approach to Lifelong Learning in Kinematic Control of Tendon-Driven Quadrupeds 

**Title (ZH)**: 基于腱驱四足机器人运动控制的零先验知识 lifelong 学习硬件实现 

**Authors**: Hesam Azadjou, Suraj Chakravarthi Raja, Ali Marjaninejad, Francisco J. Valero-Cuevas  

**Link**: [PDF](https://arxiv.org/pdf/2508.15160)  

**Abstract**: Like mammals, robots must rapidly learn to control their bodies and interact with their environment despite incomplete knowledge of their body structure and surroundings. They must also adapt to continuous changes in both. This work presents a bio-inspired learning algorithm, General-to-Particular (G2P), applied to a tendon-driven quadruped robotic system developed and fabricated in-house. Our quadruped robot undergoes an initial five-minute phase of generalized motor babbling, followed by 15 refinement trials (each lasting 20 seconds) to achieve specific cyclical movements. This process mirrors the exploration-exploitation paradigm observed in mammals. With each refinement, the robot progressively improves upon its initial "good enough" solution. Our results serve as a proof-of-concept, demonstrating the hardware-in-the-loop system's ability to learn the control of a tendon-driven quadruped with redundancies in just a few minutes to achieve functional and adaptive cyclical non-convex movements. By advancing autonomous control in robotic locomotion, our approach paves the way for robots capable of dynamically adjusting to new environments, ensuring sustained adaptability and performance. 

**Abstract (ZH)**: 类哺乳动物的机器人必须在不完全了解自身结构和环境的情况下，迅速学会控制身体并与其环境交互，同时还需要适应两者持续的变化。本工作提出了一种受生物启发的学习算法——从一般到具体（General-to-Particular, G2P），应用于自主研发的肌腱驱动四足机器人系统。我们的四足机器人首先经历一个初始的五分钟通用运动babbling阶段，随后进行15次细化试验（每次20秒），以实现特定的周期性运动。这一过程反映了哺乳动物观察到的探索-利用范式。每次细化后，机器人都会逐步完善其初始的“足够好”的解决方案。我们的结果提供了概念验证，展示了闭环硬件系统仅在几分钟内就能学习控制具有冗余性的肌腱驱动四足机器人，并实现功能性、适应性的非凸周期性运动。通过在机器人行走自主控制方面的进展，本方法为能够动态适应新环境的机器人铺平了道路，确保持续的适应性和性能。 

---
# Open-Universe Assistance Games 

**Title (ZH)**: 开放宇宙辅助游戏 

**Authors**: Rachel Ma, Jingyi Qu, Andreea Bobu, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2508.15119)  

**Abstract**: Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations. 

**Abstract (ZH)**: 开放领域协助博弈：从开放对话中提取目标的高效在线方法 

---
# Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving 

**Title (ZH)**: 学习驾驶伦理：将道德推理嵌入自动驾驶 

**Authors**: Dianzhao Li, Ostap Okhrin  

**Link**: [PDF](https://arxiv.org/pdf/2508.14926)  

**Abstract**: Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL in real-world scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy in complex, human-mixed traffic environments. 

**Abstract (ZH)**: 自主驾驶车辆蕴含着降低交通事故死亡率和提升运输效率的巨大潜力，但其广泛采用取决于将坚实的伦理推理嵌入到常规和紧急操作中。在这里，我们提出了一种分层的安全强化学习（Safe RL）框架，该框架明确地将道德考虑与标准驾驶目标相结合。在决策层面，安全RL代理通过结合碰撞概率和伤害严重性复合伦理风险成本进行训练，以生成高级运动目标。动态优先经验回放机制增强了对罕见但关键的高风险事件的学习。在执行层面，多项式路径规划结合比例积分微分（PID）和斯坦利控制器将这些目标转化为平滑可行的轨迹，确保准确性和舒适性。我们通过涵盖多种车辆、自行车和行人的丰富实际交通数据集训练和验证了该方法，并证明它在降低伦理风险和保持驾驶性能方面优于基线方法。据我们所知，这是首次在真实场景中通过安全RL进行自主驾驶车辆的伦理决策研究。我们的结果强调了将形式控制理论与数据驱动学习相结合以在复杂混有人类的交通环境中推进可问责自主性的潜力。 

---
# Understanding Action Effects through Instrumental Empowerment in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过工具性授权理解行动效果在多代理 reinforcement 学习中的应用 

**Authors**: Ardian Selmonaj, Miroslav Strupl, Oleg Szehr, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2508.15652)  

**Abstract**: To reliably deploy Multi-Agent Reinforcement Learning (MARL) systems, it is crucial to understand individual agent behaviors within a team. While prior work typically evaluates overall team performance based on explicit reward signals or learned value functions, it is unclear how to infer agent contributions in the absence of any value feedback. In this work, we investigate whether meaningful insights into agent behaviors can be extracted that are consistent with the underlying value functions, solely by analyzing the policy distribution. Inspired by the phenomenon that intelligent agents tend to pursue convergent instrumental values, which generally increase the likelihood of task success, we introduce Intended Cooperation Values (ICVs), a method based on information-theoretic Shapley values for quantifying each agent's causal influence on their co-players' instrumental empowerment. Specifically, ICVs measure an agent's action effect on its teammates' policies by assessing their decision uncertainty and preference alignment. The analysis across cooperative and competitive MARL environments reveals the extent to which agents adopt similar or diverse strategies. By comparing action effects between policies and value functions, our method identifies which agent behaviors are beneficial to team success, either by fostering deterministic decisions or by preserving flexibility for future action choices. Our proposed method offers novel insights into cooperation dynamics and enhances explainability in MARL systems. 

**Abstract (ZH)**: 基于策略分布提取agents的意图合作价值以理解Multi-Agent Reinforcement Learning系统中的个体行为 

---
# Search-Based Credit Assignment for Offline Preference-Based Reinforcement Learning 

**Title (ZH)**: 基于搜索的信用分配用于离线基于偏好强化学习 

**Authors**: Xiancheng Gao, Yufeng Shi, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15327)  

**Abstract**: Offline reinforcement learning refers to the process of learning policies from fixed datasets, without requiring additional environment interaction. However, it often relies on well-defined reward functions, which are difficult and expensive to design. Human feedback is an appealing alternative, but its two common forms, expert demonstrations and preferences, have complementary limitations. Demonstrations provide stepwise supervision, but they are costly to collect and often reflect limited expert behavior modes. In contrast, preferences are easier to collect, but it is unclear which parts of a behavior contribute most to a trajectory segment, leaving credit assignment unresolved. In this paper, we introduce a Search-Based Preference Weighting (SPW) scheme to unify these two feedback sources. For each transition in a preference labeled trajectory, SPW searches for the most similar state-action pairs from expert demonstrations and directly derives stepwise importance weights based on their similarity scores. These weights are then used to guide standard preference learning, enabling more accurate credit assignment that traditional approaches struggle to achieve. We demonstrate that SPW enables effective joint learning from preferences and demonstrations, outperforming prior methods that leverage both feedback types on challenging robot manipulation tasks. 

**Abstract (ZH)**: 基于搜索的偏好加权（SPW）方案结合示范与偏好 

---
# Multiple Memory Systems for Enhancing the Long-term Memory of Agent 

**Title (ZH)**: 多记忆系统增强智能体的长期记忆 

**Authors**: Gaoke Zhang, Bo Wang, Yunlong Ma, Dongming Zhao, Zifei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15294)  

**Abstract**: An agent powered by large language models have achieved impressive results, but effectively handling the vast amounts of historical data generated during interactions remains a challenge. The current approach is to design a memory module for the agent to process these data. However, existing methods, such as MemoryBank and A-MEM, have poor quality of stored memory content, which affects recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multiple memory system (MMS) inspired by cognitive psychology theory. The system processes short-term memory to multiple long-term memory fragments, and constructs retrieval memory units and contextual memory units based on these fragments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. Experiments on LoCoMo dataset compared our method with three others, proving its effectiveness. Ablation studies confirmed the rationality of our memory units. We also analyzed the robustness regarding the number of selected memory segments and the storage overhead, demonstrating its practical value. 

**Abstract (ZH)**: 基于大型语言模型的智能体取得了显著成果，但有效地处理交互过程中产生的大量历史数据仍是一项挑战。当前的做法是为智能体设计一个记忆模块来处理这些数据。然而，现有的方法，如MemoryBank和A-MEM，存储的记忆内容质量较差，影响了检索性能和响应质量。为了更好地构建高质量的长期记忆内容，我们受到认知心理学理论的启发，设计了一个多记忆系统（MMS）。该系统将短期记忆处理为多个长期记忆片段，并基于这些片段构建检索记忆单元和上下文记忆单元，两者之间存在一一对应关系。在检索阶段，MMS将根据用户的查询匹配最相关的检索记忆单元。然后，相应的上下文记忆单元用作响应阶段的背景，以增强知识，从而有效利用历史数据。在LoCoMo数据集上的实验将我们的方法与三种其他方法进行了对比，证明了其有效性。消融研究表明了我们记忆单元的合理性。我们还分析了所选记忆片段数量以及存储开销的鲁棒性，展示了其实用价值。 

---
# Emergent Crowds Dynamics from Language-Driven Multi-Agent Interactions 

**Title (ZH)**: 语言驱动多agent交互中的涌现人群动力学 

**Authors**: Yibo Liu, Liam Shatzel, Brandon Haworth, Teseo Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2508.15047)  

**Abstract**: Animating and simulating crowds using an agent-based approach is a well-established area where every agent in the crowd is individually controlled such that global human-like behaviour emerges. We observe that human navigation and movement in crowds are often influenced by complex social and environmental interactions, driven mainly by language and dialogue. However, most existing work does not consider these dimensions and leads to animations where agent-agent and agent-environment interactions are largely limited to steering and fixed higher-level goal extrapolation.
We propose a novel method that exploits large language models (LLMs) to control agents' movement. Our method has two main components: a dialogue system and language-driven navigation. We periodically query agent-centric LLMs conditioned on character personalities, roles, desires, and relationships to control the generation of inter-agent dialogue when necessitated by the spatial and social relationships with neighbouring agents. We then use the conversation and each agent's personality, emotional state, vision, and physical state to control the navigation and steering of each agent. Our model thus enables agents to make motion decisions based on both their perceptual inputs and the ongoing dialogue.
We validate our method in two complex scenarios that exemplify the interplay between social interactions, steering, and crowding. In these scenarios, we observe that grouping and ungrouping of agents automatically occur. Additionally, our experiments show that our method serves as an information-passing mechanism within the crowd. As a result, our framework produces more realistic crowd simulations, with emergent group behaviours arising naturally from any environmental setting. 

**Abstract (ZH)**: 使用基于代理的方法进行人群动画和模拟：利用大型语言模型控制代理的运动 

---
# End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning 

**Title (ZH)**: 端到端自主RAG系统训练以实现可追溯的诊断推理 

**Authors**: Qiaoyu Zheng, Yuze Sun, Chaoyi Wu, Weike Zhao, Pengcheng Qiu, Yongguo Yu, Kun Sun, Yanfeng Wang, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15746)  

**Abstract**: Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL.
Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See this https URL. 

**Abstract (ZH)**: 医学大型语言模型进行准确诊断受限于知识缺口和幻觉。检索和工具增强的方法有所帮助，但其影响受限于外部知识的弱使用和反馈推理 traceability 差。为应对这些挑战，我们提出了 Deep-DxSearch，这是一个通过强化学习 (RL) 端到端训练的代理性 RAG 系统，能够引导可追踪的检索增强推理以进行医学诊断。在 Deep-DxSearch 中，我们首先构建了一个包含患者记录和可靠医学知识来源的大规模医学检索语料库，以支持诊断场景中的检索感知推理。更为关键的是，我们将 LLM 作为核心代理，将检索语料库作为其环境，并使用针对格式、检索、推理结构和诊断准确性定制的奖励，从而通过 RL 从大规模数据中进化代理性 RAG 策略。实验表明，我们的端到端代理性 RL 训练框架在多个数据中心上均优于提示工程和无需训练的 RAG 方法。训练后，Deep-DxSearch 在诊断准确性方面取得了显著提升，超越了如 GPT-4o、DeepSeek-R1 等强诊断基线方法，适用于常见和罕见疾病的诊断，无论是在分布内还是分布外场景。此外，奖励设计和检索语料库组件的消融研究证实了它们的关键作用，突显了我们方法的独特性和有效性，与传统的实现方式相比更加独特和有效。最后，案例研究和可解释性分析强调了 Deep-DxSearch 的诊断政策改进，提供了其性能提升的更深入理解，并支持临床医生提供更可靠和精确的初步诊断。 

---
