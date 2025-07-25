# Scene-Aware Conversational ADAS with Generative AI for Real-Time Driver Assistance 

**Title (ZH)**: 基于场景aware的生成AI实时驾驶辅助对话系统 

**Authors**: Kyungtae Han, Yitao Chen, Rohit Gupta, Onur Altintas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10500)  

**Abstract**: While autonomous driving technologies continue to advance, current Advanced Driver Assistance Systems (ADAS) remain limited in their ability to interpret scene context or engage with drivers through natural language. These systems typically rely on predefined logic and lack support for dialogue-based interaction, making them inflexible in dynamic environments or when adapting to driver intent. This paper presents Scene-Aware Conversational ADAS (SC-ADAS), a modular framework that integrates Generative AI components including large language models, vision-to-text interpretation, and structured function calling to enable real-time, interpretable, and adaptive driver assistance. SC-ADAS supports multi-turn dialogue grounded in visual and sensor context, allowing natural language recommendations and driver-confirmed ADAS control. Implemented in the CARLA simulator with cloud-based Generative AI, the system executes confirmed user intents as structured ADAS commands without requiring model fine-tuning. We evaluate SC-ADAS across scene-aware, conversational, and revisited multi-turn interactions, highlighting trade-offs such as increased latency from vision-based context retrieval and token growth from accumulated dialogue history. These results demonstrate the feasibility of combining conversational reasoning, scene perception, and modular ADAS control to support the next generation of intelligent driver assistance. 

**Abstract (ZH)**: 基于场景的对话式先进驾驶辅助系统（SC-ADAS） 

---
# Prompt Informed Reinforcement Learning for Visual Coverage Path Planning 

**Title (ZH)**: 基于提示强化学习的视觉覆盖路径规划 

**Authors**: Venkat Margapuri  

**Link**: [PDF](https://arxiv.org/pdf/2507.10284)  

**Abstract**: Visual coverage path planning with unmanned aerial vehicles (UAVs) requires agents to strategically coordinate UAV motion and camera control to maximize coverage, minimize redundancy, and maintain battery efficiency. Traditional reinforcement learning (RL) methods rely on environment-specific reward formulations that lack semantic adaptability. This study proposes Prompt-Informed Reinforcement Learning (PIRL), a novel approach that integrates the zero-shot reasoning ability and in-context learning capability of large language models with curiosity-driven RL. PIRL leverages semantic feedback from an LLM, GPT-3.5, to dynamically shape the reward function of the Proximal Policy Optimization (PPO) RL policy guiding the agent in position and camera adjustments for optimal visual coverage. The PIRL agent is trained using OpenAI Gym and evaluated in various environments. Furthermore, the sim-to-real-like ability and zero-shot generalization of the agent are tested by operating the agent in Webots simulator which introduces realistic physical dynamics. Results show that PIRL outperforms multiple learning-based baselines such as PPO with static rewards, PPO with exploratory weight initialization, imitation learning, and an LLM-only controller. Across different environments, PIRL outperforms the best-performing baseline by achieving up to 14% higher visual coverage in OpenAI Gym and 27% higher in Webots, up to 25% higher battery efficiency, and up to 18\% lower redundancy, depending on the environment. The results highlight the effectiveness of LLM-guided reward shaping in complex spatial exploration tasks and suggest a promising direction for integrating natural language priors into RL for robotics. 

**Abstract (ZH)**: 基于无人飞行器（UAV）的视觉覆盖路径规划：Prompt-Informed Reinforcement Learning (PIRL)的方法研究 

---
# Robust RL Control for Bipedal Locomotion with Closed Kinematic Chains 

**Title (ZH)**: 闭运动链下鲁棒的双足步行RL控制 

**Authors**: Egor Maslennikov, Eduard Zaliaev, Nikita Dudorov, Oleg Shamanin, Karanov Dmitry, Gleb Afanasev, Alexey Burkov, Egor Lygin, Simeon Nedelchev, Evgeny Ponomarev  

**Link**: [PDF](https://arxiv.org/pdf/2507.10164)  

**Abstract**: Developing robust locomotion controllers for bipedal robots with closed kinematic chains presents unique challenges, particularly since most reinforcement learning (RL) approaches simplify these parallel mechanisms into serial models during training. We demonstrate that this simplification significantly impairs sim-to-real transfer by failing to capture essential aspects such as joint coupling, friction dynamics, and motor-space control characteristics. In this work, we present an RL framework that explicitly incorporates closed-chain dynamics and validate it on our custom-built robot TopA. Our approach enhances policy robustness through symmetry-aware loss functions, adversarial training, and targeted network regularization. Experimental results demonstrate that our integrated approach achieves stable locomotion across diverse terrains, significantly outperforming methods based on simplified kinematic models. 

**Abstract (ZH)**: 具有闭链运动学的双足机器人鲁棒运动控制器开发面临独特挑战，特别是因为大多数强化学习（RL）方法在训练过程中将这些并行机制简化为串联模型。我们证明，这种简化严重削弱了模拟到现实世界的转移能力，无法捕捉到关节耦合、摩擦动力学和电机空间控制特性等关键方面。在此工作中，我们提出了一种明确纳入闭链动力学的RL框架，并在我们自建的TopA机器人上进行了验证。我们的方法通过对称意识损失函数、对抗训练和目标网络正则化来增强策略的鲁棒性。实验结果表明，我们的集成方法能够在多种地形上实现稳定的运动，显著优于基于简化运动学模型的方法。 

---
# Probabilistic Human Intent Prediction for Mobile Manipulation: An Evaluation with Human-Inspired Constraints 

**Title (ZH)**: 基于人类启发约束的移动操作中人类意图概率预测评估 

**Authors**: Cesar Alan Contreras, Manolis Chiou, Alireza Rastegarpanah, Michal Szulik, Rustam Stolkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10131)  

**Abstract**: Accurate inference of human intent enables human-robot collaboration without constraining human control or causing conflicts between humans and robots. We present GUIDER (Global User Intent Dual-phase Estimation for Robots), a probabilistic framework that enables a robot to estimate the intent of human operators. GUIDER maintains two coupled belief layers, one tracking navigation goals and the other manipulation goals. In the Navigation phase, a Synergy Map blends controller velocity with an occupancy grid to rank interaction areas. Upon arrival at a goal, an autonomous multi-view scan builds a local 3D cloud. The Manipulation phase combines U2Net saliency, FastSAM instance saliency, and three geometric grasp-feasibility tests, with an end-effector kinematics-aware update rule that evolves object probabilities in real-time. GUIDER can recognize areas and objects of intent without predefined goals. We evaluated GUIDER on 25 trials (five participants x five task variants) in Isaac Sim, and compared it with two baselines, one for navigation and one for manipulation. Across the 25 trials, GUIDER achieved a median stability of 93-100% during navigation, compared with 60-100% for the BOIR baseline, with an improvement of 39.5% in a redirection scenario (T5). During manipulation, stability reached 94-100% (versus 69-100% for Trajectron), with a 31.4% difference in a redirection task (T3). In geometry-constrained trials (manipulation), GUIDER recognized the object intent three times earlier than Trajectron (median remaining time to confident prediction 23.6 s vs 7.8 s). These results validate our dual-phase framework and show improvements in intent inference in both phases of mobile manipulation tasks. 

**Abstract (ZH)**: 准确推断人类意图使机器人能够在不限制人类控制或引起人类与机器人冲突的情况下进行协作：GUIDER（全局用户意图双阶段估计框架）在机器人中的应用 

---
# Simulations and experiments with assemblies of fiber-reinforced soft actuators 

**Title (ZH)**: 纤维增强软执行器组装的 simulations 和 experiments 

**Authors**: Seung Hyun Kim, Jiamiao Guo, Arman Tekinalp, Heng-Sheng Chang, Ugur Akcal, Tixian Wang, Darren Biskup, Benjamin Walt, Girish Chowdhary, Girish Krishnan, Prashant G. Mehta, Mattia Gazzola  

**Link**: [PDF](https://arxiv.org/pdf/2507.10121)  

**Abstract**: Soft continuum arms (SCAs) promise versatile manipulation through mechanical compliance, for assistive devices, agriculture, search applications, or surgery. However, SCAs' real-world use is challenging, partly due to their hard-to-control non-linear behavior. Here, a simulation framework for SCAs modularly assembled out of fiber reinforced elastomeric enclosures (FREEs) is developed and integrated with a video-tracking system for experimental testing and control design. 

**Abstract (ZH)**: 基于纤维增强弹性封装（FREEs）模块化组装的软连续臂模拟框架及其视频追踪系统集成 

---
# Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots 

**Title (ZH)**: 基于物理引导的神经网络与无迹卡尔曼滤波的人形机器人无传感器关节扭矩估计 

**Authors**: Ines Sorrentino, Giulio Romualdi, Lorenzo Moretti, Silvio Traversaro, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2507.10105)  

**Abstract**: This paper presents a novel framework for whole-body torque control of humanoid robots without joint torque sensors, designed for systems with electric motors and high-ratio harmonic drives. The approach integrates Physics-Informed Neural Networks (PINNs) for friction modeling and Unscented Kalman Filtering (UKF) for joint torque estimation, within a real-time torque control architecture. PINNs estimate nonlinear static and dynamic friction from joint and motor velocity readings, capturing effects like motor actuation without joint movement. The UKF utilizes PINN-based friction estimates as direct measurement inputs, improving torque estimation robustness. Experimental validation on the ergoCub humanoid robot demonstrates improved torque tracking accuracy, enhanced energy efficiency, and superior disturbance rejection compared to the state-of-the-art Recursive Newton-Euler Algorithm (RNEA), using a dynamic balancing experiment. The framework's scalability is shown by consistent performance across robots with similar hardware but different friction characteristics, without re-identification. Furthermore, a comparative analysis with position control highlights the advantages of the proposed torque control approach. The results establish the method as a scalable and practical solution for sensorless torque control in humanoid robots, ensuring torque tracking, adaptability, and stability in dynamic environments. 

**Abstract (ZH)**: 一种无关节扭矩传感器的 humanoid 机器人全身扭矩控制新型框架：基于物理导向神经网络和无迹卡尔曼滤波的实时扭矩控制 

---
# Foundation Model Driven Robotics: A Comprehensive Review 

**Title (ZH)**: 基础模型驱动的机器人技术：一篇全面综述 

**Authors**: Muhammad Tayyab Khan, Ammar Waheed  

**Link**: [PDF](https://arxiv.org/pdf/2507.10087)  

**Abstract**: The rapid emergence of foundation models, particularly Large Language Models (LLMs) and Vision-Language Models (VLMs), has introduced a transformative paradigm in robotics. These models offer powerful capabilities in semantic understanding, high-level reasoning, and cross-modal generalization, enabling significant advances in perception, planning, control, and human-robot interaction. This critical review provides a structured synthesis of recent developments, categorizing applications across simulation-driven design, open-world execution, sim-to-real transfer, and adaptable robotics. Unlike existing surveys that emphasize isolated capabilities, this work highlights integrated, system-level strategies and evaluates their practical feasibility in real-world environments. Key enabling trends such as procedural scene generation, policy generalization, and multimodal reasoning are discussed alongside core bottlenecks, including limited embodiment, lack of multimodal data, safety risks, and computational constraints. Through this lens, this paper identifies both the architectural strengths and critical limitations of foundation model-based robotics, highlighting open challenges in real-time operation, grounding, resilience, and trust. The review concludes with a roadmap for future research aimed at bridging semantic reasoning and physical intelligence through more robust, interpretable, and embodied models. 

**Abstract (ZH)**: 基础模型的快速兴起，特别是大规模语言模型（LLMs）和跨模态视觉语言模型（VLMs），在机器人技术中引入了变革性的范式。这些模型在语义理解、高层次推理和跨模态泛化方面提供了强大的能力，推动了感知、规划、控制和人机交互的重要进展。本文提供了一种结构化的综述，系统地总结了近期的发展，按模拟驱动设计、开放世界执行、模拟到现实的迁移以及可适应机器人技术对应用进行了分类。与现有的侧重于孤立能力的综述不同，本文强调了集成的、系统级的策略，并评估了这些策略在现实环境中的可行性。文章讨论了关键使能趋势，如过程化场景生成、策略泛化和多模态推理，同时还探讨了主要瓶颈，包括有限的具身性、缺乏多模态数据、安全风险和计算约束。通过这一视角，本文指出了基于基础模型的机器人技术的架构优势和关键局限，突出了实时操作、定位、韧性和信任方面的开放挑战。综述最后提出了一个未来研究路线图，旨在通过更稳健、可解释和具身的模型，弥合语义推理与物理智能之间的鸿沟。 

---
# Hand Gesture Recognition for Collaborative Robots Using Lightweight Deep Learning in Real-Time Robotic Systems 

**Title (ZH)**: 基于轻量级深度学习的实时机器人系统中手部手势识别 

**Authors**: Muhtadin, I Wayan Agus Darmawan, Muhammad Hilmi Rusydiansyah, I Ketut Eddy Purnama, Chastine Fatichah, Mauridhi Hery Purnomo  

**Link**: [PDF](https://arxiv.org/pdf/2507.10055)  

**Abstract**: Direct and natural interaction is essential for intuitive human-robot collaboration, eliminating the need for additional devices such as joysticks, tablets, or wearable sensors. In this paper, we present a lightweight deep learning-based hand gesture recognition system that enables humans to control collaborative robots naturally and efficiently. This model recognizes eight distinct hand gestures with only 1,103 parameters and a compact size of 22 KB, achieving an accuracy of 93.5%. To further optimize the model for real-world deployment on edge devices, we applied quantization and pruning using TensorFlow Lite, reducing the final model size to just 7 KB. The system was successfully implemented and tested on a Universal Robot UR5 collaborative robot within a real-time robotic framework based on ROS2. The results demonstrate that even extremely lightweight models can deliver accurate and responsive hand gesture-based control for collaborative robots, opening new possibilities for natural human-robot interaction in constrained environments. 

**Abstract (ZH)**: 直接且自然的人机交互对于直观的人机协同作业是必不可少的，这消除了对额外设备（如操纵杆、平板电脑或穿戴式传感器）的需求。本文提出了一种基于轻量级深度学习的手势识别系统，使人类能够自然且高效地控制协作机器人。该模型仅使用1,103个参数和22 KB的紧凑大小，准确率达到93.5%。为进一步优化模型以适应边缘设备的实际部署，我们使用TensorFlow Lite应用量化和剪枝技术，将最终模型大小减少到仅7 KB。该系统已在基于ROS2的实时机器人框架中成功实施并测试在一台UR5协作机器人上。结果表明，即使是非常轻量级的模型也能提供准确且响应迅速的手势控制，从而为受限环境中的自然人机交互开辟了新可能性。 

---
# Finetuning Deep Reinforcement Learning Policies with Evolutionary Strategies for Control of Underactuated Robots 

**Title (ZH)**: 使用进化策略 fine-tuning 深度强化学习策略以控制欠驱动机器人 

**Authors**: Marco Calì, Alberto Sinigaglia, Niccolò Turcato, Ruggero Carli, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2507.10030)  

**Abstract**: Deep Reinforcement Learning (RL) has emerged as a powerful method for addressing complex control problems, particularly those involving underactuated robotic systems. However, in some cases, policies may require refinement to achieve optimal performance and robustness aligned with specific task objectives. In this paper, we propose an approach for fine-tuning Deep RL policies using Evolutionary Strategies (ES) to enhance control performance for underactuated robots. Our method involves initially training an RL agent with Soft-Actor Critic (SAC) using a surrogate reward function designed to approximate complex specific scoring metrics. We subsequently refine this learned policy through a zero-order optimization step employing the Separable Natural Evolution Strategy (SNES), directly targeting the original score. Experimental evaluations conducted in the context of the 2nd AI Olympics with RealAIGym at IROS 2024 demonstrate that our evolutionary fine-tuning significantly improves agent performance while maintaining high robustness. The resulting controllers outperform established baselines, achieving competitive scores for the competition tasks. 

**Abstract (ZH)**: 深度强化学习（RL）已成为解决复杂控制问题的强大方法，尤其是涉及未饱和机器人系统的任务。然而，在某些情况下，策略可能需要进一步调整以实现与特定任务目标一致的最优性能和鲁棒性。本文提出了一种使用进化策略（ES）对深度RL策略进行微调的方法，以提高未饱和机器人控制性能。该方法包括使用设计用于近似复杂特定评分指标的代理奖励函数，先用Soft-Actor Critic (SAC)训练RL代理，然后通过使用可分离自然进化策略（SNES）的零阶优化步骤进一步优化此学习策略，直接针对原始评分进行优化。在2024年IROS举办的第二届AI奥运会RealAIGym竞赛环境中进行的实验评估表明，我们的进化微调显著提高了代理性能并保持了高鲁棒性。生成的控制器优于现有基准，实现了与竞赛任务相当的评分。 

---
# Demonstrating the Octopi-1.5 Visual-Tactile-Language Model 

**Title (ZH)**: 展示Octopi-1.5视觉-触觉-语言模型 

**Authors**: Samson Yu, Kelvin Lin, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2507.09985)  

**Abstract**: Touch is recognized as a vital sense for humans and an equally important modality for robots, especially for dexterous manipulation, material identification, and scenarios involving visual occlusion. Building upon very recent work in touch foundation models, this demonstration will feature Octopi-1.5, our latest visual-tactile-language model. Compared to its predecessor, Octopi-1.5 introduces the ability to process tactile signals from multiple object parts and employs a simple retrieval-augmented generation (RAG) module to improve performance on tasks and potentially learn new objects on-the-fly. The system can be experienced live through a new handheld tactile-enabled interface, the TMI, equipped with GelSight and TAC-02 tactile sensors. This convenient and accessible setup allows users to interact with Octopi-1.5 without requiring a robot. During the demonstration, we will showcase Octopi-1.5 solving tactile inference tasks by leveraging tactile inputs and commonsense knowledge. For example, in a Guessing Game, Octopi-1.5 will identify objects being grasped and respond to follow-up queries about how to handle it (e.g., recommending careful handling for soft fruits). We also plan to demonstrate Octopi-1.5's RAG capabilities by teaching it new items. With live interactions, this demonstration aims to highlight both the progress and limitations of VTLMs such as Octopi-1.5 and to foster further interest in this exciting field. Code for Octopi-1.5 and design files for the TMI gripper are available at this https URL. 

**Abstract (ZH)**: 触觉被认定为人类的一个重要感官，对于机器人来说，特别是在灵巧操作、材料识别以及涉及视觉遮挡的场景中，触觉也是一个同等重要的模态。基于近期的触觉基础模型研究，我们将展示Octopi-1.5，这是我们的最新视觉-触觉-语言模型。相比其前代产品，Octipi-1.5 增强了处理多个物体部位触觉信号的能力，并采用简单的检索增强生成（RAG）模块以提高任务性能，并且有可能在不预先训练的情况下学习新物体。用户可以通过新的便携式触觉接口TMI直接与系统互动，该接口配备了GelSight和TAC-02触觉传感器，无需机器人即可操作。在展示中，我们将通过利用触觉输入和常识知识，展示Octopi-1.5解决触觉推理任务的能力。例如，在一个猜物游戏中，Octopi-1.5会识别被握住的物体，并回应关于如何处理该物体的后续查询（如推荐小心处理软水果）。我们还将演示Octopi-1.5的RAG能力，通过教导它新物品。通过现场互动，本次展示旨在突出如Octopi-1.5这样的VTLM的进展和局限性，并进一步激发对该领域兴趣。Octopi-1.5的代码和TMI夹爪的设计文件可在<a href="这个链接">此处</a>获取。 

---
# Visual Homing in Outdoor Robots Using Mushroom Body Circuits and Learning Walks 

**Title (ZH)**: 户外机器人使用蘑菇体电路和学习行走的视觉归巢技术 

**Authors**: Gabriel G. Gattaux, Julien R. Serres, Franck Ruffier, Antoine Wystrach  

**Link**: [PDF](https://arxiv.org/pdf/2507.09725)  

**Abstract**: Ants achieve robust visual homing with minimal sensory input and only a few learning walks, inspiring biomimetic solutions for autonomous navigation. While Mushroom Body (MB) models have been used in robotic route following, they have not yet been applied to visual homing. We present the first real-world implementation of a lateralized MB architecture for visual homing onboard a compact autonomous car-like robot. We test whether the sign of the angular path integration (PI) signal can categorize panoramic views, acquired during learning walks and encoded in the MB, into "goal on the left" and "goal on the right" memory banks, enabling robust homing in natural outdoor settings. We validate this approach through four incremental experiments: (1) simulation showing attractor-like nest dynamics; (2) real-world homing after decoupled learning walks, producing nest search behavior; (3) homing after random walks using noisy PI emulated with GPS-RTK; and (4) precise stopping-at-the-goal behavior enabled by a fifth MB Output Neuron (MBON) encoding goal-views to control velocity. This mimics the accurate homing behavior of ants and functionally resembles waypoint-based position control in robotics, despite relying solely on visual input. Operating at 8 Hz on a Raspberry Pi 4 with 32x32 pixel views and a memory footprint under 9 kB, our system offers a biologically grounded, resource-efficient solution for autonomous visual homing. 

**Abstract (ZH)**: 蚂蚁通过最少的感觉输入和几次学习行走实现稳健的视觉归巢，这启发了自主导航的生物模拟解决方案。虽然蘑菇体（MB）模型在机器人路径跟随中已被使用，但尚未应用于视觉归巢。我们首次在紧凑型自主车形机器人上实现了偏侧化MB架构的实地视觉归巢实施。我们测试角路径积分（PI）信号的符号是否能够分类学习行走期间获取并编码在蘑菇体中的全景视图，形成“目标在左”和“目标在右”的记忆库，从而在自然户外环境中实现稳健的归巢。我们通过四个逐步实验验证了这一方法：（1）仿真显示类似吸引子的巢穴动力学；（2）解耦学习行走后的实地归巢，产生巢穴搜索行为；（3）使用GPS-RTK模拟噪声路径积分的随机行走归巢；（4）通过第五个蘑菇体输出神经元（MBON）编码目标视图来控制速度，实现精确的到达目标行为。这模拟了蚂蚁精确的归巢行为，并在很大程度上类似于机器人中基于航点的位置控制，尽管仅依赖于视觉输入。以8 Hz运行在Raspberry Pi 4上，具有32x32像素视图和内存占用小于9 kB，我们的系统提供了一种基于生物学的、资源高效的自主视觉归巢解决方案。 

---
# Constrained Style Learning from Imperfect Demonstrations under Task Optimality 

**Title (ZH)**: 在任务最优性下的受限风格学习从不完美示范 

**Authors**: Kehan Wen, Chenhao Li, Junzhe He, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.09371)  

**Abstract**: Learning from demonstration has proven effective in robotics for acquiring natural behaviors, such as stylistic motions and lifelike agility, particularly when explicitly defining style-oriented reward functions is challenging. Synthesizing stylistic motions for real-world tasks usually requires balancing task performance and imitation quality. Existing methods generally depend on expert demonstrations closely aligned with task objectives. However, practical demonstrations are often incomplete or unrealistic, causing current methods to boost style at the expense of task performance. To address this issue, we propose formulating the problem as a constrained Markov Decision Process (CMDP). Specifically, we optimize a style-imitation objective with constraints to maintain near-optimal task performance. We introduce an adaptively adjustable Lagrangian multiplier to guide the agent to imitate demonstrations selectively, capturing stylistic nuances without compromising task performance. We validate our approach across multiple robotic platforms and tasks, demonstrating both robust task performance and high-fidelity style learning. On ANYmal-D hardware we show a 14.5% drop in mechanical energy and a more agile gait pattern, showcasing real-world benefits. 

**Abstract (ZH)**: 从演示学习在机器人领域证明了能够有效获取自然行为，如风格化的运动和生机勃勃的敏捷性，尤其是在明确定义风格导向的奖励函数具有挑战性的情况下。为了合成适用于真实世界任务的风格化运动，通常需要在任务性能和模仿质量之间进行权衡。现有的方法通常依赖于紧密符合任务目标的专家演示。然而，实际的演示往往是不完整或不现实的，导致当前的方法在提高风格的同时牺牲了任务性能。为了解决这一问题，我们提出将问题形式化为受限马尔可夫决策过程（CMDP）。具体而言，我们优化一个风格模仿目标，并通过约束条件来保持近似最优的任务性能。我们引入了一个可自适应调整的拉格朗日乘数，以引导代理选择性地模仿演示，捕捉风格化细节而不牺牲任务性能。我们在多个机器人平台上和任务中验证了该方法，展示了鲁棒的任务性能和高保真的风格学习。在ANYmal-D硬件上，我们展示了14.5%的机械能量下降和更加敏捷的步伐模式，展示了其实用价值。 

---
# PRAG: Procedural Action Generator 

**Title (ZH)**: 普动作生成器 

**Authors**: Michal Vavrecka, Radoslav Skoviera, Gabriela Sejnova, Karla Stepanova  

**Link**: [PDF](https://arxiv.org/pdf/2507.09167)  

**Abstract**: We present a novel approach for the procedural construction of multi-step contact-rich manipulation tasks in robotics. Our generator takes as input user-defined sets of atomic actions, objects, and spatial predicates and outputs solvable tasks of a given length for the selected robotic environment. The generator produces solvable tasks by constraining all possible (nonsolvable) combinations by symbolic and physical validation. The symbolic validation checks each generated sequence for logical and operational consistency, and also the suitability of object-predicate relations. Physical validation checks whether tasks can be solved in the selected robotic environment. Only the tasks that passed both validators are retained. The output from the generator can be directly interfaced with any existing framework for training robotic manipulation tasks, or it can be stored as a dataset of curated robotic tasks with detailed information about each task. This is beneficial for RL training as there are dense reward functions and initial and goal states paired with each subgoal. It allows the user to measure the semantic similarity of all generated tasks. We tested our generator on sequences of up to 15 actions resulting in millions of unique solvable multi-step tasks. 

**Abstract (ZH)**: 我们提出了一种用于机器人多步骤接触丰富操作任务程序化构造的新方法。生成器接受用户定义的基本动作集、物体集和空间谓词作为输入，并输出选定机器人环境下的可解任务序列，任务长度为给定长度。生成器通过符号验证和物理验证来生成可解任务，限制所有可能的（不可解）组合。符号验证检查每个生成的序列的逻辑和操作一致性，以及物体-谓词关系的适当性；物理验证检查任务在选定的机器人环境中的可解性。只有通过两个验证器的任务才被保留。生成器的输出可以直接与任何现有的机器人操作任务训练框架对接，或者作为包含每个任务详细信息的精心设计的任务数据集存储。这对于基于强化学习的训练是有益的，因为每个子目标都有稠密的奖励函数和初始状态及目标状态。它允许用户测量所有生成任务的语义相似性。我们在多达15个动作的序列上测试了生成器，结果产生了数百万个独特的多步骤可解任务。 

---
# Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization 

**Title (ZH)**: 触觉-VLA：解锁视觉-语言-行动模型的物理知识以实现触觉泛化 

**Authors**: Jialei Huang, Shuo Wang, Fanqi Lin, Yihang Hu, Chuan Wen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09160)  

**Abstract**: Vision-Language-Action (VLA) models have shown remarkable achievements, driven by the rich implicit knowledge of their vision-language components. However, achieving generalist robotic agents demands precise grounding into physical interactions, especially in contact-rich scenarios where fine-grained force control is essential. We advance VLAs' implicit knowledge beyond identifying what to do, towards guiding how to physically interact with real world. This paper introduces Tactile-VLA, a novel framework that deeply fuses vision, language, action, and tactile sensing. This framework incorporates a hybrid position-force controller to translate the model's intentions into precise physical actions and a reasoning module that allows the robot to adapt its strategy based on tactile feedback. Experiments demonstrate Tactile-VLA's effectiveness and generalizability in three key aspects: (1) enabling tactile-aware instruction following, (2) utilizing tactile-relevant commonsense, and (3) facilitating adaptive tactile-involved reasoning. A key finding is that the VLM's prior knowledge already contains semantic understanding of physical interaction; by connecting it to the robot's tactile sensors with only a few demonstrations, we can activate this prior knowledge to achieve zero-shot generalization in contact-rich tasks. 

**Abstract (ZH)**: 触觉-Vision-Language-Action (Tactile-VLA) 框架 

---
# Towards Human-level Dexterity via Robot Learning 

**Title (ZH)**: 通过机器人学习实现人类水平的灵巧性 

**Authors**: Gagan Khandate  

**Link**: [PDF](https://arxiv.org/pdf/2507.09117)  

**Abstract**: Dexterous intelligence -- the ability to perform complex interactions with multi-fingered hands -- is a pinnacle of human physical intelligence and emergent higher-order cognitive skills. However, contrary to Moravec's paradox, dexterous intelligence in humans appears simple only superficially. Many million years were spent co-evolving the human brain and hands including rich tactile sensing. Achieving human-level dexterity with robotic hands has long been a fundamental goal in robotics and represents a critical milestone toward general embodied intelligence. In this pursuit, computational sensorimotor learning has made significant progress, enabling feats such as arbitrary in-hand object reorientation. However, we observe that achieving higher levels of dexterity requires overcoming very fundamental limitations of computational sensorimotor learning.
I develop robot learning methods for highly dexterous multi-fingered manipulation by directly addressing these limitations at their root cause. Chiefly, through key studies, this disseration progressively builds an effective framework for reinforcement learning of dexterous multi-fingered manipulation skills. These methods adopt structured exploration, effectively overcoming the limitations of random exploration in reinforcement learning. The insights gained culminate in a highly effective reinforcement learning that incorporates sampling-based planning for direct exploration. Additionally, this thesis explores a new paradigm of using visuo-tactile human demonstrations for dexterity, introducing corresponding imitation learning techniques. 

**Abstract (ZH)**: 灵巧智能——多指手进行复杂交互的能力是人类物理智能和高级认知技能的顶峰。然而，与莫拉韦克悖论相反，人类的灵巧智能在表面上看似简单，但实际上经过了数百万年的大脑和手的协同进化，包含了丰富的触觉感知。实现类人的灵巧智能一直是机器人领域的根本目标，并代表着通向一般 embodded 智能的关键里程碑。在这一追求中，计算感知运动学习取得了显著进展，能够实现任意的手中物体重新定向等壮举。然而，我们观察到，实现更高的灵巧水平需要克服计算感知运动学习的基本局限性。

我通过直接针对这些局限性的根本原因，开发了机器人学习方法以实现高度灵巧的多指操纵。主要地，通过关键研究，本论文逐步建立了一个有效的强化学习框架，用于学习灵巧的多指操纵技能。这些方法采用结构化探索，有效地克服了强化学习中随机探索的局限性。获得的洞见最终在结合采样规划直接探索的强化学习中得到了充分体现。此外，本论文还探讨了一种新的使用视触觉人类示范进行灵巧的新范式，并引入相应的模仿学习技术。 

---
# AirScape: An Aerial Generative World Model with Motion Controllability 

**Title (ZH)**: AirScape: 一种具备运动可控性的空中生成世界模型 

**Authors**: Baining Zhao, Rongze Tang, Mingyuan Jia, Ziyou Wang, Fanghang Man, Xin Zhang, Yu Shang, Weichen Zhang, Chen Gao, Wei Wu, Xin Wang, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08885)  

**Abstract**: How to enable robots to predict the outcomes of their own motion intentions in three-dimensional space has been a fundamental problem in embodied intelligence. To explore more general spatial imagination capabilities, here we present AirScape, the first world model designed for six-degree-of-freedom aerial agents. AirScape predicts future observation sequences based on current visual inputs and motion intentions. Specifically, we construct an dataset for aerial world model training and testing, which consists of 11k video-intention pairs. This dataset includes first-person-view videos capturing diverse drone actions across a wide range of scenarios, with over 1,000 hours spent annotating the corresponding motion intentions. Then we develop a two-phase training schedule to train a foundation model -- initially devoid of embodied spatial knowledge -- into a world model that is controllable by motion intentions and adheres to physical spatio-temporal constraints. 

**Abstract (ZH)**: 如何使机器人在三维空间中预测自身运动意图的结果是嵌入式智能领域的基础问题。为了探索更广泛的空间想象能力，我们提出了AirScape，这是第一个为六自由度飞行代理设计的世界模型。AirScape根据当前视觉输入和运动意图预测未来观测序列。具体地，我们构建了一个用于飞行世界模型训练和测试的数据集，包含11000个视频-意图配对。该数据集包括第一人称视角视频，涵盖了多种无人机动作和广泛场景，相应的运动意图标注时间超过1000小时。然后我们开发了一个两阶段训练计划，将一个初始缺乏嵌入式空间知识的基础模型训练成一个可通过运动意图控制且遵循物理时空约束的世界模型。 

---
# Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation 

**Title (ZH)**: 基于半监督联邦学习和机器人视觉确认的隐私保护多阶段跌倒检测框架 

**Authors**: Seyed Alireza Rahimi Azghadi, Truong-Thanh-Hung Nguyen, Helene Fournier, Monica Wachowicz, Rene Richard, Francis Palma, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10474)  

**Abstract**: The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time. However, to provide timely intervention and avoid unnecessary alarms, detection systems must be effective and reliable while addressing privacy concerns regarding the user. In this work, we propose a framework for detecting falls using several complementary systems: a semi-supervised federated learning-based fall detection system (SF2D), an indoor localization and navigation system, and a vision-based human fall recognition system. A wearable device and an edge device identify a fall scenario in the first system. On top of that, the second system uses an indoor localization technique first to localize the fall location and then navigate a robot to inspect the scenario. A vision-based detection system running on an edge device with a mounted camera on a robot is used to recognize fallen people. Each of the systems of this proposed framework achieves different accuracy rates. Specifically, the SF2D has a 0.81% failure rate equivalent to 99.19% accuracy, while the vision-based fallen people detection achieves 96.3% accuracy. However, when we combine the accuracy of these two systems with the accuracy of the navigation system (95% success rate), our proposed framework creates a highly reliable performance for fall detection, with an overall accuracy of 99.99%. Not only is the proposed framework safe for older adults, but it is also a privacy-preserving solution for detecting falls. 

**Abstract (ZH)**: 老龄化人口快速增长，老年人跌倒的风险也在增加。跌倒是导致伤害的主要原因，及时检测可以大大节省医疗费用和恢复时间。然而，为了提供及时干预并避免不必要的警报，检测系统必须在解决用户隐私问题的同时具备有效性与可靠性。本文提出了一种利用多种互补系统的跌倒检测框架：基于半监督联邦学习的跌倒检测系统（SF2D）、室内定位与导航系统以及基于视觉的人体跌倒识别系统。第一种系统通过可穿戴设备和边缘设备识别跌倒场景。在此基础上，第二种系统使用室内定位技术首先定位跌倒地点，然后导航机器人检查场景。边缘设备搭载摄像头的机器人上运行的基于视觉的检测系统用于识别跌倒人员。该框架中的每个系统都实现了不同的准确率。具体而言，SF2D 的失败率为 0.81%，相当于 99.19% 的准确率，基于视觉的跌倒人员识别准确率为 96.3%。然而，当我们将这两种系统的准确率与导航系统（95% 的成功率）的准确率结合起来时，我们提出的框架在跌倒检测方面创造了高度可靠的性能，总体准确率为 99.99%。不仅该框架对老年人安全有效，还是一种保护隐私的跌倒检测解决方案。 

---
# Towards Emotion Co-regulation with LLM-powered Socially Assistive Robots: Integrating LLM Prompts and Robotic Behaviors to Support Parent-Neurodivergent Child Dyads 

**Title (ZH)**: 基于LLM赋能的社会辅助机器人的情感共调节：结合LLM提示和机器人行为以支持家长-神经多样性儿童互动对 

**Authors**: Jing Li, Felix Schijve, Sheng Li, Yuye Yang, Jun Hu, Emilia Barakova  

**Link**: [PDF](https://arxiv.org/pdf/2507.10427)  

**Abstract**: Socially Assistive Robotics (SAR) has shown promise in supporting emotion regulation for neurodivergent children. Recently, there has been increasing interest in leveraging advanced technologies to assist parents in co-regulating emotions with their children. However, limited research has explored the integration of large language models (LLMs) with SAR to facilitate emotion co-regulation between parents and children with neurodevelopmental disorders. To address this gap, we developed an LLM-powered social robot by deploying a speech communication module on the MiRo-E robotic platform. This supervised autonomous system integrates LLM prompts and robotic behaviors to deliver tailored interventions for both parents and neurodivergent children. Pilot tests were conducted with two parent-child dyads, followed by a qualitative analysis. The findings reveal MiRo-E's positive impacts on interaction dynamics and its potential to facilitate emotion regulation, along with identified design and technical challenges. Based on these insights, we provide design implications to advance the future development of LLM-powered SAR for mental health applications. 

**Abstract (ZH)**: 社交辅助机器人（SAR）在支持神经多样性儿童的情绪调节方面展现了潜力。近年来，越来越多的研究关注利用先进技术帮助父母与其孩子共同调节情绪。然而，有限的研究探索了将大型语言模型（LLMs）与SAR集成以促进神经发育障碍儿童与其父母之间的情绪共同调节。为弥补这一空白，我们在MiRo-E机器人平台上部署语音通信模块，开发出一种基于LLM的社会机器人。该监督自主系统将LLM提示与机器人行为结合，为父母和神经多样性儿童提供个性化干预。我们进行了初步测试，并进行了定性分析。研究结果表明MiRo-E对互动动态的积极影响及其在促进情绪调节方面的潜力，同时也指出了设计和技术挑战。基于这些洞见，我们提供了设计建议，以推动未来基于LLM的SAR在心理健康应用中的发展。 

---
# Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs 

**Title (ZH)**: 跨足机器人：从感知到行动的多模态语言模型与知识图谱桥梁 

**Authors**: Margherita Martorana, Francesca Urgese, Mark Adamik, Ilaria Tiddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09617)  

**Abstract**: Personal service robots are deployed to support daily living in domestic environments, particularly for elderly and individuals requiring assistance. These robots must perceive complex and dynamic surroundings, understand tasks, and execute context-appropriate actions. However, current systems rely on proprietary, hard-coded solutions tied to specific hardware and software, resulting in siloed implementations that are difficult to adapt and scale across platforms. Ontologies and Knowledge Graphs (KGs) offer a solution to enable interoperability across systems, through structured and standardized representations of knowledge and reasoning. However, symbolic systems such as KGs and ontologies struggle with raw and noisy sensory input. In contrast, multimodal language models are well suited for interpreting input such as images and natural language, but often lack transparency, consistency, and knowledge grounding. In this work, we propose a neurosymbolic framework that combines the perceptual strengths of multimodal language models with the structured representations provided by KGs and ontologies, with the aim of supporting interoperability in robotic applications. Our approach generates ontology-compliant KGs that can inform robot behavior in a platform-independent manner. We evaluated this framework by integrating robot perception data, ontologies, and five multimodal models (three LLaMA and two GPT models), using different modes of neural-symbolic interaction. We assess the consistency and effectiveness of the generated KGs across multiple runs and configurations, and perform statistical analyzes to evaluate performance. Results show that GPT-o1 and LLaMA 4 Maverick consistently outperform other models. However, our findings also indicate that newer models do not guarantee better results, highlighting the critical role of the integration strategy in generating ontology-compliant KGs. 

**Abstract (ZH)**: 基于神经符号框架的多模态语言模型与知识图谱及本体的结合在机器人应用中的互操作性支持 

---
# GenAI-based Multi-Agent Reinforcement Learning towards Distributed Agent Intelligence: A Generative-RL Agent Perspective 

**Title (ZH)**: 基于GenAI的多智能体强化学习 toward 分布式智能体智能：生成式-RL智能体视角 

**Authors**: Hang Wang, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09495)  

**Abstract**: Multi-agent reinforcement learning faces fundamental challenges that conventional approaches have failed to overcome: exponentially growing joint action spaces, non-stationary environments where simultaneous learning creates moving targets, and partial observability that constrains coordination. Current methods remain reactive, employing stimulus-response mechanisms that fail when facing novel scenarios. We argue for a transformative paradigm shift from reactive to proactive multi-agent intelligence through generative AI-based reinforcement learning. This position advocates reconceptualizing agents not as isolated policy optimizers, but as sophisticated generative models capable of synthesizing complex multi-agent dynamics and making anticipatory decisions based on predictive understanding of future interactions. Rather than responding to immediate observations, generative-RL agents can model environment evolution, predict other agents' behaviors, generate coordinated action sequences, and engage in strategic reasoning accounting for long-term dynamics. This approach leverages pattern recognition and generation capabilities of generative AI to enable proactive decision-making, seamless coordination through enhanced communication, and dynamic adaptation to evolving scenarios. We envision this paradigm shift will unlock unprecedented possibilities for distributed intelligence, moving beyond individual optimization toward emergent collective behaviors representing genuine collaborative intelligence. The implications extend across autonomous systems, robotics, and human-AI collaboration, promising solutions to coordination challenges intractable under traditional reactive frameworks. 

**Abstract (ZH)**: 基于生成AI的强化学习：多代理智能从被动到主动的范式转型 

---
# RoHOI: Robustness Benchmark for Human-Object Interaction Detection 

**Title (ZH)**: RoHOI: 人类物体交互检测的 robustness 基准 

**Authors**: Di Wen, Kunyu Peng, Kailun Yang, Yufan Chen, Ruiping Liu, Junwei Zheng, Alina Roitberg, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09111)  

**Abstract**: Human-Object Interaction (HOI) detection is crucial for robot-human assistance, enabling context-aware support. However, models trained on clean datasets degrade in real-world conditions due to unforeseen corruptions, leading to inaccurate prediction. To address this, we introduce the first robustness benchmark for HOI detection, evaluating model resilience under diverse challenges. Despite advances, current models struggle with environmental variability, occlusion, and noise. Our benchmark, RoHOI, includes 20 corruption types based on HICO-DET and V-COCO datasets and a new robustness-focused metric. We systematically analyze existing models in the related field, revealing significant performance drops under corruptions. To improve robustness, we propose a Semantic-Aware Masking-based Progressive Learning (SAMPL) strategy to guide the model to be optimized based on holistic and partial cues, dynamically adjusting the model's optimization to enhance robust feature learning. Extensive experiments show our approach outperforms state-of-the-art methods, setting a new standard for robust HOI detection. Benchmarks, datasets, and code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于语义感知掩码的渐进学习方法在鲁棒性人类-对象交互检测中的应用 

---
# Assuring the Safety of Reinforcement Learning Components: AMLAS-RL 

**Title (ZH)**: 确保强化学习组件的安全性：AMLAS-RL 

**Authors**: Calum Corrie Imrie, Ioannis Stefanakos, Sepeedeh Shahbeigi, Richard Hawkins, Simon Burton  

**Link**: [PDF](https://arxiv.org/pdf/2507.08848)  

**Abstract**: The rapid advancement of machine learning (ML) has led to its increasing integration into cyber-physical systems (CPS) across diverse domains. While CPS offer powerful capabilities, incorporating ML components introduces significant safety and assurance challenges. Among ML techniques, reinforcement learning (RL) is particularly suited for CPS due to its capacity to handle complex, dynamic environments where explicit models of interaction between system and environment are unavailable or difficult to construct. However, in safety-critical applications, this learning process must not only be effective but demonstrably safe. Safe-RL methods aim to address this by incorporating safety constraints during learning, yet they fall short in providing systematic assurance across the RL lifecycle. The AMLAS methodology offers structured guidance for assuring the safety of supervised learning components, but it does not directly apply to the unique challenges posed by RL. In this paper, we adapt AMLAS to provide a framework for generating assurance arguments for an RL-enabled system through an iterative process; AMLAS-RL. We demonstrate AMLAS-RL using a running example of a wheeled vehicle tasked with reaching a target goal without collision. 

**Abstract (ZH)**: 机器学习在 cyber-物理系统中的快速进步及其安全保证挑战：从强化学习视角出发——AMLAS的方法论 

---
# View Invariant Learning for Vision-Language Navigation in Continuous Environments 

**Title (ZH)**: 连续环境中基于视知觉语言导航的视差不变学习 

**Authors**: Josh Qixuan Sun, Xiaoying Xing, Huaiyuan Weng, Chul Min Yeum, Mark Crowley  

**Link**: [PDF](https://arxiv.org/pdf/2507.08831)  

**Abstract**: Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method. 

**Abstract (ZH)**: 基于变异视角的视觉-语言导航连续环境（V2-VLNCE）中的视不变学习（VIL） 

---
# Should We Ever Prefer Decision Transformer for Offline Reinforcement Learning? 

**Title (ZH)**: 我们应该 ever 偏好决策转换器进行离线强化学习吗？ 

**Authors**: Yumi Omori, Zixuan Dong, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2507.10174)  

**Abstract**: In recent years, extensive work has explored the application of the Transformer architecture to reinforcement learning problems. Among these, Decision Transformer (DT) has gained particular attention in the context of offline reinforcement learning due to its ability to frame return-conditioned policy learning as a sequence modeling task. Most recently, Bhargava et al. (2024) provided a systematic comparison of DT with more conventional MLP-based offline RL algorithms, including Behavior Cloning (BC) and Conservative Q-Learning (CQL), and claimed that DT exhibits superior performance in sparse-reward and low-quality data settings.
In this paper, through experimentation on robotic manipulation tasks (Robomimic) and locomotion benchmarks (D4RL), we show that MLP-based Filtered Behavior Cloning (FBC) achieves competitive or superior performance compared to DT in sparse-reward environments. FBC simply filters out low-performing trajectories from the dataset and then performs ordinary behavior cloning on the filtered dataset. FBC is not only very straightforward, but it also requires less training data and is computationally more efficient. The results therefore suggest that DT is not preferable for sparse-reward environments. From prior work, arguably, DT is also not preferable for dense-reward environments. Thus, we pose the question: Is DT ever preferable? 

**Abstract (ZH)**: 近年来，广泛的工作探索了Transformer架构在强化学习问题中的应用。其中，决策Transformer（DT）在离线强化学习领域尤为引人关注，因其能够将基于回报的策略学习框定为序列建模任务。最近，Bhargava等人（2024）系统性地比较了DT与传统的基于MLP的离线RL算法，包括行为克隆（BC）和保守Q学习（CQL），并声称在稀疏奖励和低质量数据环境中，DT表现出更优的性能。

在本文中，通过在机器人操纵任务（Robomimic）和运动基准测试（D4RL）上的实验，我们展示了基于MLP的过滤行为克隆（FBC）在稀疏奖励环境中可达到与DT相当甚至更优的性能。FBC简单地从数据集中过滤掉低性能的轨迹，然后在过滤后的数据集上进行普通的行为克隆。FBC不仅非常简洁，而且所需训练数据较少，计算效率更高。因此，这些结果表明，对于稀疏奖励环境，DT并不占优势。从以前的工作来看，DT对于密集奖励环境也不占优势。因此，我们提出了一个问题：决策Transformer（DT）是否有时更优？ 

---
# Model-Grounded Symbolic Artificial Intelligence Systems Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems 

**Title (ZH)**: 基于模型的符号人工智能系统学习与推理 

**Authors**: Aniruddha Chattopadhyay, Raj Dandekar, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2507.09854)  

**Abstract**: Neurosymbolic artificial intelligence (AI) systems combine neural network and classical symbolic AI mechanisms to exploit the complementary strengths of large scale, generalizable learning and robust, verifiable reasoning. Numerous classifications of neurosymbolic AI illustrate how these two components can be integrated in distinctly different ways. In this work, we propose reinterpreting instruction tuned large language models as model grounded symbolic AI systems where natural language serves as the symbolic layer and grounding is achieved through the models internal representation space. Within this framework, we investigate and develop novel learning and reasoning approaches that preserve structural similarities to traditional learning and reasoning paradigms. Preliminary evaluations across axiomatic deductive reasoning procedures of varying complexity provide insights into the effectiveness of our approach in improving learning efficiency and reasoning reliability. 

**Abstract (ZH)**: 神经符号人工智能系统将神经网络和经典符号人工智能机制相结合，利用大规模、泛化学习的强大能力以及稳健、可验证推理的优势。本论文提出将指令调整的大语言模型重新解释为模型本接地符号人工智能系统，其中自然语言作为符号层，通过模型的内部表示空间实现接地。在这一框架下，我们探讨并开发了新型的学习和推理方法，这些方法在结构上保留了传统学习和推理 paradigm 的相似性。初步评估涵盖了不同复杂度的公理演绎推理过程，提供了对我们方法在提高学习效率和推理可靠性方面的有效性见解。 

---
# humancompatible.interconnect: Testing Properties of Repeated Uses of Interconnections of AI Systems 

**Title (ZH)**: humancompatible.interconnect: 检测AI系统互联重复使用属性的研究 

**Authors**: Rodion Nazarov, Anthony Quinn, Robert Shorten, Jakub Marecek  

**Link**: [PDF](https://arxiv.org/pdf/2507.09626)  

**Abstract**: Artificial intelligence (AI) systems often interact with multiple agents. The regulation of such AI systems often requires that {\em a priori\/} guarantees of fairness and robustness be satisfied. With stochastic models of agents' responses to the outputs of AI systems, such {\em a priori\/} guarantees require non-trivial reasoning about the corresponding stochastic systems. Here, we present an open-source PyTorch-based toolkit for the use of stochastic control techniques in modelling interconnections of AI systems and properties of their repeated uses. It models robustness and fairness desiderata in a closed-loop fashion, and provides {\em a priori\/} guarantees for these interconnections. The PyTorch-based toolkit removes much of the complexity associated with the provision of fairness guarantees for closed-loop models of multi-agent systems. 

**Abstract (ZH)**: 人工智能系统 often 与多个代理交互。对于此类人工智能系统，监管通常要求事先保证公平性和鲁棒性。通过代理对人工智能系统输出的随机响应模型，这些事先保证需要对相应的随机系统进行非平凡推理。在此，我们提出一个基于 PyTorch 的开源工具包，用于使用随机控制技术建模人工智能系统的相互连接及其重复使用属性。该工具包以闭环方式建模鲁棒性和公平性需求，并为这些相互连接提供事先保证。基于 PyTorch 的工具包消除了为多代理系统的闭环模型提供公平性保证的相关复杂性。 

---
# Learning to Control Dynamical Agents via Spiking Neural Networks and Metropolis-Hastings Sampling 

**Title (ZH)**: 通过尖峰神经网络和梅特罗波利斯-哈斯特斯采样学习控制动力学代理 

**Authors**: Ali Safa, Farida Mohsen, Ali Al-Zawqari  

**Link**: [PDF](https://arxiv.org/pdf/2507.09540)  

**Abstract**: Spiking Neural Networks (SNNs) offer biologically inspired, energy-efficient alternatives to traditional Deep Neural Networks (DNNs) for real-time control systems. However, their training presents several challenges, particularly for reinforcement learning (RL) tasks, due to the non-differentiable nature of spike-based communication. In this work, we introduce what is, to our knowledge, the first framework that employs Metropolis-Hastings (MH) sampling, a Bayesian inference technique, to train SNNs for dynamical agent control in RL environments without relying on gradient-based methods. Our approach iteratively proposes and probabilistically accepts network parameter updates based on accumulated reward signals, effectively circumventing the limitations of backpropagation while enabling direct optimization on neuromorphic platforms. We evaluated this framework on two standard control benchmarks: AcroBot and CartPole. The results demonstrate that our MH-based approach outperforms conventional Deep Q-Learning (DQL) baselines and prior SNN-based RL approaches in terms of maximizing the accumulated reward while minimizing network resources and training episodes. 

**Abstract (ZH)**: 基于Metropolis-Hastings采样的Spiking神经网络在强化学习中的动态代理控制方法 

---
# Hide-and-Shill: A Reinforcement Learning Framework for Market Manipulation Detection in Symphony-a Decentralized Multi-Agent System 

**Title (ZH)**: 隐匿与诈欺：Symphony去中心化多智能体系统中的市场操纵检测强化学习框架 

**Authors**: Ronghua Shi, Yiou Liu, Xinyu Ying, Yang Tan, Yuchun Feng, Lynn Ai, Bill Shi, Xuhui Wang, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09179)  

**Abstract**: Decentralized finance (DeFi) has introduced a new era of permissionless financial innovation but also led to unprecedented market manipulation. Without centralized oversight, malicious actors coordinate shilling campaigns and pump-and-dump schemes across various platforms. We propose a Multi-Agent Reinforcement Learning (MARL) framework for decentralized manipulation detection, modeling the interaction between manipulators and detectors as a dynamic adversarial game. This framework identifies suspicious patterns using delayed token price reactions as financial this http URL method introduces three innovations: (1) Group Relative Policy Optimization (GRPO) to enhance learning stability in sparse-reward and partially observable settings; (2) a theory-based reward function inspired by rational expectations and information asymmetry, differentiating price discovery from manipulation noise; and (3) a multi-modal agent pipeline that integrates LLM-based semantic features, social graph signals, and on-chain market data for informed this http URL framework is integrated within the Symphony system, a decentralized multi-agent architecture enabling peer-to-peer agent execution and trust-aware learning through distributed logs, supporting chain-verifiable evaluation. Symphony promotes adversarial co-evolution among strategic actors and maintains robust manipulation detection without centralized oracles, enabling real-time surveillance across global DeFi this http URL on 100,000 real-world discourse episodes and validated in adversarial simulations, Hide-and-Shill achieves top performance in detection accuracy and causal attribution. This work bridges multi-agent systems with financial surveillance, advancing a new paradigm for decentralized market intelligence. All resources are available at the Hide-and-Shill GitHub repository to promote open research and reproducibility. 

**Abstract (ZH)**: 去中心化金融（DeFi）引入了新的去许可金融创新时代，但也导致了前所未有的市场操纵。没有中心化的监督，恶意行为者协调跨各种平台的拉抬打压和炒专辑活动。我们提出了一种多智能体强化学习（MARL）框架，用于去中心化的操纵检测，将操纵者和检测者之间的互动建模为动态的对抗性博弈。该框架利用延迟的代币价格反应识别可疑模式，方法引入了三项创新：（1）组相对策略优化（GRPO）以增强在稀疏奖励和部分可观测环境中的学习稳定性；（2）基于理论的奖励函数，受理性预期和信息不对称的启发，区分价格发现与操纵噪声；（3）多模态智能体流水线，结合基于LLM的语义特征、社会图信号和链上市场数据，以进行有信息量的检测。该框架嵌入在Symphony系统中，这是一种去中心化的多智能体架构，通过分布式日志支持节点间的智能体执行和信任感知学习，并通过链验证方式进行评估。Symphony促进了战略行为者之间的对抗共生，并在没有集中式预言机的情况下保持了强大的操纵检测能力，实现了全球DeFi市场的实时监控。该工作将多智能体系统与金融监督相结合，推进了一种新的去中心化市场智能新范式。所有资源均可在Hide-and-Shill GitHub仓库中获取，以促进开放研究和可再现性。 

---
# EmbRACE-3K: Embodied Reasoning and Action in Complex Environments 

**Title (ZH)**: EmbRACE-3K: 体态化复杂环境中的推理与行动 

**Authors**: Mingxian Lin, Wei Huang, Yitang Li, Chengjie Jiang, Kui Wu, Fangwei Zhong, Shengju Qian, Xin Wang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10548)  

**Abstract**: Recent advanced vision-language models(VLMs) have demonstrated strong performance on passive, offline image and video understanding tasks. However, their effectiveness in embodied settings, which require online interaction and active scene understanding remains limited. In such scenarios, an agent perceives the environment from a first-person perspective, with each action dynamically shaping subsequent observations. Even state-of-the-art models such as GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro struggle in open-environment interactions, exhibiting clear limitations in spatial reasoning and long-horizon planning. To address this gap, we introduce EmRACE-3K, a dataset of over 3,000 language-guided tasks situated in diverse, photorealistic environments constructed using Unreal Engine and the UnrealCV-Zoo framework. The tasks encompass a wide range of embodied challenges, including navigation, object manipulation, and multi-stage goal execution. Each task unfolds as a multi-step trajectory, pairing first-person visual observations with high-level instructions, grounded actions, and natural language rationales that express the agent's intent at every step. Using EmRACE-3K, we establish a benchmark to evaluate the embodied reasoning capabilities of VLMs across three key dimensions: Exploration, Dynamic Spatial-Semantic Reasoning, and Multi-stage Goal Execution. In zero-shot settings, all models achieve success rates below 20%, underscoring the challenge posed by our benchmark and the current limitations of VLMs in interactive environments. To demonstrate the utility of EmRACE-3K, we further fine-tune Qwen2.5-VL-7B using supervised learning followed by reinforcement learning. This approach yields substantial improvements across all three challenge categories, highlighting the dataset's effectiveness in enabling the development of embodied reasoning capabilities. 

**Abstract (ZH)**: 近期的先进视觉-语言模型(VLMs)在被动的、离线的图像和视频理解任务中展现了强大的性能。然而，在需要在线交互和主动场景理解的浸入式环境中，它们的有效性仍然有限。在这种场景中，智能体从第一人称视角感知环境，每次行动都会动态地塑造后续观察。即使是最先进的模型如GPT-4o、Claude 3.5 Sonnet和Gemini 2.5 Pro，在开放环境交互中也表现不佳，显示出在空间推断和长期规划方面的明显局限性。为解决这一差距，我们引入了EmRACE-3K数据集，包含超过3000个由语言引导的任务，这些任务置身于使用Unreal Engine和UnrealCV-Zoo框架构建的多样且逼真的环境中。这些任务涵盖了各种浸入式挑战，包括导航、对象操作和多阶段目标执行。每个任务分为多步轨迹，配以第一人称视觉观察、高层指令、接地动作以及每一步表达智能体意图的自然语言推理。利用EmRACE-3K，我们建立了一个基准来评估VLMs在三种关键维度上的浸入式推理能力：探索、动态空间语义推理和多阶段目标执行。在零样本设置下，所有模型的成功率均低于20%，突显了基准的挑战和当前VLMs在交互环境中面临的局限性。为了展示EmRACE-3K的实用性，我们进一步使用监督学习和强化学习的方式对Qwen2.5-VL-7B进行微调。这种方法在所有三个挑战类别中均取得显著改进，突显了该数据集在促进浸入式推理能力发展方面的有效性。 

---
# FaceLLM: A Multimodal Large Language Model for Face Understanding 

**Title (ZH)**: FaceLLM：一种用于面部理解的多模态大型语言模型 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10300)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable performance in vision-language tasks. However, existing MLLMs are primarily trained on generic datasets, limiting their ability to reason on domain-specific visual cues such as those in facial images. In particular, tasks that require detailed understanding of facial structure, expression, emotion, and demographic features remain underexplored by MLLMs due to the lack of large-scale annotated face image-text datasets. In this work, we introduce FaceLLM, a multimodal large language model trained specifically for facial image understanding. To construct the training data, we propose a novel weakly supervised pipeline that uses ChatGPT with attribute-aware prompts to generate high-quality question-answer pairs based on images from the FairFace dataset. The resulting corpus, called FairFaceGPT, covers a diverse set of attributes including expression, pose, skin texture, and forensic information. Our experiments demonstrate that FaceLLM improves the performance of MLLMs on various face-centric tasks and achieves state-of-the-art performance. This work highlights the potential of synthetic supervision via language models for building domain-specialized MLLMs, and sets a precedent for trustworthy, human-centric multimodal AI systems. FairFaceGPT dataset and pretrained FaceLLM models are publicly available in the project page. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在视觉-语言任务中展现了卓越的表现。然而，现有的MLLMs主要在通用数据集上进行训练，限制了它们在面部图像等领域特定视觉线索上的推理能力。特别是那些需要详细了解面部结构、表情、情感以及人口统计特征的任务，由于缺乏大规模注解的人脸图像-文本数据集，MLLMs仍处于未被充分探索的状态。在本工作中，我们介绍了FaceLLM，这是一种专门用于面部图像理解的多模态大语言模型。为了构建训练数据，我们提出了一种新的弱监督管道，利用带有属性感知提示的ChatGPT生成基于FairFace数据集图像的高质量问题-答案对。由此产生的语料库称为FairFaceGPT，涵盖了表情、姿态、皮肤纹理和法医信息等多种属性。我们的实验表明，FaceLLM在各种以人脸为中心的任务上提高了MLLMs的表现，并达到了最先进的性能。本工作突显了通过语言模型合成监督构建领域专用的MLLMs的潜力，并为值得信赖、以人为中心的多模态AI系统树立了先例。FairFaceGPT数据集和预训练的FaceLLM模型已在项目页面上公开。 

---
# Intersection of Reinforcement Learning and Bayesian Optimization for Intelligent Control of Industrial Processes: A Safe MPC-based DPG using Multi-Objective BO 

**Title (ZH)**: 基于多目标贝叶斯优化的强化学习与安全模型预测控制相结合的智能工业过程控制方法：一种安全的基于多目标贝叶斯优化的DPG方法 

**Authors**: Hossein Nejatbakhsh Esfahani, Javad Mohammadpour Velni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09864)  

**Abstract**: Model Predictive Control (MPC)-based Reinforcement Learning (RL) offers a structured and interpretable alternative to Deep Neural Network (DNN)-based RL methods, with lower computational complexity and greater transparency. However, standard MPC-RL approaches often suffer from slow convergence, suboptimal policy learning due to limited parameterization, and safety issues during online adaptation. To address these challenges, we propose a novel framework that integrates MPC-RL with Multi-Objective Bayesian Optimization (MOBO). The proposed MPC-RL-MOBO utilizes noisy evaluations of the RL stage cost and its gradient, estimated via a Compatible Deterministic Policy Gradient (CDPG) approach, and incorporates them into a MOBO algorithm using the Expected Hypervolume Improvement (EHVI) acquisition function. This fusion enables efficient and safe tuning of the MPC parameters to achieve improved closed-loop performance, even under model imperfections. A numerical example demonstrates the effectiveness of the proposed approach in achieving sample-efficient, stable, and high-performance learning for control systems. 

**Abstract (ZH)**: 基于MPC的RL与多目标贝叶斯优化的融合：一种具有高效和安全调参能力的方法 

---
# Universal Physics Simulation: A Foundational Diffusion Approach 

**Title (ZH)**: 通用物理仿真：一种基础扩散方法 

**Authors**: Bradley Camburn  

**Link**: [PDF](https://arxiv.org/pdf/2507.09733)  

**Abstract**: We present the first foundational AI model for universal physics simulation that learns physical laws directly from boundary-condition data without requiring a priori equation encoding. Traditional physics-informed neural networks (PINNs) and finite-difference methods necessitate explicit mathematical formulation of governing equations, fundamentally limiting their generalizability and discovery potential. Our sketch-guided diffusion transformer approach reimagines computational physics by treating simulation as a conditional generation problem, where spatial boundary conditions guide the synthesis of physically accurate steady-state solutions.
By leveraging enhanced diffusion transformer architectures with novel spatial relationship encoding, our model achieves direct boundary-to-equilibrium mapping and is generalizable to diverse physics domains. Unlike sequential time-stepping methods that accumulate errors over iterations, our approach bypasses temporal integration entirely, directly generating steady-state solutions with SSIM > 0.8 while maintaining sub-pixel boundary accuracy. Our data-informed approach enables physics discovery through learned representations analyzable via Layer-wise Relevance Propagation (LRP), revealing emergent physical relationships without predetermined mathematical constraints. This work represents a paradigm shift from AI-accelerated physics to AI-discovered physics, establishing the first truly universal physics simulation framework. 

**Abstract (ZH)**: 我们提出了第一个用于通用物理仿真的一般性AI模型，该模型能够直接从边界条件数据中学习物理定律，无需事先编码微分方程。传统的物理感知神经网络（PINNs）和有限差分方法需要显式地制定支配方程，从根本上限制了其泛化能力和发现潜力。我们的草图引导扩散变换器方法重新构想了计算物理，将仿真视为一个条件生成问题，其中空间边界条件指导生成物理准确的稳态解。

通过利用增强的扩散变换器架构和新颖的空间关系编码，我们的模型实现了直接从边界到平衡状态的映射，并能够泛化到多种物理领域。不同于累积误差的顺序时间步进方法，我们的方法完全绕过了时间积分，直接生成SSIM > 0.8的稳态解，同时保持亚像素边界精度。我们的数据驱动方法通过可解释层间相关性传播（LRP）学习表示，使物理发现成为可能，揭示了无预先数学约束的新兴物理关系。这项工作代表了从AI加速的物理到AI发现的物理的范式转变，建立了第一个真正通用的物理仿真框架。 

---
# Continual Reinforcement Learning by Planning with Online World Models 

**Title (ZH)**: 持续强化学习中的在线世界模型规划 

**Authors**: Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09177)  

**Abstract**: Continual reinforcement learning (CRL) refers to a naturalistic setting where an agent needs to endlessly evolve, by trial and error, to solve multiple tasks that are presented sequentially. One of the largest obstacles to CRL is that the agent may forget how to solve previous tasks when learning a new task, known as catastrophic forgetting. In this paper, we propose to address this challenge by planning with online world models. Specifically, we learn a Follow-The-Leader shallow model online to capture the world dynamics, in which we plan using model predictive control to solve a set of tasks specified by any reward functions. The online world model is immune to forgetting by construction with a proven regret bound of $\mathcal{O}(\sqrt{K^2D\log(T)})$ under mild assumptions. The planner searches actions solely based on the latest online model, thus forming a FTL Online Agent (OA) that updates incrementally. To assess OA, we further design Continual Bench, a dedicated environment for CRL, and compare with several strong baselines under the same model-planning algorithmic framework. The empirical results show that OA learns continuously to solve new tasks while not forgetting old skills, outperforming agents built on deep world models with various continual learning techniques. 

**Abstract (ZH)**: 持续强化学习（CRL）是指一种自然环境，其中智能体需要通过不断尝试和错误来解决依次呈现的多个任务并持续演化。CRL的一大挑战是，智能体在学习新任务时可能会忘记之前任务的解决方法，这被称为灾难性遗忘。在本文中，我们提出通过在线世界模型规划来应对这一挑战。具体而言，我们在线学习一个跟随领导者（Follow-The-Leader）的浅层模型以捕捉世界动力学，并使用模型预测控制来解决任意奖励函数指定的任务集。该在线世界模型通过在温和假设下具有证明的遗憾界$\mathcal{O}(\sqrt{K^2D\log(T)})$而设计为不会遗忘。规划器仅基于最新的在线模型搜索动作，从而形成一个增量更新的FTL在线智能体（OA）。为了评估OA，我们进一步设计了一个专用环境Continual Bench，并在相同的模型-规划算法框架下与多个强基线进行对比。实验证明，OA能够连续学习以解决新任务而不忘记旧技能，在与各种持续学习技术结合的深层世界模型构建的智能体中表现出色。 

---
# Deep Reinforcement Learning with Gradient Eligibility Traces 

**Title (ZH)**: 基于梯度有效性追溯的深度强化学习 

**Authors**: Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2507.09087)  

**Abstract**: Achieving fast and stable off-policy learning in deep reinforcement learning (RL) is challenging. Most existing methods rely on semi-gradient temporal-difference (TD) methods for their simplicity and efficiency, but are consequently susceptible to divergence. While more principled approaches like Gradient TD (GTD) methods have strong convergence guarantees, they have rarely been used in deep RL. Recent work introduced the Generalized Projected Bellman Error ($\GPBE$), enabling GTD methods to work efficiently with nonlinear function approximation. However, this work is only limited to one-step methods, which are slow at credit assignment and require a large number of samples. In this paper, we extend the $\GPBE$ objective to support multistep credit assignment based on the $\lambda$-return and derive three gradient-based methods that optimize this new objective. We provide both a forward-view formulation compatible with experience replay and a backward-view formulation compatible with streaming algorithms. Finally, we evaluate the proposed algorithms and show that they outperform both PPO and StreamQ in MuJoCo and MinAtar environments, respectively. Code available at this https URL\_algos 

**Abstract (ZH)**: 实现深度强化学习中的快速稳定离策学习颇具挑战性。大多数现有方法依赖于半梯度时差（TD）方法以保持简单性和高效性，但因此容易发散。虽然具有更原理基础的方法如梯度TD（GTD）方法能够提供强收敛保证，但它们在深度RL中鲜有应用。近期研究引入了广义投影贝尔曼误差（$\GPBE$），使GTD方法能够在非线性函数逼近下高效工作。然而，这项工作仅限于单步方法，后者在功劳分配上速度慢且需要大量样本。在本文中，我们扩展了$\GPBE$目标，支持基于$\lambda$-回报的多步功劳分配，并推导出三种基于梯度的方法来优化这一新目标。我们提供了与经验回放兼容的前向视图形式化和与流式算法兼容的后向视图形式化。最后，我们评估了所提算法，并分别证明它们在MuJoCo和MinAtar环境中优于PPO和StreamQ。代码可在此访问：this https URL\_algos。 

---
# Contrastive Language-Image Pre-Training Model based Semantic Communication Performance Optimization 

**Title (ZH)**: 基于对比语言-图像预训练模型的语义通信性能优化 

**Authors**: Shaoran Yang, Dongyu Wei, Hanzhi Yu, Zhaohui Yang, Yuchen Liu, Mingzhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08873)  

**Abstract**: In this paper, a novel contrastive language-image pre-training (CLIP) model based semantic communication framework is designed. Compared to standard neural network (e.g.,convolutional neural network) based semantic encoders and decoders that require joint training over a common dataset, our CLIP model based method does not require any training procedures thus enabling a transmitter to extract data meanings of the original data without neural network model training, and the receiver to train a neural network for follow-up task implementation without the communications with the transmitter. Next, we investigate the deployment of the CLIP model based semantic framework over a noisy wireless network. Since the semantic information generated by the CLIP model is susceptible to wireless noise and the spectrum used for semantic information transmission is limited, it is necessary to jointly optimize CLIP model architecture and spectrum resource block (RB) allocation to maximize semantic communication performance while considering wireless noise, the delay and energy used for semantic communication. To achieve this goal, we use a proximal policy optimization (PPO) based reinforcement learning (RL) algorithm to learn how wireless noise affect the semantic communication performance thus finding optimal CLIP model and RB for each user. Simulation results show that our proposed method improves the convergence rate by up to 40%, and the accumulated reward by 4x compared to soft actor-critic. 

**Abstract (ZH)**: 基于CLIP模型的语义通信框架设计及其在嘈杂无线网络中的部署 

---
# Can We Predict Your Next Move Without Breaking Your Privacy? 

**Title (ZH)**: 我们能在不侵犯您的隐私的情况下预测您的下一行动吗？ 

**Authors**: Arpita Soni, Sahil Tripathi, Gautam Siddharth Kashyap, Manaswi Kulahara, Mohammad Anas Azeez, Zohaib Hasan Siddiqui, Nipun Joshi, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08843)  

**Abstract**: We propose FLLL3M--Federated Learning with Large Language Models for Mobility Modeling--a privacy-preserving framework for Next-Location Prediction (NxLP). By retaining user data locally and leveraging LLMs through an efficient outer product mechanism, FLLL3M ensures high accuracy with low resource demands. It achieves SOT results on Gowalla (Acc@1: 12.55, MRR: 0.1422), WeePlace (10.71, 0.1285), Brightkite (10.42, 0.1169), and FourSquare (8.71, 0.1023), while reducing parameters by up to 45.6% and memory usage by 52.7%. 

**Abstract (ZH)**: 使用大型语言模型的联邦学习框架FLLL3M——移动性建模中的隐私保护框架——下一位置预测（NxLP） 

---
