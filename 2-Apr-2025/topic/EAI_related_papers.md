# Visual Environment-Interactive Planning for Embodied Complex-Question Answering 

**Title (ZH)**: 基于视觉环境交互规划的实体化复杂问题回答 

**Authors**: Ning Lan, Baoshan Ou, Xuemei Xie, Guangming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00775)  

**Abstract**: This study focuses on Embodied Complex-Question Answering task, which means the embodied robot need to understand human questions with intricate structures and abstract semantics. The core of this task lies in making appropriate plans based on the perception of the visual environment. Existing methods often generate plans in a once-for-all manner, i.e., one-step planning. Such approach rely on large models, without sufficient understanding of the environment. Considering multi-step planning, the framework for formulating plans in a sequential manner is proposed in this paper. To ensure the ability of our framework to tackle complex questions, we create a structured semantic space, where hierarchical visual perception and chain expression of the question essence can achieve iterative interaction. This space makes sequential task planning possible. Within the framework, we first parse human natural language based on a visual hierarchical scene graph, which can clarify the intention of the question. Then, we incorporate external rules to make a plan for current step, weakening the reliance on large models. Every plan is generated based on feedback from visual perception, with multiple rounds of interaction until an answer is obtained. This approach enables continuous feedback and adjustment, allowing the robot to optimize its action strategy. To test our framework, we contribute a new dataset with more complex questions. Experimental results demonstrate that our approach performs excellently and stably on complex tasks. And also, the feasibility of our approach in real-world scenarios has been established, indicating its practical applicability. 

**Abstract (ZH)**: 本研究聚焦于具身复杂问题回答任务，即具身机器人需要理解具有复杂结构和抽象语义的人类问题。该任务的核心在于基于对视觉环境的感知制定合适的计划。现有方法通常采用一次性规划方式，即一步规划。此类方法依赖于大型模型，而未能充分理解环境。考虑多步规划，本文提出了顺序方式制定计划的框架。为了确保框架解决复杂问题的能力，我们构建了一个结构化的语义空间，在该空间中，层次视觉感知和问题核心的链式表达可以实现迭代交互，使顺序任务规划成为可能。在框架内，我们首先基于视觉层次场景图解析人类自然语言，以明确问题意图。然后，结合外部规则制定当前步骤的计划，减少对大型模型的依赖。每个计划都基于视觉感知的反馈生成，并通过多轮交互直至获得答案。这种方法允许持续反馈和调整，使得机器人能够优化其行动策略。为测试框架，我们贡献了一个包含更多复杂问题的新数据集。实验结果表明，该方法在复杂任务中表现出色且稳定，并且在真实场景中的可行性已得到验证，显示了其实用性。 

---
# Immersive Explainability: Visualizing Robot Navigation Decisions through XAI Semantic Scene Projections in Virtual Reality 

**Title (ZH)**: 沉浸式可解释性：通过虚拟现实中的XAI语义场景投影可视化机器人导航决策 

**Authors**: Jorge de Heuvel, Sebastian Müller, Marlene Wessels, Aftab Akhtar, Christian Bauckhage, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00682)  

**Abstract**: End-to-end robot policies achieve high performance through neural networks trained via reinforcement learning (RL). Yet, their black box nature and abstract reasoning pose challenges for human-robot interaction (HRI), because humans may experience difficulty in understanding and predicting the robot's navigation decisions, hindering trust development. We present a virtual reality (VR) interface that visualizes explainable AI (XAI) outputs and the robot's lidar perception to support intuitive interpretation of RL-based navigation behavior. By visually highlighting objects based on their attribution scores, the interface grounds abstract policy explanations in the scene context. This XAI visualization bridges the gap between obscure numerical XAI attribution scores and a human-centric semantic level of explanation. A within-subjects study with 24 participants evaluated the effectiveness of our interface for four visualization conditions combining XAI and lidar. Participants ranked scene objects across navigation scenarios based on their importance to the robot, followed by a questionnaire assessing subjective understanding and predictability. Results show that semantic projection of attributions significantly enhances non-expert users' objective understanding and subjective awareness of robot behavior. In addition, lidar visualization further improves perceived predictability, underscoring the value of integrating XAI and sensor for transparent, trustworthy HRI. 

**Abstract (ZH)**: 端到端机器人政策通过强化学习训练的神经网络实现高性能，但其黑箱性质和抽象推理给机器人与人类交互（HRI）带来挑战，因为人类可能难以理解并预测机器人的导航决策，阻碍了信任的建立。我们提出一个虚拟现实（VR）界面，可视化可解释人工智能（XAI）输出和机器人的激光雷达感知，以支持对基于强化学习（RL）导航行为的直观解释。通过根据注意力分数可视化突出显示对象，该界面将抽象的策略解释与场景上下文联系起来。这种XAI可视化填补了模糊的数值XAI注意力分数与以人类为中心的意义层面解释之间的差距。一项涉及24名参与者的被试内研究评估了在结合XAI和激光雷达的四种可视化条件下，该界面的有效性。参与者在导航场景中根据其对机器人的重要性对场景对象进行排序，随后填写问卷评估主观理解和可预测性。结果表明，意义投射的注意力显著增强了非专家用户对机器人行为的客观理解和主观意识。此外，激光雷达可视化进一步提高了感知的可预测性，强调了结合XAI和传感器以实现透明、可信赖的HRI的价值。 

---
# Learning Bipedal Locomotion on Gear-Driven Humanoid Robot Using Foot-Mounted IMUs 

**Title (ZH)**: 基于脚部安装IMU的学习齿轮驱动人形机器人 bipedal 行走方法 

**Authors**: Sotaro Katayama, Yuta Koda, Norio Nagatsuka, Masaya Kinoshita  

**Link**: [PDF](https://arxiv.org/pdf/2504.00614)  

**Abstract**: Sim-to-real reinforcement learning (RL) for humanoid robots with high-gear ratio actuators remains challenging due to complex actuator dynamics and the absence of torque sensors. To address this, we propose a novel RL framework leveraging foot-mounted inertial measurement units (IMUs). Instead of pursuing detailed actuator modeling and system identification, we utilize foot-mounted IMU measurements to enhance rapid stabilization capabilities over challenging terrains. Additionally, we propose symmetric data augmentation dedicated to the proposed observation space and random network distillation to enhance bipedal locomotion learning over rough terrain. We validate our approach through hardware experiments on a miniature-sized humanoid EVAL-03 over a variety of environments. The experimental results demonstrate that our method improves rapid stabilization capabilities over non-rigid surfaces and sudden environmental transitions. 

**Abstract (ZH)**: 基于足部加速度计的高减速比腿式机器人Sim-to-real强化学习研究 

---
# Learning-Based Approximate Nonlinear Model Predictive Control Motion Cueing 

**Title (ZH)**: 基于学习的近似非线性模型预测控制运动模拟 

**Authors**: Camilo Gonzalez Arango, Houshyar Asadi, Mohammad Reza Chalak Qazani, Chee Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00469)  

**Abstract**: Motion Cueing Algorithms (MCAs) encode the movement of simulated vehicles into movement that can be reproduced with a motion simulator to provide a realistic driving experience within the capabilities of the machine. This paper introduces a novel learning-based MCA for serial robot-based motion simulators. Building on the differentiable predictive control framework, the proposed method merges the advantages of Nonlinear Model Predictive Control (NMPC) - notably nonlinear constraint handling and accurate kinematic modeling - with the computational efficiency of machine learning. By shifting the computational burden to offline training, the new algorithm enables real-time operation at high control rates, thus overcoming the key challenge associated with NMPC-based motion cueing. The proposed MCA incorporates a nonlinear joint-space plant model and a policy network trained to mimic NMPC behavior while accounting for joint acceleration, velocity, and position limits. Simulation experiments across multiple motion cueing scenarios showed that the proposed algorithm performed on par with a state-of-the-art NMPC-based alternative in terms of motion cueing quality as quantified by the RMSE and correlation coefficient with respect to reference signals. However, the proposed algorithm was on average 400 times faster than the NMPC baseline. In addition, the algorithm successfully generalized to unseen operating conditions, including motion cueing scenarios on a different vehicle and real-time physics-based simulations. 

**Abstract (ZH)**: 基于学习的串联机器人运动模拟器运动引导算法 

---
# Egocentric Conformal Prediction for Safe and Efficient Navigation in Dynamic Cluttered Environments 

**Title (ZH)**: 基于自我中心可信预测的动态杂乱环境中安全高效导航 

**Authors**: Jaeuk Shin, Jungjin Lee, Insoon Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00447)  

**Abstract**: Conformal prediction (CP) has emerged as a powerful tool in robotics and control, thanks to its ability to calibrate complex, data-driven models with formal guarantees. However, in robot navigation tasks, existing CP-based methods often decouple prediction from control, evaluating models without considering whether prediction errors actually compromise safety. Consequently, ego-vehicles may become overly conservative or even immobilized when all potential trajectories appear infeasible. To address this issue, we propose a novel CP-based navigation framework that responds exclusively to safety-critical prediction errors. Our approach introduces egocentric score functions that quantify how much closer obstacles are to a candidate vehicle position than anticipated. These score functions are then integrated into a model predictive control scheme, wherein each candidate state is individually evaluated for safety. Combined with an adaptive CP mechanism, our framework dynamically adjusts to changes in obstacle motion without resorting to unnecessary conservatism. Theoretical analyses indicate that our method outperforms existing CP-based approaches in terms of cost-efficiency while maintaining the desired safety levels, as further validated through experiments on real-world datasets featuring densely populated pedestrian environments. 

**Abstract (ZH)**: 面向安全关键预测误差的配准预测导航框架 

---
# Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation 

**Title (ZH)**: think 小，做 大： lifelong 机器人操作 的原始提示学习 

**Authors**: Yuanqi Yao, Siao Liu, Haoming Song, Delin Qu, Qizhi Chen, Yan Ding, Bin Zhao, Zhigang Wang, Xuelong Li, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00420)  

**Abstract**: Building a lifelong robot that can effectively leverage prior knowledge for continuous skill acquisition remains significantly challenging. Despite the success of experience replay and parameter-efficient methods in alleviating catastrophic forgetting problem, naively applying these methods causes a failure to leverage the shared primitives between skills. To tackle these issues, we propose Primitive Prompt Learning (PPL), to achieve lifelong robot manipulation via reusable and extensible primitives. Within our two stage learning scheme, we first learn a set of primitive prompts to represent shared primitives through multi-skills pre-training stage, where motion-aware prompts are learned to capture semantic and motion shared primitives across different skills. Secondly, when acquiring new skills in lifelong span, new prompts are appended and optimized with frozen pretrained prompts, boosting the learning via knowledge transfer from old skills to new ones. For evaluation, we construct a large-scale skill dataset and conduct extensive experiments in both simulation and real-world tasks, demonstrating PPL's superior performance over state-of-the-art methods. 

**Abstract (ZH)**: 构建能够有效利用先验知识进行连续技能获取的终身机器人still remains significantly challenging. 

---
# Safe Navigation in Dynamic Environments Using Data-Driven Koopman Operators and Conformal Prediction 

**Title (ZH)**: 使用数据驱动的Koopman算子和齐性预测在动态环境中的安全导航 

**Authors**: Kaier Liang, Guang Yang, Mingyu Cai, Cristian-Ioan Vasile  

**Link**: [PDF](https://arxiv.org/pdf/2504.00352)  

**Abstract**: We propose a novel framework for safe navigation in dynamic environments by integrating Koopman operator theory with conformal prediction. Our approach leverages data-driven Koopman approximation to learn nonlinear dynamics and employs conformal prediction to quantify uncertainty, providing statistical guarantees on approximation errors. This uncertainty is effectively incorporated into a Model Predictive Controller (MPC) formulation through constraint tightening, ensuring robust safety guarantees. We implement a layered control architecture with a reference generator providing waypoints for safe navigation. The effectiveness of our methods is validated in simulation. 

**Abstract (ZH)**: 我们提出了一种结合科赫曼算子理论与双曲预测的新颖框架，以实现动态环境下的安全导航。该方法利用数据驱动的科赫曼近似来学习非线性动力学，并采用双曲预测来量化不确定性，从而提供关于近似误差的统计保证。通过约束收紧将这种不确定性有效纳入模型预测控制器（MPC）的构架中，确保了 robust 的安全保证。我们实现了一种分层控制架构，其中参考生成器提供安全导航的航点。我们的方法在仿真实验中得到了验证。 

---
# Dynamics-aware Diffusion Models for Planning and Control 

**Title (ZH)**: 具备动力学意识的扩散模型在规划与控制中的应用 

**Authors**: Darshan Gadginmath, Fabio Pasqualetti  

**Link**: [PDF](https://arxiv.org/pdf/2504.00236)  

**Abstract**: This paper addresses the problem of generating dynamically admissible trajectories for control tasks using diffusion models, particularly in scenarios where the environment is complex and system dynamics are crucial for practical application. We propose a novel framework that integrates system dynamics directly into the diffusion model's denoising process through a sequential prediction and projection mechanism. This mechanism, aligned with the diffusion model's noising schedule, ensures generated trajectories are both consistent with expert demonstrations and adhere to underlying physical constraints. Notably, our approach can generate maximum likelihood trajectories and accurately recover trajectories generated by linear feedback controllers, even when explicit dynamics knowledge is unavailable. We validate the effectiveness of our method through experiments on standard control tasks and a complex non-convex optimal control problem involving waypoint tracking and collision avoidance, demonstrating its potential for efficient trajectory generation in practical applications. 

**Abstract (ZH)**: 本文探讨了使用扩散模型生成控制任务中动态容许轨迹的问题，特别是在环境复杂且系统动力学对于实际应用至关重要的场景中。我们提出了一种新的框架，通过顺序预测和投影机制将系统动力学直接整合到扩散模型的去噪过程中。该机制与扩散模型的加噪时间表相一致，确保生成的轨迹既符合专家演示，又遵守基本的物理约束。值得注意的是，即使在缺乏显式动力学知识的情况下，我们的方法也能生成最大似然轨迹，并且能够准确恢复由线性反馈控制器生成的轨迹。我们通过在标准控制任务和一个涉及航点跟踪和避障的复杂非凸最优控制问题上的实验验证了该方法的有效性，展示了其在实际应用场景中高效轨迹生成的潜力。 

---
# Enhancing Physical Human-Robot Interaction: Recognizing Digits via Intrinsic Robot Tactile Sensing 

**Title (ZH)**: 增强物理人机交互：通过内在机器人触觉感知识别数字 

**Authors**: Teresa Sinico, Giovanni Boschetti, Pedro Neto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00167)  

**Abstract**: Physical human-robot interaction (pHRI) remains a key challenge for achieving intuitive and safe interaction with robots. Current advancements often rely on external tactile sensors as interface, which increase the complexity of robotic systems. In this study, we leverage the intrinsic tactile sensing capabilities of collaborative robots to recognize digits drawn by humans on an uninstrumented touchpad mounted to the robot's flange. We propose a dataset of robot joint torque signals along with corresponding end-effector (EEF) forces and moments, captured from the robot's integrated torque sensors in each joint, as users draw handwritten digits (0-9) on the touchpad. The pHRI-DIGI-TACT dataset was collected from different users to capture natural variations in handwriting. To enhance classification robustness, we developed a data augmentation technique to account for reversed and rotated digits inputs. A Bidirectional Long Short-Term Memory (Bi-LSTM) network, leveraging the spatiotemporal nature of the data, performs online digit classification with an overall accuracy of 94\% across various test scenarios, including those involving users who did not participate in training the system. This methodology is implemented on a real robot in a fruit delivery task, demonstrating its potential to assist individuals in everyday life. Dataset and video demonstrations are available at: this https URL. 

**Abstract (ZH)**: 基于机器人的物理人类-机器人交互 (pHRI)：利用协作机器人内嵌的触觉感知能力识别人类在未装备传感器的触控板上手写数字 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Title (ZH)**: 将多模态LLMgrounding到寻求帮助的具身代理上，并应用于强化学习 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

**Abstract (ZH)**: 具身代理在现实世界环境中操作时必须解释模糊和不明确的人类指令。一个 capable 的家用机器人应该识别出指令的模糊性，并提出相关的问题以准确推断用户意图，从而提高任务执行的有效性。为研究这一问题，我们引入了“Act and Ask”任务，即在一个家庭环境中，具身代理必须根据模糊的指令获取特定物体实例。代理必须在部分可观测性下，战略性地提出最少但相关的问题以解决模糊性。为解决这一问题，我们提出了一种新的方法，利用大型语言模型（LLM）的在线强化学习（RL）微调，将其作为视觉-语言-行动（VLA）策略使用，同时使用LLM生成的奖励。我们的方法消除了大规模人工演示或手动工程奖励的需要。我们在我们的任务上与强零-shot 基线（包括GPT-4o）和监督微调的LLM进行评估。结果表明，我们的RL微调LLM在所有基线中表现出显著的领先优势（19.1%-40.3%），并能很好地泛化到新的场景和任务中。据我们所知，这是首次将LLM适应为能够使用LLM生成奖励进行在线RL的VLA代理并能主动寻求帮助的示范。 

---
# Personality-Driven Decision-Making in LLM-Based Autonomous Agents 

**Title (ZH)**: 基于LLM的自主代理的人格驱动决策制定 

**Authors**: Lewis Newsham, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2504.00727)  

**Abstract**: The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies. 

**Abstract (ZH)**: 大型语言模型嵌入自主代理中的研究：诱导个性特征对基于LLM代理任务选择过程的影响及其在主动网络防御策略中的可行性 

---
# Exploration and Adaptation in Non-Stationary Tasks with Diffusion Policies 

**Title (ZH)**: 非稳态任务中扩散策略的探索与适应 

**Authors**: Gunbir Singh Baveja  

**Link**: [PDF](https://arxiv.org/pdf/2504.00280)  

**Abstract**: This paper investigates the application of Diffusion Policy in non-stationary, vision-based RL settings, specifically targeting environments where task dynamics and objectives evolve over time. Our work is grounded in practical challenges encountered in dynamic real-world scenarios such as robotics assembly lines and autonomous navigation, where agents must adapt control strategies from high-dimensional visual inputs. We apply Diffusion Policy -- which leverages iterative stochastic denoising to refine latent action representations-to benchmark environments including Procgen and PointMaze. Our experiments demonstrate that, despite increased computational demands, Diffusion Policy consistently outperforms standard RL methods such as PPO and DQN, achieving higher mean and maximum rewards with reduced variability. These findings underscore the approach's capability to generate coherent, contextually relevant action sequences in continuously shifting conditions, while also highlighting areas for further improvement in handling extreme non-stationarity. 

**Abstract (ZH)**: 本文研究了扩散策略在非站稳态、基于视觉的强化学习环境中的应用，特别是针对任务动力学和目标随时间演变的环境。我们的工作基于在动态现实场景中如机器人装配线和自主导航中遇到的实用挑战，其中代理必须从高维视觉输入中适应控制策略。我们应用扩散策略——利用迭代随机降噪来细化潜在动作表示——在Procgen和PointMaze等基准环境中进行了测试。实验结果表明，尽管增加了计算需求，扩散策略在平均奖励和最大奖励方面始终优于标准的RL方法如PPO和DQN，并且具有更低的奖励变异性。这些发现突显了该方法在连续变化条件下生成连贯且上下文相关动作序列的能力，同时也指出了在处理极端非站稳态方面需要改进的领域。 

---
# WorldScore: A Unified Evaluation Benchmark for World Generation 

**Title (ZH)**: 世界生成统一评估基准：WorldScore 

**Authors**: Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00983)  

**Abstract**: We introduce the WorldScore benchmark, the first unified benchmark for world generation. We decompose world generation into a sequence of next-scene generation tasks with explicit camera trajectory-based layout specifications, enabling unified evaluation of diverse approaches from 3D and 4D scene generation to video generation models. The WorldScore benchmark encompasses a curated dataset of 3,000 test examples that span diverse worlds: static and dynamic, indoor and outdoor, photorealistic and stylized. The WorldScore metrics evaluate generated worlds through three key aspects: controllability, quality, and dynamics. Through extensive evaluation of 19 representative models, including both open-source and closed-source ones, we reveal key insights and challenges for each category of models. Our dataset, evaluation code, and leaderboard can be found at this https URL 

**Abstract (ZH)**: 世界评分基准：首个统一的世界生成基准 

---
# IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval 

**Title (ZH)**: IDMR：迈向实例驱动的精确多模态检索视图对应 

**Authors**: Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00954)  

**Abstract**: Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at this https URL. 

**Abstract (ZH)**: 多模态检索系统对于 embodied AI 和 AI 驱动的数字内容产业等前沿 AI 技术变得越来越重要。然而，当前的多模态检索任务缺乏足够的复杂性，展示出有限的实际应用价值。因此，我们设计了基于实例的多模态图像检索 (IDMR)，这是一个需要模型检索包含查询图像中相同实例的同时匹配文本描述场景的新型任务。与现有专注于全局图像相似度或类别级别匹配的检索任务不同，IDMR 要求在多种情境下保持精细的实例级一致性。为评估这一能力，我们使用真实世界的对象跟踪和第一人称视频数据开发了 IDMR-bench。为应对训练数据稀缺的问题，我们提出了一种跨领域合成方法，通过从标准检测数据集中裁剪对象生成了 55.7 万训练样本。基于 Multimodal 大语言模型 (MLLM) 的检索模型，该模型在 120 万样本上训练，不仅在传统基准测试上超过了现有方法，还在我们提出的零样本 IDMR-bench 上表现出色。实验结果表明了先前模型在实例感知检索方面的局限性，并强调了 MLLM 在高级检索应用中的潜力。整个训练数据集、代码和模型，具有广泛的大小范围，可在以下网址获得。 

---
# Improved Visual-Spatial Reasoning via R1-Zero-Like Training 

**Title (ZH)**: 通过R1-Zero-like训练提高视觉-空间推理能力 

**Authors**: Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00883)  

**Abstract**: Increasing attention has been placed on improving the reasoning capacities of multi-modal large language models (MLLMs). As the cornerstone for AI agents that function in the physical realm, video-based visual-spatial intelligence (VSI) emerges as one of the most pivotal reasoning capabilities of MLLMs. This work conducts a first, in-depth study on improving the visual-spatial reasoning of MLLMs via R1-Zero-like training. Technically, we first identify that the visual-spatial reasoning capacities of small- to medium-sized Qwen2-VL models cannot be activated via Chain of Thought (CoT) prompts. We then incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset, following DeepSeek-R1-Zero. During the investigation, we identify the necessity to keep the KL penalty (even with a small value) in GRPO. With just 120 GPU hours, our vsGRPO-2B model, fine-tuned from Qwen2-VL-2B, can outperform the base model by 12.1% and surpass GPT-4o. Moreover, our vsGRPO-7B model, fine-tuned from Qwen2-VL-7B, achieves performance comparable to that of the best open-source model LLaVA-NeXT-Video-72B. Additionally, we compare vsGRPO to supervised fine-tuning and direct preference optimization baselines and observe strong performance superiority. The code and dataset will be available soon. 

**Abstract (ZH)**: 提升多模态大型语言模型的视觉-空间推理能力：基于R1-Zero-like训练的方法 

---
# Science Autonomy using Machine Learning for Astrobiology 

**Title (ZH)**: 基于机器学习的天体生物学自主科学发现 

**Authors**: Victoria Da Poian, Bethany Theiling, Eric Lyness, David Burtt, Abigail R. Azari, Joey Pasterski, Luoth Chou, Melissa Trainer, Ryan Danell, Desmond Kaplan, Xiang Li, Lily Clough, Brett McKinney, Lukas Mandrake, Bill Diamond, Caroline Freissinet  

**Link**: [PDF](https://arxiv.org/pdf/2504.00709)  

**Abstract**: In recent decades, artificial intelligence (AI) including machine learning (ML) have become vital for space missions enabling rapid data processing, advanced pattern recognition, and enhanced insight extraction. These tools are especially valuable in astrobiology applications, where models must distinguish biotic patterns from complex abiotic backgrounds. Advancing the integration of autonomy through AI and ML into space missions is a complex challenge, and we believe that by focusing on key areas, we can make significant progress and offer practical recommendations for tackling these obstacles. 

**Abstract (ZH)**: 近年来，包括机器学习在内的人工智能在航天任务中变得至关重要，能够实现快速数据处理、高级模式识别和增强的信息提取。这些工具在 astrobiology 应用中尤为重要，因为模型必须在复杂的非生物背景下区分生物模式。通过人工智能和机器学习增强航天任务中的自主集成是一项复杂挑战，我们相信通过聚焦关键领域，可以实现显著进展并提出应对这些挑战的实际建议。 

---
# Suite-IN++: A FlexiWear BodyNet Integrating Global and Local Motion Features from Apple Suite for Robust Inertial Navigation 

**Title (ZH)**: Suite-IN++：结合Apple Suite全局和局部 motion特征的柔性穿戴BodyNet稳健惯性导航 

**Authors**: Lan Sun, Songpengcheng Xia, Jiarui Yang, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00438)  

**Abstract**: The proliferation of wearable technology has established multi-device ecosystems comprising smartphones, smartwatches, and headphones as critical enablers for ubiquitous pedestrian localization. However, traditional pedestrian dead reckoning (PDR) struggles with diverse motion modes, while data-driven methods, despite improving accuracy, often lack robustness due to their reliance on a single-device setup. Therefore, a promising solution is to fully leverage existing wearable devices to form a flexiwear bodynet for robust and accurate pedestrian localization. This paper presents Suite-IN++, a deep learning framework for flexiwear bodynet-based pedestrian localization. Suite-IN++ integrates motion data from wearable devices on different body parts, using contrastive learning to separate global and local motion features. It fuses global features based on the data reliability of each device to capture overall motion trends and employs an attention mechanism to uncover cross-device correlations in local features, extracting motion details helpful for accurate localization. To evaluate our method, we construct a real-life flexiwear bodynet dataset, incorporating Apple Suite (iPhone, Apple Watch, and AirPods) across diverse walking modes and device configurations. Experimental results demonstrate that Suite-IN++ achieves superior localization accuracy and robustness, significantly outperforming state-of-the-art models in real-life pedestrian tracking scenarios. 

**Abstract (ZH)**: 穿戴设备技术的普及已经建立了以智能手机、智能手表和耳机为核心的多设备生态系统，这些设备是实现泛在行人定位的关键使能器。然而，传统的行人航位推算（PDR）难以应对多种运动模式，而基于数据驱动的方法尽管提高了精度，但由于依赖单一设备设置，往往会缺乏鲁棒性。因此，一个有前景的解决方案是充分利用现有的穿戴设备，构建一个灵活穿戴体络（flexiwear bodynet），以实现稳健且精确的行人定位。本文提出了Suite-IN++，这是一种基于灵活穿戴体络的行人定位深度学习框架。Suite-IN++整合了不同身体部位穿戴设备的运动数据，利用对比学习分离全局和局部运动特征。它基于每个设备数据的可靠性融合全局特征，以捕捉整体运动趋势，并采用注意力机制揭示局部特征中的跨设备关联，提取有助于精确定位的运动细节。为了评估我们的方法，我们构建了一个实际生活中的灵活穿戴体络数据集，该数据集包含了在多种行走模式和设备配置下的Apple Suite（iPhone、Apple Watch和AirPods）。实验结果表明，Suite-IN++在实现卓越的定位精度和鲁棒性方面显著优于现有最先进的模型，在实际生活中的行人跟踪场景中表现优异。 

---
