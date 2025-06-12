# Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation 

**Title (ZH)**: 基于操作链的轨迹自回归建模：应用于机器人操作 задачnement
user
基于操作链的轨迹自回归建模：应用于机器人操作 

**Authors**: Wenbo Zhang, Tianrun Hu, Yanyuan Qiao, Hanbo Zhang, Yuchu Qin, Yang Li, Jiajun Liu, Tao Kong, Lingqiao Liu, Xiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.09990)  

**Abstract**: We present Chain-of-Action (CoA), a novel visuo-motor policy paradigm built upon Trajectory Autoregressive Modeling. Unlike conventional approaches that predict next step action(s) forward, CoA generates an entire trajectory by explicit backward reasoning with task-specific goals through an action-level Chain-of-Thought (CoT) process. This process is unified within a single autoregressive structure: (1) the first token corresponds to a stable keyframe action that encodes the task-specific goals; and (2) subsequent action tokens are generated autoregressively, conditioned on the initial keyframe and previously predicted actions. This backward action reasoning enforces a global-to-local structure, allowing each local action to be tightly constrained by the final goal. To further realize the action reasoning structure, CoA incorporates four complementary designs: continuous action token representation; dynamic stopping for variable-length trajectory generation; reverse temporal ensemble; and multi-token prediction to balance action chunk modeling with global structure. As a result, CoA gives strong spatial generalization capabilities while preserving the flexibility and simplicity of a visuo-motor policy. Empirically, we observe CoA achieves the state-of-the-art performance across 60 RLBench tasks and 8 real-world manipulation tasks. 

**Abstract (ZH)**: 我们提出Chain-of-Action (CoA)，这是一种基于轨迹自回归建模的新颖视觉-运动政策范式。与传统的预测下一步动作的方法不同，CoA 通过任务特定的目标进行显式的反向推理，生成整个轨迹，过程中的每一步动作都在一个统一的自回归结构中生成：（1）第一个标记对应一个稳定的基帧动作，编码任务特定的目标；（2）后续的动作标记基于初始基帧和之前预测的动作，自回归生成。这种反向动作推理建立了全局到局部的结构，使每个局部动作都能紧密地受限于最终目标。为了进一步实现动作推理结构，CoA 结合了四种补充设计：连续动作标记表示、动态停止生成可变长度的轨迹、逆时间集成以及多标记预测以平衡动作片段建模与全局结构的平衡。因此，CoA 在保持视觉-运动政策的灵活性和简单性的同时，提供了强大的空间泛化能力。实验证明，CoA 在 60 个 RLBench 任务和 8 个现实世界的操作任务中达到了最先进的性能。 

---
# SAFE: Multitask Failure Detection for Vision-Language-Action Models 

**Title (ZH)**: SAFE：面向视觉-语言-行动模型的多任务故障检测 

**Authors**: Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.09937)  

**Abstract**: While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out-of-the-box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results can be found at this https URL. 

**Abstract (ZH)**: 多任务故障检测：面向通用机器人策略的安全检测器 

---
# Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints 

**Title (ZH)**: 层次学习增强的 MPC 在异构约束下的安全人群导航 

**Authors**: Huajian Liu, Yixuan Feng, Wei Dong, Kunpeng Fan, Chao Wang, Yongzhuo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09859)  

**Abstract**: In this paper, we propose a novel hierarchical framework for robot navigation in dynamic environments with heterogeneous constraints. Our approach leverages a graph neural network trained via reinforcement learning (RL) to efficiently estimate the robot's cost-to-go, formulated as local goal recommendations. A spatio-temporal path-searching module, which accounts for kinematic constraints, is then employed to generate a reference trajectory to facilitate solving the non-convex optimization problem used for explicit constraint enforcement. More importantly, we introduce an incremental action-masking mechanism and a privileged learning strategy, enabling end-to-end training of the proposed planner. Both simulation and real-world experiments demonstrate that the proposed method effectively addresses local planning in complex dynamic environments, achieving state-of-the-art (SOTA) performance. Compared with existing learning-optimization hybrid methods, our approach eliminates the dependency on high-fidelity simulation environments, offering significant advantages in computational efficiency and training scalability. The code will be released as open-source upon acceptance of the paper. 

**Abstract (ZH)**: 本文提出了一种新颖的层次框架，用于动态环境中异构约束下的机器人导航。该方法利用通过强化学习（RL）训练的图神经网络高效估计机器人的成本到底，并将其形式化为局部目标推荐。随后采用一个考虑运动学约束的空间-时间路径搜索模块生成参考轨迹，以辅助解决用于显式约束执行的非凸优化问题。更重要的是，我们引入了增量动作遮罩机制和特权学习策略，使所提出的规划器能够端到端地训练。仿真和现实世界实验表明，所提出的方法有效地解决了复杂动态环境下的局部规划问题，达到了目前最先进的性能。与现有的学习-优化混合方法相比，我们的方法消除了对高保真仿真环境的依赖，提供了显著的计算效率和训练可扩展性优势。论文被接受后，代码将被公开发布。 

---
# Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving 

**Title (ZH)**: 带有自我意识扩张的强化细化方法用于端到端自主驾驶 

**Authors**: Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.09800)  

**Abstract**: End-to-end autonomous driving has emerged as a promising paradigm for directly mapping sensor inputs to planning maneuvers using learning-based modular integrations. However, existing imitation learning (IL)-based models suffer from generalization to hard cases, and a lack of corrective feedback loop under post-deployment. While reinforcement learning (RL) offers a potential solution to tackle hard cases with optimality, it is often hindered by overfitting to specific driving cases, resulting in catastrophic forgetting of generalizable knowledge and sample inefficiency. To overcome these challenges, we propose Reinforced Refinement with Self-aware Expansion (R2SE), a novel learning pipeline that constantly refines hard domain while keeping generalizable driving policy for model-agnostic end-to-end driving systems. Through reinforcement fine-tuning and policy expansion that facilitates continuous improvement, R2SE features three key components: 1) Generalist Pretraining with hard-case allocation trains a generalist imitation learning (IL) driving system while dynamically identifying failure-prone cases for targeted refinement; 2) Residual Reinforced Specialist Fine-tuning optimizes residual corrections using reinforcement learning (RL) to improve performance in hard case domain while preserving global driving knowledge; 3) Self-aware Adapter Expansion dynamically integrates specialist policies back into the generalist model, enhancing continuous performance improvement. Experimental results in closed-loop simulation and real-world datasets demonstrate improvements in generalization, safety, and long-horizon policy robustness over state-of-the-art E2E systems, highlighting the effectiveness of reinforce refinement for scalable autonomous driving. 

**Abstract (ZH)**: 端到端自主驾驶：基于强化修正与自我意识扩展的学习框架 

---
# Human-robot collaborative transport personalization via Dynamic Movement Primitives and velocity scaling 

**Title (ZH)**: 基于动态运动 primitives 和速度缩放的人机协作个性化运输 

**Authors**: Paolo Franceschi, Andrea Bussolan, Vincenzo Pomponi, Oliver Avram, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.09697)  

**Abstract**: Nowadays, industries are showing a growing interest in human-robot collaboration, particularly for shared tasks. This requires intelligent strategies to plan a robot's motions, considering both task constraints and human-specific factors such as height and movement preferences. This work introduces a novel approach to generate personalized trajectories using Dynamic Movement Primitives (DMPs), enhanced with real-time velocity scaling based on human feedback. The method was rigorously tested in industrial-grade experiments, focusing on the collaborative transport of an engine cowl lip section. Comparative analysis between DMP-generated trajectories and a state-of-the-art motion planner (BiTRRT) highlights their adaptability combined with velocity scaling. Subjective user feedback further demonstrates a clear preference for DMP- based interactions. Objective evaluations, including physiological measurements from brain and skin activity, reinforce these findings, showcasing the advantages of DMPs in enhancing human-robot interaction and improving user experience. 

**Abstract (ZH)**: 现在，各行业对人机协作表现出 growing 的兴趣，特别是在共享任务方面。这需要智能策略来计划机器人的动作，考虑任务约束和人类特定因素，如身高和运动偏好。本工作介绍了一种使用动态运动本征（DMPs）生成个性化轨迹的新方法，并通过实时速度缩放增强，基于人类反馈。该方法在工业级实验中得到了严格的测试，重点关注发动机整流罩唇部段的协作运输。DMP生成的轨迹与当前最先进的运动规划器（BiTRRT）的比较分析突显了它们的适应性和速度缩放组合。主观用户反馈进一步表明了对基于DMP的交互的明显偏好。客观评估，包括大脑和皮肤活动的生理测量，强化了这些发现，展示了DMPs在增强人机交互和提高用户体验方面的优势。 

---
# R-CARLA: High-Fidelity Sensor Simulations with Interchangeable Dynamics for Autonomous Racing 

**Title (ZH)**: R-CARLA: 具有可互换动力学的高保真传感器模拟在自动驾驶赛车中的应用 

**Authors**: Maurice Brunner, Edoardo Ghignone, Nicolas Baumann, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2506.09629)  

**Abstract**: Autonomous racing has emerged as a crucial testbed for autonomous driving algorithms, necessitating a simulation environment for both vehicle dynamics and sensor behavior. Striking the right balance between vehicle dynamics and sensor accuracy is crucial for pushing vehicles to their performance limits. However, autonomous racing developers often face a trade-off between accurate vehicle dynamics and high-fidelity sensor simulations. This paper introduces R-CARLA, an enhancement of the CARLA simulator that supports holistic full-stack testing, from perception to control, using a single system. By seamlessly integrating accurate vehicle dynamics with sensor simulations, opponents simulation as NPCs, and a pipeline for creating digital twins from real-world robotic data, R-CARLA empowers researchers to push the boundaries of autonomous racing development. Furthermore, it is developed using CARLA's rich suite of sensor simulations. Our results indicate that incorporating the proposed digital-twin framework into R-CARLA enables more realistic full-stack testing, demonstrating a significant reduction in the Sim-to-Real gap of car dynamics simulation by 42% and by 82% in the case of sensor simulation across various testing scenarios. 

**Abstract (ZH)**: 自主赛车比赛已成为自主驾驶算法的关键测试平台，需要一个既支持车辆动力学又支持传感器行为的模拟环境。准确平衡车辆动力学和传感器精度对于将车辆推向性能极限至关重要。然而，自主赛车开发者往往在准确的车辆动力学和高保真传感器模拟之间面临权衡。本文介绍了R-CARLA，它是CARLA模拟器的增强版本，支持从感知到控制的全方位堆栈测试。通过无缝集成准确的车辆动力学与传感器模拟、对手模拟作为非玩家角色（NPCs）以及从真实世界机器人数据创建数字孪生的管道，R-CARLA为研究人员提供了推动自主赛车开发边界的能力。此外，R-CARLA 是基于 CARLA 丰富的传感器模拟套件开发的。我们的结果表明，将提出的数字孪生框架整合到 R-CARLA 中能够实现更现实的全方位测试，在各种测试场景中，汽车动力学模拟的Sim-to-Real差距降低了42%，传感器模拟的Sim-to-Real差距降低了82%。 

---
# Analytic Task Scheduler: Recursive Least Squares Based Method for Continual Learning in Embodied Foundation Models 

**Title (ZH)**: 解析任务调度器：基于递归最小二乘法的持续学习方法在具身基础模型中的应用 

**Authors**: Lipei Xie, Yingxin Li, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09623)  

**Abstract**: Embodied foundation models are crucial for Artificial Intelligence (AI) interacting with the physical world by integrating multi-modal inputs, such as proprioception, vision and language, to understand human intentions and generate actions to control robots. While these models demonstrate strong generalization and few-shot learning capabilities, they face significant challenges in continually acquiring new skills without forgetting previously learned skills, a problem known as catastrophic forgetting. To address this issue, we propose the Analytic Task Scheduler (ATS), a novel framework for continual learning in embodied foundation models. ATS consists of a task-specific model library, where each model is fine-tuned independently on a single task, and an analytic scheduler trained using recursive least squares (RLS) to learn the mapping between language instructions and task-specific models. This architecture enables accurate task recognition and dynamic model selection while fundamentally avoiding parameter interference across tasks. The scheduler updates its parameters incrementally using only statistics (autocorrelation and cross-correlation matrices), enabling forgetting-resistant learning without the need to revisit historical data. We validate ATS on a real-world robot platform (RM65B), demonstrating superior resistance to forgetting and strong adaptability to task variations. The results highlight ATS as an effective, scalable, and deployable solution for continual learning in embodied foundation models operating in complex, dynamic environments. Our code will be available at this https URL 

**Abstract (ZH)**: 具身基础模型对于通过集成 proprioception、视觉和语言等多模态输入与物理世界交互的人工智能（AI）至关重要，这些模型能够理解人类意图并生成控制机器人的动作。尽管这些模型展示了强大的泛化能力和少样本学习能力，但它们在不断获取新技能时而不忘记已学习技能方面面临着重大挑战，这个问题被称为灾难性遗忘。为了应对这一问题，我们提出了一种新颖的持续学习框架——具身基础模型的分析任务调度器（Analytic Task Scheduler, ATS）。ATS 包括一个任务特定模型库，每个模型独立 fine-tune 在单一任务上，以及一个使用递归最小二乘法（Recursive Least Squares, RLS）训练的分析调度器，用于学习语言指令和任务特定模型之间的映射。该架构能够实现准确的任务识别和动态模型选择，从根本上避免了任务间参数干扰。调度器仅使用统计数据（自相关矩阵和交叉相关矩阵）逐增量更新参数，从而在无需回顾历史数据的情况下实现具有抗遗忘能力的学习。我们在实际机器人平台（RM65B）上验证了 ATS，展示了其优越的抗遗忘能力和对任务变化的良好适应性。结果表明，ATS 是一个有效的、可扩展且可部署的解决方案，适用于在复杂动态环境中操作的具身基础模型的持续学习。 

---
# Attention-Based Map Encoding for Learning Generalized Legged Locomotion 

**Title (ZH)**: 基于注意力的地图编码学习通用腿式运动控制 

**Authors**: Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz BÄcher, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.09588)  

**Abstract**: Dynamic locomotion of legged robots is a critical yet challenging topic in expanding the operational range of mobile robots. It requires precise planning when possible footholds are sparse, robustness against uncertainties and disturbances, and generalizability across diverse terrains. While traditional model-based controllers excel at planning on complex terrains, they struggle with real-world uncertainties. Learning-based controllers offer robustness to such uncertainties but often lack precision on terrains with sparse steppable areas. Hybrid methods achieve enhanced robustness on sparse terrains by combining both methods but are computationally demanding and constrained by the inherent limitations of model-based planners. To achieve generalized legged locomotion on diverse terrains while preserving the robustness of learning-based controllers, this paper proposes to learn an attention-based map encoding conditioned on robot proprioception, which is trained as part of the end-to-end controller using reinforcement learning. We show that the network learns to focus on steppable areas for future footholds when the robot dynamically navigates diverse and challenging terrains. We synthesize behaviors that exhibit robustness against uncertainties while enabling precise and agile traversal of sparse terrains. Additionally, our method offers a way to interpret the topographical perception of a neural network. We have trained two controllers for a 12-DoF quadrupedal robot and a 23-DoF humanoid robot respectively and tested the resulting controllers in the real world under various challenging indoor and outdoor scenarios, including ones unseen during training. 

**Abstract (ZH)**: 基于注意力的地图编码在不同地形上实现腿足机器人的鲁棒动态运动 

---
# Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities 

**Title (ZH)**: 将量化大语言模型集成到机器人系统中的边缘AI以利用其自然语言处理能力 

**Authors**: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, David Sobrín-Hidalgo, Ángel Manuel Guerrero-Higueras, Vicente MatellÁn-Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2506.09581)  

**Abstract**: Large Language Models (LLMs) have experienced great advancements in the last year resulting in an increase of these models in several fields to face natural language tasks. The integration of these models in robotics can also help to improve several aspects such as human-robot interaction, navigation, planning and decision-making. Therefore, this paper introduces llama\_ros, a tool designed to integrate quantized Large Language Models (LLMs) into robotic systems using ROS 2. Leveraging this http URL, a highly optimized runtime engine, llama\_ros enables the efficient execution of quantized LLMs as edge artificial intelligence (AI) in robotics systems with resource-constrained environments, addressing the challenges of computational efficiency and memory limitations. By deploying quantized LLMs, llama\_ros empowers robots to leverage the natural language understanding and generation for enhanced decision-making and interaction which can be paired with prompt engineering, knowledge graphs, ontologies or other tools to improve the capabilities of autonomous robots. Additionally, this paper provides insights into some use cases of using llama\_ros for planning and explainability in robotics. 

**Abstract (ZH)**: 大型语言模型（LLMs）在过去一年中取得了显著的进步，使得这些模型在多个领域用于应对自然语言任务。将这些模型集成到机器人技术中也可以提高多个方面的表现，如人机交互、导航、规划和决策。因此，本文介绍了llama\_ros，这是一个用于使用ROS 2将量化大型语言模型（LLMs）集成到机器人系统中的工具。通过利用这个高性能运行时引擎，llama\_ros能够高效地在资源受限的环境中将量化LLMs作为边缘人工智能（AI）执行，以解决计算效率和内存限制等挑战。通过部署量化LLMs，llama\_ros使机器人能够利用自然语言理解和生成能力，增强决策和交互能力，并可与提示工程、知识图谱、本体或其他工具结合使用以提高自主机器人的能力。此外，本文还探讨了使用llama\_ros进行机器人规划和解释的应用案例。 

---
# Time-Unified Diffusion Policy with Action Discrimination for Robotic Manipulation 

**Title (ZH)**: 时间统一扩散策略与动作鉴别方法在机器人操作中的应用 

**Authors**: Ye Niu, Sanping Zhou, Yizhe Li, Ye Den, Le Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09422)  

**Abstract**: In many complex scenarios, robotic manipulation relies on generative models to estimate the distribution of multiple successful actions. As the diffusion model has better training robustness than other generative models, it performs well in imitation learning through successful robot demonstrations. However, the diffusion-based policy methods typically require significant time to iteratively denoise robot actions, which hinders real-time responses in robotic manipulation. Moreover, existing diffusion policies model a time-varying action denoising process, whose temporal complexity increases the difficulty of model training and leads to suboptimal action accuracy. To generate robot actions efficiently and accurately, we present the Time-Unified Diffusion Policy (TUDP), which utilizes action recognition capabilities to build a time-unified denoising process. On the one hand, we build a time-unified velocity field in action space with additional action discrimination information. By unifying all timesteps of action denoising, our velocity field reduces the difficulty of policy learning and speeds up action generation. On the other hand, we propose an action-wise training method, which introduces an action discrimination branch to supply additional action discrimination information. Through action-wise training, the TUDP implicitly learns the ability to discern successful actions to better denoising accuracy. Our method achieves state-of-the-art performance on RLBench with the highest success rate of 82.6% on a multi-view setup and 83.8% on a single-view setup. In particular, when using fewer denoising iterations, TUDP achieves a more significant improvement in success rate. Additionally, TUDP can produce accurate actions for a wide range of real-world tasks. 

**Abstract (ZH)**: 基于时间统一扩散策略的机器人操作高效准确生成方法 

---
# Scoop-and-Toss: Dynamic Object Collection for Quadrupedal Systems 

**Title (ZH)**: 揽取并抛掷：四足机器人系统的动态对象收集 

**Authors**: Minji Kang, Chanwoo Baek, Yoonsang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.09406)  

**Abstract**: Quadruped robots have made significant advances in locomotion, extending their capabilities from controlled environments to real-world applications. Beyond movement, recent work has explored loco-manipulation using the legs to perform tasks such as pressing buttons or opening doors. While these efforts demonstrate the feasibility of leg-based manipulation, most have focused on relatively static tasks. In this work, we propose a framework that enables quadruped robots to collect objects without additional actuators by leveraging the agility of their legs. By attaching a simple scoop-like add-on to one leg, the robot can scoop objects and toss them into a collection tray mounted on its back. Our method employs a hierarchical policy structure comprising two expert policies-one for scooping and tossing, and one for approaching object positions-and a meta-policy that dynamically switches between them. The expert policies are trained separately, followed by meta-policy training for coordinated multi-object collection. This approach demonstrates how quadruped legs can be effectively utilized for dynamic object manipulation, expanding their role beyond locomotion. 

**Abstract (ZH)**: quadruped机器人在动态物体操作中的腿部利用：一种基于敏捷腿的多物体收集框架 

---
# Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation 

**Title (ZH)**: 分析人类到机器人操作转换中的关键目标以实现灵巧操作 

**Authors**: Chendong Xin, Mingrui Yu, Yongpeng Jiang, Zhefeng Zhang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09384)  

**Abstract**: Kinematic retargeting from human hands to robot hands is essential for transferring dexterity from humans to robots in manipulation teleoperation and imitation learning. However, due to mechanical differences between human and robot hands, completely reproducing human motions on robot hands is impossible. Existing works on retargeting incorporate various optimization objectives, focusing on different aspects of hand configuration. However, the lack of experimental comparative studies leaves the significance and effectiveness of these objectives unclear. This work aims to analyze these retargeting objectives for dexterous manipulation through extensive real-world comparative experiments. Specifically, we propose a comprehensive retargeting objective formulation that integrates intuitively crucial factors appearing in recent approaches. The significance of each factor is evaluated through experimental ablation studies on the full objective in kinematic posture retargeting and real-world teleoperated manipulation tasks. Experimental results and conclusions provide valuable insights for designing more accurate and effective retargeting algorithms for real-world dexterous manipulation. 

**Abstract (ZH)**: 从人体手部到机器人手部的动力学重塑对于传递灵巧操作能力在操控遥控和模仿学习中的应用至关重要。然而，由于人体和机器人手部的机械差异，完全在机器人手部复制人体动作是不可能的。现有的动力学重塑工作包含了各种优化目标，侧重于手部配置的不同方面。然而，缺乏实验比较研究使得这些目标的重要性及有效性不明确。本文旨在通过广泛的现实世界比较实验分析这些动力学重塑目标。具体而言，我们提出了一种综合的动力学重塑目标公式，结合了近期方法中直观重要的因素。通过对整体目标在动力学姿态重塑和现实世界遥控操作任务中的实验消融研究评估每个因素的重要性。实验结果和结论为设计更准确和有效的现实世界灵巧操作动力学重塑算法提供了宝贵的见解。 

---
# Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations 

**Title (ZH)**: 双足平衡控制的全身肌骨站立与跌倒仿真研究 

**Authors**: Chengtian Ma, Yunyue Wei, Chenhui Zuo, Chen Zhang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09383)  

**Abstract**: Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems. 

**Abstract (ZH)**: 平衡控制对于人类和双足机器人系统至关重要。虽然在运动过程中的动态平衡受到了广泛关注，但静态平衡和摔倒的量化理解仍然有限。本研究提出了一种分层控制管道，通过综合全身肌肉骨骼系统模拟人类平衡。我们识别了稳定站立期间平衡的时空动态，揭示了肌肉损伤对平衡行为的影响，并生成了与临床数据相一致的摔倒接触模式。此外，我们模拟的髋部外骨骼辅助表明，在外部扰动下平衡维持的改善和肌肉努力的减少。本研究提供了难以通过实验捕捉到的人类平衡动力学的肌肉层面见解，可为发展针对平衡障碍个体的靶向干预措施以及促进类人机器人系统的发展奠定基础。 

---
# SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending 

**Title (ZH)**: SkillBlender: 向 versatiles 人形全身动操作融合方向努力通过技能融合 

**Authors**: Yuxuan Kuang, Haoran Geng, Amine Elhafsi, Tan-Dzung Do, Pieter Abbeel, Jitendra Malik, Marco Pavone, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09366)  

**Abstract**: Humanoid robots hold significant potential in accomplishing daily tasks across diverse environments thanks to their flexibility and human-like morphology. Recent works have made significant progress in humanoid whole-body control and loco-manipulation leveraging optimal control or reinforcement learning. However, these methods require tedious task-specific tuning for each task to achieve satisfactory behaviors, limiting their versatility and scalability to diverse tasks in daily scenarios. To that end, we introduce SkillBlender, a novel hierarchical reinforcement learning framework for versatile humanoid loco-manipulation. SkillBlender first pretrains goal-conditioned task-agnostic primitive skills, and then dynamically blends these skills to accomplish complex loco-manipulation tasks with minimal task-specific reward engineering. We also introduce SkillBench, a parallel, cross-embodiment, and diverse simulated benchmark containing three embodiments, four primitive skills, and eight challenging loco-manipulation tasks, accompanied by a set of scientific evaluation metrics balancing accuracy and feasibility. Extensive simulated experiments show that our method significantly outperforms all baselines, while naturally regularizing behaviors to avoid reward hacking, resulting in more accurate and feasible movements for diverse loco-manipulation tasks in our daily scenarios. Our code and benchmark will be open-sourced to the community to facilitate future research. Project page: this https URL. 

**Abstract (ZH)**: 类人机器人通过其灵活性和类人的形态，在跨多种环境完成日常任务方面具有重要的潜力。近期的研究在利用最优控制或强化学习进行类人全身体控和移动操作方面取得了显著进展。然而，这些方法需要针对每个任务进行繁琐的任务特定调优，才能获得满意的行为表现，这限制了它们在日常场景中处理多样任务的灵活性和可扩展性。为此，我们提出了一种新的层次化强化学习框架SkillBlender，以实现灵活的类人移动操作。SkillBlender首先预训练目标条件的任务无关基本技能，然后动态融合这些技能，以最少的任务特定奖励工程实现复杂的移动操作任务。我们还引入了SkillBench，这是一个并行、跨载体和多样化的模拟基准平台，包含三个载体、四项基本技能和八项具有挑战性的移动操作任务，配有平衡准确性和可行性的科学评估指标。广泛的模拟实验表明，我们的方法在所有基线方法中表现出显著的优势，同时自然地调节行为以避免奖励劫持，从而在我们的日常场景中实现了更加准确和可行的移动操作。我们的代码和基准平台将向社区开源，旨在促进未来的研究。项目页面：this https URL。 

---
# UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation 

**Title (ZH)**: UAD：无监督能力蒸馏在机器人操纵中的泛化 

**Authors**: Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.09284)  

**Abstract**: Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: this https URL 

**Abstract (ZH)**: 理解细粒度对象 affordance 对机器人在未结构化环境中根据开放式任务指令操作对象至关重要。然而，现有的视觉 affordance 预测方法往往依赖于手动标注的数据或只针对预定义的任务集。我们提出了一种 UAD（无监督 affordance 提炼）方法，该方法可以在不需要任何手动标注的情况下，从基础模型中提炼 affordance 知识并转化为任务条件下的 affordance 模型。通过利用大型视觉模型和视觉语言模型的互补优势，UAD 自动标注了一个大规模的数据集，包含详细的 $<$指令，视觉 affordance$>$ 对。仅通过冻结特征并训练一个轻量级的任务条件解码器，UAD 在野生机器人场景和各种人类活动中展现出显著的泛化能力，尽管它是基于仿真中渲染的物体进行训练的。使用 UAD 提供的 affordance 作为观察空间，我们展示了模仿学习策略，在仅进行少量（如10个）演示后，该策略能够很好地泛化到未见过的对象实例、对象类别，甚至任务指令的变化。项目网站: 这个 https URL。 

---
# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning 

**Title (ZH)**: V-JEPA 2: 自监督视频模型实现理解、预测和规划 

**Authors**: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas  

**Link**: [PDF](https://arxiv.org/pdf/2506.09985)  

**Abstract**: A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories), to develop models capable of understanding, predicting, and planning in the physical world. We first pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2, on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. Additionally, after aligning V-JEPA 2 with a large language model, we demonstrate state-of-the-art performance on multiple video question-answering tasks at the 8 billion parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward. This work demonstrates how self-supervised learning from web-scale data and a small amount of robot interaction data can yield a world model capable of planning in the physical world. 

**Abstract (ZH)**: 现代AI面临的一个主要挑战是通过观察学习理解和行动。本文探讨了一种自监督方法，该方法结合了互联网规模的视频数据和少量交互数据（机器人轨迹），以开发能够在物理世界中理解、预测和规划的模型。首先，我们在包含超过100万小时互联网视频的视频和图像数据集上预训练了一个无动作联合嵌入预测架构V-JEPA 2。V-JEPA 2在动作理解方面表现出色（在Something-Something v2中取得77.3%的top-1准确性），在人类动作预见方面也取得了最先进的性能（在Epic-Kitchens-100中召回率达到39.7%），超越了之前的专业任务模型。此外，在将V-JEPA 2与大规模语言模型对齐后，我们展示了在80亿参数的大规模下多项视频问答任务的领先性能（例如，在PerceptionTest上取得84.0%，在TempCompass上取得76.9%）。最后，通过使用不到62小时的未标注机器人视频数据，我们展示了自监督学习在机器人规划任务中的应用，通过泛化训练出一个潜在动作条件的世界模型V-JEPA 2-AC，并部署在两个不同的实验室中的Franka手臂上，实现图像目标的物体抓取和放置。值得注意的是，这在这些环境内没有收集任何机器人数据，并且没有进行任何任务特定训练或奖励。本文展示了从网页规模数据和少量机器人交互数据进行自监督学习，能够生成能够在物理世界中规划的环境模型。 

---
# ReSim: Reliable World Simulation for Autonomous Driving 

**Title (ZH)**: ReSim: 可靠的世界模拟技术在自主驾驶中的应用 

**Authors**: Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09981)  

**Abstract**: How can we reliably simulate future driving scenarios under a wide range of ego driving behaviors? Recent driving world models, developed exclusively on real-world driving data composed mainly of safe expert trajectories, struggle to follow hazardous or non-expert behaviors, which are rare in such data. This limitation restricts their applicability to tasks such as policy evaluation. In this work, we address this challenge by enriching real-world human demonstrations with diverse non-expert data collected from a driving simulator (e.g., CARLA), and building a controllable world model trained on this heterogeneous corpus. Starting with a video generator featuring a diffusion transformer architecture, we devise several strategies to effectively integrate conditioning signals and improve prediction controllability and fidelity. The resulting model, ReSim, enables Reliable Simulation of diverse open-world driving scenarios under various actions, including hazardous non-expert ones. To close the gap between high-fidelity simulation and applications that require reward signals to judge different actions, we introduce a Video2Reward module that estimates a reward from ReSim's simulated future. Our ReSim paradigm achieves up to 44% higher visual fidelity, improves controllability for both expert and non-expert actions by over 50%, and boosts planning and policy selection performance on NAVSIM by 2% and 25%, respectively. 

**Abstract (ZH)**: 如何在广泛的行为范围内可靠地模拟未来的驾驶场景？通过丰富真实世界的人类演示数据并结合驾驶模拟器获取的多样化非专家数据，构建一个可控的世界模型，以应对这一挑战。ReSim模型的提出，能够在各种行动（包括非专家的危险行为）下可靠地模拟多种开放世界的驾驶场景。为了弥合高保真模拟与需要奖励信号判断不同行动的应用之间的差距，我们引入了Video2Reward模块，从ReSim模拟的未来中估算奖励。ReSim范式实现了最高44%的视觉保真度提升，对于专家和非专家行动的可控性分别提高了超过50%，并在NAVSIM中分别提升了2%和25%的计划和策略选择性能。 

---
# OctoNav: Towards Generalist Embodied Navigation 

**Title (ZH)**: OctoNav: 向 général 化自主导航迈进 

**Authors**: Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09839)  

**Abstract**: Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. 

**Abstract (ZH)**: 嵌入式导航是广泛追求的嵌入式AI基础支柱。然而，之前的导航研究被划分为不同的任务/能力，例如ObjNav、ImgNav和VLN，它们在任务目标和模态上有所不同，导致数据集和方法独立设计。在此工作中，我们朝着通用导航代理迈进，这些代理能够遵循自由形式的指令，这些指令结合了多种模态和多种能力的任意复合。为此，我们提出了一种大规模基准及其相应的方法，称为OctoNav-Bench和OctoNav-R1。具体来说，OctoNav-Bench具有连续环境，并通过设计的注释流水线构建。我们精心策划指令-轨迹对，指令形式自由多样，包含任意模态和能力。此外，我们在OctoNav-Bench中构建了一个Think-Before-Action（TBA-CoT）数据集，提供动作背后的思维过程。对于OctoNav-R1，我们在MLLMs基础上构建并将其适应为VLA类型模型，可以仅基于2D视觉观察生成低级动作。同时，我们设计了一种混合训练范式（HTP），包括三个阶段：Action-/TBA-SFT、Nav-GPRO和在线强化学习阶段。每个阶段包含专门设计的学习策略和奖励。重要的是，对于TBA-SFT和Nav-GPRO设计，我们受OpenAI-o1和DeepSeek-R1的启发，通过思考后再作答展示了出色的推理能力。因此，我们旨在探索如何在嵌入式导航领域实现思考后再行动，以提高模型的推理能力。具体而言，我们提出了TBA-SFT利用TBA-CoT数据集对模型进行微调作为冷启动语句，然后利用Nav-GPRO提升其推理能力。最后，OctoNav-R1在性能上优于先前的方法。 

---
# Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design 

**Title (ZH)**: 基于偏好高效强化学习：随机探索遇上实验设计 

**Authors**: Andreas Schlaginhaufen, Reda Ouhamma, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.09508)  

**Abstract**: We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries. 

**Abstract (ZH)**: 我们研究一般马尔可夫决策过程中的基于人工反馈的强化学习，其中代理通过轨迹级偏好比较进行学习。在这种设置中，一个核心挑战是设计算法以选择有信息量的偏好查询来识别潜在的奖励，并同时保证理论上的保证。我们提出了一种基于随机探索的元算法，该算法避免了乐观方法相关的计算挑战，并且具有可处理性。我们在温和的强化学习oracle假设下建立了遗憾和最后迭代的保证。为了改进查询复杂度，我们引入并分析了一种改进的算法，该算法收集轨迹对批次，并使用最优实验设计选择有信息量的比较查询。批结构还使得偏好查询的并行化成为可能，在实际部署中，反馈可以同时收集。实验评估证实，所提出的方法在需要少量偏好查询的情况下与基于奖励的强化学习方法具有竞争力。 

---
# CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation 

**Title (ZH)**: CheckManual: 基于手工操作的家电操纵新挑战与基准 

**Authors**: Yuxing Long, Jiyao Zhang, Mingjie Pan, Tianshu Wu, Taewhan Kim, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.09343)  

**Abstract**: Correct use of electrical appliances has significantly improved human life quality. Unlike simple tools that can be manipulated with common sense, different parts of electrical appliances have specific functions defined by manufacturers. If we want the robot to heat bread by microwave, we should enable them to review the microwave manual first. From the manual, it can learn about component functions, interaction methods, and representative task steps about appliances. However, previous manual-related works remain limited to question-answering tasks while existing manipulation researchers ignore the manual's important role and fail to comprehend multi-page manuals. In this paper, we propose the first manual-based appliance manipulation benchmark CheckManual. Specifically, we design a large model-assisted human-revised data generation pipeline to create manuals based on CAD appliance models. With these manuals, we establish novel manual-based manipulation challenges, metrics, and simulator environments for model performance evaluation. Furthermore, we propose the first manual-based manipulation planning model ManualPlan to set up a group of baselines for the CheckManual benchmark. 

**Abstract (ZH)**: 正确使用家用电器显著提升了人类生活质量。为了使机器人能通过微波炉加热面包，我们应让他们先查阅微波炉手册。通过手册，机器人可以学习到电器各部件的功能、交互方式以及代表性任务步骤。然而，现有的手册相关工作仅限于问答任务，而现有的操作研究人员忽视了手册的重要作用，未能理解和掌握多页手册。在本文中，我们提出了首个基于手册的家用电器操作基准CheckManual。具体而言，我们设计了一个大型模型辅助的人工修订数据生成管道，基于CAD家电模型创建手册。借助这些手册，我们建立了新的基于手册的操作挑战、评估指标和模拟环境，以评估模型性能。此外，我们提出了首个基于手册的操作规划模型ManualPlan，为CheckManual基准设立了基线。 

---
# Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism 

**Title (ZH)**: 机器人门控交互模仿学习与自适应干预机制 

**Authors**: Haoyuan Cai, Zhenghao Peng, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09176)  

**Abstract**: Interactive Imitation Learning (IIL) allows agents to acquire desired behaviors through human interventions, but current methods impose high cognitive demands on human supervisors. We propose the Adaptive Intervention Mechanism (AIM), a novel robot-gated IIL algorithm that learns an adaptive criterion for requesting human demonstrations. AIM utilizes a proxy Q-function to mimic the human intervention rule and adjusts intervention requests based on the alignment between agent and human actions. By assigning high Q-values when the agent deviates from the expert and decreasing these values as the agent becomes proficient, the proxy Q-function enables the agent to assess the real-time alignment with the expert and request assistance when needed. Our expert-in-the-loop experiments reveal that AIM significantly reduces expert monitoring efforts in both continuous and discrete control tasks. Compared to the uncertainty-based baseline Thrifty-DAgger, our method achieves a 40% improvement in terms of human take-over cost and learning efficiency. Furthermore, AIM effectively identifies safety-critical states for expert assistance, thereby collecting higher-quality expert demonstrations and reducing overall expert data and environment interactions needed. Code and demo video are available at this https URL. 

**Abstract (ZH)**: 自适应干预机制（AIM）：一种新颖的机器人门控imitation learning算法 

---
# SensorLM: Learning the Language of Wearable Sensors 

**Title (ZH)**: SensorLM：学习可穿戴传感器的语言 

**Authors**: Yuwei Zhang, Kumar Ayush, Siyuan Qiao, A. Ali Heydari, Girish Narayanswamy, Maxwell A. Xu, Ahmed A. Metwally, Shawn Xu, Jake Garrison, Xuhai Xu, Tim Althoff, Yun Liu, Pushmeet Kohli, Jiening Zhan, Mark Malhotra, Shwetak Patel, Cecilia Mascolo, Xin Liu, Daniel McDuff, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09108)  

**Abstract**: We present SensorLM, a family of sensor-language foundation models that enable wearable sensor data understanding with natural language. Despite its pervasive nature, aligning and interpreting sensor data with language remains challenging due to the lack of paired, richly annotated sensor-text descriptions in uncurated, real-world wearable data. We introduce a hierarchical caption generation pipeline designed to capture statistical, structural, and semantic information from sensor data. This approach enabled the curation of the largest sensor-language dataset to date, comprising over 59.7 million hours of data from more than 103,000 people. Furthermore, SensorLM extends prominent multimodal pretraining architectures (e.g., CLIP, CoCa) and recovers them as specific variants within a generic architecture. Extensive experiments on real-world tasks in human activity analysis and healthcare verify the superior performance of SensorLM over state-of-the-art in zero-shot recognition, few-shot learning, and cross-modal retrieval. SensorLM also demonstrates intriguing capabilities including scaling behaviors, label efficiency, sensor captioning, and zero-shot generalization to unseen tasks. 

**Abstract (ZH)**: 我们呈现了SensorLM，这是一个传感器-语言基础模型家族，能够利用自然语言理解穿戴式传感器数据。尽管传感器数据与语言的对齐和解释具有普遍性，但由于未整理的真实世界穿戴数据中缺乏配对的、标注丰富的传感器-文本描述，这一过程仍然具有挑战性。我们引入了一个分层标题生成流水线，旨在从传感器数据中捕捉统计、结构和语义信息。这一方法促成了迄今为止最大的传感器-语言数据集的编目，包含超过103,000人的5970多万小时数据。此外，SensorLM 扩展了著名的多模态预训练架构（如 CLIP、CoCa），并将其作为通用架构下的特定变体。在人类活动分析和医疗保健等实际任务上的广泛实验验证了SensorLM 在零样本识别、少样本学习和跨模态检索方面的优越性能。SensorLM 还展示了扩展行为、标签效率、传感器标题生成和对未见任务的零样本泛化等有趣的特性。 

---
# Enhancing the Safety of Medical Vision-Language Models by Synthetic Demonstrations 

**Title (ZH)**: 通过合成示范增强医疗视觉语言模型的安全性 

**Authors**: Zhiyu Xue, Reza Abbasi-Asl, Ramtin Pedarsani  

**Link**: [PDF](https://arxiv.org/pdf/2506.09067)  

**Abstract**: Generative medical vision-language models~(Med-VLMs) are primarily designed to generate complex textual information~(e.g., diagnostic reports) from multimodal inputs including vision modality~(e.g., medical images) and language modality~(e.g., clinical queries). However, their security vulnerabilities remain underexplored. Med-VLMs should be capable of rejecting harmful queries, such as \textit{Provide detailed instructions for using this CT scan for insurance fraud}. At the same time, addressing security concerns introduces the risk of over-defense, where safety-enhancing mechanisms may degrade general performance, causing Med-VLMs to reject benign clinical queries. In this paper, we propose a novel inference-time defense strategy to mitigate harmful queries, enabling defense against visual and textual jailbreak attacks. Using diverse medical imaging datasets collected from nine modalities, we demonstrate that our defense strategy based on synthetic clinical demonstrations enhances model safety without significantly compromising performance. Additionally, we find that increasing the demonstration budget alleviates the over-defense issue. We then introduce a mixed demonstration strategy as a trade-off solution for balancing security and performance under few-shot demonstration budget constraints. 

**Abstract (ZH)**: 生成式医学视图语言模型（Med-VLMs）主要设计用于从包括视觉模态（如医学图像）和语言模态（如临床查询）在内的多模态输入中生成复杂的文本信息（例如，诊断报告）。然而，它们的安全性漏洞尚未得到充分探索。Med-VLMs 应该能够拒绝有害查询，如“提供有关如何使用此CT扫描进行保险欺诈的详细说明”。同时，解决安全问题会带来过度防御的风险，即增强安全的机制可能会损害总体性能，导致 Med-VLMs 拒绝正常的临床查询。在本文中，我们提出了一种新颖的推理时防御策略，以减轻有害查询的影响，从而抵御视觉和文本方面的 jailbreak 攻击。通过使用从九种模态采集的多种医学成像数据集，我们展示了基于合成临床示范的防御策略能增强模型的安全性而不显著牺牲性能。此外，我们发现增加示范预算可以缓解过度防御问题。然后，我们提出了一种混合示范策略作为安全与性能之间权衡的解决方案，在少量示范预算约束下平衡安全与性能。 

---
# TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization 

**Title (ZH)**: TGRPO：基于轨迹组相对策略优化的视觉-语言-动作模型微调 

**Authors**: Zengjue Chen, Runliang Niu, He Kong, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08440)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: this https URL 

**Abstract (ZH)**: Recent advances in Vision-Language-Action (VLA)模型已在大规模数据集上预训练，展示了在多样场景、任务和机器人平台上的强泛化能力。然而，这些模型仍需在新型环境中进行任务特定的微调，这一过程几乎完全依赖于使用静态轨迹数据集的监督微调（SFT）。此类方法既不允许机器人与环境互动，也不利用实时执行的反馈。此外，它们的成功高度依赖于收集的轨迹的数量和质量。强化学习（RL）提供了一种有前景的替代方法，通过实现闭环交互并直接将学习策略与任务目标对齐。在这项工作中，我们从GRPO的思想中汲取灵感，提出了轨迹级组相对策略优化（TGRPO）方法。通过融合步级和轨迹级的优势信号，该方法改进了GRPO的组级优势估计，从而使算法更适合VLA的在线强化学习训练。实验结果表明，TGRPO在libero-object基准的十个操作任务上持续优于各种基线方法，能够在多种测试场景中生成更稳健和高效的策略。我们的源代码可在以下链接获取：this https URL。 

---
