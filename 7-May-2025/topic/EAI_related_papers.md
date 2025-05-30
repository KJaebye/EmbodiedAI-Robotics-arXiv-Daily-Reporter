# AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control 

**Title (ZH)**: AMO：自适应运动优化在超灵巧人形全身控制中的应用 

**Authors**: Jialong Li, Xuxin Cheng, Tianshu Huang, Shiqi Yang, Ri-Zhao Qiu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03738)  

**Abstract**: Humanoid robots derive much of their dexterity from hyper-dexterous whole-body movements, enabling tasks that require a large operational workspace: such as picking objects off the ground. However, achieving these capabilities on real humanoids remains challenging due to their high degrees of freedom (DoF) and nonlinear dynamics. We propose Adaptive Motion Optimization (AMO), a framework that integrates sim-to-real reinforcement learning (RL) with trajectory optimization for real-time, adaptive whole-body control. To mitigate distribution bias in motion imitation RL, we construct a hybrid AMO dataset and train a network capable of robust, on-demand adaptation to potentially O.O.D. commands. We validate AMO in simulation and on a 29-DoF Unitree G1 humanoid robot, demonstrating superior stability and an expanded workspace compared to strong baselines. Finally, we show that AMO's consistent performance supports autonomous task execution via imitation learning, underscoring the system's versatility and robustness. 

**Abstract (ZH)**: 类人机器人通过超灵巧的全身运动获得其灵巧性，能够执行需要大操作空间的任务，如捡拾地上的物体。然而，由于其高自由度和非线性动力学，要在实际的类人机器人上实现这些能力仍然具有挑战性。我们提出了自适应运动优化（AMO）框架，该框架将模拟到现实的强化学习（RL）与轨迹优化结合，以实现实时、自适应的全身控制。为减轻运动模仿RL中的分布偏差，我们构建了一个混合AMO数据集，并训练一个能够在潜在O.O.D.命令下实现鲁棒、按需适应的网络。我们在模拟中验证了AMO，并在具有29个自由度的Unitree G1类人机器人上进行了实验，结果显示出优于强基线系统的优越稳定性和扩展的工作空间。最后，我们展示了AMO的一致性能支持通过模仿学习实现自主任务执行，突显了该系统的多功能性和鲁棒性。 

---
# Visual Imitation Enables Contextual Humanoid Control 

**Title (ZH)**: 视觉模仿使情境驱动的人形控制成为可能 

**Authors**: Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.03729)  

**Abstract**: How can we teach humanoids to climb staircases and sit on chairs using the surrounding environment context? Arguably, the simplest way is to just show them-casually capture a human motion video and feed it to humanoids. We introduce VIDEOMIMIC, a real-to-sim-to-real pipeline that mines everyday videos, jointly reconstructs the humans and the environment, and produces whole-body control policies for humanoid robots that perform the corresponding skills. We demonstrate the results of our pipeline on real humanoid robots, showing robust, repeatable contextual control such as staircase ascents and descents, sitting and standing from chairs and benches, as well as other dynamic whole-body skills-all from a single policy, conditioned on the environment and global root commands. VIDEOMIMIC offers a scalable path towards teaching humanoids to operate in diverse real-world environments. 

**Abstract (ZH)**: 如何利用环境上下文教机器人爬楼梯和坐在椅子上？一种方法是直接向它们展示—随意捕捉人类动作视频并输入机器人中。我们提出了一种从现实到仿真再到现实的管道——VIDEOMIMIC，它挖掘日常生活中的视频，联合重建人类和环境，并生成使得类人机器人执行相应技能的全身控制策略。我们在真实的类人机器人上展示了该管道的结果，展示了如爬楼梯、上下楼梯、从椅子和长凳上坐下和站立等鲁棒且可重复的上下文控制，以及其他动态全身技能—这一切都来自一个单一的策略，该策略基于环境和全局基础命令进行条件控制。VIDEOMIMIC 提供了一条可扩展的路径，以便教会类人机器人操作多样化的现实世界环境。 

---
# Meta-Optimization and Program Search using Language Models for Task and Motion Planning 

**Title (ZH)**: 使用语言模型进行元优化和程序搜索以实现任务与运动规划 

**Authors**: Denis Shcherba, Eckart Cobo-Briesewitz, Cornelius V. Braun, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2505.03725)  

**Abstract**: Intelligent interaction with the real world requires robotic agents to jointly reason over high-level plans and low-level controls. Task and motion planning (TAMP) addresses this by combining symbolic planning and continuous trajectory generation. Recently, foundation model approaches to TAMP have presented impressive results, including fast planning times and the execution of natural language instructions. Yet, the optimal interface between high-level planning and low-level motion generation remains an open question: prior approaches are limited by either too much abstraction (e.g., chaining simplified skill primitives) or a lack thereof (e.g., direct joint angle prediction). Our method introduces a novel technique employing a form of meta-optimization to address these issues by: (i) using program search over trajectory optimization problems as an interface between a foundation model and robot control, and (ii) leveraging a zero-order method to optimize numerical parameters in the foundation model output. Results on challenging object manipulation and drawing tasks confirm that our proposed method improves over prior TAMP approaches. 

**Abstract (ZH)**: 智能与现实世界交互需要机器人代理联合推理高层规划和低层控制。任务与运动规划（TAMP）通过结合符号规划和连续轨迹生成来解决这一问题。近期，基础模型在TAMP方面的应用取得了令人印象深刻的成果，包括快速规划时间和执行自然语言指令。然而，高层规划与低层运动生成的最佳接口仍是一个开放问题：先前的方法要么过于抽象（例如，组合简化技能原语），要么不够抽象（例如，直接预测关节角度）。我们的方法引入了一种新颖的技术，通过元优化解决这些问题，包括：（i）使用轨迹优化问题上的程序搜索作为基础模型与机器人控制之间的接口；（ii）利用零阶方法优化基础模型输出中的数值参数。在具有挑战性的物体操作和绘画任务上的结果证实，我们提出的方法优于先前的TAMP方法。 

---
# Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach 

**Title (ZH)**: 自主监督学习在机器人树叶操作中的应用：几何-神经混合方法 

**Authors**: Srecharan Selvam, Abhishesh Silwal, George Kanter  

**Link**: [PDF](https://arxiv.org/pdf/2505.03702)  

**Abstract**: Automating leaf manipulation in agricultural settings faces significant challenges, including the variability of plant morphologies and deformable leaves. We propose a novel hybrid geometric-neural approach for autonomous leaf grasping that combines traditional computer vision with neural networks through self-supervised learning. Our method integrates YOLOv8 for instance segmentation and RAFT-Stereo for 3D depth estimation to build rich leaf representations, which feed into both a geometric feature scoring pipeline and a neural refinement module (GraspPointCNN). The key innovation is our confidence-weighted fusion mechanism that dynamically balances the contribution of each approach based on prediction certainty. Our self-supervised framework uses the geometric pipeline as an expert teacher to automatically generate training data. Experiments demonstrate that our approach achieves an 88.0% success rate in controlled environments and 84.7% in real greenhouse conditions, significantly outperforming both purely geometric (75.3%) and neural (60.2%) methods. This work establishes a new paradigm for agricultural robotics where domain expertise is seamlessly integrated with machine learning capabilities, providing a foundation for fully automated crop monitoring systems. 

**Abstract (ZH)**: 在农业生产环境中自动执行叶片操作面临显著挑战，包括植物形态的变异性与可变形的叶片。我们提出了一种结合传统计算机视觉与神经网络的新型几何-神经混合方法，通过自监督学习将两者结合起来进行自主叶片抓取。该方法利用YOLOv8进行实例分割和RAFT-Stereo进行3D深度估计，构建丰富的叶片表示，为几何特征评分管道和神经精炼模块（GraspPointCNN）提供输入。关键创新在于我们的置信加权融合机制，该机制基于预测的确定性动态平衡每种方法的贡献。我们的自监督框架使用几何管道作为专家教师，自动生成训练数据。实验表明，该方法在受控环境中的成功率达到了88.0%，在实际温室条件下的成功率为84.7%，显著优于纯几何方法（75.3%）和纯神经方法（60.2%）。这项研究建立了农业机器人领域的新范式，将领域专业知识无缝集成到机器学习能力中，为完全自动化的农作物监测系统提供了基础。 

---
# RoboOS: A Hierarchical Embodied Framework for Cross-Embodiment and Multi-Agent Collaboration 

**Title (ZH)**: RoboOS: 一种多层次 embodied 框架for 跨身躯和多智能体协作 

**Authors**: Huajie Tan, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Yaoxu Lyu, Mingyu Cao, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03673)  

**Abstract**: The dawn of embodied intelligence has ushered in an unprecedented imperative for resilient, cognition-enabled multi-agent collaboration across next-generation ecosystems, revolutionizing paradigms in autonomous manufacturing, adaptive service robotics, and cyber-physical production architectures. However, current robotic systems face significant limitations, such as limited cross-embodiment adaptability, inefficient task scheduling, and insufficient dynamic error correction. While End-to-end VLA models demonstrate inadequate long-horizon planning and task generalization, hierarchical VLA models suffer from a lack of cross-embodiment and multi-agent coordination capabilities. To address these challenges, we introduce RoboOS, the first open-source embodied system built on a Brain-Cerebellum hierarchical architecture, enabling a paradigm shift from single-agent to multi-agent intelligence. Specifically, RoboOS consists of three key components: (1) Embodied Brain Model (RoboBrain), a MLLM designed for global perception and high-level decision-making; (2) Cerebellum Skill Library, a modular, plug-and-play toolkit that facilitates seamless execution of multiple skills; and (3) Real-Time Shared Memory, a spatiotemporal synchronization mechanism for coordinating multi-agent states. By integrating hierarchical information flow, RoboOS bridges Embodied Brain and Cerebellum Skill Library, facilitating robust planning, scheduling, and error correction for long-horizon tasks, while ensuring efficient multi-agent collaboration through Real-Time Shared Memory. Furthermore, we enhance edge-cloud communication and cloud-based distributed inference to facilitate high-frequency interactions and enable scalable deployment. Extensive real-world experiments across various scenarios, demonstrate RoboOS's versatility in supporting heterogeneous embodiments. Project website: this https URL 

**Abstract (ZH)**: 拟人化智能的兴起带来了对未来代际生态系统中具备认知能力的多智能体协作的新迫切需求，革新了自主制造、自适应服务机器人和网络物理生产架构的范式。然而，现有机器人系统面临诸多限制，如跨体适应性有限、任务调度效率低下以及动态错误校正能力不足。尽管端到端的VLA模型在长期规划和任务通用性方面表现不足，而分层的VLA模型则缺乏跨体和多智能体协调能力。为应对这些挑战，我们提出了RoboOS，这是首个建立在脑-小脑分层架构上的开源拟人化系统，推动了从单智能体到多智能体智能的范式转变。RoboOS 包含三个关键组件：(1) 拟人化脑模型（RoboBrain），一种面向全局感知和高层次决策的MLLM；(2) 小脑技能库，一种模块化、即插即用工具包，使多技能无缝执行成为可能；(3) 实时共享内存，一种时空同步机制，用于协调多智能体状态。通过集成分层信息流，RoboOS 连接了拟人化脑和小脑技能库，支持长期任务的稳健规划、调度和错误校正，同时借助实时共享内存确保高效的多智能体协作。此外，我们增强了边缘-云通信和基于云的分布式推理，以促进高频交互并实现可扩展部署。在各种场景下的广泛真实世界实验证明了RoboOS 在支持异构体方面的灵活性。项目网站：this https URL。 

---
# Meta-reasoning Using Attention Maps and Its Applications in Cloud Robotics 

**Title (ZH)**: 基于注意力图的元推理及其在云机器人中的应用 

**Authors**: Adrian Lendinez, Renxi Qiu, Lanfranco Zanzi, Dayou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03587)  

**Abstract**: Metareasoning, a branch of AI, focuses on reasoning about reasons. It has the potential to enhance robots' decision-making processes in unexpected situations. However, the concept has largely been confined to theoretical discussions and case-by-case investigations, lacking general and practical solutions when the Value of Computation (VoC) is undefined, which is common in unexpected situations. In this work, we propose a revised meta-reasoning framework that significantly improves the scalability of the original approach in unexpected situations. This is achieved by incorporating semantic attention maps and unsupervised 'attention' updates into the metareasoning processes. To accommodate environmental dynamics, 'lines of thought' are used to bridge context-specific objects with abstracted attentions, while meta-information is monitored and controlled at the meta-level for effective reasoning. The practicality of the proposed approach is demonstrated through cloud robots deployed in real-world scenarios, showing improved performance and robustness. 

**Abstract (ZH)**: 元推理，人工智能的一个分支，专注于关于原因的推理。它有可能在意外情况下增强机器人的决策过程。然而，这一概念主要局限于理论讨论和个案研究，在计算价值（VoC）未定义的情况下缺乏一般性和实用性，这种情况在意外情况下很常见。在本工作中，我们提出了一种修订的元推理框架，该框架在意外情况下显著提高了原始方法的可伸缩性。这通过将语义注意力图和无监督的“注意力”更新纳入元推理过程而实现。为了适应环境动态，“思路线”被用于连接具体环境对象与抽象的注意力，同时在元层次上监控和控制元信息以实现有效的推理。通过实际部署在真实场景中的云机器人，证明了所提方法的实用性，并展示了其改进的性能和鲁棒性。 

---
# Task Reconstruction and Extrapolation for $π_0$ using Text Latent 

**Title (ZH)**: $π_0$ 的任务重建与外推 Using Text Latent 

**Authors**: Quanyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03500)  

**Abstract**: Vision-language-action models (VLAs) often achieve high performance on demonstrated tasks but struggle significantly when required to extrapolate, combining skills learned from different tasks in novel ways. For instance, VLAs might successfully put the cream cheese in the bowl and put the bowl on top of the cabinet, yet still fail to put the cream cheese on top of the cabinet. In this work, we demonstrate that behaviors from distinct tasks can be effectively recombined by manipulating the VLA's internal representations at inference time. Concretely, we identify the text latent by averaging the text tokens' hidden states across all demonstrated trajectories for a specific base task. For executing an extrapolated task, we can temporally interpolate the text latent of the two base tasks and add it back to the text hidden states, so sub-behaviors from the two tasks will be activated sequentially. We evaluate this approach using the newly created libero-ood benchmark, featuring 20 tasks extrapolated from standard LIBERO suites. The results on libero-ood show that all SOTA VLAs achieve < 15% success rate, while $\pi0$ with text latent interpolation reaches an 83% success rate. Further qualitative analysis reveals a tendency for VLAs to exhibit spatial overfitting, mapping object names to demonstrated locations rather than achieving genuine object and goal understanding. Additionally, we find that decoding the text latent yields human-unreadable prompts that can nevertheless instruct the VLA to achieve a 70% success rate on standard LIBERO suites, enabling private instruction or backdoor attacks. 

**Abstract (ZH)**: Vision-语言-动作模型（VLAs）在执行示范任务时通常表现出高水平性能，但在需要外推、以新颖方式结合来自不同任务所学技能时会面临显著挑战。例如，VLAs 可能能够成功地将奶油奶酪放入碗中，再将碗放在橱柜上，但仍无法将奶油奶酪放在橱柜上。在本项工作中，我们证明了可以通过在推断时操控 VLAs 的内部表示，从而有效重组来自不同任务的行为。具体而言，我们通过在特定基础任务的所有演示轨迹中平均文本标记的隐藏状态来识别文本潜在表示。执行外推任务时，我们可以按时间插值来自两个基础任务的文本潜在表示并重新添加到文本隐藏状态中，这样两个任务的子行为将依次被激活。我们使用包含20个从标准LIBERO套件外推而来的新创建的libero-ood基准对其进行评估。在libero-ood上的结果显示，所有最新VLAs的成功率均低于15%，而$\pi0$结合文本潜在表示插值法达到了83%的成功率。进一步的定性分析表明，VLAs可能表现出空间过拟合倾向，将物体名称映射到演示的位置上，而不是真正理解物体和目标。此外，我们发现解码文本潜在表示可以生成人类难以理解的提示，但这些提示仍能指导VLAs在标准LIBERO套件中达到70%的成功率，从而实现私人指导或后门攻击。 

---
# LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs 

**Title (ZH)**: LogisticsVLN：基于代理型无人机的低空终端配送视觉语言导航 

**Authors**: Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03460)  

**Abstract**: The growing demand for intelligent logistics, particularly fine-grained terminal delivery, underscores the need for autonomous UAV (Unmanned Aerial Vehicle)-based delivery systems. However, most existing last-mile delivery studies rely on ground robots, while current UAV-based Vision-Language Navigation (VLN) tasks primarily focus on coarse-grained, long-range goals, making them unsuitable for precise terminal delivery. To bridge this gap, we propose LogisticsVLN, a scalable aerial delivery system built on multimodal large language models (MLLMs) for autonomous terminal delivery. LogisticsVLN integrates lightweight Large Language Models (LLMs) and Visual-Language Models (VLMs) in a modular pipeline for request understanding, floor localization, object detection, and action-decision making. To support research and evaluation in this new setting, we construct the Vision-Language Delivery (VLD) dataset within the CARLA simulator. Experimental results on the VLD dataset showcase the feasibility of the LogisticsVLN system. In addition, we conduct subtask-level evaluations of each module of our system, offering valuable insights for improving the robustness and real-world deployment of foundation model-based vision-language delivery systems. 

**Abstract (ZH)**: Growing需求下的智能物流特别是精细末端配送强化了自主无人机交付系统的需求。然而，目前大多数最后一公里配送研究依赖地面机器人，而当前基于视觉-语言导航(VLN)的无人机任务主要侧重于粗粒度的远距离目标，使其不适合精确的末端配送。为弥合这一差距，我们提出LogisticsVLN，这是一种基于多模态大规模语言模型（MLLMs）构建的可扩展的空中交付系统，用于自主末端配送。LogisticsVLN 在模块化流水线中整合了轻量级的大规模语言模型（LLMs）和视觉-语言模型（VLMs），用于请求理解、楼层定位、物体检测和动作决策。为了支持这一新环境中的研究和评估，我们在CARLA模拟器中构建了Vision-Language Delivery (VLD) 数据集。LogisticsVLN系统在VLD数据集上的实验结果展示了该系统的可行性。此外，我们还对系统中每个模块的子任务水平进行了评估，为提高基于基础模型的视觉-语言交付系统的鲁棒性和实际部署提供了有价值的见解。 

---
# Close-Fitting Dressing Assistance Based on State Estimation of Feet and Garments with Semantic-based Visual Attention 

**Title (ZH)**: 基于足部和服装状态估计的语义引导视觉注意力适配穿着辅助 

**Authors**: Takuma Tsukakoshi, Tamon Miyake, Tetsuya Ogata, Yushi Wang, Takumi Akaishi, Shigeki Sugano  

**Link**: [PDF](https://arxiv.org/pdf/2505.03400)  

**Abstract**: As the population continues to age, a shortage of caregivers is expected in the future. Dressing assistance, in particular, is crucial for opportunities for social participation. Especially dressing close-fitting garments, such as socks, remains challenging due to the need for fine force adjustments to handle the friction or snagging against the skin, while considering the shape and position of the garment. This study introduces a method uses multi-modal information including not only robot's camera images, joint angles, joint torques, but also tactile forces for proper force interaction that can adapt to individual differences in humans. Furthermore, by introducing semantic information based on object concepts, rather than relying solely on RGB data, it can be generalized to unseen feet and background. In addition, incorporating depth data helps infer relative spatial relationship between the sock and the foot. To validate its capability for semantic object conceptualization and to ensure safety, training data were collected using a mannequin, and subsequent experiments were conducted with human subjects. In experiments, the robot successfully adapted to previously unseen human feet and was able to put socks on 10 participants, achieving a higher success rate than Action Chunking with Transformer and Diffusion Policy. These results demonstrate that the proposed model can estimate the state of both the garment and the foot, enabling precise dressing assistance for close-fitting garments. 

**Abstract (ZH)**: 随着人口老龄化趋势的加剧，未来护理人员短缺将是一个严峻的问题。特别是穿衣援助对于提高老年人的社会参与机会至关重要。尤其是穿紧身袜子等衣物仍然具有挑战性，因为需要精细地调整力量来处理与皮肤的摩擦或钩挂，同时考虑衣物的形状和位置。本研究介绍了一种方法，该方法利用多模态信息，包括不仅限于机器人摄像头图像、关节角度、关节扭矩，还利用触觉力进行适当的力量交互，以适应人类个体差异。此外，通过引入基于对象概念的语义信息，而不是仅依赖RGB数据，该方法可以泛化到未见过的脚和背景。此外，结合深度数据有助于推断袜子和脚之间的相对空间关系。为了验证其实现语义对象概念化的能力并确保安全，在人体模型上采集了训练数据，并在后续实验中使用人类受试者进行了实验。实验结果显示，机器人能够适应之前未见过的人类脚，并成功为10名参与者穿上了袜子，其成功率为Action Chunking with Transformer和Diffusion Policy的更高。这些结果表明，所提出模型能够估计衣物和脚的状态，从而实现对紧身衣物的精确穿衣辅助。 

---
# Effective Reinforcement Learning Control using Conservative Soft Actor-Critic 

**Title (ZH)**: 使用保守软Actor-Critic的有效强化学习控制 

**Authors**: Xinyi Yuan, Zhiwei Shang, Wenjun Huang, Yunduan Cui, Di Chen, Meixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03356)  

**Abstract**: Reinforcement Learning (RL) has shown great potential in complex control tasks, particularly when combined with deep neural networks within the Actor-Critic (AC) framework. However, in practical applications, balancing exploration, learning stability, and sample efficiency remains a significant challenge. Traditional methods such as Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) address these issues by incorporating entropy or relative entropy regularization, but often face problems of instability and low sample efficiency. In this paper, we propose the Conservative Soft Actor-Critic (CSAC) algorithm, which seamlessly integrates entropy and relative entropy regularization within the AC framework. CSAC improves exploration through entropy regularization while avoiding overly aggressive policy updates with the use of relative entropy regularization. Evaluations on benchmark tasks and real-world robotic simulations demonstrate that CSAC offers significant improvements in stability and efficiency over existing methods. These findings suggest that CSAC provides strong robustness and application potential in control tasks under dynamic environments. 

**Abstract (ZH)**: 保守Soft Actor-Critic (CSAC)算法在Actor-Critic框架中的应用 

---
# The Unreasonable Effectiveness of Discrete-Time Gaussian Process Mixtures for Robot Policy Learning 

**Title (ZH)**: 离散时间高斯过程混合模型在机器人策略学习中的意外有效性 

**Authors**: Jan Ole von Hartz, Adrian Röfer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03296)  

**Abstract**: We present Mixture of Discrete-time Gaussian Processes (MiDiGap), a novel approach for flexible policy representation and imitation learning in robot manipulation. MiDiGap enables learning from as few as five demonstrations using only camera observations and generalizes across a wide range of challenging tasks. It excels at long-horizon behaviors such as making coffee, highly constrained motions such as opening doors, dynamic actions such as scooping with a spatula, and multimodal tasks such as hanging a mug. MiDiGap learns these tasks on a CPU in less than a minute and scales linearly to large datasets. We also develop a rich suite of tools for inference-time steering using evidence such as collision signals and robot kinematic constraints. This steering enables novel generalization capabilities, including obstacle avoidance and cross-embodiment policy transfer. MiDiGap achieves state-of-the-art performance on diverse few-shot manipulation benchmarks. On constrained RLBench tasks, it improves policy success by 76 percentage points and reduces trajectory cost by 67%. On multimodal tasks, it improves policy success by 48 percentage points and increases sample efficiency by a factor of 20. In cross-embodiment transfer, it more than doubles policy success. We make the code publicly available at this https URL. 

**Abstract (ZH)**: 混合离散时间高斯过程在机器人操作中的灵活策略表示与模仿学习 

---
# Enabling Robots to Autonomously Search Dynamic Cluttered Post-Disaster Environments 

**Title (ZH)**: 使机器人能够自主搜索动态杂乱灾害后环境 

**Authors**: Karlo Rado, Mirko Baglioni, Anahita Jamshidnejad  

**Link**: [PDF](https://arxiv.org/pdf/2505.03283)  

**Abstract**: Robots will bring search and rescue (SaR) in disaster response to another level, in case they can autonomously take over dangerous SaR tasks from humans. A main challenge for autonomous SaR robots is to safely navigate in cluttered environments with uncertainties, while avoiding static and moving obstacles. We propose an integrated control framework for SaR robots in dynamic, uncertain environments, including a computationally efficient heuristic motion planning system that provides a nominal (assuming there are no uncertainties) collision-free trajectory for SaR robots and a robust motion tracking system that steers the robot to track this reference trajectory, taking into account the impact of uncertainties. The control architecture guarantees a balanced trade-off among various SaR objectives, while handling the hard constraints, including safety. The results of various computer-based simulations, presented in this paper, showed significant out-performance (of up to 42.3%) of the proposed integrated control architecture compared to two commonly used state-of-the-art methods (Rapidly-exploring Random Tree and Artificial Potential Function) in reaching targets (e.g., trapped victims in SaR) safely, collision-free, and in the shortest possible time. 

**Abstract (ZH)**: 机器人将在灾害救援响应中将搜索与救援（SaR）提升到一个新的水平，前提是它们能够自主接管危险的SaR任务。自主SaR机器人的主要挑战是在具有不确定性且杂乱的环境中安全导航，同时避免静态和移动障碍物。我们提出了一种集成控制框架，用于动态、不确定环境中的SaR机器人，包括一个计算高效的启发式运动规划系统，该系统为SaR机器人提供了一个名义上的（假设没有不确定性）无碰撞轨迹，并且包括一个鲁棒运动跟踪系统，该系统引导机器人跟踪参考轨迹，同时考虑不确定性的影响。该控制架构保证在处理各种SaR目标的同时实现平衡的权衡，并满足包括安全在内的严格约束。本文中提出的各种计算机仿真的结果表明，与两种常用的先进方法（快速扩展随机树和人工势场方法）相比，提出的集成控制架构在安全、无碰撞和最短时间到达目标（如SaR中的被困受害者）方面表现出显著的优越性（最高达42.3%）。 

---
# RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning 

**Title (ZH)**: RobotxR1：通过闭环强化学习在大规模语言模型中实现机器人的体験智能 

**Authors**: Liam Boyle, Nicolas Baumann, Paviththiren Sivasothilingam, Michele Magno, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2505.03238)  

**Abstract**: Future robotic systems operating in real-world environments will require on-board embodied intelligence without continuous cloud connection, balancing capabilities with constraints on computational power and memory. This work presents an extension of the R1-zero approach, which enables the usage of low parameter-count Large Language Models (LLMs) in the robotic domain. The R1-Zero approach was originally developed to enable mathematical reasoning in LLMs using static datasets. We extend it to the robotics domain through integration in a closed-loop Reinforcement Learning (RL) framework. This extension enhances reasoning in Embodied Artificial Intelligence (Embodied AI) settings without relying solely on distillation of large models through Supervised Fine-Tuning (SFT). We show that small-scale LLMs can achieve effective reasoning performance by learning through closed-loop interaction with their environment, which enables tasks that previously required significantly larger models. In an autonomous driving setting, a performance gain of 20.2%-points over the SFT-based baseline is observed with a Qwen2.5-1.5B model. Using the proposed training procedure, Qwen2.5-3B achieves a 63.3% control adaptability score, surpassing the 58.5% obtained by the much larger, cloud-bound GPT-4o. These results highlight that practical, on-board deployment of small LLMs is not only feasible but can outperform larger models if trained through environmental feedback, underscoring the importance of an interactive learning framework for robotic Embodied AI, one grounded in practical experience rather than static supervision. 

**Abstract (ZH)**: 未来在现实世界环境中运行的机器人系统将需要在不依赖连续云连接的情况下具备嵌入式智能，平衡计算能力和内存约束下的能力。本文扩展了R1-zero方法，使其能够在机器人领域使用低参数量大型语言模型（LLMs）。R1-Zero方法最初是为了解决在静态数据集上使用大语言模型进行数学推理的问题。通过将其集成到闭环强化学习（RL）框架中，我们将其扩展到机器人领域。此扩展在无需依赖通过监督微调（SFT）缩小大模型的基础上增强了嵌入式人工智能（Embodied AI）环境中的推理能力。实验表明，通过与环境的闭环交互学习，小型语言模型可以实现有效的推理性能，并能够执行先前需要更大模型的任务。在自动驾驶设置中，Qwen2.5-1.5B模型相较于基于SFT的基线模型性能提高了20.2%。使用建议的训练程序，Qwen2.5-3B实现了63.3%的控制适应性得分，超过了更大且依赖云服务的GPT-4o所获得的58.5%。这些结果表明，在实践中将小型LLMs部署在机器人嵌入式智能中不仅是可行的，而且通过环境反馈进行训练可以超越更大模型，突显了闭环学习框架的重要性，这一框架应基于实际经验而非静态监督。 

---
# GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data 

**Title (ZH)**: GraspVLA：一种预训练在十亿规模合成行动数据上的抓取基础模型 

**Authors**: Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03233)  

**Abstract**: Embodied foundation models are gaining increasing attention for their zero-shot generalization, scalability, and adaptability to new tasks through few-shot post-training. However, existing models rely heavily on real-world data, which is costly and labor-intensive to collect. Synthetic data offers a cost-effective alternative, yet its potential remains largely underexplored. To bridge this gap, we explore the feasibility of training Vision-Language-Action models entirely with large-scale synthetic action data. We curate SynGrasp-1B, a billion-frame robotic grasping dataset generated in simulation with photorealistic rendering and extensive domain randomization. Building on this, we present GraspVLA, a VLA model pretrained on large-scale synthetic action data as a foundational model for grasping tasks. GraspVLA integrates autoregressive perception tasks and flow-matching-based action generation into a unified Chain-of-Thought process, enabling joint training on synthetic action data and Internet semantics data. This design helps mitigate sim-to-real gaps and facilitates the transfer of learned actions to a broader range of Internet-covered objects, achieving open-vocabulary generalization in grasping. Extensive evaluations across real-world and simulation benchmarks demonstrate GraspVLA's advanced zero-shot generalizability and few-shot adaptability to specific human preferences. We will release SynGrasp-1B dataset and pre-trained weights to benefit the community. 

**Abstract (ZH)**: 基于体态的本体模型因其实现零样本泛化、扩展性和通过少样本后训练适应新任务的能力而日益受到关注。然而，现有模型高度依赖真实世界数据，收集这些数据既昂贵又耗时。合成数据提供了一种成本效益较高的替代方案，但其潜力尚未得到充分探索。为解决这一问题，我们探索了使用大规模合成动作数据完全训练视觉-语言-行动模型的可能性。我们编纂了SynGrasp-1B，这是一个在模拟中通过照片写实渲染和广泛领域随机化生成的一百亿帧的机器人抓取数据集。在此基础上，我们提出了GraspVLA，这是一种基于大规模合成动作数据预训练的视觉-语言-行动模型，作为抓取任务的基础模型。GraspVLA 将自回归感知任务和流动匹配基于的动作生成统一到一个思考链过程中，使其能够在合成动作数据和互联网语义数据上进行联合训练。这种设计有助于缩小模拟与现实之间的差距，并促进学习到的动作转移到更广泛的互联网覆盖对象上，实现开放词汇的抓取泛化。针对现实世界和模拟基准的广泛评估展示了GraspVLA 高级的零样本泛化能力和特定人类偏好的少样本适应能力。我们将发布SynGrasp-1B 数据集和预训练权重以造福社区。 

---
# Learn to Swim: Data-Driven LSTM Hydrodynamic Model for Quadruped Robot Gait Optimization 

**Title (ZH)**: 学会游泳：基于数据驱动的LSTM水动力模型在四足机器人步态优化中的应用 

**Authors**: Fei Han, Pengming Guo, Hao Chen, Weikun Li, Jingbo Ren, Naijun Liu, Ning Yang, Dixia Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03146)  

**Abstract**: This paper presents a Long Short-Term Memory network-based Fluid Experiment Data-Driven model (FED-LSTM) for predicting unsteady, nonlinear hydrodynamic forces on the underwater quadruped robot we constructed. Trained on experimental data from leg force and body drag tests conducted in both a recirculating water tank and a towing tank, FED-LSTM outperforms traditional Empirical Formulas (EF) commonly used for flow prediction over flat surfaces. The model demonstrates superior accuracy and adaptability in capturing complex fluid dynamics, particularly in straight-line and turning-gait optimizations via the NSGA-II algorithm. FED-LSTM reduces deflection errors during straight-line swimming and improves turn times without increasing the turning radius. Hardware experiments further validate the model's precision and stability over EF. This approach provides a robust framework for enhancing the swimming performance of legged robots, laying the groundwork for future advances in underwater robotic locomotion. 

**Abstract (ZH)**: 基于长短期记忆网络的流体实验数据驱动模型（FED-LSTM）用于预测我们构建的水下四足机器人不稳态、非线性水动力力 

---
# Latent Adaptive Planner for Dynamic Manipulation 

**Title (ZH)**: 动态操作的潜适应规划器 

**Authors**: Donghun Noh, Deqian Kong, Minglu Zhao, Andrew Lizarraga, Jianwen Xie, Ying Nian Wu, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.03077)  

**Abstract**: This paper presents Latent Adaptive Planner (LAP), a novel approach for dynamic nonprehensile manipulation tasks that formulates planning as latent space inference, effectively learned from human demonstration videos. Our method addresses key challenges in visuomotor policy learning through a principled variational replanning framework that maintains temporal consistency while efficiently adapting to environmental changes. LAP employs Bayesian updating in latent space to incrementally refine plans as new observations become available, striking an optimal balance between computational efficiency and real-time adaptability. We bridge the embodiment gap between humans and robots through model-based proportional mapping that regenerates accurate kinematic-dynamic joint states and object positions from human demonstrations. Experimental evaluations across multiple complex manipulation benchmarks demonstrate that LAP achieves state-of-the-art performance, outperforming existing approaches in success rate, trajectory smoothness, and energy efficiency, particularly in dynamic adaptation scenarios. Our approach enables robots to perform complex interactions with human-like adaptability while providing an expandable framework applicable to diverse robotic platforms using the same human demonstration videos. 

**Abstract (ZH)**: 基于潜在空间推断的latent adaptive planner（LAP）：一种新颖的动态非拾取操作规划方法 

---
# MORE: Mobile Manipulation Rearrangement Through Grounded Language Reasoning 

**Title (ZH)**: 基于接地语言推理的移动 manipulator 重组 

**Authors**: Mohammad Mohammadi, Daniel Honerkamp, Martin Büchner, Matteo Cassinelli, Tim Welschehold, Fabien Despinoy, Igor Gilitschenski, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03035)  

**Abstract**: Autonomous long-horizon mobile manipulation encompasses a multitude of challenges, including scene dynamics, unexplored areas, and error recovery. Recent works have leveraged foundation models for scene-level robotic reasoning and planning. However, the performance of these methods degrades when dealing with a large number of objects and large-scale environments. To address these limitations, we propose MORE, a novel approach for enhancing the capabilities of language models to solve zero-shot mobile manipulation planning for rearrangement tasks. MORE leverages scene graphs to represent environments, incorporates instance differentiation, and introduces an active filtering scheme that extracts task-relevant subgraphs of object and region instances. These steps yield a bounded planning problem, effectively mitigating hallucinations and improving reliability. Additionally, we introduce several enhancements that enable planning across both indoor and outdoor environments. We evaluate MORE on 81 diverse rearrangement tasks from the BEHAVIOR-1K benchmark, where it becomes the first approach to successfully solve a significant share of the benchmark, outperforming recent foundation model-based approaches. Furthermore, we demonstrate the capabilities of our approach in several complex real-world tasks, mimicking everyday activities. We make the code publicly available at this https URL. 

**Abstract (ZH)**: 自主长时移动操作涵盖了多种挑战，包括场景动态、未探索区域和错误恢复。近期研究利用基础模型进行场景级别的机器人推理和规划。然而，当处理大量对象和大规模环境时，这些方法的性能会下降。为克服这些限制，我们提出MORE，一种增强语言模型解决零样本移动操作规划以进行重排任务能力的新方法。MORE利用场景图表示环境，引入实例差异化，并引入一种积极筛选方案，提取与任务相关的对象和区域实例子图。这些步骤产生了有界规划问题，有效减弱了幻觉并提高了可靠性。此外，我们引入了几种增强措施，使规划能够在室内外环境中进行。我们在BEHAVIOR-1K基准上的81个多样重排任务中评估了MORE，使其成为首个成功解决基准测试显著比例方法，并优于基于基础模型的近期方法。此外，我们展示了该方法在多个复杂真实世界的任务中的能力，模拟日常活动。代码已公开于此链接。 

---
# Artificial Behavior Intelligence: Technology, Challenges, and Future Directions 

**Title (ZH)**: 人工行为智能：技术、挑战及未来方向 

**Authors**: Kanghyun Jo, Jehwan Choi, Kwanho Kim, Seongmin Kim, Duy-Linh Nguyen, Xuan-Thuy Vo, Adri Priadana, Tien-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2505.03315)  

**Abstract**: Understanding and predicting human behavior has emerged as a core capability in various AI application domains such as autonomous driving, smart healthcare, surveillance systems, and social robotics. This paper defines the technical framework of Artificial Behavior Intelligence (ABI), which comprehensively analyzes and interprets human posture, facial expressions, emotions, behavioral sequences, and contextual cues. It details the essential components of ABI, including pose estimation, face and emotion recognition, sequential behavior analysis, and context-aware modeling. Furthermore, we highlight the transformative potential of recent advances in large-scale pretrained models, such as large language models (LLMs), vision foundation models, and multimodal integration models, in significantly improving the accuracy and interpretability of behavior recognition. Our research team has a strong interest in the ABI domain and is actively conducting research, particularly focusing on the development of intelligent lightweight models capable of efficiently inferring complex human behaviors. This paper identifies several technical challenges that must be addressed to deploy ABI in real-world applications including learning behavioral intelligence from limited data, quantifying uncertainty in complex behavior prediction, and optimizing model structures for low-power, real-time inference. To tackle these challenges, our team is exploring various optimization strategies including lightweight transformers, graph-based recognition architectures, energy-aware loss functions, and multimodal knowledge distillation, while validating their applicability in real-time environments. 

**Abstract (ZH)**: 理解与预测人类行为已成为自动驾驶、智能医疗、监控系统和社会机器人等领域的核心能力。本文定义了人工行为智能（ABI）的技术框架，全面分析和解释了人类的姿态、面部表情、情绪、行为序列和上下文线索。详细阐述了ABI的关键组成部分，包括姿态估计、面部和情绪识别、序列行为分析以及上下文感知建模。此外，本文还强调了大规模预训练模型（如大型语言模型、视觉基础模型和多模态集成模型）的最新进展在大幅提高行为识别的准确性和可解释性方面的潜力。我们的研究团队对ABI领域非常感兴趣，并积极进行研究，特别是侧重于开发高效推理复杂人类行为的智能轻量级模型。本文指出了在实际应用中部署ABI所必须解决的技术挑战，包括从有限数据中学习行为智能、在复杂行为预测中量化不确定性以及优化低功耗、实时推理的模型结构。为了应对这些挑战，我们的团队正在探索各种优化策略，包括轻量级变换器、基于图的识别架构、能量感知损失函数以及多模态知识精简技术，并在实时环境中验证其适用性。 

---
# Null Counterfactual Factor Interactions for Goal-Conditioned Reinforcement Learning 

**Title (ZH)**: 无偏反事实因素交互作用在目标条件 reinforcement 学习中 

**Authors**: Caleb Chuck, Fan Feng, Carl Qi, Chang Shi, Siddhant Agarwal, Amy Zhang, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2505.03172)  

**Abstract**: Hindsight relabeling is a powerful tool for overcoming sparsity in goal-conditioned reinforcement learning (GCRL), especially in certain domains such as navigation and locomotion. However, hindsight relabeling can struggle in object-centric domains. For example, suppose that the goal space consists of a robotic arm pushing a particular target block to a goal location. In this case, hindsight relabeling will give high rewards to any trajectory that does not interact with the block. However, these behaviors are only useful when the object is already at the goal -- an extremely rare case in practice. A dataset dominated by these kinds of trajectories can complicate learning and lead to failures. In object-centric domains, one key intuition is that meaningful trajectories are often characterized by object-object interactions such as pushing the block with the gripper. To leverage this intuition, we introduce Hindsight Relabeling using Interactions (HInt), which combines interactions with hindsight relabeling to improve the sample efficiency of downstream RL. However because interactions do not have a consensus statistical definition tractable for downstream GCRL, we propose a definition of interactions based on the concept of null counterfactual: a cause object is interacting with a target object if, in a world where the cause object did not exist, the target object would have different transition dynamics. We leverage this definition to infer interactions in Null Counterfactual Interaction Inference (NCII), which uses a "nulling'' operation with a learned model to infer interactions. NCII is able to achieve significantly improved interaction inference accuracy in both simple linear dynamics domains and dynamic robotic domains in Robosuite, Robot Air Hockey, and Franka Kitchen and HInt improves sample efficiency by up to 4x. 

**Abstract (ZH)**: hindsight relabeling for object-centric reinforcement learning using interactions (hint) 

---
# Cognitio Emergens: Agency, Dimensions, and Dynamics in Human-AI Knowledge Co-Creation 

**Title (ZH)**: 认知涌现：人类-人工智能知识共创的代理、维度与动力机制 

**Authors**: Xule Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03105)  

**Abstract**: Scientific knowledge creation is fundamentally transforming as humans and AI systems evolve beyond tool-user relationships into co-evolutionary epistemic partnerships. When AlphaFold revolutionized protein structure prediction, researchers described engaging with an epistemic partner that reshaped how they conceptualized fundamental relationships. This article introduces Cognitio Emergens (CE), a framework addressing critical limitations in existing models that focus on static roles or narrow metrics while failing to capture how scientific understanding emerges through recursive human-AI interaction over time. CE integrates three components addressing these limitations: Agency Configurations describing how authority distributes between humans and AI (Directed, Contributory, Partnership), with partnerships dynamically oscillating between configurations rather than following linear progression; Epistemic Dimensions capturing six specific capabilities emerging through collaboration across Discovery, Integration, and Projection axes, creating distinctive "capability signatures" that guide development; and Partnership Dynamics identifying forces shaping how these relationships evolve, particularly the risk of epistemic alienation where researchers lose interpretive control over knowledge they formally endorse. Drawing from autopoiesis theory, social systems theory, and organizational modularity, CE reveals how knowledge co-creation emerges through continuous negotiation of roles, values, and organizational structures. By reconceptualizing human-AI scientific collaboration as fundamentally co-evolutionary, CE offers a balanced perspective that neither uncritically celebrates nor unnecessarily fears AI's evolving role, instead providing conceptual tools for cultivating partnerships that maintain meaningful human participation while enabling transformative scientific breakthroughs. 

**Abstract (ZH)**: 科学知识的创造正在从根本上被重塑，随着人类和AI系统的进化超越工具使用者的关系，进入到共生认知伙伴关系。当AlphaFold革新蛋白质结构预测时，研究人员描述了与一个重塑他们基本认知关系的共生认知伙伴互动的经历。本文介绍了认知涌现（Cognitio Emergens，CE）框架，该框架解决现有模型中聚焦于静态角色或狭窄指标的局限性，而未能捕捉到随着时间的推移，通过反复的人类-AI互动中科学理解是如何涌现的。CE整合了三个组成部分来解决这些局限性：主体配置（Agency Configurations），描述权力在人类和AI之间的分配（定向、贡献、伙伴关系），伙伴关系在这些配置之间动态摆动而非线性进展；认知维度（Epistemic Dimensions），捕捉协作过程中在发现、整合和预测轴上六种特定能力的涌现，创造出独特的“能力特征”，指导发展；以及伙伴关系动力学（Partnership Dynamics），识别塑造这些关系演变的力量，尤其是知识异化的风险，即研究人员失去了他们正式认可的知识的解释控制。借助自动调节理论、社会系统理论和组织模块化理论，CE揭示了知识共同创造如何通过持续的角色、价值观和组织结构的协商而涌现。通过将人类-AI科学合作重新构建为根本上的共生进化，CE提供了一种平衡的观点，既不过分庆祝也不无必要地恐惧AI角色的演变，而是提供了实现有意义的人类参与与颠覆性科学突破之间的概念工具。 

---
