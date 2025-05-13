# H$^{\mathbf{3}}$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning 

**Title (ZH)**: H$^{\mathbf{3}}$DP: 三层次扩散策略用于视听运动学习 

**Authors**: Yiyang Lu, Yufeng Tian, Zhecheng Yuan, Xianbang Wang, Pu Hua, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07819)  

**Abstract**: Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce $\textbf{Triply-Hierarchical Diffusion Policy}~(\textbf{H$^{\mathbf{3}}$DP})$, a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^{3}$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^{3}$DP yields a $\mathbf{+27.5\%}$ average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: this https URL. 

**Abstract (ZH)**: 三重嵌套扩散政策（Triply-Hierarchical Diffusion Policy, H³DP）：强化视觉与动作生成集成的visuomotor学习框架 

---
# Pixel Motion as Universal Representation for Robot Control 

**Title (ZH)**: 像素运动作为机器人控制的通用表示 

**Authors**: Kanchana Ranasinghe, Xiang Li, Cristina Mata, Jongwoo Park, Michael S Ryoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07817)  

**Abstract**: We present LangToMo, a vision-language-action framework structured as a dual-system architecture that uses pixel motion forecasts as intermediate representations. Our high-level System 2, an image diffusion model, generates text-conditioned pixel motion sequences from a single frame to guide robot control. Pixel motion-a universal, interpretable, and motion-centric representation-can be extracted from videos in a self-supervised manner, enabling diffusion model training on web-scale video-caption data. Treating generated pixel motion as learned universal representations, our low level System 1 module translates these into robot actions via motion-to-action mapping functions, which can be either hand-crafted or learned with minimal supervision. System 2 operates as a high-level policy applied at sparse temporal intervals, while System 1 acts as a low-level policy at dense temporal intervals. This hierarchical decoupling enables flexible, scalable, and generalizable robot control under both unsupervised and supervised settings, bridging the gap between language, motion, and action. Checkout this https URL for visualizations. 

**Abstract (ZH)**: LangToMo：一种基于像素运动预测的视觉-语言-动作框架 

---
# Imagine, Verify, Execute: Memory-Guided Agentic Exploration with Vision-Language Models 

**Title (ZH)**: 想象、验证与执行：基于视觉-语言模型的记忆引导代理探索 

**Authors**: Seungjae Lee, Daniel Ekpo, Haowen Liu, Furong Huang, Abhinav Shrivastava, Jia-Bin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07815)  

**Abstract**: Exploration is essential for general-purpose robotic learning, especially in open-ended environments where dense rewards, explicit goals, or task-specific supervision are scarce. Vision-language models (VLMs), with their semantic reasoning over objects, spatial relations, and potential outcomes, present a compelling foundation for generating high-level exploratory behaviors. However, their outputs are often ungrounded, making it difficult to determine whether imagined transitions are physically feasible or informative. To bridge the gap between imagination and execution, we present IVE (Imagine, Verify, Execute), an agentic exploration framework inspired by human curiosity. Human exploration is often driven by the desire to discover novel scene configurations and to deepen understanding of the environment. Similarly, IVE leverages VLMs to abstract RGB-D observations into semantic scene graphs, imagine novel scenes, predict their physical plausibility, and generate executable skill sequences through action tools. We evaluate IVE in both simulated and real-world tabletop environments. The results show that IVE enables more diverse and meaningful exploration than RL baselines, as evidenced by a 4.1 to 7.8x increase in the entropy of visited states. Moreover, the collected experience supports downstream learning, producing policies that closely match or exceed the performance of those trained on human-collected demonstrations. 

**Abstract (ZH)**: 探索对于通用机器人学习至关重要，特别是在奖励稀疏、缺乏明确目标或任务特定监督的开放环境中。视觉语言模型（VLMs）通过其对对象、空间关系和潜在结果的语义推理，为生成高层次的探索行为提供了有力的基础。然而，它们的输出往往是未grounded的，难以确定想象中的过渡是否物理上可行或具有信息性。为了弥合想象与执行之间的差距，我们提出了IVE（Imagine, Verify, Execute）探索框架，该框架受到人类好奇心的启发。人类的探索往往受到发现新型场景配置和加深对环境理解的驱动。同样，IVE 利用VLMs将RGB-D观察抽象成语义场景图，设想新的场景，预测它们的物理可行性，并通过动作工具生成可执行的能力序列。我们在模拟和真实世界的桌面上评估了IVE。结果表明，IVE 使探索变得更加多样和有意义，如访问状态的熵相比RL基准提高了4.1到7.8倍。此外，收集的经验支持下游学习，产生的策略与或超过了在人类收集的示范上训练的性能。 

---
# DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies 

**Title (ZH)**: DexWild: 动手动脚的户外机器人政策中的灵巧人类交互 

**Authors**: Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2505.07813)  

**Abstract**: Large-scale, diverse robot datasets have emerged as a promising path toward enabling dexterous manipulation policies to generalize to novel environments, but acquiring such datasets presents many challenges. While teleoperation provides high-fidelity datasets, its high cost limits its scalability. Instead, what if people could use their own hands, just as they do in everyday life, to collect data? In DexWild, a diverse team of data collectors uses their hands to collect hours of interactions across a multitude of environments and objects. To record this data, we create DexWild-System, a low-cost, mobile, and easy-to-use device. The DexWild learning framework co-trains on both human and robot demonstrations, leading to improved performance compared to training on each dataset individually. This combination results in robust robot policies capable of generalizing to novel environments, tasks, and embodiments with minimal additional robot-specific data. Experimental results demonstrate that DexWild significantly improves performance, achieving a 68.5% success rate in unseen environments-nearly four times higher than policies trained with robot data only-and offering 5.8x better cross-embodiment generalization. Video results, codebases, and instructions at this https URL 

**Abstract (ZH)**: 大规模多样化的机器人数据集已成为实现灵巧操作策略泛化到新型环境的一种有前景的方法，但获取这样的数据集面临着许多挑战。虽然远程操作可以提供高保真数据集，但其高昂的成本限制了其 scalability。相反，如果人们可以像日常生活一样使用他们的手来收集数据会怎样？在 DexWild 中，一个多样化的数据收集团队使用他们的手在多种环境和对象上收集数小时的交互数据。为了记录这些数据，我们创建了 DexWild-System，这是一个低成本、便携且易于使用的设备。DexWild 学习框架在人类和机器人演示数据上共同训练，与单独训练每个数据集相比，能够取得更好的性能。这种组合产生了鲁棒性强的机器人策略，能够在最小限度增加特定于机器人数据的情况下泛化到新型环境、任务和不同体态。实验结果表明，DexWild 显著提高了性能，在未见过的环境中取得了 68.5% 的成功率——几乎是仅使用机器人数据训练的策略成功率的四倍，并且在不同体态泛化方面表现出了 5.8 倍的改进。更多信息、视频结果、代码库和使用说明请访问：this https URL。 

---
# Intuitive Human-Robot Interfaces Leveraging on Autonomy Features for the Control of Highly-redundant Robots 

**Title (ZH)**: 基于自主性特征的直观人机机器人接口及其在高冗余度机器人控制中的应用 

**Authors**: Davide Torielli  

**Link**: [PDF](https://arxiv.org/pdf/2505.07668)  

**Abstract**: [...] With the TelePhysicalOperation interface, the user can teleoperate the different capabilities of a robot (e.g., single/double arm manipulation, wheel/leg locomotion) by applying virtual forces on selected robot body parts. This approach emulates the intuitiveness of physical human-robot interaction, but at the same time it permits to teleoperate the robot from a safe distance, in a way that resembles a "Marionette" interface. The system is further enhanced with wearable haptic feedback functions to align better with the "Marionette" metaphor, and a user study has been conducted to validate its efficacy with and without the haptic channel enabled. Considering the importance of robot independence, the TelePhysicalOperation interface incorporates autonomy modules to face, for example, the teleoperation of dual-arm mobile base robots for bimanual object grasping and transportation tasks.
With the laser-guided interface, the user can indicate points of interest to the robot through the utilization of a simple but effective laser emitter device. With a neural network-based vision system, the robot tracks the laser projection in real time, allowing the user to indicate not only fixed goals, like objects, but also paths to follow. With the implemented autonomous behavior, a mobile manipulator employs its locomanipulation abilities to follow the indicated goals. The behavior is modeled using Behavior Trees, exploiting their reactivity to promptly respond to changes in goal positions, and their modularity to adapt the motion planning to the task needs. The proposed laser interface has also been employed in an assistive scenario. In this case, users with upper limbs impairments can control an assistive manipulator by directing a head-worn laser emitter to the point of interests, to collaboratively address activities of everyday life. [...] 

**Abstract (ZH)**: 通过TelePhysicalOperation界面，用户可以通过在选定的机器人身体部位应用虚拟力来远程操作机器人的不同能力（如单臂/双臂操作、轮式/腿式移动）。这种方法模仿了物理的人机互动直觉性，同时允许用户从安全距离远程操作机器人，类似于“提线木偶”接口。该系统进一步增强了穿戴式力反馈功能，更好地契合“提线木偶”的隐喻，并进行了用户研究以验证其在启用和未启用力反馈通道时的有效性。考虑到机器人独立性的重要性，TelePhysicalOperation界面集成自主模块，例如，用于远程操作具有双臂移动基座的机器人进行双臂对象抓取和运输任务。

通过激光引导界面，用户可以通过简单的激光发射器设备指示机器人感兴趣点。借助基于神经网络的视觉系统，机器人可以实时跟踪激光投影，从而让用户不仅指示固定目标（如物体），还可以指示跟随的路径。在实现的自主行为中，移动机器人利用其定位操作能力跟随指示目标。行为被建模为行为树，利用其实时响应能力对目标位置变化做出迅速反应，并利用其模块化特性以适应任务需求进行运动规划。所提出的激光界面还应用于辅助场景。在这种情况下，上肢受损的用户可以通过头部佩戴的激光发射器控制辅助操作器，以协作应对日常生活活动。 

---
# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents 

**Title (ZH)**: 神经脑：一种受神经科学启发的体态智能框架 

**Authors**: Jian Liu, Xiongtao Shi, Thai Duy Nguyen, Haitian Zhang, Tianxiang Zhang, Wei Sun, Yanjie Li, Athanasios V. Vasilakos, Giovanni Iacca, Arshad Ali Khan, Arvind Kumar, Jae Won Cho, Ajmal Mian, Lihua Xie, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07634)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios. 

**Abstract (ZH)**: 人工智能的快速进化已从静态、数据驱动的模型转向能够感知和互动的动态系统。尽管在模式识别和符号推理方面取得了进展，当前的人工智能系统，如大型语言模型，依然缺乏实体性，无法实际与世界互动。这一限制推动了具身人工智能的发展，其中自主代理，如类人机器人，必须具备人类般的适应性来导航和操控非结构化的环境。这一挑战的核心在于神经脑的概念，这是一种设计用于驱动具有人类适应性的具身代理的中枢智能系统。神经脑必须无缝地整合多模态感知与认知能力。实现这一点还需要一个适应性记忆系统和高效的硬件-软件协同设计，以实现动态环境中的实时行动。本文介绍了具身代理的统一神经脑框架，解决了两个基本挑战：（1）定义神经脑的核心组件；（2）弥合静态人工智能模型与现实世界部署所需动态适应性之间的差距。为此，我们提出了一种受生物学启发的架构，该架构整合了多模态主动感知、感知-认知-行动功能、基于神经可塑性的记忆存储与更新，以及神经形态硬件/软件优化。此外，我们还回顾了在这些四个方面的最新研究成果，并分析了当前人工智能系统与人类智能之间的差距。通过综合神经科学的见解，我们勾勒出了一条发展通用、自主代理以在现实场景中实现人类级智能的道路。 

---
# Average-Reward Maximum Entropy Reinforcement Learning for Global Policy in Double Pendulum Tasks 

**Title (ZH)**: 双摆任务中的平均奖励最大熵强化学习全局策略 

**Authors**: Jean Seong Bjorn Choe, Bumkyu Choi, Jong-kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07516)  

**Abstract**: This report presents our reinforcement learning-based approach for the swing-up and stabilisation tasks of the acrobot and pendubot, tailored specifcially to the updated guidelines of the 3rd AI Olympics at ICRA 2025. Building upon our previously developed Average-Reward Entropy Advantage Policy Optimization (AR-EAPO) algorithm, we refined our solution to effectively address the new competition scenarios and evaluation metrics. Extensive simulations validate that our controller robustly manages these revised tasks, demonstrating adaptability and effectiveness within the updated framework. 

**Abstract (ZH)**: 基于强化学习的Acrobot和Pendubot摆动-up和稳定任务解决方案——面向ICRA 2025 AI Olympics更新指南 

---
# GelFusion: Enhancing Robotic Manipulation under Visual Constraints via Visuotactile Fusion 

**Title (ZH)**: 视觉接触感知融合增强机器人在视觉约束下的操作 Manipulation under Visual Constraints via Visuotactile Fusion 

**Authors**: Shulong Jiang, Shiqi Zhao, Yuxuan Fan, Peng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07455)  

**Abstract**: Visuotactile sensing offers rich contact information that can help mitigate performance bottlenecks in imitation learning, particularly under vision-limited conditions, such as ambiguous visual cues or occlusions. Effectively fusing visual and visuotactile modalities, however, presents ongoing challenges. We introduce GelFusion, a framework designed to enhance policies by integrating visuotactile feedback, specifically from high-resolution GelSight sensors. GelFusion using a vision-dominated cross-attention fusion mechanism incorporates visuotactile information into policy learning. To better provide rich contact information, the framework's core component is our dual-channel visuotactile feature representation, simultaneously leveraging both texture-geometric and dynamic interaction features. We evaluated GelFusion on three contact-rich tasks: surface wiping, peg insertion, and fragile object pick-and-place. Outperforming baselines, GelFusion shows the value of its structure in improving the success rate of policy learning. 

**Abstract (ZH)**: 视觉触觉感知提供了丰富的接触信息，可以在视觉受限条件下，如含糊的视觉提示或遮挡情况下，帮助缓解模仿学习中的性能瓶颈。然而，有效地融合视觉与触觉模态仍然存在挑战。我们提出了GelFusion框架，旨在通过整合高分辨率GelSight传感器的触觉反馈来增强策略。GelFusion采用以视觉为主导的交叉注意力融合机制，将触觉信息融入到策略学习中。框架的核心组件是我们的双通道触觉特征表示，同时利用纹理几何和动态交互特征以提供丰富的接触信息。我们在三项接触丰富的任务（表面擦拭、 Peg插入和易碎物体拾放）上评估了GelFusion，结果显示其结构在提高策略学习成功率方面的价值。 

---
# TPT-Bench: A Large-Scale, Long-Term and Robot-Egocentric Dataset for Benchmarking Target Person Tracking 

**Title (ZH)**: TPT-Bench：一个用于目标人员跟踪基准测试的大规模、长期和机器人本体中心数据集 

**Authors**: Hanjing Ye, Yu Zhan, Weixi Situ, Guangcheng Chen, Jingwen Yu, Kuanqi Cai, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07446)  

**Abstract**: Tracking a target person from robot-egocentric views is crucial for developing autonomous robots that provide continuous personalized assistance or collaboration in Human-Robot Interaction (HRI) and Embodied AI. However, most existing target person tracking (TPT) benchmarks are limited to controlled laboratory environments with few distractions, clean backgrounds, and short-term occlusions. In this paper, we introduce a large-scale dataset designed for TPT in crowded and unstructured environments, demonstrated through a robot-person following task. The dataset is collected by a human pushing a sensor-equipped cart while following a target person, capturing human-like following behavior and emphasizing long-term tracking challenges, including frequent occlusions and the need for re-identification from numerous pedestrians. It includes multi-modal data streams, including odometry, 3D LiDAR, IMU, panoptic, and RGB-D images, along with exhaustively annotated 2D bounding boxes of the target person across 35 sequences, both indoors and outdoors. Using this dataset and visual annotations, we perform extensive experiments with existing TPT methods, offering a thorough analysis of their limitations and suggesting future research directions. 

**Abstract (ZH)**: 面向拥挤和非结构化环境的目标人员跟踪数据集对于开发在人机交互（HRI）和具身AI中提供连续个性化协助或合作的自主机器人至关重要。然而，现有的大多数目标人员跟踪（TPT）基准仅限于少量干扰、干净背景和短暂遮挡的受控实验室环境。在本文中，我们介绍了一个大规模数据集，该数据集旨在用于拥挤和非结构化环境下的目标人员跟踪，并通过机器人追踪任务进行了展示。该数据集通过一名人类推动装有传感器的手推车并追随目标人员收集，捕捉类似人类的跟随行为，强调长期跟踪挑战，包括频繁遮挡和在众多行人的背景下重新识别的需求。该数据集包括多模态数据流，包括里程计、3D激光雷达、IMU、全景以及RGB-D图像，并且包含35个序列（室内和室外）中目标人员的详尽注释2D边界框。利用该数据集和视觉注释，我们对现有的TPT方法进行了广泛实验，提供了对其局限性的全面分析，并提出了未来研究方向。 

---
# ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning 

**Title (ZH)**: ReinboT: 用强化学习增强机器人视觉-语言操作 

**Authors**: Hongyin Zhang, Zifeng Zhuang, Han Zhao, Pengxiang Ding, Hongchao Lu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07395)  

**Abstract**: Vision-Language-Action (VLA) models have shown great potential in general robotic decision-making tasks via imitation learning. However, the variable quality of training data often constrains the performance of these models. On the other hand, offline Reinforcement Learning (RL) excels at learning robust policy models from mixed-quality data. In this paper, we introduce Reinforced robot GPT (ReinboT), a novel end-to-end VLA model that integrates the RL principle of maximizing cumulative reward. ReinboT achieves a deeper understanding of the data quality distribution by predicting dense returns that capture the nuances of manipulation tasks. The dense return prediction capability enables the robot to generate more robust decision-making actions, oriented towards maximizing future benefits. Extensive experiments show that ReinboT achieves state-of-the-art performance on the CALVIN mixed-quality dataset and exhibits superior few-shot learning and out-of-distribution generalization capabilities in real-world tasks. 

**Abstract (ZH)**: 基于强化学习的视觉-语言-动作模型ReinboT：一种通过预测密集回报实现高效鲁棒决策的新范式 

---
# Drive Fast, Learn Faster: On-Board RL for High Performance Autonomous Racing 

**Title (ZH)**: 快速驾驶，更快学习：车载强化学习在高性能自动驾驶赛车中的应用 

**Authors**: Benedict Hildisch, Edoardo Ghignone, Nicolas Baumann, Cheng Hu, Andrea Carron, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2505.07321)  

**Abstract**: Autonomous racing presents unique challenges due to its non-linear dynamics, the high speed involved, and the critical need for real-time decision-making under dynamic and unpredictable conditions. Most traditional Reinforcement Learning (RL) approaches rely on extensive simulation-based pre-training, which faces crucial challenges in transfer effectively to real-world environments. This paper introduces a robust on-board RL framework for autonomous racing, designed to eliminate the dependency on simulation-based pre-training enabling direct real-world adaptation. The proposed system introduces a refined Soft Actor-Critic (SAC) algorithm, leveraging a residual RL structure to enhance classical controllers in real-time by integrating multi-step Temporal-Difference (TD) learning, an asynchronous training pipeline, and Heuristic Delayed Reward Adjustment (HDRA) to improve sample efficiency and training stability. The framework is validated through extensive experiments on the F1TENTH racing platform, where the residual RL controller consistently outperforms the baseline controllers and achieves up to an 11.5 % reduction in lap times compared to the State-of-the-Art (SotA) with only 20 min of training. Additionally, an End-to-End (E2E) RL controller trained without a baseline controller surpasses the previous best results with sustained on-track learning. These findings position the framework as a robust solution for high-performance autonomous racing and a promising direction for other real-time, dynamic autonomous systems. 

**Abstract (ZH)**: 自主赛车比赛由于其非线性动力学、高速度以及在动态和不可预测条件下对实时决策的高需求，面临独特的挑战。大多数传统的强化学习（RL）方法依赖于基于仿真的预训练，但在将仿真实验结果有效转移到真实环境方面存在重大挑战。本文介绍了一种鲁棒的车载RL框架，旨在消除对基于仿真的预训练的依赖，使系统能够直接适应真实世界环境。该系统引入了一种改进的Soft Actor-Critic（SAC）算法，利用残差RL结构，通过集成多步时差（TD）学习、异步训练管道和启发式延迟奖励调整（HDRA），增强了实时控制性能，提高了样本效率并增强了训练稳定性。该框架通过在F1TENTH赛车平台上进行广泛实验得到验证，实验证明残差RL控制器在性能上持续优于基线控制器，并且仅通过20分钟的训练即可达到比当前最佳技术水平（SotA）快11.5%的圈速。此外，无需基线控制器的端到端（E2E）RL控制器在持续赛道学习中取得了超越以往最佳结果的表现。这些发现将该框架定位为高性能自主赛车的强大解决方案，并为其他实时动态自主系统提供了有前景的方向。 

---
# HuB: Learning Extreme Humanoid Balance 

**Title (ZH)**: HuB: 学习极端人形机器人平衡 

**Authors**: Tong Zhang, Boyuan Zheng, Ruiqian Nai, Yingdong Hu, Yen-Jen Wang, Geng Chen, Fanqi Lin, Jiongye Li, Chuye Hong, Koushil Sreenath, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07294)  

**Abstract**: The human body demonstrates exceptional motor capabilities-such as standing steadily on one foot or performing a high kick with the leg raised over 1.5 meters-both requiring precise balance control. While recent research on humanoid control has leveraged reinforcement learning to track human motions for skill acquisition, applying this paradigm to balance-intensive tasks remains challenging. In this work, we identify three key obstacles: instability from reference motion errors, learning difficulties due to morphological mismatch, and the sim-to-real gap caused by sensor noise and unmodeled dynamics. To address these challenges, we propose HuB (Humanoid Balance), a unified framework that integrates reference motion refinement, balance-aware policy learning, and sim-to-real robustness training, with each component targeting a specific challenge. We validate our approach on the Unitree G1 humanoid robot across challenging quasi-static balance tasks, including extreme single-legged poses such as Swallow Balance and Bruce Lee's Kick. Our policy remains stable even under strong physical disturbances-such as a forceful soccer strike-while baseline methods consistently fail to complete these tasks. Project website: this https URL 

**Abstract (ZH)**: 人类身体展示了卓越的运动能力——如单脚站立或抬腿超过1.5米的高踢动作，都要求精确的平衡控制。尽管最近关于类人控制器的研究利用强化学习来跟踪人体运动以获取技能，但在平衡密集型任务上应用这一范式仍存在挑战。在本文中，我们确定了三个关键障碍：参考运动错误引起的不稳定性、由于形态差异导致的学习困难，以及由传感器噪声和未建模动力学引起的仿真实验到真实环境的差距。为了应对这些挑战，我们提出了一种统一框架HuB（类人平衡），该框架融合了参考运动精细化、平衡感知策略学习和仿真实验到真实环境的鲁棒性训练，每个组件针对特定挑战。我们通过在Unitree G1类人机器人上执行具有挑战性的准静态平衡任务来验证我们的方法，包括极端的单腿姿态如燕子平衡和李小龙踢腿。即使在强烈的物理干扰下（如足球射门），我们的策略仍保持稳定，而基线方法则无法完成这些任务。项目网站：this https URL。 

---
# BETTY Dataset: A Multi-modal Dataset for Full-Stack Autonomy 

**Title (ZH)**: 贝蒂数据集：全栈自主性的多模态数据集 

**Authors**: Micah Nye, Ayoub Raji, Andrew Saba, Eidan Erlich, Robert Exley, Aragya Goyal, Alexander Matros, Ritesh Misra, Matthew Sivaprakasam, Marko Bertogna, Deva Ramanan, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07266)  

**Abstract**: We present the BETTY dataset, a large-scale, multi-modal dataset collected on several autonomous racing vehicles, targeting supervised and self-supervised state estimation, dynamics modeling, motion forecasting, perception, and more. Existing large-scale datasets, especially autonomous vehicle datasets, focus primarily on supervised perception, planning, and motion forecasting tasks. Our work enables multi-modal, data-driven methods by including all sensor inputs and the outputs from the software stack, along with semantic metadata and ground truth information. The dataset encompasses 4 years of data, currently comprising over 13 hours and 32TB, collected on autonomous racing vehicle platforms. This data spans 6 diverse racing environments, including high-speed oval courses, for single and multi-agent algorithm evaluation in feature-sparse scenarios, as well as high-speed road courses with high longitudinal and lateral accelerations and tight, GPS-denied environments. It captures highly dynamic states, such as 63 m/s crashes, loss of tire traction, and operation at the limit of stability. By offering a large breadth of cross-modal and dynamic data, the BETTY dataset enables the training and testing of full autonomy stack pipelines, pushing the performance of all algorithms to the limits. The current dataset is available at this https URL. 

**Abstract (ZH)**: BETTY数据集：一种多模态的自主赛车大数据集，用于监督和自监督状态估计、动力学建模、运动预测、感知等任务 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Title (ZH)**: UAV-CodeAgents: 通过多智能体ReAct和视觉-语言推理实现可扩展的无人机任务规划 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

**Abstract (ZH)**: UAV-CodeAgents：基于大型语言和多模态模型的可扩展多Agent自主无人机任务生成框架 

---
# X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real 

**Title (ZH)**: X-Sim：通过实境到仿真再到实境的跨体态学习 

**Authors**: Prithwish Dan, Kushal Kedia, Angela Chao, Edward Weiyi Duan, Maximus Adrian Pace, Wei-Chiu Ma, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.07096)  

**Abstract**: Human videos offer a scalable way to train robot manipulation policies, but lack the action labels needed by standard imitation learning algorithms. Existing cross-embodiment approaches try to map human motion to robot actions, but often fail when the embodiments differ significantly. We propose X-Sim, a real-to-sim-to-real framework that uses object motion as a dense and transferable signal for learning robot policies. X-Sim starts by reconstructing a photorealistic simulation from an RGBD human video and tracking object trajectories to define object-centric rewards. These rewards are used to train a reinforcement learning (RL) policy in simulation. The learned policy is then distilled into an image-conditioned diffusion policy using synthetic rollouts rendered with varied viewpoints and lighting. To transfer to the real world, X-Si introduces an online domain adaptation technique that aligns real and simulated observations during deployment. Importantly, X-Sim does not require any robot teleoperation data. We evaluate it across 5 manipulation tasks in 2 environments and show that it: (1) improves task progress by 30% on average over hand-tracking and sim-to-real baselines, (2) matches behavior cloning with 10x less data collection time, and (3) generalizes to new camera viewpoints and test-time changes. Code and videos are available at this https URL. 

**Abstract (ZH)**: 人类视频为机器人操作策略训练提供了可扩展的方法，但缺乏标准imitation learning算法所需的动作标签。现有的跨体态方法尝试将人类动作映射到机器人动作，但在体态差异显著时往往失败。我们提出X-Sim，一种从真实到模拟再到真实的世界框架，使用物体运动作为密集且可转移的信号来学习机器人策略。X-Sim 首先从RGBD人类视频中重构逼真的模拟，并跟踪物体轨迹以定义以物体为中心的奖励。这些奖励用于在模拟中训练强化学习（RL）策略。学习到的策略随后通过多样视角和光照渲染合成轨迹提炼为条件图像扩散策略。为了转移到真实世界，X-Si 引入了一种在线领域适应技术，在部署过程中对真实和模拟观察进行对齐。重要的是，X-Sim 不需要任何机器人远程操作数据。我们在两个环境中对5个操作任务进行了评估，并展示了它：(1) 平均在手部追踪和sim-to-real基线方法上提高任务进度30%，(2) 用1/10的数据收集时间匹配行为克隆，(3) 能够适应新的相机视角和测试时的变化。代码和视频可在以下链接获取。 

---
# YOPOv2-Tracker: An End-to-End Agile Tracking and Navigation Framework from Perception to Action 

**Title (ZH)**: YOPOv2-Tracker：从感知到行动的端到端敏捷跟踪与导航框架 

**Authors**: Junjie Lu, Yulin Hui, Xuewei Zhang, Wencan Feng, Hongming Shen, Zhiyu Li, Bailing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.06923)  

**Abstract**: Traditional target tracking pipelines including detection, mapping, navigation, and control are comprehensive but introduce high latency, limitting the agility of quadrotors. On the contrary, we follow the design principle of "less is more", striving to simplify the process while maintaining effectiveness. In this work, we propose an end-to-end agile tracking and navigation framework for quadrotors that directly maps the sensory observations to control commands. Importantly, leveraging the multimodal nature of navigation and detection tasks, our network maintains interpretability by explicitly integrating the independent modules of the traditional pipeline, rather than a crude action regression. In detail, we adopt a set of motion primitives as anchors to cover the searching space regarding the feasible region and potential target. Then we reformulate the trajectory optimization as regression of primitive offsets and associated costs considering the safety, smoothness, and other metrics. For tracking task, the trajectories are expected to approach the target and additional objectness scores are predicted. Subsequently, the predictions, after compensation for the estimated lumped disturbance, are transformed into thrust and attitude as control commands for swift response. During training, we seamlessly integrate traditional motion planning with deep learning by directly back-propagating the gradients of trajectory costs to the network, eliminating the need for expert demonstration in imitation learning and providing more direct guidance than reinforcement learning. Finally, we deploy the algorithm on a compact quadrotor and conduct real-world validations in both forest and building environments to demonstrate the efficiency of the proposed method. 

**Abstract (ZH)**: 一种基于端到端优化的四旋翼敏捷跟踪与导航框架 

---
# FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots 

**Title (ZH)**: FACET：基于阻抗参考跟踪的力自适应控制方法应用于腿式机器人 

**Authors**: Botian Xu, Haoyang Weng, Qingzhou Lu, Yang Gao, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06883)  

**Abstract**: Reinforcement learning (RL) has made significant strides in legged robot control, enabling locomotion across diverse terrains and complex loco-manipulation capabilities. However, the commonly used position or velocity tracking-based objectives are agnostic to forces experienced by the robot, leading to stiff and potentially dangerous behaviors and poor control during forceful interactions. To address this limitation, we present \emph{Force-Adaptive Control via Impedance Reference Tracking} (FACET). Inspired by impedance control, we use RL to train a control policy to imitate a virtual mass-spring-damper system, allowing fine-grained control under external forces by manipulating the virtual spring. In simulation, we demonstrate that our quadruped robot achieves improved robustness to large impulses (up to 200 Ns) and exhibits controllable compliance, achieving an 80% reduction in collision impulse. The policy is deployed to a physical robot to showcase both compliance and the ability to engage with large forces by kinesthetic control and pulling payloads up to 2/3 of its weight. Further extension to a legged loco-manipulator and a humanoid shows the applicability of our method to more complex settings to enable whole-body compliance control. Project Website: this https URL 

**Abstract (ZH)**: 基于阻抗参考跟踪的力自适应控制（Force-Adaptive Control via Impedance Reference Tracking） 

---
# Efficient Robotic Policy Learning via Latent Space Backward Planning 

**Title (ZH)**: 通过潜在空间逆向规划实现高效的机器人政策学习 

**Authors**: Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06861)  

**Abstract**: Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: this https URL 

**Abstract (ZH)**: 当前的机器人规划方法往往依赖于预测多帧具有全像素细节的图像。虽然这种精细的方法可以作为通用的世界模型，但对于下游策略学习来说，它引入了两个重大挑战：巨大的计算成本阻碍了实时部署，以及累积的不准确数据可能会误导动作提取。使用粗粒度的子目标进行规划部分缓解了效率问题。然而，其前瞻性的规划方案仍然可能由于累积误差导致离任务目标的预测，从而与长期目标产生偏差。这提出了一个关键问题：机器人规划能否在长时间多阶段任务中既高效又足够准确以实现实时控制？为了解决这个问题，我们提出了一种潜空间反向规划方案（LBP），该方案首先将任务细化为最终的潜目标，然后通过递归预测与当前状态更接近的中间子目标。最终目标的先验性使得反向子目标规划总是能够保持对任务完成的意识，促进在整个规划时段内的任务相关预测。子目标条件策略结合了一个可学习的标记来总结子目标序列，并决定每个子目标如何引导动作提取。通过大量的模拟实验和长时间的真实机器人实验，我们展示了LBP在性能上优于现有的精细和前瞻规划方法，达到了目前最佳的性能。项目页面: [this URL](this https URL) 

---
# UniDiffGrasp: A Unified Framework Integrating VLM Reasoning and VLM-Guided Part Diffusion for Open-Vocabulary Constrained Grasping with Dual Arms 

**Title (ZH)**: UniDiffGrasp: 一种集成VLM推理和VLM指导的部件扩散的统一框架，用于双臂开放词汇约束抓取 

**Authors**: Xueyang Guo, Hongwei Hu, Chengye Song, Jiale Chen, Zilin Zhao, Yu Fu, Bowen Guan, Zhenze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06832)  

**Abstract**: Open-vocabulary, task-oriented grasping of specific functional parts, particularly with dual arms, remains a key challenge, as current Vision-Language Models (VLMs), while enhancing task understanding, often struggle with precise grasp generation within defined constraints and effective dual-arm coordination. We innovatively propose UniDiffGrasp, a unified framework integrating VLM reasoning with guided part diffusion to address these limitations. UniDiffGrasp leverages a VLM to interpret user input and identify semantic targets (object, part(s), mode), which are then grounded via open-vocabulary segmentation. Critically, the identified parts directly provide geometric constraints for a Constrained Grasp Diffusion Field (CGDF) using its Part-Guided Diffusion, enabling efficient, high-quality 6-DoF grasps without retraining. For dual-arm tasks, UniDiffGrasp defines distinct target regions, applies part-guided diffusion per arm, and selects stable cooperative grasps. Through extensive real-world deployment, UniDiffGrasp achieves grasp success rates of 0.876 in single-arm and 0.767 in dual-arm scenarios, significantly surpassing existing state-of-the-art methods, demonstrating its capability to enable precise and coordinated open-vocabulary grasping in complex real-world scenarios. 

**Abstract (ZH)**: 开放式词汇、任务导向的特定功能性部件抓取，尤其是双臂抓取，仍然是一个关键挑战，现有的视觉-语言模型虽然提升了任务理解能力，但往往在精确抓取生成和双臂有效协调方面存在困难。我们创新地提出了UniDiffGrasp，这是一种结合视觉-语言模型推理与部分导向扩散的统一框架，以解决这些问题。UniDiffGrasp利用视觉-语言模型解读用户输入并识别语义目标（对象、部件、模式），并通过开放词汇分割进行语义grounding。关键的是，识别出的部件直接为受部件导向扩散约束的抓取扩散场（CGDF）提供几何约束，从而在无需重新训练的情况下实现高效、高质量的6-DoF抓取。对于双臂任务，UniDiffGrasp定义了不同的目标区域，每臂应用部分导向扩散，并选择稳定的协同抓取。通过广泛的实地部署，UniDiffGrasp在单臂场景中达到了0.876的抓取成功率，在双臂场景中达到了0.767，显著超越了现有的先进方法，展示了其在复杂现实场景中实现精确且协调的开放词汇抓取的能力。 

---
# Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson's Equation 

**Title (ZH)**: 复杂环境中的动态安全：基于泊松方程合成安全过滤器 

**Authors**: Gilbert Bahati, Ryan M. Bena, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2505.06794)  

**Abstract**: Synthesizing safe sets for robotic systems operating in complex and dynamically changing environments is a challenging problem. Solving this problem can enable the construction of safety filters that guarantee safe control actions -- most notably by employing Control Barrier Functions (CBFs). This paper presents an algorithm for generating safe sets from perception data by leveraging elliptic partial differential equations, specifically Poisson's equation. Given a local occupancy map, we solve Poisson's equation subject to Dirichlet boundary conditions, with a novel forcing function. Specifically, we design a smooth guidance vector field, which encodes gradient information required for safety. The result is a variational problem for which the unique minimizer -- a safety function -- characterizes the safe set. After establishing our theoretical result, we illustrate how safety functions can be used in CBF-based safety filtering. The real-time utility of our synthesis method is highlighted through hardware demonstrations on quadruped and humanoid robots navigating dynamically changing obstacle-filled environments. 

**Abstract (ZH)**: 在复杂且动态变化环境中合成安全集是机器人系统面临的一个具有挑战性的问题。解决这一问题可以使得通过控制障碍函数（CBFs）等方式构建保证安全控制动作的安全过滤器成为可能。本文提出了一种利用椭圆偏微分方程（尤其是泊松方程）从感知数据生成安全集的算法。给定局部占用地图，我们通过Dirichlet边界条件求解泊松方程，并设计了一个新型的激励函数。具体地，我们设计了一个光滑的引导向量场，编码了用于安全性的梯度信息。结果是一个变分问题，其唯一的最小值解——安全函数——定义了安全集。在建立我们的理论结果后，本文展示了如何使用安全函数来进行基于CBF的安全过滤。通过在具有动态变化障碍物环境中的 quadruped 和类人机器人上的硬件演示，突显了我们合成方法的实时实用性。 

---
# FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation 

**Title (ZH)**: FALCON: 学习自适应力 humanoid 操纵-developération 

**Authors**: Yuanhang Zhang, Yifu Yuan, Prajwal Gurunath, Tairan He, Shayegan Omidshafiei, Ali-akbar Agha-mohammadi, Marcell Vazquez-Chanlatte, Liam Pedersen, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06776)  

**Abstract**: Humanoid loco-manipulation holds transformative potential for daily service and industrial tasks, yet achieving precise, robust whole-body control with 3D end-effector force interaction remains a major challenge. Prior approaches are often limited to lightweight tasks or quadrupedal/wheeled platforms. To overcome these limitations, we propose FALCON, a dual-agent reinforcement-learning-based framework for robust force-adaptive humanoid loco-manipulation. FALCON decomposes whole-body control into two specialized agents: (1) a lower-body agent ensuring stable locomotion under external force disturbances, and (2) an upper-body agent precisely tracking end-effector positions with implicit adaptive force compensation. These two agents are jointly trained in simulation with a force curriculum that progressively escalates the magnitude of external force exerted on the end effector while respecting torque limits. Experiments demonstrate that, compared to the baselines, FALCON achieves 2x more accurate upper-body joint tracking, while maintaining robust locomotion under force disturbances and achieving faster training convergence. Moreover, FALCON enables policy training without embodiment-specific reward or curriculum tuning. Using the same training setup, we obtain policies that are deployed across multiple humanoids, enabling forceful loco-manipulation tasks such as transporting payloads (0-20N force), cart-pulling (0-100N), and door-opening (0-40N) in the real world. 

**Abstract (ZH)**: 人形机器人 manipulative 运动在日常服务和工业任务中具有变革性的潜力，但实现精确且健壮的三维末端执行器力交互全程体控制仍然是一个重大挑战。先前的方法通常局限于轻量级任务或四足或轮式平台。为克服这些限制，我们提出了 FALCON，一个基于双代理强化学习的人形机器人健壮力自适应 manipulative 运动框架。FALCON 将全程体控制分解为两个专门的代理：（1）下肢代理，确保在外部力干扰下稳定行走；（2）上肢代理，精确跟踪末端执行器位置并隐式进行自适应力补偿。这两个代理在模拟中联合训练，通过渐进增强作用于末端执行器的外部力的大小来构建力课程，同时遵守扭矩限制。实验表明，与基线方法相比，FALCON 在保持稳健行走能力的同时，上肢关节追踪更精确，训练收敛速度更快。此外，FALCON 允许在无需针对具体实体的奖励或课程进行调整的情况下训练策略。使用相同的训练设置，我们获得了可在多个机器人上部署的策略，实现了诸如搬运载荷（0-20N 力）、拉小车（0-100N）和开门（0-40N）等真实的力交互 maniuplation 任务。 

---
# JaxRobotarium: Training and Deploying Multi-Robot Policies in 10 Minutes 

**Title (ZH)**: JaxRobotarium：十分钟内训练和部署多机器人策略 

**Authors**: Shalin Anand Jain, Jiazhen Liu, Siva Kailas, Harish Ravichandar  

**Link**: [PDF](https://arxiv.org/pdf/2505.06771)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a promising solution for learning complex and scalable coordination behaviors in multi-robot systems. However, established MARL platforms (e.g., SMAC and MPE) lack robotics relevance and hardware deployment, leaving multi-robot learning researchers to develop bespoke environments and hardware testbeds dedicated to the development and evaluation of their individual contributions. The Multi-Agent RL Benchmark and Learning Environment for the Robotarium (MARBLER) is an exciting recent step in providing a standardized robotics-relevant platform for MARL, by bridging the Robotarium testbed with existing MARL software infrastructure. However, MARBLER lacks support for parallelization and GPU/TPU execution, making the platform prohibitively slow compared to modern MARL environments and hindering adoption. We contribute JaxRobotarium, a Jax-powered end-to-end simulation, learning, deployment, and benchmarking platform for the Robotarium. JaxRobotarium enables rapid training and deployment of multi-robot reinforcement learning (MRRL) policies with realistic robot dynamics and safety constraints, supporting both parallelization and hardware acceleration. Our generalizable learning interface provides an easy-to-use integration with SOTA MARL libraries (e.g., JaxMARL). In addition, JaxRobotarium includes eight standardized coordination scenarios, including four novel scenarios that bring established MARL benchmark tasks (e.g., RWARE and Level-Based Foraging) to a realistic robotics setting. We demonstrate that JaxRobotarium retains high simulation fidelity while achieving dramatic speedups over baseline (20x in training and 150x in simulation), and provides an open-access sim-to-real evaluation pipeline through the Robotarium testbed, accelerating and democratizing access to multi-robot learning research and evaluation. 

**Abstract (ZH)**: 基于Jax的Robotarium多智能体强化学习平台：加速和普及多机器人学习研究 

---
# Learned IMU Bias Prediction for Invariant Visual Inertial Odometry 

**Title (ZH)**: 基于学习的IMU偏置预测用于不变视觉惯性里程计 

**Authors**: Abdullah Altawaitan, Jason Stanley, Sambaran Ghosal, Thai Duong, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2505.06748)  

**Abstract**: Autonomous mobile robots operating in novel environments depend critically on accurate state estimation, often utilizing visual and inertial measurements. Recent work has shown that an invariant formulation of the extended Kalman filter improves the convergence and robustness of visual-inertial odometry by utilizing the Lie group structure of a robot's position, velocity, and orientation states. However, inertial sensors also require measurement bias estimation, yet introducing the bias in the filter state breaks the Lie group symmetry. In this paper, we design a neural network to predict the bias of an inertial measurement unit (IMU) from a sequence of previous IMU measurements. This allows us to use an invariant filter for visual inertial odometry, relying on the learned bias prediction rather than introducing the bias in the filter state. We demonstrate that an invariant multi-state constraint Kalman filter (MSCKF) with learned bias predictions achieves robust visual-inertial odometry in real experiments, even when visual information is unavailable for extended periods and the system needs to rely solely on IMU measurements. 

**Abstract (ZH)**: 自主移动机器人在新型环境中的操作高度依赖于准确的状态估计，通常利用视觉和惯性测量。最近的研究表明，扩展卡尔曼滤波器的一种不变形式通过利用机器人位置、速度和姿态状态的李群结构，可以改善视觉-惯性里程计的收敛性和鲁棒性。然而，惯性传感器还需要进行偏置估计，而在滤波器状态中引入偏置会破坏李群对称性。在本文中，我们设计了一个神经网络来从惯性测量单元(IMU)的先前测量序列中预测偏置。这使我们能够使用不变滤波器进行视觉-惯性里程计，依靠学习到的偏置预测，而不是在滤波器状态中引入偏置。我们展示了在实际实验中，基于学习偏置预测的不变多状态约束Kalman滤波器(MSCKF)可以在长时间缺乏视觉信息且系统仅依赖IMU测量的情况下实现稳健的视觉-惯性里程计。 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Title (ZH)**: 平衡进展与安全：自动驾驶中强化学习的新颖风险感知目标 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

**Abstract (ZH)**: 强化学习（RL）在实现自主驾驶中的应用由于其强大的决策能力而充满 promise，并通过在交通场景中基于奖励函数的试错学习驾驶策略。然而，奖励函数的设计未能得到充分关注，导致定义模糊且存在诸多问题。特别是在安全性方面，长期仅将安全视为碰撞的惩罚，忽视了导致碰撞之前的行为风险，限制了 RL 在实际场景中的应用。为解决这些问题，我们的工作集中在增强奖励的制定，通过定义一系列驾驶目标并将其分层结构化来改善奖励公式。此外，我们以规范化的方式讨论这些目标的制定，以透明地确定它们对总体奖励的贡献。我们还引入了一个基于椭圆函数和扩展责任敏感安全（RSS）概念的新颖风险感知目标，用于各种驾驶交互。我们在不同交通密度的无信号交叉口场景中评估了我们提出奖励的有效性，该方法平均将碰撞率降低了 21%，并且在路线进展和累计奖励方面持续超过基线奖励，证明了其在促进更安全驾驶行为的同时保持高性能的能力。 

---
# STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation 

**Title (ZH)**: STRIVE: 结构化表示集成VLM推理以实现高效物体导航 

**Authors**: Haokun Zhu, Zongtai Li, Zhixuan Liu, Wenshan Wang, Ji Zhang, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.06729)  

**Abstract**: Vision-Language Models (VLMs) have been increasingly integrated into object navigation tasks for their rich prior knowledge and strong reasoning abilities. However, applying VLMs to navigation poses two key challenges: effectively representing complex environment information and determining \textit{when and how} to query VLMs. Insufficient environment understanding and over-reliance on VLMs (e.g. querying at every step) can lead to unnecessary backtracking and reduced navigation efficiency, especially in continuous environments. To address these challenges, we propose a novel framework that constructs a multi-layer representation of the environment during navigation. This representation consists of viewpoint, object nodes, and room nodes. Viewpoints and object nodes facilitate intra-room exploration and accurate target localization, while room nodes support efficient inter-room planning. Building on this representation, we propose a novel two-stage navigation policy, integrating high-level planning guided by VLM reasoning with low-level VLM-assisted exploration to efficiently locate a goal object. We evaluated our approach on three simulated benchmarks (HM3D, RoboTHOR, and MP3D), and achieved state-of-the-art performance on both the success rate ($\mathord{\uparrow}\, 7.1\%$) and navigation efficiency ($\mathord{\uparrow}\, 12.5\%$). We further validate our method on a real robot platform, demonstrating strong robustness across 15 object navigation tasks in 10 different indoor environments. Project page is available at this https URL . 

**Abstract (ZH)**: 视觉语言模型（VLMs）已越来越多地集成到对象导航任务中，利用其丰富的先验知识和强大的推理能力。然而，将VLMs应用于导航面临着两个关键挑战：有效地表示复杂环境信息以及确定何时及如何查询VLMs。环境理解不足和过度依赖VLMs（例如，在每一步都查询）会导致不必要的回溯和导航效率降低，尤其是在连续环境中。为了解决这些挑战，我们提出了一种新型框架，在导航过程中构建环境的多层表示。该表示由视点、物体节点和房间节点组成。视点和物体节点促进了房间内的探索和精确的目标定位，而房间节点支持高效的跨房间规划。基于这种表示，我们提出了一种新的两阶段导航策略，整合了由VLM推理引导的高层次规划和VLM辅助的低层次探索，以高效地定位目标物体。我们在三个模拟基准（HM3D、RoboTHOR和MP3D）上评估了我们的方法，并在成功率（提高7.1%）和导航效率（提高12.5%）上达到了最先进的性能。我们在真实的机器人平台上进一步验证了该方法，展示了在10种不同的室内环境中执行15种物体导航任务的强大鲁棒性。项目页面请访问此链接。 

---
# 3D Characterization of Smoke Plume Dispersion Using Multi-View Drone Swarm 

**Title (ZH)**: 多视点无人机群用于烟雾羽流扩散的三维表征 

**Authors**: Nikil Krishnakumar, Shashank Sharma, Srijan Kumar Pal, Jiarong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06638)  

**Abstract**: This study presents an advanced multi-view drone swarm imaging system for the three-dimensional characterization of smoke plume dispersion dynamics. The system comprises a manager drone and four worker drones, each equipped with high-resolution cameras and precise GPS modules. The manager drone uses image feedback to autonomously detect and position itself above the plume, then commands the worker drones to orbit the area in a synchronized circular flight pattern, capturing multi-angle images. The camera poses of these images are first estimated, then the images are grouped in batches and processed using Neural Radiance Fields (NeRF) to generate high-resolution 3D reconstructions of plume dynamics over time. Field tests demonstrated the ability of the system to capture critical plume characteristics including volume dynamics, wind-driven directional shifts, and lofting behavior at a temporal resolution of about 1 s. The 3D reconstructions generated by this system provide unique field data for enhancing the predictive models of smoke plume dispersion and fire spread. Broadly, the drone swarm system offers a versatile platform for high resolution measurements of pollutant emissions and transport in wildfires, volcanic eruptions, prescribed burns, and industrial processes, ultimately supporting more effective fire control decisions and mitigating wildfire risks. 

**Abstract (ZH)**: 本研究提出了一种先进的多视角无人机群成像系统，用于三维表征烟羽分散动力学。系统包括一架管理者无人机和四架工人无人机，每架都装备有高分辨率相机和精确的GPS模块。管理者无人机通过图像反馈自主检测并定位在烟羽上方，然后指挥工人无人机以同步圆形飞行模式环绕区域飞行，捕获多角度图像。首先估计这些图像的相机姿态，然后将图像分批处理并使用神经辐射场（NeRF）生成烟羽动力学随时间演化的高分辨率三维重构。实地测试证明，该系统能够在大约1秒的时分辨率下捕获关键的烟羽特性，包括体积动态、风驱动的方向偏移以及抬升行为。由该系统生成的三维重构提供了独特的现场数据，以增强烟羽分散和火灾蔓延的预测模型。总体而言，无人机群系统为高分辨率测量野火、火山爆发、规定烧毁、工业过程中的污染物排放和传输提供了多功能平台，最终支持更有效的火灾控制决策并减轻野火风险。 

---
# ACORN: Adaptive Contrastive Optimization for Safe and Robust Fine-Grained Robotic Manipulation 

**Title (ZH)**: ACORN: 自适应对比优化以实现安全鲁棒的细粒度机器人操作 

**Authors**: Zhongquan Zhou, Shuhao Li, Zixian Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.06628)  

**Abstract**: Embodied AI research has traditionally emphasized performance metrics such as success rate and cumulative reward, overlooking critical robustness and safety considerations that emerge during real-world deployment. In actual environments, agents continuously encounter unpredicted situations and distribution shifts, causing seemingly reliable policies to experience catastrophic failures, particularly in manipulation tasks. To address this gap, we introduce four novel safety-centric metrics that quantify an agent's resilience to environmental perturbations. Building on these metrics, we present Adaptive Contrastive Optimization for Robust Manipulation (ACORN), a plug-and-play algorithm that enhances policy robustness without sacrificing performance. ACORN leverages contrastive learning to simultaneously align trajectories with expert demonstrations while diverging from potentially unsafe behaviors. Our approach efficiently generates informative negative samples through structured Gaussian noise injection, employing a double perturbation technique that maintains sample diversity while minimizing computational overhead. Comprehensive experiments across diverse manipulation environments validate ACORN's effectiveness, yielding improvements of up to 23% in safety metrics under disturbance compared to baseline methods. These findings underscore ACORN's significant potential for enabling reliable deployment of embodied agents in safety-critical real-world applications. 

**Abstract (ZH)**: 实体AI研究传统上强调成功 rate 和累计奖励等性能指标，忽视了实际部署中出现的关键稳健性和安全性考虑。在实际环境中，智能体不断遇到不可预测的情况和分布偏移，导致看似可靠的策略在操作任务中经历灾难性的失败。为解决这一问题，我们引入了四个新型的安全中心化指标，量化智能体对环境扰动的抗性。在这些指标的基础上，我们提出了适用于稳健操作的自适应对比优化算法（ACORN），该算法能够在不牺牲性能的情况下增强策略的稳健性。ACORN 利用对比学习同时使轨迹与专家演示对齐，同时避免潜在的不安全行为。通过结构化高斯噪声注入生成有效的负样本，采用双重扰动技术维持样本多样性同时减少计算开销。跨多种操作环境的全面实验验证了 ACORN 的有效性，在干扰条件下安全指标提高了高达 23%，基线方法表现出显著改进。这些发现突显了 ACORN 在安全关键的实际应用中可靠部署实体智能体方面的巨大潜力。 

---
# JAEGER: Dual-Level Humanoid Whole-Body Controller 

**Title (ZH)**: JAEGER: 双层次 humanoid 整体体动控制算法 

**Authors**: Ziluo Ding, Haobin Jiang, Yuxuan Wang, Zhenguo Sun, Yu Zhang, Xiaojie Niu, Ming Yang, Weishuai Zeng, Xinrun Xu, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06584)  

**Abstract**: This paper presents JAEGER, a dual-level whole-body controller for humanoid robots that addresses the challenges of training a more robust and versatile policy. Unlike traditional single-controller approaches, JAEGER separates the control of the upper and lower bodies into two independent controllers, so that they can better focus on their distinct tasks. This separation alleviates the dimensionality curse and improves fault tolerance. JAEGER supports both root velocity tracking (coarse-grained control) and local joint angle tracking (fine-grained control), enabling versatile and stable movements. To train the controller, we utilize a human motion dataset (AMASS), retargeting human poses to humanoid poses through an efficient retargeting network, and employ a curriculum learning approach. This method performs supervised learning for initialization, followed by reinforcement learning for further exploration. We conduct our experiments on two humanoid platforms and demonstrate the superiority of our approach against state-of-the-art methods in both simulation and real environments. 

**Abstract (ZH)**: JAEGER： humanoid机器人的双层次全身控制器及其训练方法 

---
# Quadrupedal Robot Skateboard Mounting via Reverse Curriculum Learning 

**Title (ZH)**: 基于逆序课程学习的四足机器人滑板搭载 

**Authors**: Danil Belov, Artem Erkhov, Elizaveta Pestova, Ilya Osokin, Dzmitry Tsetserukou, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06561)  

**Abstract**: The aim of this work is to enable quadrupedal robots to mount skateboards using Reverse Curriculum Reinforcement Learning. Although prior work has demonstrated skateboarding for quadrupeds that are already positioned on the board, the initial mounting phase still poses a significant challenge. A goal-oriented methodology was adopted, beginning with the terminal phases of the task and progressively increasing the complexity of the problem definition to approximate the desired objective. The learning process was initiated with the skateboard rigidly fixed within the global coordinate frame and the robot positioned directly above it. Through gradual relaxation of these initial conditions, the learned policy demonstrated robustness to variations in skateboard position and orientation, ultimately exhibiting a successful transfer to scenarios involving a mobile skateboard. The code, trained models, and reproducible examples are available at the following link: this https URL 

**Abstract (ZH)**: 本工作旨在利用反向 Curriculum 强化学习使四足机器人能够骑上滑板。尽管之前的工作已经展示了四足机器人在滑板上定位好之后进行滑板骑行，初始上板阶段仍然存在重大挑战。采用了一种以目标为导向的方法，从任务的最终阶段开始，逐步增加问题定义的复杂性，以逼近所需目标。学习过程从滑板刚性固定在全局坐标系中且机器人直接在其上方开始。通过逐渐放宽这些初始条件，学习到的策略显示了对滑板位置和方向变化的鲁棒性，并最终成功地转移到涉及移动滑板的场景中。相关代码、训练模型及可重复实验示例可在以下链接获取：this https URL 

---
# CompSLAM: Complementary Hierarchical Multi-Modal Localization and Mapping for Robot Autonomy in Underground Environments 

**Title (ZH)**: CompSLAM: 地下环境机器人自主性中互补分层多模态定位与地图构建 

**Authors**: Shehryar Khattak, Timon Homberger, Lukas Bernreiter, Julian Nubert, Olov Andersson, Roland Siegwart, Kostas Alexis, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.06483)  

**Abstract**: Robot autonomy in unknown, GPS-denied, and complex underground environments requires real-time, robust, and accurate onboard pose estimation and mapping for reliable operations. This becomes particularly challenging in perception-degraded subterranean conditions under harsh environmental factors, including darkness, dust, and geometrically self-similar structures. This paper details CompSLAM, a highly resilient and hierarchical multi-modal localization and mapping framework designed to address these challenges. Its flexible architecture achieves resilience through redundancy by leveraging the complementary nature of pose estimates derived from diverse sensor modalities. Developed during the DARPA Subterranean Challenge, CompSLAM was successfully deployed on all aerial, legged, and wheeled robots of Team Cerberus during their competition-winning final run. Furthermore, it has proven to be a reliable odometry and mapping solution in various subsequent projects, with extensions enabling multi-robot map sharing for marsupial robotic deployments and collaborative mapping. This paper also introduces a comprehensive dataset acquired by a manually teleoperated quadrupedal robot, covering a significant portion of the DARPA Subterranean Challenge finals course. This dataset evaluates CompSLAM's robustness to sensor degradations as the robot traverses 740 meters in an environment characterized by highly variable geometries and demanding lighting conditions. The CompSLAM code and the DARPA SubT Finals dataset are made publicly available for the benefit of the robotics community 

**Abstract (ZH)**: 地下未知、GPS受限和复杂环境中的机器人自主性要求具备实时、 robust 和精确的机载位姿估计和建图能力，以确保可靠运行。在黑暗、灰尘和几何相似结构等恶劣环境因素导致感知退化的地下条件下，这一要求变得尤为具有挑战性。本文详细介绍了CompSLAM，这是一种针对这些挑战设计的高度鲁棒性和分层多模态定位与建图框架。该框架通过利用来自多种传感模态的位姿估计的互补性，其灵活的架构实现了通过冗余性增强鲁棒性。CompSLAM在DARPA地下挑战赛期间开发，并成功部署在团队Cerberus的所有空中、 legged 和轮式机器人上，用于其竞赛获胜的最终运行。此外，其可靠性和建图解决方案在各种后续项目中得到了验证，并扩展了多机器人地图共享功能，以支持袋鼠机器人部署和协同建图。本文还介绍了由手动遥控四足机器人收集的全面数据集，涵盖了DARPA地下挑战赛决赛赛道的重要部分。数据集评估了机器人在具有高度变化几何结构和严苛光照条件的环境中穿越740米过程中CompSLAM对传感器退化的影响。CompSLAM代码和DARPA SubT决赛数据集已公开发布，以造福于机器人社区。 

---
# Autonomous Vision-Based Magnetic Microrobotic Pushing of Micro-Objects and Cells 

**Title (ZH)**: 基于自主视觉的磁微机器人对微物体和细胞的操控推移 

**Authors**: Max Sokolich, Ceren Kirmizitas, Sambeeta Das, Ron Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.06450)  

**Abstract**: Accurate and autonomous transportation of micro-objects and biological cells can enable significant advances in a wide variety of research disciplines. Here, we present a novel, vision-based, model-free microrobotic pushing algorithm for the autonomous manipulation of micro objects and biological cells. The algorithm adjusts the axis of a rotating magnetic field that in turn controls the heading angle and spin axis of a spherical Janus rolling microrobot. We introduce the concept of a microrobotic guiding corridor to constrain the object and to avoid pushing failures. We then show that employing only two simple conditions, the microrobot is able to successfully and autonomously push microscale objects along predefined trajectories. We evaluate the performance of the algorithm by measuring the mean absolute error and completion time relative to a desired path at different actuation frequencies and guiding corridor widths. Finally, we demonstrate biomedical applicability by autonomously transporting a single biological cell, highlighting the methods potential for applications in tissue engineering, drug delivery and synthetic biology. 

**Abstract (ZH)**: 基于视觉的无模型微机器人推动物算法可实现微小物体和生物细胞的精确自主运输，从而推动多个研究领域的重大进展。该算法通过调整旋转磁场的轴线来控制Janus滚珠微机器人的航向角和自旋轴，从而实现对微小物体和生物细胞的自主操作。我们提出了微机器人引导走廊的概念，以约束目标物体并避免推动物操作失败。我们证明，仅通过满足两个简单条件，微机器人就能够成功地沿着预定义的轨迹自主推动物体。通过在不同驱动频率和引导走廊宽度下测量平均绝对误差和完成时间来评估算法性能。最后，通过自主运输单个生物细胞，展示了其在生物医学应用中的潜力，包括组织工程、药物递送和合成生物学领域。 

---
# DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition 

**Title (ZH)**: DAPPER：基于查询高效性的可辨别性感知政策偏好强化学习技能获取方法 

**Authors**: Yuki Kadokawa, Jonas Frey, Takahiro Miki, Takamitsu Matsubara, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.06357)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions. 

**Abstract (ZH)**: 基于偏好增强学习的判别能力感知多策略高效学习（Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning, DAPPER） 

---
# Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach 

**Title (ZH)**: 基于模型的视频增强离线强化学习 

**Authors**: Minting Pan, Yitao Zheng, Jiajian Li, Yunbo Wang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06482)  

**Abstract**: Offline reinforcement learning (RL) enables policy optimization in static datasets, avoiding the risks and costs of real-world exploration. However, it struggles with suboptimal behavior learning and inaccurate value estimation due to the lack of environmental interaction. In this paper, we present Video-Enhanced Offline RL (VeoRL), a model-based approach that constructs an interactive world model from diverse, unlabeled video data readily available online. Leveraging model-based behavior guidance, VeoRL transfers commonsense knowledge of control policy and physical dynamics from natural videos to the RL agent within the target domain. Our method achieves substantial performance gains (exceeding 100% in some cases) across visuomotor control tasks in robotic manipulation, autonomous driving, and open-world video games. 

**Abstract (ZH)**: 视频增强的离线强化学习（VeoRL） 

---
# Emotion-Gradient Metacognitive RSI (Part I): Theoretical Foundations and Single-Agent Architecture 

**Title (ZH)**: 情感梯度元认知RSI（第一部分）：理论基础与单智能体架构 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.07757)  

**Abstract**: We present the Emotion-Gradient Metacognitive Recursive Self-Improvement (EG-MRSI) framework, a novel architecture that integrates introspective metacognition, emotion-based intrinsic motivation, and recursive self-modification into a unified theoretical system. The framework is explicitly capable of overwriting its own learning algorithm under formally bounded risk. Building upon the Noise-to-Meaning RSI (N2M-RSI) foundation, EG-MRSI introduces a differentiable intrinsic reward function driven by confidence, error, novelty, and cumulative success. This signal regulates both a metacognitive mapping and a self-modification operator constrained by provable safety mechanisms. We formally define the initial agent configuration, emotion-gradient dynamics, and RSI trigger conditions, and derive a reinforcement-compatible optimization objective that guides the agent's development trajectory. Meaning Density and Meaning Conversion Efficiency are introduced as quantifiable metrics of semantic learning, closing the gap between internal structure and predictive informativeness. This Part I paper establishes the single-agent theoretical foundations of EG-MRSI. Future parts will extend this framework to include safety certificates and rollback protocols (Part II), collective intelligence mechanisms (Part III), and feasibility constraints including thermodynamic and computational limits (Part IV). Together, the EG-MRSI series provides a rigorous, extensible foundation for open-ended and safe AGI. 

**Abstract (ZH)**: 基于情感梯度元认知递归自我改进的框架：EG-MRSI 

---
# Belief Injection for Epistemic Control in Linguistic State Space 

**Title (ZH)**: 信念注入实现知识控制的语言状态空间中 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.07693)  

**Abstract**: This work introduces belief injection, a proactive epistemic control mechanism for artificial agents whose cognitive states are structured as dynamic ensembles of linguistic belief fragments. Grounded in the Semantic Manifold framework, belief injection directly incorporates targeted linguistic beliefs into an agent's internal cognitive state, influencing reasoning and alignment proactively rather than reactively. We delineate various injection strategies, such as direct, context-aware, goal-oriented, and reflective approaches, and contrast belief injection with related epistemic control mechanisms, notably belief filtering. Additionally, this work discusses practical applications, implementation considerations, ethical implications, and outlines promising directions for future research into cognitive governance using architecturally embedded belief injection. 

**Abstract (ZH)**: This work introduces belief injection, 一种基于语义流形框架的主动主义态控制机制，适用于其认知状态由动态语言信念片段组成的 artificial agents。通过直接将目标语言信念注入代理的内部认知状态，信念注入主动影响推理和对齐，而非被动地响应。我们阐明了各种注入策略，如直接注入、语境感知注入、目标导向注入和反思性方法，并将信念注入与相关主义态控制机制，特别是信念过滤进行对比。此外，本文讨论了实际应用、实现考虑、伦理影响，并概述了在认知治理中嵌入信念注入的有前途的研究方向。 

---
# RefPentester: A Knowledge-Informed Self-Reflective Penetration Testing Framework Based on Large Language Models 

**Title (ZH)**: RefPentester：一种基于大型语言模型的知识导向型自省式渗透测试框架 

**Authors**: Hanzheng Dai, Yuanliang Li, Zhibo Zhang, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07089)  

**Abstract**: Automated penetration testing (AutoPT) powered by large language models (LLMs) has gained attention for its ability to automate ethical hacking processes and identify vulnerabilities in target systems by leveraging the intrinsic knowledge of LLMs. However, existing LLM-based AutoPT frameworks often underperform compared to human experts in challenging tasks for several reasons: the imbalanced knowledge used in LLM training, short-sighted planning in the planning process, and hallucinations during command generation. In addition, the penetration testing (PT) process, with its trial-and-error nature, is limited by existing frameworks that lack mechanisms to learn from previous failed operations, restricting adaptive improvement of PT strategies. To address these limitations, we propose a knowledge-informed self-reflective PT framework powered by LLMs, called RefPentester, which is an AutoPT framework designed to assist human operators in identifying the current stage of the PT process, selecting appropriate tactic and technique for the stage, choosing suggested action, providing step-by-step operational guidance, and learning from previous failed operations. We also modeled the PT process as a seven-state Stage Machine to integrate the proposed framework effectively. The evaluation shows that RefPentester can successfully reveal credentials on Hack The Box's Sau machine, outperforming the baseline GPT-4o model by 16.7\%. Across PT stages, RefPentester also demonstrates superior success rates on PT stage transitions. 

**Abstract (ZH)**: 基于大型语言模型的自适应渗透测试框架：RefPentester 

---
# Embodied Intelligence: The Key to Unblocking Generalized Artificial Intelligence 

**Title (ZH)**: 具身智能：通向通用人工智能的关键 

**Authors**: Jinhao Jiang, Changlin Chen, Shile Feng, Wanru Geng, Zesheng Zhou, Ni Wang, Shuai Li, Feng-Qi Cui, Erbao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06897)  

**Abstract**: The ultimate goal of artificial intelligence (AI) is to achieve Artificial General Intelligence (AGI). Embodied Artificial Intelligence (EAI), which involves intelligent systems with physical presence and real-time interaction with the environment, has emerged as a key research direction in pursuit of AGI. While advancements in deep learning, reinforcement learning, large-scale language models, and multimodal technologies have significantly contributed to the progress of EAI, most existing reviews focus on specific technologies or applications. A systematic overview, particularly one that explores the direct connection between EAI and AGI, remains scarce. This paper examines EAI as a foundational approach to AGI, systematically analyzing its four core modules: perception, intelligent decision-making, action, and feedback. We provide a detailed discussion of how each module contributes to the six core principles of AGI. Additionally, we discuss future trends, challenges, and research directions in EAI, emphasizing its potential as a cornerstone for AGI development. Our findings suggest that EAI's integration of dynamic learning and real-world interaction is essential for bridging the gap between narrow AI and AGI. 

**Abstract (ZH)**: 人工通用人工智能导向的具身人工智能：从感知到反馈的系统分析及其未来展望 

---
# Control Plane as a Tool: A Scalable Design Pattern for Agentic AI Systems 

**Title (ZH)**: 控制面作为工具：自主人工智能系统的大规模设计模式 

**Authors**: Sivasathivel Kandasamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.06817)  

**Abstract**: Agentic AI systems represent a new frontier in artificial intelligence, where agents often based on large language models(LLMs) interact with tools, environments, and other agents to accomplish tasks with a degree of autonomy. These systems show promise across a range of domains, but their architectural underpinnings remain immature. This paper conducts a comprehensive review of the types of agents, their modes of interaction with the environment, and the infrastructural and architectural challenges that emerge. We identify a gap in how these systems manage tool orchestration at scale and propose a reusable design abstraction: the "Control Plane as a Tool" pattern. This pattern allows developers to expose a single tool interface to an agent while encapsulating modular tool routing logic behind it. We position this pattern within the broader context of agent design and argue that it addresses several key challenges in scaling, safety, and extensibility. 

**Abstract (ZH)**: 基于大型语言模型的代理AI系统构成了人工智能的新前沿，这些代理通常与工具、环境和其他代理交互以实现具有一定程度自主性的任务。这些系统在多个领域显示出潜力，但其架构基础仍不够成熟。本文对代理的类型、它们与环境的交互模式以及随之而来的基础设施和架构挑战进行了全面评审。我们发现这些系统在大规模工具编排管理方面存在差距，并提出了一种可重复使用的设计抽象：“控制平面即工具”模式。该模式允许开发者为代理暴露单一工具接口，同时将模块化工具路由逻辑封装在其后。我们将该模式置于更广泛的代理设计背景下，并论证它在扩展、安全性和可扩展性方面解决了多个关键问题。 

---
# A Point-Based Algorithm for Distributional Reinforcement Learning in Partially Observable Domains 

**Title (ZH)**: 基于点的算法在部分可观测域中的分布强化学习 

**Authors**: Larry Preuett III  

**Link**: [PDF](https://arxiv.org/pdf/2505.06518)  

**Abstract**: In many real-world planning tasks, agents must tackle uncertainty about the environment's state and variability in the outcomes of any chosen policy. We address both forms of uncertainty as a first step toward safer algorithms in partially observable settings. Specifically, we extend Distributional Reinforcement Learning (DistRL)-which models the entire return distribution for fully observable domains-to Partially Observable Markov Decision Processes (POMDPs), allowing an agent to learn the distribution of returns for each conditional plan. Concretely, we introduce new distributional Bellman operators for partial observability and prove their convergence under the supremum p-Wasserstein metric. We also propose a finite representation of these return distributions via psi-vectors, generalizing the classical alpha-vectors in POMDP solvers. Building on this, we develop Distributional Point-Based Value Iteration (DPBVI), which integrates psi-vectors into a standard point-based backup procedure-bridging DistRL and POMDP planning. By tracking return distributions, DPBVI naturally enables risk-sensitive control in domains where rare, high-impact events must be carefully managed. We provide source code to foster further research in robust decision-making under partial observability. 

**Abstract (ZH)**: 在部分可观测环境中处理不确定性形式的安全算法初步研究：将分布强化学习扩展到部分可观测马尔可夫决策过程 

---
# A Grounded Memory System For Smart Personal Assistants 

**Title (ZH)**: 基于内存的智能个人助理系统 

**Authors**: Felix Ocker, Jörg Deigmöller, Pavel Smirnov, Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2505.06328)  

**Abstract**: A wide variety of agentic AI applications - ranging from cognitive assistants for dementia patients to robotics - demand a robust memory system grounded in reality. In this paper, we propose such a memory system consisting of three components. First, we combine Vision Language Models for image captioning and entity disambiguation with Large Language Models for consistent information extraction during perception. Second, the extracted information is represented in a memory consisting of a knowledge graph enhanced by vector embeddings to efficiently manage relational information. Third, we combine semantic search and graph query generation for question answering via Retrieval Augmented Generation. We illustrate the system's working and potential using a real-world example. 

**Abstract (ZH)**: 多种代理型AI应用——从痴呆患者的认知辅助到机器人技术——需要一个基于现实的稳健记忆系统。本文提出了一种这样的记忆系统，由三个组件构成。首先，我们将视觉语言模型用于图像描述和实体消歧，与大型语言模型结合，实现感知过程中一致的信息提取。其次，提取的信息被表示在一个由向量嵌入增强的知识图谱中，以高效管理关系信息。第三，我们通过检索增强生成结合语义搜索和图查询生成来进行问答。我们通过一个实际例子展示该系统的运作及其潜力。 

---
# Internet of Agents: Fundamentals, Applications, and Challenges 

**Title (ZH)**: 代理互联网：基础、应用与挑战 

**Authors**: Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2505.07176)  

**Abstract**: With the rapid proliferation of large language models and vision-language models, AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the Internet of Agents (IoA) as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems. 

**Abstract (ZH)**: 随着大规模语言模型和视觉语言模型的迅速 proliferation，AI 代理从孤立的任务特定系统演变为无需人类干预即可感知、推理和行动的自主交互实体。随着这些代理在虚拟和物理环境中普及，从虚拟助手到具身机器人，构建统一的以代理为中心的基础架构变得至关重要。在本文综述中，我们介绍了代理互联网（IoA）作为一种基础框架，使大规模异构代理能够实现无缝互联、动态发现和协作编排。我们首先概述了通用的 IoA 架构，强调其分层组织、相对于传统互联网的独特特性以及新兴应用。接着，我们分析了 IoA 的关键操作使能技术，包括能力通知和发现、自适应通信协议、动态任务匹配、共识和冲突解决机制以及激励模型。最后，我们确定了朝着构建稳健且可信赖的 IoA 生态系统的研究方向。 

---
# ParaView-MCP: An Autonomous Visualization Agent with Direct Tool Use 

**Title (ZH)**: ParaView-MCP：一个具备直接工具使用的自主可视化代理 

**Authors**: Shusen Liu, Haichao Miao, Peer-Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07064)  

**Abstract**: While powerful and well-established, tools like ParaView present a steep learning curve that discourages many potential users. This work introduces ParaView-MCP, an autonomous agent that integrates modern multimodal large language models (MLLMs) with ParaView to not only lower the barrier to entry but also augment ParaView with intelligent decision support. By leveraging the state-of-the-art reasoning, command execution, and vision capabilities of MLLMs, ParaView-MCP enables users to interact with ParaView through natural language and visual inputs. Specifically, our system adopted the Model Context Protocol (MCP) - a standardized interface for model-application communication - that facilitates direct interaction between MLLMs with ParaView's Python API to allow seamless information exchange between the user, the language model, and the visualization tool itself. Furthermore, by implementing a visual feedback mechanism that allows the agent to observe the viewport, we unlock a range of new capabilities, including recreating visualizations from examples, closed-loop visualization parameter updates based on user-defined goals, and even cross-application collaboration involving multiple tools. Broadly, we believe such an agent-driven visualization paradigm can profoundly change the way we interact with visualization tools. We expect a significant uptake in the development of such visualization tools, in both visualization research and industry. 

**Abstract (ZH)**: ParaView-MCP：基于现代多模态大型语言模型的自主代理增强可视化工具 

---
# MAGE:A Multi-stage Avatar Generator with Sparse Observations 

**Title (ZH)**: MAGE：一种基于稀疏观测的多阶段_avatar生成器 

**Authors**: Fangyu Du, Yang Yang, Xuehao Gao, Hongye Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06411)  

**Abstract**: Inferring full-body poses from Head Mounted Devices, which capture only 3-joint observations from the head and wrists, is a challenging task with wide AR/VR applications. Previous attempts focus on learning one-stage motion mapping and thus suffer from an over-large inference space for unobserved body joint motions. This often leads to unsatisfactory lower-body predictions and poor temporal consistency, resulting in unrealistic or incoherent motion sequences. To address this, we propose a powerful Multi-stage Avatar GEnerator named MAGE that factorizes this one-stage direct motion mapping learning with a progressive prediction strategy. Specifically, given initial 3-joint motions, MAGE gradually inferring multi-scale body part poses at different abstract granularity levels, starting from a 6-part body representation and gradually refining to 22 joints. With decreasing abstract levels step by step, MAGE introduces more motion context priors from former prediction stages and thus improves realistic motion completion with richer constraint conditions and less ambiguity. Extensive experiments on large-scale datasets verify that MAGE significantly outperforms state-of-the-art methods with better accuracy and continuity. 

**Abstract (ZH)**: 从头戴设备推断全身体态：一种逐步Avatar生成器MAGE的方法 

---
# Bi-LSTM based Multi-Agent DRL with Computation-aware Pruning for Agent Twins Migration in Vehicular Embodied AI Networks 

**Title (ZH)**: 基于Bi-LSTM的考虑计算量的多Agent DRL与代理双胞胎迁移修剪方法在车载实体AI网络中的应用 

**Authors**: Yuxiang Wei, Zhuoqi Zeng, Yue Zhong, Jiawen Kang, Ryan Wen Liu, M. Shamim Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.06378)  

**Abstract**: With the advancement of large language models and embodied Artificial Intelligence (AI) in the intelligent transportation scenarios, the combination of them in intelligent transportation spawns the Vehicular Embodied AI Network (VEANs). In VEANs, Autonomous Vehicles (AVs) are typical agents whose local advanced AI applications are defined as vehicular embodied AI agents, enabling capabilities such as environment perception and multi-agent collaboration. Due to computation latency and resource constraints, the local AI applications and services running on vehicular embodied AI agents need to be migrated, and subsequently referred to as vehicular embodied AI agent twins, which drive the advancement of vehicular embodied AI networks to offload intensive tasks to Roadside Units (RSUs), mitigating latency problems while maintaining service quality. Recognizing workload imbalance among RSUs in traditional approaches, we model AV-RSU interactions as a Stackelberg game to optimize bandwidth resource allocation for efficient migration. A Tiny Multi-Agent Bidirectional LSTM Proximal Policy Optimization (TMABLPPO) algorithm is designed to approximate the Stackelberg equilibrium through decentralized coordination. Furthermore, a personalized neural network pruning algorithm based on Path eXclusion (PX) dynamically adapts to heterogeneous AV computation capabilities by identifying task-critical parameters in trained models, reducing model complexity with less performance degradation. Experimental validation confirms the algorithm's effectiveness in balancing system load and minimizing delays, demonstrating significant improvements in vehicular embodied AI agent deployment. 

**Abstract (ZH)**: 基于大规模语言模型和具身人工智能的智能交通场景下具身人工智能网络（VEANs）：自治车辆与道路侧单元的工作负载优化 

---
# ARDNS-FN-Quantum: A Quantum-Enhanced Reinforcement Learning Framework with Cognitive-Inspired Adaptive Exploration for Dynamic Environments 

**Title (ZH)**: ARDNS-FN-量子增强强化学习框架：认知启发式自适应探索用于动态环境 

**Authors**: Umberto Gonçalves de Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06300)  

**Abstract**: Reinforcement learning (RL) has transformed sequential decision making, yet traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents ARDNS-FN-Quantum (Adaptive Reward-Driven Neural Simulator with Quantum enhancement), a novel framework that integrates a 2-qubit quantum circuit for action selection, a dual-memory system inspired by human cognition, and adaptive exploration strategies modulated by reward variance and curiosity. Evaluated in a 10X10 grid-world over 20,000 episodes, ARDNS-FN-Quantum achieves a 99.5% success rate (versus 81.3% for DQN and 97.0% for PPO), a mean reward of 9.0528 across all episodes (versus 1.2941 for DQN and 7.6196 for PPO), and an average of 46.7 steps to goal (versus 135.9 for DQN and 62.5 for PPO). In the last 100 episodes, it records a mean reward of 9.1652 (versus 7.0916 for DQN and 9.0310 for PPO) and 37.2 steps to goal (versus 52.7 for DQN and 53.4 for PPO). Graphical analyses, including learning curves, steps-to-goal trends, reward variance, and reward distributions, demonstrate ARDNS-FN-Quantum's superior stability (reward variance 5.424 across all episodes versus 252.262 for DQN and 76.583 for PPO) and efficiency. By bridging quantum computing, cognitive science, and RL, ARDNS-FN-Quantum offers a scalable, human-like approach to adaptive learning in uncertain environments, with potential applications in robotics, autonomous systems, and decision-making under uncertainty. 

**Abstract (ZH)**: 强化学习（RL）已重塑序列决策过程，但传统的算法如深度Q网络（DQN）和渐近策略优化（PPO）往往在动态环境中的高效探索、稳定性和适应性方面存在问题。本文提出了一种名为ARDNS-FN-量子（Adaptive Reward-Driven Neural Simulator with Quantum Enhancement）的新颖框架，该框架整合了用于动作选择的2-qubit量子电路、受人类认知启发的双记忆系统以及由奖励方差和好奇性调节的自适应探索策略。在10×10的网格世界中经过20,000个episode的评估，ARDNS-FN-量子实现了99.5%的成功率（相比之下，DQN为81.3%，PPO为97.0%）、平均每回合9.0528的奖励（相比之下，DQN为1.2941，PPO为7.6196）以及平均46.7步达到目标的性能（相比之下，DQN为135.9，PPO为62.5）。最后100个episode中，平均奖励为9.1652（相比之下，DQN为7.0916，PPO为9.0310），平均达到目标步数为37.2（相比之下，DQN为52.7，PPO为53.4）。图形分析包括学习曲线、达到目标步数趋势、奖励方差和奖励分布，证明了ARDNS-FN-量子在稳定性和效率方面的优越性能。通过结合量子计算、认知科学和强化学习，ARDNS-FN-量子提供了一种可扩展、类人的适应性学习方法，适用于不确定环境下的机器人、自主系统和不确定性决策，具有潜在应用价值。 

---
# Beyond Attention: Toward Machines with Intrinsic Higher Mental States 

**Title (ZH)**: 超越注意力：迈向拥有内在高级心理状态的机器 

**Authors**: Ahsan Adeel  

**Link**: [PDF](https://arxiv.org/pdf/2505.06257)  

**Abstract**: Attending to what is relevant is fundamental to both the mammalian brain and modern machine learning models such as Transformers. Yet, determining relevance remains a core challenge, traditionally offloaded to learning algorithms like backpropagation. Inspired by recent cellular neurobiological evidence linking neocortical pyramidal cells to distinct mental states, this work shows how models (e.g., Transformers) can emulate high-level perceptual processing and awake thought (imagination) states to pre-select relevant information before applying attention. Triadic neuronal-level modulation loops among questions ($Q$), clues (keys, $K$), and hypotheses (values, $V$) enable diverse, deep, parallel reasoning chains at the representation level and allow a rapid shift from initial biases to refined understanding. This leads to orders-of-magnitude faster learning with significantly reduced computational demand (e.g., fewer heads, layers, and tokens), at an approximate cost of $\mathcal{O}(N)$, where $N$ is the number of input tokens. Results span reinforcement learning (e.g., CarRacing in a high-dimensional visual setup), computer vision, and natural language question answering. 

**Abstract (ZH)**: 关注相关信息是哺乳动物大脑和现代机器学习模型如变换器的基本要素。然而，确定相关性仍然是一个核心挑战，传统上由反向传播等学习算法承担。受最近细胞神经生物学证据启发，该工作展示了如何使模型（如变换器）模拟高级感知处理和清醒思考（想象）状态，在应用注意力之前预先筛选相关信息。由问题（$Q$）、线索（键，$K$）和假设（值，$V$）三元神经级调节回路实现多样、深入、并行的表示层面推理链，并允许从初始偏见迅速转向精炼理解。这导致了比传统方法快得多的学习速度，计算需求显著降低（如较少的头部、层和标记），并以约$\mathcal{O}(N)$的成本，其中$N$为输入标记的数量。结果涵盖强化学习（例如，高维视觉设置中的CarRacing）、计算机视觉和自然语言问答领域。 

---
