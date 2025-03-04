# CurviTrack: Curvilinear Trajectory Tracking for High-speed Chase of a USV 

**Title (ZH)**: CurviTrack: 曲线轨迹跟踪用于高速追逐USV 

**Authors**: Parakh M. Gupta, Ondřej Procházka, Tiago Nascimento, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2502.21303)  

**Abstract**: Heterogeneous robot teams used in marine environments incur time-and-energy penalties when the marine vehicle has to halt the mission to allow the autonomous aerial vehicle to land for recharging. In this paper, we present a solution for this problem using a novel drag-aware model formulation which is coupled with MPC, and therefore, enables tracking and landing during high-speed curvilinear trajectories of an USV without any communication. Compared to the state-of-the-art, our approach yields 40% decrease in prediction errors, and provides a 3-fold increase in certainty of predictions. Consequently, this leads to a 30% improvement in tracking performance and 40% higher success in landing on a moving USV even during aggressive turns that are unfeasible for conventional marine missions. We test our approach in two different real-world scenarios with marine vessels of two different sizes and further solidify our results through statistical analysis in simulation to demonstrate the robustness of our method. 

**Abstract (ZH)**: 海洋环境下异构机器人团队在自主水下车辆需要暂停任务以允许自主无人航空器降落充电时会遭受时间与能源的损失。本文提出了一种解决方案，该方案结合了一种新颖的考虑阻力的模型 formulations，并与 MPC 相耦合，从而能够在无通信的情况下跟踪和着陆高速曲线轨迹的 USV。与现有技术相比，我们的方法将预测误差降低了 40%，并提供了 3 倍的预测 certainty 提高。这导致跟踪性能提高了 30%，并在即使对于传统海洋任务不可行的激进转弯中，着陆成功率也提高了 40%。我们在两种不同尺寸的海洋船只的两个不同真实场景中测试了这种方法，并通过仿真中的统计分析进一步验证了结果，以证明该方法的鲁棒性。 

---
# RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete 

**Title (ZH)**: RoboBrain: 从抽象到具体的机器人操作统一脑模型 

**Authors**: Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21257)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various multimodal contexts. However, their application in robotic scenarios, particularly for long-horizon manipulation tasks, reveals significant limitations. These limitations arise from the current MLLMs lacking three essential robotic brain capabilities: Planning Capability, which involves decomposing complex manipulation instructions into manageable sub-tasks; Affordance Perception, the ability to recognize and interpret the affordances of interactive objects; and Trajectory Prediction, the foresight to anticipate the complete manipulation trajectory necessary for successful execution. To enhance the robotic brain's core capabilities from abstract to concrete, we introduce ShareRobot, a high-quality heterogeneous dataset that labels multi-dimensional information such as task planning, object affordance, and end-effector trajectory. ShareRobot's diversity and accuracy have been meticulously refined by three human annotators. Building on this dataset, we developed RoboBrain, an MLLM-based model that combines robotic and general multi-modal data, utilizes a multi-stage training strategy, and incorporates long videos and high-resolution images to improve its robotic manipulation capabilities. Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various robotic tasks, highlighting its potential to advance robotic brain capabilities. 

**Abstract (ZH)**: 近期多模态大型语言模型（MLLMs）在各种多模态场景中的应用显示出了显著的能力，但在机器人应用场景中，特别是对于远期操作任务，其应用却暴露出显著的局限性。这些局限性源于当前MLLMs缺乏三种关键的机器人脑能力：规划能力（将复杂的操作指令分解为可管理的子任务）、可利用性感知（识别和解释交互对象的可利用性）以及轨迹预测（预见完成操作所需的完整轨迹）。为了使机器人脑的能力从抽象层面提升到具体层面，我们介绍了ShareRobot，这是一个高质量的异构数据集，标注了任务规划、物体可利用性和末端执行器轨迹等多维度信息。ShareRobot的多样性和准确性经过三位人工标注者的精细调整。基于这一数据集，我们开发了RoboBrain，这是一种结合了机器人和通用多模态数据的MLLM模型，采用多阶段训练策略，并结合长视频和高分辨率图像以提高其操作能力。广泛的实验表明，RoboBrain在各种机器人任务中达到了最先进的性能，突显了其提升机器人脑能力的潜力。 

---
# Dynamically Local-Enhancement Planner for Large-Scale Autonomous Driving 

**Title (ZH)**: 大规模自主驾驶的动态局部增强规划者 

**Authors**: Nanshan Deng, Weitao Zhou, Bo Zhang, Junze Wen, Kun Jiang, Zhong Cao, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21134)  

**Abstract**: Current autonomous vehicles operate primarily within limited regions, but there is increasing demand for broader applications. However, as models scale, their limited capacity becomes a significant challenge for adapting to novel scenarios. It is increasingly difficult to improve models for new situations using a single monolithic model. To address this issue, we introduce the concept of dynamically enhancing a basic driving planner with local driving data, without permanently modifying the planner itself. This approach, termed the Dynamically Local-Enhancement (DLE) Planner, aims to improve the scalability of autonomous driving systems without significantly expanding the planner's size. Our approach introduces a position-varying Markov Decision Process formulation coupled with a graph neural network that extracts region-specific driving features from local observation data. The learned features describe the local behavior of the surrounding objects, which is then leveraged to enhance a basic reinforcement learning-based policy. We evaluated our approach in multiple scenarios and compared it with a one-for-all driving model. The results show that our method outperforms the baseline policy in both safety (collision rate) and average reward, while maintaining a lighter scale. This approach has the potential to benefit large-scale autonomous vehicles without the need for largely expanding on-device driving models. 

**Abstract (ZH)**: 动态局部增强的驾驶规划器：提高自主驾驶系统的可扩展性 

---
# Jointly Assigning Processes to Machines and Generating Plans for Autonomous Mobile Robots in a Smart Factory 

**Title (ZH)**: 智能工厂中联合分配过程和自主移动机器人生成计划的研究 

**Authors**: Christopher Leet, Aidan Sciortino, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2502.21101)  

**Abstract**: A modern smart factory runs a manufacturing procedure using a collection of programmable machines. Typically, materials are ferried between these machines using a team of mobile robots. To embed a manufacturing procedure in a smart factory, a factory operator must a) assign its processes to the smart factory's machines and b) determine how agents should carry materials between machines. A good embedding maximizes the smart factory's throughput; the rate at which it outputs products. Existing smart factory management systems solve the aforementioned problems sequentially, limiting the throughput that they can achieve. In this paper we introduce ACES, the Anytime Cyclic Embedding Solver, the first solver which jointly optimizes the assignment of processes to machines and the assignment of paths to agents. We evaluate ACES and show that it can scale to real industrial scenarios. 

**Abstract (ZH)**: 一种现代智能工厂使用可编程机器运行制造过程。通常，材料在这些机器之间由一组移动机器人传递。要将制造过程嵌入智能工厂，工厂操作员必须完成以下两项任务：a) 将其过程分配给智能工厂的机器，b) 确定代理在机器之间携带材料的路径。良好的嵌入可以最大化智能工厂的 throughput；即其产出产品速率。现有的智能工厂管理系统按顺序解决上述问题，限制了它们能达到的 throughput。本文介绍了一种即席循环嵌入求解器 ACES，这是第一个同时优化过程分配给机器和路径分配给代理的求解器。我们评估了 ACES，并展示了它能够适应实际工业场景。 

---
# AuthSim: Towards Authentic and Effective Safety-critical Scenario Generation for Autonomous Driving Tests 

**Title (ZH)**: AuthSim: 向Towards真实有效的自动驾驶安全场景生成ALSEscientific标准 

**Authors**: Yukuan Yang, Xucheng Lu, Zhili Zhang, Zepeng Wu, Guoqi Li, Lingzhong Meng, Yunzhi Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.21100)  

**Abstract**: Generating adversarial safety-critical scenarios is a pivotal method for testing autonomous driving systems, as it identifies potential weaknesses and enhances system robustness and reliability. However, existing approaches predominantly emphasize unrestricted collision scenarios, prompting non-player character (NPC) vehicles to attack the ego vehicle indiscriminately. These works overlook these scenarios' authenticity, rationality, and relevance, resulting in numerous extreme, contrived, and largely unrealistic collision events involving aggressive NPC vehicles. To rectify this issue, we propose a three-layer relative safety region model, which partitions the area based on danger levels and increases the likelihood of NPC vehicles entering relative boundary regions. This model directs NPC vehicles to engage in adversarial actions within relatively safe boundary regions, thereby augmenting the scenarios' authenticity. We introduce AuthSim, a comprehensive platform for generating authentic and effective safety-critical scenarios by integrating the three-layer relative safety region model with reinforcement learning. To our knowledge, this is the first attempt to address the authenticity and effectiveness of autonomous driving system test scenarios comprehensively. Extensive experiments demonstrate that AuthSim outperforms existing methods in generating effective safety-critical scenarios. Notably, AuthSim achieves a 5.25% improvement in average cut-in distance and a 27.12% enhancement in average collision interval time, while maintaining higher efficiency in generating effective safety-critical scenarios compared to existing methods. This underscores its significant advantage in producing authentic scenarios over current methodologies. 

**Abstract (ZH)**: 生成相对安全区域模型以提高自主驾驶系统测试场景的真实性和有效性 

---
# Robust Deterministic Policy Gradient for Disturbance Attenuation and Its Application to Quadrotor Control 

**Title (ZH)**: 稳健的确定性策略梯度方法在干扰抑制中的应用及其在四旋翼控制中的应用 

**Authors**: Taeho Lee, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.21057)  

**Abstract**: Practical control systems pose significant challenges in identifying optimal control policies due to uncertainties in the system model and external disturbances. While $H_\infty$ control techniques are commonly used to design robust controllers that mitigate the effects of disturbances, these methods often require complex and computationally intensive calculations. To address this issue, this paper proposes a reinforcement learning algorithm called Robust Deterministic Policy Gradient (RDPG), which formulates the $H_\infty$ control problem as a two-player zero-sum dynamic game. In this formulation, one player (the user) aims to minimize the cost, while the other player (the adversary) seeks to maximize it. We then employ deterministic policy gradient (DPG) and its deep reinforcement learning counterpart to train a robust control policy with effective disturbance attenuation. In particular, for practical implementation, we introduce an algorithm called robust deep deterministic policy gradient (RDDPG), which employs a deep neural network architecture and integrates techniques from the twin-delayed deep deterministic policy gradient (TD3) to enhance stability and learning efficiency. To evaluate the proposed algorithm, we implement it on an unmanned aerial vehicle (UAV) tasked with following a predefined path in a disturbance-prone environment. The experimental results demonstrate that the proposed method outperforms other control approaches in terms of robustness against disturbances, enabling precise real-time tracking of moving targets even under severe disturbance conditions. 

**Abstract (ZH)**: 实用控制系统由于系统模型的不确定性与外部干扰，在确定最优控制策略上面临重大挑战。虽然 $H_\infty$ 控制技术常被用于设计鲁棒控制器以减轻干扰的影响，但这些方法往往需要复杂的且计算密集的计算。为解决此问题，本文提出了一种称为鲁棒确定性策略梯度（RDPG）的强化学习算法，将 $H_\infty$ 控制问题形式化为两人零和动态博弈问题。在该形式化中，一方（用户）力求最小化成本，而另一方（对手）则力求最大化成本。我们随后利用确定性策略梯度（DPG）及其深度强化学习的相应技术来训练具有有效干扰衰减的鲁棒控制策略。特别是从实际实施的角度出发，我们引入了一种称为鲁棒深度确定性策略梯度（RDDPG）的算法，该算法采用深度神经网络架构，并结合了双延迟深度确定性策略梯度（TD3）的技术来提升稳定性和学习效率。为了评估所提出的算法，我们将其应用于一个无人机，该无人机在多干扰环境中需要遵循预定义的飞行路径。实验结果表明，所提出的方法在抵抗干扰方面优于其他控制方法，能够在严重干扰条件下实现精确的实时目标跟踪。 

---
# Vibrotactile information coding strategies for a body-worn vest to aid robot-human collaboration 

**Title (ZH)**: 基于穿戴式背心的振动触觉信息编码策略以辅助机器人-人类协作 

**Authors**: Adrian Vecina Tercero, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2502.21056)  

**Abstract**: This paper explores the use of a body-worn vibrotactile vest to convey real-time information from robot to operator. Vibrotactile communication could be useful in providing information without compropmising or loading a person's visual or auditory perception. This paper considers applications in Urban Search and Rescue (USAR) scenarios where a human working alongside a robot is likely to be operating in high cognitive load conditions. The focus is on understanding how best to convey information considering different vibrotactile information coding strategies to enhance scene understanding in scenarios where a robot might be operating remotely as a scout. In exploring information representation, this paper introduces Semantic Haptics, using shapes and patterns to represent certain events as if the skin was a screen, and shows how these lead to bettter learnability and interpreation accuracy. 

**Abstract (ZH)**: 本文探讨了使用穿戴式震动触觉背心来实时传输机器人到操作员的信息。触觉通信在无需牺牲或负担人员的视觉或听觉感知的情况下提供信息可能非常有用。本文考虑了在城市搜救（USAR）场景中的应用，在这些场景中，与机器人并肩工作的人员可能处于高认知负荷条件。重点是理解如何最好地传递信息，以考虑不同的触觉信息编码策略，以增强在机器人可能作为侦察员远程操作的情况下场景理解。在探索信息表示时，本文引入了语义触觉概念，使用形状和模式表示特定事件，仿佛皮肤是一个屏幕，并展示了这些方法如何提高学习能力和解释准确性。 

---
# Sixth-Sense: Self-Supervised Learning of Spatial Awareness of Humans from a Planar Lidar 

**Title (ZH)**: 六感：基于平面激光雷达的人类空间意识自监督学习 

**Authors**: Simone Arreghini, Nicholas Carlotti, Mirko Nava, Antonio Paolillo, Alessandro Giusti  

**Link**: [PDF](https://arxiv.org/pdf/2502.21029)  

**Abstract**: Localizing humans is a key prerequisite for any service robot operating in proximity to people. In these scenarios, robots rely on a multitude of state-of-the-art detectors usually designed to operate with RGB-D cameras or expensive 3D LiDARs. However, most commercially available service robots are equipped with cameras with a narrow field of view, making them blind when a user is approaching from other directions, or inexpensive 1D LiDARs whose readings are difficult to interpret. To address these limitations, we propose a self-supervised approach to detect humans and estimate their 2D pose from 1D LiDAR data, using detections from an RGB-D camera as a supervision source. Our approach aims to provide service robots with spatial awareness of nearby humans. After training on 70 minutes of data autonomously collected in two environments, our model is capable of detecting humans omnidirectionally from 1D LiDAR data in a novel environment, with 71% precision and 80% recall, while retaining an average absolute error of 13 cm in distance and 44° in orientation. 

**Abstract (ZH)**: 面向人体局部化的自监督方法：基于1D LiDAR数据的服务机器人环境感知 

---
# Nano Drone-based Indoor Crime Scene Analysis 

**Title (ZH)**: 基于纳米无人机的室内犯罪现场分析 

**Authors**: Martin Cooney, Sivadinesh Ponrajan, Fernando Alonso-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2502.21019)  

**Abstract**: Technologies such as robotics, Artificial Intelligence (AI), and Computer Vision (CV) can be applied to crime scene analysis (CSA) to help protect lives, facilitate justice, and deter crime, but an overview of the tasks that can be automated has been lacking. Here we follow a speculate prototyping approach: First, the STAIR tool is used to rapidly review the literature and identify tasks that seem to have not received much attention, like accessing crime sites through a window, mapping/gathering evidence, and analyzing blood smears. Secondly, we present a prototype of a small drone that implements these three tasks with 75%, 85%, and 80% performance, to perform a minimal analysis of an indoor crime scene. Lessons learned are reported, toward guiding next work in the area. 

**Abstract (ZH)**: 机器人、人工智能（AI）和计算机视觉（CV）技术可以应用于犯罪现场分析（CSA），以保护生命、促进正义和威慑犯罪，但自动化可执行的任务概述尚缺乏。在这里，我们采用推测性原型设计方法：首先，使用STAIR工具迅速回顾文献，识别出尚未得到足够关注的任务，例如通过窗户访问犯罪现场、制图/收集证据以及分析血痕。其次，我们提出一种小型无人机原型，该原型可实现上述三项任务，性能分别为75%、85%和80%，以执行室内犯罪现场的最小化分析。报告了所学经验教训，以指导该领域的后续工作。 

---
# Motion ReTouch: Motion Modification Using Four-Channel Bilateral Control 

**Title (ZH)**: 运动精修：基于四通道双边控制的运动修改 

**Authors**: Koki Inami, Sho Sakaino, Toshiaki Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.20982)  

**Abstract**: Recent research has demonstrated the usefulness of imitation learning in autonomous robot operation. In particular, teaching using four-channel bilateral control, which can obtain position and force information, has been proven effective. However, control performance that can easily execute high-speed, complex tasks in one go has not yet been achieved. We propose a method called Motion ReTouch, which retroactively modifies motion data obtained using four-channel bilateral control. The proposed method enables modification of not only position but also force information. This was achieved by the combination of multilateral control and motion-copying system. The proposed method was verified in experiments with a real robot, and the success rate of the test tube transfer task was improved, demonstrating the possibility of modification force information. 

**Abstract (ZH)**: Recent Research Demonstrates the Utility of Imitation Learning in Autonomous Robot Operation: Motion ReTouch method Retroactively Modifies Motion Data with Position and Force Information 

---
# DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping 

**Title (ZH)**: DexGraspVLA：一种面向通用灵巧抓取的视觉-语言-行动框架 

**Authors**: Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Yitao Liang, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20900)  

**Abstract**: Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on specific assumptions, such as single-object settings or limited environments, leading to constrained generalization. Our solution is DexGraspVLA, a hierarchical framework that utilizes a pre-trained Vision-Language model as the high-level task planner and learns a diffusion-based policy as the low-level Action controller. The key insight lies in iteratively transforming diverse language and visual inputs into domain-invariant representations, where imitation learning can be effectively applied due to the alleviation of domain shift. Thus, it enables robust generalization across a wide range of real-world scenarios. Notably, our method achieves a 90+% success rate under thousands of unseen object, lighting, and background combinations in a ``zero-shot'' environment. Empirical analysis further confirms the consistency of internal model behavior across environmental variations, thereby validating our design and explaining its generalization performance. We hope our work can be a step forward in achieving general dexterous grasping. Our demo and code can be found at this https URL. 

**Abstract (ZH)**: Dexterous抓取仍然是机器人领域一个基本而又具有挑战性的问题。通用机器人必须能够在任意场景中抓取多样化的物体。然而，现有的研究通常依赖于特定的前提假设，如单一物体设置或有限的环境，导致泛化能力受限。我们的解决方案是DexGraspVLA，这是一种分层框架，利用预训练的视觉-语言模型作为高层次的任务规划器，并学习基于扩散的策略作为低层次的动作控制器。关键在于迭代地将多样化的语言和视觉输入转换为领域不变的表示，从而在领域偏移减轻的情况下可以有效应用模仿学习，因此能够在广泛的实际应用场景中实现稳健的泛化。值得注意的是，我们的方法在数千种未见过的物体、光照和背景组合的“零样本”环境中实现了90%以上的成功率。经验分析进一步证实了模型在环境变化下内部行为一致性，从而验证了我们的设计并解释了其泛化性能。我们希望我们的工作能为实现通用灵巧抓取迈出一步。我们的演示和代码可以在以下链接找到。 

---
# Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments 

**Title (ZH)**: 非抓握操作在通用环境中的分层与模块化网络 

**Authors**: Yoonyoung Cho, Junhyek Han, Jisu Han, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.20843)  

**Abstract**: For robots to operate in general environments like households, they must be able to perform non-prehensile manipulation actions such as toppling and rolling to manipulate ungraspable objects. However, prior works on non-prehensile manipulation cannot yet generalize across environments with diverse geometries. The main challenge lies in adapting to varying environmental constraints: within a cabinet, the robot must avoid walls and ceilings; to lift objects to the top of a step, the robot must account for the step's pose and extent. While deep reinforcement learning (RL) has demonstrated impressive success in non-prehensile manipulation, accounting for such variability presents a challenge for the generalist policy, as it must learn diverse strategies for each new combination of constraints. To address this, we propose a modular and reconfigurable architecture that adaptively reconfigures network modules based on task requirements. To capture the geometric variability in environments, we extend the contact-based object representation (CORN) to environment geometries, and propose a procedural algorithm for generating diverse environments to train our agent. Taken together, the resulting policy can zero-shot transfer to novel real-world environments and objects despite training entirely within a simulator. We additionally release a simulation-based benchmark featuring nine digital twins of real-world scenes with 353 objects to facilitate non-prehensile manipulation research in realistic domains. 

**Abstract (ZH)**: 针对一般环境如家庭中操作的机器人，必须能够执行非接触性操作，如推倒和滚动，以操纵无法抓取的对象。然而，目前对非接触性操作的研究还不能跨不同几何结构的环境进行泛化。主要挑战在于适应变化的环境约束：在橱柜内，机器人必须避开墙壁和天花板；为了将物体举到台阶顶部，机器人必须考虑台阶的姿态和范围。虽然深度强化学习在非接触性操作上取得了显著成果，但适应这些变化性对通用策略构成了挑战，因为策略必须学会应对每种新的约束组合所采用的多样化策略。为此，我们提出了一种模块化且可重构的架构，该架构可以根据任务需求自适应地重构网络模块。为了捕捉环境中的几何变化性，我们将基于接触的对象表示（CORN）扩展到环境几何结构，并提出了一种生成多样环境的程序化算法，用于训练我们的代理。总体而言，生成的策略能够在完全在模拟器中训练的情况下，零样本迁移至全新的真实世界环境和对象。此外，我们还发布了一个基于模拟的基准测试，其中包括九个真实世界场景的九个数字孪生体和353个物体，以促进在真实场景中进行非接触性操作的研究。 

---
# Learning-Based Leader Localization for Underwater Vehicles With Optical-Acoustic-Pressure Sensor Fusion 

**Title (ZH)**: 基于学习的水下自主车辆 optics-acoustics-pressure 传感器融合领导者定位 

**Authors**: Mingyang Yang, Zeyu Sha, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20817)  

**Abstract**: Underwater vehicles have emerged as a critical technology for exploring and monitoring aquatic environments. The deployment of multi-vehicle systems has gained substantial interest due to their capability to perform collaborative tasks with improved efficiency. However, achieving precise localization of a leader underwater vehicle within a multi-vehicle configuration remains a significant challenge, particularly in dynamic and complex underwater conditions. To address this issue, this paper presents a novel tri-modal sensor fusion neural network approach that integrates optical, acoustic, and pressure sensors to localize the leader vehicle. The proposed method leverages the unique strengths of each sensor modality to improve localization accuracy and robustness. Specifically, optical sensors provide high-resolution imaging for precise relative positioning, acoustic sensors enable long-range detection and ranging, and pressure sensors offer environmental context awareness. The fusion of these sensor modalities is implemented using a deep learning architecture designed to extract and combine complementary features from raw sensor data. The effectiveness of the proposed method is validated through a custom-designed testing platform. Extensive data collection and experimental evaluations demonstrate that the tri-modal approach significantly improves the accuracy and robustness of leader localization, outperforming both single-modal and dual-modal methods. 

**Abstract (ZH)**: 水下无人车辆已成为探索和监测水生环境的关键技术。多车辆系统的部署因其能够执行协作任务并提高效率而引起了广泛关注。然而，在动态和复杂的水下条件下，精确定位多车辆配置中的领航车辆仍然是一个重大挑战。为了应对这一问题，本文提出了一种新颖的三模态传感器融合神经网络方法，该方法结合了光学、声学和压力传感器以定位领航车辆。所提出的方法利用了每种传感器模态的独特优势，以提高定位精度和鲁棒性。具体而言，光学传感器提供高分辨率成像以实现精确的相对定位，声学传感器实现远距离检测和测距，而压力传感器提供环境上下文意识。这三种传感器模态的融合是通过一种设计用于从原始传感器数据中提取和结合互补特征的深度学习架构实现的。通过一个定制设计的测试平台验证了所提出方法的有效性。大量数据收集和实验评估表明，三模态方法显著提高了领航车辆定位的准确性和鲁棒性，优于单模态和双模态方法。 

---
# Towards Semantic 3D Hand-Object Interaction Generation via Functional Text Guidance 

**Title (ZH)**: 基于功能性文本指导的语义三维手物交互生成 

**Authors**: Yongqi Tian, Xueyu Sun, Haoyuan He, Linji Hao, Ning Ding, Caigui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20805)  

**Abstract**: Hand-object interaction(HOI) is the fundamental link between human and environment, yet its dexterous and complex pose significantly challenges for gesture control. Despite significant advances in AI and robotics, enabling machines to understand and simulate hand-object interactions, capturing the semantics of functional grasping tasks remains a considerable challenge. While previous work can generate stable and correct 3D grasps, they are still far from achieving functional grasps due to unconsidered grasp semantics. To address this challenge, we propose an innovative two-stage framework, Functional Grasp Synthesis Net (FGS-Net), for generating 3D HOI driven by functional text. This framework consists of a text-guided 3D model generator, Functional Grasp Generator (FGG), and a pose optimization strategy, Functional Grasp Refiner (FGR). FGG generates 3D models of hands and objects based on text input, while FGR fine-tunes the poses using Object Pose Approximator and energy functions to ensure the relative position between the hand and object aligns with human intent and remains physically plausible. Extensive experiments demonstrate that our approach achieves precise and high-quality HOI generation without requiring additional 3D annotation data. 

**Abstract (ZH)**: 功能化抓取合成网络（FGS-Net）：基于功能文本驱动的3D手物互动生成 

---
# Characteristics Analysis of Autonomous Vehicle Pre-crash Scenarios 

**Title (ZH)**: 自主驾驶车辆预碰撞场景特征分析 

**Authors**: Yixuan Li, Xuesong Wang, Tianyi Wang, Qian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20789)  

**Abstract**: To date, hundreds of crashes have occurred in open road testing of automated vehicles (AVs), highlighting the need for improving AV reliability and safety. Pre-crash scenario typology classifies crashes based on vehicle dynamics and kinematics features. Building on this, characteristics analysis can identify similar features under comparable crashes, offering a more effective reflection of general crash patterns and providing more targeted recommendations for enhancing AV performance. However, current studies primarily concentrated on crashes among conventional human-driven vehicles, leaving a gap in research dedicated to in-depth AV crash analyses. In this paper, we analyzed the latest California AV collision reports and used the newly revised pre-crash scenario typology to identify pre-crash scenarios. We proposed a set of mapping rules for automatically extracting these AV pre-crash scenarios, successfully identifying 24 types with a 98.1% accuracy rate, and obtaining two key scenarios of AV crashes (i.e., rear-end scenarios and intersection scenarios) through detailed analysis. Association analyses of rear-end scenarios showed that the significant environmental influencing factors were traffic control type, location type, light, etc. For intersection scenarios prone to severe crashes with detailed descriptions, we employed causal analyses to obtain the significant causal factors: habitual violations and expectations of certain behavior. Optimization recommendations were then formulated, addressing both governmental oversight and AV manufacturers' potential improvements. The findings of this paper could guide government authorities to develop related regulations, help manufacturers design AV test scenarios, and identify potential shortcomings in control algorithms specific to various real-world scenarios, thereby optimizing AV systems effectively. 

**Abstract (ZH)**: 到目前为止，开放道路测试中已经发生了数百起自动驾驶车辆（AVs）事故，凸显了提高AV可靠性和安全性的重要性。基于车辆动力学和运动学特征的预事故场景分类能够根据事故特征对事故进行分类，通过特征分析可识别出具有相似特征的可比事故，更有效地反映一般事故模式，提供更具针对性的建议以提升AV性能。然而，当前的研究主要集中在传统人类驾驶车辆的事故上，对AV事故的深入分析研究相对缺乏。本文分析了最新的加州AV碰撞报告，并使用修订后的预事故场景分类来识别预事故场景。我们提出了自动提取AV预事故场景的一套映射规则，成功识别出24种类型，准确率达到98.1%，并通过详细分析确定了两类关键的AV事故场景（即追尾场景和交叉口场景）。追尾场景的相关性分析显示，主要环境影响因素包括交通控制类型、位置类型、光线等。针对描述详细的易发生严重事故的交叉口场景，我们采用了因果分析以获得关键的因果因素：习惯性违法行为和特定行为的预期。随后，我们提出了优化建议，涉及政府监管和AV制造商的潜在改进。本文的研究成果可指导政府部门制定相关法规，帮助制造商设计AV测试场景，并识别特定实际场景下控制算法的潜在不足，从而有效地优化AV系统。 

---
# CSubBT: A Self-Adjusting Execution Framework for Mobile Manipulation System 

**Title (ZH)**: CSubBT: 一种移动操作系统的自适应执行框架 

**Authors**: Huihui Guo, Huizhang Luo, Huilong Pi, Mingxing Duan, Kenli Li, Chubo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20771)  

**Abstract**: With the advancements in modern intelligent technologies, mobile robots equipped with manipulators are increasingly operating in unstructured environments. These robots can plan sequences of actions for long-horizon tasks based on perceived information. However, in practice, the planned actions often fail due to discrepancies between the perceptual information used for planning and the actual conditions. In this paper, we introduce the {\itshape Conditional Subtree} (CSubBT), a general self-adjusting execution framework for mobile manipulation tasks based on Behavior Trees (BTs). CSubBT decomposes symbolic action into sub-actions and uses BTs to control their execution, addressing any potential anomalies during the process. CSubBT treats common anomalies as constraint non-satisfaction problems and continuously guides the robot in performing tasks by sampling new action parameters in the constraint space when anomalies are detected. We demonstrate the robustness of our framework through extensive manipulation experiments on different platforms, both in simulation and real-world settings. 

**Abstract (ZH)**: 基于行为树的条件子树：移动操纵任务的自适应执行框架 

---
# A2DO: Adaptive Anti-Degradation Odometry with Deep Multi-Sensor Fusion for Autonomous Navigation 

**Title (ZH)**: 自适应抗退化里程计：基于深度多传感器融合的自主导航 

**Authors**: Hui Lai, Qi Chen, Junping Zhang, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20767)  

**Abstract**: Accurate localization is essential for the safe and effective navigation of autonomous vehicles, and Simultaneous Localization and Mapping (SLAM) is a cornerstone technology in this context. However, The performance of the SLAM system can deteriorate under challenging conditions such as low light, adverse weather, or obstructions due to sensor degradation. We present A2DO, a novel end-to-end multi-sensor fusion odometry system that enhances robustness in these scenarios through deep neural networks. A2DO integrates LiDAR and visual data, employing a multi-layer, multi-scale feature encoding module augmented by an attention mechanism to mitigate sensor degradation dynamically. The system is pre-trained extensively on simulated datasets covering a broad range of degradation scenarios and fine-tuned on a curated set of real-world data, ensuring robust adaptation to complex scenarios. Our experiments demonstrate that A2DO maintains superior localization accuracy and robustness across various degradation conditions, showcasing its potential for practical implementation in autonomous vehicle systems. 

**Abstract (ZH)**: 准确的定位对于自主车辆的安全有效导航至关重要，而同时定位与建图（SLAM）是这一过程中的核心技术。然而，在低光照、恶劣天气或传感器退化等具有挑战性条件下，SLAM系统的性能可能会下降。我们提出了一种名为A2DO的新型端到端多传感器融合里程计系统，通过深度神经网络增强在这些场景下的鲁棒性。A2DO结合了LiDAR和视觉数据，通过具有注意机制的多层多尺度特征编码模块动态减轻传感器退化的影响。该系统在覆盖广泛退化场景的模拟数据集上进行广泛的预训练，并在精心收集的真实数据集上进行微调，以确保其在复杂场景中的鲁棒适应性。我们的实验结果表明，A2DO能够在各种退化条件下保持卓越的定位准确性和鲁棒性，展示了其在自主车辆系统中的实际应用潜力。 

---
# Indoor Localization for Autonomous Robot Navigation 

**Title (ZH)**: 室内定位以实现自主机器人导航 

**Authors**: Sean Kouma, Rachel Masters  

**Link**: [PDF](https://arxiv.org/pdf/2502.20731)  

**Abstract**: Indoor positioning systems (IPSs) have gained attention as outdoor navigation becomes prevalent in everyday life. Research is being actively conducted on how indoor smartphone navigation can be accomplished and improved using received signal strength indication (RSSI) and machine learning (ML). IPSs have more use cases that need further exploration, and we aim to explore using IPSs for the indoor navigation of an autonomous robot. We collected a dataset and trained models to test on a robot. We also developed an A* path-planning algorithm so that our robot could navigate itself using predicted directions. After testing different network structures, our robot was able to successfully navigate corners around 50 percent of the time. The findings of this paper indicate that using IPSs for autonomous robots is a promising area of future research. 

**Abstract (ZH)**: Indoor定位系统（IPSs）随着户外导航在日常生活中的普及而引起了关注。研究人员正在积极探讨如何利用接收信号强度指示（RSSI）和机器学习（ML）来实现和改进室内的智能手机导航。IPSs具有更多的应用场景有待进一步探索，我们旨在利用IPSs实现自主机器人在室内的导航。我们收集了数据集并对模型进行了训练，以便在机器人上进行测试。我们还开发了A*路径规划算法，使我们的机器人能够根据预测的方向自主导航。经过测试不同的网络结构后，我们发现机器人能够成功导航拐角处的比例约为50%。本文的研究成果表明，利用IPSs进行自主机器人导航是一个充满前景的研究领域。 

---
# FSMP: A Frontier-Sampling-Mixed Planner for Fast Autonomous Exploration of Complex and Large 3-D Environments 

**Title (ZH)**: FSMP：一种前沿采样混合规划器，用于快速探索复杂大型3D环境 

**Authors**: Shiyong Zhang, Xuebo Zhang, Qianli Dong, Ziyu Wang, Haobo Xi, Jing Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.20707)  

**Abstract**: In this paper, we propose a systematic framework for fast exploration of complex and large 3-D environments using micro aerial vehicles (MAVs). The key insight is the organic integration of the frontier-based and sampling-based strategies that can achieve rapid global exploration of the environment. Specifically, a field-of-view-based (FOV) frontier detector with the guarantee of completeness and soundness is devised for identifying 3-D map frontiers. Different from random sampling-based methods, the deterministic sampling technique is employed to build and maintain an incremental road map based on the recorded sensor FOVs and newly detected frontiers. With the resulting road map, we propose a two-stage path planner. First, it quickly computes the global optimal exploration path on the road map using the lazy evaluation strategy. Then, the best exploration path is smoothed for further improving the exploration efficiency. We validate the proposed method both in simulation and real-world experiments. The comparative results demonstrate the promising performance of our planner in terms of exploration efficiency, computational time, and explored volume. 

**Abstract (ZH)**: 本文提出了一种用于快速探索复杂大型3D环境的微型 aerial车辆（MAVs）系统框架。 

---
# From Safety Standards to Safe Operation with Mobile Robotic Systems Deployment 

**Title (ZH)**: 从安全标准到移动机器人系统部署的安全运行 

**Authors**: Bruno Belzile, Tatiana Wanang-Siyapdjie, Sina Karimi, Rafael Gomes Braga, Ivanka Iordanova, David St-Onge  

**Link**: [PDF](https://arxiv.org/pdf/2502.20693)  

**Abstract**: Mobile robotic systems are increasingly used in various work environments to support productivity. However, deploying robots in workplaces crowded by human workers and interacting with them results in safety challenges and concerns, namely robot-worker collisions and worker distractions in hazardous environments. Moreover, the literature on risk assessment as well as the standard specific to mobile platforms is rather limited. In this context, this paper first conducts a review of the relevant standards and methodologies and then proposes a risk assessment for the safe deployment of mobile robots on construction sites. The approach extends relevant existing safety standards to encompass uncovered scenarios. Safety recommendations are made based on the framework, after its validation by field experts. 

**Abstract (ZH)**: 移动机器人系统在各种工作环境中的应用日益增多，以支持生产效率。然而，将机器人部署在拥挤的人工作业环境并与人交互时，会带来安全挑战和担忧，包括机器人与工人的碰撞以及在危险环境中工人注意力分散。此外，关于风险评估的文献以及针对移动平台的标准相当有限。在此背景下，本文首先回顾了相关标准和方法，并提出了一种适用于建筑工地的移动机器人安全部署的风险评估方法。该方法将现有相关安全标准扩展到涵盖未覆盖的场景，并基于该框架的现场专家验证提出安全建议。 

---
# Delayed-Decision Motion Planning in the Presence of Multiple Predictions 

**Title (ZH)**: 带有多个预测的延迟决策运动规划 

**Authors**: David Isele, Alexandre Miranda Anon, Faizan M. Tariq, Goro Yeh, Avinash Singh, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2502.20636)  

**Abstract**: Reliable automated driving technology is challenged by various sources of uncertainties, in particular, behavioral uncertainties of traffic agents. It is common for traffic agents to have intentions that are unknown to others, leaving an automated driving car to reason over multiple possible behaviors. This paper formalizes a behavior planning scheme in the presence of multiple possible futures with corresponding probabilities. We present a maximum entropy formulation and show how, under certain assumptions, this allows delayed decision-making to improve safety. The general formulation is then turned into a model predictive control formulation, which is solved as a quadratic program or a set of quadratic programs. We discuss implementation details for improving computation and verify operation in simulation and on a mobile robot. 

**Abstract (ZH)**: 可靠的自动驾驶技术受到多种不确定性挑战，特别是在交通代理行为不确定性方面的挑战。交通代理常常有他人未知的意图，使得自动驾驶车辆需要推理多种可能的行为。本文在存在多种可能未来的背景下形式化了一种行为规划方案，并提出了最大熵表述，展示了在某些假设下，这如何推迟决策以提高安全性。接着，将通用表述转换为模型预测控制表述，该表述可以作为二次规划或多组二次规划求解。讨论了改进计算的实现细节，并在仿真和移动机器人上验证了操作。 

---
# Subtask-Aware Visual Reward Learning from Segmented Demonstrations 

**Title (ZH)**: 面向子任务的分割演示视觉奖励学习 

**Authors**: Changyeon Kim, Minho Heo, Doohyun Lee, Jinwoo Shin, Honglak Lee, Joseph J. Lim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20630)  

**Abstract**: Reinforcement Learning (RL) agents have demonstrated their potential across various robotic tasks. However, they still heavily rely on human-engineered reward functions, requiring extensive trial-and-error and access to target behavior information, often unavailable in real-world settings. This paper introduces REDS: REward learning from Demonstration with Segmentations, a novel reward learning framework that leverages action-free videos with minimal supervision. Specifically, REDS employs video demonstrations segmented into subtasks from diverse sources and treats these segments as ground-truth rewards. We train a dense reward function conditioned on video segments and their corresponding subtasks to ensure alignment with ground-truth reward signals by minimizing the Equivalent-Policy Invariant Comparison distance. Additionally, we employ contrastive learning objectives to align video representations with subtasks, ensuring precise subtask inference during online interactions. Our experiments show that REDS significantly outperforms baseline methods on complex robotic manipulation tasks in Meta-World and more challenging real-world tasks, such as furniture assembly in FurnitureBench, with minimal human intervention. Moreover, REDS facilitates generalization to unseen tasks and robot embodiments, highlighting its potential for scalable deployment in diverse environments. 

**Abstract (ZH)**: 基于示例和分割的奖励学习框架：REDS 

---
# LV-DOT: LiDAR-visual dynamic obstacle detection and tracking for autonomous robot navigation 

**Title (ZH)**: 基于LiDAR-视觉的动态障碍物检测与跟踪方法在自主机器人导航中的应用 

**Authors**: Zhefan Xu, Haoyu Shen, Xinming Han, Hanyu Jin, Kanlong Ye, Kenji Shimada  

**Link**: [PDF](https://arxiv.org/pdf/2502.20607)  

**Abstract**: Accurate perception of dynamic obstacles is essential for autonomous robot navigation in indoor environments. Although sophisticated 3D object detection and tracking methods have been investigated and developed thoroughly in the fields of computer vision and autonomous driving, their demands on expensive and high-accuracy sensor setups and substantial computational resources from large neural networks make them unsuitable for indoor robotics. Recently, more lightweight perception algorithms leveraging onboard cameras or LiDAR sensors have emerged as promising alternatives. However, relying on a single sensor poses significant limitations: cameras have limited fields of view and can suffer from high noise, whereas LiDAR sensors operate at lower frequencies and lack the richness of visual features. To address this limitation, we propose a dynamic obstacle detection and tracking framework that uses both onboard camera and LiDAR data to enable lightweight and accurate perception. Our proposed method expands on our previous ensemble detection approach, which integrates outputs from multiple low-accuracy but computationally efficient detectors to ensure real-time performance on the onboard computer. In this work, we propose a more robust fusion strategy that integrates both LiDAR and visual data to enhance detection accuracy further. We then utilize a tracking module that adopts feature-based object association and the Kalman filter to track and estimate detected obstacles' states. Besides, a dynamic obstacle classification algorithm is designed to robustly identify moving objects. The dataset evaluation demonstrates a better perception performance compared to benchmark methods. The physical experiments on a quadcopter robot confirms the feasibility for real-world navigation. 

**Abstract (ZH)**: 基于车载摄像头和LiDAR数据的动态障碍检测与跟踪框架 

---
# Map Space Belief Prediction for Manipulation-Enhanced Mapping 

**Title (ZH)**: 基于操作增强制图的地图空间信念预测 

**Authors**: Joao Marcos Correia Marques, Nils Dengler, Tobias Zaenker, Jesper Mucke, Shenlong Wang, Maren Bennewitz, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2502.20606)  

**Abstract**: Searching for objects in cluttered environments requires selecting efficient viewpoints and manipulation actions to remove occlusions and reduce uncertainty in object locations, shapes, and categories. In this work, we address the problem of manipulation-enhanced semantic mapping, where a robot has to efficiently identify all objects in a cluttered shelf. Although Partially Observable Markov Decision Processes~(POMDPs) are standard for decision-making under uncertainty, representing unstructured interactive worlds remains challenging in this formalism. To tackle this, we define a POMDP whose belief is summarized by a metric-semantic grid map and propose a novel framework that uses neural networks to perform map-space belief updates to reason efficiently and simultaneously about object geometries, locations, categories, occlusions, and manipulation physics. Further, to enable accurate information gain analysis, the learned belief updates should maintain calibrated estimates of uncertainty. Therefore, we propose Calibrated Neural-Accelerated Belief Updates (CNABUs) to learn a belief propagation model that generalizes to novel scenarios and provides confidence-calibrated predictions for unknown areas. Our experiments show that our novel POMDP planner improves map completeness and accuracy over existing methods in challenging simulations and successfully transfers to real-world cluttered shelves in zero-shot fashion. 

**Abstract (ZH)**: 在杂乱环境中搜索物体需要选择高效的视角和操作动作以移除遮挡并降低物体位置、形状和类别不确定性的程度。在本文中，我们解决了操作增强语义映射问题，要求机器人高效地在杂乱货架上识别所有物体。尽管部分可观测马尔可夫决策过程（POMDPs）是不确定性决策的标准方法，但在这种形式isms下表示非结构化交互世界仍具挑战性。为了解决这个问题，我们定义了一个POMDP，其信念由度量语义网格图总结，并提出了一种新的框架，使用神经网络在地图空间中进行信念更新，从而高效且同时地推理物体几何形状、位置、类别、遮挡和操作物理。此外，为了实现准确的信息增益分析，所学习的信念更新应保持对不确定性的校准估计。因此，我们提出了一种校准神经加速信念更新（CNABUs）方法，学习一个能够泛化到新场景并为未知区域提供置信校准预测的信念传播模型。我们的实验表明，我们的新型POMDP规划者在具有挑战性的模拟中提高了地图的完整性和准确性，并以零样本方式成功转移到实际杂乱货架上。 

---
# Close-Proximity Satellite Operations through Deep Reinforcement Learning and Terrestrial Testing Environments 

**Title (ZH)**: 通过深度强化学习和陆基测试环境实现近距离卫星操作 

**Authors**: Henry Lei, Joshua Aurand, Zachary S. Lippay, Sean Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2502.20554)  

**Abstract**: With the increasingly congested and contested space environment, safe and effective satellite operation has become increasingly challenging. As a result, there is growing interest in autonomous satellite capabilities, with common machine learning techniques gaining attention for their potential to address complex decision-making in the space domain. However, the "black-box" nature of many of these methods results in difficulty understanding the model's input/output relationship and more specifically its sensitivity to environmental disturbances, sensor noise, and control intervention. This paper explores the use of Deep Reinforcement Learning (DRL) for satellite control in multi-agent inspection tasks. The Local Intelligent Network of Collaborative Satellites (LINCS) Lab is used to test the performance of these control algorithms across different environments, from simulations to real-world quadrotor UAV hardware, with a particular focus on understanding their behavior and potential degradation in performance when deployed beyond the training environment. 

**Abstract (ZH)**: 随着太空环境日益拥挤和竞争激烈，卫星安全有效运行面临更大挑战。因此，自主卫星能力的研究兴趣日益增长，其中常见的机器学习技术因其在空间域复杂决策方面的潜在应用而受到关注。然而，许多这些方法的“黑盒”性质使得难以理解模型的输入/输出关系及其对环境扰动、传感器噪声和控制干预的敏感性。本文探讨了在多智能体检查任务中使用深度强化学习（DRL）进行卫星控制的应用。通过Local Intelligent Network of Collaborative Satellites（LINCS）实验室，这些控制算法在从仿真到真实四旋翼无人机硬件的不同环境中进行了测试，特别是在部署到训练环境以外时对其行为和性能退化进行了重点研究。 

---
# Toward Fully Autonomous Flexible Chunk-Based Aerial Additive Manufacturing: Insights from Experimental Validation 

**Title (ZH)**: 面向完全自主柔性块状 aerial 融合制造的探究：来自实验验证的见解 

**Authors**: Marios-Nektarios Stamatopoulos, Jakub Haluska, Elias Small, Jude Marroush, Avijit Banerjee, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.20549)  

**Abstract**: A novel autonomous chunk-based aerial additive manufacturing framework is presented, supported with experimental demonstration advancing aerial 3D printing. An optimization-based decomposition algorithm transforms structures into sub-components, or chunks, treated as individual tasks coordinated via a dependency graph, ensuring sequential assignment to UAVs considering inter-dependencies and printability constraints for seamless execution. A specially designed hexacopter equipped with a pressurized canister for lightweight expandable foam extrusion is utilized to deposit the material in a controlled manner. To further enhance precise execution of the printing, an offset-free Model Predictive Control mechanism is considered compensating reactively for disturbances and ground effect during execution. Additionally, an interlocking mechanism is introduced in the chunking process to enhance structural cohesion and improve layer adhesion. Extensive experiments demonstrate the framework's effectiveness in constructing precise structures of various shapes while seamlessly adapting to practical challenges, proving its potential for a transformative leap in aerial robotic capability for autonomous construction. 

**Abstract (ZH)**: 一种基于分块的新型自主空中增材制造框架，结合实验演示推进空中3D打印。基于优化分解算法将结构分解为子组件或分块，作为个体任务通过依赖图协调，确保考虑相互依赖性和可打印性约束顺序分配给无人机，实现无缝执行。一种特别设计的六旋翼无人机配备有轻型可扩展泡沫挤出装置的压力罐，用于在受控状态下沉积材料。为了进一步提高打印执行的精确性，考虑了一种无偏移的模型预测控制机制，以在执行过程中反应性地补偿干扰和地面效应。此外，在分块过程中引入了一种互锁机制，以增强结构的凝聚力和提高层间粘附性。广泛的实验证明了该框架在构建各种形状的精确结构时能无缝适应实际挑战，证明了其在自主建筑领域为空中机器人能力带来的革命性飞跃。 

---
# Unified Feedback Linearization for Nonlinear Systems with Dexterous and Energy-Saving Modes 

**Title (ZH)**: 统一反馈线性化方法用于具有灵巧和节能模式的非线性系统 

**Authors**: Mirko Mizzoni, Pieter van Goor, Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20524)  

**Abstract**: Systems with a high number of inputs compared to the degrees of freedom (e.g. a mobile robot with Mecanum wheels) often have a minimal set of energy-efficient inputs needed to achieve a main task (e.g. position tracking) and a set of energy-intense inputs needed to achieve an additional auxiliary task (e.g. orientation tracking). This letter presents a unified control scheme, derived through feedback linearization, that can switch between two modes: an energy-saving mode, which tracks the main task using only the energy-efficient inputs while forcing the energy-intense inputs to zero, and a dexterous mode, which also uses the energy-intense inputs to track the auxiliary task as needed. The proposed control guarantees the exponential tracking of the main task and that the dynamics associated with the main task evolve independently of the a priori unknown switching signal. When the control is operating in dexterous mode, the exponential tracking of the auxiliary task is also guaranteed. Numerical simulations on an omnidirectional Mecanum wheel robot validate the effectiveness of the proposed approach and demonstrate the effect of the switching signal on the exponential tracking behavior of the main and auxiliary tasks. 

**Abstract (ZH)**: 具有大量输入相对于自由度而言的系统（例如配有Mecanum轮的移动机器人）通常仅需要一组能量高效输入来完成主要任务（例如位置跟踪），同时还需要一组能量密集输入来完成辅助任务（例如定向跟踪）。本通信提出了一种通过反馈线性化导出的统一控制方案，该方案可以在两种模式之间切换：节能模式下仅使用能量高效输入跟踪主要任务，并强制能量密集输入为零；灵巧模式下利用能量密集输入根据需要跟踪辅助任务。所提出的控制方案保证了主要任务的指数级跟踪，并确保与主要任务相关的动态演化与先验未知的切换信号独立。当控制处于灵巧模式时，副任务的指数级跟踪也被保证。数值仿真验证了所提出方法的有效性，并展示了切换信号对主要任务和辅助任务指数级跟踪行为的影响。 

---
# Equivariant Reinforcement Learning Frameworks for Quadrotor Low-Level Control 

**Title (ZH)**: 四旋翼低级控制的 equivariant 强化学习框架 

**Authors**: Beomyeol Yu, Taeyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20500)  

**Abstract**: Improving sampling efficiency and generalization capability is critical for the successful data-driven control of quadrotor unmanned aerial vehicles (UAVs) that are inherently unstable. While various reinforcement learning (RL) approaches have been applied to autonomous quadrotor flight, they often require extensive training data, posing multiple challenges and safety risks in practice. To address these issues, we propose data-efficient, equivariant monolithic and modular RL frameworks for quadrotor low-level control. Specifically, by identifying the rotational and reflectional symmetries in quadrotor dynamics and encoding these symmetries into equivariant network models, we remove redundancies of learning in the state-action space. This approach enables the optimal control action learned in one configuration to automatically generalize into other configurations via symmetry, thereby enhancing data efficiency. Experimental results demonstrate that our equivariant approaches significantly outperform their non-equivariant counterparts in terms of learning efficiency and flight performance. 

**Abstract (ZH)**: 提高采样效率和泛化能力是成功实现固有不稳定的四旋翼无人机数据驱动控制的关键。虽然各种强化学习（RL）方法已被应用于自主四旋翼飞行，但它们往往需要大量的训练数据，实践中这会带来多个挑战和安全风险。为解决这些问题，我们提出了四旋翼低级控制的数据高效且共变的统一和模块化RL框架。具体而言，通过识别四旋翼动力学中的旋转和反射对称性，并将这些对称性编码到共变网络模型中，我们消除了状态-动作空间中的冗余学习。这种方法通过对称性使得在一个配置中学习到的最优控制动作能够自动泛化到其他配置，从而提高数据效率。实验结果表明，我们的共变方法在学习效率和飞行性能方面明显优于非共变方法。 

---
# Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction 

**Title (ZH)**: 通过学习时间抽象实现 stochastic 环境下的可扩展决策制定 

**Authors**: Baiting Luo, Ava Pettet, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2502.21186)  

**Abstract**: Sequential decision-making in high-dimensional continuous action spaces, particularly in stochastic environments, faces significant computational challenges. We explore this challenge in the traditional offline RL setting, where an agent must learn how to make decisions based on data collected through a stochastic behavior policy. We present \textit{Latent Macro Action Planner} (L-MAP), which addresses this challenge by learning a set of temporally extended macro-actions through a state-conditional Vector Quantized Variational Autoencoder (VQ-VAE), effectively reducing action dimensionality. L-MAP employs a (separate) learned prior model that acts as a latent transition model and allows efficient sampling of plausible actions. During planning, our approach accounts for stochasticity in both the environment and the behavior policy by using Monte Carlo tree search (MCTS). In offline RL settings, including stochastic continuous control tasks, L-MAP efficiently searches over discrete latent actions to yield high expected returns. Empirical results demonstrate that L-MAP maintains low decision latency despite increased action dimensionality. Notably, across tasks ranging from continuous control with inherently stochastic dynamics to high-dimensional robotic hand manipulation, L-MAP significantly outperforms existing model-based methods and performs on-par with strong model-free actor-critic baselines, highlighting the effectiveness of the proposed approach in planning in complex and stochastic environments with high-dimensional action spaces. 

**Abstract (ZH)**: 高维连续动作空间下顺序决策在随机环境中的挑战及其解决方法：基于潜在宏动作规划器（L-MAP）的方法 

---
# EDENet: Echo Direction Encoding Network for Place Recognition Based on Ground Penetrating Radar 

**Title (ZH)**: EDENet：基于地面穿透雷达的回声方向编码网络在位置识别中的应用 

**Authors**: Pengyu Zhang, Xieyuanli Chen, Yuwei Chen, Beizhen Bi, Zhuo Xu, Tian Jin, Xiaotao Huang, Liang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20643)  

**Abstract**: Ground penetrating radar (GPR) based localization has gained significant recognition in robotics due to its ability to detect stable subsurface features, offering advantages in environments where traditional sensors like cameras and LiDAR may struggle. However, existing methods are primarily focused on small-scale place recognition (PR), leaving the challenges of PR in large-scale maps unaddressed. These challenges include the inherent sparsity of underground features and the variability in underground dielectric constants, which complicate robust localization. In this work, we investigate the geometric relationship between GPR echo sequences and underground scenes, leveraging the robustness of directional features to inform our network design. We introduce learnable Gabor filters for the precise extraction of directional responses, coupled with a direction-aware attention mechanism for effective geometric encoding. To further enhance performance, we incorporate a shift-invariant unit and a multi-scale aggregation strategy to better accommodate variations in di-electric constants. Experiments conducted on public datasets demonstrate that our proposed EDENet not only surpasses existing solutions in terms of PR performance but also offers advantages in model size and computational efficiency. 

**Abstract (ZH)**: 基于地面穿透雷达（GPR）的定位在机器人领域由于其能够检测稳定的地下特征，而在传统传感器如相机和LiDAR表现不佳的环境中展现出优势，获得了显著的认可。然而，现有的方法主要集中在小规模场所识别（PR）上，大型规模地图的PR挑战未得到充分解决。这些挑战包括地下特征的固有稀疏性和地下介电常数的变异性，这给稳健定位带来了复杂性。在本工作中，我们研究了GPR回波序列与地下场景之间的几何关系，利用方向特征的鲁棒性指导网络设计。我们引入可学习的Gabor滤波器来精确提取方向响应，并结合方向感知的注意机制进行有效的几何编码。为了进一步提升性能，我们加入了移不变单元和多尺度聚合策略，更好地适应介电常数的变化。在公开数据集上的实验表明，我们提出的EDENet不仅在PR性能上优于现有解决方案，还在模型大小和计算效率方面具有优势。 

---
