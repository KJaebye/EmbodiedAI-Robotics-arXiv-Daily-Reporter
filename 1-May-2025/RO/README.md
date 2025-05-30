# Neuro-Symbolic Generation of Explanations for Robot Policies with Weighted Signal Temporal Logic 

**Title (ZH)**: 基于加权信号时序逻辑的神经符号机器人策略解释生成 

**Authors**: Mikihisa Yuasa, Ramavarapu S. Sreenivas, Huy T. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2504.21841)  

**Abstract**: Neural network-based policies have demonstrated success in many robotic applications, but often lack human-explanability, which poses challenges in safety-critical deployments. To address this, we propose a neuro-symbolic explanation framework that generates a weighted signal temporal logic (wSTL) specification to describe a robot policy in a interpretable form. Existing methods typically produce explanations that are verbose and inconsistent, which hinders explainability, and loose, which do not give meaningful insights into the underlying policy. We address these issues by introducing a simplification process consisting of predicate filtering, regularization, and iterative pruning. We also introduce three novel explainability evaluation metrics -- conciseness, consistency, and strictness -- to assess explanation quality beyond conventional classification metrics. Our method is validated in three simulated robotic environments, where it outperforms baselines in generating concise, consistent, and strict wSTL explanations without sacrificing classification accuracy. This work bridges policy learning with formal methods, contributing to safer and more transparent decision-making in robotics. 

**Abstract (ZH)**: 基于神经网络的政策在许多机器人应用中取得了成功，但常常缺乏人类可解释性，这在安全关键部署中提出了挑战。为应对这一问题，我们提出了一种神经符号解释框架，生成加权信号时序逻辑（wSTL）规范，以一种可解释的形式描述机器人政策。现有方法通常生成冗长且不一致的解释，妨碍了可解释性，并且宽松的解释无法提供有关潜在政策的有意义见解。我们通过引入包括谓词过滤、正则化和迭代剪枝的简化过程来解决这些问题。我们还引入了三种新的可解释性评估指标——简洁性、一致性和严格性——以超越传统分类指标来评估解释质量。该方法在三个模拟机器人环境中得到了验证，能够在不牺牲分类准确性的情况下生成简洁、一致和严格的wSTL解释，从而将策略学习与形式方法相结合，为机器人中的更安全和透明决策做出了贡献。 

---
# An Underwater, Fault-Tolerant, Laser-Aided Robotic Multi-Modal Dense SLAM System for Continuous Underwater In-Situ Observation 

**Title (ZH)**: 一种基于水下激光辅助的容错多模态密集SLAM系统，实现连续原位水下观测 

**Authors**: Yaming Ou, Junfeng Fan, Chao Zhou, Pengju Zhang, Zongyuan Shen, Yichen Fu, Xiaoyan Liu, Zengguang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21826)  

**Abstract**: Existing underwater SLAM systems are difficult to work effectively in texture-sparse and geometrically degraded underwater environments, resulting in intermittent tracking and sparse mapping. Therefore, we present Water-DSLAM, a novel laser-aided multi-sensor fusion system that can achieve uninterrupted, fault-tolerant dense SLAM capable of continuous in-situ observation in diverse complex underwater scenarios through three key innovations: Firstly, we develop Water-Scanner, a multi-sensor fusion robotic platform featuring a self-designed Underwater Binocular Structured Light (UBSL) module that enables high-precision 3D perception. Secondly, we propose a fault-tolerant triple-subsystem architecture combining: 1) DP-INS (DVL- and Pressure-aided Inertial Navigation System): fusing inertial measurement unit, doppler velocity log, and pressure sensor based Error-State Kalman Filter (ESKF) to provide high-frequency absolute odometry 2) Water-UBSL: a novel Iterated ESKF (IESKF)-based tight coupling between UBSL and DP-INS to mitigate UBSL's degeneration issues 3) Water-Stereo: a fusion of DP-INS and stereo camera for accurate initialization and tracking. Thirdly, we introduce a multi-modal factor graph back-end that dynamically fuses heterogeneous sensor data. The proposed multi-sensor factor graph maintenance strategy efficiently addresses issues caused by asynchronous sensor frequencies and partial data loss. Experimental results demonstrate Water-DSLAM achieves superior robustness (0.039 m trajectory RMSE and 100\% continuity ratio during partial sensor dropout) and dense mapping (6922.4 points/m^3 in 750 m^3 water volume, approximately 10 times denser than existing methods) in various challenging environments, including pools, dark underwater scenes, 16-meter-deep sinkholes, and field rivers. Our project is available at this https URL. 

**Abstract (ZH)**: 水下SLAM系统Water-DSLAM：一种适用于复杂水下场景的多传感器融合系统 

---
# LLM-based Interactive Imitation Learning for Robotic Manipulation 

**Title (ZH)**: 基于LLM的交互式模仿学习在机器人操作中的应用 

**Authors**: Jonas Werner, Kun Chu, Cornelius Weber, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2504.21769)  

**Abstract**: Recent advancements in machine learning provide methods to train autonomous agents capable of handling the increasing complexity of sequential decision-making in robotics. Imitation Learning (IL) is a prominent approach, where agents learn to control robots based on human demonstrations. However, IL commonly suffers from violating the independent and identically distributed (i.i.d) assumption in robotic tasks. Interactive Imitation Learning (IIL) achieves improved performance by allowing agents to learn from interactive feedback from human teachers. Despite these improvements, both approaches come with significant costs due to the necessity of human involvement. Leveraging the emergent capabilities of Large Language Models (LLMs) in reasoning and generating human-like responses, we introduce LLM-iTeach -- a novel IIL framework that utilizes an LLM as an interactive teacher to enhance agent performance while alleviating the dependence on human resources. Firstly, LLM-iTeach uses a hierarchical prompting strategy that guides the LLM in generating a policy in Python code. Then, with a designed similarity-based feedback mechanism, LLM-iTeach provides corrective and evaluative feedback interactively during the agent's training. We evaluate LLM-iTeach against baseline methods such as Behavior Cloning (BC), an IL method, and CEILing, a state-of-the-art IIL method using a human teacher, on various robotic manipulation tasks. Our results demonstrate that LLM-iTeach surpasses BC in the success rate and achieves or even outscores that of CEILing, highlighting the potential of LLMs as cost-effective, human-like teachers in interactive learning environments. We further demonstrate the method's potential for generalization by evaluating it on additional tasks. The code and prompts are provided at: this https URL. 

**Abstract (ZH)**: 新兴的机器学习技术为训练能够处理机器人领域日益复杂的序列决策的自主代理提供了方法。模仿学习（IL）是一种突出的方法，其中代理基于人类示范学习控制机器人。然而，IL 在机器人任务中通常会违反独立且同分布（i.i.d）的假设。互动模仿学习（IIL）通过允许代理从人类教师的互动反馈中学习来实现性能的提升。尽管如此，这两种方法都因为需要人类的参与而存在较大的成本。利用大型语言模型（LLMs）在推理和生成类似人类回应方面出现的能力，我们引入了 LLM-iTeach ——一种新颖的 IIL 框架，该框架利用 LLM 作为互动教师以增强代理性能并减轻对人力资源的依赖。首先，LLM-iTeach 使用分级提示策略来引导 LLM 生成 Python 代码策略。然后，通过设计的一种基于相似性的反馈机制，LLM-iTeach 在代理训练过程中提供纠正性和评价性反馈。我们在各种机器人操作任务上将 LLM-iTeach 与基线方法行为克隆（BC）、一种 IL 方法以及 CEILing、一种最先进的 IIL 方法（使用人类教师）进行了比较评估。结果显示，LLM-iTeach 在成功率上优于 BC，并且在某些情况下甚至优于 CEILing，突显了 LLM 作为互动学习环境中成本效益高且类似人类的教师的潜力。我们进一步通过额外任务评估展示了该方法的泛化能力。相关代码和提示可在以下链接获取：this https URL。 

---
# Whleaper: A 10-DOF Flexible Bipedal Wheeled Robot 

**Title (ZH)**: Whleaper: 一种10自由度柔性 bipedal 轮式机器人 

**Authors**: Yinglei Zhu, Sixiao He, Zhenghao Qi, Zhuoyuan Yong, Yihua Qin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21767)  

**Abstract**: Wheel-legged robots combine the advantages of both wheeled robots and legged robots, offering versatile locomotion capabilities with excellent stability on challenging terrains and high efficiency on flat surfaces. However, existing wheel-legged robots typically have limited hip joint mobility compared to humans, while hip joint plays a crucial role in locomotion. In this paper, we introduce Whleaper, a novel 10-degree-of-freedom (DOF) bipedal wheeled robot, with 3 DOFs at the hip of each leg. Its humanoid joint design enables adaptable motion in complex scenarios, ensuring stability and flexibility. This paper introduces the details of Whleaper, with a focus on innovative mechanical design, control algorithms and system implementation. Firstly, stability stems from the increased DOFs at the hip, which expand the range of possible postures and improve the robot's foot-ground contact. Secondly, the extra DOFs also augment its mobility. During walking or sliding, more complex movements can be adopted to execute obstacle avoidance tasks. Thirdly, we utilize two control algorithms to implement multimodal motion for walking and sliding. By controlling specific DOFs of the robot, we conducted a series of simulations and practical experiments, demonstrating that a high-DOF hip joint design can effectively enhance the stability and flexibility of wheel-legged robots. Whleaper shows its capability to perform actions such as squatting, obstacle avoidance sliding, and rapid turning in real-world scenarios. 

**Abstract (ZH)**: 轮腿机器人结合了轮式机器人和腿式机器人各自的优点，提供在挑战性地形中出色的稳定性和在平坦表面上的高效率的多功能移动能力。然而，现有的轮腿机器人相比人类通常具有有限的髋关节 Mobility，而髋关节在移动中起着至关重要的作用。本文介绍了Whleaper，一种新型10自由度（DOF）双足轮式机器人，每条腿具有3个髋关节自由度。其类人关节设计使其能够在复杂场景中适应运动，确保稳定性和灵活性。本文详细介绍Whleaper，重点介绍了其创新机械设计、控制算法和系统实现。首先，稳定性来自于髋关节处增加的自由度，这扩展了可能的姿态范围并提高了机器人足底着地接触的质量。其次，额外的自由度也增强了其机动性。在行走或滑行时，可以采用更复杂的动作执行避障任务。第三，我们利用两种控制算法实现了行走和滑行的多模态运动。通过控制机器人特定的自由度，我们进行了系列仿真和实验证明，高自由度髋关节设计可以有效地增强轮腿机器人的稳定性和灵活性。Whleaper展示了其在真实场景中执行蹲伏、避障滑行和快速转弯等动作的能力。 

---
# LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning 

**Title (ZH)**: LangWBC: 通过端到端学习的语言导向 humanoid 全身控制 

**Authors**: Yiyang Shao, Xiaoyu Huang, Bike Zhang, Qiayuan Liao, Yuman Gao, Yufeng Chi, Zhongyu Li, Sophia Shao, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2504.21738)  

**Abstract**: General-purpose humanoid robots are expected to interact intuitively with humans, enabling seamless integration into daily life. Natural language provides the most accessible medium for this purpose. However, translating language into humanoid whole-body motion remains a significant challenge, primarily due to the gap between linguistic understanding and physical actions. In this work, we present an end-to-end, language-directed policy for real-world humanoid whole-body control. Our approach combines reinforcement learning with policy distillation, allowing a single neural network to interpret language commands and execute corresponding physical actions directly. To enhance motion diversity and compositionality, we incorporate a Conditional Variational Autoencoder (CVAE) structure. The resulting policy achieves agile and versatile whole-body behaviors conditioned on language inputs, with smooth transitions between various motions, enabling adaptation to linguistic variations and the emergence of novel motions. We validate the efficacy and generalizability of our method through extensive simulations and real-world experiments, demonstrating robust whole-body control. Please see our website at this http URL for more information. 

**Abstract (ZH)**: 通用 humanoid 机器人期望能够直观地与人类交互，从而实现无缝融入日常生活。自然语言是最具可访问性的媒介。然而，将语言翻译成类人全身动作仍然是一项显著的挑战，主要是由于语言理解和物理动作之间的差距。在本文中，我们提出了一种端到端、基于语言的政策，用于现实中的类人全身控制。我们的方法结合了强化学习与策略蒸馏，使得单个神经网络能够直接解释语言命令并执行相应的物理动作。为了增强动作的多样性和组合性，我们引入了一个条件变分自编码器（CVAE）结构。最终的政策能够在语言输入的条件下实现灵活且多功能的整体动作行为，具有平滑的动作过渡，能够适应语言变化并产生新的动作。我们通过广泛的仿真实验和现实世界实验验证了该方法的有效性和普适性，展示了稳健的整体动作控制能力。更多信息请参见我们的网站：this http URL。 

---
# LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics 

**Title (ZH)**: 基于LLM的强大记忆增强任务规划家庭机器人代理 

**Authors**: Marc Glocker, Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.21716)  

**Abstract**: We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2-Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: this https URL. 

**Abstract (ZH)**: 我们提出一种受-bodied机器人系统，采用由大型语言模型驱动的代理 orchestration 架构，实现自主家庭物品管理。该系统集成了记忆增强的任务规划，使机器人能够执行高级用户命令并追踪过往操作。系统采用三个专门的代理：路由代理、任务规划代理和知识库代理，每个代理由任务特定的大语言模型驱动。通过利用上下文学习，我们的系统避免了显式模型训练的需要。基于关联检索（RAG）的能力使系统能够从过往交互中检索上下文，从而改善长期对象跟踪。结合使用 Grounded SAM 和 LLaMa3.2-Vision 提供了稳健的对象检测，有助于任务规划中的语义场景理解。跨三种家庭场景的评估展示了高任务规划准确性，并且由于 RAG，记忆力召回率得到提升。具体而言，Qwen2.5 在专门代理中表现出最佳性能，而 LLaMA3.1 在路由任务中表现优异。源代码可在以下链接获取：this https URL。 

---
# Self-Supervised Monocular Visual Drone Model Identification through Improved Occlusion Handling 

**Title (ZH)**: 通过改进遮挡处理的自监督单目视觉无人驾驶车辆模型识别 

**Authors**: Stavrow A. Bahnam, Christophe De Wagter, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21695)  

**Abstract**: Ego-motion estimation is vital for drones when flying in GPS-denied environments. Vision-based methods struggle when flight speed increases and close-by objects lead to difficult visual conditions with considerable motion blur and large occlusions. To tackle this, vision is typically complemented by state estimation filters that combine a drone model with inertial measurements. However, these drone models are currently learned in a supervised manner with ground-truth data from external motion capture systems, limiting scalability to different environments and drones. In this work, we propose a self-supervised learning scheme to train a neural-network-based drone model using only onboard monocular video and flight controller data (IMU and motor feedback). We achieve this by first training a self-supervised relative pose estimation model, which then serves as a teacher for the drone model. To allow this to work at high speed close to obstacles, we propose an improved occlusion handling method for training self-supervised pose estimation models. Due to this method, the root mean squared error of resulting odometry estimates is reduced by an average of 15%. Moreover, the student neural drone model can be successfully obtained from the onboard data. It even becomes more accurate at higher speeds compared to its teacher, the self-supervised vision-based model. We demonstrate the value of the neural drone model by integrating it into a traditional filter-based VIO system (ROVIO), resulting in superior odometry accuracy on aggressive 3D racing trajectories near obstacles. Self-supervised learning of ego-motion estimation represents a significant step toward bridging the gap between flying in controlled, expensive lab environments and real-world drone applications. The fusion of vision and drone models will enable higher-speed flight and improve state estimation, on any drone in any environment. 

**Abstract (ZH)**: 基于自我监督学习的无人机 ego-运动估计 

---
# Path Planning on Multi-level Point Cloud with a Weighted Traversability Graph 

**Title (ZH)**: 多级点云权重通过性图路径规划 

**Authors**: Yujie Tang, Quan Li, Hao Geng, Yangmin Xie, Hang Shi, Yusheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21622)  

**Abstract**: This article proposes a new path planning method for addressing multi-level terrain situations. The proposed method includes innovations in three aspects: 1) the pre-processing of point cloud maps with a multi-level skip-list structure and data-slimming algorithm for well-organized and simplified map formalization and management, 2) the direct acquisition of local traversability indexes through vehicle and point cloud interaction analysis, which saves work in surface fitting, and 3) the assignment of traversability indexes on a multi-level connectivity graph to generate a weighted traversability graph for generally search-based path planning. The A* algorithm is modified to utilize the traversability graph to generate a short and safe path. The effectiveness and reliability of the proposed method are verified through indoor and outdoor experiments conducted in various environments, including multi-floor buildings, woodland, and rugged mountainous regions. The results demonstrate that the proposed method can properly address 3D path planning problems for ground vehicles in a wide range of situations. 

**Abstract (ZH)**: 本文提出了一种应对多级地形情况的新型路径规划方法。该方法在三个方面进行了创新：1) 使用多级跳跃列表结构和数据瘦身算法对点云地图进行预处理，以实现地图的有序化和简化管理，2) 通过车辆与点云的交互分析直接获取局部可通行性指标，省去了表面拟合的工作，3) 在多级连接图上分配可通行性指标以生成加权可通行性图，用于基于搜索的路径规划。对A*算法进行了修改，使其能够利用可通行性图生成短且安全的路径。通过在多种环境，包括多层建筑物、林地和崎岖的山区等地进行室内外实验，验证了所提出方法的有效性和可靠性。该研究成果表明，所提出的方法能够妥善解决地面车辆在多种情况下的三维路径规划问题。 

---
# LRBO2: Improved 3D Vision Based Hand-Eye Calibration for Collaborative Robot Arm 

**Title (ZH)**: LRBO2: 基于改进的3D视觉的手眼标定方法用于协作机器人臂 

**Authors**: Leihui Li, Lixuepiao Wan, Volker Krueger, Xuping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21619)  

**Abstract**: Hand-eye calibration is a common problem in the field of collaborative robotics, involving the determination of the transformation matrix between the visual sensor and the robot flange to enable vision-based robotic tasks. However, this process typically requires multiple movements of the robot arm and an external calibration object, making it both time-consuming and inconvenient, especially in scenarios where frequent recalibration is necessary. In this work, we extend our previous method, Look at Robot Base Once (LRBO), which eliminates the need for external calibration objects such as a chessboard. We propose a generic dataset generation approach for point cloud registration, focusing on aligning the robot base point cloud with the scanned data. Furthermore, a more detailed simulation study is conducted involving several different collaborative robot arms, followed by real-world experiments in an industrial setting. Our improved method is simulated and evaluated using a total of 14 robotic arms from 9 different brands, including KUKA, Universal Robots, UFACTORY, and Franka Emika, all of which are widely used in the field of collaborative robotics. Physical experiments demonstrate that our extended approach achieves performance comparable to existing commercial hand-eye calibration solutions, while completing the entire calibration procedure in just a few seconds. In addition, we provide a user-friendly hand-eye calibration solution, with the code publicly available at this http URL. 

**Abstract (ZH)**: 一次查看机器人基座（LRBO）方法的扩展：用于协作机器人的眼手标定 

---
# Real Time Semantic Segmentation of High Resolution Automotive LiDAR Scans 

**Title (ZH)**: 实时高分辨率汽车LiDAR扫描语义分割 

**Authors**: Hannes Reichert, Benjamin Serfling, Elijah Schüssler, Kerim Turacan, Konrad Doll, Bernhard Sick  

**Link**: [PDF](https://arxiv.org/pdf/2504.21602)  

**Abstract**: In recent studies, numerous previous works emphasize the importance of semantic segmentation of LiDAR data as a critical component to the development of driver-assistance systems and autonomous vehicles. However, many state-of-the-art methods are tested on outdated, lower-resolution LiDAR sensors and struggle with real-time constraints. This study introduces a novel semantic segmentation framework tailored for modern high-resolution LiDAR sensors that addresses both accuracy and real-time processing demands. We propose a novel LiDAR dataset collected by a cutting-edge automotive 128 layer LiDAR in urban traffic scenes. Furthermore, we propose a semantic segmentation method utilizing surface normals as strong input features. Our approach is bridging the gap between cutting-edge research and practical automotive applications. Additionaly, we provide a Robot Operating System (ROS2) implementation that we operate on our research vehicle. Our dataset and code are publicly available: this https URL. 

**Abstract (ZH)**: 近年来，许多先前研究强调了LiDAR数据语义分割在驾驶员辅助系统和自动驾驶车辆开发中的重要性，是其关键组成部分。然而，许多最先进的方法是在过时且分辨率较低的LiDAR传感器上进行测试，并且难以满足实时约束。本研究提出了一种针对现代高分辨率LiDAR传感器的新型语义分割框架，旨在同时满足准确性和实时处理需求。我们提出了一种新型LiDAR数据集，该数据集由先进的汽车128层LiDAR在城市交通场景中收集。此外，我们提出了一种利用表面法线作为强大输入特征的语义分割方法。我们的方法填补了尖端研究与实际汽车应用之间的差距。另外，我们提供了在研究车辆上运行的基于Robot Operating System (ROS2)的实现。我们的数据集和代码已公开提供：[此链接]。 

---
# Leveraging Pre-trained Large Language Models with Refined Prompting for Online Task and Motion Planning 

**Title (ZH)**: 利用精炼提示调优预先训练的大语言模型进行在线任务与运动规划 

**Authors**: Huihui Guo, Huilong Pi, Yunchuan Qin, Zhuo Tang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21596)  

**Abstract**: With the rapid advancement of artificial intelligence, there is an increasing demand for intelligent robots capable of assisting humans in daily tasks and performing complex operations. Such robots not only require task planning capabilities but must also execute tasks with stability and robustness. In this paper, we present a closed-loop task planning and acting system, LLM-PAS, which is assisted by a pre-trained Large Language Model (LLM). While LLM-PAS plans long-horizon tasks in a manner similar to traditional task and motion planners, it also emphasizes the execution phase of the task. By transferring part of the constraint-checking process from the planning phase to the execution phase, LLM-PAS enables exploration of the constraint space and delivers more accurate feedback on environmental anomalies during execution. The reasoning capabilities of the LLM allow it to handle anomalies that cannot be addressed by the robust executor. To further enhance the system's ability to assist the planner during replanning, we propose the First Look Prompting (FLP) method, which induces LLM to generate effective PDDL goals. Through comparative prompting experiments and systematic experiments, we demonstrate the effectiveness and robustness of LLM-PAS in handling anomalous conditions during task execution. 

**Abstract (ZH)**: 基于预训练大语言模型的闭环任务规划与执行系统：结合首次观察提示方法处理执行过程中的异常条件 

---
# One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms 

**Title (ZH)**: 一网统管：跨不同平台的四旋翼飞行器比赛中的领域随机化 

**Authors**: Robin Ferede, Till Blaha, Erin Lucassen, Christophe De Wagter, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21586)  

**Abstract**: In high-speed quadcopter racing, finding a single controller that works well across different platforms remains challenging. This work presents the first neural network controller for drone racing that generalizes across physically distinct quadcopters. We demonstrate that a single network, trained with domain randomization, can robustly control various types of quadcopters. The network relies solely on the current state to directly compute motor commands. The effectiveness of this generalized controller is validated through real-world tests on two substantially different crafts (3-inch and 5-inch race quadcopters). We further compare the performance of this generalized controller with controllers specifically trained for the 3-inch and 5-inch drone, using their identified model parameters with varying levels of domain randomization (0%, 10%, 20%, 30%). While the generalized controller shows slightly slower speeds compared to the fine-tuned models, it excels in adaptability across different platforms. Our results show that no randomization fails sim-to-real transfer while increasing randomization improves robustness but reduces speed. Despite this trade-off, our findings highlight the potential of domain randomization for generalizing controllers, paving the way for universal AI controllers that can adapt to any platform. 

**Abstract (ZH)**: 在高速四旋翼竞速中，找到一款适合不同平台的单一控制器仍具有挑战性。本文提出了首个能够在物理上不同的四旋翼无人机间通用的神经网络控制器。我们证明，通过领域随机化训练的单个网络能够稳健地控制不同类型四旋翼无人机。该网络仅依赖当前状态直接计算电机命令。我们通过在两个明显不同的机型（3英寸和5英寸竞速四旋翼无人机）上的实际测试验证了这一通用控制器的有效性。我们还将此通用控制器的性能与专门针对3英寸和5英寸无人机训练的控制器进行了比较，后者使用不同水平的领域随机化（0%，10%，20%，30%）确定的模型参数。虽然通用控制器的速度略慢于微调模型，但它在不同平台上的适应性更佳。我们的结果显示，没有随机化会导致从仿真实验到现实世界的过渡失败，而增加随机化则提高了鲁棒性但降低了速度。尽管存在这种权衡，我们的研究结果突显了领域随机化在控制器通用化方面的潜力，为能够适应任何平台的通用人工智能控制器铺平了道路。 

---
# Multi-Goal Dexterous Hand Manipulation using Probabilistic Model-based Reinforcement Learning 

**Title (ZH)**: 基于概率模型的强化学习在多目标灵巧手操作中的应用 

**Authors**: Yingzhuo Jiang, Wenjun Huang, Rongdun Lin, Chenyang Miao, Tianfu Sun, Yunduan Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.21585)  

**Abstract**: This paper tackles the challenge of learning multi-goal dexterous hand manipulation tasks using model-based Reinforcement Learning. We propose Goal-Conditioned Probabilistic Model Predictive Control (GC-PMPC) by designing probabilistic neural network ensembles to describe the high-dimensional dexterous hand dynamics and introducing an asynchronous MPC policy to meet the control frequency requirements in real-world dexterous hand systems. Extensive evaluations on four simulated Shadow Hand manipulation scenarios with randomly generated goals demonstrate GC-PMPC's superior performance over state-of-the-art baselines. It successfully drives a cable-driven Dexterous hand, DexHand 021 with 12 Active DOFs and 5 tactile sensors, to learn manipulating a cubic die to three goal poses within approximately 80 minutes of interactions, demonstrating exceptional learning efficiency and control performance on a cost-effective dexterous hand platform. 

**Abstract (ZH)**: 基于模型的强化学习中多目标灵巧手操作任务的学习：目标条件概率模型预测控制 

---
# RoboGround: Robotic Manipulation with Grounded Vision-Language Priors 

**Title (ZH)**: RoboGround: 基于地面视觉-语言先验的机器人操作 

**Authors**: Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan Wang, Tai Wang, Jiangmiao Pang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.21530)  

**Abstract**: Recent advancements in robotic manipulation have highlighted the potential of intermediate representations for improving policy generalization. In this work, we explore grounding masks as an effective intermediate representation, balancing two key advantages: (1) effective spatial guidance that specifies target objects and placement areas while also conveying information about object shape and size, and (2) broad generalization potential driven by large-scale vision-language models pretrained on diverse grounding datasets. We introduce RoboGround, a grounding-aware robotic manipulation system that leverages grounding masks as an intermediate representation to guide policy networks in object manipulation tasks. To further explore and enhance generalization, we propose an automated pipeline for generating large-scale, simulated data with a diverse set of objects and instructions. Extensive experiments show the value of our dataset and the effectiveness of grounding masks as intermediate guidance, significantly enhancing the generalization abilities of robot policies. 

**Abstract (ZH)**: 近期机器人操作领域的进展突显了中间表示在提高策略泛化能力方面的潜力。在本文中，我们探索基于掩码的中间表示作为有效的方案，结合了两项关键优势：（1）有效的空间指导，能够指定目标对象及其放置区域，并传达对象的形状和大小信息；（2）由大规模预训练的视觉-语言模型驱动的广泛泛化能力，这些模型基于多样化的基础现实数据集。我们提出了RoboGround，这是一种基于掩码的机器人操作系统，利用掩码作为中间表示来指导物体操作任务中的策略网络。为了进一步探索和增强泛化能力，我们提出了一种自动生成多样对象和指令的大规模模拟数据的自动化管道。广泛的实验表明，我们数据集的价值以及掩码作为中间指导的有效性，显著提升了机器人策略的泛化能力。 

---
# Provably-Safe, Online System Identification 

**Title (ZH)**: 可证明安全的在线系统辨识 

**Authors**: Bohao Zhang, Zichang Zhou, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21486)  

**Abstract**: Precise manipulation tasks require accurate knowledge of payload inertial parameters. Unfortunately, identifying these parameters for unknown payloads while ensuring that the robotic system satisfies its input and state constraints while avoiding collisions with the environment remains a significant challenge. This paper presents an integrated framework that enables robotic manipulators to safely and automatically identify payload parameters while maintaining operational safety guarantees. The framework consists of two synergistic components: an online trajectory planning and control framework that generates provably-safe exciting trajectories for system identification that can be tracked while respecting robot constraints and avoiding obstacles and a robust system identification method that computes rigorous overapproximative bounds on end-effector inertial parameters assuming bounded sensor noise. Experimental validation on a robotic manipulator performing challenging tasks with various unknown payloads demonstrates the framework's effectiveness in establishing accurate parameter bounds while maintaining safety throughout the identification process. The code is available at our project webpage: this https URL. 

**Abstract (ZH)**: 精确操作任务需要准确的负载惯性参数知识。不幸的是，对于未知负载，同时确保机器人系统满足其输入和状态约束并避免与环境发生碰撞以识别这些参数仍然是一项重大的挑战。本文提出了一种集成框架，使机器人操作器能够安全地自动识别负载参数，同时保持操作安全性保证。该框架由两个协同工作的组件组成：一个在线轨迹规划与控制框架，生成可证明安全的激发轨迹进行系统识别，同时遵守机器人约束并避开障碍物；以及一种鲁棒的系统识别方法，基于有界传感器噪声计算末端执行器惯性参数的严谨包络上界。在各种未知负载下执行具有挑战性的任务的机器人操作器实验验证表明，该框架在识别过程中保持准确的参数边界的同时确保安全性。代码可在我们的项目网页上获取：this https URL。 

---
# SimPRIVE: a Simulation framework for Physical Robot Interaction with Virtual Environments 

**Title (ZH)**: SimPRIVE：一个物理机器人与虚拟环境交互的仿真框架 

**Authors**: Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.21454)  

**Abstract**: The use of machine learning in cyber-physical systems has attracted the interest of both industry and academia. However, no general solution has yet been found against the unpredictable behavior of neural networks and reinforcement learning agents. Nevertheless, the improvements of photo-realistic simulators have paved the way towards extensive testing of complex algorithms in different virtual scenarios, which would be expensive and dangerous to implement in the real world.
This paper presents SimPRIVE, a simulation framework for physical robot interaction with virtual environments, which operates as a vehicle-in-the-loop platform, rendering a virtual world while operating the vehicle in the real world.
Using SimPRIVE, any physical mobile robot running on ROS 2 can easily be configured to move its digital twin in a virtual world built with the Unreal Engine 5 graphic engine, which can be populated with objects, people, or other vehicles with programmable behavior.
SimPRIVE has been designed to accommodate custom or pre-built virtual worlds while being light-weight to contain execution times and allow fast rendering. Its main advantage lies in the possibility of testing complex algorithms on the full software and hardware stack while minimizing the risks and costs of a test campaign. The framework has been validated by testing a reinforcement learning agent trained for obstacle avoidance on an AgileX Scout Mini rover that navigates a virtual office environment where everyday objects and people are placed as obstacles. The physical rover moves with no collision in an indoor limited space, thanks to a LiDAR-based heuristic. 

**Abstract (ZH)**: 机器学习在 cyber-物理系统中的应用引起了工业和学术界的兴趣。然而，尚未找到针对神经网络和强化学习代理不可预测行为的通用解决方案。尽管如此，逼真模拟器性能的改进为在不同虚拟场景中广泛测试复杂算法铺平了道路，而在真实世界中实施这些算法将非常昂贵且危险。
本文提出了一种名为SimPRIVE的仿真框架，用于物理机器人与虚拟环境的交互，该框架作为车辆在环平台运作，同时在真实世界中运行车辆并渲染虚拟世界。
使用SimPRIVE，任何运行在ROS 2上的物理移动机器人可以轻松配置为其虚拟世界中的数字双胞胎移动，该虚拟世界由Unreal Engine 5图形引擎构建，可以包含具有可编程行为的物体、人物或其他车辆。
SimPRIVE设计目的是兼容自定义或预构建的虚拟世界，同时保持轻量级以控制执行时间和允许快速渲染。其主要优势在于能够在完整软件和硬件堆栈上测试复杂算法，同时将测试活动的风险和成本降到最低。该框架通过在AgileX Scout Mini漫游车上测试用于障碍物回避的强化学习代理，并在充满日常物体和人物的虚拟办公室环境中导航来得到验证。得益于基于LiDAR的启发式方法，物理漫游车在室内有限空间内移动时未发生碰撞。 

---
# UAV-VLN: End-to-End Vision Language guided Navigation for UAVs 

**Title (ZH)**: UAV-VLN: 端到端视觉语言引导的无人机导航 

**Authors**: Pranav Saxena, Nishant Raghuvanshi, Neena Goveas  

**Link**: [PDF](https://arxiv.org/pdf/2504.21432)  

**Abstract**: A core challenge in AI-guided autonomy is enabling agents to navigate realistically and effectively in previously unseen environments based on natural language commands. We propose UAV-VLN, a novel end-to-end Vision-Language Navigation (VLN) framework for Unmanned Aerial Vehicles (UAVs) that seamlessly integrates Large Language Models (LLMs) with visual perception to facilitate human-interactive navigation. Our system interprets free-form natural language instructions, grounds them into visual observations, and plans feasible aerial trajectories in diverse environments.
UAV-VLN leverages the common-sense reasoning capabilities of LLMs to parse high-level semantic goals, while a vision model detects and localizes semantically relevant objects in the environment. By fusing these modalities, the UAV can reason about spatial relationships, disambiguate references in human instructions, and plan context-aware behaviors with minimal task-specific supervision. To ensure robust and interpretable decision-making, the framework includes a cross-modal grounding mechanism that aligns linguistic intent with visual context.
We evaluate UAV-VLN across diverse indoor and outdoor navigation scenarios, demonstrating its ability to generalize to novel instructions and environments with minimal task-specific training. Our results show significant improvements in instruction-following accuracy and trajectory efficiency, highlighting the potential of LLM-driven vision-language interfaces for safe, intuitive, and generalizable UAV autonomy. 

**Abstract (ZH)**: AI引导的自主性核心挑战在于使代理能够在以前未见过的环境中根据自然语言指令进行现实有效的导航。我们提出了UAV-V LN，这是一种新颖的端到端视觉-语言导航（VLN）框架，用于无人驾驶飞机（UAVs），该框架无缝地将大型语言模型（LLMs）与视觉感知结合在一起，以促进人机交互导航。该系统解释自由形式的自然语言指令，将其转化为视觉观察，并在多种环境中规划可行的空中轨迹。UAV-VLN利用大型语言模型的常识推理能力解析高层语义目标，同时视觉模型检测并定位环境中的语义相关物。通过融合这些模态，无人驾驶飞机可以推理空间关系，消解人类指令中的指代歧义，并在最少的任务特定监督下规划上下文相关的行为。为了确保稳健和可解释的决策制定，框架包括一个跨模态定位机制，使语言意图与视觉上下文对齐。我们在多种室内外导航场景中评估了UAV-VLN，展示了其在最少任务特定训练的情况下，能够泛化到新指令和新环境的能力。结果表明，指令遵循的准确性显著提高，轨迹效率也得到了提升，突显了基于大型语言模型的视觉-语言接口在安全、直观和泛化的无人驾驶飞机自主性方面的潜力。 

---
# UAV Marketplace Simulation Tool for BVLOS Operations 

**Title (ZH)**: UAV市场交易平台模拟工具missive for BVLOS操作 

**Authors**: Kıvanç Şerefoğlu, Önder Gürcan, Reyhan Aydoğan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21428)  

**Abstract**: We present a simulation tool for evaluating team formation in autonomous multi-UAV (Unmanned Aerial Vehicle) missions that operate Beyond Visual Line of Sight (BVLOS). The tool models UAV collaboration and mission execution in dynamic and adversarial conditions, where Byzantine UAVs attempt to disrupt operations. Our tool allows researchers to integrate and compare various team formation strategies in a controlled environment with configurable mission parameters and adversarial behaviors. The log of each simulation run is stored in a structured way along with performance metrics so that statistical analysis could be done straightforwardly. The tool is versatile for testing and improving UAV coordination strategies in real-world applications. 

**Abstract (ZH)**: 一种用于评估自主多无人机（Unmanned Aerial Vehicle）越视距（Beyond Visual Line of Sight）任务中团队形成的仿真工具 

---
# A Koopman Operator-based NMPC Framework for Mobile Robot Navigation under Uncertainty 

**Title (ZH)**: 基于Koopman算子的移动机器人导航鲁棒NMPC框架 

**Authors**: Xiaobin Zhang, Mohamed Karim Bouafoura, Lu Shi, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2504.21215)  

**Abstract**: Mobile robot navigation can be challenged by system uncertainty. For example, ground friction may vary abruptly causing slipping, and noisy sensor data can lead to inaccurate feedback control. Traditional model-based methods may be limited when considering such variations, making them fragile to varying types of uncertainty. One way to address this is by leveraging learned prediction models by means of the Koopman operator into nonlinear model predictive control (NMPC). This paper describes the formulation of, and provides the solution to, an NMPC problem using a lifted bilinear model that can accurately predict affine input systems with stochastic perturbations. System constraints are defined in the Koopman space, while the optimization problem is solved in the state space to reduce computational complexity. Training data to estimate the Koopman operator for the system are given via randomized control inputs. The output of the developed method enables closed-loop navigation control over environments populated with obstacles. The effectiveness of the proposed method has been tested through numerical simulations using a wheeled robot with additive stochastic velocity perturbations, Gazebo simulations with a realistic digital twin robot, and physical hardware experiments without knowledge of the true dynamics. 

**Abstract (ZH)**: 移动机器人导航可能受到系统不确定性的影响。例如，地面摩擦可能突然变化导致打滑，且嘈杂的传感器数据可能导致不准确的反馈控制。传统的基于模型的方法在考虑此类变化时可能会受到限制，从而使其对不同类型的不确定性变得脆弱。一种解决方法是通过Koopman算子来利用学习到的预测模型来实现非线性模型预测控制（NMPC）。本文描述了使用提升的双线性模型来制定并解决一个NMPC问题，该模型可以准确预测具有随机扰动的仿射输入系统。系统约束在Koopman空间中定义，而优化问题在状态空间中求解以降低计算复杂度。通过随机控制输入来估计系统的Koopman算子。所开发方法的输出使闭环导航控制能够应用于具有障碍物的环境中。通过带有加性随机速度扰动的轮式机器人数值仿真、使用现实数字双胞胎机器人的Gazebo仿真，以及无需了解真实动力学的物理硬件实验，验证了所提出方法的有效性。 

---
# Task and Joint Space Dual-Arm Compliant Control 

**Title (ZH)**: 双臂顺应控制及任务空间-联合空间协同控制 

**Authors**: Alexander L. Mitchell, Tobit Flatscher, Ingmar Posner  

**Link**: [PDF](https://arxiv.org/pdf/2504.21159)  

**Abstract**: Robots that interact with humans or perform delicate manipulation tasks must exhibit compliance. However, most commercial manipulators are rigid and suffer from significant friction, limiting end-effector tracking accuracy in torque-controlled modes. To address this, we present a real-time, open-source impedance controller that smoothly interpolates between joint-space and task-space compliance. This hybrid approach ensures safe interaction and precise task execution, such as sub-centimetre pin insertions. We deploy our controller on Frank, a dual-arm platform with two Kinova Gen3 arms, and compensate for modelled friction dynamics using a model-free observer. The system is real-time capable and integrates with standard ROS tools like MoveIt!. It also supports high-frequency trajectory streaming, enabling closed-loop execution of trajectories generated by learning-based methods, optimal control, or teleoperation. Our results demonstrate robust tracking and compliant behaviour even under high-friction conditions. The complete system is available open-source at this https URL. 

**Abstract (ZH)**: 与人类交互或执行精细操作任务的机器人必须表现出顺应性。然而，大多数商业 manipulator 是刚性的，且摩擦显著，限制了扭矩控制模式下末端执行器的跟踪精度。为了解决这个问题，我们提出了一种实时开源阻抗控制器，可以在关节空间和任务空间顺应性之间平滑插值。这种混合方法确保了安全的交互和精确的任务执行，例如毫米级针插入。我们在配备两个 Kinova Gen3 手臂的双臂平台 Frank 上部署了该控制器，并使用模型自由观测器补偿了建模的摩擦动态。该系统具有实时能力，并与标准 ROS 工具（如 MoveIt!）集成。它还支持高频率轨迹流式传输，使得基于学习的方法、最优控制或遥操作生成的轨迹能够进行闭环执行。我们的结果表明，即使在高摩擦条件下，该系统也能实现稳健的跟踪和顺应性行为。完整系统已开源，可从以下链接访问：this https URL。 

---
# Composite Safety Potential Field for Highway Driving Risk Assessment 

**Title (ZH)**: 高速公路驾驶风险评估的复合安全势场方法 

**Authors**: Dachuan Zuo, Zilin Bian, Fan Zuo, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2504.21158)  

**Abstract**: In the era of rapid advancements in vehicle safety technologies, driving risk assessment has become a focal point of attention. Technologies such as collision warning systems, advanced driver assistance systems (ADAS), and autonomous driving require driving risks to be evaluated proactively and in real time. To be effective, driving risk assessment metrics must not only accurately identify potential collisions but also exhibit human-like reasoning to enable safe and seamless interactions between vehicles. Existing safety potential field models assess driving risks by considering both objective and subjective safety factors. However, their practical applicability in real-world risk assessment tasks is limited. These models are often challenging to calibrate due to the arbitrary nature of their structures, and calibration can be inefficient because of the scarcity of accident statistics. Additionally, they struggle to generalize across both longitudinal and lateral risks. To address these challenges, we propose a composite safety potential field framework, namely C-SPF, involving a subjective field to capture drivers' risk perception about spatial proximity and an objective field to quantify the imminent collision probability, to comprehensively evaluate driving risks. The C-SPF is calibrated using abundant two-dimensional spacing data from trajectory datasets, enabling it to effectively capture drivers' proximity risk perception and provide a more realistic explanation of driving behaviors. Analysis of a naturalistic driving dataset demonstrates that the C-SPF can capture both longitudinal and lateral risks that trigger drivers' safety maneuvers. Further case studies highlight the C-SPF's ability to explain lateral driver behaviors, such as abandoning lane changes or adjusting lateral position relative to adjacent vehicles, which are capabilities that existing models fail to achieve. 

**Abstract (ZH)**: 在车辆安全技术飞速发展的时代，驾驶风险评估已成为关注的焦点。现有的安全潜在领域模型通过考虑客观和主观的安全因素来评估驾驶风险，但由于其结构的任意性和事故统计数据的稀缺性，这些模型在实际风险评估任务中的应用受到限制，且难以泛化到纵向和横向风险。为此，我们提出了一种复合安全潜在领域框架C-SPF，该框架结合了主观领域来捕捉驾驶者对空间接近性的风险感知和客观领域来量化即将发生碰撞的概率，以全面评估驾驶风险。C-SPF通过使用轨迹数据集中的丰富二维间距数据进行校准，能够有效捕捉驾驶者对接近性的风险感知，并提供更现实的驾驶行为解释。对自然驾驶数据集的分析表明，C-SPF能够捕捉到触发驾驶者安全行为的纵向和横向风险。进一步的案例研究还突显了C-SPF解释横向驾驶行为（如放弃车道变更或调整相对于相邻车辆的位置）的能力，这是现有模型无法实现的。 

---
# How to Coordinate UAVs and UGVs for Efficient Mission Planning? Optimizing Energy-Constrained Cooperative Routing with a DRL Framework 

**Title (ZH)**: 如何协调无人机和地面机器人进行高效的任务规划？基于DRL框架的能量约束协同路径优化 

**Authors**: Md Safwan Mondal, Subramanian Ramasamy, Luca Russo, James D. Humann, James M. Dotterweich, Pranav Bhounsule  

**Link**: [PDF](https://arxiv.org/pdf/2504.21111)  

**Abstract**: Efficient mission planning for cooperative systems involving Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs) requires addressing energy constraints, scalability, and coordination challenges between agents. UAVs excel in rapidly covering large areas but are constrained by limited battery life, while UGVs, with their extended operational range and capability to serve as mobile recharging stations, are hindered by slower speeds. This heterogeneity makes coordination between UAVs and UGVs critical for achieving optimal mission outcomes. In this work, we propose a scalable deep reinforcement learning (DRL) framework to address the energy-constrained cooperative routing problem for multi-agent UAV-UGV teams, aiming to visit a set of task points in minimal time with UAVs relying on UGVs for recharging during the mission. The framework incorporates sortie-wise agent switching to efficiently manage multiple agents, by allocating task points and coordinating actions. Using an encoder-decoder transformer architecture, it optimizes routes and recharging rendezvous for the UAV-UGV team in the task scenario. Extensive computational experiments demonstrate the framework's superior performance over heuristic methods and a DRL baseline, delivering significant improvements in solution quality and runtime efficiency across diverse scenarios. Generalization studies validate its robustness, while dynamic scenario highlights its adaptability to real-time changes with a case study. This work advances UAV-UGV cooperative routing by providing a scalable, efficient, and robust solution for multi-agent mission planning. 

**Abstract (ZH)**: 基于无人机（UAV）与地面机器人（UGV）的多智能体协同航路规划：一种考虑能量约束的可扩展深度强化学习框架 

---
# Automated Parking Trajectory Generation Using Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的自动化停车轨迹生成 

**Authors**: Zheyu Zhang, Yutong Luo, Yongzhou Chen, Haopeng Zhao, Zhichao Ma, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21071)  

**Abstract**: Autonomous parking is a key technology in modern autonomous driving systems, requiring high precision, strong adaptability, and efficiency in complex environments. This paper proposes a Deep Reinforcement Learning (DRL) framework based on the Soft Actor-Critic (SAC) algorithm to optimize autonomous parking tasks. SAC, an off-policy method with entropy regularization, is particularly well-suited for continuous action spaces, enabling fine-grained vehicle control. We model the parking task as a Markov Decision Process (MDP) and train an agent to maximize cumulative rewards while balancing exploration and exploitation through entropy maximization. The proposed system integrates multiple sensor inputs into a high-dimensional state space and leverages SAC's dual critic networks and policy network to achieve stable learning. Simulation results show that the SAC-based approach delivers high parking success rates, reduced maneuver times, and robust handling of dynamic obstacles, outperforming traditional rule-based methods and other DRL algorithms. This study demonstrates SAC's potential in autonomous parking and lays the foundation for real-world applications. 

**Abstract (ZH)**: 基于Soft Actor-Critic算法的深度强化学习自主泊车框架 

---
# REHEARSE-3D: A Multi-modal Emulated Rain Dataset for 3D Point Cloud De-raining 

**Title (ZH)**: REHEARSE-3D: 一种多模态模拟雨滴点云去雨数据集 

**Authors**: Abu Mohammed Raisuddin, Jesper Holmblad, Hamed Haghighi, Yuri Poledna, Maikol Funk Drechsler, Valentina Donzella, Eren Erdal Aksoy  

**Link**: [PDF](https://arxiv.org/pdf/2504.21699)  

**Abstract**: Sensor degradation poses a significant challenge in autonomous driving. During heavy rainfall, the interference from raindrops can adversely affect the quality of LiDAR point clouds, resulting in, for instance, inaccurate point measurements. This, in turn, can potentially lead to safety concerns if autonomous driving systems are not weather-aware, i.e., if they are unable to discern such changes. In this study, we release a new, large-scale, multi-modal emulated rain dataset, REHEARSE-3D, to promote research advancements in 3D point cloud de-raining. Distinct from the most relevant competitors, our dataset is unique in several respects. First, it is the largest point-wise annotated dataset, and second, it is the only one with high-resolution LiDAR data (LiDAR-256) enriched with 4D Radar point clouds logged in both daytime and nighttime conditions in a controlled weather environment. Furthermore, REHEARSE-3D involves rain-characteristic information, which is of significant value not only for sensor noise modeling but also for analyzing the impact of weather at a point level. Leveraging REHEARSE-3D, we benchmark raindrop detection and removal in fused LiDAR and 4D Radar point clouds. Our comprehensive study further evaluates the performance of various statistical and deep-learning models. Upon publication, the dataset and benchmark models will be made publicly available at: this https URL. 

**Abstract (ZH)**: 传感器退化对自主驾驶构成显著挑战。在暴雨中，雨滴的干扰会负面影响LiDAR点云质量，例如导致不准确的点测量。如果自主驾驶系统不具备天气 aware 性能，即无法识别此类变化，这可能会导致安全隐患。在这项研究中，我们发布了一个新的大规模多模式模拟降雨数据集REHEARSE-3D，以促进3D点云除雨研究的进步。与最相关竞争对手相比，我们的数据集在多个方面具有独特性。首先，它是最大的逐点标注数据集；其次，它是唯一一个包含在受控天气环境中记录的高分辨率LiDAR数据（LiDAR-256）和4D雷达点云的数据集，无论白天还是黑夜。此外，REHEARSE-3D包括了降雨特性信息，这对于传感器噪声建模和点级天气影响分析都具有重要意义。利用REHEARSE-3D，我们对融合LiDAR和4D雷达点云的雨滴检测与去除进行了基准测试。我们进一步的全面研究还评估了各种统计和深度学习模型的性能。发布后，数据集和基准模型将公开发布于：this https URL。 

---
# Designing Control Barrier Function via Probabilistic Enumeration for Safe Reinforcement Learning Navigation 

**Title (ZH)**: 基于概率枚举的控制障碍函数设计以实现安全的强化学习导航 

**Authors**: Luca Marzari, Francesco Trotti, Enrico Marchesini, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.21643)  

**Abstract**: Achieving safe autonomous navigation systems is critical for deploying robots in dynamic and uncertain real-world environments. In this paper, we propose a hierarchical control framework leveraging neural network verification techniques to design control barrier functions (CBFs) and policy correction mechanisms that ensure safe reinforcement learning navigation policies. Our approach relies on probabilistic enumeration to identify unsafe regions of operation, which are then used to construct a safe CBF-based control layer applicable to arbitrary policies. We validate our framework both in simulation and on a real robot, using a standard mobile robot benchmark and a highly dynamic aquatic environmental monitoring task. These experiments demonstrate the ability of the proposed solution to correct unsafe actions while preserving efficient navigation behavior. Our results show the promise of developing hierarchical verification-based systems to enable safe and robust navigation behaviors in complex scenarios. 

**Abstract (ZH)**: 实现安全自主导航系统对于在动态和不确定的现实环境中部署机器人至关重要。本文提出了一种层次控制框架，利用神经网络验证技术设计控制障碍函数（CBFs）及策略修正机制，以确保安全的强化学习导航策略。我们的方法依赖于概率枚举来识别操作的不安全区域，进而构建适用于任意策略的安全CBFs控制层。我们在仿真和实际机器人上对框架进行了验证，使用标准的移动机器人基准和高度动态的水下环境监测任务。这些实验表明，所提出的方法能够纠正不安全的行为同时保持高效的导航行为。我们的结果展示了基于层次验证的系统在复杂场景中实现安全和稳健的导航行为的潜力。 

---
# Leveraging Systems and Control Theory for Social Robotics: A Model-Based Behavioral Control Approach to Human-Robot Interaction 

**Title (ZH)**: 利用系统与控制理论进行社会机器人研究：基于模型的行为控制方法在人机交互中的应用 

**Authors**: Maria Morão Patrício, Anahita Jamshidnejad  

**Link**: [PDF](https://arxiv.org/pdf/2504.21548)  

**Abstract**: Social robots (SRs) should autonomously interact with humans, while exhibiting proper social behaviors associated to their role. By contributing to health-care, education, and companionship, SRs will enhance life quality. However, personalization and sustaining user engagement remain a challenge for SRs, due to their limited understanding of human mental states. Accordingly, we leverage a recently introduced mathematical dynamic model of human perception, cognition, and decision-making for SRs. Identifying the parameters of this model and deploying it in behavioral steering system of SRs allows to effectively personalize the responses of SRs to evolving mental states of their users, enhancing long-term engagement and personalization. Our approach uniquely enables autonomous adaptability of SRs by modeling the dynamics of invisible mental states, significantly contributing to the transparency and awareness of SRs. We validated our model-based control system in experiments with 10 participants who interacted with a Nao robot over three chess puzzle sessions, 45 - 90 minutes each. The identified model achieved a mean squared error (MSE) of 0.067 (i.e., 1.675% of the maximum possible MSE) in tracking beliefs, goals, and emotions of participants. Compared to a model-free controller that did not track mental states of participants, our approach increased engagement by 16% on average. Post-interaction feedback of participants (provided via dedicated questionnaires) further confirmed the perceived engagement and awareness of the model-driven robot. These results highlight the unique potential of model-based approaches and control theory in advancing human-SR interactions. 

**Abstract (ZH)**: 社会机器人（SRs）应自主与人类互动，并表现出与其角色相关的适当社会行为。通过在健康care、教育和陪伴等方面发挥作用，SRs将提高生活质量。然而，个性化和维持用户参与度仍然是SRs面临的挑战，原因在于它们对人类心理状态的有限理解。因此，我们利用一种最近引入的人类感知、认知和决策过程的数学动态模型为SRs发挥作用。通过识别该模型的参数并在SRs的行为引导系统中部署它，可以有效地使SRs的响应个性化，以适应用户不断变化的心理状态，从而增强长期的参与度和个性化程度。我们的方法独特地使SRs具备了对无形心理状态动态建模的自主适应能力，显著提升了SRs的透明度和意识水平。我们通过与10名参与者进行实验来验证基于模型的控制系统，参与者与一个NAO机器人进行了三次国际象棋难题会话（每次会话时长45-90分钟）。识别出的模型在跟踪参与者信念、目标和情绪时的均方误差（MSE）为0.067（即最大可能MSE的1.675%）。与不跟踪参与者心理状态的无模型控制器相比，我们的方法平均增加了16%的参与度。参与者在互动后的反馈（通过专门的问卷提供）进一步证实了模型驱动机器人感知到的交互和意识。这些结果突显了基于模型的方法和控制理论在提升人类-SR互动方面的独特潜力。 

---
# CMD: Constraining Multimodal Distribution for Domain Adaptation in Stereo Matching 

**Title (ZH)**: CMD：约束多模态分布的领域适应立体匹配 

**Authors**: Zhelun Shen, Zhuo Li, Chenming Wu, Zhibo Rao, Lina Liu, Yuchao Dai, Liangjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21302)  

**Abstract**: Recently, learning-based stereo matching methods have achieved great improvement in public benchmarks, where soft argmin and smooth L1 loss play a core contribution to their success. However, in unsupervised domain adaptation scenarios, we observe that these two operations often yield multimodal disparity probability distributions in target domains, resulting in degraded generalization. In this paper, we propose a novel approach, Constrain Multi-modal Distribution (CMD), to address this issue. Specifically, we introduce \textit{uncertainty-regularized minimization} and \textit{anisotropic soft argmin} to encourage the network to produce predominantly unimodal disparity distributions in the target domain, thereby improving prediction accuracy. Experimentally, we apply the proposed method to multiple representative stereo-matching networks and conduct domain adaptation from synthetic data to unlabeled real-world scenes. Results consistently demonstrate improved generalization in both top-performing and domain-adaptable stereo-matching models. The code for CMD will be available at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于约束多模态分布的无监督场景适配立体匹配方法 

---
# PhysicsFC: Learning User-Controlled Skills for a Physics-Based Football Player Controller 

**Title (ZH)**: PhysicsFC：学习用户控制技能的物理基础足球玩家控制器 

**Authors**: Minsu Kim, Eunho Jung, Yoonsang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.21216)  

**Abstract**: We propose PhysicsFC, a method for controlling physically simulated football player characters to perform a variety of football skills--such as dribbling, trapping, moving, and kicking--based on user input, while seamlessly transitioning between these skills. Our skill-specific policies, which generate latent variables for each football skill, are trained using an existing physics-based motion embedding model that serves as a foundation for reproducing football motions. Key features include a tailored reward design for the Dribble policy, a two-phase reward structure combined with projectile dynamics-based initialization for the Trap policy, and a Data-Embedded Goal-Conditioned Latent Guidance (DEGCL) method for the Move policy. Using the trained skill policies, the proposed football player finite state machine (PhysicsFC FSM) allows users to interactively control the character. To ensure smooth and agile transitions between skill policies, as defined in the FSM, we introduce the Skill Transition-Based Initialization (STI), which is applied during the training of each skill policy. We develop several interactive scenarios to showcase PhysicsFC's effectiveness, including competitive trapping and dribbling, give-and-go plays, and 11v11 football games, where multiple PhysicsFC agents produce natural and controllable physics-based football player behaviors. Quantitative evaluations further validate the performance of individual skill policies and the transitions between them, using the presented metrics and experimental designs. 

**Abstract (ZH)**: 我们提出PhysicsFC方法，该方法基于用户输入控制物理模拟的足球球员角色执行多种足球技能——如带球、控球、移动和射门——同时无缝过渡到这些技能。关键特征包括为Dribble策略量身设计的奖励设计、Trap策略中结合轨迹动力学初始化的两阶段奖励结构以及嵌入数据的目标条件潜变量指导（DEGCL）方法。利用训练好的技能策略，提出的足球玩家有限状态机（PhysicsFC FSM）允许用户交互控制角色。为了确保有限状态机（FSM）中定义的技能策略之间的平滑和灵活过渡，我们引入了技能过渡基于初始化（STI），并将其应用于每个技能策略的训练中。我们开发了多个交互场景以展示PhysicsFC的有效性，包括竞争性控球和带球、传切配合以及11对11足球比赛，其中多个PhysicsFC代理产生自然可控的基于物理的足球玩家行为。定量评估进一步验证了每个技能策略及其过渡之间的性能，使用呈现的度量标准和实验设计。 

---
# NavEX: A Multi-Agent Coverage in Non-Convex and Uneven Environments via Exemplar-Clustering 

**Title (ZH)**: NavEX: 一种基于范例聚类的非凸不均匀环境多agents覆盖算法 

**Authors**: Donipolo Ghimire, Carlos Nieto-Granda, Solmaz S. Kia  

**Link**: [PDF](https://arxiv.org/pdf/2504.21113)  

**Abstract**: This paper addresses multi-agent deployment in non-convex and uneven environments. To overcome the limitations of traditional approaches, we introduce Navigable Exemplar-Based Dispatch Coverage (NavEX), a novel dispatch coverage framework that combines exemplar-clustering with obstacle-aware and traversability-aware shortest distances, offering a deployment framework based on submodular optimization. NavEX provides a unified approach to solve two critical coverage tasks: (a) fair-access deployment, aiming to provide equitable service by minimizing agent-target distances, and (b) hotspot deployment, prioritizing high-density target regions. A key feature of NavEX is the use of exemplar-clustering for the coverage utility measure, which provides the flexibility to employ non-Euclidean distance metrics that do not necessarily conform to the triangle inequality. This allows NavEX to incorporate visibility graphs for shortest-path computation in environments with planar obstacles, and traversability-aware RRT* for complex, rugged terrains. By leveraging submodular optimization, the NavEX framework enables efficient, near-optimal solutions with provable performance guarantees for multi-agent deployment in realistic and complex settings, as demonstrated by our simulations. 

**Abstract (ZH)**: 基于导航典范的分布式覆盖框架（NavEX）：非凸不规则环境下的多agent部署 

---
# GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction 

**Title (ZH)**: GauSS-MI: 高斯点云 Shannon 互信息active 3D重建 

**Authors**: Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21067)  

**Abstract**: This research tackles the challenge of real-time active view selection and uncertainty quantification on visual quality for active 3D reconstruction. Visual quality is a critical aspect of 3D reconstruction. Recent advancements such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have notably enhanced the image rendering quality of reconstruction models. Nonetheless, the efficient and effective acquisition of input images for reconstruction-specifically, the selection of the most informative viewpoint-remains an open challenge, which is crucial for active reconstruction. Existing studies have primarily focused on evaluating geometric completeness and exploring unobserved or unknown regions, without direct evaluation of the visual uncertainty within the reconstruction model. To address this gap, this paper introduces a probabilistic model that quantifies visual uncertainty for each Gaussian. Leveraging Shannon Mutual Information, we formulate a criterion, Gaussian Splatting Shannon Mutual Information (GauSS-MI), for real-time assessment of visual mutual information from novel viewpoints, facilitating the selection of next best view. GauSS-MI is implemented within an active reconstruction system integrated with a view and motion planner. Extensive experiments across various simulated and real-world scenes showcase the superior visual quality and reconstruction efficiency performance of the proposed system. 

**Abstract (ZH)**: 本研究解决了实时主动视角选择和视觉质量不确定性量化在主动3D重建中的挑战。视觉质量是3D重建的关键 aspects。最近的进展如神经辐射场（NeRF）和三维高斯散点图（3DGS）显著地提高了重建模型的图像渲染质量。然而，为重建高效和有效地获取输入图像，特别是在选择最有信息量的视角方面的挑战仍然未解决，这是主动重建的关键。现有研究主要集中在评估几何完整性并探索未观察或未知区域，而没有直接评估重建模型内的视觉不确定性。为弥补这一差距，本文提出一种概率模型来量化每个高斯的视觉不确定性。基于香农互信息，我们提出了高斯散点图香农互信息（GauSS-MI）准则，用于实时评估新视角下的视觉互信息，辅助选择下一个最佳视角。GauSS-MI 在结合视图和运动规划器的主动重建系统中实现。广泛实验在各种模拟和真实场景中展示了所提系统的卓越视觉质量和重建效率性能。 

---
# ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees 

**Title (ZH)**: ConformalNL2LTL: 将自然语言指令转化为具有符合正确性保证的时间逻辑公式 

**Authors**: Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2504.21022)  

**Abstract**: Linear Temporal Logic (LTL) has become a prevalent specification language for robotic tasks. To mitigate the significant manual effort and expertise required to define LTL-encoded tasks, several methods have been proposed for translating Natural Language (NL) instructions into LTL formulas, which, however, lack correctness guarantees. To address this, we introduce a new NL-to-LTL translation method, called ConformalNL2LTL, that can achieve user-defined translation success rates over unseen NL commands. Our method constructs LTL formulas iteratively by addressing a sequence of open-vocabulary Question-Answering (QA) problems with LLMs. To enable uncertainty-aware translation, we leverage conformal prediction (CP), a distribution-free uncertainty quantification tool for black-box models. CP enables our method to assess the uncertainty in LLM-generated answers, allowing it to proceed with translation when sufficiently confident and request help otherwise. We provide both theoretical and empirical results demonstrating that ConformalNL2LTL achieves user-specified translation accuracy while minimizing help rates. 

**Abstract (ZH)**: 一种新的不确定导向的自然语言到线性时序逻辑的翻译方法：ConformalNL2LTL 

---
