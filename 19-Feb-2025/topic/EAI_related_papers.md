# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation 

**Title (ZH)**: SoFar: 语言导向的空间导航连接空间推理与物体操作 

**Authors**: Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13143)  

**Abstract**: Spatial intelligence is a critical component of embodied AI, promoting robots to understand and interact with their environments. While recent advances have enhanced the ability of VLMs to perceive object locations and positional relationships, they still lack the capability to precisely understand object orientations-a key requirement for tasks involving fine-grained manipulations. Addressing this limitation not only requires geometric reasoning but also an expressive and intuitive way to represent orientation. In this context, we propose that natural language offers a more flexible representation space than canonical frames, making it particularly suitable for instruction-following robotic systems. In this paper, we introduce the concept of semantic orientation, which defines object orientations using natural language in a reference-frame-free manner (e.g., the ''plug-in'' direction of a USB or the ''handle'' direction of a knife). To support this, we construct OrienText300K, a large-scale dataset of 3D models annotated with semantic orientations that link geometric understanding to functional semantics. By integrating semantic orientation into a VLM system, we enable robots to generate manipulation actions with both positional and orientational constraints. Extensive experiments in simulation and real world demonstrate that our approach significantly enhances robotic manipulation capabilities, e.g., 48.7% accuracy on Open6DOR and 74.9% accuracy on SIMPLER. 

**Abstract (ZH)**: 空间智能是体现式AI的关键组成部分，促进机器人理解并与其环境互动。尽管近年来视觉语言模型的能力得到了增强，使其能够感知物体位置和空间关系，但它们仍然缺乏精确理解物体姿态的能力——这是进行精细操作任务所需的关键能力。为解决这一局限，不仅需要几何推理，还需要一种灵活且直观的方式来表示姿态。在此背景下，我们认为自然语言提供了比标准坐标系更具弹性的表示空间，使其特别适合于指令跟随的机器人系统。在这项研究中，我们提出了语义姿态的概念，用自然语言在无参考坐标系的方式下定义物体姿态（例如，USB的“插接方向”或刀具的“把手方向”）。为了支持这一概念，我们构建了OrienText300K数据集，该数据集包含大量标注有语义姿态的3D模型，将几何理解与功能语义联系起来。通过将语义姿态集成到视觉语言模型系统中，我们使机器人能够生成带有位置和姿态约束的操纵动作。在模拟和真实环境中的广泛实验表明，我们的方法大大提升了一体机操作能力，例如，在Open6DOR上的准确率为48.7%，在SIMPLER上的准确率为74.9%。 

---
# Pre-training Auto-regressive Robotic Models with 4D Representations 

**Title (ZH)**: 使用4D表示预训练自回归机器人模型 

**Authors**: Dantong Niu, Yuvan Sharma, Haoru Xue, Giscard Biamby, Junyi Zhang, Ziteng Ji, Trevor Darrell, Roei Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13142)  

**Abstract**: Foundation models pre-trained on massive unlabeled datasets have revolutionized natural language and computer vision, exhibiting remarkable generalization capabilities, thus highlighting the importance of pre-training. Yet, efforts in robotics have struggled to achieve similar success, limited by either the need for costly robotic annotations or the lack of representations that effectively model the physical world. In this paper, we introduce ARM4R, an Auto-regressive Robotic Model that leverages low-level 4D Representations learned from human video data to yield a better pre-trained robotic model. Specifically, we focus on utilizing 3D point tracking representations from videos derived by lifting 2D representations into 3D space via monocular depth estimation across time. These 4D representations maintain a shared geometric structure between the points and robot state representations up to a linear transformation, enabling efficient transfer learning from human video data to low-level robotic control. Our experiments show that ARM4R can transfer efficiently from human video data to robotics and consistently improves performance on tasks across various robot environments and configurations. 

**Abstract (ZH)**: 基于人类视频数据中学习到的低级四维表示的自回归机器人模型ARM4R 

---
# RHINO: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations 

**Title (ZH)**: RHINO：从人类示范学习实时人体-人体-物体交互 

**Authors**: Jingxiao Chen, Xinyao Li, Jiahang Cao, Zhengbang Zhu, Wentao Dong, Minghuan Liu, Ying Wen, Yong Yu, Liqing Zhang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13134)  

**Abstract**: Humanoid robots have shown success in locomotion and manipulation. Despite these basic abilities, humanoids are still required to quickly understand human instructions and react based on human interaction signals to become valuable assistants in human daily life. Unfortunately, most existing works only focus on multi-stage interactions, treating each task separately, and neglecting real-time feedback. In this work, we aim to empower humanoid robots with real-time reaction abilities to achieve various tasks, allowing human to interrupt robots at any time, and making robots respond to humans immediately. To support such abilities, we propose a general humanoid-human-object interaction framework, named RHINO, i.e., Real-time Humanoid-human Interaction and Object manipulation. RHINO provides a unified view of reactive motion, instruction-based manipulation, and safety concerns, over multiple human signal modalities, such as languages, images, and motions. RHINO is a hierarchical learning framework, enabling humanoids to learn reaction skills from human-human-object demonstrations and teleoperation data. In particular, it decouples the interaction process into two levels: 1) a high-level planner inferring human intentions from real-time human behaviors; and 2) a low-level controller achieving reactive motion behaviors and object manipulation skills based on the predicted intentions. We evaluate the proposed framework on a real humanoid robot and demonstrate its effectiveness, flexibility, and safety in various scenarios. 

**Abstract (ZH)**: 实时人形机器人-human物体交互框架RHINO：实时反应能力与人机互动 

---
# RobotIQ: Empowering Mobile Robots with Human-Level Planning for Real-World Execution 

**Title (ZH)**: RobotIQ：赋予移动机器人类人级别的规划能力以实现实际执行 

**Authors**: Emmanuel K. Raptis, Athanasios Ch. Kapoutsis, Elias B. Kosmatopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.12862)  

**Abstract**: This paper introduces RobotIQ, a framework that empowers mobile robots with human-level planning capabilities, enabling seamless communication via natural language instructions through any Large Language Model. The proposed framework is designed in the ROS architecture and aims to bridge the gap between humans and robots, enabling robots to comprehend and execute user-expressed text or voice commands. Our research encompasses a wide spectrum of robotic tasks, ranging from fundamental logical, mathematical, and learning reasoning for transferring knowledge in domains like navigation, manipulation, and object localization, enabling the application of learned behaviors from simulated environments to real-world operations. All encapsulated within a modular crafted robot library suite of API-wise control functions, RobotIQ offers a fully functional AI-ROS-based toolset that allows researchers to design and develop their own robotic actions tailored to specific applications and robot configurations. The effectiveness of the proposed system was tested and validated both in simulated and real-world experiments focusing on a home service scenario that included an assistive application designed for elderly people. RobotIQ with an open-source, easy-to-use, and adaptable robotic library suite for any robot can be found at this https URL. 

**Abstract (ZH)**: 本文介绍了RobotIQ，这是一种框架，使移动机器人具备类似于人类的规划能力，通过任何大型语言模型实现自然语言指令的无缝通信。该提出的框架基于ROS架构设计，旨在弥合人类与机器人之间的差距，使机器人能够理解并执行用户表达的文本或语音命令。我们的研究涵盖了一系列机器人任务，包括导航、操作和物体定位等领域的基本逻辑、数学和学习推理，以便将模拟环境中学到的行为应用于现实世界操作。RobotIQ通过模块化的机器人库API控制函数封装，提供了一个功能齐全的基于AI-ROS的工具集，允许研究人员设计和开发针对特定应用和机器人配置的机器人动作。在家庭服务场景中，该系统包括一个辅助应用程序，旨在为老年人提供服务。RobotIQ附带一个开源、使用简便且可适应任何机器人的机器人库套件，可访问此链接：https://github.com/robotiq/RobotIQ。 

---
# InstructRobot: A Model-Free Framework for Mapping Natural Language Instructions into Robot Motion 

**Title (ZH)**: InstructRobot: 一种无需模型的自然语言指令到机器人运动映射框架 

**Authors**: Iury Cleveston, Alana C. Santana, Paula D. P. Costa, Ricardo R. Gudwin, Alexandre S. Simões, Esther L. Colombini  

**Link**: [PDF](https://arxiv.org/pdf/2502.12861)  

**Abstract**: The ability to communicate with robots using natural language is a significant step forward in human-robot interaction. However, accurately translating verbal commands into physical actions is promising, but still presents challenges. Current approaches require large datasets to train the models and are limited to robots with a maximum of 6 degrees of freedom. To address these issues, we propose a framework called InstructRobot that maps natural language instructions into robot motion without requiring the construction of large datasets or prior knowledge of the robot's kinematics model. InstructRobot employs a reinforcement learning algorithm that enables joint learning of language representations and inverse kinematics model, simplifying the entire learning process. The proposed framework is validated using a complex robot with 26 revolute joints in object manipulation tasks, demonstrating its robustness and adaptability in realistic environments. The framework can be applied to any task or domain where datasets are scarce and difficult to create, making it an intuitive and accessible solution to the challenges of training robots using linguistic communication. Open source code for the InstructRobot framework and experiments can be accessed at this https URL. 

**Abstract (ZH)**: 使用自然语言与机器人交流的能力是人类-机器人交互领域的一项重要进展。然而，将口头指令准确地翻译成物理动作尽管充满希望，但仍面临挑战。现有方法需要大量数据集来训练模型，并且限制在具有最多6个自由度的机器人上。为了解决这些问题，我们提出了一种名为InstructRobot的框架，该框架将自然语言指令映射为机器人运动，而无需构建大量数据集或预先了解机器人的运动学模型。InstructRobot采用了强化学习算法，能够联合学习语言表示和逆运动学模型，简化了整个学习过程。该提出的框架通过使用具有26个转动关节的复杂机器人在物体操作任务中进行了验证，展示了其在现实环境中的稳健性和适应性。该框架可以应用于数据集稀缺且难以创建的任务或领域，使其成为训练使用语言交流的机器人的一种直观且易用的解决方案。InstructRobot框架的开源代码和实验可以在以下网址访问：this https URL。 

---
# Design Optimization of Musculoskeletal Humanoids with Maximization of Redundancy to Compensate for Muscle Rupture 

**Title (ZH)**: 具备肌肉断裂补偿能力的最大冗余度设计优化的人体机器人类体设计优化 

**Authors**: Kento Kawaharazuka, Yasunori Toshimitsu, Manabu Nishiura, Yuya Koga, Yusuke Omura, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12803)  

**Abstract**: Musculoskeletal humanoids have various biomimetic advantages, and the redundant muscle arrangement allowing for variable stiffness control is one of the most important. In this study, we focus on one feature of the redundancy, which enables the humanoid to keep moving even if one of its muscles breaks, an advantage that has not been dealt with in many studies. In order to make the most of this advantage, the design of muscle arrangement is optimized by considering the maximization of minimum available torque that can be exerted when one muscle breaks. This method is applied to the elbow of a musculoskeletal humanoid Musashi with simulations, the design policy is extracted from the optimization results, and its effectiveness is confirmed with the actual robot. 

**Abstract (ZH)**: 具有冗余肌肉排列的肌骨骼人形机器人具有多种生物模仿优势，其中允许变量刚度控制的多余肌肉排列尤为关键。本研究聚焦于冗余的一种特性，即使一根肌肉断裂，机器人仍能继续移动，这一点在许多研究中尚未得到充分探讨。为了充分利用这一优势，通过最大化单根肌肉断裂时仍可利用的最小可用扭矩来优化肌肉排列设计。该方法应用到肌骨骼人形机器人Musashi的肘部，并通过仿真提取设计策略，最终通过实际机器人验证其有效性。 

---
# Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control 

**Title (ZH)**: 响应性噪声传递扩散策略：响应性强且高效的感觉运动控制 

**Authors**: Zhuoqun Chen, Xiu Yuan, Tongzhou Mu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12724)  

**Abstract**: Imitation learning is an efficient method for teaching robots a variety of tasks. Diffusion Policy, which uses a conditional denoising diffusion process to generate actions, has demonstrated superior performance, particularly in learning from multi-modal demonstrates. However, it relies on executing multiple actions to retain performance and prevent mode bouncing, which limits its responsiveness, as actions are not conditioned on the most recent observations. To address this, we introduce Responsive Noise-Relaying Diffusion Policy (RNR-DP), which maintains a noise-relaying buffer with progressively increasing noise levels and employs a sequential denoising mechanism that generates immediate, noise-free actions at the head of the sequence, while appending noisy actions at the tail. This ensures that actions are responsive and conditioned on the latest observations, while maintaining motion consistency through the noise-relaying buffer. This design enables the handling of tasks requiring responsive control, and accelerates action generation by reusing denoising steps. Experiments on response-sensitive tasks demonstrate that, compared to Diffusion Policy, ours achieves 18% improvement in success rate. Further evaluation on regular tasks demonstrates that RNR-DP also exceeds the best acceleration method by 6.9%, highlighting its computational efficiency advantage in scenarios where responsiveness is less critical. 

**Abstract (ZH)**: 响应式噪声传递扩散策略在模仿学习中的应用研究 

---
# SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning 

**Title (ZH)**: SATA：受动物学习启发的安全自适应扭矩基运动策略 

**Authors**: Peizhuo Li, Hongyi Li, Ge Sun, Jin Cheng, Xinrong Yang, Guillaume Bellegarda, Milad Shafiee, Yuhong Cao, Auke Ijspeert, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.12674)  

**Abstract**: Despite recent advances in learning-based controllers for legged robots, deployments in human-centric environments remain limited by safety concerns. Most of these approaches use position-based control, where policies output target joint angles that must be processed by a low-level controller (e.g., PD or impedance controllers) to compute joint torques. Although impressive results have been achieved in controlled real-world scenarios, these methods often struggle with compliance and adaptability when encountering environments or disturbances unseen during training, potentially resulting in extreme or unsafe behaviors. Inspired by how animals achieve smooth and adaptive movements by controlling muscle extension and contraction, torque-based policies offer a promising alternative by enabling precise and direct control of the actuators in torque space. In principle, this approach facilitates more effective interactions with the environment, resulting in safer and more adaptable behaviors. However, challenges such as a highly nonlinear state space and inefficient exploration during training have hindered their broader adoption. To address these limitations, we propose SATA, a bio-inspired framework that mimics key biomechanical principles and adaptive learning mechanisms observed in animal locomotion. Our approach effectively addresses the inherent challenges of learning torque-based policies by significantly improving early-stage exploration, leading to high-performance final policies. Remarkably, our method achieves zero-shot sim-to-real transfer. Our experimental results indicate that SATA demonstrates remarkable compliance and safety, even in challenging environments such as soft/slippery terrain or narrow passages, and under significant external disturbances, highlighting its potential for practical deployments in human-centric and safety-critical scenarios. 

**Abstract (ZH)**: 基于扭矩的学习控制方法在人本环境中应用的研究 

---
# Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum 

**Title (ZH)**: 基于系统奖励分析和视觉-语言模型引导课程的学习高质量机器人擦拭策略 

**Authors**: Yihong Liu, Dongyeop Kang, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2502.12599)  

**Abstract**: Autonomous robotic wiping is an important task in various industries, ranging from industrial manufacturing to sanitization in healthcare. Deep reinforcement learning (Deep RL) has emerged as a promising algorithm, however, it often suffers from a high demand for repetitive reward engineering. Instead of relying on manual tuning, we first analyze the convergence of quality-critical robotic wiping, which requires both high-quality wiping and fast task completion, to show the poor convergence of the problem and propose a new bounded reward formulation to make the problem feasible. Then, we further improve the learning process by proposing a novel visual-language model (VLM) based curriculum, which actively monitors the progress and suggests hyperparameter tuning. We demonstrate that the combined method can find a desirable wiping policy on surfaces with various curvatures, frictions, and waypoints, which cannot be learned with the baseline formulation. The demo of this project can be found at: this https URL. 

**Abstract (ZH)**: 自主机器人擦拭是一项重要任务，应用于从工业制造到医疗消毒的各个领域。深度强化学习（Deep RL）已成为一种有前景的算法，然而它通常会遭受重复奖励工程需求高的困扰。我们首先分析关键质量要求的机器人擦拭的收敛性，该要求既需要高质量擦拭又需要快速任务完成，展示了该问题的不良收敛性，并提出了一种新的有界奖励形式来使问题可行。然后，我们通过提出一种基于视觉语言模型（VLM）的新颖课程学习方法进一步改进学习过程，该方法积极监控进度并建议超参数调整。我们证明，结合方法能够在具有各种曲率、摩擦力和航点的表面上找到一个理想的擦拭策略，这是基线形式无法学习到的。该项目的演示可以在以下链接找到：this https URL。 

---
# USPilot: An Embodied Robotic Assistant Ultrasound System with Large Language Model Enhanced Graph Planner 

**Title (ZH)**: USPilot: 一种增强图规划的大语言模型驱动的 embodied 超声机器人助手系统 

**Authors**: Mingcong Chen, Siqi Fan, Guanglin Cao, Hongbin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12498)  

**Abstract**: In the era of Large Language Models (LLMs), embodied artificial intelligence presents transformative opportunities for robotic manipulation tasks. Ultrasound imaging, a widely used and cost-effective medical diagnostic procedure, faces challenges due to the global shortage of professional sonographers. To address this issue, we propose USPilot, an embodied robotic assistant ultrasound system powered by an LLM-based framework to enable autonomous ultrasound acquisition. USPilot is designed to function as a virtual sonographer, capable of responding to patients' ultrasound-related queries and performing ultrasound scans based on user intent. By fine-tuning the LLM, USPilot demonstrates a deep understanding of ultrasound-specific questions and tasks. Furthermore, USPilot incorporates an LLM-enhanced Graph Neural Network (GNN) to manage ultrasound robotic APIs and serve as a task planner. Experimental results show that the LLM-enhanced GNN achieves unprecedented accuracy in task planning on public datasets. Additionally, the system demonstrates significant potential in autonomously understanding and executing ultrasound procedures. These advancements bring us closer to achieving autonomous and potentially unmanned robotic ultrasound systems, addressing critical resource gaps in medical imaging. 

**Abstract (ZH)**: 在大型语言模型时代，具身人工智能为机器人操作任务提供了变革性的机会。超声成像作为一种广泛使用且成本效益高的医疗诊断程序，由于专业超声技师的全球短缺而面临挑战。为解决这一问题，我们提出了USPilot，一种基于大型语言模型框架的动力具身机器人辅助超声系统，用于实现自动超声成像。USPilot被设计为一种虚拟超声技师，能够响应患者的超声相关查询，并根据用户意图执行超声扫描。通过微调大型语言模型，USPilot展示了对超声特定问题和任务的深刻理解。此外，USPilot还集成了增强的图形神经网络（GNN），用于管理和作为任务规划器处理超声机器人API。实验结果表明，增强的GNN在公共数据集上的任务规划中达到了前所未有的准确性。此外，该系统在自主理解和执行超声程序方面显示出显著的潜力。这些进步使我们更接近实现自主的、可能是无人驾驶的机器人超声系统，解决医学成像中的关键资源缺口。 

---
# IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation 

**Title (ZH)**: IMLE策略：通过隐式最大似然估计实现快速和样本高效的空间知觉运动政策学习 

**Authors**: Krishan Rana, Robert Lee, David Pershouse, Niko Suenderhauf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12371)  

**Abstract**: Recent advances in imitation learning, particularly using generative modelling techniques like diffusion, have enabled policies to capture complex multi-modal action distributions. However, these methods often require large datasets and multiple inference steps for action generation, posing challenges in robotics where the cost for data collection is high and computation resources are limited. To address this, we introduce IMLE Policy, a novel behaviour cloning approach based on Implicit Maximum Likelihood Estimation (IMLE). IMLE Policy excels in low-data regimes, effectively learning from minimal demonstrations and requiring 38\% less data on average to match the performance of baseline methods in learning complex multi-modal behaviours. Its simple generator-based architecture enables single-step action generation, improving inference speed by 97.3\% compared to Diffusion Policy, while outperforming single-step Flow Matching. We validate our approach across diverse manipulation tasks in simulated and real-world environments, showcasing its ability to capture complex behaviours under data constraints. Videos and code are provided on our project page: this https URL. 

**Abstract (ZH)**: 最近在生成建模技术（如扩散模型）驱动的模仿学习方面的进展，使策略能够捕捉到复杂的多模态动作分布。然而，这些方法通常需要大量数据和多次推理步骤来进行动作生成，在机器人领域，数据收集成本高且计算资源有限，这带来了挑战。为了解决这个问题，我们提出了基于隐式最大似然估计（IMLE）的IMLE策略，这是一种新颖的行为克隆方法。IMLE策略在数据稀缺的情况下表现优异，能够有效地从少量示例中学习，并且平均只需要少38%的数据就能达到基线方法在学习复杂多模态行为时的性能。其基于生成器的简单架构允许一步生成动作，与扩散策略相比，将推理速度提高97.3%，同时优于一步流匹配方法。我们在模拟和真实环境中的多种操作任务中验证了该方法，展示了其在数据受限条件下捕捉复杂行为的能力。项目页面提供了相关的视频和代码：this https URL。 

---
# Hovering Flight of Soft-Actuated Insect-Scale Micro Aerial Vehicles using Deep Reinforcement Learning 

**Title (ZH)**: 软驱动昆虫尺度微型飞行器的悬浮飞行基于深度强化学习 

**Authors**: Yi-Hsuan Hsiao, Wei-Tung Chen, Yun-Sheng Chang, Pulkit Agrawal, YuFeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12355)  

**Abstract**: Soft-actuated insect-scale micro aerial vehicles (IMAVs) pose unique challenges for designing robust and computationally efficient controllers. At the millimeter scale, fast robot dynamics ($\sim$ms), together with system delay, model uncertainty, and external disturbances significantly affect flight performances. Here, we design a deep reinforcement learning (RL) controller that addresses system delay and uncertainties. To initialize this neural network (NN) controller, we propose a modified behavior cloning (BC) approach with state-action re-matching to account for delay and domain-randomized expert demonstration to tackle uncertainty. Then we apply proximal policy optimization (PPO) to fine-tune the policy during RL, enhancing performance and smoothing commands. In simulations, our modified BC substantially increases the mean reward compared to baseline BC; and RL with PPO improves flight quality and reduces command fluctuations. We deploy this controller on two different insect-scale aerial robots that weigh 720 mg and 850 mg, respectively. The robots demonstrate multiple successful zero-shot hovering flights, with the longest lasting 50 seconds and root-mean-square errors of 1.34 cm in lateral direction and 0.05 cm in altitude, marking the first end-to-end deep RL-based flight on soft-driven IMAVs. 

**Abstract (ZH)**: 软驱动昆虫尺度微型飞行器（IMAVs）的软化执行控制器设计提出独特挑战：面向鲁棒性和计算效率的控制设计。在毫米尺度上，快速机器人动力学（≈ms）、系统延迟、模型不确定性以及外部干扰严重影响飞行性能。在这里，我们设计了一种深度强化学习（RL）控制器，以应对系统延迟和不确定性。为了初始化这个神经网络（NN）控制器，我们提出了带有状态-动作重新匹配的修改行为 cloning（BC）方法，并利用域随机化专家演示来应对不确定性。然后我们使用近端策略优化（PPO）在RL过程中精细调整策略，提升性能并平滑命令。在仿真中，我们修改后的BC相比于基线BC显著提高了平均奖励；而使用PPO的RL进一步提升了飞行质量并减少了命令波动。我们将此控制器部署在两个不同重量的昆虫尺度微型飞行器上，分别为720 mg和850 mg。这些飞行器展现了多次成功的一次性悬停飞行，最长持续50秒，侧向方向的均方根误差为1.34 cm，海拔方向的均方根误差为0.05 cm，标志着软驱动IMAVs端到端深度RL控制的首次实现。 

---
# X-IL: Exploring the Design Space of Imitation Learning Policies 

**Title (ZH)**: X-IL：探索模仿学习策略的设计空间 

**Authors**: Xiaogang Jia, Atalay Donat, Xi Huang, Xuan Zhao, Denis Blessing, Hongyi Zhou, Hanyi Zhang, Han A. Wang, Qian Wang, Rudolf Lioutikov, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.12330)  

**Abstract**: Designing modern imitation learning (IL) policies requires making numerous decisions, including the selection of feature encoding, architecture, policy representation, and more. As the field rapidly advances, the range of available options continues to grow, creating a vast and largely unexplored design space for IL policies. In this work, we present X-IL, an accessible open-source framework designed to systematically explore this design space. The framework's modular design enables seamless swapping of policy components, such as backbones (e.g., Transformer, Mamba, xLSTM) and policy optimization techniques (e.g., Score-matching, Flow-matching). This flexibility facilitates comprehensive experimentation and has led to the discovery of novel policy configurations that outperform existing methods on recent robot learning benchmarks. Our experiments demonstrate not only significant performance gains but also provide valuable insights into the strengths and weaknesses of various design choices. This study serves as both a practical reference for practitioners and a foundation for guiding future research in imitation learning. 

**Abstract (ZH)**: 设计现代模仿学习（IL）策略需要做出众多决策，包括特征编码、架构、策略表示的选择等。随着该领域的快速发展，可供选择的范围不断扩大，为IL策略创造了广阔的、尚未充分探索的设计空间。在本文中，我们介绍了一种开源框架X-IL，旨在系统地探索这一设计空间。该框架模块化的设计使得可以无缝地替换策略组件，如骨干网络（例如，Transformer、Mamba、xLSTM）和策略优化技术（例如，Score-matching、Flow-matching）。这种灵活性促进了全面的实验，并发现了优于现有方法的新颖策略配置。我们的实验证明了显著的性能提升，并提供了关于各种设计选择的优缺点的重要见解。这项研究不仅为实践者提供了一个实用的参考，也为指导未来模仿学习的研究奠定了基础。 

---
# RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning 

**Title (ZH)**: RAD: 通过大规模3DGS为基础的强化学习训练端到端驾驶策略 

**Authors**: Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13144)  

**Abstract**: Existing end-to-end autonomous driving (AD) algorithms typically follow the Imitation Learning (IL) paradigm, which faces challenges such as causal confusion and the open-loop gap. In this work, we establish a 3DGS-based closed-loop Reinforcement Learning (RL) training paradigm. By leveraging 3DGS techniques, we construct a photorealistic digital replica of the real physical world, enabling the AD policy to extensively explore the state space and learn to handle out-of-distribution scenarios through large-scale trial and error. To enhance safety, we design specialized rewards that guide the policy to effectively respond to safety-critical events and understand real-world causal relationships. For better alignment with human driving behavior, IL is incorporated into RL training as a regularization term. We introduce a closed-loop evaluation benchmark consisting of diverse, previously unseen 3DGS environments. Compared to IL-based methods, RAD achieves stronger performance in most closed-loop metrics, especially 3x lower collision rate. Abundant closed-loop results are presented at this https URL. 

**Abstract (ZH)**: 基于3DGS的闭环强化学习训练 paradigm 在自主驾驶中的应用：克服imitation learning的挑战并实现更强的安全性和泛化能力 

---
# Magma: A Foundation Model for Multimodal AI Agents 

**Title (ZH)**: Magma：多模态AI代理的基础模型 

**Authors**: Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13130)  

**Abstract**: We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visual-spatial world (spatial-temporal intelligence) and complete agentic tasks ranging from UI navigation to robot manipulation. To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action planning. Extensive experiments show that SoM and ToM reach great synergy and facilitate the acquisition of spatial-temporal intelligence for our Magma model, which is fundamental to a wide range of tasks as shown in Fig.1. In particular, Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are specifically tailored to these tasks. On image and video-related multimodal tasks, Magma also compares favorably to popular large multimodal models that are trained on much larger datasets. We make our model and code public for reproducibility at this https URL. 

**Abstract (ZH)**: 我们呈现了Magma，一个服务于数字世界和物理世界多模态AI代理任务的基础模型。Magma是视觉语言（VL）模型的一个重要扩展，它不仅保留了后者在视觉语言理解方面的能力（言语智能），而且还具备在视觉空间世界中规划和执行任务的能力（时空智能），能够完成从UI导航到机器人操作等一系列代理任务。为赋予这些代理能力，Magma在从图像、视频到机器人数据等各种异质数据集上进行了预训练，在图像中的可操作视觉对象（如GUI中的可点击按钮）上通过Set-of-Mark (SoM)进行标注以实现动作接地，在视频中物体的运动轨迹（如人类手部或机器人手臂的轨迹）上通过Trace-of-Mark (ToM)进行标注以实现动作规划。大量实验表明，SoM和ToM能够产生巨大的协同效应，促进我们Magma模型在时空智能方面的获取，这对广泛的任务至关重要，如图1所示。特别是，Magma在UI导航和机器人操作任务上创造了新的最具表现力的结果，超越了专门为这些任务设计的先前模型。对于图像和视频相关的多模态任务，Magma也优于在更大数据集上训练的流行多模态模型。我们在此处公开了我们的模型和代码以确保可再现性。 

---
# Interactive Agents to Overcome Ambiguity in Software Engineering 

**Title (ZH)**: 克服软件工程中歧义性的交互式代理 

**Authors**: Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13069)  

**Abstract**: AI agents are increasingly being deployed to automate tasks, often based on ambiguous and underspecified user instructions. Making unwarranted assumptions and failing to ask clarifying questions can lead to suboptimal outcomes, safety risks due to tool misuse, and wasted computational resources. In this work, we study the ability of LLM agents to handle ambiguous instructions in interactive code generation settings by evaluating proprietary and open-weight models on their performance across three key steps: (a) leveraging interactivity to improve performance in ambiguous scenarios, (b) detecting ambiguity, and (c) asking targeted questions. Our findings reveal that models struggle to distinguish between well-specified and underspecified instructions. However, when models interact for underspecified inputs, they effectively obtain vital information from the user, leading to significant improvements in performance and underscoring the value of effective interaction. Our study highlights critical gaps in how current state-of-the-art models handle ambiguity in complex software engineering tasks and structures the evaluation into distinct steps to enable targeted improvements. 

**Abstract (ZH)**: AI代理在自动化任务中的模糊指令处理能力研究：通过评估proprietary和open-weight模型在交互式代码生成中的表现，考察其实现关键步骤的能力，包括利用互动性提高模糊场景下的性能、检测模糊性以及提出针对性的问题。研究发现模型难以区分明确和模糊的指令，但在处理输入模糊的任务时，通过互动有效获取了用户的重要信息，显著提升了性能，突显了有效互动的价值。本研究指出了当前最先进的模型在复杂软件工程任务中处理模糊性的重要缺陷，并将评估结构化为不同的步骤，以促进针对性的改进。 

---
# Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks 

**Title (ZH)**: 代理深度图推理构建自组织知识网络 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2502.13025)  

**Abstract**: We present an agentic, autonomous graph expansion framework that iteratively structures and refines knowledge in situ. Unlike conventional knowledge graph construction methods relying on static extraction or single-pass learning, our approach couples a reasoning-native large language model with a continually updated graph representation. At each step, the system actively generates new concepts and relationships, merges them into a global graph, and formulates subsequent prompts based on its evolving structure. Through this feedback-driven loop, the model organizes information into a scale-free network characterized by hub formation, stable modularity, and bridging nodes that link disparate knowledge clusters. Over hundreds of iterations, new nodes and edges continue to appear without saturating, while centrality measures and shortest path distributions evolve to yield increasingly distributed connectivity. Our analysis reveals emergent patterns, such as the rise of highly connected 'hub' concepts and the shifting influence of 'bridge' nodes, indicating that agentic, self-reinforcing graph construction can yield open-ended, coherent knowledge structures. Applied to materials design problems, we present compositional reasoning experiments by extracting node-specific and synergy-level principles to foster genuinely novel knowledge synthesis, yielding cross-domain ideas that transcend rote summarization and strengthen the framework's potential for open-ended scientific discovery. We discuss other applications in scientific discovery and outline future directions for enhancing scalability and interpretability. 

**Abstract (ZH)**: 一种自主的图扩展框架：迭代结构化和精炼知识 

---
# Integrating Reinforcement Learning, Action Model Learning, and Numeric Planning for Tackling Complex Tasks 

**Title (ZH)**: 结合强化学习、行动模型学习和数值规划以应对复杂任务 

**Authors**: Yarin Benyamin, Argaman Mordoch, Shahaf S. Shperberg, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2502.13006)  

**Abstract**: Automated Planning algorithms require a model of the domain that specifies the preconditions and effects of each action. Obtaining such a domain model is notoriously hard. Algorithms for learning domain models exist, yet it remains unclear whether learning a domain model and planning is an effective approach for numeric planning environments, i.e., where states include discrete and numeric state variables. In this work, we explore the benefits of learning a numeric domain model and compare it with alternative model-free solutions. As a case study, we use two tasks in Minecraft, a popular sandbox game that has been used as an AI challenge. First, we consider an offline learning setting, where a set of expert trajectories are available to learn from. This is the standard setting for learning domain models. We used the Numeric Safe Action Model Learning (NSAM) algorithm to learn a numeric domain model and solve new problems with the learned domain model and a numeric planner. We call this model-based solution NSAM_(+p), and compare it to several model-free Imitation Learning (IL) and Offline Reinforcement Learning (RL) algorithms. Empirical results show that some IL algorithms can learn faster to solve simple tasks, while NSAM_(+p) allows solving tasks that require long-term planning and enables generalizing to solve problems in larger environments. Then, we consider an online learning setting, where learning is done by moving an agent in the environment. For this setting, we introduce RAMP. In RAMP, observations collected during the agent's execution are used to simultaneously train an RL policy and learn a planning domain action model. This forms a positive feedback loop between the RL policy and the learned domain model. We demonstrate experimentally the benefits of using RAMP, showing that it finds more efficient plans and solves more problems than several RL baselines. 

**Abstract (ZH)**: 自动规划算法需要一个领域模型，该模型指定了每个操作的先决条件和效果。获得这样的领域模型非常困难。存在学习领域模型的算法，但仍不清楚在包含离散和数值状态变量的数值规划环境中，学习领域模型和规划是否是一种有效的方法。在本文中，我们探索学习数值领域模型的benefits，并将其与无模型解决方案进行比较。作为案例研究，我们在Minecraft中使用了两个任务，这是一个流行的沙盒游戏，曾被用作AI挑战。首先，我们考虑了一个离线学习设置，在该设置中，可用一组专家轨迹来学习。这是学习领域模型的标准设置。我们使用了Numeric Safe Action Model Learning (NSAM)算法来学习数值领域模型，并使用所学到的领域模型和数值规划器解决新问题。我们称之为基于模型的解决方案NSAM_(+p)，并将它与几种无模型的模仿学习（IL）和离线强化学习（RL）算法进行比较。实验结果表明，某些IL算法可以更快地学习解决简单任务，而NSAM_(+p)允许解决需要长期规划的任务，并能够泛化以在较大环境中解决问题。然后，我们考虑了一个在线学习设置，在此设置中，学习是通过在环境中移动代理来完成的。为此设置，我们引入了RAMP。在RAMP中，代理执行期间收集的观察结果被用作同时训练RL策略并学习规划领域动作模型的训练数据。这形成了一种RL策略与所学习领域模型之间的正反馈循环。我们实验展示了使用RAMP的好处，显示它找到了更有效的计划并解决了比几种RL基准更多的问题。 

---
# You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations 

**Title (ZH)**: 你需要效仿以获得名声：多Agent对话解决会议纪要短缺问题 

**Authors**: Frederic Kirstein, Muneeb Khan, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.13001)  

**Abstract**: Meeting summarization suffers from limited high-quality data, mainly due to privacy restrictions and expensive collection processes. We address this gap with FAME, a dataset of 500 meetings in English and 300 in German produced by MIMIC, our new multi-agent meeting synthesis framework that generates meeting transcripts on a given knowledge source by defining psychologically grounded participant profiles, outlining the conversation, and orchestrating a large language model (LLM) debate. A modular post-processing step refines these outputs, mitigating potential repetitiveness and overly formal tones, ensuring coherent, credible dialogues at scale. We also propose a psychologically grounded evaluation framework assessing naturalness, social behavior authenticity, and transcript difficulties. Human assessments show that FAME approximates real-meeting spontaneity (4.5/5 in naturalness), preserves speaker-centric challenges (3/5 in spoken language), and introduces richer information-oriented difficulty (4/5 in difficulty). These findings highlight that FAME is a good and scalable proxy for real-world meeting conditions. It enables new test scenarios for meeting summarization research and other conversation-centric applications in tasks requiring conversation data or simulating social scenarios under behavioral constraints. 

**Abstract (ZH)**: FAME：一种基于多智能体的会议合成框架，弥合会议总结数据不足的差距 

---
# CityEQA: A Hierarchical LLM Agent on Embodied Question Answering Benchmark in City Space 

**Title (ZH)**: CityEQA：城市空间中层次化大语言模型代理的实体问答基准 

**Authors**: Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, Jincai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12532)  

**Abstract**: Embodied Question Answering (EQA) has primarily focused on indoor environments, leaving the complexities of urban settings - spanning environment, action, and perception - largely unexplored. To bridge this gap, we introduce CityEQA, a new task where an embodied agent answers open-vocabulary questions through active exploration in dynamic city spaces. To support this task, we present CityEQA-EC, the first benchmark dataset featuring 1,412 human-annotated tasks across six categories, grounded in a realistic 3D urban simulator. Moreover, we propose Planner-Manager-Actor (PMA), a novel agent tailored for CityEQA. PMA enables long-horizon planning and hierarchical task execution: the Planner breaks down the question answering into sub-tasks, the Manager maintains an object-centric cognitive map for spatial reasoning during the process control, and the specialized Actors handle navigation, exploration, and collection sub-tasks. Experiments demonstrate that PMA achieves 60.7% of human-level answering accuracy, significantly outperforming frontier-based baselines. While promising, the performance gap compared to humans highlights the need for enhanced visual reasoning in CityEQA. This work paves the way for future advancements in urban spatial intelligence. Dataset and code are available at this https URL. 

**Abstract (ZH)**: 基于城市环境的实体化问答（CityEQA）：一种新的任务及其实验研究 

---
# Computational Safety for Generative AI: A Signal Processing Perspective 

**Title (ZH)**: 生成式AI的计算安全：一个信号处理视角 

**Authors**: Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12445)  

**Abstract**: AI safety is a rapidly growing area of research that seeks to prevent the harm and misuse of frontier AI technology, particularly with respect to generative AI (GenAI) tools that are capable of creating realistic and high-quality content through text prompts. Examples of such tools include large language models (LLMs) and text-to-image (T2I) diffusion models. As the performance of various leading GenAI models approaches saturation due to similar training data sources and neural network architecture designs, the development of reliable safety guardrails has become a key differentiator for responsibility and sustainability. This paper presents a formalization of the concept of computational safety, which is a mathematical framework that enables the quantitative assessment, formulation, and study of safety challenges in GenAI through the lens of signal processing theory and methods. In particular, we explore two exemplary categories of computational safety challenges in GenAI that can be formulated as hypothesis testing problems. For the safety of model input, we show how sensitivity analysis and loss landscape analysis can be used to detect malicious prompts with jailbreak attempts. For the safety of model output, we elucidate how statistical signal processing and adversarial learning can be used to detect AI-generated content. Finally, we discuss key open research challenges, opportunities, and the essential role of signal processing in computational AI safety. 

**Abstract (ZH)**: AI安全是快速发展的一个研究领域，旨在防止前沿AI技术带来的危害和滥用，尤其是在生成AI（GenAI）工具方面，这些工具能够通过文本提示生成真实性和高质量的内容。这类工具包括大型语言模型（LLMs）和文本到图像（T2I）扩散模型。随着各种领先的GenAI模型的性能接近饱和，由于类似的数据来源和神经网络架构设计，开发可靠的安全护栏已成为责任和可持续性的关键区别点。本文提出了计算安全这一概念的数学化，这是一种通过信号处理理论和方法来定量评估、形式化和研究GenAI安全挑战的数学框架。特别是，我们探讨了计算安全中两个可以形式化为假设检验问题的范例类别。在模型输入的安全方面，我们展示了如何使用灵敏度分析和损失景观分析来检测具有破解企图的恶意提示。在模型输出的安全方面，我们阐明了如何使用统计信号处理和对抗性学习来检测AI生成的内容。最后，我们讨论了关键的开放性研究挑战、机遇以及信号处理在计算AI安全中的核心作用。 

---
# MediaMind: Revolutionizing Media Monitoring using Agentification 

**Title (ZH)**: MediaMind: 通过代理化革命性地推动媒体监控 

**Authors**: Ahmet Gunduz, Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12745)  

**Abstract**: In an era of rapid technological advancements, agentification of software tools has emerged as a critical innovation, enabling systems to function autonomously and adaptively. This paper introduces MediaMind as a case study to demonstrate the agentification process, highlighting how existing software can be transformed into intelligent agents capable of independent decision-making and dynamic interaction. Developed by aiXplain, MediaMind leverages agent-based architecture to autonomously monitor, analyze, and provide insights from multilingual media content in real time. The focus of this paper is on the technical methodologies and design principles behind agentifying MediaMind, showcasing how agentification enhances adaptability, efficiency, and responsiveness. Through detailed case studies and practical examples, we illustrate how the agentification of MediaMind empowers organizations to streamline workflows, optimize decision-making, and respond to evolving trends. This work underscores the broader potential of agentification to revolutionize software tools across various domains. 

**Abstract (ZH)**: 在技术飞速发展的时代，软件工具的智能化已成为一项关键创新，使系统能够自主适应和动态交互。本文以MediaMind为例，展示智能化的过程，突出如何通过改造现有软件使其成为能够独立做出决策并进行动态交互的智能代理。由aiXplain开发的MediaMind利用基于代理的架构，能够实时监控、分析并提供多语言媒体内容的洞察。本文的重点在于MediaMind智能化的技术方法和设计原则，展示智能化如何增强系统的适应性、效率和响应性。通过详细的案例研究和实际示例，本文阐述了MediaMind的智能化如何使组织优化工作流程、增强决策效率并回应不断变化的趋势。本文强调了软件工具在各个领域通过智能化革命性的潜力。 

---
# Score-Based Diffusion Policy Compatible with Reinforcement Learning via Optimal Transport 

**Title (ZH)**: 基于分数扩散政策：通过最优传输与强化学习相兼容 

**Authors**: Mingyang Sun, Pengxiang Ding, Weinan Zhang, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12631)  

**Abstract**: Diffusion policies have shown promise in learning complex behaviors from demonstrations, particularly for tasks requiring precise control and long-term planning. However, they face challenges in robustness when encountering distribution shifts. This paper explores improving diffusion-based imitation learning models through online interactions with the environment. We propose OTPR (Optimal Transport-guided score-based diffusion Policy for Reinforcement learning fine-tuning), a novel method that integrates diffusion policies with RL using optimal transport theory. OTPR leverages the Q-function as a transport cost and views the policy as an optimal transport map, enabling efficient and stable fine-tuning. Moreover, we introduce masked optimal transport to guide state-action matching using expert keypoints and a compatibility-based resampling strategy to enhance training stability. Experiments on three simulation tasks demonstrate OTPR's superior performance and robustness compared to existing methods, especially in complex and sparse-reward environments. In sum, OTPR provides an effective framework for combining IL and RL, achieving versatile and reliable policy learning. The code will be released at this https URL. 

**Abstract (ZH)**: 基于最优传输的评分扩散政策在强化学习微调中的应用：改善分布偏移下的 imitation 学习 

---
# A Graph-Enhanced Deep-Reinforcement Learning Framework for the Aircraft Landing Problem 

**Title (ZH)**: 一种增强图表示的深度强化学习框架用于航空器降落问题 

**Authors**: Vatsal Maru  

**Link**: [PDF](https://arxiv.org/pdf/2502.12617)  

**Abstract**: The Aircraft Landing Problem (ALP) is one of the challenging problems in aircraft transportation and management. The challenge is to schedule the arriving aircraft in a sequence so that the cost and delays are optimized. There are various solution approaches to solving this problem, most of which are based on operations research algorithms and meta-heuristics. Although traditional methods perform better on one or the other factors, there remains a problem of solving real-time rescheduling and computational scalability altogether. This paper presents a novel deep reinforcement learning (DRL) framework that combines graph neural networks with actor-critic architectures to address the ALP. This paper introduces three key contributions: A graph-based state representation that efficiently captures temporal and spatial relationships between aircraft, a specialized actor-critic architecture designed to handle multiple competing objectives in landing scheduling, and a runway balance strategy that ensures efficient resource utilization while maintaining safety constraints. The results show that the trained algorithm can be tested on different problem sets and the results are competitive to operation research algorithms. The experimental results on standard benchmark data sets demonstrate a 99.95 reduction in computational time compared to Mixed Integer Programming (MIP) and 38 higher runway throughput over First Come First Serve (FCFS) approaches. Therefore, the proposed solution is competitive to traditional approaches and achieves substantial advancements. Notably, it does not require retraining, making it particularly suitable for industrial deployment. The frameworks capability to generate solutions within 1 second enables real-time rescheduling, addressing critical requirements of air traffic management. 

**Abstract (ZH)**: 基于图神经网络的深强化学习在航空着陆问题中的应用 

---
# From Abstract to Actionable: Pairwise Shapley Values for Explainable AI 

**Title (ZH)**: 从抽象到可行：成对谢普利值实现可解释AI 

**Authors**: Jiaxin Xu, Hung Chau, Angela Burden  

**Link**: [PDF](https://arxiv.org/pdf/2502.12525)  

**Abstract**: Explainable AI (XAI) is critical for ensuring transparency, accountability, and trust in machine learning systems as black-box models are increasingly deployed within high-stakes domains. Among XAI methods, Shapley values are widely used for their fairness and consistency axioms. However, prevalent Shapley value approximation methods commonly rely on abstract baselines or computationally intensive calculations, which can limit their interpretability and scalability. To address such challenges, we propose Pairwise Shapley Values, a novel framework that grounds feature attributions in explicit, human-relatable comparisons between pairs of data instances proximal in feature space. Our method introduces pairwise reference selection combined with single-value imputation to deliver intuitive, model-agnostic explanations while significantly reducing computational overhead. Here, we demonstrate that Pairwise Shapley Values enhance interpretability across diverse regression and classification scenarios--including real estate pricing, polymer property prediction, and drug discovery datasets. We conclude that the proposed methods enable more transparent AI systems and advance the real-world applicability of XAI. 

**Abstract (ZH)**: 可解释的人工智能（XAI）对于确保在高风险领域部署的黑盒模型具有透明性、问责制和信任至关重要。在XAI方法中，舍勒值由于其公平性和一致性公理而被广泛使用。然而，常见的舍勒值近似方法通常依赖于抽象的基本面或计算强度大的计算，这会限制其可解释性和可扩展性。为了应对这些挑战，我们提出了一种新的Pairwise舍勒值框架，该框架将特征归因基于特征空间中临近数据实例之间明确的人类可理解的对比。我们的方法结合了成对参考选择与单一值插补，以提供直观且模型无关的解释，同时显著降低了计算开销。我们证明Pairwise舍勒值在各种回归和分类场景中增强了可解释性，包括房地产定价、聚合物性质预测和药物发现数据集。我们得出结论，所提出的方 法能够实现更透明的人工智能系统，并推进XAI在实际应用中的适用性。 

---
# LM Agents for Coordinating Multi-User Information Gathering 

**Title (ZH)**: 多用户信息搜集协调的LM代理 

**Authors**: Harsh Jhamtani, Jacob Andreas, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.12328)  

**Abstract**: This paper introduces PeopleJoin, a benchmark for evaluating LM-mediated collaborative problem solving. Given a user request, PeopleJoin agents must identify teammates who might be able to assist, converse with these teammates to gather information, and finally compile a useful answer or summary for the original user. PeopleJoin comprises two evaluation domains: PeopleJoin-QA, focused on questions about tabular data, and PeopleJoin-DocCreation, focused on document creation tasks. The two domains are adapted from existing NLP benchmarks for database question answering and multi-document summarization; here, however, the information needed to complete these tasks is distributed across synthetic ``organizations'' of 2--20 users, simulating natural multi-user collaboration scenarios. We implemented several popular LM agent architectures, evaluating their accuracy and efficiency at completing tasks, and highlight new research questions that can be studied using PeopleJoin. 

**Abstract (ZH)**: PeopleJoin：一种评价LM介导协作问题解决能力的基准 

---
# NeuroStrata: Harnessing Neurosymbolic Paradigms for Improved Design, Testability, and Verifiability of Autonomous CPS 

**Title (ZH)**: NeuroStrata: 利用神经符号范式以提高自主 CPS 的设计、可测试性和可验证性 

**Authors**: Xi Zheng, Ziyang Li, Ivan Ruchkin, Ruzica Piskac, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2502.12267)  

**Abstract**: Autonomous cyber-physical systems (CPSs) leverage AI for perception, planning, and control but face trust and safety certification challenges due to inherent uncertainties. The neurosymbolic paradigm replaces stochastic layers with interpretable symbolic AI, enabling determinism. While promising, challenges like multisensor fusion, adaptability, and verification remain. This paper introduces NeuroStrata, a neurosymbolic framework to enhance the testing and verification of autonomous CPS. We outline its key components, present early results, and detail future plans. 

**Abstract (ZH)**: 自主认知物理系统（CPS）利用AI进行感知、规划和控制，但由于固有的不确定性，面临着信任和安全认证的挑战。神经符号范式用可解释的符号AI替代随机层，从而实现确定性。虽然具有前景，但仍面临多传感器融合、适应性和验证等方面的挑战。本文介绍了NeuroStrata，一个神经符号框架，用于增强自主CPS的测试和验证。我们概述了其关键组件，展示了早期结果，并详细说明了未来计划。 

---
# IMPACTX: Improving Model Performance by Appropriately predicting CorrecT eXplanations 

**Title (ZH)**: IMPACTX: 通过适当预测正确解释来提高模型性能 

**Authors**: Andrea Apicella, Salvatore Giugliano, Francesco Isgrò, Roberto Prevete  

**Link**: [PDF](https://arxiv.org/pdf/2502.12222)  

**Abstract**: The eXplainable Artificial Intelligence (XAI) research predominantly concentrates to provide explainations about AI model decisions, especially Deep Learning (DL) models. However, there is a growing interest in using XAI techniques to automatically improve the performance of the AI systems themselves.
This paper proposes IMPACTX, a novel approach that leverages XAI as a fully automated attention mechanism, without requiring external knowledge or human feedback. Experimental results show that IMPACTX has improved performance respect to the standalone ML model by integrating an attention mechanism based an XAI method outputs during the model training. Furthermore, IMPACTX directly provides proper feature attribution maps for the model's decisions, without relying on external XAI methods during the inference process.
Our proposal is evaluated using three widely recognized DL models (EfficientNet-B2, MobileNet, and LeNet-5) along with three standard image datasets: CIFAR-10, CIFAR-100, and STL-10. The results show that IMPACTX consistently improves the performance of all the inspected DL models across all evaluated datasets, and it directly provides appropriate explanations for its responses. 

**Abstract (ZH)**: 可解释的人工智能（XAI）研究主要集中在提供关于AI模型决策的解释，尤其是深度学习（DL）模型。然而，越来越多的研究兴趣在于使用XAI技术自动提升AI系统的性能本身。

本文提出了一种名为IMPACTX的新型方法，该方法利用XAI作为完全自动化的注意力机制，无需外部知识或人工反馈。实验结果表明，IMPACTX在通过XAI方法输出集成注意力机制进行模型训练后，相对于独立的机器学习模型具有更好的性能。此外，IMPACTX直接提供适合的特征归因图，用于模型决策，无需在推断过程中依赖外部XAI方法。

该提案使用三种广泛认可的深度学习模型（EfficientNet-B2、MobileNet和LeNet-5）以及三种标准图像数据集（CIFAR-10、CIFAR-100和STL-10）进行评估。结果表明，无论在何种数据集上，IMPACTX都能一致地提高所有受检深度学习模型的性能，并直接提供适当的操作解释。 

---
