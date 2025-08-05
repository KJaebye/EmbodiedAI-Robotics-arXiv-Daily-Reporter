# Manip4Care: Robotic Manipulation of Human Limbs for Solving Assistive Tasks 

**Title (ZH)**: Manip4Care: 人为肢体的机器人操作以解决辅助任务 

**Authors**: Yubin Koh, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02649)  

**Abstract**: Enabling robots to grasp and reposition human limbs can significantly enhance their ability to provide assistive care to individuals with severe mobility impairments, particularly in tasks such as robot-assisted bed bathing and dressing. However, existing assistive robotics solutions often assume that the human remains static or quasi-static, limiting their effectiveness. To address this issue, we present Manip4Care, a modular simulation pipeline that enables robotic manipulators to grasp and reposition human limbs effectively. Our approach features a physics simulator equipped with built-in techniques for grasping and repositioning while considering biomechanical and collision avoidance constraints. Our grasping method employs antipodal sampling with force closure to grasp limbs, and our repositioning system utilizes the Model Predictive Path Integral (MPPI) and vector-field-based control method to generate motion trajectories under collision avoidance and biomechanical constraints. We evaluate this approach across various limb manipulation tasks in both supine and sitting positions and compare outcomes for different age groups with differing shoulder joint limits. Additionally, we demonstrate our approach for limb manipulation using a real-world mannequin and further showcase its effectiveness in bed bathing tasks. 

**Abstract (ZH)**: 使机器人能够抓握和重新定位人类肢体可以显著增强它们为严重行动障碍个体提供辅助护理的能力，特别是在辅助沐浴和 dressing 等任务中。然而，现有的辅助机器人解决方案通常假设人类保持静止或准静止状态，限制了其有效性。为了解决这一问题，我们提出了 Manip4Care，一个模块化的仿真流水线，使机器人操作器能够有效抓握和重新定位人类肢体。我们的方法配备了一个具有抓握和重新定位内置技术的物理学仿真器，并考虑了生物力学和碰撞避免约束。我们的抓握方法采用了反握采样与力闭合技术来抓取肢体，而我们的重新定位系统则利用模型预测路径积分（MPPI）和基于向量场的控制方法，在避免碰撞和生物力学约束条件下生成运动轨迹。我们在仰卧和坐姿位置下的多种肢体操作任务中评估了这种方法，并对不同年龄组和不同的肩关节限制进行了比较。此外，我们使用实物人形模特展示了肢体操作的方法，并进一步展示了其在辅助沐浴任务中的有效性。 

---
# HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents 

**Title (ZH)**: HyCodePolicy: 混合语言控制器在具身智能体的多模态监测与决策中的应用 

**Authors**: Yibin Liu, Zhixuan Liang, Zanxin Chen, Tianxing Chen, Mengkang Hu, Wanxi Dong, Congsheng Xu, Zhaoming Han, Yusen Qin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02629)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines. 

**Abstract (ZH)**: 近期多模态大型语言模型的发展为体感代理的代码策略生成提供了更丰富的知觉基础。然而，现有系统大多缺乏有效的机制来适应性地监控策略执行并修复代码以完成任务。在本工作中，我们提出了HyCodePolicy，这是一种综合语言控制框架，系统地将代码合成、几何约束、知觉监控和迭代修复整合到体感代理的闭环编程循环中。技术上，给定自然语言指令，我们的系统首先将其分解为子目标，并生成基于以对象为中心的几何原语的初始可执行程序。该程序随后在模拟中执行，同时通过视觉语言模型(VLM)观察选定的检查点以检测和定位执行失败并推断失败原因。通过融合捕捉程序级事件的结构化执行轨迹与VLM基于的知觉反馈，HyCodePolicy推断失败原因并修复程序。这种混合双反馈机制能够在最少的人工监督下实现自纠正的程序合成。我们的结果表明，HyCodePolicy显著提高了机器人操作策略的鲁棒性和样本效率，提供了一种将多模态推理整合到自主决策管道中的可扩展策略。 

---
# Vision-based Navigation of Unmanned Aerial Vehicles in Orchards: An Imitation Learning Approach 

**Title (ZH)**: 基于视觉的果园无人无人机导航：一种模仿学习方法 

**Authors**: Peng Wei, Prabhash Ragbir, Stavros G. Vougioukas, Zhaodan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02617)  

**Abstract**: Autonomous unmanned aerial vehicle (UAV) navigation in orchards presents significant challenges due to obstacles and GPS-deprived environments. In this work, we introduce a learning-based approach to achieve vision-based navigation of UAVs within orchard rows. Our method employs a variational autoencoder (VAE)-based controller, trained with an intervention-based learning framework that allows the UAV to learn a visuomotor policy from human experience. We validate our approach in real orchard environments with a custom-built quadrotor platform. Field experiments demonstrate that after only a few iterations of training, the proposed VAE-based controller can autonomously navigate the UAV based on a front-mounted camera stream. The controller exhibits strong obstacle avoidance performance, achieves longer flying distances with less human assistance, and outperforms existing algorithms. Furthermore, we show that the policy generalizes effectively to novel environments and maintains competitive performance across varying conditions and speeds. This research not only advances UAV autonomy but also holds significant potential for precision agriculture, improving efficiency in orchard monitoring and management. 

**Abstract (ZH)**: 基于学习的视觉导航在果园中自主无人机飞行面临显著挑战，因为空中障碍和GPS受限环境。本文提出了一种基于学习的方法，实现无人机在果园行间基于视觉的自主导航。该方法采用基于变分自编码器（VAE）的控制器，并通过基于干预的学习框架进行训练，使无人机能够从人类经验中学习视觉运动策略。我们在自建的四旋翼平台上在真实果园环境中验证了该方法。实地实验表明，在几次训练迭代后，提出的基于VAE的控制器能够基于前向安装的摄像头流自主导航无人机，并表现出强大的避障性能，在较少的人工干预下实现了更远的飞行距离，且优于现有算法。此外，我们还展示了该策略在新颖环境中的有效推广，并在不同条件和速度下保持竞争力。这项研究不仅推动了无人机自主性的发展，还对精准农业具有重要意义，有助于提高果园监测和管理的效率。 

---
# Periodic robust robotic rock chop via virtual model control 

**Title (ZH)**: 周期性鲁棒岩石切割的虚拟模型控制 

**Authors**: Yi Zhang, Fumiya Iida, Fulvio Forni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02604)  

**Abstract**: Robotic cutting is a challenging contact-rich manipulation task where the robot must simultaneously negotiate unknown object mechanics, large contact forces, and precise motion requirements. We introduce a new virtual-model control scheme that enables knife rocking motion for robot manipulators, without pre-planned trajectories or precise information of the environment. Motion is generated through interconnection with virtual mechanisms, given by virtual springs, dampers, and masses arranged in a suitable way. Through analysis and experiments, we demonstrate that the controlled robot behavior settles into a periodic motion. Experiments with a Franka manipulator demonstrate robust cuts with five different vegetables, and sub-millimeter slice accuracy from 1 mm to 6 mm at nearly one cut per second. The same controller survives changes in knife shape and cutting board height, and adaptation to a different humanoid manipulator, demonstrating robustness and platform independence. 

**Abstract (ZH)**: 机器人切削是一项接触丰富的操作任务，其中机器人必须同时应对未知物体的机械特性、大的接触力以及精确的运动要求。我们提出了一种新的虚拟模型控制方案，使机器人 manipulator 能够实现刀具摇动运动，无需预先规划轨迹或精确的环境信息。运动通过与虚拟弹簧、阻尼器和质量的相互连接生成。通过分析和实验，我们证明控制的机器人行为会稳定在周期性运动中。使用 Franka maniuplators 的实验展示了对五种不同蔬菜的稳健切割，并实现了从 1 mm 到 6 mm 的亚毫米级切片精度，几乎每秒一次切割。相同的控制器能够适应刀具形状的变化、切割板高度的变化，并适用于不同的类人 manipulator，显示了其稳健性和平台独立性。 

---
# An RGB-D Camera-Based Multi-Small Flying Anchors Control for Wire-Driven Robots Connecting to the Environment 

**Title (ZH)**: 基于RGB-D摄像机的多小型悬挂锚点控制方法及其在环境连接的线驱动机器人中的应用 

**Authors**: Shintaro Inoue, Kento Kawaharazuka, Keita Yoneda, Sota Yuzaki, Yuta Sahara, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2508.02544)  

**Abstract**: In order to expand the operational range and payload capacity of robots, wire-driven robots that leverage the external environment have been proposed. It can exert forces and operate in spaces far beyond those dictated by its own structural limits. However, for practical use, robots must autonomously attach multiple wires to the environment based on environmental recognition-an operation so difficult that many wire-driven robots remain restricted to specialized, pre-designed environments. Here, in this study, we propose a robot that autonomously connects multiple wires to the environment by employing a multi-small flying anchor system, as well as an RGB-D camera-based control and environmental recognition method. Each flying anchor is a drone with an anchoring mechanism at the wire tip, allowing the robot to attach wires by flying into position. Using the robot's RGB-D camera to identify suitable attachment points and a flying anchor position, the system can connect wires in environments that are not specially prepared, and can also attach multiple wires simultaneously. Through this approach, a wire-driven robot can autonomously attach its wires to the environment, thereby realizing the benefits of wire-driven operation at any location. 

**Abstract (ZH)**: 基于多小型飞行锚系统和RGB-D相机控制与环境识别的环境自主连接多条牵引线的机器人 

---
# Failure-Aware Multi-Robot Coordination for Resilient and Adaptive Target Tracking 

**Title (ZH)**: 面向故障的多机器人协调以实现鲁棒性和适应性目标跟踪 

**Authors**: Peihan Li, Jiazhen Liu, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02529)  

**Abstract**: Multi-robot coordination is crucial for autonomous systems, yet real-world deployments often encounter various failures. These include both temporary and permanent disruptions in sensing and communication, which can significantly degrade system robustness and performance if not explicitly modeled. Despite its practical importance, failure-aware coordination remains underexplored in the literature. To bridge the gap between idealized conditions and the complexities of real-world environments, we propose a unified failure-aware coordination framework designed to enable resilient and adaptive multi-robot target tracking under both temporary and permanent failure conditions. Our approach systematically distinguishes between two classes of failures: (1) probabilistic and temporary disruptions, where robots recover from intermittent sensing or communication losses by dynamically adapting paths and avoiding inferred danger zones, and (2) permanent failures, where robots lose sensing or communication capabilities irreversibly, requiring sustained, decentralized behavioral adaptation. To handle these scenarios, the robot team is partitioned into subgroups. Robots that remain connected form a communication group and collaboratively plan using partially centralized nonlinear optimization. Robots experiencing permanent disconnection or failure continue to operate independently through decentralized or individual optimization, allowing them to contribute to the task within their local context. We extensively evaluate our method across a range of benchmark variations and conduct a comprehensive assessment under diverse real-world failure scenarios. Results show that our framework consistently achieves robust performance in realistic environments with unknown danger zones, offering a practical and generalizable solution for the multi-robot systems community. 

**Abstract (ZH)**: 面向故障的多机器人协调框架：在临时和永久故障条件下的稳健和适应性目标跟踪 

---
# QuaDreamer: Controllable Panoramic Video Generation for Quadruped Robots 

**Title (ZH)**: QuaDreamer: 可控全景视频生成方法在四足机器人中的应用 

**Authors**: Sheng Wu, Fei Teng, Hao Shi, Qi Jiang, Kai Luo, Kaiwei Wang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02512)  

**Abstract**: Panoramic cameras, capturing comprehensive 360-degree environmental data, are suitable for quadruped robots in surrounding perception and interaction with complex environments. However, the scarcity of high-quality panoramic training data-caused by inherent kinematic constraints and complex sensor calibration challenges-fundamentally limits the development of robust perception systems tailored to these embodied platforms. To address this issue, we propose QuaDreamer-the first panoramic data generation engine specifically designed for quadruped robots. QuaDreamer focuses on mimicking the motion paradigm of quadruped robots to generate highly controllable, realistic panoramic videos, providing a data source for downstream tasks. Specifically, to effectively capture the unique vertical vibration characteristics exhibited during quadruped locomotion, we introduce Vertical Jitter Encoding (VJE). VJE extracts controllable vertical signals through frequency-domain feature filtering and provides high-quality prompts. To facilitate high-quality panoramic video generation under jitter signal control, we propose a Scene-Object Controller (SOC) that effectively manages object motion and boosts background jitter control through the attention mechanism. To address panoramic distortions in wide-FoV video generation, we propose the Panoramic Enhancer (PE)-a dual-stream architecture that synergizes frequency-texture refinement for local detail enhancement with spatial-structure correction for global geometric consistency. We further demonstrate that the generated video sequences can serve as training data for the quadruped robot's panoramic visual perception model, enhancing the performance of multi-object tracking in 360-degree scenes. The source code and model weights will be publicly available at this https URL. 

**Abstract (ZH)**: 全景相机捕获全方位360度环境数据，适合四足机器人在环境感知和与复杂环境互动中的应用。然而，由于固有的运动学约束和传感器校准挑战导致的高质量全景训练数据稀缺性，从根本上限制了针对这些实体平台的鲁棒感知系统的开发。为了解决这一问题，我们提出QuaDreamer——首个专门设计用于四足机器人的全景数据生成引擎。QuaDreamer旨在模拟四足机器人的运动模式，生成高度可控、真实的全景视频，为下游任务提供数据源。具体而言，为了有效捕捉四足行走过程中特有的垂直振动特性，我们引入了垂直抖动编码（VJE）。VJE通过频域特征过滤提取可控制的垂直信号，并提供高质量的提示。为了在抖动信号控制下生成高质量的全景视频，我们提出了一种场景-对象控制器（SOC），该控制器通过注意机制有效管理对象运动并增强背景抖动控制。为了解决宽视场视频生成中的全景失真问题，我们提出了一种全景增强器（PE），这是一种双流架构，能够通过局部细节增强和全局几何一致性校正来协同实现频率-纹理细化和空间-结构校正。我们进一步证明，生成的视频序列可以作为四足机器人全景视觉感知模型的训练数据，提高360度场景中多目标跟踪的性能。源代码和模型权重将在该网址公开。 

---
# Would you let a humanoid play storytelling with your child? A usability study on LLM-powered narrative Humanoid-Robot Interaction 

**Title (ZH)**: 你会让类人机器人给你孩子讲故事吗？基于大语言模型的叙事类人机器人-机器人交互易用性研究 

**Authors**: Maria Lombardi, Carmela Calabrese, Davide Ghiglino, Caterina Foglino, Davide De Tommaso, Giulia Da Lisca, Lorenzo Natale, Agnieszka Wykowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.02505)  

**Abstract**: A key challenge in human-robot interaction research lies in developing robotic systems that can effectively perceive and interpret social cues, facilitating natural and adaptive interactions. In this work, we present a novel framework for enhancing the attention of the iCub humanoid robot by integrating advanced perceptual abilities to recognise social cues, understand surroundings through generative models, such as ChatGPT, and respond with contextually appropriate social behaviour. Specifically, we propose an interaction task implementing a narrative protocol (storytelling task) in which the human and the robot create a short imaginary story together, exchanging in turn cubes with creative images placed on them. To validate the protocol and the framework, experiments were performed to quantify the degree of usability and the quality of experience perceived by participants interacting with the system. Such a system can be beneficial in promoting effective human robot collaborations, especially in assistance, education and rehabilitation scenarios where the social awareness and the robot responsiveness play a pivotal role. 

**Abstract (ZH)**: 人类与机器人交互研究中的一个重要挑战在于开发能够有效感知和解读社交线索的机器人系统，以促进自然且适应性的交互。本文提出了一种新的框架，通过集成先进的感知能力来增强iCub人形机器人的注意能力，利用生成模型（如ChatGPT）理解周围环境，并以情境合适的社会行为作出响应。具体而言，我们提出了一项交互任务，实施基于叙述协议（故事讲述任务），人类与机器人一起创作一段简短的想象故事，轮流交换带有创意图像的立方体。为了验证协议和框架的有效性，进行了实验以量化参与者在使用系统时的易用性和感知体验质量。此类系统在促进有效的机器人协作方面颇具益处，尤其是在需要社会意识和机器人响应能力的辅助、教育和康复场景中。 

---
# Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing 

**Title (ZH)**: 基于本体感觉的机器人 manipulator 多类人类/物体检测 

**Authors**: Justin Hehli, Marco Heiniger, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.02425)  

**Abstract**: In physical human-robot collaboration (pHRC) settings, humans and robots collaborate directly in shared environments. Robots must analyze interactions with objects to ensure safety and facilitate meaningful workflows. One critical aspect is human/object detection, where the contacted object is identified. Past research introduced binary machine learning classifiers to distinguish between soft and hard objects. This study improves upon those results by evaluating three-class human/object detection models, offering more detailed contact analysis. A dataset was collected using the Franka Emika Panda robot manipulator, exploring preprocessing strategies for time-series analysis. Models including LSTM, GRU, and Transformers were trained on these datasets. The best-performing model achieved 91.11\% accuracy during real-time testing, demonstrating the feasibility of multi-class detection models. Additionally, a comparison of preprocessing strategies suggests a sliding window approach is optimal for this task. 

**Abstract (ZH)**: 在物理人机协作（pHRC）环境中的人类/机器人协作中，人类和机器人在共享环境中直接协作。机器人必须分析与物体的交互以确保安全并促进有意义的工作流程。一个关键方面是人类/物体检测，其中需要识别被接触的物体。以往研究引入了二元机器学习分类器来区分软物体和硬物体。本研究在此基础上通过评估三类人类/物体检测模型来改进以往成果，提供更详细的接触分析。使用Franka Emika Panda 机器人操作器收集数据，探索时间序列分析的预处理策略。在这些数据集上训练了包括LSTM、GRU和变换器在内的模型。最佳模型在实时测试中达到91.11%的准确率，证明了多类检测模型的可行性。此外，预处理策略比较表明滑动窗口方法在这种任务中是最优的。 

---
# Improving Generalization of Language-Conditioned Robot Manipulation 

**Title (ZH)**: 改进基于语言条件的机器人操作的泛化能力 

**Authors**: Chenglin Cui, Chaoran Zhu, Changjae Oh, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.02405)  

**Abstract**: The control of robots for manipulation tasks generally relies on visual input. Recent advances in vision-language models (VLMs) enable the use of natural language instructions to condition visual input and control robots in a wider range of environments. However, existing methods require a large amount of data to fine-tune VLMs for operating in unseen environments. In this paper, we present a framework that learns object-arrangement tasks from just a few demonstrations. We propose a two-stage framework that divides object-arrangement tasks into a target localization stage, for picking the object, and a region determination stage for placing the object. We present an instance-level semantic fusion module that aligns the instance-level image crops with the text embedding, enabling the model to identify the target objects defined by the natural language instructions. We validate our method on both simulation and real-world robotic environments. Our method, fine-tuned with a few demonstrations, improves generalization capability and demonstrates zero-shot ability in real-robot manipulation scenarios. 

**Abstract (ZH)**: 基于少量示范学习物体排列任务的框架 

---
# Adaptive Lattice-based Motion Planning 

**Title (ZH)**: 自适应格子基运动规划 

**Authors**: Abhishek Dhar, Sarthak Mishra, Spandan Roy, Daniel Axehill  

**Link**: [PDF](https://arxiv.org/pdf/2508.02350)  

**Abstract**: This paper proposes an adaptive lattice-based motion planning solution to address the problem of generating feasible trajectories for systems, represented by a linearly parameterizable non-linear model operating within a cluttered environment. The system model is considered to have uncertain model parameters. The key idea here is to utilize input/output data online to update the model set containing the uncertain system parameter, as well as a dynamic estimated parameter of the model, so that the associated model estimation error reduces over time. This in turn improves the quality of the motion primitives generated by the lattice-based motion planner using a nominal estimated model selected on the basis of suitable criteria. The motion primitives are also equipped with tubes to account for the model mismatch between the nominal estimated model and the true system model, to guarantee collision-free overall motion. The tubes are of uniform size, which is directly proportional to the size of the model set containing the uncertain system parameter. The adaptive learning module guarantees a reduction in the diameter of the model set as well as in the parameter estimation error between the dynamic estimated parameter and the true system parameter. This directly implies a reduction in the size of the implemented tubes and guarantees that the utilized motion primitives go arbitrarily close to the resolution-optimal motion primitives associated with the true model of the system, thus significantly improving the overall motion planning performance over time. The efficiency of the motion planner is demonstrated by a suitable simulation example that considers a drone model represented by Euler-Lagrange dynamics containing uncertain parameters and operating within a cluttered environment. 

**Abstract (ZH)**: 基于自适应格形的运动规划方案：解决具不确定参数的非线性模型在复杂环境下的可行轨迹生成问题 

---
# Framework for Robust Motion Planning of Tethered Multi-Robot Systems in Marine Environments 

**Title (ZH)**: tethered多机器人系统在海洋环境中的鲁棒运动规划框架 

**Authors**: Markus Buchholz, Ignacio Carlucho, Zebin Huang, Michele Grimaldi, Pierre Nicolay, Sumer Tuncay, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2508.02287)  

**Abstract**: This paper introduces CoralGuide, a novel framework designed for path planning and trajectory optimization for tethered multi-robot systems. We focus on marine robotics, which commonly have tethered configurations of an Autonomous Surface Vehicle (ASV) and an Autonomous Underwater Vehicle (AUV). CoralGuide provides safe navigation in marine environments by enhancing the A* algorithm with specialized heuristics tailored for tethered ASV-AUV systems. Our method integrates catenary curve modelling for tether management and employs Bezier curve interpolation for smoother trajectory planning, ensuring efficient and synchronized operations without compromising safety. Through simulations and real-world experiments, we have validated CoralGuides effectiveness in improving path planning and trajectory optimization, demonstrating its potential to significantly enhance operational capabilities in marine research and infrastructure inspection. 

**Abstract (ZH)**: CoralGuide：一种用于 tethered 多机器人系统路径规划与轨迹优化的新框架 

---
# Tethered Multi-Robot Systems in Marine Environments 

**Title (ZH)**: 海洋环境中的 tethered 多机器人系统 

**Authors**: Markus Buchholz, Ignacio Carlucho, Michele Grimaldi, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2508.02264)  

**Abstract**: This paper introduces a novel simulation framework for evaluating motion control in tethered multi-robot systems within dynamic marine environments. Specifically, it focuses on the coordinated operation of an Autonomous Underwater Vehicle (AUV) and an Autonomous Surface Vehicle(ASV). The framework leverages GazeboSim, enhanced with realistic marine environment plugins and ArduPilots SoftwareIn-The-Loop (SITL) mode, to provide a high-fidelity simulation platform. A detailed tether model, combining catenary equations and physical simulation, is integrated to accurately represent the dynamic interactions between the vehicles and the environment. This setup facilitates the development and testing of advanced control strategies under realistic conditions, demonstrating the frameworks capability to analyze complex tether interactions and their impact on system performance. 

**Abstract (ZH)**: 本文介绍了一种用于评估缆绳约束多机器人系统在动态海洋环境中的运动控制的新模拟框架，具体聚焦于自主水下车辆(AUV)与自主水面车辆(ASV)的协调操作。该框架借助增强现实海洋环境插件的GazeboSim及ArduPilots软件在环(SITL)模式，提供了一个高保真度的模拟平台。结合缆绳方程和物理模拟的详细缆绳模型被集成进来，以准确表示车辆与环境之间的动态交互。该设置促进了在实际条件下先进控制策略的开发与测试，展示了该框架分析复杂缆绳交互及其对系统性能影响的能力。 

---
# CO-RFT: Efficient Fine-Tuning of Vision-Language-Action Models through Chunked Offline Reinforcement Learning 

**Title (ZH)**: CO-RFT: 通过分块离线强化学习高效微调视觉-语言-动作模型 

**Authors**: Dongchi Huang, Zhirui Fang, Tianle Zhang, Yihang Li, Lin Zhao, Chunhe Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.02219)  

**Abstract**: Vision-Language-Action (VLA) models demonstrate significant potential for developing generalized policies in real-world robotic control. This progress inspires researchers to explore fine-tuning these models with Reinforcement Learning (RL). However, fine-tuning VLA models with RL still faces challenges related to sample efficiency, compatibility with action chunking, and training stability. To address these challenges, we explore the fine-tuning of VLA models through offline reinforcement learning incorporating action chunking. In this work, we propose Chunked RL, a novel reinforcement learning framework specifically designed for VLA models. Within this framework, we extend temporal difference (TD) learning to incorporate action chunking, a prominent characteristic of VLA models. Building upon this framework, we propose CO-RFT, an algorithm aimed at fine-tuning VLA models using a limited set of demonstrations (30 to 60 samples). Specifically, we first conduct imitation learning (IL) with full parameter fine-tuning to initialize both the backbone and the policy. Subsequently, we implement offline RL with action chunking to optimize the pretrained policy. Our empirical results in real-world environments demonstrate that CO-RFT outperforms previous supervised methods, achieving a 57% improvement in success rate and a 22.3% reduction in cycle time. Moreover, our method exhibits robust positional generalization capabilities, attaining a success rate of 44.3% in previously unseen positions. 

**Abstract (ZH)**: Vision-LANGUAGE-Action (VLA)模型在现实机器人控制中开发通用策略方面展现出显著潜力。这种进展激励研究人员探索使用强化学习（RL）微调这些模型。然而，使用RL微调VLA模型仍然面临样本效率、与动作分块的兼容性以及训练稳定性等方面的问题。为应对这些挑战，我们通过结合动作分块的离线强化学习探索VLA模型的微调方法。在本工作中，我们提出了Chunked RL，这是一种专门为VLA模型设计的新型强化学习框架。在此框架内，我们将时间差分（TD）学习扩展以纳入动作分块，这是VLA模型的一个显著特征。基于此框架，我们提出了CO-RFT算法，该算法旨在使用有限数量的演示（30至60个样本）微调VLA模型。首先，我们使用完整的参数微调进行模仿学习（IL）以初始化骨干网络和策略。随后，我们应用结合动作分块的离线RL优化预训练策略。我们在真实环境中的实验证据表明，CO-RFT优于以往的监督方法，成功率达到57%的改进和22.3%的循环时间减少。此外，我们的方法展示了鲁棒的位置泛化能力，在未见过的位置实现了44.3%的成功率。 

---
# TacMan-Turbo: Proactive Tactile Control for Robust and Efficient Articulated Object Manipulation 

**Title (ZH)**: TacMan-Turbo: 主动触觉控制以实现 robust 和 efficient 的刚性对象操控 

**Authors**: Zihang Zhao, Zhenghao Qi, Yuyang Li, Leiyao Cui, Zhi Han, Lecheng Ruan, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02204)  

**Abstract**: Adept manipulation of articulated objects is essential for robots to operate successfully in human environments. Such manipulation requires both effectiveness -- reliable operation despite uncertain object structures -- and efficiency -- swift execution with minimal redundant steps and smooth actions. Existing approaches struggle to achieve both objectives simultaneously: methods relying on predefined kinematic models lack effectiveness when encountering structural variations, while tactile-informed approaches achieve robust manipulation without kinematic priors but compromise efficiency through reactive, step-by-step exploration-compensation cycles. This paper introduces TacMan-Turbo, a novel proactive tactile control framework for articulated object manipulation that resolves this fundamental trade-off. Unlike previous approaches that treat tactile contact deviations merely as error signals requiring compensation, our method interprets these deviations as rich sources of local kinematic information. This new perspective enables our controller to predict optimal future interactions and make proactive adjustments, significantly enhancing manipulation efficiency. In comprehensive evaluations across 200 diverse simulated articulated objects and real-world experiments, our approach maintains a 100% success rate while significantly outperforming the previous tactile-informed method in time efficiency, action efficiency, and trajectory smoothness (all p-values < 0.0001). These results demonstrate that the long-standing trade-off between effectiveness and efficiency in articulated object manipulation can be successfully resolved without relying on prior kinematic knowledge. 

**Abstract (ZH)**: 灵巧操作 articulated 物体是机器人在人类环境中成功操作的关键。这种操作既需要有效性——在遇到结构不确定性时能够可靠运行——也需要效率——快速执行且步骤最少、动作流畅。现有方法难以同时实现这两个目标：依赖预定义动力学模型的方法在遇到结构变化时有效性不足，而基于触觉的信息的方法不依赖动力学先验从而实现稳健操作，但通过反应性的、逐步的探索—补偿循环降低了效率。本文介绍了一种新的前瞻触觉控制框架 TacMan-Turbo，解决了这种基本的权衡问题。与以往方法仅将触觉接触偏差视作需要补偿的误差信号不同，我们的方法将这些偏差视为丰富的局部动力学信息来源。这种新的视角使我们的控制器能够预测最优的未来交互并采取积极的调整，显著提高了操作效率。在针对 200 种不同模拟 articulated 物体以及真实世界实验的全面评估中，我们的方法保持了 100% 的成功率，在时间效率、动作效率和轨迹流畅度方面显著优于之前的触觉信息方法（所有 p 值 < 0.0001）。这些结果表明，articulated 物体操作中的长期有效性与效率权衡可以通过不依赖先验动力学知识的方法成功解决。 

---
# Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot 

**Title (ZH)**: 约束强化学习在Bolt机器人不稳定的点足 bipedal 行走中的应用 

**Authors**: Constant Roux, Elliot Chane-Sane, Ludovic De Matteïs, Thomas Flayols, Jérôme Manhes, Olivier Stasse, Philippe Souères  

**Link**: [PDF](https://arxiv.org/pdf/2508.02194)  

**Abstract**: Bipedal locomotion is a key challenge in robotics, particularly for robots like Bolt, which have a point-foot design. This study explores the control of such underactuated robots using constrained reinforcement learning, addressing their inherent instability, lack of arms, and limited foot actuation. We present a methodology that leverages Constraints-as-Terminations and domain randomization techniques to enable sim-to-real transfer. Through a series of qualitative and quantitative experiments, we evaluate our approach in terms of balance maintenance, velocity control, and responses to slip and push disturbances. Additionally, we analyze autonomy through metrics like the cost of transport and ground reaction force. Our method advances robust control strategies for point-foot bipedal robots, offering insights into broader locomotion. 

**Abstract (ZH)**: 点足两足步行机器人控制的约束强化学习研究 

---
# FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation 

**Title (ZH)**: FedVLA：用于机器人操作的双门控混合专家联邦视觉-语言-动作学习 

**Authors**: Cui Miao, Tao Chang, Meihan Wu, Hongbin Xu, Chun Li, Ming Li, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02190)  

**Abstract**: Vision-language-action (VLA) models have significantly advanced robotic manipulation by enabling robots to interpret language instructions for task execution. However, training these models often relies on large-scale user-specific data, raising concerns about privacy and security, which in turn limits their broader adoption. To address this, we propose FedVLA, the first federated VLA learning framework, enabling distributed model training that preserves data privacy without compromising performance. Our framework integrates task-aware representation learning, adaptive expert selection, and expert-driven federated aggregation, enabling efficient and privacy-preserving training of VLA models. Specifically, we introduce an Instruction Oriented Scene-Parsing mechanism, which decomposes and enhances object-level features based on task instructions, improving contextual understanding. To effectively learn diverse task patterns, we design a Dual Gating Mixture-of-Experts (DGMoE) mechanism, where not only input tokens but also self-aware experts adaptively decide their activation. Finally, we propose an Expert-Driven Aggregation strategy at the federated server, where model aggregation is guided by activated experts, ensuring effective cross-client knowledge this http URL simulations and real-world robotic experiments demonstrate the effectiveness of our proposals. Notably, DGMoE significantly improves computational efficiency compared to its vanilla counterpart, while FedVLA achieves task success rates comparable to centralized training, effectively preserving data privacy. 

**Abstract (ZH)**: 联邦视觉-语言-行动（FedVLA）学习框架：分布式的高效和隐私保护机器人操作模型训练 

---
# A Moment Matching-Based Method for Sparse and Noisy Point Cloud Registration 

**Title (ZH)**: 基于矩匹配的稀疏且噪声点云配准方法 

**Authors**: Xingyi Li, Han Zhang, Ziliang Wang, Yukai Yang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02187)  

**Abstract**: Point cloud registration is a key step in robotic perception tasks, such as Simultaneous Localization and Mapping (SLAM). It is especially challenging in conditions with sparse points and heavy noise. Traditional registration methods, such as Iterative Closest Point (ICP) and Normal Distributions Transform (NDT), often have difficulties in achieving a robust and accurate alignment under these conditions. In this paper, we propose a registration framework based on moment matching. In particular, the point clouds are regarded as i.i.d. samples drawn from the same distribution observed in the source and target frames. We then match the generalized Gaussian Radial Basis moments calculated from the point clouds to estimate the rigid transformation between two frames. Moreover, such method does not require explicit point-to-point correspondences among the point clouds. We further show the consistency of the proposed method. Experiments on synthetic and real-world datasets show that our approach achieves higher accuracy and robustness than existing methods. In addition, we integrate our framework into a 4D Radar SLAM system. The proposed method significantly improves the localization performance and achieves results comparable to LiDAR-based systems. These findings demonstrate the potential of moment matching technique for robust point cloud registration in sparse and noisy scenarios. 

**Abstract (ZH)**: 基于矩匹配的点云注册框架及其在稀疏噪声条件下的应用 

---
# Towards High Precision: An Adaptive Self-Supervised Learning Framework for Force-Based Verification 

**Title (ZH)**: 面向高精度：基于自适应自监督学习框架的力基验证方法 

**Authors**: Zebin Duan, Frederik Hagelskjær, Aljaz Kramberger, Juan Heredia and, Norbert Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02153)  

**Abstract**: The automation of robotic tasks requires high precision and adaptability, particularly in force-based operations such as insertions. Traditional learning-based approaches either rely on static datasets, which limit their ability to generalize, or require frequent manual intervention to maintain good performances. As a result, ensuring long-term reliability without human supervision remains a significant challenge. To address this, we propose an adaptive self-supervised learning framework for insertion classification that continuously improves its precision over time. The framework operates in real-time, incrementally refining its classification decisions by integrating newly acquired force data. Unlike conventional methods, it does not rely on pre-collected datasets but instead evolves dynamically with each task execution. Through real-world experiments, we demonstrate how the system progressively reduces execution time while maintaining near-perfect precision as more samples are processed. This adaptability ensures long-term reliability in force-based robotic tasks while minimizing the need for manual intervention. 

**Abstract (ZH)**: 基于力的操作的机器人任务自动化需要高精度和适应性，传统的基于学习的方法要么依赖于静态数据集，限制了其泛化能力，要么需要频繁的手动干预以保持性能。因此，在无需人工监督的情况下确保长期可靠性仍然是一个重大挑战。为此，我们提出了一种适应性自我监督学习框架，用于插入分类，该框架能够随着时间的推移持续提高其精度。该框架实时运行，通过集成新获取的力数据，逐步精炼其分类决策。与传统方法不同，它不依赖于预先收集的数据集，而是随着每次任务执行动态演变。通过实际实验，我们展示了该系统如何在处理更多样本时逐步减少执行时间，并保持接近完美的精度。这种适应性确保了力基机器人任务的长期可靠性，同时最大限度地减少了手动干预的需要。 

---
# ScrewSplat: An End-to-End Method for Articulated Object Recognition 

**Title (ZH)**: 螺栓爆裂：一种端到端的articulated对象识别方法 

**Authors**: Seungyeon Kim, Junsu Ha, Young Hun Kim, Yonghyeon Lee, Frank C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.02146)  

**Abstract**: Articulated object recognition -- the task of identifying both the geometry and kinematic joints of objects with movable parts -- is essential for enabling robots to interact with everyday objects such as doors and laptops. However, existing approaches often rely on strong assumptions, such as a known number of articulated parts; require additional inputs, such as depth images; or involve complex intermediate steps that can introduce potential errors -- limiting their practicality in real-world settings. In this paper, we introduce ScrewSplat, a simple end-to-end method that operates solely on RGB observations. Our approach begins by randomly initializing screw axes, which are then iteratively optimized to recover the object's underlying kinematic structure. By integrating with Gaussian Splatting, we simultaneously reconstruct the 3D geometry and segment the object into rigid, movable parts. We demonstrate that our method achieves state-of-the-art recognition accuracy across a diverse set of articulated objects, and further enables zero-shot, text-guided manipulation using the recovered kinematic model. 

**Abstract (ZH)**: 带关节物体识别——旨在识别具有可动部件的物体的几何形状和运动关节——对于使机器人能够与日常物品如门窗和笔记本电脑交互至关重要。然而，现有方法通常依赖强有力的假设，如已知的可动部件数量；需要额外输入，如深度图像；或者涉及复杂的中间步骤，可能导致潜在错误——这限制了它们在实际环境中的实用性。在本文中，我们引入了ScrewSplat，这是一种仅基于RGB观察的简单端到端方法。我们的方法首先随机初始化螺纹轴，然后通过迭代优化来恢复物体的基本运动结构。通过与高斯点划法结合，我们同时重建了3D几何形状，并将物体分割成刚性可动部分。我们证明了该方法在多种带关节物体上的识别准确性达到了最先进的水平，并通过恢复的动力学模型进一步实现了零样本、基于文本的操纵能力。 

---
# "Set It Up": Functional Object Arrangement with Compositional Generative Models 

**Title (ZH)**: “设置好它”：基于组合生成模型的功能性物体布局 

**Authors**: Yiqing Xu, Jiayuan Mao, Linfeng Li, Yilun Du, Tomas Lozáno-Pérez, Leslie Pack Kaelbling, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02068)  

**Abstract**: Functional object arrangement (FORM) is the task of arranging objects to fulfill a function, e.g., "set up a dining table for two". One key challenge here is that the instructions for FORM are often under-specified and do not explicitly specify the desired object goal poses. This paper presents SetItUp, a neuro-symbolic framework that learns to specify the goal poses of objects from a few training examples and a structured natural-language task specification. SetItUp uses a grounding graph, which is composed of abstract spatial relations among objects (e.g., left-of), as its intermediate representation. This decomposes the FORM problem into two stages: (i) predicting this graph among objects and (ii) predicting object poses given the grounding graph. For (i), SetItUp leverages large language models (LLMs) to induce Python programs from a task specification and a few training examples. This program can be executed to generate grounding graphs in novel scenarios. For (ii), SetItUp pre-trains a collection of diffusion models to capture primitive spatial relations and online composes these models to predict object poses based on the grounding graph. We evaluated SetItUp on a dataset spanning three distinct task families: arranging tableware on a dining table, organizing items on a bookshelf, and laying out furniture in a bedroom. Experiments show that SetItUp outperforms existing models in generating functional, physically feasible, and aesthetically pleasing object arrangements. This article extends our conference paper published at Robotics: Science and Systems (RSS) 2024. 

**Abstract (ZH)**: 功能对象布局（FORM）：一种基于神经符号框架的学习方法 

---
# RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models 

**Title (ZH)**: RICL：向预训练的多模态模型添加基于上下文的适应性 

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, Insup Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02062)  

**Abstract**: Multi-task ``vision-language-action'' (VLA) models have recently demonstrated increasing promise as generalist foundation models for robotics, achieving non-trivial performance out of the box on new tasks in new environments. However, for such models to be truly useful, an end user must have easy means to teach them to improve. For language and vision models, the emergent ability to perform in-context learning (ICL) has proven to be a versatile and highly useful interface to easily teach new tasks with no parameter finetuning. Unfortunately, VLAs pre-trained with imitation learning objectives do not naturally acquire ICL abilities. In this paper, we demonstrate that, with the right finetuning recipe and a small robot demonstration dataset, it is possible to inject in-context adaptability post hoc into such a VLA. After retraining for in-context learning (RICL), our system permits an end user to provide a small number (10-20) of demonstrations for a new task. RICL then fetches the most relevant portions of those demonstrations into the VLA context to exploit ICL, performing the new task and boosting task performance. We apply RICL to inject ICL into the $\pi_{0}$-FAST VLA, and show that it permits large in-context improvements for a variety of new manipulation tasks with only 20 demonstrations per task, without any parameter updates. When parameter updates on the target task demonstrations is possible, RICL finetuning further boosts performance. We release code and model weights for RICL-$\pi_{0}$-FAST alongside the paper to enable, for the first time, a simple in-context learning interface for new manipulation tasks. Website: this https URL. 

**Abstract (ZH)**: 多任务“视觉-语言-行动”（VLA）模型最近在作为机器人领域的通才基础模型方面展现了越来越大的潜力，能够在新的环境中以非平凡的方式完成新的任务。然而，要使这些模型真正有用，最终用户必须有简便的方法来教导它们从而改进。对于语言和视觉模型而言，出现的即席学习（ICL）能力已经成为了一个功能强大且实用的界面，能够轻松教授新任务而不需调整参数。不幸的是，使用模仿学习目标预训练的VLA们并不会自然地获得ICL能力。在本文中，我们展示了通过正确的调整训练方法和一个小机器人示范数据集，可以在这样的VLA中后置注入即席适应性。经过针对即席学习（RICL）的重新训练后，我们的系统允许最终用户仅需提供少量（10-20个）新任务示范。RICL随后将这些示范中最相关的部分注入到VLA的上下文中，利用ICL执行新任务并提升任务性能。我们将RICL注入到$\pi_{0}$-FAST VLA中，并展示了它在仅提供每任务20个示范的情况下，能够对多种新的操作任务实现大幅的即席改进，无需任何参数更新。当在目标任务示范上进行参数更新时，RICL进一步提升了性能。我们在论文中一同发布了RICL-$\pi_{0}$-FAST的代码和模型权重，以首次实现出即席学习界面来教导新的操作任务。网址：this https URL。 

---
# NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks 

**Title (ZH)**: NaviMaster: 学习统一策略开展GUI和具身导航任务 

**Authors**: Zhihao Luo, Wentao Yan abd Jingyu Gong, Min Wang, Zhizhong Zhang, Xuhong Wang, Yuan Xie, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02046)  

**Abstract**: Recent advances in Graphical User Interface (GUI) and embodied navigation have driven significant progress, yet these domains have largely evolved in isolation, with disparate datasets and training paradigms. In this paper, we observe that both tasks can be formulated as Markov Decision Processes (MDP), suggesting a foundational principle for their unification. Hence, we present NaviMaster, the first unified agent capable of seamlessly integrating GUI navigation and embodied navigation within a single framework. Specifically, NaviMaster (i) proposes a visual-target trajectory collection pipeline that generates trajectories for both GUI and embodied tasks in one formulation. (ii) employs a unified reinforcement learning framework on the mix data for better generalization. (iii) designs a novel distance-aware reward to ensure efficient learning from the trajectories. Through extensive experiments on out-of-domain benchmarks, NaviMaster is shown to outperform state-of-the-art agents in GUI navigation, spatial affordance prediction, and embodied navigation. Ablation studies further confirm the efficacy of our unified training strategy, data mixing strategy, and reward design. 

**Abstract (ZH)**: 近期图形用户界面（GUI）和体感导航的发展取得了显著进步，然而这两个领域主要独立演化，拥有各异的数据集和训练范式。本文观察到，这两个任务都可以形式化为马尔可夫决策过程（MDP），这暗示了一个统一基础原则。因此，我们提出了NaviMaster，这是首个能够无缝集成GUI导航和体感导航于单一框架中的统一代理。具体而言，NaviMaster：(i) 提出了一个视觉目标轨迹收集管道，用于在单一表述中生成GUI任务和体感任务的轨迹。(ii) 在混合数据上采用统一的强化学习框架以获得更好的泛化能力。(iii) 设计了一个新的距离感知奖励，以确保从轨迹中高效学习。通过在跨域基准上的广泛实验，NaviMaster 在GUI导航、空间行动预测和体感导航方面均优于现有最先进的代理。消融研究进一步证实了我们统一训练策略、数据混合策略和奖励设计的有效性。 

---
# Design and Control of an Actively Morphing Quadrotor with Vertically Foldable Arms 

**Title (ZH)**: 可主动变形且臂可竖折的四旋翼无人机的设计与控制 

**Authors**: Tingyu Yeh, Mengxin Xu, Lijun Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.02022)  

**Abstract**: In this work, we propose a novel quadrotor design capable of folding its arms vertically to grasp objects and navigate through narrow spaces. The transformation is controlled actively by a central servomotor, gears, and racks. The arms connect the motor bases to the central frame, forming a parallelogram structure that ensures the propellers maintain a constant orientation during morphing. In its stretched state, the quadrotor resembles a conventional design, and when contracted, it functions as a gripper with grasping components emerging from the motor bases. To mitigate disturbances during transforming and grasping payloads, we employ an adaptive sliding mode controller with a disturbance observer. After fully folded, the quadrotor frame shrinks to 67% of its original size. The control performance and versatility of the morphing quadrotor are validated through real-world experiments. 

**Abstract (ZH)**: 本研究提出了一种新颖的四旋翼设计，能够在垂直折叠其臂部以抓取物体并穿梭于狭窄空间。转换由中央伺服电机、齿轮和齿条主动控制。臂部将电机底座连接到中央框架，形成一个确保推进器在变形过程中保持恒定方向的平行四边形结构。在拉伸状态下，该四旋翼机类似于常规设计；当收缩时，它变为具有从电机底座伸出的抓取部件的夹持器。为减轻转换和抓取载荷时的干扰，我们采用带有干扰观测器的自适应滑模控制器。完全折叠后，四旋翼机框架缩小至原尺寸的67%。通过实际实验验证了变形四旋翼机的控制性能和多功能性。 

---
# From Photons to Physics: Autonomous Indoor Drones and the Future of Objective Property Assessment 

**Title (ZH)**: 从光子到物理：自主室内无人机与客观属性评估的未来 

**Authors**: Petteri Teikari, Mike Jarrell, Irene Bandera Moreno, Harri Pesola  

**Link**: [PDF](https://arxiv.org/pdf/2508.01965)  

**Abstract**: The convergence of autonomous indoor drones with physics-aware sensing technologies promises to transform property assessment from subjective visual inspection to objective, quantitative measurement. This comprehensive review examines the technical foundations enabling this paradigm shift across four critical domains: (1) platform architectures optimized for indoor navigation, where weight constraints drive innovations in heterogeneous computing, collision-tolerant design, and hierarchical control systems; (2) advanced sensing modalities that extend perception beyond human vision, including hyperspectral imaging for material identification, polarimetric sensing for surface characterization, and computational imaging with metaphotonics enabling radical miniaturization; (3) intelligent autonomy through active reconstruction algorithms, where drones equipped with 3D Gaussian Splatting make strategic decisions about viewpoint selection to maximize information gain within battery constraints; and (4) integration pathways with existing property workflows, including Building Information Modeling (BIM) systems and industry standards like Uniform Appraisal Dataset (UAD) 3.6. 

**Abstract (ZH)**: 自主室内无人机结合物理感知技术的收敛有望将房产评估从主观的视觉检查转变为客观的定量测量。本文综述了促成这一范式转变的技术基础，涵盖了四个关键领域：(1) 优化室内导航的平台架构，其中重量限制推动了异构计算、防碰撞设计和分层控制系统的发展；(2) 超越人类视觉的先进感知模式，包括用于材料识别的超光谱成像、用于表面表征的偏振光 sensing 以及通过元光子学实现的计算成像以实现根本性的微型化；(3) 通过主动重建算法实现的智能自主性，其中配备三维正态斑图化算法的无人机在电池限制条件下做出关于视点选择的策略性决策以最大化信息增益；以及 (4) 与现有房产工作流程的集成途径，包括建筑信息建模（BIM）系统和行业标准如统一评估数据集（UAD）3.6。 

---
# Beyond Simulation: Benchmarking World Models for Planning and Causality in Autonomous Driving 

**Title (ZH)**: 超越模拟：自主驾驶中世界模型的规划与因果性基准测试 

**Authors**: Hunter Schofield, Mohammed Elmahgiubi, Kasra Rezaee, Jinjun Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01922)  

**Abstract**: World models have become increasingly popular in acting as learned traffic simulators. Recent work has explored replacing traditional traffic simulators with world models for policy training. In this work, we explore the robustness of existing metrics to evaluate world models as traffic simulators to see if the same metrics are suitable for evaluating a world model as a pseudo-environment for policy training. Specifically, we analyze the metametric employed by the Waymo Open Sim-Agents Challenge (WOSAC) and compare world model predictions on standard scenarios where the agents are fully or partially controlled by the world model (partial replay). Furthermore, since we are interested in evaluating the ego action-conditioned world model, we extend the standard WOSAC evaluation domain to include agents that are causal to the ego vehicle. Our evaluations reveal a significant number of scenarios where top-ranking models perform well under no perturbation but fail when the ego agent is forced to replay the original trajectory. To address these cases, we propose new metrics to highlight the sensitivity of world models to uncontrollable objects and evaluate the performance of world models as pseudo-environments for policy training and analyze some state-of-the-art world models under these new metrics. 

**Abstract (ZH)**: 世界模型已成为作为学习交通模拟器越来越受欢迎的选择。近期工作探索用世界模型替代传统交通模拟器进行策略训练。在本研究中，我们考察现有用于评估世界模型作为交通模拟器的指标的鲁棒性，以确定这些指标是否适合评估世界模型作为策略训练伪环境的工具。具体而言，我们分析了Waymo Open Sim-Agents挑战赛（WOSAC）中使用的元指标，并将世界模型在标准场景中的预测进行对比，其中代理完全或部分由世界模型控制（部分回放）。由于我们关注ego动作条件的世界模型，我们将标准的WOSAC评估域扩展到包括对ego车辆有因果影响的代理。评估结果显示，在无干扰情况下排名靠前的模型在ego代理被迫重放原始轨迹时表现不佳。为此，我们提出新的指标来突出世界模型对不可控对象的敏感性，并评估世界模型作为策略训练伪环境的性能，在这些新指标下分析了一些最先进的世界模型。 

---
# L3M+P: Lifelong Planning with Large Language Models 

**Title (ZH)**: L3M+P: 基于大型语言模型的终身规划 

**Authors**: Krish Agarwal, Yuqian Jiang, Jiaheng Hu, Bo Liu, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2508.01917)  

**Abstract**: By combining classical planning methods with large language models (LLMs), recent research such as LLM+P has enabled agents to plan for general tasks given in natural language. However, scaling these methods to general-purpose service robots remains challenging: (1) classical planning algorithms generally require a detailed and consistent specification of the environment, which is not always readily available; and (2) existing frameworks mainly focus on isolated planning tasks, whereas robots are often meant to serve in long-term continuous deployments, and therefore must maintain a dynamic memory of the environment which can be updated with multi-modal inputs and extracted as planning knowledge for future tasks. To address these two issues, this paper introduces L3M+P (Lifelong LLM+P), a framework that uses an external knowledge graph as a representation of the world state. The graph can be updated from multiple sources of information, including sensory input and natural language interactions with humans. L3M+P enforces rules for the expected format of the absolute world state graph to maintain consistency between graph updates. At planning time, given a natural language description of a task, L3M+P retrieves context from the knowledge graph and generates a problem definition for classical planners. Evaluated on household robot simulators and on a real-world service robot, L3M+P achieves significant improvement over baseline methods both on accurately registering natural language state changes and on correctly generating plans, thanks to the knowledge graph retrieval and verification. 

**Abstract (ZH)**: 通过将经典规划方法与大规模语言模型结合，最近的研究如LLM+P使得代理能够规划自然语言描述的一般任务。然而，将这些方法扩展到通用服务机器人仍然具有挑战性：（1）经典规划算法通常需要详细的且一致的环境规范，而这些规范并不总是容易获得；（2）现有框架主要集中在孤立的规划任务上，而机器人通常旨在长期连续部署，因此必须维护一个可以随着多模态输入更新并提取为未来任务规划知识的动态环境记忆。为了解决这两个问题，本文引入了L3M+P（终身LLM+P）框架，该框架使用外部知识图作为世界状态的表示。该图可以从包括感官输入和与人类的自然语言交互等多种信息源进行更新。L3M+P规定了绝对世界状态图的预期格式以保持图更新间的连贯性。在规划时，给定一个自然语言描述的任务，L3M+P从知识图检索上下文并为经典规划者生成问题定义。在家庭机器人模拟器和真实世界的服务机器人上对L3M+P进行评估，通过知识图检索和验证，该方法在准确记录自然语言状态变化和正确生成规划方面显著优于基线方法。 

---
# Exploring environment exploitation for self-reconfiguration in modular robotics 

**Title (ZH)**: 探索环境利用以实现模块化机器人的自主重构 

**Authors**: Philippe Martin Wyder, Haorui Li, Andrew Bae, Henry Zhao, Mark Yim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01829)  

**Abstract**: Modular robotics research has long been preoccupied with perfecting the modules themselves -- their actuation methods, connectors, controls, communication, and fabrication. This inward focus results, in part, from the complexity of the task and largely confines modular robots to sterile laboratory settings. The latest generation of truss modular robots, such as the Variable Topology Truss and the Truss Link, have begun to focus outward and reveal a key insight: the environment is not just a backdrop; it is a tool. In this work, we shift the paradigm from building better robots to building better robot environment interactions for modular truss robots. We study how modular robots can effectively exploit their surroundings to achieve faster locomotion, adaptive self-reconfiguration, and complex three-dimensional assembly from simple two-dimensional robot assemblies. By using environment features -- ledges, gaps, and slopes -- we show how the environment can extend the robots' capabilities. Nature has long mastered this principle: organisms not only adapt, but exploit their environments to their advantage. Robots must learn to do the same. This study is a step towards modular robotic systems that transcend their limitations by exploiting environmental features. 

**Abstract (ZH)**: 模块化机器人研究长期以来一直专注于完善模块本身——其执行机构、连接方式、控制、通信和制造。这种内向的焦点部分上源于任务的复杂性，很大程度上限制了模块化机器人的应用场景。最新一代桁架模块化机器人，如可变拓扑桁架和桁架链接，已经开始关注外部环境，揭示了一个关键见解：环境不仅仅是一个背景，而是一种工具。在此项工作中，我们将范式从构建更好的机器人转移到构建更好的模块化桁架机器人与环境的交互。我们研究模块化机器人如何有效地利用其周围环境以实现更快的移动、自适应重构和从二维机器人组装实现复杂三维组装。通过利用环境特征（如凸起、缝隙和斜坡），我们展示了环境如何扩展机器人的能力。自然界早已掌握了这一原则：生物不仅能适应环境，还能利用环境。机器人必须学会这样做。这项研究朝着能够通过利用环境特征超越自身限制的模块化机器人系统迈出了一步。 

---
# Unraveling the Connection: How Cognitive Workload Shapes Intent Recognition in Robot-Assisted Surgery 

**Title (ZH)**: 探索认知负荷与机器人辅助手术中意图识别之间的关联 

**Authors**: Mansi Sharma, Antonio Kruger  

**Link**: [PDF](https://arxiv.org/pdf/2508.01823)  

**Abstract**: Robot-assisted surgery has revolutionized the healthcare industry by providing surgeons with greater precision, reducing invasiveness, and improving patient outcomes. However, the success of these surgeries depends heavily on the robotic system ability to accurately interpret the intentions of the surgical trainee or even surgeons. One critical factor impacting intent recognition is the cognitive workload experienced during the procedure. In our recent research project, we are building an intelligent adaptive system to monitor cognitive workload and improve learning outcomes in robot-assisted surgery. The project will focus on achieving a semantic understanding of surgeon intents and monitoring their mental state through an intelligent multi-modal assistive framework. This system will utilize brain activity, heart rate, muscle activity, and eye tracking to enhance intent recognition, even in mentally demanding situations. By improving the robotic system ability to interpret the surgeons intentions, we can further enhance the benefits of robot-assisted surgery and improve surgery outcomes. 

**Abstract (ZH)**: 机器人辅助手术通过提供更高的精确度、降低侵入性并改善患者预后， đã彻底改变了医疗行业。然而，这些手术的成功高度依赖于机器人系统准确解读外科医生学员乃至外科医生的意图的能力。影响意图识别的一个关键因素是在手术过程中经历的认知负荷。在我们最近的研究项目中，我们正在构建一个智能自适应系统，以监测认知负荷并改善机器人辅助手术中的学习效果。该项目将专注于实现对手术医生意图的语义理解，并通过智能多模态辅助框架监控其心理状态。该系统将利用脑活动、心率、肌肉活动和眼动追踪来增强意图识别，即使在认知负荷较高的情况下也不例外。通过提高机器人系统对医生意图的解读能力，我们可以进一步增强机器人辅助手术的优点并改善手术结果。 

---
# Exploring Stiffness Gradient Effects in Magnetically Induced Metamorphic Materials via Continuum Simulation and Validation 

**Title (ZH)**: 通过连续模拟与验证探索磁场诱导 metamorphic 材料中的刚度梯度效应 

**Authors**: Wentao Shi, Yang Yang, Yiming Huang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01810)  

**Abstract**: Magnetic soft continuum robots are capable of bending with remote control in confined space environments, and they have been applied in various bioengineering contexts. As one type of ferromagnetic soft continuums, the Magnetically Induced Metamorphic Materials (MIMMs)-based continuum (MC) exhibits similar bending behaviors. Based on the characteristics of its base material, MC is flexible in modifying unit stiffness and convenient in molding fabrication. However, recent studies on magnetic continuum robots have primarily focused on one or two design parameters, limiting the development of a comprehensive magnetic continuum bending model. In this work, we constructed graded-stiffness MCs (GMCs) and developed a numerical model for GMCs' bending performance, incorporating four key parameters that determine their performance. The simulated bending results were validated with real bending experiments in four different categories: varying magnetic field, cross-section, unit stiffness, and unit length. The graded-stiffness design strategy applied to GMCs prevents sharp bending at the fixed end and results in a more circular curvature. We also trained an expansion model for GMCs' bending performance that is highly efficient and accurate compared to the simulation process. An extensive library of bending prediction for GMCs was built using the trained model. 

**Abstract (ZH)**: 磁诱导 metamorphic 材料为基础的渐变刚度连续体机器人能够在受限空间中远程控制弯曲，并已应用于多种生物工程领域。在利用基材特性可调节单一刚度和方便成型加工的基础上，磁连续体机器人的近期研究主要集中在一两个设计参数上，限制了综合磁连续体弯曲模型的发展。在这项工作中，我们构建了渐变刚度连续体（GMC），并发展了适用于 GMC 的弯曲性能数值模型，综合了影响性能的四个关键参数。通过四种不同类别（磁场强度、截面形状、单一刚度和单一长度）的模拟弯曲实验和实际弯曲实验验证了模拟结果。渐变刚度设计策略应用于 GMC 避免了固定端的尖锐弯曲，产生了更圆的曲率。此外，我们还训练了一个高效的高精度扩展模型，用于预测 GMC 的弯曲性能，并构建了基于该模型的大量弯曲预测库。 

---
# Learning to Perform Low-Contact Autonomous Nasotracheal Intubation by Recurrent Action-Confidence Chunking with Transformer 

**Title (ZH)**: 学习通过递归动作-信心片段方法运用变压器进行低接触自主鼻.RESET插管 

**Authors**: Yu Tian, Ruoyi Hao, Yiming Huang, Dihong Xie, Catherine Po Ling Chan, Jason Ying Kuen Chan, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01808)  

**Abstract**: Nasotracheal intubation (NTI) is critical for establishing artificial airways in clinical anesthesia and critical care. Current manual methods face significant challenges, including cross-infection, especially during respiratory infection care, and insufficient control of endoluminal contact forces, increasing the risk of mucosal injuries. While existing studies have focused on automated endoscopic insertion, the automation of NTI remains unexplored despite its unique challenges: Nasotracheal tubes exhibit greater diameter and rigidity than standard endoscopes, substantially increasing insertion complexity and patient risks. We propose a novel autonomous NTI system with two key components to address these challenges. First, an autonomous NTI system is developed, incorporating a prosthesis embedded with force sensors, allowing for safety assessment and data filtering. Then, the Recurrent Action-Confidence Chunking with Transformer (RACCT) model is developed to handle complex tube-tissue interactions and partial visual observations. Experimental results demonstrate that the RACCT model outperforms the ACT model in all aspects and achieves a 66% reduction in average peak insertion force compared to manual operations while maintaining equivalent success rates. This validates the system's potential for reducing infection risks and improving procedural safety. 

**Abstract (ZH)**: 鼻咽气管插管（NTI）是临床麻醉和重症监护中建立人工气道的关键技术。现有的手动方法面临显著挑战，包括交叉感染，尤其是在呼吸道感染护理中，以及内腔接触力控制不足，增加黏膜损伤的风险。尽管已有研究集中在自动化内镜插入上，但鼻咽气管插管的自动化仍未得到探索，其独特挑战包括鼻咽气管导管直径和刚度大于标准内窥镜，显著增加插入复杂性和患者风险。我们提出了一种新的自主鼻咽气管插管系统，其中包括两个关键组件以应对这些挑战。首先，开发了一种自主鼻咽气管插管系统，该系统包含一个内置力传感器的植入物，以实现安全性评估和数据过滤。然后，开发了基于变换器的循环动作-置信度片段模型（RACCT），以处理复杂管-组织相互作用和不完整视觉观察。实验结果表明，RACCT模型在各方面都优于ACT模型，与手动操作相比，平均峰值插入力降低了66%，同时保持了同等的成功率。这验证了该系统在降低感染风险和提高操作安全方面的潜在价值。 

---
# Set the Stage: Enabling Storytelling with Multiple Robots through Roleplaying Metaphors 

**Title (ZH)**: 铺垫场景：通过角色扮演隐喻实现多机器人讲故事功能 

**Authors**: Tyrone Justin Sta Maria, Faith Griffin, Jordan Aiko Deja  

**Link**: [PDF](https://arxiv.org/pdf/2508.01736)  

**Abstract**: Gestures are an expressive input modality for controlling multiple robots, but their use is often limited by rigid mappings and recognition constraints. To move beyond these limitations, we propose roleplaying metaphors as a scaffold for designing richer interactions. By introducing three roles: Director, Puppeteer, and Wizard, we demonstrate how narrative framing can guide the creation of diverse gesture sets and interaction styles. These roles enable a variety of scenarios, showing how roleplay can unlock new possibilities for multi-robot systems. Our approach emphasizes creativity, expressiveness, and intuitiveness as key elements for future human-robot interaction design. 

**Abstract (ZH)**: 手势是一种控制多台机器人的情感输入方式，但其使用往往受限于僵硬的映射关系和识别约束。为进一步突破这些限制，我们提出了角色扮演元喻作为设计更丰富互动的支架。通过引入导演、提线木偶师和魔术师三种角色，我们展示了如何通过叙事框架指导多样手势集和交互风格的创设。这些角色使得多种场景成为可能，展示了角色扮演如何为多机器人系统解锁新可能性。我们的方法强调创造力、表达性和直观性作为未来人机交互设计的关键要素。 

---
# OpenMap: Instruction Grounding via Open-Vocabulary Visual-Language Mapping 

**Title (ZH)**: OpenMap: 基于开放词汇视觉语言映射的指令 grounding 

**Authors**: Danyang Li, Zenghui Yang, Guangpeng Qi, Songtao Pang, Guangyong Shang, Qiang Ma, Zheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01723)  

**Abstract**: Grounding natural language instructions to visual observations is fundamental for embodied agents operating in open-world environments. Recent advances in visual-language mapping have enabled generalizable semantic representations by leveraging vision-language models (VLMs). However, these methods often fall short in aligning free-form language commands with specific scene instances, due to limitations in both instance-level semantic consistency and instruction interpretation. We present OpenMap, a zero-shot open-vocabulary visual-language map designed for accurate instruction grounding in navigation tasks. To address semantic inconsistencies across views, we introduce a Structural-Semantic Consensus constraint that jointly considers global geometric structure and vision-language similarity to guide robust 3D instance-level aggregation. To improve instruction interpretation, we propose an LLM-assisted Instruction-to-Instance Grounding module that enables fine-grained instance selection by incorporating spatial context and expressive target descriptions. We evaluate OpenMap on ScanNet200 and Matterport3D, covering both semantic mapping and instruction-to-target retrieval tasks. Experimental results show that OpenMap outperforms state-of-the-art baselines in zero-shot settings, demonstrating the effectiveness of our method in bridging free-form language and 3D perception for embodied navigation. 

**Abstract (ZH)**: 将自然语言指令与视觉观察对接对于在开放世界环境中操作的实体代理至关重要。近期在视觉-语言映射方面的进展通过利用视觉-语言模型（VLMs）实现了可泛化的语义表示。然而，这些方法往往难以将开放形式的语言指令与具体的场景实例对齐，这主要归因于实例级别的语义一致性和指令解释方面的局限性。我们提出OpenMap，一种零样本开放词汇视觉-语言地图，旨在导航任务中实现精确的指令对接。为了解决视角间的语义不一致，我们引入了一种结构语义共识约束，它同时考虑全局几何结构和视觉-语言相似性，以指导稳健的三维实例级聚合。为了提高指令解释，我们提出了一种基于大语言模型的指令到实例对接模块，该模块通过整合空间上下文和表达性目标描述实现精细的实例选择。我们在ScanNet200和Matterport3D上评估OpenMap，涵盖了语义映射和指令到目标检索任务。实验结果表明，在零样本设置下，OpenMap优于现有最先进的基线方法，展示了我们方法在将开放形式语言与三维感知对接以实现实体导航方面的有效性。 

---
# Towards Zero-Shot Terrain Traversability Estimation: Challenges and Opportunities 

**Title (ZH)**: 面向零样本地形可通过性估计：挑战与机遇 

**Authors**: Ida Germann, Mark O. Mints, Peer Neubert  

**Link**: [PDF](https://arxiv.org/pdf/2508.01715)  

**Abstract**: Terrain traversability estimation is crucial for autonomous robots, especially in unstructured environments where visual cues and reasoning play a key role. While vision-language models (VLMs) offer potential for zero-shot estimation, the problem remains inherently ill-posed. To explore this, we introduce a small dataset of human-annotated water traversability ratings, revealing that while estimations are subjective, human raters still show some consensus. Additionally, we propose a simple pipeline that integrates VLMs for zero-shot traversability estimation. Our experiments reveal mixed results, suggesting that current foundation models are not yet suitable for practical deployment but provide valuable insights for further research. 

**Abstract (ZH)**: 地形可通过性估计对于自主机器人至关重要，尤其是在视觉线索和推理起关键作用的未结构化环境中。尽管视觉语言模型（VLMs）为零样本估计提供了潜力，但该问题依然本质上存在不明确性。为探索这一问题，我们引入了一个小型的人工标注水域可通过性评价数据集，揭示出虽然估计具有主观性，但人类评价者仍表现出一定程度的一致性。此外，我们提出一个简单的管道，将VLMs集成用于零样本可通过性估计。我们的实验结果显示了混合的结果，表明当前的基础模型尚未准备好实际部署，但提供了进一步研究的宝贵见解。 

---
# DexReMoE:In-hand Reorientation of General Object via Mixtures of Experts 

**Title (ZH)**: DexReMoE: 通过专家混合实现的手握状态下通用物体的重新定向 

**Authors**: Jun Wan, Xing Liu, Yunlong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01695)  

**Abstract**: In hand object reorientation provides capability for dexterous manipulation, requiring robust control policies to manage diverse object geometries, maintain stable grasps, and execute precise complex orientation trajectories. However, prior works focus on single objects or simple geometries and struggle to generalize to complex shapes. In this work, we introduce DexReMoE (Dexterous Reorientation Mixture-of-Experts), in which multiple expert policies are trained for different complex shapes and integrated within a Mixture-of-Experts (MoE) framework, making the approach capable of generalizing across a wide range of objects. Additionally, we incorporate object category information as privileged inputs to enhance shape representation. Our framework is trained in simulation using reinforcement learning (RL) and evaluated on novel out-of-distribution objects in the most challenging scenario of reorienting objects held in the air by a downward-facing hand. In terms of the average consecutive success count, DexReMoE achieves a score of 19.5 across a diverse set of 150 objects. In comparison to the baselines, it also enhances the worst-case performance, increasing it from 0.69 to 6.05. These results underscore the scalability and adaptability of the DexReMoE framework for general-purpose in-hand reorientation. 

**Abstract (ZH)**: Dexterous Reorientation Mixture-of-Experts for Robust Hand Object Manipulation Across Complex Shapes 

---
# Energy-Predictive Planning for Optimizing Drone Service Delivery 

**Title (ZH)**: 基于能量预测的无人机服务交付优化规划 

**Authors**: Guanting Ren, Babar Shahzaad, Balsam Alkouz, Abdallah Lakhdari, Athman Bouguettaya  

**Link**: [PDF](https://arxiv.org/pdf/2508.01671)  

**Abstract**: We propose a novel Energy-Predictive Drone Service (EPDS) framework for efficient package delivery within a skyway network. The EPDS framework incorporates a formal modeling of an EPDS and an adaptive bidirectional Long Short-Term Memory (Bi-LSTM) machine learning model. This model predicts the energy status and stochastic arrival times of other drones operating in the same skyway network. Leveraging these predictions, we develop a heuristic optimization approach for composite drone services. This approach identifies the most time-efficient and energy-efficient skyway path and recharging schedule for each drone in the network. We conduct extensive experiments using a real-world drone flight dataset to evaluate the performance of the proposed framework. 

**Abstract (ZH)**: 一种新型能量预测无人机服务（EPDS）框架：在天道网络中实现高效的包裹交付 

---
# VFP: Variational Flow-Matching Policy for Multi-Modal Robot Manipulation 

**Title (ZH)**: VFP：变分流匹配策略用于多模态机器人操纵 

**Authors**: Xuanran Zhai, Ce Hao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01622)  

**Abstract**: Flow-matching-based policies have recently emerged as a promising approach for learning-based robot manipulation, offering significant acceleration in action sampling compared to diffusion-based policies. However, conventional flow-matching methods struggle with multi-modality, often collapsing to averaged or ambiguous behaviors in complex manipulation tasks. To address this, we propose the Variational Flow-Matching Policy (VFP), which introduces a variational latent prior for mode-aware action generation and effectively captures both task-level and trajectory-level multi-modality. VFP further incorporates Kantorovich Optimal Transport (K-OT) for distribution-level alignment and utilizes a Mixture-of-Experts (MoE) decoder for mode specialization and efficient inference. We comprehensively evaluate VFP on 41 tasks across four benchmark environments, demonstrating its effectiveness and sampling efficiency in both task and path multi-modality settings. Results show that VFP achieves a $49\%$ relative improvement in task success rate over standard flow-based baselines, while maintaining fast inference and compact model size. More details are available on our project page: this https URL 

**Abstract (ZH)**: 基于流匹配的政策最近成为学习驱动机器人操作的有前景的方法，相比基于扩散的政策，在动作采样上提供了显著加速。然而，传统的流匹配方法在处理多模态时存在困难，经常在复杂操作任务中退化为平均或模棱两可的行为。为了解决这一问题，我们提出了变分流匹配策略（VFP），引入了变分潜在先验以实现模式感知的动作生成，并有效地捕捉任务级和轨迹级的多模态。VFP 进一步结合了柯尔莫哥洛夫最优传输（K-OT）进行分布级对齐，并利用混合专家（MoE）解码器实现模式专业化和高效推理。我们在四个基准环境的41个任务上全面评估了VFP，展示了其在任务和路径多模态设置中的有效性和采样效率。结果表明，与标准基于流的基线相比，VFP 在任务成功率上实现了49%的相对提升，同时保持快速推理和紧凑的模型大小。更多细节请参见我们的项目页面：this https URL。 

---
# CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation 

**Title (ZH)**: CLASS: 基于动作序列监督的对比学习方法在机器人操作中的应用 

**Authors**: Sung-Wook Lee, Xuhui Kang, Brandon Yang, Yen-Ling Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01600)  

**Abstract**: Recent advances in Behavior Cloning (BC) have led to strong performance in robotic manipulation, driven by expressive models, sequence modeling of actions, and large-scale demonstration data. However, BC faces significant challenges when applied to heterogeneous datasets, such as visual shift with different camera poses or object appearances, where performance degrades despite the benefits of learning at scale. This stems from BC's tendency to overfit individual demonstrations rather than capture shared structure, limiting generalization. To address this, we introduce Contrastive Learning via Action Sequence Supervision (CLASS), a method for learning behavioral representations from demonstrations using supervised contrastive learning. CLASS leverages weak supervision from similar action sequences identified via Dynamic Time Warping (DTW) and optimizes a soft InfoNCE loss with similarity-weighted positive pairs. We evaluate CLASS on 5 simulation benchmarks and 3 real-world tasks to achieve competitive results using retrieval-based control with representations only. Most notably, for downstream policy learning under significant visual shifts, Diffusion Policy with CLASS pre-training achieves an average success rate of 75%, while all other baseline methods fail to perform competitively. Project webpage: this https URL. 

**Abstract (ZH)**: 近期行为克隆（BC）的进展在机器人操作方面的性能显著提升，得益于表现力更强的模型、行动序列建模以及大规模演示数据。然而，当将BC应用于异构数据集时，如不同摄像机姿态或物体外观导致的视觉偏移，其性能会下降，尽管大规模学习带来了好处。这源于BC倾向于过度拟合个体演示，而不是捕捉共享结构，从而限制了泛化能力。为解决这一问题，我们引入了通过行动序列监督的对比学习（CLASS）方法，这是一种从通过动态时间规整（DTW）识别的相似行动序列中学习行为表示的方法。CLASS利用动态时间规整（DTW）识别的相似行动序列的弱监督，并使用基于相似性加权的正样本优化软InfoNCE损失。我们通过5个模拟基准和3个真实世界任务评估CLASS，采用基于检索的控制方法仅使用表示实现了竞争力的结果。尤其值得注意的是，在显著视觉偏移下进行下游策略学习时，通过CLASS预训练的扩散策略的成功率为75%，而所有其他基线方法表现不佳。项目网页：这个httpsURL。 

---
# Adverse Weather-Independent Framework Towards Autonomous Driving Perception through Temporal Correlation and Unfolded Regularization 

**Title (ZH)**: 基于 temporal 相关性和展开正则化的独立恶劣天气自主驾驶感知框架 

**Authors**: Wei-Bin Kou, Guangxu Zhu, Rongguang Ye, Jingreng Lei, Shuai Wang, Qingfeng Lin, Ming Tang, Yik-Chung Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01583)  

**Abstract**: Various adverse weather conditions such as fog and rain pose a significant challenge to autonomous driving (AD) perception tasks like semantic segmentation, object detection, etc. The common domain adaption strategy is to minimize the disparity between images captured in clear and adverse weather conditions. However, domain adaption faces two challenges: (I) it typically relies on utilizing clear image as a reference, which is challenging to obtain in practice; (II) it generally targets single adverse weather condition and performs poorly when confronting the mixture of multiple adverse weather conditions. To address these issues, we introduce a reference-free and Adverse weather condition-independent (Advent) framework (rather than a specific model architecture) that can be implemented by various backbones and heads. This is achieved by leveraging the homogeneity over short durations, getting rid of clear reference and being generalizable to arbitrary weather condition. Specifically, Advent includes three integral components: (I) Locally Sequential Mechanism (LSM) leverages temporal correlations between adjacent frames to achieve the weather-condition-agnostic effect thanks to the homogeneity behind arbitrary weather condition; (II) Globally Shuffled Mechanism (GSM) is proposed to shuffle segments processed by LSM from different positions of input sequence to prevent the overfitting to LSM-induced temporal patterns; (III) Unfolded Regularizers (URs) are the deep unfolding implementation of two proposed regularizers to penalize the model complexity to enhance across-weather generalization. We take the semantic segmentation task as an example to assess the proposed Advent framework. Extensive experiments demonstrate that the proposed Advent outperforms existing state-of-the-art baselines with large margins. 

**Abstract (ZH)**: 无参考且独立于恶劣天气条件的(Advent)框架：针对语义分割任务的评估 

---
# HALO: Human Preference Aligned Offline Reward Learning for Robot Navigation 

**Title (ZH)**: HALO: 人类偏好对齐的离线奖励学习在机器人导航中的应用 

**Authors**: Gershom Seneviratne, Jianyu An, Sahire Ellahy, Kasun Weerakoon, Mohamed Bashir Elnoor, Jonathan Deepak Kannan, Amogha Thalihalla Sunil, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2508.01539)  

**Abstract**: In this paper, we introduce HALO, a novel Offline Reward Learning algorithm that quantifies human intuition in navigation into a vision-based reward function for robot navigation. HALO learns a reward model from offline data, leveraging expert trajectories collected from mobile robots. During training, actions are uniformly sampled around a reference action and ranked using preference scores derived from a Boltzmann distribution centered on the preferred action, and shaped based on binary user feedback to intuitive navigation queries. The reward model is trained via the Plackett-Luce loss to align with these ranked preferences. To demonstrate the effectiveness of HALO, we deploy its reward model in two downstream applications: (i) an offline learned policy trained directly on the HALO-derived rewards, and (ii) a model-predictive-control (MPC) based planner that incorporates the HALO reward as an additional cost term. This showcases the versatility of HALO across both learning-based and classical navigation frameworks. Our real-world deployments on a Clearpath Husky across diverse scenarios demonstrate that policies trained with HALO generalize effectively to unseen environments and hardware setups not present in the training data. HALO outperforms state-of-the-art vision-based navigation methods, achieving at least a 33.3% improvement in success rate, a 12.9% reduction in normalized trajectory length, and a 26.6% reduction in Frechet distance compared to human expert trajectories. 

**Abstract (ZH)**: 本文介绍了HALO，一种新颖的离线奖励学习算法，将人类在导航中的直觉量化为基于视觉的奖励函数，用于机器人导航。HALO通过利用从移动机器人收集的专家轨迹，从离线数据中学习奖励模型。在训练过程中，动作在参考动作周围均匀采样，并使用基于首选动作中心的玻尔兹曼分布衍生的偏好得分进行排序，同时根据二元用户反馈对直觉导航查询进行塑造。通过Plackett-Luce损失训练奖励模型，使其与这些排序偏好保持一致。为了展示HALO的有效性，我们将其奖励模型部署在两个下游应用中：(i) 一个直接在HALO衍生奖励上训练的离线学习策略，以及(ii) 一个基于模型预测控制(MPC)的规划器，将HALO奖励作为额外的成本项纳入其中。这展示了HALO在学习驱动和经典导航框架中的 versatility。我们的实际部署在Clearpath Husky上，涵盖了多种场景，证明了使用HALO训练的策略能够有效地泛化到未出现在训练数据中的未知环境和硬件配置。与基于视觉的人类专家轨迹相比，HALO在成功率上至少提高了33.3%，平均轨迹长度降低了12.9%，弗雷彻距离降低了26.6%。 

---
# Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于多智能体强化学习的分布式缆索悬挂负载操作 

**Authors**: Jack Zeng, Andreu Matoses Gimenez, Eugene Vinitsky, Javier Alonso-Mora, Sihao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01522)  

**Abstract**: This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: this https URL 

**Abstract (ZH)**: 本文提出了一种分散式方法，利用一组微型空中车辆（MAVs）实现悬索负载的六自由度现实世界操作。该方法利用多代理强化学习（MARL）为每个MAV训练外部循环控制策略。与现有利用集中式方案的控制器不同，我们的策略不需要全局状态、MAV间通信或邻近MAV的信息。相反，代理仅通过负载姿态观察进行隐式通信，这使得高可扩展性和灵活性成为可能。这种方法还显著减少了推理时的计算成本，使得策略能够在机载部署。此外，我们还为MAVs引入了一种新的动作空间设计，使用线性加速度和机体角速率。这种选择结合了一个鲁棒的低级控制器，即使在动态三维运动中因缆绳张力引起的显著不确定性下，也能实现可靠的仿真到现实的过渡。我们在各种实际实验中验证了该方法，包括在负载模型不确定性下的全姿态控制，展示了与现有最佳集中式方法相当的定值跟踪性能。我们还展示了具有不同控制策略的代理之间的合作，并证明了在一台MAV完全失联时的鲁棒性。实验视频：this https URL 

---
# Physically-based Lighting Augmentation for Robotic Manipulation 

**Title (ZH)**: 基于物理的照明增强方法在机器人操作中的应用 

**Authors**: Shutong Jin, Lezhong Wang, Ben Temming, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2508.01442)  

**Abstract**: Despite advances in data augmentation, policies trained via imitation learning still struggle to generalize across environmental variations such as lighting changes. To address this, we propose the first framework that leverages physically-based inverse rendering for lighting augmentation on real-world human demonstrations. Specifically, inverse rendering decomposes the first frame in each demonstration into geometric (surface normal, depth) and material (albedo, roughness, metallic) properties, which are then used to render appearance changes under different lighting. To ensure consistent augmentation across each demonstration, we fine-tune Stable Video Diffusion on robot execution videos for temporal lighting propagation. We evaluate our framework by measuring the structural and temporal consistency of the augmented sequences, and by assessing its effectiveness in reducing the behavior cloning generalization gap (40.1%) on a 7-DoF robot across 6 lighting conditions using 720 real-world evaluations. We further showcase three downstream applications enabled by the proposed framework. 

**Abstract (ZH)**: 尽管在数据增强方面取得了进展，通过imitation learning训练的策略仍然难以在光照变化等环境变化中泛化。为了解决这个问题，我们提出了第一个利用基于物理的逆渲染进行光照增强的框架，以增强现实人类示范。具体而言，逆渲染将每个示范的第一帧分解为几何（法线、深度）和材料（反射率、粗糙度、金属度）属性，并利用这些属性在不同光照条件下渲染外观变化。为了确保每个示范中增强的连贯性，我们在机器人执行视频上微调了Stable Video Diffusion以实现时间光照传播。我们通过测量增强序列的结构和时间一致性，并通过在6种光照条件下使用720个真实场景评估，在7-DoF机器人上减少行为克隆泛化差距（40.1%）来评估该框架的有效性。我们进一步展示了该框架使三个下游应用成为可能。 

---
# RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems 

**Title (ZH)**: RoboMemory: 一种类脑多记忆自主框架，用于物理实体系统的终身学习 

**Authors**: Mingcong Lei, Honghao Cai, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.01415)  

**Abstract**: We present RoboMemory, a brain-inspired multi-memory framework for lifelong learning in physical embodied systems, addressing critical challenges in real-world environments: continuous learning, multi-module memory latency, task correlation capture, and infinite-loop mitigation in closed-loop planning. Grounded in cognitive neuroscience, it integrates four core modules: the Information Preprocessor (thalamus-like), the Lifelong Embodied Memory System (hippocampus-like), the Closed-Loop Planning Module (prefrontal lobe-like), and the Low-Level Executer (cerebellum-like) to enable long-term planning and cumulative learning. The Lifelong Embodied Memory System, central to the framework, alleviates inference speed issues in complex memory frameworks via parallelized updates/retrieval across Spatial, Temporal, Episodic, and Semantic submodules. It incorporates a dynamic Knowledge Graph (KG) and consistent architectural design to enhance memory consistency and scalability. Evaluations on EmbodiedBench show RoboMemory outperforms the open-source baseline (Qwen2.5-VL-72B-Ins) by 25% in average success rate and surpasses the closed-source State-of-the-Art (SOTA) (Claude3.5-Sonnet) by 5%, establishing new SOTA. Ablation studies validate key components (critic, spatial memory, long-term memory), while real-world deployment confirms its lifelong learning capability with significantly improved success rates across repeated tasks. RoboMemory alleviates high latency challenges with scalability, serving as a foundational reference for integrating multi-modal memory systems in physical robots. 

**Abstract (ZH)**: RoboMemory：一种脑启发的多记忆框架，用于物理具身系统的终身学习 

---
# MoRe-ERL: Learning Motion Residuals using Episodic Reinforcement Learning 

**Title (ZH)**: MoRe-ERL：基于情景强化学习的学习运动残差 

**Authors**: Xi Huang, Hongyi Zhou, Ge Li, Yucheng Tang, Weiran Liao, Björn Hein, Tamim Asfour, Rudolf Lioutikov  

**Link**: [PDF](https://arxiv.org/pdf/2508.01409)  

**Abstract**: We propose MoRe-ERL, a framework that combines Episodic Reinforcement Learning (ERL) and residual learning, which refines preplanned reference trajectories into safe, feasible, and efficient task-specific trajectories. This framework is general enough to incorporate into arbitrary ERL methods and motion generators seamlessly. MoRe-ERL identifies trajectory segments requiring modification while preserving critical task-related maneuvers. Then it generates smooth residual adjustments using B-Spline-based movement primitives to ensure adaptability to dynamic task contexts and smoothness in trajectory refinement. Experimental results demonstrate that residual learning significantly outperforms training from scratch using ERL methods, achieving superior sample efficiency and task performance. Hardware evaluations further validate the framework, showing that policies trained in simulation can be directly deployed in real-world systems, exhibiting a minimal sim-to-real gap. 

**Abstract (ZH)**: MoRe-ERL: 结合 episodic强化学习和残差学习的轨迹优化框架 

---
# VLH: Vision-Language-Haptics Foundation Model 

**Title (ZH)**: VLH：视觉-语言-触觉基础模型 

**Authors**: Luis Francisco Moreno Fuentes, Muhammad Haris Khan, Miguel Altamirano Cabrera, Valerii Serpiva, Dmitri Iarchuk, Yara Mahmoud, Issatay Tokmurziyev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2508.01361)  

**Abstract**: We present VLH, a novel Visual-Language-Haptic Foundation Model that unifies perception, language, and tactile feedback in aerial robotics and virtual reality. Unlike prior work that treats haptics as a secondary, reactive channel, VLH synthesizes mid-air force and vibration cues as a direct consequence of contextual visual understanding and natural language commands. Our platform comprises an 8-inch quadcopter equipped with dual inverse five-bar linkage arrays for localized haptic actuation, an egocentric VR camera, and an exocentric top-down view. Visual inputs and language instructions are processed by a fine-tuned OpenVLA backbone - adapted via LoRA on a bespoke dataset of 450 multimodal scenarios - to output a 7-dimensional action vector (Vx, Vy, Vz, Hx, Hy, Hz, Hv). INT8 quantization and a high-performance server ensure real-time operation at 4-5 Hz. In human-robot interaction experiments (90 flights), VLH achieved a 56.7% success rate for target acquisition (mean reach time 21.3 s, pose error 0.24 m) and 100% accuracy in texture discrimination. Generalization tests yielded 70.0% (visual), 54.4% (motion), 40.0% (physical), and 35.0% (semantic) performance on novel tasks. These results demonstrate VLH's ability to co-evolve haptic feedback with perceptual reasoning and intent, advancing expressive, immersive human-robot interactions. 

**Abstract (ZH)**: VLH：一种统一视觉、语言和触觉反馈的新型视觉-语言-触觉基础模型 

---
# Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy 

**Title (ZH)**: 基于对称等变强化学习策略的协调人类形机器人运动控制 

**Authors**: Buqing Nie, Yang Zhang, Rongjun Jin, Zhanxiang Cao, Huangxuan Lin, Xiaokang Yang, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01247)  

**Abstract**: The human nervous system exhibits bilateral symmetry, enabling coordinated and balanced movements. However, existing Deep Reinforcement Learning (DRL) methods for humanoid robots neglect morphological symmetry of the robot, leading to uncoordinated and suboptimal behaviors. Inspired by human motor control, we propose Symmetry Equivariant Policy (SE-Policy), a new DRL framework that embeds strict symmetry equivariance in the actor and symmetry invariance in the critic without additional hyperparameters. SE-Policy enforces consistent behaviors across symmetric observations, producing temporally and spatially coordinated motions with higher task performance. Extensive experiments on velocity tracking tasks, conducted in both simulation and real-world deployment with the Unitree G1 humanoid robot, demonstrate that SE-Policy improves tracking accuracy by up to 40% compared to state-of-the-art baselines, while achieving superior spatial-temporal coordination. These results demonstrate the effectiveness of SE-Policy and its broad applicability to humanoid robots. 

**Abstract (ZH)**: 人体神经系统表现出双边对称性，使协调和平衡的运动成为可能。然而，现有的类人机器人深度强化学习（DRL）方法未考虑机器人的形态对称性，导致不协调和次优行为。受人类运动控制的启发，我们提出了一种新的DRL框架——对称不变策略（SE-Policy），该框架在演员中嵌入严格的对称不变性，在评论者中嵌入对称性，而无需额外的超参数。SE-Policy在对称观察下强制执行一致的行为，产生更协调的时空运动，具有更高的任务性能。在使用Unitree G1类人机器人进行的仿真和实际应用中的速度追踪任务的广泛实验中，SE-Policy相较于最先进的基线提高了多达40%的跟踪精度，同时实现了更优的时空协调。这些结果表明SE-Policy的有效性及其在类人机器人领域的广泛应用。 

---
# Unified Generation-Refinement Planning: Bridging Flow Matching and Sampling-Based MPC 

**Title (ZH)**: 统一生成-精炼规划：流匹配与基于采样的MPC的桥梁 

**Authors**: Kazuki Mizuta, Karen Leung  

**Link**: [PDF](https://arxiv.org/pdf/2508.01192)  

**Abstract**: Planning safe and effective robot behavior in dynamic, human-centric environments remains a core challenge due to the need to handle uncertainty, adapt in real-time, and ensure safety. Optimization-based planners offer explicit constraint handling but rely on oversimplified initialization, reducing solution quality. Learning-based planners better capture multimodal possible solutions but struggle to enforce constraints such as safety. In this paper, we introduce a unified generation-refinement framework bridging learning and optimization with a novel reward-guided conditional flow matching (CFM) model and model predictive path integral (MPPI) control. Our key innovation is in the incorporation of a bidirectional information exchange: samples from a reward-guided CFM model provide informed priors for MPPI refinement, while the optimal trajectory from MPPI warm-starts the next CFM generation. Using autonomous social navigation as a motivating application, we demonstrate that our approach can flexibly adapt to dynamic environments to satisfy safety requirements in real-time. 

**Abstract (ZH)**: 在动态的人本中心环境中规划安全有效的机器人行为仍是一项核心挑战，由于需要处理不确定性、实时适应并确保安全性。基于优化的规划器提供明确的约束处理机制，但依赖于过度简化的初始化，降低了解决方案的质量。基于学习的规划器能够更好地捕捉多模态可能的解决方案，但在执行约束，如安全性方面却举步维艰。在本文中，我们提出了一种统一的生成-精炼框架，结合了学习和优化，并引入了一种新型的基于奖励的条件流动匹配（CFM）模型和模型预测路径积分（MPPI）控制。我们创新之处在于实现了双向信息交换：基于奖励的CFM模型的样本为MPPI的精炼提供有信息的先验知识，而MPPI的最优轨迹则为下一个CFM生成提供初始条件。以自主社会导航为例，我们展示了该方法能够灵活适应动态环境，并在实时满足安全要求方面具有优势。 

---
# Design of Q8bot: A Miniature, Low-Cost, Dynamic Quadruped Built with Zero Wires 

**Title (ZH)**: Q8bot的设计：一款无线、低成本、动态四足机器人 

**Authors**: Yufeng Wu, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01149)  

**Abstract**: This paper introduces Q8bot, an open-source, miniature quadruped designed for robotics research and education. We present the robot's novel zero-wire design methodology, which leads to its superior form factor, robustness, replicability, and high performance. With a size and weight similar to a modern smartphone, this standalone robot can walk for over an hour on a single battery charge and survive meter-high drops with simple repairs. Its 300-dollar bill of materials includes minimal off-the-shelf components, readily available custom electronics from online vendors, and structural parts that can be manufactured on hobbyist 3D printers. A preliminary user assembly study confirms that Q8bot can be easily replicated, with an average assembly time of under one hour by a single person. With heuristic open-loop control, Q8bot achieves a stable walking speed of 5.4 body lengths per second and a turning speed of 5 radians per second, along with other dynamic movements such as jumping and climbing moderate slopes. 

**Abstract (ZH)**: 本论文介绍了Q8bot，一种面向机器人研究与教育的开源微型四足机器人。我们介绍了该机器人独特的无缆设计方法论，这使其具有优异的外形、 sturdy性和可复制性以及高性能。这款单体机器人大小和重量类似现代智能手机，单次电池充电后可行走超过一个小时，并且简单修复后可以从一米高的地方跌落存活。其300美元的物料清单包括少量标准组件、易于获取的定制电子元件以及可以使用爱好者级3D打印机制造的结构部件。初步用户组装研究证实，Q8bot 可以轻松复制，单人组装时间平均小于一小时。通过启发式的开环控制，Q8bot 达到了每秒5.4个身长的稳定行走速度和每秒5弧度的转向速度，以及其他动态动作，如跳跃和攀爬较陡斜坡。 

---
# COLLAGE: Adaptive Fusion-based Retrieval for Augmented Policy Learning 

**Title (ZH)**: COLLAGE: 适应性融合为基础的增强策略学习检索方法 

**Authors**: Sateesh Kumar, Shivin Dass, Georgios Pavlakos, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2508.01131)  

**Abstract**: In this work, we study the problem of data retrieval for few-shot imitation learning: selecting data from a large dataset to train a performant policy for a specific task, given only a few target demonstrations. Prior methods retrieve data using a single-feature distance heuristic, assuming that the best demonstrations are those that most closely resemble the target examples in visual, semantic, or motion space. However, this approach captures only a subset of the relevant information and can introduce detrimental demonstrations, e.g., retrieving data from unrelated tasks due to similar scene layouts, or selecting similar motions from tasks with divergent goals. We present COLLAGE, a method for COLLective data AGgrEgation in few-shot imitation learning that uses an adaptive late fusion mechanism to guide the selection of relevant demonstrations based on a task-specific combination of multiple cues. COLLAGE follows a simple, flexible, and efficient recipe: it assigns weights to subsets of the dataset that are pre-selected using a single feature (e.g., appearance, shape, or language similarity), based on how well a policy trained on each subset predicts actions in the target demonstrations. These weights are then used to perform importance sampling during policy training, sampling data more densely or sparsely according to estimated relevance. COLLAGE is general and feature-agnostic, allowing it to combine any number of subsets selected by any retrieval heuristic, and to identify which subsets provide the greatest benefit for the target task. In extensive experiments, COLLAGE outperforms state-of-the-art retrieval and multi-task learning approaches by 5.1% in simulation across 10 tasks, and by 16.6% in the real world across 6 tasks, where we perform retrieval from the large-scale DROID dataset. More information at this https URL . 

**Abstract (ZH)**: 在少样本模仿学习中的数据检索研究：基于任务特定多线索的适应性数据聚合方法 

---
# Human-Robot Red Teaming for Safety-Aware Reasoning 

**Title (ZH)**: 人类与机器人红队协同进行安全意识推理 

**Authors**: Emily Sheetz, Emma Zemler, Misha Savchenko, Connor Rainen, Erik Holum, Jodi Graf, Andrew Albright, Shaun Azimi, Benjamin Kuipers  

**Link**: [PDF](https://arxiv.org/pdf/2508.01129)  

**Abstract**: While much research explores improving robot capabilities, there is a deficit in researching how robots are expected to perform tasks safely, especially in high-risk problem domains. Robots must earn the trust of human operators in order to be effective collaborators in safety-critical tasks, specifically those where robots operate in human environments. We propose the human-robot red teaming paradigm for safety-aware reasoning. We expect humans and robots to work together to challenge assumptions about an environment and explore the space of hazards that may arise. This exploration will enable robots to perform safety-aware reasoning, specifically hazard identification, risk assessment, risk mitigation, and safety reporting. We demonstrate that: (a) human-robot red teaming allows human-robot teams to plan to perform tasks safely in a variety of domains, and (b) robots with different embodiments can learn to operate safely in two different environments -- a lunar habitat and a household -- with varying definitions of safety. Taken together, our work on human-robot red teaming for safety-aware reasoning demonstrates the feasibility of this approach for safely operating and promoting trust on human-robot teams in safety-critical problem domains. 

**Abstract (ZH)**: whilst much research focuses on enhancing robot capabilities, there is a deficiency in investigating how robots are expected to perform tasks safely, especially in high-risk domains. We propose the human-robot red teaming paradigm for safety-aware reasoning. We expect humans and robots to collaborate in challenging assumptions about an environment and exploring potential hazards. This process will enable robots to perform safety-aware reasoning, including hazard identification, risk assessment, risk mitigation, and safety reporting. We demonstrate that: (a) human-robot red teaming allows teams to plan for safe task execution in diverse domains, and (b) robots with different physical embodiments can learn to operate safely in two distinct environments—lunar habitats and households—with varying safety definitions. Collectively, our work on human-robot red teaming for safety-aware reasoning highlights the feasibility of this approach for safe operation and trust-building on human-robot teams in critical domains. 

---
# Improving Drone Racing Performance Through Iterative Learning MPC 

**Title (ZH)**: 通过迭代学习模型预测控制提高无人机竞速 performance 

**Authors**: Haocheng Zhao, Niklas Schlüter, Lukas Brunke, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2508.01103)  

**Abstract**: Autonomous drone racing presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control~(LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations:~(1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence,~(2)~a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and~(3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85\%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05\% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing. 

**Abstract (ZH)**: 自主无人机竞速 presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control (LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations: (1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence, (2) a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and (3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing. 

---
# Learning Pivoting Manipulation with Force and Vision Feedback Using Optimization-based Demonstrations 

**Title (ZH)**: 基于优化示范的学习基于力与视觉反馈的pivot操作 manipulaton 

**Authors**: Yuki Shirai, Kei Ota, Devesh K. Jha, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2508.01082)  

**Abstract**: Non-prehensile manipulation is challenging due to complex contact interactions between objects, the environment, and robots. Model-based approaches can efficiently generate complex trajectories of robots and objects under contact constraints. However, they tend to be sensitive to model inaccuracies and require access to privileged information (e.g., object mass, size, pose), making them less suitable for novel objects. In contrast, learning-based approaches are typically more robust to modeling errors but require large amounts of data. In this paper, we bridge these two approaches to propose a framework for learning closed-loop pivoting manipulation. By leveraging computationally efficient Contact-Implicit Trajectory Optimization (CITO), we design demonstration-guided deep Reinforcement Learning (RL), leading to sample-efficient learning. We also present a sim-to-real transfer approach using a privileged training strategy, enabling the robot to perform pivoting manipulation using only proprioception, vision, and force sensing without access to privileged information. Our method is evaluated on several pivoting tasks, demonstrating that it can successfully perform sim-to-real transfer. 

**Abstract (ZH)**: 基于接触的运动规划对于复杂物体、环境和机器人之间的接触交互具有挑战性。基于模型的方法可以在接触约束下高效生成机器人和物体的复杂轨迹。然而，它们往往对模型不准确性敏感，并且需要访问特权信息（例如，物体的质量、尺寸、姿态），从而使它们不适用于新型物体。相比之下，基于学习的方法通常对建模误差更加 robust，但需要大量的数据。在本文中，我们结合这两种方法以提出一种学习闭环翻转操作的框架。利用计算效率高的接触显式轨迹优化（CITO），我们设计了演示指导的深度强化学习（DRL），实现高效学习。我们还提出了一种基于特权训练策略的从仿真到现实的转移方法，使机器人仅通过本体感觉、视觉和力感知即可执行翻转操作，而不需访问特权信息。我们的方法在多种翻转任务上进行了评估，展示了其能够成功实现从仿真到现实的转移。 

---
# Hestia: Hierarchical Next-Best-View Exploration for Systematic Intelligent Autonomous Data Collection 

**Title (ZH)**: Hestia: 分级 next-best-view 探索用于系统化智能自主数据收集 

**Authors**: Cheng-You Lu, Zhuoli Zhuang, Nguyen Thanh Trung Le, Da Xiao, Yu-Cheng Chang, Thomas Do, Srinath Sridhar, Chin-teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01014)  

**Abstract**: Advances in 3D reconstruction and novel view synthesis have enabled efficient, photorealistic rendering, but the data collection process remains largely manual, making it time-consuming and labor-intensive. To address the challenges, this study introduces Hierarchical Next-Best-View Exploration for Systematic Intelligent Autonomous Data Collection (Hestia), which leverages reinforcement learning to learn a generalizable policy for 5-DoF next-best viewpoint prediction. Unlike prior approaches, Hestia systematically defines the next-best-view task by proposing core components such as dataset choice, observation design, action space, reward calculation, and learning schemes, forming a foundation for the planner. Hestia goes beyond prior next-best-view approaches and traditional capture systems through integration and validation in a real-world setup, where a drone serves as a mobile sensor for active scene exploration. Experimental results show that Hestia performs robustly across three datasets and translated object settings in the NVIDIA IsaacLab environment, and proves feasible for real-world deployment. 

**Abstract (ZH)**: 层级最佳视图探索在系统智能自主数据采集中的进展（Hestia）：基于强化学习的5-自由度最佳视图预测策略 

---
# Service Discovery-Based Hybrid Network Middleware for Efficient Communication in Distributed Robotic Systems 

**Title (ZH)**: 基于服务发现的混合网络中间件：分布式机器人系统中的高效通信 

**Authors**: Shiyao Sang, Yinggang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2508.00947)  

**Abstract**: Robotic middleware is fundamental to ensuring reliable communication among system components and is crucial for intelligent robotics, autonomous vehicles, and smart manufacturing. However, existing robotic middleware often struggles to meet the diverse communication demands, optimize data transmission efficiency, and maintain scheduling determinism between Orin computing units in large-scale L4 autonomous vehicle deployments. This paper presents RIMAOS2C, a service discovery-based hybrid network communication middleware designed to tackle these challenges. By leveraging multi-level service discovery multicast, RIMAOS2C supports a wide variety of communication modes, including multiple cross-chip Ethernet protocols and PCIe communication capabilities. Its core mechanism, the Message Bridge, optimizes data flow forwarding and employs shared memory for centralized message distribution, reducing message redundancy and minimizing transmission delay uncertainty. Tested on L4 vehicles and Jetson Orin domain controllers, RIMAOS2C leverages TCP-based ZeroMQ to overcome the large-message transmission bottleneck in native CyberRT. In scenarios with two cross-chip subscribers, it eliminates message redundancy and improves large-data transmission efficiency by 36 to 40 percent while reducing callback latency variation by 42 to 906 percent. This research advances the communication capabilities of robotic operating systems and proposes a novel approach to optimizing communication in distributed computing architectures for autonomous driving. 

**Abstract (ZH)**: 基于服务发现的混合网络通信中间件RIMAOS2C 

---
# BarlowWalk: Self-supervised Representation Learning for Legged Robot Terrain-adaptive Locomotion 

**Title (ZH)**: BarlowWalk：自监督表示学习在适应性地形步行机器人中的应用 

**Authors**: Haodong Huang, Shilong Sun, Yuanpeng Wang, Chiyao Li, Hailin Huang, Wenfu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00939)  

**Abstract**: Reinforcement learning (RL), driven by data-driven methods, has become an effective solution for robot leg motion control problems. However, the mainstream RL methods for bipedal robot terrain traversal, such as teacher-student policy knowledge distillation, suffer from long training times, which limit development efficiency. To address this issue, this paper proposes BarlowWalk, an improved Proximal Policy Optimization (PPO) method integrated with self-supervised representation learning. This method employs the Barlow Twins algorithm to construct a decoupled latent space, mapping historical observation sequences into low-dimensional representations and implementing self-supervision. Meanwhile, the actor requires only proprioceptive information to achieve self-supervised learning over continuous time steps, significantly reducing the dependence on external terrain perception. Simulation experiments demonstrate that this method has significant advantages in complex terrain scenarios. To enhance the credibility of the evaluation, this study compares BarlowWalk with advanced algorithms through comparative tests, and the experimental results verify the effectiveness of the proposed method. 

**Abstract (ZH)**: 基于自监督表示学习的改进 proximal 策略优化方法 BarlowWalk 在两足机器人地形穿越中的应用 

---
# A Survey on Deep Multi-Task Learning in Connected Autonomous Vehicles 

**Title (ZH)**: 连接自动驾驶车辆中深度多任务学习综述 

**Authors**: Jiayuan Wang, Farhad Pourpanah, Q. M. Jonathan Wu, Ning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00917)  

**Abstract**: Connected autonomous vehicles (CAVs) must simultaneously perform multiple tasks, such as object detection, semantic segmentation, depth estimation, trajectory prediction, motion prediction, and behaviour prediction, to ensure safe and reliable navigation in complex environments. Vehicle-to-everything (V2X) communication enables cooperative driving among CAVs, thereby mitigating the limitations of individual sensors, reducing occlusions, and improving perception over long distances. Traditionally, these tasks are addressed using distinct models, which leads to high deployment costs, increased computational overhead, and challenges in achieving real-time performance. Multi-task learning (MTL) has recently emerged as a promising solution that enables the joint learning of multiple tasks within a single unified model. This offers improved efficiency and resource utilization. To the best of our knowledge, this survey is the first comprehensive review focused on MTL in the context of CAVs. We begin with an overview of CAVs and MTL to provide foundational background. We then explore the application of MTL across key functional modules, including perception, prediction, planning, control, and multi-agent collaboration. Finally, we discuss the strengths and limitations of existing methods, identify key research gaps, and provide directions for future research aimed at advancing MTL methodologies for CAV systems. 

**Abstract (ZH)**: 连接自主车辆（CAVs）必须同时执行多个任务，例如物体检测、语义分割、深度估计、轨迹预测、运动预测和行为预测，以确保在复杂环境中实现安全可靠的导航。车辆对万物（V2X）通信使CAVs之间能够实现协同驾驶，从而缓解单一传感器的限制、减少遮挡，并提高远程感知性能。传统上，这些任务是使用不同的模型分别解决的，这导致部署成本高、计算量增加，并且难以实现实时性能。多任务学习（MTL）最近被认为是一种有前途的解决方案，能够在一个统一模型中联合学习多个任务。这一方法提高了效率并优化了资源利用。据我们所知，这是首个专注于CAVs背景下MTL的全面综述。我们从介绍CAVs和MTL开始，提供基础背景。然后探讨MTL在感知、预测、规划、控制和多智能体协作等关键功能模块中的应用。最后，讨论现有方法的优势和局限性，确定关键的研究缺口，并为未来旨在推进MTL方法论的研究提供方向。 

---
# Sparse 3D Perception for Rose Harvesting Robots: A Two-Stage Approach Bridging Simulation and Real-World Applications 

**Title (ZH)**: 基于稀疏三维感知的玫瑰采摘机器人：一种连接仿真与实际应用的两阶段方法 

**Authors**: Taha Samavati, Mohsen Soryani, Sina Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00900)  

**Abstract**: The global demand for medicinal plants, such as Damask roses, has surged with population growth, yet labor-intensive harvesting remains a bottleneck for scalability. To address this, we propose a novel 3D perception pipeline tailored for flower-harvesting robots, focusing on sparse 3D localization of rose centers. Our two-stage algorithm first performs 2D point-based detection on stereo images, followed by depth estimation using a lightweight deep neural network. To overcome the challenge of scarce real-world labeled data, we introduce a photorealistic synthetic dataset generated via Blender, simulating a dynamic rose farm environment with precise 3D annotations. This approach minimizes manual labeling costs while enabling robust model training. We evaluate two depth estimation paradigms: a traditional triangulation-based method and our proposed deep learning framework. Results demonstrate the superiority of our method, achieving an F1 score of 95.6% (synthetic) and 74.4% (real) in 2D detection, with a depth estimation error of 3% at a 2-meter range on synthetic data. The pipeline is optimized for computational efficiency, ensuring compatibility with resource-constrained robotic systems. By bridging the domain gap between synthetic and real-world data, this work advances agricultural automation for specialty crops, offering a scalable solution for precision harvesting. 

**Abstract (ZH)**: 面向花卉收割机器人的三维感知管道：基于稀疏玫瑰中心三维定位的探索 

---
# MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming 

**Title (ZH)**: MonoDream：单目视觉-语言导航与全景梦境 

**Authors**: Shuo Wang, Yongcai Wang, Wanting Li, Yucheng Wang, Maiyue Chen, Kaihui Wang, Zhizhong Su, Xudong Cai, Yeying Jin, Deying Li, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02549)  

**Abstract**: Vision-Language Navigation (VLN) tasks often leverage panoramic RGB and depth inputs to provide rich spatial cues for action planning, but these sensors can be costly or less accessible in real-world deployments. Recent approaches based on Vision-Language Action (VLA) models achieve strong results with monocular input, yet they still lag behind methods using panoramic RGB-D information. We present MonoDream, a lightweight VLA framework that enables monocular agents to learn a Unified Navigation Representation (UNR). This shared feature representation jointly aligns navigation-relevant visual semantics (e.g., global layout, depth, and future cues) and language-grounded action intent, enabling more reliable action prediction. MonoDream further introduces Latent Panoramic Dreaming (LPD) tasks to supervise the UNR, which train the model to predict latent features of panoramic RGB and depth observations at both current and future steps based on only monocular input. Experiments on multiple VLN benchmarks show that MonoDream consistently improves monocular navigation performance and significantly narrows the gap with panoramic-based agents. 

**Abstract (ZH)**: MonoDream：基于单目输入的统一导航表示学习框架 

---
# Uncertainty-Aware Perception-Based Control for Autonomous Racing 

**Title (ZH)**: 基于感知的、考虑不确定性自主赛车控制 

**Authors**: Jelena Trisovic, Andrea Carron, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02494)  

**Abstract**: Autonomous systems operating in unknown environments often rely heavily on visual sensor data, yet making safe and informed control decisions based on these measurements remains a significant challenge. To facilitate the integration of perception and control in autonomous vehicles, we propose a novel perception-based control approach that incorporates road estimation, quantification of its uncertainty, and uncertainty-aware control based on this estimate. At the core of our method is a parametric road curvature model, optimized using visual measurements of the road through a constrained nonlinear optimization problem. This process ensures adherence to constraints on both model parameters and curvature. By leveraging the Frenet frame formulation, we embed the estimated track curvature into the system dynamics, allowing the controller to explicitly account for perception uncertainty and enhancing robustness to estimation errors based on visual input. We validate our approach in a simulated environment, using a high-fidelity 3D rendering engine, and demonstrate its effectiveness in achieving reliable and uncertainty-aware control for autonomous racing. 

**Abstract (ZH)**: 自主系统在未知环境中运作时常依赖视觉传感器数据，然而基于这些测量数据做出安全、可靠的控制决策仍然是一个重大挑战。为了促进自主车辆中感知与控制的整合，我们提出了一种新颖的基于感知的控制方法，该方法结合了道路估计、估计的不确定性量化以及基于此估计的不确定性感知控制。该方法的核心是一种子模型参数化的道路曲率模型，通过对视觉测量的道路进行约束非线性优化问题求解来优化。这一过程确保了模型参数和曲率的约束满足。通过利用弗朗et帧公式，我们将估计的道路曲率嵌入系统动力学中，使得控制器能够明确考虑感知的不确定性，并基于视觉输入的估计误差提高系统的鲁棒性。我们在一个高保真3D渲染引擎模拟环境中验证了该方法，并展示了其在自主赛车中的有效性和不确定性感知控制能力。 

---
# mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera 

**Title (ZH)**: 基于毫米波雷达和摄像头提取道路布局的T路口非视距行人定位方法 

**Authors**: Byeonggyu Park, Hee-Yeun Kim, Byonghyok Choi, Hansang Cho, Byungkwan Kim, Soomok Lee, Mingu Jeon, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.02348)  

**Abstract**: Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method. 

**Abstract (ZH)**: 城市环境中非视线（NLoS）区域的人行者定位是自主驾驶系统面临的一项重大挑战。尽管毫米波雷达在这些场景下检测物体方面显示出潜力，但由于多径反射造成的2D雷达点云（PCD）数据失真，使得准确的空间推断变得困难。此外，尽管相机图像提供了高分辨率的视觉信息，但它们缺乏深度感知，无法直接观察NLoS区域中的物体。在本文中，我们提出了一种新颖的框架，通过从相机推断的道路布局来解释雷达PCD，以实现NLoS行人定位。所提出的方法利用相机的视觉信息来解释2D雷达PCD，使空间场景重建成为可能。通过在真实车辆上安装雷达-相机系统进行的实验验证了所提出方法的有效性。定位性能通过在户外NLoS驾驶环境中收集的数据集进行评估，展示了该方法的实际适用性。 

---
# Correspondence-Free Fast and Robust Spherical Point Pattern Registration 

**Title (ZH)**: correspondence-free快速且稳健的球面点模式注册 

**Authors**: Anik Sarker, Alan T. Asbeck  

**Link**: [PDF](https://arxiv.org/pdf/2508.02339)  

**Abstract**: Existing methods for rotation estimation between two spherical ($\mathbb{S}^2$) patterns typically rely on spherical cross-correlation maximization between two spherical function. However, these approaches exhibit computational complexities greater than cubic $O(n^3)$ with respect to rotation space discretization and lack extensive evaluation under significant outlier contamination. To this end, we propose a rotation estimation algorithm between two spherical patterns with linear time complexity $O(n)$. Unlike existing spherical-function-based methods, we explicitly represent spherical patterns as discrete 3D point sets on the unit sphere, reformulating rotation estimation as a spherical point-set alignment (i.e., Wahba problem for 3D unit vectors). Given the geometric nature of our formulation, our spherical pattern alignment algorithm naturally aligns with the Wahba problem framework for 3D unit vectors. Specifically, we introduce three novel algorithms: (1) SPMC (Spherical Pattern Matching by Correlation), (2) FRS (Fast Rotation Search), and (3) a hybrid approach (SPMC+FRS) that combines the advantages of the previous two methods. Our experiments demonstrate that in the $\mathbb{S}^2$ domain and in correspondence-free settings, our algorithms are over 10x faster and over 10x more accurate than current state-of-the-art methods for the Wahba problem with outliers. We validate our approach through extensive simulations on a new dataset of spherical patterns, the ``Robust Vector Alignment Dataset. "Furthermore, we adapt our methods to two real-world tasks: (i) Point Cloud Registration (PCR) and (ii) rotation estimation for spherical images. 

**Abstract (ZH)**: 基于球面模式的旋转估计：线性时间复杂度的新型算法 

---
# Vision Language Model-based Testing of Industrial Autonomous Mobile Robots 

**Title (ZH)**: 基于视觉语言模型的工业自主移动机器人测试 

**Authors**: Jiahui Wu, Chengjie Lu, Aitor Arrieta, Shaukat Ali, Thomas Peyrucain  

**Link**: [PDF](https://arxiv.org/pdf/2508.02338)  

**Abstract**: Autonomous Mobile Robots (AMRs) are deployed in diverse environments (e.g., warehouses, retail spaces, and offices), where they work alongside humans. Given that human behavior can be unpredictable and that AMRs may not have been trained to handle all possible unknown and uncertain behaviors, it is important to test AMRs under a wide range of human interactions to ensure their safe behavior. Moreover, testing in real environments with actual AMRs and humans is often costly, impractical, and potentially hazardous (e.g., it could result in human injury). To this end, we propose a Vision Language Model (VLM)-based testing approach (RVSG) for industrial AMRs developed by PAL Robotics in Spain. Based on the functional and safety requirements, RVSG uses the VLM to generate diverse human behaviors that violate these requirements. We evaluated RVSG with several requirements and navigation routes in a simulator using the latest AMR from PAL Robotics. Our results show that, compared with the baseline, RVSG can effectively generate requirement-violating scenarios. Moreover, RVSG-generated scenarios increase variability in robot behavior, thereby helping reveal their uncertain behaviors. 

**Abstract (ZH)**: 基于视觉语言模型的工业AMR测试方法（RVSG） 

---
# An Event-based Fast Intensity Reconstruction Scheme for UAV Real-time Perception 

**Title (ZH)**: 基于事件的快速强度重建方案用于UAV实时感知 

**Authors**: Xin Dong, Yiwei Zhang, Yangjie Cui, Jinwu Xiang, Daochun Li, Zhan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02238)  

**Abstract**: Event cameras offer significant advantages, including a wide dynamic range, high temporal resolution, and immunity to motion blur, making them highly promising for addressing challenging visual conditions. Extracting and utilizing effective information from asynchronous event streams is essential for the onboard implementation of event cameras. In this paper, we propose a streamlined event-based intensity reconstruction scheme, event-based single integration (ESI), to address such implementation challenges. This method guarantees the portability of conventional frame-based vision methods to event-based scenarios and maintains the intrinsic advantages of event cameras. The ESI approach reconstructs intensity images by performing a single integration of the event streams combined with an enhanced decay algorithm. Such a method enables real-time intensity reconstruction at a high frame rate, typically 100 FPS. Furthermore, the relatively low computation load of ESI fits onboard implementation suitably, such as in UAV-based visual tracking scenarios. Extensive experiments have been conducted to evaluate the performance comparison of ESI and state-of-the-art algorithms. Compared to state-of-the-art algorithms, ESI demonstrates remarkable runtime efficiency improvements, superior reconstruction quality, and a high frame rate. As a result, ESI enhances UAV onboard perception significantly under visual adversary surroundings. In-flight tests, ESI demonstrates effective performance for UAV onboard visual tracking under extremely low illumination conditions(2-10lux), whereas other comparative algorithms fail due to insufficient frame rate, poor image quality, or limited real-time performance. 

**Abstract (ZH)**: 事件相机因其宽动态范围、高时间分辨率和运动模糊免疫等显著优势，成为应对复杂视觉条件的理想选择。从异步事件流中提取并利用有效信息对于事件相机的机载实现至关重要。本文提出了一种简化的事件驱动强度重建方案——事件驱动单积分(ESI)，以应对这些实现挑战。该方法确保了传统基于帧的视觉方法在事件驱动场景中的可移植性，并保持事件相机的固有优势。ESI通过结合改进的衰减算法对事件流进行单次积分，从而实现了实时高帧率（通常为100 FPS）强度重建。此外，ESI较低的计算负载使其适合于机载实现，如无人机视觉跟踪场景。通过广泛实验，ESI在与最新算法性能对比中展示了显著的运行时效率提升、卓越的重建质量和高帧率。因此，ESI在视觉对抗环境下显著增强了无人机的机载感知能力。飞行测试表明，在极低光照条件下（2-10lux），ESI表现出有效的无人机机载视觉跟踪性能，而其他比较算法因帧率不足、图像质量差或实时性能受限而失败。 

---
# Towards Immersive Human-X Interaction: A Real-Time Framework for Physically Plausible Motion Synthesis 

**Title (ZH)**: 面向沉浸式人机交互的实时物理 plausible 运动合成框架 

**Authors**: Kaiyang Ji, Ye Shi, Zichen Jin, Kangyi Chen, Lan Xu, Yuexin Ma, Jingyi Yu, Jingya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02106)  

**Abstract**: Real-time synthesis of physically plausible human interactions remains a critical challenge for immersive VR/AR systems and humanoid robotics. While existing methods demonstrate progress in kinematic motion generation, they often fail to address the fundamental tension between real-time responsiveness, physical feasibility, and safety requirements in dynamic human-machine interactions. We introduce Human-X, a novel framework designed to enable immersive and physically plausible human interactions across diverse entities, including human-avatar, human-humanoid, and human-robot systems. Unlike existing approaches that focus on post-hoc alignment or simplified physics, our method jointly predicts actions and reactions in real-time using an auto-regressive reaction diffusion planner, ensuring seamless synchronization and context-aware responses. To enhance physical realism and safety, we integrate an actor-aware motion tracking policy trained with reinforcement learning, which dynamically adapts to interaction partners' movements while avoiding artifacts like foot sliding and penetration. Extensive experiments on the Inter-X and InterHuman datasets demonstrate significant improvements in motion quality, interaction continuity, and physical plausibility over state-of-the-art methods. Our framework is validated in real-world applications, including virtual reality interface for human-robot interaction, showcasing its potential for advancing human-robot collaboration. 

**Abstract (ZH)**: 实时合成物理上合理的真人互动仍然是沉浸式VR/AR系统和类人机器人领域的关键挑战。现有的方法在运动生成的运动学方面取得了进展，但往往未能解决动态人机互动中实时响应性、物理可行性与安全性之间的基本矛盾。我们提出了Human-X，一种新型框架，旨在促进多样实体间的沉浸式和物理上合理的互动，包括人-avatar、人-类人机器人和人-机器人系统。与现有的侧重于事后对齐或简化物理的方法不同，我们的方法使用自回归反应扩散规划器在实时中联合预测动作与反应，确保无缝同步和上下文感知的响应。为了增强物理真实性和安全性，我们集成了一个基于强化学习训练的意识动作跟踪策略，该策略能够动态适应互动伙伴的运动，同时避免诸如脚滑和穿插等伪影。在Inter-X和InterHuman数据集上的广泛实验表明，与现有最先进的方法相比，在运动质量、互动连续性和物理合理性方面取得了显著改善。我们的框架在实际应用中得到了验证，包括用于人-机器人交互的虚拟现实接口，展示了其在推进人-机器人协作方面的潜力。 

---
# ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks 

**Title (ZH)**: ROVER：带有视觉-语言模型的递归视频推理用于嵌入式任务 

**Authors**: Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2508.01943)  

**Abstract**: Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: this https URL 

**Abstract (ZH)**: Vision-language模型(VLMs)在各种视觉理解任务中展现了令人印象深刻的性能，但在需要处理视频中长时间序列镜头的推理任务上仍存在不足。这限制了它们在需要连续视觉输入中长时间序列镜头推理的实体环境中的应用。为了解决这一限制，我们提出了一种名为ROVER（递归视频推理）的框架，该框架使模型能够递归地将长时间视频轨迹分解为与轨迹中较短子任务相对应的段。通过这种方式，ROVER促进了对时间局部镜头序列的更集中和准确的推理，同时保留全局上下文。我们使用上下文学习方法实现的ROVER在多样的OpenX Embodiment视频以及源自RoboCasa的新数据集（包含543个视频，展示了27个机器人操作任务中专家和扰动非专家轨迹）上，针对三种视频推理任务（任务进度估计、帧级自然语言推理和视频问题回答）进行了评估。ROVER在所有任务中都优于强基线。我们观察到，通过减少模型在每个时间步长上需要推理的镜头数量，ROVER减轻了意外或非最优时刻的幻觉。此外，通过启用特定于子任务的滑动上下文窗口，ROVER的时间复杂度随着视频长度线性增长，从而在渐近上优于基线。更多演示、代码和数据请访问：this https URL。 

---
# CVD-SfM: A Cross-View Deep Front-end Structure-from-Motion System for Sparse Localization in Multi-Altitude Scenes 

**Title (ZH)**: CVD-SfM：多海拔场景稀疏定位的跨视图深度前端结构从运动系统 

**Authors**: Yaxuan Li, Yewei Huang, Bijay Gaudel, Hamidreza Jafarnejadsani, Brendan Englot  

**Link**: [PDF](https://arxiv.org/pdf/2508.01936)  

**Abstract**: We present a novel multi-altitude camera pose estimation system, addressing the challenges of robust and accurate localization across varied altitudes when only considering sparse image input. The system effectively handles diverse environmental conditions and viewpoint variations by integrating the cross-view transformer, deep features, and structure-from-motion into a unified framework. To benchmark our method and foster further research, we introduce two newly collected datasets specifically tailored for multi-altitude camera pose estimation; datasets of this nature remain rare in the current literature. The proposed framework has been validated through extensive comparative analyses on these datasets, demonstrating that our system achieves superior performance in both accuracy and robustness for multi-altitude sparse pose estimation tasks compared to existing solutions, making it well suited for real-world robotic applications such as aerial navigation, search and rescue, and automated inspection. 

**Abstract (ZH)**: 我们提出了一种新颖的多海拔相机姿态估计系统，解决了仅考虑稀疏图像输入时在不同海拔高度下实现稳健且准确定位的挑战。该系统通过将跨视图变换器、深度特征和结构从运动集成到统一框架中，有效地处理了各种环境条件和视角变化。为了对该方法进行基准测试并推动进一步研究，我们引入了两个专门为多海拔相机姿态估计收集的新数据集；这类数据集目前在文献中较为罕见。所提出的框架通过对这些数据集进行广泛的比较分析得到了验证，表明我们的系统在多海拔稀疏姿态估计任务中比现有解决方案在准确性和鲁棒性方面表现更优，使其非常适合用于实际的机器人应用，如高空导航、搜救和自动化检测。 

---
# MUTE-DSS: A Digital-Twin-Based Decision Support System for Minimizing Underwater Radiated Noise in Ship Voyage Planning 

**Title (ZH)**: MUTE-DSS: 基于数字孪生的海底辐射噪声最小化船舶航程规划决策支持系统 

**Authors**: Akash Venkateshwaran, Indu Kant Deo, Rajeev K. Jaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.01907)  

**Abstract**: We present a novel MUTE-DSS, a digital-twin-based decision support system for minimizing underwater radiated noise (URN) during ship voyage planning. It is a ROS2-centric framework that integrates state-of-the-art acoustic models combining a semi-empirical reference spectrum for near-field modeling with 3D ray tracing for propagation losses for far-field modeling, offering real-time computation of the ship noise signature, alongside a data-driven Southern resident killer whale distribution model. The proposed DSS performs a two-stage optimization pipeline: Batch Informed Trees for collision-free ship routing and a genetic algorithm for adaptive ship speed profiling under voyage constraints that minimizes cumulative URN exposure to marine mammals. The effectiveness of MUTE-DSS is demonstrated through case studies of ships operating between the Strait of Georgia and the Strait of Juan de Fuca, comparing optimized voyages against baseline trajectories derived from automatic identification system data. Results show substantial reductions in noise exposure level, up to 7.14 dB, corresponding to approximately an 80.68% reduction in a simplified scenario, and an average 4.90 dB reduction, corresponding to approximately a 67.6% reduction in a more realistic dynamic setting. These results illustrate the adaptability and practical utility of the proposed decision support system. 

**Abstract (ZH)**: 基于数字孪生的MUTE-DSS决策支持系统：用于 ship 航行规划期间最小化水下辐射噪声（URN） 

---
# A Simple Algebraic Solution for Estimating the Pose of a Camera from Planar Point Features 

**Title (ZH)**: 从平面点特征估算相机姿态的简单代数解法 

**Authors**: Tarek Bouazza, Tarek Hamel, Claude Samson  

**Link**: [PDF](https://arxiv.org/pdf/2508.01836)  

**Abstract**: This paper presents a simple algebraic method to estimate the pose of a camera relative to a planar target from $n \geq 4$ reference points with known coordinates in the target frame and their corresponding bearing measurements in the camera frame. The proposed approach follows a hierarchical structure; first, the unit vector normal to the target plane is determined, followed by the camera's position vector, its distance to the target plane, and finally, the full orientation. To improve the method's robustness to measurement noise, an averaging methodology is introduced to refine the estimation of the target's normal direction. The accuracy and robustness of the approach are validated through extensive experiments. 

**Abstract (ZH)**: 本文提出了一种简单代数方法，用于从目标框架中已知坐标且在摄像机框架中有相应视线测量的至少4个参考点，估计摄像机相对于平面目标的姿态。该提出的方案遵循层次结构；首先确定目标平面的单位法向量，随后确定摄像机的位置矢量、到目标平面的距离，最后确定完整的姿态方向。为了提高方法对测量噪声的鲁棒性，引入了一种平均方法来细化目标法向方向的估计。通过广泛的实验验证了该方法的准确性和鲁棒性。 

---
# DiffSemanticFusion: Semantic Raster BEV Fusion for Autonomous Driving via Online HD Map Diffusion 

**Title (ZH)**: DiffSemanticFusion: 基于在线高精度地图扩散的语义栅格BEV融合技术在自动驾驶中的应用 

**Authors**: Zhigang Sun, Yiru Wang, Anqing Jiang, Shuo Wang, Yu Gao, Yuwen Heng, Shouyi Zhang, An He, Hao Jiang, Jinhao Chai, Zichong Gu, Wang Jijun, Shichen Tang, Lavdim Halilaj, Juergen Luettin, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01778)  

**Abstract**: Autonomous driving requires accurate scene understanding, including road geometry, traffic agents, and their semantic relationships. In online HD map generation scenarios, raster-based representations are well-suited to vision models but lack geometric precision, while graph-based representations retain structural detail but become unstable without precise maps. To harness the complementary strengths of both, we propose DiffSemanticFusion -- a fusion framework for multimodal trajectory prediction and planning. Our approach reasons over a semantic raster-fused BEV space, enhanced by a map diffusion module that improves both the stability and expressiveness of online HD map representations. We validate our framework on two downstream tasks: trajectory prediction and planning-oriented end-to-end autonomous driving. Experiments on real-world autonomous driving benchmarks, nuScenes and NAVSIM, demonstrate improved performance over several state-of-the-art methods. For the prediction task on nuScenes, we integrate DiffSemanticFusion with the online HD map informed QCNet, achieving a 5.1\% performance improvement. For end-to-end autonomous driving in NAVSIM, DiffSemanticFusion achieves state-of-the-art results, with a 15\% performance gain in NavHard scenarios. In addition, extensive ablation and sensitivity studies show that our map diffusion module can be seamlessly integrated into other vector-based approaches to enhance performance. All artifacts are available at this https URL. 

**Abstract (ZH)**: 自主驾驶需要准确的场景理解，包括道路几何、交通代理及其语义关系。在在线高清地图生成场景中，基于栅格的表示适合视觉模型但缺乏几何精度，而基于图的表示保留了结构细节但在没有精确地图的情况下会变得不稳定。为了充分利用两者的互补优势，我们提出了DiffSemanticFusion——一种多模态轨迹预测和规划的融合框架。我们的方法在语义栅格融合的BEV空间中进行推理，并通过地图扩散模块提高了在线高清地图表示的稳定性和表达性。我们在轨迹预测和规划导向的端到端自主驾驶两个下游任务上验证了该框架。在nuScenes和NASIM等真实世界自主驾驶基准上的实验表明，该框架相对于几种最先进的方法具有更好的性能。在nuScenes预测任务中，我们将DiffSemanticFusion与基于在线高清地图的QCNet结合使用，实现了5.1%的性能提升。在NASIM端到端自主驾驶中，DiffSemanticFusion在NavHard场景中达到了最先进的结果，性能提高了15%。此外，广泛的消融和敏感性研究显示，我们的地图扩散模块可以无缝集成到其他向量方法中以提升性能。所有成果均可在以下链接访问：this https URL。 

---
# Dynamic Robot-Assisted Surgery with Hierarchical Class-Incremental Semantic Segmentation 

**Title (ZH)**: 动态机器人辅助手术与层次类增量语义分割 

**Authors**: Julia Hindel, Ema Mekic, Enamundram Naga Karthik, Rohit Mohan, Daniele Cattaneo, Maria Kalweit, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2508.01713)  

**Abstract**: Robot-assisted surgeries rely on accurate and real-time scene understanding to safely guide surgical instruments. However, segmentation models trained on static datasets face key limitations when deployed in these dynamic and evolving surgical environments. Class-incremental semantic segmentation (CISS) allows models to continually adapt to new classes while avoiding catastrophic forgetting of prior knowledge, without training on previous data. In this work, we build upon the recently introduced Taxonomy-Oriented Poincaré-regularized Incremental Class Segmentation (TOPICS) approach and propose an enhanced variant, termed TOPICS+, specifically tailored for robust segmentation of surgical scenes. Concretely, we incorporate the Dice loss into the hierarchical loss formulation to handle strong class imbalances, introduce hierarchical pseudo-labeling, and design tailored label taxonomies for robotic surgery environments. We also propose six novel CISS benchmarks designed for robotic surgery environments including multiple incremental steps and several semantic categories to emulate realistic class-incremental settings in surgical environments. In addition, we introduce a refined set of labels with more than 144 classes on the Syn-Mediverse synthetic dataset, hosted online as an evaluation benchmark. We make the code and trained models publicly available at this http URL. 

**Abstract (ZH)**: 机器人辅助手术依赖于准确的实时场景理解以安全地引导手术器械。然而，训练于静态数据集的分割模型在这些动态且不断变化的手术环境中部署时面临关键限制。类别增量语义分割(CISS)允许模型在不断适应新类别时避免遗忘先前的知识，而无需重新训练之前的数据。在本工作中，我们建立在最近提出的分类学导向的庞加莱正则化增量类别分割(TOPICS)方法之上，并提出了一种增强变体，称为TOPICS+，特别针对机器人手术环境中的稳健分割进行了优化。具体来说，我们将Dice损失纳入层次损失公式以处理类别不平衡，引入层次伪标签，并为机器人手术环境设计专门的标签分类学。我们还提出了六个针对机器人手术环境设计的新CISS基准，包括多步增量和多个语义类别，以模拟手术环境中真实的类别增量设置。此外，我们引入了一组精炼的标签，涵盖Syn-Mediverse合成数据集中的超过144个类别，并在线作为评估基准提供。我们已将代码和训练模型公开发布在特定网址。 

---
# 3DRot: 3D Rotation Augmentation for RGB-Based 3D Tasks 

**Title (ZH)**: 3DRot: 基于RGB的3D任务的3D旋转增强 

**Authors**: Shitian Yang, Deyu Li, Xiaoke Jiang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01423)  

**Abstract**: RGB-based 3D tasks, e.g., 3D detection, depth estimation, 3D keypoint estimation, still suffer from scarce, expensive annotations and a thin augmentation toolbox, since most image transforms, including resize and rotation, disrupt geometric consistency. In this paper, we introduce 3DRot, a plug-and-play augmentation that rotates and mirrors images about the camera's optical center while synchronously updating RGB images, camera intrinsics, object poses, and 3D annotations to preserve projective geometry-achieving geometry-consistent rotations and reflections without relying on any scene depth. We validate 3DRot with a classical 3D task, monocular 3D detection. On SUN RGB-D dataset, 3DRot raises $IoU_{3D}$ from 43.21 to 44.51, cuts rotation error (ROT) from 22.91$^\circ$ to 20.93$^\circ$, and boosts $mAP_{0.5}$ from 35.70 to 38.11. As a comparison, Cube R-CNN adds 3 other datasets together with SUN RGB-D for monocular 3D estimation, with a similar mechanism and test dataset, increases $IoU_{3D}$ from 36.2 to 37.8, boosts $mAP_{0.5}$ from 34.7 to 35.4. Because it operates purely through camera-space transforms, 3DRot is readily transferable to other 3D tasks. 

**Abstract (ZH)**: 基于RGB的3D任务，如3D检测、深度估计、3D关键点估计，仍然受到稀缺且昂贵的标注以及有限的增强工具箱的困扰，因为大多数图像变换，包括缩放和旋转，会破坏几何一致性。在本文中，我们介绍了一种即插即用增强方法3DRot，该方法在保持投影几何关系的前提下，围绕相机光学中心旋转和镜像图像，并同步更新RGB图像、相机内参、物体姿态和3D标注。3DRot实现了几何一致的旋转和反射，无需依赖任何场景深度。我们通过单目3D检测这一经典3D任务验证了3DRot。在SUN RGB-D数据集上，3DRot将$IoU_{3D}$从43.21提高到44.51，将旋转误差（ROT）从22.91°降低到20.93°，并将$ mAP_{0.5}$从35.70提升到38.11。相比之下，Cube R-CNN通过结合SUN RGB-D和其他3个数据集进行单目3D估计，采用相似的机制和测试集，将$IoU_{3D}$从36.2提升到37.8，将$ mAP_{0.5}$从34.7提升到35.4。由于仅通过相机空间变换操作，3DRot易于应用于其他3D任务。 

---
# NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration 

**Title (ZH)**: NarrativeGuide: 一种基于LLM的叙事移动机器人，用于远程地点探索 

**Authors**: Yaxin Hu, Arissa J. Sato, Jingxin Du, Chenming Ye, Anjun Zhu, Pragathi Praveena, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01235)  

**Abstract**: Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments. 

**Abstract (ZH)**: 基于位置感知的LLMNarrative能力集成的移动机器人支持远程探索 

---
# Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization? 

**Title (ZH)**: 从更广泛的视角来看：房间风格知识能帮助视觉floorplan定位吗？ 

**Authors**: Bolei Chen, Shengsheng Yan, Yongzheng Cui, Jiaxu Kang, Ping Zhong, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01216)  

**Abstract**: Since a building's floorplan remains consistent over time and is inherently robust to changes in visual appearance, visual Floorplan Localization (FLoc) has received increasing attention from researchers. However, as a compact and minimalist representation of the building's layout, floorplans contain many repetitive structures (e.g., hallways and corners), thus easily result in ambiguous localization. Existing methods either pin their hopes on matching 2D structural cues in floorplans or rely on 3D geometry-constrained visual pre-trainings, ignoring the richer contextual information provided by visual images. In this paper, we suggest using broader visual scene context to empower FLoc algorithms with scene layout priors to eliminate localization uncertainty. In particular, we propose an unsupervised learning technique with clustering constraints to pre-train a room discriminator on self-collected unlabeled room images. Such a discriminator can empirically extract the hidden room type of the observed image and distinguish it from other room types. By injecting the scene context information summarized by the discriminator into an FLoc algorithm, the room style knowledge is effectively exploited to guide definite visual FLoc. We conducted sufficient comparative studies on two standard visual Floc benchmarks. Our experiments show that our approach outperforms state-of-the-art methods and achieves significant improvements in robustness and accuracy. 

**Abstract (ZH)**: 基于广泛视觉场景上下文的建筑平面图定位方法 

---
# A Coarse-to-Fine Approach to Multi-Modality 3D Occupancy Grounding 

**Title (ZH)**: 从粗到细的多模态3D占用语义 grounding 方法 

**Authors**: Zhan Shi, Song Wang, Junbo Chen, Jianke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01197)  

**Abstract**: Visual grounding aims to identify objects or regions in a scene based on natural language descriptions, essential for spatially aware perception in autonomous driving. However, existing visual grounding tasks typically depend on bounding boxes that often fail to capture fine-grained details. Not all voxels within a bounding box are occupied, resulting in inaccurate object representations. To address this, we introduce a benchmark for 3D occupancy grounding in challenging outdoor scenes. Built on the nuScenes dataset, it integrates natural language with voxel-level occupancy annotations, offering more precise object perception compared to the traditional grounding task. Moreover, we propose GroundingOcc, an end-to-end model designed for 3D occupancy grounding through multi-modal learning. It combines visual, textual, and point cloud features to predict object location and occupancy information from coarse to fine. Specifically, GroundingOcc comprises a multimodal encoder for feature extraction, an occupancy head for voxel-wise predictions, and a grounding head to refine localization. Additionally, a 2D grounding module and a depth estimation module enhance geometric understanding, thereby boosting model performance. Extensive experiments on the benchmark demonstrate that our method outperforms existing baselines on 3D occupancy grounding. The dataset is available at this https URL. 

**Abstract (ZH)**: 三维占用语义定位旨在基于自然语言描述在场景中识别对象或区域，对于自动驾驶中的空间感知至关重要。然而，现有的语义定位任务通常依赖于边界框，往往无法捕捉到细粒度的细节。边界盒内的所有体素并不一定被占用，导致对象表示不准确。为了解决这一问题，我们引入了一个针对具有挑战性的室外场景的三维占用语义定位基准。该基准基于nuScenes数据集，将自然语言与体素级别占用标注相结合，相比传统的语义定位任务提供了更精确的对象感知。此外，我们提出了一种名为GroundingOcc的端到端模型，用于通过多模态学习进行三维占用语义定位。该模型结合视觉、文本和点云特征，从粗到细预测对象位置和占用信息。具体而言，GroundingOcc包括一个用于特征提取的多模态编码器、一个用于体素级预测的占用头和一个用于细化定位的语义定位头。此外，2D语义定位模块和深度估计模块增强了几何理解，从而提高了模型性能。基准上的大量实验表明，我们的方法在三维占用语义定位上优于现有基线。数据集可在以下链接获取。 

---
# T2S: Tokenized Skill Scaling for Lifelong Imitation Learning 

**Title (ZH)**: T2S: 分词技能扩展for 全生命周期imitation learning 

**Authors**: Hongquan Zhang, Jingyu Gong, Zhizhong Zhang, Xin Tan, Yanyun Qu, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.01167)  

**Abstract**: The main challenge in lifelong imitation learning lies in the balance between mitigating catastrophic forgetting of previous skills while maintaining sufficient capacity for acquiring new ones. However, current approaches typically address these aspects in isolation, overlooking their internal correlation in lifelong skill acquisition. We address this limitation with a unified framework named Tokenized Skill Scaling (T2S). Specifically, by tokenizing the model parameters, the linear parameter mapping of the traditional transformer is transformed into cross-attention between input and learnable tokens, thereby enhancing model scalability through the easy extension of new tokens. Additionally, we introduce language-guided skill scaling to transfer knowledge across tasks efficiently and avoid linearly growing parameters. Extensive experiments across diverse tasks demonstrate that T2S: 1) effectively prevents catastrophic forgetting (achieving an average NBT of 1.0% across the three LIBERO task suites), 2) excels in new skill scaling with minimal increases in trainable parameters (needing only 8.0% trainable tokens in an average of lifelong tasks), and 3) enables efficient knowledge transfer between tasks (achieving an average FWT of 77.7% across the three LIBERO task suites), offering a promising solution for lifelong imitation learning. 

**Abstract (ZH)**: lifelong imitation学习的主要挑战在于在减轻先前技能灾难性遗忘的同时维持足够容量以获取新技能之间的平衡。然而，当前方法通常将这两方面分离处理，忽略了它们在终身技能获取中的内在关联。我们通过一个名为Tokenized Skill Scaling (T2S)的统一框架来克服这一限制。具体而言，通过模型参数的-token化-，传统的变压器的线性参数映射被转换为输入和可学习标记之间的交叉注意，从而通过新标记的易于扩展性增强模型的可扩展性。此外，我们引入了由语言指导的技能缩放，以高效地跨任务转移知识并避免参数线性增长。广泛的跨多种任务的实验证明了T2S：1) 有效防止了灾难性遗忘（在三个LIBERO任务套件中平均NBT为1.0%），2) 在新技能缩放方面表现优异，需要的可训练参数小幅增加（平均终身任务中仅为8.0%的可训练标记），3) 使任务之间的知识转移变得高效（在三个LIBERO任务套件中平均FWT为77.7%），提供了一个有前景的终身模仿学习解决方案。 

---
# RoboLinker: A Diffusion-model-based Matching Clothing Generator Between Humans and Companion Robots 

**Title (ZH)**: RoboLinker: 一种基于扩散模型的人与伙伴机器人服饰匹配生成器 

**Authors**: Jing Tang, Qing Xiao, Kunxu Du, Zaiqiao Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.01165)  

**Abstract**: We present RoboLinker, a generative design system that creates matching outfits for humans and their robots. Using a diffusion-based model, the system takes a robot image and a style prompt from users as input, and outputs a human outfit that visually complements the robot's attire. Through an interactive interface, users can refine the generated designs. We evaluate RoboLinker with both humanoid and pet-like robots, demonstrating its capacity to produce stylistically coherent and emotionally resonant results. 

**Abstract (ZH)**: RoboLinker：一种为人类和其机器人创建匹配服装的生成设计系统 

---
# REACT: A Real-Time Edge-AI Based V2X Framework for Accident Avoidance in Autonomous Driving System 

**Title (ZH)**: REACT：基于边缘AI的实时V2X框架以避免自动驾驶系统中的事故 

**Authors**: Fengze Yang, Bo Yu, Yang Zhou, Xuewen Luo, Zhengzhong Tu, Chenxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01057)  

**Abstract**: Collisions caused by human error are the most common type of multi-vehicle crash, highlighting the critical need for autonomous driving (AD) systems to leverage cooperative perception through Vehicle-to-Everything (V2X) communication. This capability extends situational awareness beyond the limitations of onboard sensors. However, current transformer-based V2X frameworks suffer from limited generalization, shallow contextual reasoning, and reliance on mono-modal inputs. Vision-Language Models (VLMs) offer enhanced reasoning and multimodal integration but typically fall short of real-time performance requirements in safety-critical applications. This paper presents REACT, a real-time, V2X-integrated trajectory optimization framework built upon a fine-tuned lightweight VLM. REACT integrates a set of specialized modules that process multimodal inputs into optimized, risk-aware trajectories. To ensure real-time performance on edge devices, REACT incorporates edge adaptation strategies that reduce model complexity and accelerate inference. Evaluated on the DeepAccident benchmark, REACT achieves state-of-the-art performance, a 77% collision rate reduction, a 48.2% Video Panoptic Quality (VPQ), and a 0.57-second inference latency on the Jetson AGX Orin. Ablation studies validate the contribution of each input, module, and edge adaptation strategy. These results demonstrate the feasibility of lightweight VLMs for real-time edge-based cooperative planning and showcase the potential of language-guided contextual reasoning to improve safety and responsiveness in autonomous driving. 

**Abstract (ZH)**: 由人类错误引起的碰撞是最常见的多车事故类型，突显了自动驾驶系统通过Vehicle-to-Everything (V2X)通信进行协作感知的迫切需求。这一能力能够超越车载传感器的局限性扩展情况感知。然而，当前基于变换器的V2X框架存在泛化能力有限、浅层上下文推理以及依赖单模输入等问题。视觉-语言模型（VLMs）提供了增强的推理能力和多模态整合，但在关键安全应用中通常无法满足实时性能要求。本文提出了一种基于微调轻量级VLM的实时V2X集成轨迹优化框架——REACT。REACT结合了一系列专门模块，将多模态输入处理成优化且风险意识强的轨迹。为了确保边缘设备上的实时性能，REACT采用了边端适配策略来减少模型复杂性和加速推理。在DeepAccident基准上评估，REACT实现了最先进的性能，碰撞率降低了77%，视频泛光质量（VPQ）为48.2%，推理延迟为0.57秒（Jetson AGX Orin）。消融实验验证了每个输入、模块和边端适配策略的贡献。这些结果表明了轻量级VLM在实时边缘基于协作规划的可行性，并展示了语言引导上下文推理在提升自动驾驶安全性和响应性方面的潜力。 

---
# Cooperative Perception: A Resource-Efficient Framework for Multi-Drone 3D Scene Reconstruction Using Federated Diffusion and NeRF 

**Title (ZH)**: 协同感知：一种基于联邦扩散和NeRF的多无人机3D场景重建高效框架 

**Authors**: Massoud Pourmandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00967)  

**Abstract**: The proposal introduces an innovative drone swarm perception system that aims to solve problems related to computational limitations and low-bandwidth communication, and real-time scene reconstruction. The framework enables efficient multi-agent 3D/4D scene synthesis through federated learning of shared diffusion model and YOLOv12 lightweight semantic extraction and local NeRF updates while maintaining privacy and scalability. The framework redesigns generative diffusion models for joint scene reconstruction, and improves cooperative scene understanding, while adding semantic-aware compression protocols. The approach can be validated through simulations and potential real-world deployment on drone testbeds, positioning it as a disruptive advancement in multi-agent AI for autonomous systems. 

**Abstract (ZH)**: 一种解决计算限制、低带宽通信和实时场景重建问题的创新无人机群感知系统及其框架 

---
# Visuo-Acoustic Hand Pose and Contact Estimation 

**Title (ZH)**: 视觉-听觉手部姿态和接触估计 

**Authors**: Yuemin Ma, Uksang Yoo, Yunchao Yao, Shahram Najam Syed, Luca Bondi, Jonathan Francis, Jean Oh, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.00852)  

**Abstract**: Accurately estimating hand pose and hand-object contact events is essential for robot data-collection, immersive virtual environments, and biomechanical analysis, yet remains challenging due to visual occlusion, subtle contact cues, limitations in vision-only sensing, and the lack of accessible and flexible tactile sensing. We therefore introduce VibeMesh, a novel wearable system that fuses vision with active acoustic sensing for dense, per-vertex hand contact and pose estimation. VibeMesh integrates a bone-conduction speaker and sparse piezoelectric microphones, distributed on a human hand, emitting structured acoustic signals and capturing their propagation to infer changes induced by contact. To interpret these cross-modal signals, we propose a graph-based attention network that processes synchronized audio spectra and RGB-D-derived hand meshes to predict contact with high spatial resolution. We contribute: (i) a lightweight, non-intrusive visuo-acoustic sensing platform; (ii) a cross-modal graph network for joint pose and contact inference; (iii) a dataset of synchronized RGB-D, acoustic, and ground-truth contact annotations across diverse manipulation scenarios; and (iv) empirical results showing that VibeMesh outperforms vision-only baselines in accuracy and robustness, particularly in occluded or static-contact settings. 

**Abstract (ZH)**: 准确估计手部姿态和手物接触事件对于机器人数据收集、沉浸式虚拟环境和生物力学分析至关重要，但由于视觉遮挡、微妙的接触提示、单一视觉感知的限制以及缺乏可访问和灵活的触觉传感技术，这仍然是一个难题。因此，我们提出了VibeMesh，一种结合视觉与主动声学传感的新型穿戴系统，用于密集的手部顶点接触和姿态估计。VibeMesh集成了骨传导扬声器和分布在人体手部的稀疏压电微电话筒，发射结构化声信号并捕获其传播，以推断接触引起的改变。为了解释这些跨模态信号，我们提出了一种基于图的注意力网络，该网络处理同步的音频频谱和RGB-D转换的手部网格，以在高空间分辨率下预测接触。我们贡献了：(i) 一种轻量级、非侵入性的视听传感平台；(ii) 一种跨模态图网络，用于联合姿态和接触推断；(iii) 一个跨越多种操作场景的同步RGB-D、声学和地面真实接触标注的数据集；(iv) 实验结果表明，VibeMesh在准确性和鲁棒性方面优于单一视觉基线，尤其是在被遮挡或静态接触的场景中。 

---
