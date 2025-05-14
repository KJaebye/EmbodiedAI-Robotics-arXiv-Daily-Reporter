# UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations 

**Title (ZH)**: UniSkill: 通过跨躯体技能表示模仿人类视频 

**Authors**: Hanjung Kim, Jaehyun Kang, Hyolim Kang, Meedeum Cho, Seon Joo Kim, Youngwoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.08787)  

**Abstract**: Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: this https URL. 

**Abstract (ZH)**: 模仿是人类基本的学习机制，使个体能够通过观察和模仿专家来学习新任务。然而，将这一能力应用到机器人上由于人类和机器人在视觉外观和物理能力上的本质差异而面临重大挑战。尽管先前的方法使用共享场景和任务的跨体征数据集来弥合这一差距，但大规模收集人类和机器人之间的对齐数据并不容易。在本文中，我们提出了一种名为UniSkill的新型框架，该框架可以从大规模的跨体征视频数据中学习体征无关的技能表示，无需任何标签，使从人类视频提示中提取的技能能够有效转移到仅基于机器人数据训练的机器人策略中。我们的仿真和真实环境实验结果表明，我们的跨体征技能能够成功指导机器人选择合适的行动，即使是对未见过的视频提示。项目网站可访问：this https URL。 

---
# NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance 

**Title (ZH)**: NavDP：学习基于优先信息指导的模拟到现实导航扩散策略 

**Authors**: Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08712)  

**Abstract**: Learning navigation in dynamic open-world environments is an important yet challenging skill for robots. Most previous methods rely on precise localization and mapping or learn from expensive real-world demonstrations. In this paper, we propose the Navigation Diffusion Policy (NavDP), an end-to-end framework trained solely in simulation and can zero-shot transfer to different embodiments in diverse real-world environments. The key ingredient of NavDP's network is the combination of diffusion-based trajectory generation and a critic function for trajectory selection, which are conditioned on only local observation tokens encoded from a shared policy transformer. Given the privileged information of the global environment in simulation, we scale up the demonstrations of good quality to train the diffusion policy and formulate the critic value function targets with contrastive negative samples. Our demonstration generation approach achieves about 2,500 trajectories/GPU per day, 20$\times$ more efficient than real-world data collection, and results in a large-scale navigation dataset with 363.2km trajectories across 1244 scenes. Trained with this simulation dataset, NavDP achieves state-of-the-art performance and consistently outstanding generalization capability on quadruped, wheeled, and humanoid robots in diverse indoor and outdoor environments. In addition, we present a preliminary attempt at using Gaussian Splatting to make in-domain real-to-sim fine-tuning to further bridge the sim-to-real gap. Experiments show that adding such real-to-sim data can improve the success rate by 30\% without hurting its generalization capability. 

**Abstract (ZH)**: 学习动态开放世界环境中的导航是一项重要但具有挑战性的技能，对于机器人而言尤为重要。大多数先前的方法依赖于精确的定位和建图，或者从昂贵的真实世界演示中学习。本文提出了一种名为导航扩散政策（NavDP）的端到端框架，该框架仅在仿真中训练，并能够零样本转移到不同载体在多样化的现实环境中。NavDP网络的关键成分是基于扩散的轨迹生成与用于轨迹选择的评论函数的结合，后者仅基于共享策略变换器编码的局部观察 token。借助仿真中对全局环境的先验信息，我们将高质量的演示扩展以训练扩散策略，并使用反例样本来构建评论函数目标。我们的演示生成方法每天每个GPU可以生成约2500条轨迹，比真实世界的数据收集效率高20倍，并生成了一个大规模的导航数据集，包含1244个场景中的363.2公里轨迹。使用此仿真数据集训练的NavDP在各种室内和室外环境中实现了最先进的性能，并在四足、轮式和类人机器人上表现出一致的出色泛化能力。此外，我们还提出了使用高斯点积进行领域内真实到仿真的微调的初步尝试，以进一步缩小仿真实用性差距。实验结果显示，添加这种真实到仿真的数据可以使成功率达到提高30%而不会损害其泛化能力。 

---
# A Social Robot with Inner Speech for Dietary Guidance 

**Title (ZH)**: 带有内心语言的社会机器人饮食指导 

**Authors**: Valerio Belcamino, Alessandro Carfì, Valeria Seidita, Fulvio Mastrogiovanni, Antonio Chella  

**Link**: [PDF](https://arxiv.org/pdf/2505.08664)  

**Abstract**: We explore the use of inner speech as a mechanism to enhance transparency and trust in social robots for dietary advice. In humans, inner speech structures thought processes and decision-making; in robotics, it improves explainability by making reasoning explicit. This is crucial in healthcare scenarios, where trust in robotic assistants depends on both accurate recommendations and human-like dialogue, which make interactions more natural and engaging. Building on this, we developed a social robot that provides dietary advice, and we provided the architecture with inner speech capabilities to validate user input, refine reasoning, and generate clear justifications. The system integrates large language models for natural language understanding and a knowledge graph for structured dietary information. By making decisions more transparent, our approach strengthens trust and improves human-robot interaction in healthcare. We validated this by measuring the computational efficiency of our architecture and conducting a small user study, which assessed the reliability of inner speech in explaining the robot's behavior. 

**Abstract (ZH)**: 我们探索内心言语作为机制以增强社交机器人在饮食建议中的透明度和信任度。基于此，我们开发了一种提供饮食建议的社交机器人，并为其赋予内心言语能力，以验证用户输入、改进推理并生成清晰的解释。该系统整合了大规模语言模型进行自然语言理解，以及知识图谱进行结构化的饮食信息。通过使决策过程更加透明，我们的方法可以增强信任并改善医疗保健场景中的人机交互。我们通过测量架构的计算效率和进行小型用户研究来验证这一点，该研究评估了内心言语在解释机器人行为方面的可靠性。 

---
# Augmented Reality for RObots (ARRO): Pointing Visuomotor Policies Towards Visual Robustness 

**Title (ZH)**: 增强现实用于机器人（ARRO）：指向视觉稳健性的指针知觉运动策略 

**Authors**: Reihaneh Mirjalili, Tobias Jülg, Florian Walter, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2505.08627)  

**Abstract**: Visuomotor policies trained on human expert demonstrations have recently shown strong performance across a wide range of robotic manipulation tasks. However, these policies remain highly sensitive to domain shifts stemming from background or robot embodiment changes, which limits their generalization capabilities. In this paper, we present ARRO, a novel calibration-free visual representation that leverages zero-shot open-vocabulary segmentation and object detection models to efficiently mask out task-irrelevant regions of the scene without requiring additional training. By filtering visual distractors and overlaying virtual guides during both training and inference, ARRO improves robustness to scene variations and reduces the need for additional data collection. We extensively evaluate ARRO with Diffusion Policy on several tabletop manipulation tasks in both simulation and real-world environments, and further demonstrate its compatibility and effectiveness with generalist robot policies, such as Octo and OpenVLA. Across all settings in our evaluation, ARRO yields consistent performance gains, allows for selective masking to choose between different objects, and shows robustness even to challenging segmentation conditions. Videos showcasing our results are available at: this http URL 

**Abstract (ZH)**: 基于人类专家演示训练的知觉运动策略在多种机器人操作任务中表现出强大性能。然而，这些策略仍然高度敏感于背景或机器人身体特征变化引起的领域转换，这限制了它们的泛化能力。本文中，我们提出ARRO，这是一种新型无需校准的视觉表示，它利用零样本开放式词汇分割和对象检测模型来高效地屏蔽场景中的任务无关区域，而无需额外训练。通过在训练和推理过程中过滤视觉干扰并叠加虚拟引导，ARRO 提高了对场景变化的鲁棒性，并减少了额外数据收集的需求。我们使用 Diffusion Policy 在多种桌面操作任务的模拟和真实环境中进行了广泛评估，并进一步展示了其与通用机器人策略（如 Octo 和 OpenVLA）的兼容性和有效性。在我们评估的所有设置中，ARRO 均实现了一致的性能提升，允许选择性屏蔽以选择不同的对象，并且即使在具有挑战性的分割条件下也表现出鲁棒性。有关我们的结果的视频可在以下链接查看：this http URL。 

---
# Beyond Predefined Actions: Integrating Behavior Trees and Dynamic Movement Primitives for Robot Learning from Demonstration 

**Title (ZH)**: 超越预定义动作：将行为树与动态运动 primitives 结合用于机器人演示学习 

**Authors**: David Cáceres Domínguez, Erik Schaffernicht, Todor Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08625)  

**Abstract**: Interpretable policy representations like Behavior Trees (BTs) and Dynamic Motion Primitives (DMPs) enable robot skill transfer from human demonstrations, but each faces limitations: BTs require expert-crafted low-level actions, while DMPs lack high-level task logic. We address these limitations by integrating DMP controllers into a BT framework, jointly learning the BT structure and DMP actions from single demonstrations, thereby removing the need for predefined actions. Additionally, by combining BT decision logic with DMP motion generation, our method enhances policy interpretability, modularity, and adaptability for autonomous systems. Our approach readily affords both learning to replicate low-level motions and combining partial demonstrations into a coherent and easy-to-modify overall policy. 

**Abstract (ZH)**: 可解释的策略表示方法，如行为树（BTs）和动态运动基元（DMPs），能够通过人类示范将机器人技能进行迁移，但每种方法都面临局限：BTs需要专家设计的低级动作，而DMPs缺乏高级任务逻辑。我们通过将DMP控制器整合到BT框架中，从单个示范中联合学习BT结构和DMP动作，从而消除了预定义动作的需要。此外，结合BT决策逻辑和DMP运动生成，我们的方法增强了策略的可解释性、模块化和适应性，以提升自主系统的性能。我们的方法既可以直接学习模仿低级运动，也可以将不完整的示范合并为一个连贯且易于修改的整体策略。 

---
# End-to-End Multi-Task Policy Learning from NMPC for Quadruped Locomotion 

**Title (ZH)**: 基于NMPC的 quadruped 行走多任务端到端策略学习 

**Authors**: Anudeep Sajja, Shahram Khorshidi, Sebastian Houben, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.08574)  

**Abstract**: Quadruped robots excel in traversing complex, unstructured environments where wheeled robots often fail. However, enabling efficient and adaptable locomotion remains challenging due to the quadrupeds' nonlinear dynamics, high degrees of freedom, and the computational demands of real-time control. Optimization-based controllers, such as Nonlinear Model Predictive Control (NMPC), have shown strong performance, but their reliance on accurate state estimation and high computational overhead makes deployment in real-world settings challenging. In this work, we present a Multi-Task Learning (MTL) framework in which expert NMPC demonstrations are used to train a single neural network to predict actions for multiple locomotion behaviors directly from raw proprioceptive sensor inputs. We evaluate our approach extensively on the quadruped robot Go1, both in simulation and on real hardware, demonstrating that it accurately reproduces expert behavior, allows smooth gait switching, and simplifies the control pipeline for real-time deployment. Our MTL architecture enables learning diverse gaits within a unified policy, achieving high $R^{2}$ scores for predicted joint targets across all tasks. 

**Abstract (ZH)**: 四足机器人在复杂、未结构化环境中表现出色，而轮式机器人在这些环境中往往表现不佳。然而，由于四足机器人的非线性动力学、高自由度以及实时控制所需的高计算需求，实现高效和适应性的运动仍然具有挑战性。基于优化的控制器，如非线性模型预测控制（NMPC），显示了强大的性能，但其依赖于准确的状态估计和高计算开销使得其实现现场应用具有挑战性。在本文中，我们提出了一种多任务学习（MTL）框架，其中专家NMPC演示用于训练单个神经网络，直接从原始本体感觉传感器输入中预测多种运动行为的行动。我们广泛评估了我们的方法在四足机器人Go1上的性能，包括仿真和实物硬件上，证明了它能够准确重现专家行为、实现平滑的步伐切换，并简化了实时部署的控制流程。我们的MTL架构能够在统一策略中学习多样化的步伐，对所有任务预测的关节目标获得较高的$R^{2}$分数。 

---
# From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation 

**Title (ZH)**: 从感知到执行：实现机器人操作中的推理与决策融合 

**Authors**: Yifu Yuan, Haiqin Cui, Yibin Chen, Zibin Dong, Fei Ni, Longxin Kou, Jinyi Liu, Pengyi Li, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08548)  

**Abstract**: Achieving generalization in robotic manipulation remains a critical challenge, particularly for unseen scenarios and novel tasks. Current Vision-Language-Action (VLA) models, while building on top of general Vision-Language Models (VLMs), still fall short of achieving robust zero-shot performance due to the scarcity and heterogeneity prevalent in embodied datasets. To address these limitations, we propose FSD (From Seeing to Doing), a novel vision-language model that generates intermediate representations through spatial relationship reasoning, providing fine-grained guidance for robotic manipulation. Our approach combines a hierarchical data pipeline for training with a self-consistency mechanism that aligns spatial coordinates with visual signals. Through extensive experiments, we comprehensively validated FSD's capabilities in both "seeing" and "doing," achieving outstanding performance across 8 benchmarks for general spatial reasoning and embodied reference abilities, as well as on our proposed more challenging benchmark VABench. We also verified zero-shot capabilities in robot manipulation, demonstrating significant performance improvements over baseline methods in both SimplerEnv and real robot settings. Experimental results show that FSD achieves 54.1% success rate in SimplerEnv and 72% success rate across 8 real-world tasks, outperforming the strongest baseline by 30%. 

**Abstract (ZH)**: 在机器人操作中实现泛化仍然是一个关键挑战，特别是在未见过的场景和新型任务中。当前的视觉-语言-动作（VLA）模型虽然基于通用的视觉-语言模型（VLM），但仍因体态数据中存在的稀缺性和异质性而难以实现稳健的零样本性能。为了解决这些局限性，我们提出FSD（From Seeing to Doing），一种新颖的视觉-语言模型，通过空间关系推理生成中间表示，为机器人操作提供精细指导。我们的方法结合了分层数据管道进行训练，并引入了一种自我一致性机制，使空间坐标与视觉信号对齐。通过大量的实验，我们全面验证了FSD在“看”和“做”方面的能力，在8个基准测试中取得了出色的泛空间推理能力和体态参考能力表现，并在我们提出的更具挑战性的基准测试VABench中表现优异。我们还验证了FSD在机器人操作中的零样本能力，在SimplerEnv和真实机器人设置中均表现出显著性能提升。实验结果表明，FSD在SimplerEnv中的成功率为54.1%，在8个真实世界任务中的成功率为72%，比最强基线方法高出30%。 

---
# FOCI: Trajectory Optimization on Gaussian Splats 

**Title (ZH)**: FOCI：高斯体素上的轨迹优化 

**Authors**: Mario Gomez Andreu, Maximum Wilder-Smith, Victor Klemm, Vaishakh Patil, Jesus Tordesillas, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.08510)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at this https URL 

**Abstract (ZH)**: 3D高斯点积（3DGS）在3D重建和视角合成方法中近年来因其比神经辐射场（NeRF）更快而受到青睐。利用3DGS中编码的时空信息，本工作提出了一种直接在高斯点自身上优化轨迹的FOCI（场重叠碰撞积分）算法。FOCI利用重叠积分的新颖且可解释的碰撞形式化来利用3DGS的特性。与仅用保守的边界框低估环境可通行性的其他方法不同，我们提出将环境和机器人表示为高斯点积。这不仅具有良好的计算属性，还允许姿态感知的规划，使机器人能够通过非常狭小的空间。我们在合成和真实高斯点积上进行了广泛的测试，展示了即使环境由数十万高斯点组成，也可以在几秒内计算出无碰撞轨迹，这些轨迹适用于ANYmal腿式机器人。项目页面和代码可在以下链接访问：这个 https URL。 

---
# Zero-Shot Sim-to-Real Reinforcement Learning for Fruit Harvesting 

**Title (ZH)**: 零样本模拟到现实的强化学习在水果采摘中的应用 

**Authors**: Emlyn Williams, Athanasios Polydoros  

**Link**: [PDF](https://arxiv.org/pdf/2505.08458)  

**Abstract**: This paper presents a comprehensive sim-to-real pipeline for autonomous strawberry picking from dense clusters using a Franka Panda robot. Our approach leverages a custom Mujoco simulation environment that integrates domain randomization techniques. In this environment, a deep reinforcement learning agent is trained using the dormant ratio minimization algorithm. The proposed pipeline bridges low-level control with high-level perception and decision making, demonstrating promising performance in both simulation and in a real laboratory environment, laying the groundwork for successful transfer to real-world autonomous fruit harvesting. 

**Abstract (ZH)**: 本研究表明了一种用于利用Franka Panda机器人从密集簇中自主采摘草莓的全面仿真实验转换流水线。该方法借助一个包含领域随机化技术的定制Mujoco仿真环境，并使用休眠比最小化算法训练深度强化学习代理。所提出的流水线将低级控制与高级感知和决策相结合，在仿真和实际实验室环境中均表现出色，为成功转换到实际自主水果 harvesting 奠定了基础。 

---
# Parameter Estimation using Reinforcement Learning Causal Curiosity: Limits and Challenges 

**Title (ZH)**: 使用强化学习因果好奇心进行参数估计：限制与挑战 

**Authors**: Miguel Arana-Catania, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08453)  

**Abstract**: Causal understanding is important in many disciplines of science and engineering, where we seek to understand how different factors in the system causally affect an experiment or situation and pave a pathway towards creating effective or optimising existing models. Examples of use cases are autonomous exploration and modelling of unknown environments or assessing key variables in optimising large complex systems. In this paper, we analyse a Reinforcement Learning approach called Causal Curiosity, which aims to estimate as accurately and efficiently as possible, without directly measuring them, the value of factors that causally determine the dynamics of a system. Whilst the idea presents a pathway forward, measurement accuracy is the foundation of methodology effectiveness. Focusing on the current causal curiosity's robotic manipulator, we present for the first time a measurement accuracy analysis of the future potentials and current limitations of this technique and an analysis of its sensitivity and confounding factor disentanglement capability - crucial for causal analysis. As a result of our work, we promote proposals for an improved and efficient design of Causal Curiosity methods to be applied to real-world complex scenarios. 

**Abstract (ZH)**: 因果理解在科学和工程的许多学科中非常重要，我们旨在理解系统中不同因素如何因果影响实验或情况，并为此铺平创造有效或优化现有模型的途径。应用案例包括自主探索和建模未知环境或评估优化大型复杂系统的关键变量。在本文中，我们分析了一种名为因果好奇心的强化学习方法，该方法旨在尽可能准确且高效地估计那些因果决定系统动力学的因素价值，而无需直接测量这些因素。尽管这一想法为未来发展指出了一条路径，但测量准确性是该方法有效性的基础。本文首次专注于当前因果好奇心的机器人 manipulator，对其未来潜力和当前局限性进行测量精度分析，并对其因果分析中的因果因素分离能力进行分析。由此，我们提出改进和高效设计因果好奇心方法的建议，以应用于实际复杂场景。 

---
# HMR-ODTA: Online Diverse Task Allocation for a Team of Heterogeneous Mobile Robots 

**Title (ZH)**: HMR-ODTA:在线多样化任务分配给异质移动机器人团队 

**Authors**: Ashish Verma, Avinash Gautam, Tanishq Duhan, V. S. Shekhawat, Sudeept Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08419)  

**Abstract**: Coordinating time-sensitive deliveries in environments like hospitals poses a complex challenge, particularly when managing multiple online pickup and delivery requests within strict time windows using a team of heterogeneous robots. Traditional approaches fail to address dynamic rescheduling or diverse service requirements, typically restricting robots to single-task types. This paper tackles the Multi-Pickup and Delivery Problem with Time Windows (MPDPTW), where autonomous mobile robots are capable of handling varied service requests. The objective is to minimize late delivery penalties while maximizing task completion rates. To achieve this, we propose a novel framework leveraging a heterogeneous robot team and an efficient dynamic scheduling algorithm that supports dynamic task rescheduling. Users submit requests with specific time constraints, and our decentralized algorithm, Heterogeneous Mobile Robots Online Diverse Task Allocation (HMR-ODTA), optimizes task assignments to ensure timely service while addressing delays or task rejections. Extensive simulations validate the algorithm's effectiveness. For smaller task sets (40-160 tasks), penalties were reduced by nearly 63%, while for larger sets (160-280 tasks), penalties decreased by approximately 50%. These results highlight the algorithm's effectiveness in improving task scheduling and coordination in multi-robot systems, offering a robust solution for enhancing delivery performance in structured, time-critical environments. 

**Abstract (ZH)**: 多时间窗口条件下异构机器人团队的多取多送问题研究 

---
# ORACLE-Grasp: Zero-Shot Task-Oriented Robotic Grasping using Large Multimodal Models 

**Title (ZH)**: ORACLE-抓取：基于大型多模态模型的零样本任务导向机器人抓取 

**Authors**: Avihai Giuili, Rotem Atari, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08417)  

**Abstract**: Grasping unknown objects in unstructured environments remains a fundamental challenge in robotics, requiring both semantic understanding and spatial reasoning. Existing methods often rely on dense training datasets or explicit geometric modeling, limiting their scalability to real-world tasks. Recent advances in Large Multimodal Models (LMMs) offer new possibilities for integrating vision and language understanding, but their application to autonomous robotic grasping remains largely unexplored. We present ORACLE-Grasp, a zero-shot framework that leverages LMMs as semantic oracles to guide grasp selection without requiring additional training or human input. The system formulates grasp prediction as a structured, iterative decision process, using dual-prompt tool calling to first extract high-level object context and then select task-relevant grasp regions. By discretizing the image space and reasoning over candidate areas, ORACLE-Grasp mitigates the spatial imprecision common in LMMs and produces human-like, task-driven grasp suggestions. Early stopping and depth-based refinement steps further enhance efficiency and physical grasp reliability. Experiments demonstrate that the predicted grasps achieve low positional and orientation errors relative to human-annotated ground truth and lead to high success rates in real-world pick up tasks. These results highlight the potential of combining language-driven reasoning with lightweight vision techniques to enable robust, autonomous grasping without task-specific datasets or retraining. 

**Abstract (ZH)**: 未知物体在无结构环境中的抓取仍然是机器人技术中的一个基本挑战，需要同时具备语义理解和空间推理能力。现有的方法往往依赖于密集的训练数据集或显式的几何建模，限制了其在实际任务中的可扩展性。近期在大规模多模态模型（LMMs）方面的进展为融合视觉和语言理解提供了新的可能性，但其在自主机器人抓取中的应用仍然很大程度上尚未被探索。我们提出ORACLE-抓取框架，利用LMMs作为语义或acles来指导抓取选择，无需额外训练或人工输入。该系统将抓取预测表述为结构化的迭代决策过程，通过双提示工具调用来首先提取高层次的对象上下文，然后选择与任务相关的抓取区域。通过离散化图像空间并在候选区域内进行推理，ORACLE-抓取减少了LMMs中常见的空间精度问题，并生成了接近人类、任务驱动的抓取建议。通过早期停止和基于深度的精修步骤，进一步提高了效率和物理抓取的可靠性。实验结果表明，预测的抓取相对于由人工标注的地面真相在位置和姿态误差上较低，并在实际的任务拾取中实现了高成功率。这些结果突显了结合基于语言的推理与轻量级视觉技术以实现匹配任务数据集或重新训练的鲁棒自主抓取的潜力。 

---
# Adaptive Diffusion Policy Optimization for Robotic Manipulation 

**Title (ZH)**: 适应性扩散策略优化在机器人 manipulation 中的应用 

**Authors**: Huiyun Jiang, Zhuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08376)  

**Abstract**: Recent studies have shown the great potential of diffusion models in improving reinforcement learning (RL) by modeling complex policies, expressing a high degree of multi-modality, and efficiently handling high-dimensional continuous control tasks. However, there is currently limited research on how to optimize diffusion-based polices (e.g., Diffusion Policy) fast and stably. In this paper, we propose an Adam-based Diffusion Policy Optimization (ADPO), a fast algorithmic framework containing best practices for fine-tuning diffusion-based polices in robotic control tasks using the adaptive gradient descent method in RL. Adaptive gradient method is less studied in training RL, let alone diffusion-based policies. We confirm that ADPO outperforms other diffusion-based RL methods in terms of overall effectiveness for fine-tuning on standard robotic tasks. Concretely, we conduct extensive experiments on standard robotic control tasks to test ADPO, where, particularly, six popular diffusion-based RL methods are provided as benchmark methods. Experimental results show that ADPO acquires better or comparable performance than the baseline methods. Finally, we systematically analyze the sensitivity of multiple hyperparameters in standard robotics tasks, providing guidance for subsequent practical applications. Our video demonstrations are released in this https URL. 

**Abstract (ZH)**: 基于Adam的扩散政策优化（ADPO）：机器人控制任务中快速稳定的扩散政策优化方法 

---
# MA-ROESL: Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos 

**Title (ZH)**: MA-ROESL：基于运动感知的快速奖励优化方法以实现从单个视频中高效学习机器人技能 

**Authors**: Xianghui Wang, Xinming Zhang, Yanjun Chen, Xiaoyu Shen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08367)  

**Abstract**: Vision-language models (VLMs) have demonstrated excellent high-level planning capabilities, enabling locomotion skill learning from video demonstrations without the need for meticulous human-level reward design. However, the improper frame sampling method and low training efficiency of current methods remain a critical bottleneck, resulting in substantial computational overhead and time costs. To address this limitation, we propose Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos (MA-ROESL). MA-ROESL integrates a motion-aware frame selection method to implicitly enhance the quality of VLM-generated reward functions. It further employs a hybrid three-phase training pipeline that improves training efficiency via rapid reward optimization and derives the final policy through online fine-tuning. Experimental results demonstrate that MA-ROESL significantly enhances training efficiency while faithfully reproducing locomotion skills in both simulated and real-world settings, thereby underscoring its potential as a robust and scalable framework for efficient robot locomotion skill learning from video demonstrations. 

**Abstract (ZH)**: 基于运动感知的快速奖励优化高效单视频机器人技能学习（MA-ROESL） 

---
# Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning 

**Title (ZH)**: 自动驾驶场景下的自适应课程学习：迈向稳健高效的强化学习 

**Authors**: Ahmed Abouelazm, Tim Weinstein, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.08264)  

**Abstract**: This paper addresses the challenges of training end-to-end autonomous driving agents using Reinforcement Learning (RL). RL agents are typically trained in a fixed set of scenarios and nominal behavior of surrounding road users in simulations, limiting their generalization and real-life deployment. While domain randomization offers a potential solution by randomly sampling driving scenarios, it frequently results in inefficient training and sub-optimal policies due to the high variance among training scenarios. To address these limitations, we propose an automatic curriculum learning framework that dynamically generates driving scenarios with adaptive complexity based on the agent's evolving capabilities. Unlike manually designed curricula that introduce expert bias and lack scalability, our framework incorporates a ``teacher'' that automatically generates and mutates driving scenarios based on their learning potential -- an agent-centric metric derived from the agent's current policy -- eliminating the need for expert design. The framework enhances training efficiency by excluding scenarios the agent has mastered or finds too challenging. We evaluate our framework in a reinforcement learning setting where the agent learns a driving policy from camera images. Comparative results against baseline methods, including fixed scenario training and domain randomization, demonstrate that our approach leads to enhanced generalization, achieving higher success rates: +9\% in low traffic density, +21\% in high traffic density, and faster convergence with fewer training steps. Our findings highlight the potential of ACL in improving the robustness and efficiency of RL-based autonomous driving agents. 

**Abstract (ZH)**: 基于自动课程学习的端到端自主驾驶代理强化学习训练方法 

---
# Training Strategies for Efficient Embodied Reasoning 

**Title (ZH)**: 高效体域推理的训练策略 

**Authors**: William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2505.08243)  

**Abstract**: Robot chain-of-thought reasoning (CoT) -- wherein a model predicts helpful intermediate representations before choosing actions -- provides an effective method for improving the generalization and performance of robot policies, especially vision-language-action models (VLAs). While such approaches have been shown to improve performance and generalization, they suffer from core limitations, like needing specialized robot reasoning data and slow inference speeds. To design new robot reasoning approaches that address these issues, a more complete characterization of why reasoning helps policy performance is critical. We hypothesize several mechanisms by which robot reasoning improves policies -- (1) better representation learning, (2) improved learning curricularization, and (3) increased expressivity -- then devise simple variants of robot CoT reasoning to isolate and test each one. We find that learning to generate reasonings does lead to better VLA representations, while attending to the reasonings aids in actually leveraging these features for improved action prediction. Our results provide us with a better understanding of why CoT reasoning helps VLAs, which we use to introduce two simple and lightweight alternative recipes for robot reasoning. Our proposed approaches achieve significant performance gains over non-reasoning policies, state-of-the-art results on the LIBERO-90 benchmark, and a 3x inference speedup compared to standard robot reasoning. 

**Abstract (ZH)**: 基于推理的机器人链式思考（CoT）推理：一种通过预测有助于执行的中间表示来提升机器人策略泛化能力和性能的方法，特别适用于视觉-语言-行动模型（VLAs）。 

---
# Motion Control of High-Dimensional Musculoskeletal Systems with Hierarchical Model-Based Planning 

**Title (ZH)**: 基于分级模型规划的高维肌肉骨骼系统运动控制 

**Authors**: Yunyue Wei, Shanning Zhuang, Vincent Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.08238)  

**Abstract**: Controlling high-dimensional nonlinear systems, such as those found in biological and robotic applications, is challenging due to large state and action spaces. While deep reinforcement learning has achieved a number of successes in these domains, it is computationally intensive and time consuming, and therefore not suitable for solving large collections of tasks that require significant manual tuning. In this work, we introduce Model Predictive Control with Morphology-aware Proportional Control (MPC^2), a hierarchical model-based learning algorithm for zero-shot and near-real-time control of high-dimensional complex dynamical systems. MPC^2 uses a sampling-based model predictive controller for target posture planning, and enables robust control for high-dimensional tasks by incorporating a morphology-aware proportional controller for actuator coordination. The algorithm enables motion control of a high-dimensional human musculoskeletal model in a variety of motion tasks, such as standing, walking on different terrains, and imitating sports activities. The reward function of MPC^2 can be tuned via black-box optimization, drastically reducing the need for human-intensive reward engineering. 

**Abstract (ZH)**: 基于形态意识比例控制的模型预测控制（MPC²）：用于高维复杂动态系统的零样本和近实时控制 

---
# Reinforcement Learning-based Fault-Tolerant Control for Quadrotor with Online Transformer Adaptation 

**Title (ZH)**: 基于强化学习的四旋翼无人机在线变压器自适应容错控制 

**Authors**: Dohyun Kim, Jayden Dongwoo Lee, Hyochoong Bang, Jungho Bae  

**Link**: [PDF](https://arxiv.org/pdf/2505.08223)  

**Abstract**: Multirotors play a significant role in diverse field robotics applications but remain highly susceptible to actuator failures, leading to rapid instability and compromised mission reliability. While various fault-tolerant control (FTC) strategies using reinforcement learning (RL) have been widely explored, most previous approaches require prior knowledge of the multirotor model or struggle to adapt to new configurations. To address these limitations, we propose a novel hybrid RL-based FTC framework integrated with a transformer-based online adaptation module. Our framework leverages a transformer architecture to infer latent representations in real time, enabling adaptation to previously unseen system models without retraining. We evaluate our method in a PyBullet simulation under loss-of-effectiveness actuator faults, achieving a 95% success rate and a positional root mean square error (RMSE) of 0.129 m, outperforming existing adaptation methods with 86% success and an RMSE of 0.153 m. Further evaluations on quadrotors with varying configurations confirm the robustness of our framework across untrained dynamics. These results demonstrate the potential of our framework to enhance the adaptability and reliability of multirotors, enabling efficient fault management in dynamic and uncertain environments. Website is available at this http URL 

**Abstract (ZH)**: 多旋翼飞行器在多样化领域的机器人应用中扮演着重要角色，但仍高度易受执行器故障的影响，导致快速失稳和任务可靠性的下降。尽管已有利用强化学习（RL）的容错控制（FTC）策略得到了广泛探索，但大多数前期方法需要多旋翼飞行器模型的先验知识，或者难以适应新的布局配置。为克服这些限制，我们提出了一种结合基于变换器的在线自适应模块的新型混合RL基于FTC框架。该框架利用变换器架构在实时推断潜在表示，从而能够在无需重新训练的情况下适应未见过的系统模型。我们在PyBullet模拟环境中评估了该方法，在性能降低的执行器故障条件下实现了95%的成功率和位置均方根误差（RMSE）为0.129 m，优于现有自适应方法的86%成功率和0.153 m的RMSE。进一步在不同配置的四旋翼飞行器上的评估证实了该框架在未训练动态条件下的鲁棒性。这些结果展示了该框架增强多旋翼飞行器的适应性和可靠性的潜力，使其能够在动态和不确定环境中高效管理故障。Website is available at this http URL 

---
# What Matters for Batch Online Reinforcement Learning in Robotics? 

**Title (ZH)**: 机器人领域批量在线强化学习中什么是重要的？ 

**Authors**: Perry Dong, Suvir Mirchandani, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2505.08078)  

**Abstract**: The ability to learn from large batches of autonomously collected data for policy improvement -- a paradigm we refer to as batch online reinforcement learning -- holds the promise of enabling truly scalable robot learning by significantly reducing the need for human effort of data collection while getting benefits from self-improvement. Yet, despite the promise of this paradigm, it remains challenging to achieve due to algorithms not being able to learn effectively from the autonomous data. For example, prior works have applied imitation learning and filtered imitation learning methods to the batch online RL problem, but these algorithms often fail to efficiently improve from the autonomously collected data or converge quickly to a suboptimal point. This raises the question of what matters for effective batch online RL in robotics. Motivated by this question, we perform a systematic empirical study of three axes -- (i) algorithm class, (ii) policy extraction methods, and (iii) policy expressivity -- and analyze how these axes affect performance and scaling with the amount of autonomous data. Through our analysis, we make several observations. First, we observe that the use of Q-functions to guide batch online RL significantly improves performance over imitation-based methods. Building on this, we show that an implicit method of policy extraction -- via choosing the best action in the distribution of the policy -- is necessary over traditional policy extraction methods from offline RL. Next, we show that an expressive policy class is preferred over less expressive policy classes. Based on this analysis, we propose a general recipe for effective batch online RL. We then show a simple addition to the recipe of using temporally-correlated noise to obtain more diversity results in further performance gains. Our recipe obtains significantly better performance and scaling compared to prior methods. 

**Abstract (ZH)**: 从自主收集的大批次数据中进行策略改进的学习能力——一种我们称之为批在线强化学习的范式——有望通过显著减少数据收集的人力需求来实现真正 scalable 的机器人学习，同时还能获得自我改进的好处。然而，尽管这一范式具有巨大的前景，但由于算法难以有效学习自主数据，实现起来仍然具有挑战性。例如，先前的工作将模仿学习和过滤模仿学习方法应用于批在线 RL 问题，但这些算法往往无法有效地从自主收集的数据中提高自身，或迅速收敛到一个次优点。这提出了什么对于有效的批在线 RL 在机器人领域的研究。在这一问题的驱动下，我们系统地研究了三个维度——（i）算法类别，（ii）策略提取方法，以及（iii）策略表达性——并分析了这些维度如何影响性能和随自主数据量增加的可扩展性。通过我们的分析，我们做出了几项观察。首先，我们观察到使用 Q 函数来指导批在线 RL 显著提高了性能，超过了基于模仿的方法。在这一点基础上，我们展示了策略提取的隐式方法——通过选择策略分布中的最佳动作——比传统的离线 RL 中的策略提取方法更为必要。其次，我们展示了具有丰富表达能力的策略类别优于表达能力较弱的类别。基于这一分析，我们提出了一个有效的批在线 RL 的通用配方。然后，我们展示了一种简单的附加方法——使用时序相关噪声以获得更多的多样性，进一步提高了性能。我们的配方相对于先前的方法在性能和可扩展性方面取得了显著的改进。 

---
# Land-Coverage Aware Path-Planning for Multi-UAV Swarms in Search and Rescue Scenarios 

**Title (ZH)**: 基于土地覆盖的多无人机群搜救路径规划 

**Authors**: Pedro Antonio Alarcon Granadeno, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08060)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have become vital in search-and-rescue (SAR) missions, with autonomous mission planning improving response times and coverage efficiency. Early approaches primarily used path planning techniques such as A*, potential-fields, or Dijkstra's algorithm, while recent approaches have incorporated meta-heuristic frameworks like genetic algorithms and particle swarm optimization to balance competing objectives such as network connectivity, energy efficiency, and strategic placement of charging stations. However, terrain-aware path planning remains under-explored, despite its critical role in optimizing UAV SAR deployments. To address this gap, we present a computer-vision based terrain-aware mission planner that autonomously extracts and analyzes terrain topology to enhance SAR pre-flight planning. Our framework uses a deep segmentation network fine-tuned on our own collection of landcover datasets to transform satellite imagery into a structured, grid-based representation of the operational area. This classification enables terrain-specific UAV-task allocation, improving deployment strategies in complex environments. We address the challenge of irregular terrain partitions, by introducing a two-stage partitioning scheme that first evaluates terrain monotonicity along coordinate axes before applying a cost-based recursive partitioning process, minimizing unnecessary splits and optimizing path efficiency. Empirical validation in a high-fidelity simulation environment demonstrates that our approach improves search and dispatch time over multiple meta-heuristic techniques and against a competing state-of-the-art method. These results highlight its potential for large-scale SAR operations, where rapid response and efficient UAV coordination are critical. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）在搜索与救援（SAR）任务中变得至关重要，自主任务规划提高了响应时间和覆盖效率。早期方法主要采用路径规划技术如A*、势场法或迪杰斯特拉算法，而最近的方法则结合了遗传算法和粒子群优化等元启发式框架，以平衡网络连接、能量效率和充电站战略性位置等竞争性目标。然而，地形感知路径规划仍然鲜有研究，尽管其在优化UAV SAR部署中的作用至关重要。为填补这一空白，我们提出了一种基于计算机视觉的地形感知任务规划器，该规划器能够自主提取和分析地形拓扑，以增强SAR预飞行规划。我们的框架利用了一个在我们自己的土地覆盖数据集上微调的深度分割网络，将卫星影像转换为运营区域的结构化网格表示。这种分类使得可以根据地形进行无人机任务分配，从而在复杂环境中提高部署策略。为了应对不规则地形分区的挑战，我们引入了一种两阶段分区方案，首先沿坐标轴评估地形单调性，然后应用基于成本的递归分区过程，从而最小化不必要的分割并优化路径效率。在高保真模拟环境中的实证验证表明，我们的方法在多种元启发式技术以及与一种竞争性的最新方法相比，在搜索和调度时间方面均有改进。这些结果突显了其在大规模SAR操作中的潜力，尤其是在快速响应和高效无人机协调方面。 

---
# Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM 

**Title (ZH)**: 通过轻量级局部LLM实现基于神经符号规划的可扩展机器人自主性 

**Authors**: Nicholas Attolino, Alessio Capitanelli, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08492)  

**Abstract**: PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline. 

**Abstract (ZH)**: 基于PDDL的符号任务规划在机器人自主性中仍然至关重要，但由于扩展性、重规划需求以及计划延迟可用性等问题，在动态人机协作中面临挑战。尽管一些神经符号框架曾利用如GPT-3等大规模语言模型来解决这些问题，但依赖闭源、远程模型带来的限制包括第三方依赖性、响应时间不一致、计划长度和复杂性受限以及多领域扩展性问题。我们提出了Gideon，一个新型框架，使其能够过渡到现代的小规模本地语言模型，并扩展上下文长度。Gideon 集成了一个新型问题生成器，可以系统地为任何领域生成大规模的现实世界领域-问题-计划三元组数据集，并针对本地语言模型适应神经符号规划，从而实现设备端执行和扩展上下文以支持多领域。单领域场景下使用Qwen-2.5 1.5B，在8k-32k样本训练下，初步实验显示计划有效性比例为66.1%（32k模型），并通过增加数据可以进一步扩大比例。多领域测试下使用16k样本，计划有效性比例高达70.6%，表明其在跨领域的应用具有扩展性，并暗示数据多样性可以提升学习效率。尽管面向长远规划和较小模型规模使得Gideon的训练效率远低于基于大规模模型的基线模型，但在训练模型规模仅为基线的1/120的情况下，Gideon在推理效率、扩展性和多领域适应性方面仍能获得显著优势，这些都是人机协作中的关键因素。Gideon简化的数据生成流程可以帮助缓解训练效率问题。 

---
# SLAG: Scalable Language-Augmented Gaussian Splatting 

**Title (ZH)**: SLAG：可扩展的语言增强高斯绘制技术 

**Authors**: Laszlo Szilagyi, Francis Engelmann, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2505.08124)  

**Abstract**: Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: this https URL. 

**Abstract (ZH)**: 语言增强的场景表示在大型机器人应用中的巨大潜力，如搜救、智慧城市和采矿等领域。许多这些场景是时间敏感的，要求快速的场景编码，同时也数据密集，需要可扩展的解决方案。在计算资源有限的机器人上部署这些表示进一步增加了挑战。为了解决这个问题，我们引入了SLAG，一种基于多GPU的语言增强高斯点积框架，可以提高大型场景嵌入的速度和可扩展性。我们的方法使用SAM和CLIP将2D视觉-语言模型特征集成到3D场景中。与先前的方法不同，SLAG不需要损失函数来计算每个高斯的语言嵌入，而是通过规范化加权平均从3D高斯场景参数中衍生嵌入，从而实现高效的并行化场景编码。此外，我们还引入了一个向量数据库以实现高效的嵌入存储和检索。我们的实验表明，在16-GPU设置下，SLAG在嵌入计算方面比OpenGaussian快18倍，同时在ScanNet和LERF数据集上保持了嵌入质量。欲了解更多信息，请访问我们的项目网站：this https URL。 

---
# Agent-as-a-Service based on Agent Network 

**Title (ZH)**: 基于代理网络的代理即服务 

**Authors**: Yuhan Zhu, Haojie Liu, Jian Wang, Bing Li, Zikang Yin, Yefei Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08446)  

**Abstract**: The rise of large model-based AI agents has spurred interest in Multi-Agent Systems (MAS) for their capabilities in decision-making, collaboration, and adaptability. While the Model Context Protocol (MCP) addresses tool invocation and data exchange challenges via a unified protocol, it lacks support for organizing agent-level collaboration. To bridge this gap, we propose Agent-as-a-Service based on Agent Network (AaaS-AN), a service-oriented paradigm grounded in the Role-Goal-Process-Service (RGPS) standard. AaaS-AN unifies the entire agent lifecycle, including construction, integration, interoperability, and networked collaboration, through two core components: (1) a dynamic Agent Network, which models agents and agent groups as vertexes that self-organize within the network based on task and role dependencies; (2) service-oriented agents, incorporating service discovery, registration, and interoperability protocols. These are orchestrated by a Service Scheduler, which leverages an Execution Graph to enable distributed coordination, context tracking, and runtime task management. We validate AaaS-AN on mathematical reasoning and application-level code generation tasks, which outperforms state-of-the-art baselines. Notably, we constructed a MAS based on AaaS-AN containing agent groups, Robotic Process Automation (RPA) workflows, and MCP servers over 100 agent services. We also release a dataset containing 10,000 long-horizon multi-agent workflows to facilitate future research on long-chain collaboration in MAS. 

**Abstract (ZH)**: 基于代理网络的代理即服务多智能体系统 

---
# Explaining Autonomous Vehicles with Intention-aware Policy Graphs 

**Title (ZH)**: 基于意图意识的策略图解释自动驾驶车辆 

**Authors**: Sara Montese, Victor Gimenez-Abalos, Atia Cortés, Ulises Cortés, Sergio Alvarez-Napagao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08404)  

**Abstract**: The potential to improve road safety, reduce human driving error, and promote environmental sustainability have enabled the field of autonomous driving to progress rapidly over recent decades. The performance of autonomous vehicles has significantly improved thanks to advancements in Artificial Intelligence, particularly Deep Learning. Nevertheless, the opacity of their decision-making, rooted in the use of accurate yet complex AI models, has created barriers to their societal trust and regulatory acceptance, raising the need for explainability. We propose a post-hoc, model-agnostic solution to provide teleological explanations for the behaviour of an autonomous vehicle in urban environments. Building on Intention-aware Policy Graphs, our approach enables the extraction of interpretable and reliable explanations of vehicle behaviour in the nuScenes dataset from global and local perspectives. We demonstrate the potential of these explanations to assess whether the vehicle operates within acceptable legal boundaries and to identify possible vulnerabilities in autonomous driving datasets and models. 

**Abstract (ZH)**: 近年来，自主驾驶领域的快速发展得益于提高道路安全、减少人为驾驶错误和促进环境可持续性的潜力。得益人工智能尤其是深度学习的进步，自主车辆的表现显著提升。然而，其决策过程的不透明性，源于使用准确且复杂的AI模型，导致社会信任和监管接受度受到阻碍，因此需要提高透明度。我们提出了一种后验、模型无关的解决方案，以提供对城市环境中自主车辆行为的目的性解释。基于意图aware策略图，我们的方法可在全局和局部视角下从nuScenes数据集中提取可解释且可靠的车辆行为说明。我们展示了这些说明的潜在价值，以评估车辆是否在法律允许的范围内运行，并识别自主驾驶数据集和模型中的可能漏洞。 

---
# Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning 

**Title (ZH)**: 基于语言引导可组装因果组件的 reinforcement learning 中 unseen 环境建模 

**Authors**: Xinyue Wang, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08361)  

**Abstract**: Generalization in reinforcement learning (RL) remains a significant challenge, especially when agents encounter novel environments with unseen dynamics. Drawing inspiration from human compositional reasoning -- where known components are reconfigured to handle new situations -- we introduce World Modeling with Compositional Causal Components (WM3C). This novel framework enhances RL generalization by learning and leveraging compositional causal components. Unlike previous approaches focusing on invariant representation learning or meta-learning, WM3C identifies and utilizes causal dynamics among composable elements, facilitating robust adaptation to new tasks. Our approach integrates language as a compositional modality to decompose the latent space into meaningful components and provides theoretical guarantees for their unique identification under mild assumptions. Our practical implementation uses a masked autoencoder with mutual information constraints and adaptive sparsity regularization to capture high-level semantic information and effectively disentangle transition dynamics. Experiments on numerical simulations and real-world robotic manipulation tasks demonstrate that WM3C significantly outperforms existing methods in identifying latent processes, improving policy learning, and generalizing to unseen tasks. 

**Abstract (ZH)**: 基于组合因果组件的世界建模（WM3C）在强化学习中的泛化 

---
# Foundation Models Knowledge Distillation For Battery Capacity Degradation Forecast 

**Title (ZH)**: 基础模型知识蒸馏电池容量退化预测 

**Authors**: Joey Chan, Zhen Chen, Ershun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08151)  

**Abstract**: Accurate estimation of lithium-ion battery capacity degradation is critical for enhancing the reliability and safety of battery operations. Traditional expert models, tailored to specific scenarios, provide isolated estimations. With the rapid advancement of data-driven techniques, a series of general-purpose time-series foundation models have been developed. However, foundation models specifically designed for battery capacity degradation remain largely unexplored. To enable zero-shot generalization in battery degradation prediction using large model technology, this study proposes a degradation-aware fine-tuning strategy for time-series foundation models. We apply this strategy to fine-tune the Timer model on approximately 10 GB of open-source battery charge discharge data. Validation on our released CycleLife-SJTUIE dataset demonstrates that the fine-tuned Battery-Timer possesses strong zero-shot generalization capability in capacity degradation forecasting. To address the computational challenges of deploying large models, we further propose a knowledge distillation framework that transfers the knowledge of pre-trained foundation models into compact expert models. Distillation results across several state-of-the-art time-series expert models confirm that foundation model knowledge significantly improves the multi-condition generalization of expert models. 

**Abstract (ZH)**: 准确估计锂离子电池容量衰退对于提高电池操作的可靠性和安全性至关重要。传统的专家模型针对特定场景提供孤立的估计。随着数据驱动技术的飞速发展，一系列通用时间序列基础模型被开发出来。然而，专门针对电池容量衰退的基础模型的研究仍然相对空白。为了利用大规模模型技术在电池衰退预测中实现零样本泛化，本文提出了一种衰退感知的时间序列基础模型微调策略。我们采用该策略对大约10 GB的开源电池充放电数据进行微调Timer模型。在我们发布的CycleLife-SJTUIE数据集上的验证表明，微调后的Battery-Timer在容量衰退预测中具备强大的零样本泛化能力。为了应对部署大规模模型的计算挑战，本文进一步提出了一种知识蒸馏框架，将预训练基础模型的知识转移到紧凑的专家模型中。针对多个性能前沿的时间序列专家模型的蒸馏结果证实，基础模型知识显著提高了专家模型在多条件泛化方面的性能。 

---
# Explainable Reinforcement Learning Agents Using World Models 

**Title (ZH)**: 使用世界模型的可解释强化学习代理 

**Authors**: Madhuri Singh, Amal Alabdulkarim, Gennie Mansi, Mark O. Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2505.08073)  

**Abstract**: Explainable AI (XAI) systems have been proposed to help people understand how AI systems produce outputs and behaviors. Explainable Reinforcement Learning (XRL) has an added complexity due to the temporal nature of sequential decision-making. Further, non-AI experts do not necessarily have the ability to alter an agent or its policy. We introduce a technique for using World Models to generate explanations for Model-Based Deep RL agents. World Models predict how the world will change when actions are performed, allowing for the generation of counterfactual trajectories. However, identifying what a user wanted the agent to do is not enough to understand why the agent did something else. We augment Model-Based RL agents with a Reverse World Model, which predicts what the state of the world should have been for the agent to prefer a given counterfactual action. We show that explanations that show users what the world should have been like significantly increase their understanding of the agent policy. We hypothesize that our explanations can help users learn how to control the agents execution through by manipulating the environment. 

**Abstract (ZH)**: 可解释的人工智能（XAI）系统被提出以帮助人们理解AI系统如何生成输出和行为。由于序列决策的时序性质，可解释的强化学习（XRL）增加了复杂性。此外，并非所有非AI专家都具备修改智能体或其策略的能力。我们提出了一种使用世界模型生成基于模型深度强化学习（Model-Based Deep RL）智能体解释的技术。世界模型预测执行动作时世界将如何变化，从而允许生成反事实轨迹。然而，仅确定用户想让智能体做什么还不足以理解其为何做了其他事情。我们为基于模型的RL智能体增加了逆向世界模型，该模型预测世界应为何种状态，以便智能体更偏好某个给定的反事实动作。我们展示了向用户展示世界应为何种状态的解释显著提高了他们对智能体策略的理解。我们假设我们的解释可以帮助用户通过操纵环境学习如何控制智能体的执行。 

---
# Towards Autonomous UAV Visual Object Search in City Space: Benchmark and Agentic Methodology 

**Title (ZH)**: 面向城市空间自主无人机视觉目标搜索：基准与主体性方法论 

**Authors**: Yatai Ji, Zhengqiu Zhu, Yong Zhao, Beidan Liu, Chen Gao, Yihao Zhao, Sihang Qiu, Yue Hu, Quanjun Yin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08765)  

**Abstract**: Aerial Visual Object Search (AVOS) tasks in urban environments require Unmanned Aerial Vehicles (UAVs) to autonomously search for and identify target objects using visual and textual cues without external guidance. Existing approaches struggle in complex urban environments due to redundant semantic processing, similar object distinction, and the exploration-exploitation dilemma. To bridge this gap and support the AVOS task, we introduce CityAVOS, the first benchmark dataset for autonomous search of common urban objects. This dataset comprises 2,420 tasks across six object categories with varying difficulty levels, enabling comprehensive evaluation of UAV agents' search capabilities. To solve the AVOS tasks, we also propose PRPSearcher (Perception-Reasoning-Planning Searcher), a novel agentic method powered by multi-modal large language models (MLLMs) that mimics human three-tier cognition. Specifically, PRPSearcher constructs three specialized maps: an object-centric dynamic semantic map enhancing spatial perception, a 3D cognitive map based on semantic attraction values for target reasoning, and a 3D uncertainty map for balanced exploration-exploitation search. Also, our approach incorporates a denoising mechanism to mitigate interference from similar objects and utilizes an Inspiration Promote Thought (IPT) prompting mechanism for adaptive action planning. Experimental results on CityAVOS demonstrate that PRPSearcher surpasses existing baselines in both success rate and search efficiency (on average: +37.69% SR, +28.96% SPL, -30.69% MSS, and -46.40% NE). While promising, the performance gap compared to humans highlights the need for better semantic reasoning and spatial exploration capabilities in AVOS tasks. This work establishes a foundation for future advances in embodied target search. Dataset and source code are available at this https URL. 

**Abstract (ZH)**: 城市环境下基于视觉的空中目标搜索（AVOS）任务要求无人驾驶航空车辆（UAV）自主地利用视觉和文本线索搜索和识别目标对象，无需外部指引。现有方法在复杂城市环境中由于冗余的语义处理、相似对象区分困难以及探索与利用的困境而难以应对。为填补这一空白并支持AVOS任务，我们引入CityAVOS，这是首个用于自主搜索常见城市物体的基准数据集。该数据集包含六个类别、不同难度级别的2420个任务，能够全面评估UAV代理的搜索能力。为了解决AVOS任务，我们还提出了PRPSearcher（感知-推理-规划搜索器），这是一种由多模态大语言模型（MLLMs）驱动的新型代理方法，模拟人类三等级认知。PRPSearcher构建了三个专门的地图：以对象为中心的动力学语义图，增强空间感知；基于语义吸引值的目标推理的3D认知地图；以及平衡探索与利用的3D不确定性图。此外，我们的方法还包含去噪机制以减轻相似对象的干扰，并利用启发式促进思考（IPT）提示机制进行适应性行动规划。在CityAVOS上的实验结果表明，PRPSearcher在成功率和搜索效率方面均超过了现有基线（平均：+37.69%成功率，+28.96% SPL，-30.69% MSS，-46.40% 新错误率）。尽管表现出色，但与人类的表现差距表明AVOS任务中需要更好的语义推理和空间探索能力。本工作为未来在具身目标搜索方面的研究奠定了基础。数据集和源代码可在以下链接获取。 

---
# A Practical Introduction to Deep Reinforcement Learning 

**Title (ZH)**: 实用深度强化学习简介 

**Authors**: Yinghan Sun, Hongxi Wang, Hua Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08295)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a powerful framework for solving sequential decision-making problems, achieving remarkable success in a wide range of applications, including game AI, autonomous driving, biomedicine, and large language models. However, the diversity of algorithms and the complexity of theoretical foundations often pose significant challenges for beginners seeking to enter the field. This tutorial aims to provide a concise, intuitive, and practical introduction to DRL, with a particular focus on the Proximal Policy Optimization (PPO) algorithm, which is one of the most widely used and effective DRL methods. To facilitate learning, we organize all algorithms under the Generalized Policy Iteration (GPI) framework, offering readers a unified and systematic perspective. Instead of lengthy theoretical proofs, we emphasize intuitive explanations, illustrative examples, and practical engineering techniques. This work serves as an efficient and accessible guide, helping readers rapidly progress from basic concepts to the implementation of advanced DRL algorithms. 

**Abstract (ZH)**: 深度强化学习（DRL）已成为解决序列决策问题的一种强大框架，已在游戏AI、自主驾驶、生物医学和大规模语言模型等多种应用中取得了显著成功。然而，算法的多样性以及理论基础的复杂性常常为初学者进入该领域带来显著挑战。本教程旨在提供一个精炼、直观且实用的DRL入门介绍，特别强调广泛使用和有效的Proximal Policy Optimization（PPO）算法。为了便于学习，我们将所有算法组织在通用策略迭代（GPI）框架下，为读者提供统一和系统的视角。我们强调直观解释、示例和实用工程技巧，而非冗长的理论证明。本工作作为一项高效和易于访问的指南，帮助读者快速从基本概念过渡到高级DRL算法的实现。 

---
# Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles 

**Title (ZH)**: 基于自主车辆的水下声学跟踪中多智能体强化学习的扩展研究 

**Authors**: Matteo Gallici, Ivan Masmitja, Mario Martín  

**Link**: [PDF](https://arxiv.org/pdf/2505.08222)  

**Abstract**: Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions. 

**Abstract (ZH)**: 自主水下车辆（AV）为水下追踪等科学任务提供了经济有效的解决方案。近年来，强化学习（RL）已成为控制复杂海洋环境中的AV的强大方法。然而，将这些技术扩展到车队——这对于多目标追踪或具有快速、不可预测运动的目标至关重要——带来了显著的计算挑战。多智能体强化学习（MARL）以其样本效益差著称，尽管像Gazebo的LRAUV这样的高保真模拟器可以在单机器人模拟中实现100倍于实时的速度，但在多车辆场景中却未能提供显著的速度提升，使得MARL的训练变得不切实际。为解决这些限制，我们提出了一种迭代蒸馏方法，将高保真模拟转移至简化且GPU加速的环境，同时保持高层动力学。该方法通过并行化实现了高达30,000倍的速度提升，为端到端的GPU加速训练提供了高效途径。此外，我们引入了一种基于Transformer的新型架构（TransfMAPPO），该架构可以学习不受智能体和目标数量影响的多智能体策略，显著提高样本效率。我们通过完全在GPU上进行大规模递增学习后，在Gazebo中进行了广泛评估，结果显示，即使在存在多个快速移动目标的情况下，我们的方法也能在整个时间段内将跟踪误差保持在5米以下。这项工作填补了大规模MARL训练与高保真部署之间的差距，为实现实用海中自主舰队控制提供了可扩展的框架。 

---
# DSADF: Thinking Fast and Slow for Decision Making 

**Title (ZH)**: DSADF：快速与缓慢的决策思考 

**Authors**: Alex Zhihao Dou, Dongfei Cui, Jun Yan, Weida Wang, Benteng Chen, Haoming Wang, Zeke Xie, Shufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08189)  

**Abstract**: Although Reinforcement Learning (RL) agents are effective in well-defined environments, they often struggle to generalize their learned policies to dynamic settings due to their reliance on trial-and-error interactions. Recent work has explored applying Large Language Models (LLMs) or Vision Language Models (VLMs) to boost the generalization of RL agents through policy optimization guidance or prior knowledge. However, these approaches often lack seamless coordination between the RL agent and the foundation model, leading to unreasonable decision-making in unfamiliar environments and efficiency bottlenecks. Making full use of the inferential capabilities of foundation models and the rapid response capabilities of RL agents and enhancing the interaction between the two to form a dual system is still a lingering scientific question. To address this problem, we draw inspiration from Kahneman's theory of fast thinking (System 1) and slow thinking (System 2), demonstrating that balancing intuition and deep reasoning can achieve nimble decision-making in a complex world. In this study, we propose a Dual-System Adaptive Decision Framework (DSADF), integrating two complementary modules: System 1, comprising an RL agent and a memory space for fast and intuitive decision making, and System 2, driven by a VLM for deep and analytical reasoning. DSADF facilitates efficient and adaptive decision-making by combining the strengths of both systems. The empirical study in the video game environment: Crafter and Housekeep demonstrates the effectiveness of our proposed method, showing significant improvements in decision abilities for both unseen and known tasks. 

**Abstract (ZH)**: 基于双系统适应性决策框架的强化学习代理决策机制研究 

---
# Combining Bayesian Inference and Reinforcement Learning for Agent Decision Making: A Review 

**Title (ZH)**: 融合贝叶斯推断与强化学习的代理决策制作：一篇综述 

**Authors**: Chengmin Zhou, Ville Kyrki, Pasi Fränti, Laura Ruotsalainen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07911)  

**Abstract**: Bayesian inference has many advantages in decision making of agents (e.g. robotics/simulative agent) over a regular data-driven black-box neural network: Data-efficiency, generalization, interpretability, and safety where these advantages benefit directly/indirectly from the uncertainty quantification of Bayesian inference. However, there are few comprehensive reviews to summarize the progress of Bayesian inference on reinforcement learning (RL) for decision making to give researchers a systematic understanding. This paper focuses on combining Bayesian inference with RL that nowadays is an important approach in agent decision making. To be exact, this paper discusses the following five topics: 1) Bayesian methods that have potential for agent decision making. First basic Bayesian methods and models (Bayesian rule, Bayesian learning, and Bayesian conjugate models) are discussed followed by variational inference, Bayesian optimization, Bayesian deep learning, Bayesian active learning, Bayesian generative models, Bayesian meta-learning, and lifelong Bayesian learning. 2) Classical combinations of Bayesian methods with model-based RL (with approximation methods), model-free RL, and inverse RL. 3) Latest combinations of potential Bayesian methods with RL. 4) Analytical comparisons of methods that combine Bayesian methods with RL with respect to data-efficiency, generalization, interpretability, and safety. 5) In-depth discussions in six complex problem variants of RL, including unknown reward, partial-observability, multi-agent, multi-task, non-linear non-Gaussian, and hierarchical RL problems and the summary of how Bayesian methods work in the data collection, data processing and policy learning stages of RL to pave the way for better agent decision-making strategies. 

**Abstract (ZH)**: 贝叶斯推断在代理决策（如机器人/模拟代理）中相对于常规的数据驱动黑盒神经网络有许多优势：数据效率、泛化能力、可解释性和安全性，这些优势直接或间接地受益于贝叶斯推断的不确定性量化。然而，鲜有全面的综述总结贝叶斯推断在强化学习（RL）中的进展，以帮助研究人员系统地理解这一领域。本文专注于结合贝叶斯推断与RL，这是当前代理决策中的一个重要方法。具体来说，本文讨论了以下五个主题：1) 具有代理决策潜力的贝叶斯方法。首先讨论基础的贝叶斯方法和模型（贝叶斯规则、贝叶斯学习和共轭模型），随后讨论变分推断、贝叶斯优化、贝叶斯深度学习、贝叶斯主动学习、生成模型、元学习和终身学习。2) 贝叶斯方法与基于模型的RL（含近似方法）、无模型RL和逆RL的经典结合。3) 最新的贝叶斯方法与RL的结合。4) 在数据效率、泛化能力、可解释性和安全性方面，结合贝叶斯方法与RL的方法的分析比较。5) 对六种复杂RL问题变体的深入讨论，包括未知奖励、部分可观测性、多代理、多任务、非线性非高斯以及层次化RL问题，并总结贝叶斯方法在RL的数据收集、数据处理和策略学习阶段的工作，为更好的代理决策策略铺平道路。 

---
# Reinforcement Learning (RL) Meets Urban Climate Modeling: Investigating the Efficacy and Impacts of RL-Based HVAC Control 

**Title (ZH)**: 强化学习（RL）与城市气候 modeling相结合：基于RL的HVAC控制效果与影响调查 

**Authors**: Junjie Yu, John S. Schreck, David John Gagne, Keith W. Oleson, Jie Li, Yongtu Liang, Qi Liao, Mingfei Sun, David O. Topping, Zhonghua Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07045)  

**Abstract**: Reinforcement learning (RL)-based heating, ventilation, and air conditioning (HVAC) control has emerged as a promising technology for reducing building energy consumption while maintaining indoor thermal comfort. However, the efficacy of such strategies is influenced by the background climate and their implementation may potentially alter both the indoor climate and local urban climate. This study proposes an integrated framework combining RL with an urban climate model that incorporates a building energy model, aiming to evaluate the efficacy of RL-based HVAC control across different background climates, impacts of RL strategies on indoor climate and local urban climate, and the transferability of RL strategies across cities. Our findings reveal that the reward (defined as a weighted combination of energy consumption and thermal comfort) and the impacts of RL strategies on indoor climate and local urban climate exhibit marked variability across cities with different background climates. The sensitivity of reward weights and the transferability of RL strategies are also strongly influenced by the background climate. Cities in hot climates tend to achieve higher rewards across most reward weight configurations that balance energy consumption and thermal comfort, and those cities with more varying atmospheric temperatures demonstrate greater RL strategy transferability. These findings underscore the importance of thoroughly evaluating RL-based HVAC control strategies in diverse climatic contexts. This study also provides a new insight that city-to-city learning will potentially aid the deployment of RL-based HVAC control. 

**Abstract (ZH)**: 基于强化学习（RL）的 HVAC 控制结合城市气候模型的综合框架：评价不同背景气候下 HVAC 控制策略的有效性及其对室内和局部城市气候的影响，以及策略的跨城迁移性 

---
