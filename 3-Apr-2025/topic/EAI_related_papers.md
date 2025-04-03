# Slot-Level Robotic Placement via Visual Imitation from Single Human Video 

**Title (ZH)**: 基于单人视频的视觉模仿-slot级机器人放置 

**Authors**: Dandan Shan, Kaichun Mo, Wei Yang, Yu-Wei Chao, David Fouhey, Dieter Fox, Arsalan Mousavian  

**Link**: [PDF](https://arxiv.org/pdf/2504.01959)  

**Abstract**: The majority of modern robot learning methods focus on learning a set of pre-defined tasks with limited or no generalization to new tasks. Extending the robot skillset to novel tasks involves gathering an extensive amount of training data for additional tasks. In this paper, we address the problem of teaching new tasks to robots using human demonstration videos for repetitive tasks (e.g., packing). This task requires understanding the human video to identify which object is being manipulated (the pick object) and where it is being placed (the placement slot). In addition, it needs to re-identify the pick object and the placement slots during inference along with the relative poses to enable robot execution of the task. To tackle this, we propose SLeRP, a modular system that leverages several advanced visual foundation models and a novel slot-level placement detector Slot-Net, eliminating the need for expensive video demonstrations for training. We evaluate our system using a new benchmark of real-world videos. The evaluation results show that SLeRP outperforms several baselines and can be deployed on a real robot. 

**Abstract (ZH)**: 利用人类示范视频教学机器人执行新任务：SLeRP模块化系统及其应用 

---
# Anticipating Degradation: A Predictive Approach to Fault Tolerance in Robot Swarms 

**Title (ZH)**: 预见退化：机器人蜂群容错的预测方法 

**Authors**: James O'Keeffe  

**Link**: [PDF](https://arxiv.org/pdf/2504.01594)  

**Abstract**: An active approach to fault tolerance is essential for robot swarms to achieve long-term autonomy. Previous efforts have focused on responding to spontaneous electro-mechanical faults and failures. However, many faults occur gradually over time. Waiting until such faults have manifested as failures before addressing them is both inefficient and unsustainable in a variety of scenarios. This work argues that the principles of predictive maintenance, in which potential faults are resolved before they hinder the operation of the swarm, offer a promising means of achieving long-term fault tolerance. This is a novel approach to swarm fault tolerance, which is shown to give a comparable or improved performance when tested against a reactive approach in almost all cases tested. 

**Abstract (ZH)**: 一种积极的方法对于实现机器人 swarm 的长期自主性来说是实现容错所不可或缺的。预测性维护原理在 swarm 故障容忍中具有潜在的前景，该方法在几乎所有测试案例中表现出与反应性方法相当或更好的性能。 

---
# Building Knowledge from Interactions: An LLM-Based Architecture for Adaptive Tutoring and Social Reasoning 

**Title (ZH)**: 基于交互构建知识：一种基于大语言模型的自适应辅导与社会推理架构 

**Authors**: Luca Garello, Giulia Belgiovine, Gabriele Russo, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2504.01588)  

**Abstract**: Integrating robotics into everyday scenarios like tutoring or physical training requires robots capable of adaptive, socially engaging, and goal-oriented interactions. While Large Language Models show promise in human-like communication, their standalone use is hindered by memory constraints and contextual incoherence. This work presents a multimodal, cognitively inspired framework that enhances LLM-based autonomous decision-making in social and task-oriented Human-Robot Interaction. Specifically, we develop an LLM-based agent for a robot trainer, balancing social conversation with task guidance and goal-driven motivation. To further enhance autonomy and personalization, we introduce a memory system for selecting, storing and retrieving experiences, facilitating generalized reasoning based on knowledge built across different interactions. A preliminary HRI user study and offline experiments with a synthetic dataset validate our approach, demonstrating the system's ability to manage complex interactions, autonomously drive training tasks, and build and retrieve contextual memories, advancing socially intelligent robotics. 

**Abstract (ZH)**: 将机器人集成到如辅导或体能训练等日常生活场景中需要具备适应性、社交互动性和目标导向性的机器人。尽管大型语言模型在Humans-like交流方面显示出潜力，但其独立使用受限于记忆约束和语境不一致问题。本研究提出了一种多模态、认知启发式的框架，以增强基于大型语言模型的自主决策能力，在社交和任务导向的人机交互中发挥重要作用。具体而言，我们为机器人教练开发了一个基于大型语言模型的代理，平衡社交对话与任务指导及目标驱动的动机。为进一步增强自主性和个性化，我们引入了一个记忆系统，用于选择、存储和检索经验，基于不同交互中积累的知识促进泛化推理。初步的人机交互用户研究和基于合成数据集的离线实验验证了我们的方法，展示了该系统管理复杂交互、自主驱动训练任务以及构建和检索上下文记忆的能力，推动了社会智能机器人技术的发展。 

---
# Teaching Robots to Handle Nuclear Waste: A Teleoperation-Based Learning Approach< 

**Title (ZH)**: 基于远程操作的机器人处理核废料学习方法 

**Authors**: Joong-Ku Lee, Hyeonseok Choi, Young Soo Park, Jee-Hwan Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01405)  

**Abstract**: This paper presents a Learning from Teleoperation (LfT) framework that integrates human expertise with robotic precision to enable robots to autonomously perform skills learned from human operators. The proposed framework addresses challenges in nuclear waste handling tasks, which often involve repetitive and meticulous manipulation operations. By capturing operator movements and manipulation forces during teleoperation, the framework utilizes this data to train machine learning models capable of replicating and generalizing human skills. We validate the effectiveness of the LfT framework through its application to a power plug insertion task, selected as a representative scenario that is repetitive yet requires precise trajectory and force control. Experimental results highlight significant improvements in task efficiency, while reducing reliance on continuous operator involvement. 

**Abstract (ZH)**: 基于遥操作的学习（LfT）框架：将人类 expertise 与机器人精确性集成以使机器人自主执行从人类操作者学到的技能 

---
# Inverse RL Scene Dynamics Learning for Nonlinear Predictive Control in Autonomous Vehicles 

**Title (ZH)**: 基于逆强化学习的场景动力学学习在自动驾驶车辆的非线性预测控制中 

**Authors**: Sorin Grigorescu, Mihai Zaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.01336)  

**Abstract**: This paper introduces the Deep Learning-based Nonlinear Model Predictive Controller with Scene Dynamics (DL-NMPC-SD) method for autonomous navigation. DL-NMPC-SD uses an a-priori nominal vehicle model in combination with a scene dynamics model learned from temporal range sensing information. The scene dynamics model is responsible for estimating the desired vehicle trajectory, as well as to adjust the true system model used by the underlying model predictive controller. We propose to encode the scene dynamics model within the layers of a deep neural network, which acts as a nonlinear approximator for the high order state-space of the operating conditions. The model is learned based on temporal sequences of range sensing observations and system states, both integrated by an Augmented Memory component. We use Inverse Reinforcement Learning and the Bellman optimality principle to train our learning controller with a modified version of the Deep Q-Learning algorithm, enabling us to estimate the desired state trajectory as an optimal action-value function. We have evaluated DL-NMPC-SD against the baseline Dynamic Window Approach (DWA), as well as against two state-of-the-art End2End and reinforcement learning methods, respectively. The performance has been measured in three experiments: i) in our GridSim virtual environment, ii) on indoor and outdoor navigation tasks using our RovisLab AMTU (Autonomous Mobile Test Unit) platform and iii) on a full scale autonomous test vehicle driving on public roads. 

**Abstract (ZH)**: 基于深度学习的场景动力学非线性模型预测控制方法（DL-NMPC-SD）在自主导航中的应用 

---
# Bi-LAT: Bilateral Control-Based Imitation Learning via Natural Language and Action Chunking with Transformers 

**Title (ZH)**: 基于双边控制的自然语言和动作片段变换器辅助的模仿学习：Bi-LAT 

**Authors**: Takumi Kobayashi, Masato Kobayashi, Thanpimon Buamanee, Yuki Uranishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01301)  

**Abstract**: We present Bi-LAT, a novel imitation learning framework that unifies bilateral control with natural language processing to achieve precise force modulation in robotic manipulation. Bi-LAT leverages joint position, velocity, and torque data from leader-follower teleoperation while also integrating visual and linguistic cues to dynamically adjust applied force. By encoding human instructions such as "softly grasp the cup" or "strongly twist the sponge" through a multimodal Transformer-based model, Bi-LAT learns to distinguish nuanced force requirements in real-world tasks. We demonstrate Bi-LAT's performance in (1) unimanual cup-stacking scenario where the robot accurately modulates grasp force based on language commands, and (2) bimanual sponge-twisting task that requires coordinated force control. Experimental results show that Bi-LAT effectively reproduces the instructed force levels, particularly when incorporating SigLIP among tested language encoders. Our findings demonstrate the potential of integrating natural language cues into imitation learning, paving the way for more intuitive and adaptive human-robot interaction. For additional material, please visit: this https URL 

**Abstract (ZH)**: Bi-LAT：一种联合双臂控制与自然语言处理的新型 imitation learning 框架，实现精确的机器人操作力调控 

---
# The Social Life of Industrial Arms: How Arousal and Attention Shape Human-Robot Interaction 

**Title (ZH)**: 工业机器人的社会生活：唤醒程度与注意力如何塑造人机交互 

**Authors**: Roy El-Helou, Matthew K.X.J Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01260)  

**Abstract**: This study explores how human perceptions of a non-anthropomorphic robotic manipulator are shaped by two key dimensions of behaviour: arousal, defined as the robot's movement energy and expressiveness, and attention, defined as the robot's capacity to selectively orient toward and engage with a user. We introduce a novel control architecture that integrates a gaze-like attention engine with an arousal-modulated motion system to generate socially meaningful behaviours. In a user study, we find that robots exhibiting high attention -- actively directing their focus toward users -- are perceived as warmer and more competent, intentional, and lifelike. In contrast, high arousal -- characterized by fast, expansive, and energetic motions -- increases perceptions of discomfort and disturbance. Importantly, a combination of focused attention and moderate arousal yields the highest ratings of trust and sociability, while excessive arousal diminishes social engagement. These findings offer design insights for endowing non-humanoid robots with expressive, intuitive behaviours that support more natural human-robot interaction. 

**Abstract (ZH)**: 本研究探讨了人类对非人格化机器 manipulator 的感知如何受到两类行为维度的影响：唤醒（定义为机器人的运动能量和表达性）和注意力（定义为机器人选择性地朝向和与用户互动的能力）。我们提出了一种新颖的控制架构，结合了类似凝视的注意力引擎和受唤醒程度调节的运动系统，以生成具有社会意义的行为。在用户研究中，我们发现主动将注意力集中在用户身上的高注意力水平的机器人被感知为更加温暖、有能力、有意向并且更加拟人。相反，具有快速、扩展性和能量充沛运动的高唤醒水平增加了不适和干扰的感知。重要的是，结合集中注意力和适度唤醒的组合获得了最高的信任和社会互动评分，而过度的唤醒降低了社会互动。这些发现为赋予非人形机器人表达性和直观的行为提供了设计见解，以支持更自然的人机互动。 

---
# Plan-and-Act using Large Language Models for Interactive Agreement 

**Title (ZH)**: 使用大型语言模型进行计划与行动以达成互动共识 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01252)  

**Abstract**: Recent large language models (LLMs) are capable of planning robot actions. In this paper, we explore how LLMs can be used for planning actions with tasks involving situational human-robot interaction (HRI). A key problem of applying LLMs in situational HRI is balancing between "respecting the current human's activity" and "prioritizing the robot's task," as well as understanding the timing of when to use the LLM to generate an action plan. In this paper, we propose a necessary plan-and-act skill design to solve the above problems. We show that a critical factor for enabling a robot to switch between passive / active interaction behavior is to provide the LLM with an action text about the current robot's action. We also show that a second-stage question to the LLM (about the next timing to call the LLM) is necessary for planning actions at an appropriate timing. The skill design is applied to an Engage skill and is tested on four distinct interaction scenarios. We show that by using the skill design, LLMs can be leveraged to easily scale to different HRI scenarios with a reasonable success rate reaching 90% on the test scenarios. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）能够规划机器人动作。本文探讨了如何在涉及情境人类-机器人交互（HRI）的任务中利用LLMs进行动作规划。将LLMs应用于情境HRI的关键问题是平衡“尊重当前人类的活动”和“优先考虑机器人的任务”的关系，以及理解何时使用LLMs生成动作计划。本文提出了一种必要的计划与执行技能设计以解决上述问题。我们表明，使机器人能够切换到被动/主动交互行为的关键因素是向LLMs提供当前机器人动作的文字描述。我们还表明，在适当的时间规划动作需要LLMs的第二个阶段问题（关于何时再次调用LLMs的时间）。该技能设计应用于一种Engage技能，并在四种不同的交互场景中进行了测试。我们表明，通过使用该技能设计，可以在合理成功率达到90%的情况下，轻松地将LLMs扩展到不同的HRI场景中。 

---
# Value Iteration for Learning Concurrently Executable Robotic Control Tasks 

**Title (ZH)**: 并发可执行机器人控制任务的学习价值迭代 

**Authors**: Sheikh A. Tahmid, Gennaro Notomista  

**Link**: [PDF](https://arxiv.org/pdf/2504.01174)  

**Abstract**: Many modern robotic systems such as multi-robot systems and manipulators exhibit redundancy, a property owing to which they are capable of executing multiple tasks. This work proposes a novel method, based on the Reinforcement Learning (RL) paradigm, to train redundant robots to be able to execute multiple tasks concurrently. Our approach differs from typical multi-objective RL methods insofar as the learned tasks can be combined and executed in possibly time-varying prioritized stacks. We do so by first defining a notion of task independence between learned value functions. We then use our definition of task independence to propose a cost functional that encourages a policy, based on an approximated value function, to accomplish its control objective while minimally interfering with the execution of higher priority tasks. This allows us to train a set of control policies that can be executed simultaneously. We also introduce a version of fitted value iteration to learn to approximate our proposed cost functional efficiently. We demonstrate our approach on several scenarios and robotic systems. 

**Abstract (ZH)**: 基于强化学习的冗余机器人多任务并发执行新方法 

---
# Extended Hybrid Zero Dynamics for Bipedal Walking of the Knee-less Robot SLIDER 

**Title (ZH)**: 膝关节-less 机器人SLIDER的扩展混合零动力学双足步行控制 

**Authors**: Rui Zong, Martin Liang, Yuntian Fang, Ke Wang, Xiaoshuai Chen, Wei Chen, Petar Kormushev  

**Link**: [PDF](https://arxiv.org/pdf/2504.01165)  

**Abstract**: Knee-less bipedal robots like SLIDER have the advantage of ultra-lightweight legs and improved walking energy efficiency compared to traditional humanoid robots. In this paper, we firstly introduce an improved hardware design of the bipedal robot SLIDER with new line-feet and more optimized mass distribution which enables higher locomotion speeds. Secondly, we propose an extended Hybrid Zero Dynamics (eHZD) method, which can be applied to prismatic joint robots like SLIDER. The eHZD method is then used to generate a library of gaits with varying reference velocities in an offline way. Thirdly, a Guided Deep Reinforcement Learning (DRL) algorithm is proposed to use the pre-generated library to create walking control policies in real-time. This approach allows us to combine the advantages of both HZD (for generating stable gaits with a full-dynamics model) and DRL (for real-time adaptive gait generation). The experimental results show that this approach achieves 150% higher walking velocity than the previous MPC-based approach. 

**Abstract (ZH)**: 膝关节less双足机器人如SLIDER具有超轻 legs 和更好的步行能效比传统人形机器人。本文首先介绍了具有新型线脚和更优化质量分布的SLIDER双足机器人改进硬件设计，使其能够达到更高的移动速度。其次，提出了扩展的混合零动态(eHZD)方法，该方法适用于如SLIDER这样的普朗特尔关节机器人。然后，使用eHZD方法以离线方式生成具有不同参考速度的步态库。第三，提出了一种引导式深度强化学习(GDRL)算法，利用预先生成的库实现实时步行控制策略生成。该方法结合了混合零动态(HZD)和深度强化学习(DRL)的优势。实验结果表明，该方法比基于MPC的方法实现了150%更高的步行速度。 

---
# Making Sense of Robots in Public Spaces: A Study of Trash Barrel Robots 

**Title (ZH)**: 理解公共空间中的机器人：垃圾桶机器人研究 

**Authors**: Fanjun Bu, Kerstin Fischer, Wendy Ju  

**Link**: [PDF](https://arxiv.org/pdf/2504.01121)  

**Abstract**: In this work, we analyze video data and interviews from a public deployment of two trash barrel robots in a large public space to better understand the sensemaking activities people perform when they encounter robots in public spaces. Based on an analysis of 274 human-robot interactions and interviews with N=65 individuals or groups, we discovered that people were responding not only to the robots or their behavior, but also to the general idea of deploying robots as trashcans, and the larger social implications of that idea. They wanted to understand details about the deployment because having that knowledge would change how they interact with the robot. Based on our data and analysis, we have provided implications for design that may be topics for future human-robot design researchers who are exploring robots for public space deployment. Furthermore, our work offers a practical example of analyzing field data to make sense of robots in public spaces. 

**Abstract (ZH)**: 本研究分析了两个垃圾桶机器人在大型公共空间公共部署中的视频数据和访谈，以更好地理解人们在遇到公共空间中的机器人时所进行的意义建构活动。基于对274次人机互动和与N=65名个体或小组进行的访谈的分析，我们发现人们不仅对机器人及其行为作出反应，还对部署机器人作为垃圾桶这一行为本身以及这一行为背后更广泛的社会意义作出了反应。他们希望了解到关于部署的详细信息，因为这些知识会影响他们与机器人互动的方式。基于我们的数据和分析，我们提出了对未来探索公共空间中机器人部署的人机设计研究人员具有指导意义的设计建议。此外，我们的研究为通过分析现场数据来理解公共空间中的机器人提供了实际案例。 

---
# HomeEmergency -- Using Audio to Find and Respond to Emergencies in the Home 

**Title (ZH)**: HomeEmergency——使用音频发现并响应家庭紧急情况 

**Authors**: James F. Mullen Jr, Dhruva Kumar, Xuewei Qi, Rajasimman Madhivanan, Arnie Sen, Dinesh Manocha, Richard Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.01089)  

**Abstract**: In the United States alone accidental home deaths exceed 128,000 per year. Our work aims to enable home robots who respond to emergency scenarios in the home, preventing injuries and deaths. We introduce a new dataset of household emergencies based in the ThreeDWorld simulator. Each scenario in our dataset begins with an instantaneous or periodic sound which may or may not be an emergency. The agent must navigate the multi-room home scene using prior observations, alongside audio signals and images from the simulator, to determine if there is an emergency or not.
In addition to our new dataset, we present a modular approach for localizing and identifying potential home emergencies. Underpinning our approach is a novel probabilistic dynamic scene graph (P-DSG), where our key insight is that graph nodes corresponding to agents can be represented with a probabilistic edge. This edge, when refined using Bayesian inference, enables efficient and effective localization of agents in the scene. We also utilize multi-modal vision-language models (VLMs) as a component in our approach, determining object traits (e.g. flammability) and identifying emergencies. We present a demonstration of our method completing a real-world version of our task on a consumer robot, showing the transferability of both our task and our method. Our dataset will be released to the public upon this papers publication. 

**Abstract (ZH)**: 美国境内意外家庭死亡人数每年超过128,000人。我们的工作旨在使家用机器人能够应对家庭中的紧急情况，防止受伤和死亡。我们基于ThreeDWorld模拟器引入了一个新的家庭紧急情况数据集。每个数据集中的情景以瞬时或周期性声音开始，这可能是或可能不是紧急情况。代理必须利用先前的观察，以及模拟器中的音频信号和图像，导航多房间家居场景，以判断是否存在紧急情况。除了我们的新数据集外，我们还提出了一种模块化方法来定位和识别潜在的家庭紧急情况。我们方法的核心是新颖的概率动态场景图（P-DSG），关键见解是代理节点可以用概率边表示。通过贝叶斯推理对这条边进行细化，可以实现场景中代理的有效定位。我们还利用多模态视觉语言模型（VLMs）作为方法的一部分，确定物体特性（如易燃性）并识别紧急情况。我们展示了我们的方法在消费级机器人上完成我们任务的演示，证明了我们任务和方法的可移植性。我们的数据集将在论文发表后公开。 

---
# Quattro: Transformer-Accelerated Iterative Linear Quadratic Regulator Framework for Fast Trajectory Optimization 

**Title (ZH)**: Quattro：基于 Transformer 加速的迭代二次调节框架，用于快速轨迹优化 

**Authors**: Yue Wang, Hoayu Wang, Zhaoxing Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01806)  

**Abstract**: Real-time optimal control remains a fundamental challenge in robotics, especially for nonlinear systems with stringent performance requirements. As one of the representative trajectory optimization algorithms, the iterative Linear Quadratic Regulator (iLQR) faces limitations due to their inherently sequential computational nature, which restricts the efficiency and applicability of real-time control for robotic systems. While existing parallel implementations aim to overcome the above limitations, they typically demand additional computational iterations and high-performance hardware, leading to only modest practical improvements. In this paper, we introduce Quattro, a transformer-accelerated iLQR framework employing an algorithm-hardware co-design strategy to predict intermediate feedback and feedforward matrices. It facilitates effective parallel computations on resource-constrained devices without sacrificing accuracy. Experiments on cart-pole and quadrotor systems show an algorithm-level acceleration of up to 5.3$\times$ and 27$\times$ per iteration, respectively. When integrated into a Model Predictive Control (MPC) framework, Quattro achieves overall speedups of 2.8$\times$ for the cart-pole and 17.8$\times$ for the quadrotor compared to the one that applies traditional iLQR. Transformer inference is deployed on FPGA to maximize performance, achieving up to 27.3$\times$ speedup over commonly used computing devices, with around 2 to 4$\times$ power reduction and acceptable hardware overhead. 

**Abstract (ZH)**: 基于变压器加速的Quattro：一种算法-硬件协同设计的iLQR框架 

---
# Beyond Non-Expert Demonstrations: Outcome-Driven Action Constraint for Offline Reinforcement Learning 

**Title (ZH)**: 超越非专家演示： Offline 强化学习的基于结果的动作约束 

**Authors**: Ke Jiang, Wen Jiang, Yao Li, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01719)  

**Abstract**: We address the challenge of offline reinforcement learning using realistic data, specifically non-expert data collected through sub-optimal behavior policies. Under such circumstance, the learned policy must be safe enough to manage \textit{distribution shift} while maintaining sufficient flexibility to deal with non-expert (bad) demonstrations from offline this http URL tackle this issue, we introduce a novel method called Outcome-Driven Action Flexibility (ODAF), which seeks to reduce reliance on the empirical action distribution of the behavior policy, hence reducing the negative impact of those bad this http URL be specific, a new conservative reward mechanism is developed to deal with {\it distribution shift} by evaluating actions according to whether their outcomes meet safety requirements - remaining within the state support area, rather than solely depending on the actions' likelihood based on offline this http URL theoretical justification, we provide empirical evidence on widely used MuJoCo and various maze benchmarks, demonstrating that our ODAF method, implemented using uncertainty quantification techniques, effectively tolerates unseen transitions for improved "trajectory stitching," while enhancing the agent's ability to learn from realistic non-expert data. 

**Abstract (ZH)**: 基于现实数据的离线强化学习挑战：一种基于结果驱动的动作灵活性方法 

---
# Reasoning LLMs for User-Aware Multimodal Conversational Agents 

**Title (ZH)**: 面向用户的多模态对话代理的推理大型语言模型 

**Authors**: Hamed Rahimi, Jeanne Cattoni, Meriem Beghili, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01700)  

**Abstract**: Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment. 

**Abstract (ZH)**: 基于USER-LLM R1的用户意识对话代理框架：通过动态用户画像和模型初始化解决冷启动问题 

---
# LLM-mediated Dynamic Plan Generation with a Multi-Agent Approach 

**Title (ZH)**: 基于多代理方法的LLM调解动态计划生成 

**Authors**: Reo Abe, Akifumi Ito, Kanata Takayasu, Satoshi Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2504.01637)  

**Abstract**: Planning methods with high adaptability to dynamic environments are crucial for the development of autonomous and versatile robots. We propose a method for leveraging a large language model (GPT-4o) to automatically generate networks capable of adapting to dynamic environments. The proposed method collects environmental "status," representing conditions and goals, and uses them to generate agents. These agents are interconnected on the basis of specific conditions, resulting in networks that combine flexibility and generality. We conducted evaluation experiments to compare the networks automatically generated with the proposed method with manually constructed ones, confirming the comprehensiveness of the proposed method's networks and their higher generality. This research marks a significant advancement toward the development of versatile planning methods applicable to robotics, autonomous vehicles, smart systems, and other complex environments. 

**Abstract (ZH)**: 利用大型语言模型（GPT-4o）自动生成适用于动态环境的高适应性规划网络的方法 

---
# Cuddle-Fish: Exploring a Soft Floating Robot with Flapping Wings for Physical Interactions 

**Title (ZH)**: cuddle-鱼：探索具有拍打翅膀的软体漂浮机器人用于物理交互 

**Authors**: Mingyang Xu, Jiayi Shao, Yulan Ju, Ximing Shen, Qingyuan Gao, Weijen Chen, Qing Zhang, Yun Suen Pai, Giulia Barbareschi, Matthias Hoppe, Kouta Minamizawa, Kai Kunze  

**Link**: [PDF](https://arxiv.org/pdf/2504.01293)  

**Abstract**: Flying robots, such as quadrotor drones, offer new possibilities for human-robot interaction but often pose safety risks due to fast-spinning propellers, rigid structures, and noise. In contrast, lighter-than-air flapping-wing robots, inspired by animal movement, offer a soft, quiet, and touch-safe alternative. Building on these advantages, we present \textit{Cuddle-Fish}, a soft, flapping-wing floating robot designed for safe, close-proximity interactions in indoor spaces. Through a user study with 24 participants, we explored their perceptions of the robot and experiences during a series of co-located demonstrations in which the robot moved near them. Results showed that participants felt safe, willingly engaged in touch-based interactions with the robot, and exhibited spontaneous affective behaviours, such as patting, stroking, hugging, and cheek-touching, without external prompting. They also reported positive emotional responses towards the robot. These findings suggest that the soft floating robot with flapping wings can serve as a novel and socially acceptable alternative to traditional rigid flying robots, opening new possibilities for companionship, play, and interactive experiences in everyday indoor environments. 

**Abstract (ZH)**: 软扑翼浮空机器人Cuddle-Fish：一种安全近距离互动的新颖替代方案 

---
# Cal or No Cal? -- Real-Time Miscalibration Detection of LiDAR and Camera Sensors 

**Title (ZH)**: 有校准或不校准？-- 激光雷达和摄像头传感器的实时失校准检测 

**Authors**: Ilir Tahiraj, Jeremialie Swadiryus, Felix Fent, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.01040)  

**Abstract**: The goal of extrinsic calibration is the alignment of sensor data to ensure an accurate representation of the surroundings and enable sensor fusion applications. From a safety perspective, sensor calibration is a key enabler of autonomous driving. In the current state of the art, a trend from target-based offline calibration towards targetless online calibration can be observed. However, online calibration is subject to strict real-time and resource constraints which are not met by state-of-the-art methods. This is mainly due to the high number of parameters to estimate, the reliance on geometric features, or the dependence on specific vehicle maneuvers. To meet these requirements and ensure the vehicle's safety at any time, we propose a miscalibration detection framework that shifts the focus from the direct regression of calibration parameters to a binary classification of the calibration state, i.e., calibrated or miscalibrated. Therefore, we propose a contrastive learning approach that compares embedded features in a latent space to classify the calibration state of two different sensor modalities. Moreover, we provide a comprehensive analysis of the feature embeddings and challenging calibration errors that highlight the performance of our approach. As a result, our method outperforms the current state-of-the-art in terms of detection performance, inference time, and resource demand. The code is open source and available on this https URL. 

**Abstract (ZH)**: 目标检测中的传感器偏差检测框架：从直接回归校准参数转向校准状态的二元分类 

---
# FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs 

**Title (ZH)**: FineLIP：通过与较长文本输入的精细对齐扩展CLIP的能力 

**Authors**: Mothilal Asokan, Kebin Wu, Fatima Albreiki  

**Link**: [PDF](https://arxiv.org/pdf/2504.01916)  

**Abstract**: As a pioneering vision-language model, CLIP (Contrastive Language-Image Pre-training) has achieved significant success across various domains and a wide range of downstream vision-language tasks. However, the text encoders in popular CLIP models are limited to processing only 77 text tokens, which constrains their ability to effectively handle longer, detail-rich captions. Additionally, CLIP models often struggle to effectively capture detailed visual and textual information, which hampers their performance on tasks that require fine-grained analysis. To address these limitations, we present a novel approach, \textbf{FineLIP}, that extends the capabilities of CLIP. FineLIP enhances cross-modal text-image mapping by incorporating \textbf{Fine}-grained alignment with \textbf{L}onger text input within the CL\textbf{IP}-style framework. FineLIP first extends the positional embeddings to handle longer text, followed by the dynamic aggregation of local image and text tokens. The aggregated results are then used to enforce fine-grained token-to-token cross-modal alignment. We validate our model on datasets with long, detailed captions across two tasks: zero-shot cross-modal retrieval and text-to-image generation. Quantitative and qualitative experimental results demonstrate the effectiveness of FineLIP, outperforming existing state-of-the-art approaches. Furthermore, comprehensive ablation studies validate the benefits of key design elements within FineLIP. 

**Abstract (ZH)**: 作为先驱性的跨模态模型，CLIP（对比语言-图像预训练）已在多个领域和各种下游跨模态任务中取得了显著成功。然而，流行的CLIP模型中的文本编码器仅限于处理77个文本标记，这限制了其有效处理较长、细节丰富描述的能力。此外，CLIP模型往往难以有效捕捉详细的视觉和文本信息，这阻碍了其在需要精细分析的任务中的表现。为解决这些限制，我们提出了一种新颖的方法——\textbf{FineLIP}，该方法在CL\textbf{IP}-风格框架中增强了跨模态文本-图像映射能力，通过引入细粒度对齐与更长文本输入。FineLIP首先扩展位置嵌入以处理更长的文本，随后动态聚合局部图像和文本标记，结合结果以确保细粒度标记间的跨模态对齐。我们利用包含长细节描述的数据集在两个任务上验证了该模型：零样本跨模态检索和文本生成图像。定量和定性实验结果表明FineLIP的有效性，超越了现有的先进方法。进一步的消融实验验证了FineLIP中关键设计元素的优势。 

---
# Interpreting Emergent Planning in Model-Free Reinforcement Learning 

**Title (ZH)**: 基于模型的自由强化学习中 Emergent 规划的解释 

**Authors**: Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2504.01871)  

**Abstract**: We present the first mechanistic evidence that model-free reinforcement learning agents can learn to plan. This is achieved by applying a methodology based on concept-based interpretability to a model-free agent in Sokoban -- a commonly used benchmark for studying planning. Specifically, we demonstrate that DRC, a generic model-free agent introduced by Guez et al. (2019), uses learned concept representations to internally formulate plans that both predict the long-term effects of actions on the environment and influence action selection. Our methodology involves: (1) probing for planning-relevant concepts, (2) investigating plan formation within the agent's representations, and (3) verifying that discovered plans (in the agent's representations) have a causal effect on the agent's behavior through interventions. We also show that the emergence of these plans coincides with the emergence of a planning-like property: the ability to benefit from additional test-time compute. Finally, we perform a qualitative analysis of the planning algorithm learned by the agent and discover a strong resemblance to parallelized bidirectional search. Our findings advance understanding of the internal mechanisms underlying planning behavior in agents, which is important given the recent trend of emergent planning and reasoning capabilities in LLMs through RL 

**Abstract (ZH)**: 无模型 reinforcement 学习代理的第一个机制证据：学习规划的能力——以 Sokoban 中的概念解释方法为例 

---
# Probabilistic Curriculum Learning for Goal-Based Reinforcement Learning 

**Title (ZH)**: 基于目标的强化学习的概率性课程学习 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.01459)  

**Abstract**: Reinforcement learning (RL) -- algorithms that teach artificial agents to interact with environments by maximising reward signals -- has achieved significant success in recent years. These successes have been facilitated by advances in algorithms (e.g., deep Q-learning, deep deterministic policy gradients, proximal policy optimisation, trust region policy optimisation, and soft actor-critic) and specialised computational resources such as GPUs and TPUs. One promising research direction involves introducing goals to allow multimodal policies, commonly through hierarchical or curriculum reinforcement learning. These methods systematically decompose complex behaviours into simpler sub-tasks, analogous to how humans progressively learn skills (e.g. we learn to run before we walk, or we learn arithmetic before calculus). However, fully automating goal creation remains an open challenge. We present a novel probabilistic curriculum learning algorithm to suggest goals for reinforcement learning agents in continuous control and navigation tasks. 

**Abstract (ZH)**: 强化学习（RL）——通过最大化奖励信号使人工代理与环境互动的算法——近年来取得了显著成功。这些成功得益于算法进步（例如，深度Q学习、深度确定性策略梯度、近似策略优化、信任区域策略优化以及柔和的行动者-批判者）和专门的计算资源（如GPU和TPU）的支持。一个有前途的研究方向是引入目标以允许多模态策略，通常通过层次化或课程强化学习实现。这些方法系统地将复杂行为分解为更简单的子任务，类似于人类逐步学习技能的过程（例如，我们先学会跑步再学会走路，或者先学会算术再学会微积分）。然而，完全自动化目标创建仍是一个开放的挑战。我们提出了一种新颖的概率课程学习算法，用于在连续控制和导航任务中建议强化学习代理的目标。 

---
# One Person, One Bot 

**Title (ZH)**: 一人一机器 

**Authors**: Liat Lavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01039)  

**Abstract**: This short paper puts forward a vision for a new democratic model enabled by the recent technological advances in agentic AI. It therefore opens with drawing a clear and concise picture of the model, and only later addresses related proposals and research directions, and concerns regarding feasibility and safety. It ends with a note on the timeliness of this idea and on optimism. The model proposed is that of assigning each citizen an AI Agent that would serve as their political delegate, enabling the return to direct democracy. The paper examines this models relation to existing research, its potential setbacks and feasibility and argues for its further development. 

**Abstract (ZH)**: 这篇短文提出了由近期agency AI技术进步所赋能的新民主模式的愿景。因此，它首先绘制了一个清晰简洁的新模式图景，随后才讨论相关提案、研究方向以及可行性和安全性的担忧。结尾处对这一想法的及时性以及保持乐观态度进行了简要说明。所提出的新模式是为每位公民分配一个AI代理，使他们能够返回直接民主。本文探讨了该模式与现有研究的关系、潜在挑战及其可行性，并呼吁进一步发展这一模式。 

---
