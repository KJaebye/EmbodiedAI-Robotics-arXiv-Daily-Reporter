# GMT: General Motion Tracking for Humanoid Whole-Body Control 

**Title (ZH)**: GMT：通用运动跟踪的人形全身控制 

**Authors**: Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14770)  

**Abstract**: The ability to track general whole-body motions in the real world is a useful way to build general-purpose humanoid robots. However, achieving this can be challenging due to the temporal and kinematic diversity of the motions, the policy's capability, and the difficulty of coordination of the upper and lower bodies. To address these issues, we propose GMT, a general and scalable motion-tracking framework that trains a single unified policy to enable humanoid robots to track diverse motions in the real world. GMT is built upon two core components: an Adaptive Sampling strategy and a Motion Mixture-of-Experts (MoE) architecture. The Adaptive Sampling automatically balances easy and difficult motions during training. The MoE ensures better specialization of different regions of the motion manifold. We show through extensive experiments in both simulation and the real world the effectiveness of GMT, achieving state-of-the-art performance across a broad spectrum of motions using a unified general policy. Videos and additional information can be found at this https URL. 

**Abstract (ZH)**: 全身运动跟踪能力是构建通用人形机器人的有用手段。然而，由于动作的时空多样性和政策能力，以及上下半身协调的难度，实现这一目标具有挑战性。为了解决这些难题，我们提出了GMT，这是一种通用且可扩展的运动跟踪框架，通过训练一个统一的策略，使类人机器人能够跟踪现实世界中的多样动作。GMT 基于两种核心组件：自适应采样策略和运动混合专家（MoE）架构。自适应采样在训练过程中自动平衡简单和困难的动作。MoE 确保了运动流形不同区域的更好专业化。通过在仿真和现实世界中的广泛实验，我们展示了GMT 的有效性，使用统一的通用策略在多种动作中实现了最先进的性能。有关视频和额外信息，请访问此链接：此 https URL。 

---
# RobotSmith: Generative Robotic Tool Design for Acquisition of Complex Manipulation Skills 

**Title (ZH)**: RobotSmith: 生成式机器人工具设计以获取复杂 manipulation 技能 

**Authors**: Chunru Lin, Haotian Yuan, Yian Wang, Xiaowen Qiu, Tsun-Hsuan Wang, Minghao Guo, Bohan Wang, Yashraj Narang, Dieter Fox, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14763)  

**Abstract**: Endowing robots with tool design abilities is critical for enabling them to solve complex manipulation tasks that would otherwise be intractable. While recent generative frameworks can automatically synthesize task settings, such as 3D scenes and reward functions, they have not yet addressed the challenge of tool-use scenarios. Simply retrieving human-designed tools might not be ideal since many tools (e.g., a rolling pin) are difficult for robotic manipulators to handle. Furthermore, existing tool design approaches either rely on predefined templates with limited parameter tuning or apply generic 3D generation methods that are not optimized for tool creation. To address these limitations, we propose RobotSmith, an automated pipeline that leverages the implicit physical knowledge embedded in vision-language models (VLMs) alongside the more accurate physics provided by physics simulations to design and use tools for robotic manipulation. Our system (1) iteratively proposes tool designs using collaborative VLM agents, (2) generates low-level robot trajectories for tool use, and (3) jointly optimizes tool geometry and usage for task performance. We evaluate our approach across a wide range of manipulation tasks involving rigid, deformable, and fluid objects. Experiments show that our method consistently outperforms strong baselines in terms of both task success rate and overall performance. Notably, our approach achieves a 50.0\% average success rate, significantly surpassing other baselines such as 3D generation (21.4%) and tool retrieval (11.1%). Finally, we deploy our system in real-world settings, demonstrating that the generated tools and their usage plans transfer effectively to physical execution, validating the practicality and generalization capabilities of our approach. 

**Abstract (ZH)**: 赋予机器人工具设计能力对于使其能够解决复杂的操作任务至关重要，这些任务否则将无法解决。虽然现有的生成框架可以自动合成任务场景，如3D场景和奖励函数，但它们尚未解决工具使用场景的挑战。简单地检索人类设计的工具可能不理想，因为许多工具（如擀面杖）对机器人操作器来说难以处理。此外，现有的工具设计方法要么依赖于预定义的模板并进行有限的参数调整，要么应用通用的3D生成方法，这些方法并未针对工具创建进行优化。为了解决这些局限性，我们提出了RobotSmith，这是一种自动管道，利用嵌入在视觉语言模型（VLMs）中的隐含物理知识以及物理模拟提供的更准确的物理模型来设计和使用工具进行机器人操作。我们的系统（1）通过协作的VLM代理迭代提出工具设计，（2）生成用于工具使用的低级机器人轨迹，并（3）共同优化工具几何形状和使用方式以提高任务性能。我们在涉及刚性、变形性和流体对象的广泛操作任务中评估了我们的方法。实验表明，我们的方法在任务成功率和整体性能上都优于强基线。特别地，我们的方法实现了50.0%的平均成功率，显著优于其他基线方法，如3D生成（21.4%）和工具检索（11.1%）。最后，我们在实际场景中部署了我们的系统，证明生成的工具及其使用计划能够有效转移到实际执行中，验证了我们方法的实际可行性和泛化能力。 

---
# Tactile Beyond Pixels: Multisensory Touch Representations for Robot Manipulation 

**Title (ZH)**: 超越像素的触觉感知：用于机器人操作的多感官触觉表示 

**Authors**: Carolina Higuera, Akash Sharma, Taosha Fan, Chaithanya Krishna Bodduluri, Byron Boots, Michael Kaess, Mike Lambeta, Tingfan Wu, Zixi Liu, Francois Robert Hogan, Mustafa Mukadam  

**Link**: [PDF](https://arxiv.org/pdf/2506.14754)  

**Abstract**: We present Sparsh-X, the first multisensory touch representations across four tactile modalities: image, audio, motion, and pressure. Trained on ~1M contact-rich interactions collected with the Digit 360 sensor, Sparsh-X captures complementary touch signals at diverse temporal and spatial scales. By leveraging self-supervised learning, Sparsh-X fuses these modalities into a unified representation that captures physical properties useful for robot manipulation tasks. We study how to effectively integrate real-world touch representations for both imitation learning and tactile adaptation of sim-trained policies, showing that Sparsh-X boosts policy success rates by 63% over an end-to-end model using tactile images and improves robustness by 90% in recovering object states from touch. Finally, we benchmark Sparsh-X ability to make inferences about physical properties, such as object-action identification, material-quantity estimation, and force estimation. Sparsh-X improves accuracy in characterizing physical properties by 48% compared to end-to-end approaches, demonstrating the advantages of multisensory pretraining for capturing features essential for dexterous manipulation. 

**Abstract (ZH)**: 我们呈现了Sparsh-X，这是一种全新的多感知触觉表示，涵盖了四种触觉模态：图像、音频、运动和压力。基于Digit 360传感器收集的约100万次接触丰富的交互数据，Sparsh-X捕捉了适用于不同时间和空间尺度的互补触觉信号。通过利用自监督学习，Sparsh-X将这些模态融合成一个统一的表示，能够捕捉到用于机器人操作任务的物理属性。我们研究了如何有效集成现实世界的触觉表示，用于模仿学习和触觉适应训练策略，结果显示Sparsh-X相较于端到端模型使用触觉图像提升了63%的策略成功率，并在从触觉恢复物体状态的鲁棒性方面提高了90%。最后，我们评估了Sparsh-X在推断物理属性如物体动作识别、材料数量估计和力估计方面的能力。与端到端方法相比，Sparsh-X在表征物理属性的准确性上提高了48%，展示了多感知预训练在捕捉对灵巧操作至关重要的特征方面的优势。 

---
# Casper: Inferring Diverse Intents for Assistive Teleoperation with Vision Language Models 

**Title (ZH)**: Casper: 基于视觉语言模型的多样化辅助遥操作意图推断 

**Authors**: Huihan Liu, Rutav Shah, Shuijing Liu, Jack Pittenger, Mingyo Seo, Yuchen Cui, Yonatan Bisk, Roberto Martín-Martín, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14727)  

**Abstract**: Assistive teleoperation, where control is shared between a human and a robot, enables efficient and intuitive human-robot collaboration in diverse and unstructured environments. A central challenge in real-world assistive teleoperation is for the robot to infer a wide range of human intentions from user control inputs and to assist users with correct actions. Existing methods are either confined to simple, predefined scenarios or restricted to task-specific data distributions at training, limiting their support for real-world assistance. We introduce Casper, an assistive teleoperation system that leverages commonsense knowledge embedded in pre-trained visual language models (VLMs) for real-time intent inference and flexible skill execution. Casper incorporates an open-world perception module for a generalized understanding of novel objects and scenes, a VLM-powered intent inference mechanism that leverages commonsense reasoning to interpret snippets of teleoperated user input, and a skill library that expands the scope of prior assistive teleoperation systems to support diverse, long-horizon mobile manipulation tasks. Extensive empirical evaluation, including human studies and system ablations, demonstrates that Casper improves task performance, reduces human cognitive load, and achieves higher user satisfaction than direct teleoperation and assistive teleoperation baselines. 

**Abstract (ZH)**: 辅助远程操作，其中控制由人类和机器人共同分享，在多样化和未结构化的环境中能够实现高效和直观的人机协作。现实世界辅助远程操作中的一个核心挑战是，机器人需要从用户的控制输入中推断出广泛的人类意图，并以正确的方式帮助用户行动。现有方法要么局限于简单的预定义场景，要么仅在训练时受限于特定的任务数据分布，限制了它们对实际辅助的支持。我们提出了Casper，一种利用预训练的视觉语言模型（VLM）嵌入的常识知识进行实时意图推理和灵活技能执行的辅助远程操作系统。Casper融合了一个开放世界的感知模块，以对新型物体和场景有概括性的理解，一个利用常识推理来解释远程操作用户输入片段的VLM驱动的意图推理机制，以及一个技能库，该技能库扩展了先前辅助远程操作系统的范围，以支持多样的长时移动操作任务。广泛的实证研究，包括人类研究和系统拆分，表明Casper提高了任务性能，减轻了人类的认知负担，并在直接远程操作和辅助远程操作基线方面获得了更高的用户满意度。 

---
# Factor-Graph-Based Passive Acoustic Navigation for Decentralized Cooperative Localization Using Bearing Elevation Depth Difference 

**Title (ZH)**: 基于因子图的被动声纳导航用于基于方位仰角深度差的分布式协同定位 

**Authors**: Kalliyan Velasco, Timothy W. McLain, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.14690)  

**Abstract**: Accurate and scalable underwater multi-agent localization remains a critical challenge due to the constraints of underwater communication. In this work, we propose a multi-agent localization framework using a factor-graph representation that incorporates bearing, elevation, and depth difference (BEDD). Our method leverages inverted ultra-short baseline (inverted-USBL) derived azimuth and elevation measurements from incoming acoustic signals and relative depth measurements to enable cooperative localization for a multi-robot team of autonomous underwater vehicles (AUVs). We validate our approach in the HoloOcean underwater simulator with a fleet of AUVs, demonstrating improved localization accuracy compared to dead reckoning. Additionally, we investigate the impact of azimuth and elevation measurement outliers, highlighting the need for robust outlier rejection techniques for acoustic signals. 

**Abstract (ZH)**: 基于bearing、elevation和depth difference的可扩展水下多agent定位框架 

---
# SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning 

**Title (ZH)**: SENIOR：基于偏好的强化学习中的高效查询选择与偏好导向探索 

**Authors**: Hexian Ni, Tao Lu, Haoyuan Hu, Yinghao Cai, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14648)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) methods provide a solution to avoid reward engineering by learning reward models based on human preferences. However, poor feedback- and sample- efficiency still remain the problems that hinder the application of PbRL. In this paper, we present a novel efficient query selection and preference-guided exploration method, called SENIOR, which could select the meaningful and easy-to-comparison behavior segment pairs to improve human feedback-efficiency and accelerate policy learning with the designed preference-guided intrinsic rewards. Our key idea is twofold: (1) We designed a Motion-Distinction-based Selection scheme (MDS). It selects segment pairs with apparent motion and different directions through kernel density estimation of states, which is more task-related and easy for human preference labeling; (2) We proposed a novel preference-guided exploration method (PGE). It encourages the exploration towards the states with high preference and low visits and continuously guides the agent achieving the valuable samples. The synergy between the two mechanisms could significantly accelerate the progress of reward and policy learning. Our experiments show that SENIOR outperforms other five existing methods in both human feedback-efficiency and policy convergence speed on six complex robot manipulation tasks from simulation and four real-worlds. 

**Abstract (ZH)**: 基于偏好强化学习的高效查询选择与偏好引导探索方法：SENIOR 

---
# Latent Action Diffusion for Cross-Embodiment Manipulation 

**Title (ZH)**: 潜在动作扩散在跨体态操控中的应用 

**Authors**: Erik Bauer, Elvis Nava, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.14608)  

**Abstract**: End-to-end learning approaches offer great potential for robotic manipulation, but their impact is constrained by data scarcity and heterogeneity across different embodiments. In particular, diverse action spaces across different end-effectors create barriers for cross-embodiment learning and skill transfer. We address this challenge through diffusion policies learned in a latent action space that unifies diverse end-effector actions. We first show that we can learn a semantically aligned latent action space for anthropomorphic robotic hands, a human hand, and a parallel jaw gripper using encoders trained with a contrastive loss. Second, we show that by using our proposed latent action space for co-training on manipulation data from different end-effectors, we can utilize a single policy for multi-robot control and obtain up to 13% improved manipulation success rates, indicating successful skill transfer despite a significant embodiment gap. Our approach using latent cross-embodiment policies presents a new method to unify different action spaces across embodiments, enabling efficient multi-robot control and data sharing across robot setups. This unified representation significantly reduces the need for extensive data collection for each new robot morphology, accelerates generalization across embodiments, and ultimately facilitates more scalable and efficient robotic learning. 

**Abstract (ZH)**: 端到端学习方法为机器人操作提供了巨大潜力，但其影响受数据稀缺性和不同实体间异质性限制。特别是，不同末端执行器的动作空间差异为跨实体学习和技能转移设置了障碍。我们通过在统一不同末端执行器动作的空间中学习扩散策略来应对这一挑战。首先，我们展示了可以使用训练有素的对比损失编解码器为类人机器人手、人的手和并联夹爪学习语义对齐的潜在动作空间。其次，我们展示了通过在不同末端执行器操作数据上使用我们提出的潜在动作空间进行协同训练，可以使用单一策略进行多机器人控制，并获得最高13%的提高的操作成功率，表明即使存在显著的实体差距，技能转移也是成功的。我们使用潜在跨实体策略的方法提供了一种新的方法，以统一不同实体的动作空间，从而实现高效的多机器人控制和跨机器人配置的数据共享。这种统一的表示大大减少了为每种新机器人形态收集大量数据的需求，加速了实体间的泛化，并最终促进了更可扩展和高效的机器人学习。 

---
# NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving 

**Title (ZH)**: NetRoller: 联通通用模型与专用模型的端到端自动驾驶接口 

**Authors**: Ren Xin, Hongji Liu, Xiaodong Mei, Wenru Liu, Maosheng Ye, Zhili Chen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.14589)  

**Abstract**: Integrating General Models (GMs) such as Large Language Models (LLMs), with Specialized Models (SMs) in autonomous driving tasks presents a promising approach to mitigating challenges in data diversity and model capacity of existing specialized driving models. However, this integration leads to problems of asynchronous systems, which arise from the distinct characteristics inherent in GMs and SMs. To tackle this challenge, we propose NetRoller, an adapter that incorporates a set of novel mechanisms to facilitate the seamless integration of GMs and specialized driving models. Specifically, our mechanisms for interfacing the asynchronous GMs and SMs are organized into three key stages. NetRoller first harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. It then applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation. At last, it employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of SMs through few-epoch fine-tuning. Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the GM. Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving. The code and models are available at this https URL . 

**Abstract (ZH)**: 将通用模型（如大型语言模型）与专门模型集成到自动驾驶任务中，以缓解现有专门驾驶模型在数据多样性和模型容量方面的挑战，这是一种有前途的方法。然而，这种集成会导致异步系统的问题，这是由于通用模型和专门模型固有的不同特性所引起的。为了解决这一挑战，我们提出NetRoller，这是一种适配器，结合了一套新型机制，以促进通用模型和专门驾驶模型的无缝集成。具体而言，NetRoller用于接口异步通用模型和专门模型的机制被组织为三个关键阶段。NetRoller首先通过早期停止机制从大型语言模型的推理过程中采集语义丰富且计算高效的表示，同时保留关键的驾驶上下文洞察并保持较低的开销。然后，它应用可学习的查询嵌入、无意义嵌入和位置层嵌入，以促进稳健且高效的跨模态转换。最后，它通过少量 epochs 的微调应用计算高效的查询转换和特征转换机制，以增强专门模型的性能。基于这三个阶段机制的制定，NetRoller使专门驾驶模型在其固有频率下运行，同时保持对通用模型的环境意识。在nuScenes数据集上的实验表明，通过NetRoller集成通用模型在规划任务中显著提高了人类相似度和安全性，同时也实现了检测和建图任务中端到端自动驾驶的明显精度提升。代码和模型可在以下网址获取。 

---
# GAMORA: A Gesture Articulated Meta Operative Robotic Arm for Hazardous Material Handling in Containment-Level Environments 

**Title (ZH)**: GAMORA：一种用于受限环境危险材料处理的 gesture articulated meta operative robotic arm 

**Authors**: Farha Abdul Wasay, Mohammed Abdul Rahman, Hania Ghouse  

**Link**: [PDF](https://arxiv.org/pdf/2506.14513)  

**Abstract**: The convergence of robotics and virtual reality (VR) has enabled safer and more efficient workflows in high-risk laboratory settings, particularly virology labs. As biohazard complexity increases, minimizing direct human exposure while maintaining precision becomes essential. We propose GAMORA (Gesture Articulated Meta Operative Robotic Arm), a novel VR-guided robotic system that enables remote execution of hazardous tasks using natural hand gestures. Unlike existing scripted automation or traditional teleoperation, GAMORA integrates the Oculus Quest 2, NVIDIA Jetson Nano, and Robot Operating System (ROS) to provide real-time immersive control, digital twin simulation, and inverse kinematics-based articulation. The system supports VR-based training and simulation while executing precision tasks in physical environments via a 3D-printed robotic arm. Inverse kinematics ensure accurate manipulation for delicate operations such as specimen handling and pipetting. The pipeline includes Unity-based 3D environment construction, real-time motion planning, and hardware-in-the-loop testing. GAMORA achieved a mean positional discrepancy of 2.2 mm (improved from 4 mm), pipetting accuracy within 0.2 mL, and repeatability of 1.2 mm across 50 trials. Integrated object detection via YOLOv8 enhances spatial awareness, while energy-efficient operation (50% reduced power output) ensures sustainable deployment. The system's digital-physical feedback loop enables safe, precise, and repeatable automation of high-risk lab tasks. GAMORA offers a scalable, immersive solution for robotic control and biosafety in biomedical research environments. 

**Abstract (ZH)**: 机器人技术和虚拟现实的融合在高风险实验室环境中，特别是病毒学实验室中，实现了更安全、更高效的 workflows。GAMORA（Gesture Articulated Meta Operative Robotic Arm）：一种基于VR的新型机器人系统，通过自然手势远程执行危险任务。 

---
# Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation? 

**Title (ZH)**: 预训练的视觉-语言嵌入能否单独引导机器人导航？ 

**Authors**: Nitesh Subedi, Adam Haroon, Shreyan Ganguly, Samuel T.K. Tetteh, Prajwal Koirala, Cody Fleming, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.14507)  

**Abstract**: Foundation models have revolutionized robotics by providing rich semantic representations without task-specific training. While many approaches integrate pretrained vision-language models (VLMs) with specialized navigation architectures, the fundamental question remains: can these pretrained embeddings alone successfully guide navigation without additional fine-tuning or specialized modules? We present a minimalist framework that decouples this question by training a behavior cloning policy directly on frozen vision-language embeddings from demonstrations collected by a privileged expert. Our approach achieves a 74% success rate in navigation to language-specified targets, compared to 100% for the state-aware expert, though requiring 3.2 times more steps on average. This performance gap reveals that pretrained embeddings effectively support basic language grounding but struggle with long-horizon planning and spatial reasoning. By providing this empirical baseline, we highlight both the capabilities and limitations of using foundation models as drop-in representations for embodied tasks, offering critical insights for robotics researchers facing practical design tradeoffs between system complexity and performance in resource-constrained scenarios. Our code is available at this https URL 

**Abstract (ZH)**: 预训练模型通过提供无需任务特定训练的丰富语义表示，彻底改变了机器人技术。尽管许多方法将预训练的视觉-语言模型（VLMs）与专门的导航架构结合使用，但基本问题仍悬而未决：这些预训练嵌入本身能否在无需附加微调或专门模块的情况下成功引导导航？我们提出了一种极简主义框架，通过直接在由特权专家收集的演示中冻结的视觉-语言嵌入上训练行为克隆策略来解答这一问题。我们的方法在导航到语言指定目标方面的成功率达到了74%，尽管平均需要多出3.2倍的步骤，这一性能差距揭示了预训练嵌入能够有效支持基本语言定位但难以应对长时规划和空间推理。通过提供这一实证基线，我们突显了使用基础模型作为即插即用表示进行体感能力的能力和局限性，为机器人研究人员在资源受限场景中面对系统复杂性和性能之间的实际设计权衡提供了关键见解。我们的代码可在以下网址获取：this https URL。 

---
# ros2 fanuc interface: Design and Evaluation of a Fanuc CRX Hardware Interface in ROS2 

**Title (ZH)**: ROS2 Fanuc 接口：ROS2 中 Fanuc CRX 硬件接口的设计与评估 

**Authors**: Paolo Franceschi, Marco Faroni, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.14487)  

**Abstract**: This paper introduces the ROS2 control and the Hardware Interface (HW) integration for the Fanuc CRX- robot family. It explains basic implementation details and communication protocols, and its integration with the Moveit2 motion planning library. We conducted a series of experiments to evaluate relevant performances in the robotics field. We tested the developed ros2_fanuc_interface for four relevant robotics cases: step response, trajectory tracking, collision avoidance integrated with Moveit2, and dynamic velocity scaling, respectively. Results show that, despite a non-negligible delay between command and feedback, the robot can track the defined path with negligible errors (if it complies with joint velocity limits), ensuring collision avoidance. Full code is open source and available at this https URL. 

**Abstract (ZH)**: ROS2控制与Hardware Interface (HW)集成在Fanuc CRX-机器人家族中的实现与研究：基于Moveit2的运动规划库集成实验 

---
# Automatic Cannulation of Femoral Vessels in a Porcine Shock Model 

**Title (ZH)**: 猪休克模型中自动股血管穿刺的研究 

**Authors**: Nico Zevallos, Cecilia G. Morales, Andrew Orekhov, Tejas Rane, Hernando Gomez, Francis X. Guyette, Michael R. Pinsky, John Galeotti, Artur Dubrawski, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2506.14467)  

**Abstract**: Rapid and reliable vascular access is critical in trauma and critical care. Central vascular catheterization enables high-volume resuscitation, hemodynamic monitoring, and advanced interventions like ECMO and REBOA. While peripheral access is common, central access is often necessary but requires specialized ultrasound-guided skills, posing challenges in prehospital settings. The complexity arises from deep target vessels and the precision needed for needle placement. Traditional techniques, like the Seldinger method, demand expertise to avoid complications. Despite its importance, ultrasound-guided central access is underutilized due to limited field expertise. While autonomous needle insertion has been explored for peripheral vessels, only semi-autonomous methods exist for femoral access. This work advances toward full automation, integrating robotic ultrasound for minimally invasive emergency procedures. Our key contribution is the successful femoral vein and artery cannulation in a porcine hemorrhagic shock model. 

**Abstract (ZH)**: rapid 和可靠的血管通路在创伤和重症 care 中至关重要。中心静脉导管化使高体积复苏、血流动力学监测及体外膜氧合（ECMO）和全身性旁路（REBOA）等高级干预成为可能。虽然外周通路较为常见，但中心通路在必要时是必需的，这需要特殊的超声引导技能，在院前环境中提出了挑战。复杂性源于深部目标血管以及针头放置所需的精度。传统技术，如西德林方法，要求有专业知识以避免并发症。尽管其重要性，由于现场专业人员有限，超声引导下中心通路的应用仍然不足。虽然外周血管的自主穿刺已有所探索，但仅存在半自主的股静脉穿刺方法。本研究朝着完全自动化方向迈进，结合了微创紧急程序中的机器人超声技术。我们的主要贡献是在一种猪实验性失血性休克模型中成功实现了股静脉和动脉穿刺。 

---
# Enhancing Object Search in Indoor Spaces via Personalized Object-factored Ontologies 

**Title (ZH)**: 基于个性化对象本体的室内空间物体搜索增强 

**Authors**: Akash Chikhalikar, Ankit A. Ravankar, Jose Victorio Salazar Luces, Yasuhisa Hirata  

**Link**: [PDF](https://arxiv.org/pdf/2506.14422)  

**Abstract**: Personalization is critical for the advancement of service robots. Robots need to develop tailored understandings of the environments they are put in. Moreover, they need to be aware of changes in the environment to facilitate long-term deployment. Long-term understanding as well as personalization is necessary to execute complex tasks like prepare dinner table or tidy my room. A precursor to such tasks is that of Object Search. Consequently, this paper focuses on locating and searching multiple objects in indoor environments. In this paper, we propose two crucial novelties. Firstly, we propose a novel framework that can enable robots to deduce Personalized Ontologies of indoor environments. Our framework consists of a personalization schema that enables the robot to tune its understanding of ontologies. Secondly, we propose an Adaptive Inferencing strategy. We integrate Dynamic Belief Updates into our approach which improves performance in multi-object search tasks. The cumulative effect of personalization and adaptive inferencing is an improved capability in long-term object search. This framework is implemented on top of a multi-layered semantic map. We conduct experiments in real environments and compare our results against various state-of-the-art (SOTA) methods to demonstrate the effectiveness of our approach. Additionally, we show that personalization can act as a catalyst to enhance the performance of SOTAs. Video Link: this https URL 

**Abstract (ZH)**: 服务机器人个性化对于其发展至关重要。机器人需要对所处环境形成定制化的理解，并且需要意识到环境变化以促进长期部署。长期理解和个性化对于执行复杂的任务，如布置餐桌或整理房间，是必要的。此类任务的前提是物体搜索。因此，本文专注于在室内环境中定位和搜索多个物体。本文提出两项关键创新。首先，我们提出了一种新型框架，使机器人能够推断出室内环境的个性化本体。该框架包括一个个性化方案，使机器人能够调整其对本体的理解。其次，我们提出了自适应推理策略。我们将动态信念更新集成到我们的方法中，这在多物体搜索任务中提高了性能。个性化与自适应推理的综合效果提升了长期物体搜索的能力。该框架建立在多层语义地图之上。我们在真实环境中进行了实验，并将我们的结果与各种最先进的方法进行比较，以展示我们方法的有效性。此外，我们展示了个性化可以作为催化剂来增强最先进的方法的性能。视频链接：this https URL 

---
# Data Driven Approach to Input Shaping for Vibration Suppression in a Flexible Robot Arm 

**Title (ZH)**: 基于数据驱动的输入成型方法用于柔性机器人臂的振动抑制 

**Authors**: Jarkko Kotaniemi, Janne Saukkoriipi, Shuai Li, Markku Suomalainen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14405)  

**Abstract**: This paper presents a simple and effective method for setting parameters for an input shaper to suppress the residual vibrations in flexible robot arms using a data-driven approach. The parameters are adaptively tuned in the workspace of the robot by interpolating previously measured data of the robot's residual vibrations. Input shaping is a simple and robust technique to generate vibration-reduced shaped commands by a convolution of an impulse sequence with the desired input command. The generated impulses create waves in the material countering the natural vibrations of the system. The method is demonstrated with a flexible 3D-printed robot arm with multiple different materials, achieving a significant reduction in the residual vibrations. 

**Abstract (ZH)**: 基于数据驱动的方法在柔性机器人手臂中抑制残余振动的参数设置方法 

---
# Barrier Method for Inequality Constrained Factor Graph Optimization with Application to Model Predictive Control 

**Title (ZH)**: 不等式约束因子图优化的屏障方法及其在模型预测控制中的应用 

**Authors**: Anas Abdelkarim, Holger Voos, Daniel Görges  

**Link**: [PDF](https://arxiv.org/pdf/2506.14341)  

**Abstract**: Factor graphs have demonstrated remarkable efficiency for robotic perception tasks, particularly in localization and mapping applications. However, their application to optimal control problems -- especially Model Predictive Control (MPC) -- has remained limited due to fundamental challenges in constraint handling. This paper presents a novel integration of the Barrier Interior Point Method (BIPM) with factor graphs, implemented as an open-source extension to the widely adopted g2o framework. Our approach introduces specialized inequality factor nodes that encode logarithmic barrier functions, thereby overcoming the quadratic-form limitations of conventional factor graph formulations. To the best of our knowledge, this is the first g2o-based implementation capable of efficiently handling both equality and inequality constraints within a unified optimization backend. We validate the method through a multi-objective adaptive cruise control application for autonomous vehicles. Benchmark comparisons with state-of-the-art constraint-handling techniques demonstrate faster convergence and improved computational efficiency. (Code repository: this https URL) 

**Abstract (ZH)**: 因子图在机器人感知任务中展现了显著的效率，特别是在定位和建图应用中。然而，由于处理约束的基本挑战，它们在最优控制问题——尤其是模型预测控制（MPC）——中的应用仍受到限制。本文提出了一种将障碍内部点方法（BIPM）与因子图相结合的新型集成方法，并作为对广泛采用的g2o框架的开源扩展实现。我们的方法引入了专门的不等式因子节点，编码对数障碍函数，从而克服了传统因子图公式化的二次形式限制。据我们所知，这是首个基于g2o框架能够高效处理等式和不等式约束的统一优化后端的实现。该方法通过自主车辆的多目标自适应巡航控制应用进行了验证。基准比较显示，与最先进的约束处理技术相比，该方法具有更快的收敛性和更好的计算效率。（代码库：this https URL） 

---
# ClutterDexGrasp: A Sim-to-Real System for General Dexterous Grasping in Cluttered Scenes 

**Title (ZH)**: ClutterDexGrasp：一种用于杂乱场景通用灵巧抓取的仿真实验系统 

**Authors**: Zeyuan Chen, Qiyang Yan, Yuanpei Chen, Tianhao Wu, Jiyao Zhang, Zihan Ding, Jinzhou Li, Yaodong Yang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14317)  

**Abstract**: Dexterous grasping in cluttered scenes presents significant challenges due to diverse object geometries, occlusions, and potential collisions. Existing methods primarily focus on single-object grasping or grasp-pose prediction without interaction, which are insufficient for complex, cluttered scenes. Recent vision-language-action models offer a potential solution but require extensive real-world demonstrations, making them costly and difficult to scale. To address these limitations, we revisit the sim-to-real transfer pipeline and develop key techniques that enable zero-shot deployment in reality while maintaining robust generalization. We propose ClutterDexGrasp, a two-stage teacher-student framework for closed-loop target-oriented dexterous grasping in cluttered scenes. The framework features a teacher policy trained in simulation using clutter density curriculum learning, incorporating both a novel geometry and spatially-embedded scene representation and a comprehensive safety curriculum, enabling general, dynamic, and safe grasping behaviors. Through imitation learning, we distill the teacher's knowledge into a student 3D diffusion policy (DP3) that operates on partial point cloud observations. To the best of our knowledge, this represents the first zero-shot sim-to-real closed-loop system for target-oriented dexterous grasping in cluttered scenes, demonstrating robust performance across diverse objects and layouts. More details and videos are available at this https URL. 

**Abstract (ZH)**: 在杂乱场景中实现灵巧抓取面临着显著挑战，由于对象几何形状多样、遮挡以及潜在碰撞。现有方法主要集中在单对象抓取或抓取姿态预测，而不涉及交互，这不足以应对复杂、杂乱的场景。最近的视觉-语言-动作模型提供了一种潜在的解决方案，但需要大量的现实世界演示，使其成本高昂且难以扩展。为了解决这些局限性，我们重新审视了从仿真到现实的转移管道，并开发了关键技术，以实现零样本部署并保持稳健的泛化能力。我们提出了一种两阶段教师-学生框架ClutterDexGrasp，用于杂乱场景中的闭环目标导向灵巧抓取。该框架采用通过杂乱密度课程学习在仿真中训练的教师策略，结合新颖的几何和空间嵌入场景表示以及全面的安全课程，从而实现通用、动态且安全的抓取行为。通过模仿学习，我们将教师的知识提炼为适用于部分点云观察操作的3D扩散策略（DP3）。据我们所知，这是第一个零样本从仿真到现实的闭环系统，用于杂乱场景中的目标导向灵巧抓取，展示了在不同类型物体和布局上的稳健性能。更多详细信息和视频请访问：this https URL。 

---
# Socially Aware Robot Crowd Navigation via Online Uncertainty-Driven Risk Adaptation 

**Title (ZH)**: 基于在线不确定性驱动风险适应的具有社会意识的机器人人群导航 

**Authors**: Zhirui Sun, Xingrong Diao, Yao Wang, Bi-Ke Zhu, Jiankun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14305)  

**Abstract**: Navigation in human-robot shared crowded environments remains challenging, as robots are expected to move efficiently while respecting human motion conventions. However, many existing approaches emphasize safety or efficiency while overlooking social awareness. This article proposes Learning-Risk Model Predictive Control (LR-MPC), a data-driven navigation algorithm that balances efficiency, safety, and social awareness. LR-MPC consists of two phases: an offline risk learning phase, where a Probabilistic Ensemble Neural Network (PENN) is trained using risk data from a heuristic MPC-based baseline (HR-MPC), and an online adaptive inference phase, where local waypoints are sampled and globally guided by a Multi-RRT planner. Each candidate waypoint is evaluated for risk by PENN, and predictions are filtered using epistemic and aleatoric uncertainty to ensure robust decision-making. The safest waypoint is selected as the MPC input for real-time navigation. Extensive experiments demonstrate that LR-MPC outperforms baseline methods in success rate and social awareness, enabling robots to navigate complex crowds with high adaptability and low disruption. A website about this work is available at this https URL. 

**Abstract (ZH)**: 人类与机器人共享拥挤环境中的导航仍旧具有挑战性，机器人需要在遵循人类运动规范的同时高效移动。然而，许多现有方法侧重于安全或效率，而忽视了社会意识。本文提出了一种基于数据的导航算法——Learning-Risk Model Predictive Control (LR-MPC)，该算法能够平衡效率、安全与社会意识。LR-MPC 包含两个阶段：离线风险学习阶段，使用基于启发式 MPC 的 baseline (HR-MPC) 的风险数据训练 Probabilistic Ensemble Neural Network (PENN)；在线自适应推理阶段，利用 Multi-RRT 规划器进行局部航点采样并全局引导。每个候选航点的风险由 PENN 评估，通过先验不确定性与偶然不确定性筛选预测，以实现鲁棒决策。选出最安全的航点作为实时导航的 MPC 输入。实验结果表明，LR-MPC 在成功率与社会意识方面优于基线方法，使机器人能够在复杂人群中具有高度适应性和低干扰地导航。该项目的网站可访问此 <https://> 链接。 

---
# Uncertainty-Driven Radar-Inertial Fusion for Instantaneous 3D Ego-Velocity Estimation 

**Title (ZH)**: 基于不确定性驱动的雷达-惯导融合的即时三维 ego 速度 estimation 

**Authors**: Prashant Kumar Rai, Elham Kowsari, Nataliya Strokina, Reza Ghabcheloo  

**Link**: [PDF](https://arxiv.org/pdf/2506.14294)  

**Abstract**: We present a method for estimating ego-velocity in autonomous navigation by integrating high-resolution imaging radar with an inertial measurement unit. The proposed approach addresses the limitations of traditional radar-based ego-motion estimation techniques by employing a neural network to process complex-valued raw radar data and estimate instantaneous linear ego-velocity along with its associated uncertainty. This uncertainty-aware velocity estimate is then integrated with inertial measurement unit data using an Extended Kalman Filter. The filter leverages the network-predicted uncertainty to refine the inertial sensor's noise and bias parameters, improving the overall robustness and accuracy of the ego-motion estimation. We evaluated the proposed method on the publicly available ColoRadar dataset. Our approach achieves significantly lower error compared to the closest publicly available method and also outperforms both instantaneous and scan matching-based techniques. 

**Abstract (ZH)**: 我们提出了一种通过集成高分辨率成像雷达和惯性测量单元来估计自主导航中自我速度的方法。该提出的方案通过运用神经网络处理复杂的雷达原始数据，并同时估计瞬时线性自我速度及其相关的不确定性，解决了传统雷达基自我运动估计技术的局限性。然后，使用扩展卡尔曼滤波器将该不确定性意识下的速度估计值与惯性测量单元数据结合。滤波器利用网络预测的不确定性来细化惯性传感器的噪声和偏置参数，从而提高整体的鲁棒性和精度。我们在公开可用的ColoRadar数据集上评估了所提出的方法。我们的方法在误差方面显著低于最接近的公开可用方法，并且在瞬时速度估计和扫描匹配技术方面也表现出更好的性能。 

---
# Steering Robots with Inference-Time Interactions 

**Title (ZH)**: 在推理时进行交互的机器人操控 

**Authors**: Yanwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14287)  

**Abstract**: Imitation learning has driven the development of generalist policies capable of autonomously solving multiple tasks. However, when a pretrained policy makes errors during deployment, there are limited mechanisms for users to correct its behavior. While collecting additional data for finetuning can address such issues, doing so for each downstream use case is inefficient at deployment. My research proposes an alternative: keeping pretrained policies frozen as a fixed skill repertoire while allowing user interactions to guide behavior generation toward user preferences at inference time. By making pretrained policies steerable, users can help correct policy errors when the model struggles to generalize-without needing to finetune the policy. Specifically, I propose (1) inference-time steering, which leverages user interactions to switch between discrete skills, and (2) task and motion imitation, which enables user interactions to edit continuous motions while satisfying task constraints defined by discrete symbolic plans. These frameworks correct misaligned policy predictions without requiring additional training, maximizing the utility of pretrained models while achieving inference-time user objectives. 

**Abstract (ZH)**: 模仿学习推动了通用政策的发展，使政策能够自主解决多种任务。然而，当预训练政策在部署过程中出现错误时，用户缺乏有效的机制来纠正其行为。虽然为微调收集额外数据可以解决此类问题，但在部署中为每个下游用例这样做是低效的。我的研究提出了一种替代方案：保持预训练政策冻结作为固定的技能集合，同时允许用户交互在推断时引导行为生成以满足用户偏好。通过使预训练政策可调控，用户可以在模型难以泛化时帮助纠正政策错误，而无需微调政策。具体而言，我提出了（1）推断时调控，利用用户交互在离散技能之间切换，以及（2）任务和运动模仿，允许用户交互编辑连续动作以满足由离散符号计划定义的任务约束。这些框架可以在不需额外训练的情况下纠正不一致的政策预测，最大化预训练模型的利用率并实现推断时的用户目标。 

---
# Whole-Body Control Framework for Humanoid Robots with Heavy Limbs: A Model-Based Approach 

**Title (ZH)**: 重肢人形机器人全身控制框架：基于模型的方法 

**Authors**: Tianlin Zhang, Linzhu Yue, Hongbo Zhang, Lingwei Zhang, Xuanqi Zeng, Zhitao Song, Yun-Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14278)  

**Abstract**: Humanoid robots often face significant balance issues due to the motion of their heavy limbs. These challenges are particularly pronounced when attempting dynamic motion or operating in environments with irregular terrain. To address this challenge, this manuscript proposes a whole-body control framework for humanoid robots with heavy limbs, using a model-based approach that combines a kino-dynamics planner and a hierarchical optimization problem. The kino-dynamics planner is designed as a model predictive control (MPC) scheme to account for the impact of heavy limbs on mass and inertia distribution. By simplifying the robot's system dynamics and constraints, the planner enables real-time planning of motion and contact forces. The hierarchical optimization problem is formulated using Hierarchical Quadratic Programming (HQP) to minimize limb control errors and ensure compliance with the policy generated by the kino-dynamics planner. Experimental validation of the proposed framework demonstrates its effectiveness. The humanoid robot with heavy limbs controlled by the proposed framework can achieve dynamic walking speeds of up to 1.2~m/s, respond to external disturbances of up to 60~N, and maintain balance on challenging terrains such as uneven surfaces, and outdoor environments. 

**Abstract (ZH)**: 具有重肢的人形机器人全身控制框架：基于模型的方法 

---
# Public Acceptance of Cybernetic Avatars in the service sector: Evidence from a Large-Scale Survey in Dubai 

**Title (ZH)**: 公共领域中网络化 avatar 的接受度：来自迪拜大规模调查的证据 

**Authors**: Laura Aymerich-Franch, Tarek Taha, Takahiro Miyashita, Hiroko Kamide, Hiroshi Ishiguro, Paolo Dario  

**Link**: [PDF](https://arxiv.org/pdf/2506.14268)  

**Abstract**: Cybernetic avatars are hybrid interaction robots or digital representations that combine autonomous capabilities with teleoperated control. This study investigates the acceptance of cybernetic avatars in the highly multicultural society of Dubai, with particular emphasis on robotic avatars for customer service. Specifically, we explore how acceptance varies as a function of robot appearance (e.g., android, robotic-looking, cartoonish), deployment settings (e.g., shopping malls, hotels, hospitals), and functional tasks (e.g., providing information, patrolling). To this end, we conducted a large-scale survey with over 1,000 participants. Overall, cybernetic avatars received a high level of acceptance, with physical robot avatars receiving higher acceptance than digital avatars. In terms of appearance, robot avatars with a highly anthropomorphic robotic appearance were the most accepted, followed by cartoonish designs and androids. Animal-like appearances received the lowest level of acceptance. Among the tasks, providing information and guidance was rated as the most valued. Shopping malls, airports, public transport stations, and museums were the settings with the highest acceptance, whereas healthcare-related spaces received lower levels of support. An analysis by community cluster revealed among others that Emirati respondents showed significantly greater acceptance of android appearances compared to the overall sample, while participants from the 'Other Asia' cluster were significantly more accepting of cartoonish appearances. Our study underscores the importance of incorporating citizen feedback into the design and deployment of cybernetic avatars from the early stages to enhance acceptance of this technology in society. 

**Abstract (ZH)**: 基于控制论的化身在接受度研究：以迪拜的多功能机器人化身在客户服务中的应用为例 

---
# Robust Adaptive Time-Varying Control Barrier Function with Application to Robotic Surface Treatment 

**Title (ZH)**: 鲁棒自适应时变控制屏障函数及其在机器人表面处理中的应用 

**Authors**: Yitaek Kim, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2506.14249)  

**Abstract**: Set invariance techniques such as control barrier functions (CBFs) can be used to enforce time-varying constraints such as keeping a safe distance from dynamic objects. However, existing methods for enforcing time-varying constraints often overlook model uncertainties. To address this issue, this paper proposes a CBFs-based robust adaptive controller design endowing time-varying constraints while considering parametric uncertainty and additive disturbances. To this end, we first leverage Robust adaptive Control Barrier Functions (RaCBFs) to handle model uncertainty, along with the concept of Input-to-State Safety (ISSf) to ensure robustness towards input disturbances. Furthermore, to alleviate the inherent conservatism in robustness, we also incorporate a set membership identification scheme. We demonstrate the proposed method on robotic surface treatment that requires time-varying force bounds to ensure uniform quality, in numerical simulation and real robotic setup, showing that the quality is formally guaranteed within an acceptable range. 

**Abstract (ZH)**: 基于控制障碍函数的鲁棒自适应控制器设计：同时考虑时间varying约束、参数不确定性及增广扰动 

---
# Narrate2Nav: Real-Time Visual Navigation with Implicit Language Reasoning in Human-Centric Environments 

**Title (ZH)**: Narrate2Nav：以人为本环境中实时视觉导航与隐式语言推理 

**Authors**: Amirreza Payandeh, Anuj Pokhrel, Daeun Song, Marcos Zampieri, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.14233)  

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated potential in enhancing mobile robot navigation in human-centric environments by understanding contextual cues, human intentions, and social dynamics while exhibiting reasoning capabilities. However, their computational complexity and limited sensitivity to continuous numerical data impede real-time performance and precise motion control. To this end, we propose Narrate2Nav, a novel real-time vision-action model that leverages a novel self-supervised learning framework based on the Barlow Twins redundancy reduction loss to embed implicit natural language reasoning, social cues, and human intentions within a visual encoder-enabling reasoning in the model's latent space rather than token space. The model combines RGB inputs, motion commands, and textual signals of scene context during training to bridge from robot observations to low-level motion commands for short-horizon point-goal navigation during deployment. Extensive evaluation of Narrate2Nav across various challenging scenarios in both offline unseen dataset and real-world experiments demonstrates an overall improvement of 52.94 percent and 41.67 percent, respectively, over the next best baseline. Additionally, qualitative comparative analysis of Narrate2Nav's visual encoder attention map against four other baselines demonstrates enhanced attention to navigation-critical scene elements, underscoring its effectiveness in human-centric navigation tasks. 

**Abstract (ZH)**: Narrate2Nav：一种基于Barlow Twins冗余减少损失的新型实时视觉-行动模型，用于人类中心环境中的移动机器人导航 

---
# Pose State Perception of Interventional Robot for Cardio-cerebrovascular Procedures 

**Title (ZH)**: 介入机器人在心血管和脑血管手术中的姿态状态感知 

**Authors**: Shunhan Ji, Yanxi Chen, Zhongyu Yang, Quan Zhang, Xiaohang Nie, Jingqian Sun, Yichao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14201)  

**Abstract**: In response to the increasing demand for cardiocerebrovascular interventional surgeries, precise control of interventional robots has become increasingly important. Within these complex vascular scenarios, the accurate and reliable perception of the pose state for interventional robots is particularly crucial. This paper presents a novel vision-based approach without the need of additional sensors or markers. The core of this paper's method consists of a three-part framework: firstly, a dual-head multitask U-Net model for simultaneous vessel segment and interventional robot detection; secondly, an advanced algorithm for skeleton extraction and optimization; and finally, a comprehensive pose state perception system based on geometric features is implemented to accurately identify the robot's pose state and provide strategies for subsequent control. The experimental results demonstrate the proposed method's high reliability and accuracy in trajectory tracking and pose state perception. 

**Abstract (ZH)**: 响应心脏血管介入手术需求的增加，介入机器人精确控制变得日益重要。在这些复杂的血管场景中，介入机器人的姿态状态准确可靠感知尤为关键。本文提出了一种无需额外传感器或标记的新型视觉方法。该方法的核心由三部分框架组成：首先，一种双重头多任务U-Net模型进行血管段和介入机器人的同时检测；其次，一种先进的骨架提取和优化算法；最后，基于几何特征实现全面的姿态状态感知系统，以准确识别机器人的姿态状态并为后续控制提供策略。实验结果表明，所提出的方法在轨迹跟踪和姿态状态感知方面具有高可靠性和准确性。 

---
# AMPLIFY: Actionless Motion Priors for Robot Learning from Videos 

**Title (ZH)**: AMPLIFY：来自视频的无动作先验机器人学习 

**Authors**: Jeremy A. Collins, Loránd Cheng, Kunal Aneja, Albert Wilcox, Benjamin Joffe, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.14198)  

**Abstract**: Action-labeled data for robotics is scarce and expensive, limiting the generalization of learned policies. In contrast, vast amounts of action-free video data are readily available, but translating these observations into effective policies remains a challenge. We introduce AMPLIFY, a novel framework that leverages large-scale video data by encoding visual dynamics into compact, discrete motion tokens derived from keypoint trajectories. Our modular approach separates visual motion prediction from action inference, decoupling the challenges of learning what motion defines a task from how robots can perform it. We train a forward dynamics model on abundant action-free videos and an inverse dynamics model on a limited set of action-labeled examples, allowing for independent scaling. Extensive evaluations demonstrate that the learned dynamics are both accurate, achieving up to 3.7x better MSE and over 2.5x better pixel prediction accuracy compared to prior approaches, and broadly useful. In downstream policy learning, our dynamics predictions enable a 1.2-2.2x improvement in low-data regimes, a 1.4x average improvement by learning from action-free human videos, and the first generalization to LIBERO tasks from zero in-distribution action data. Beyond robotic control, we find the dynamics learned by AMPLIFY to be a versatile latent world model, enhancing video prediction quality. Our results present a novel paradigm leveraging heterogeneous data sources to build efficient, generalizable world models. More information can be found at this https URL. 

**Abstract (ZH)**: 基于动作标记的数据稀缺且昂贵，限制了所学策略的泛化能力。相比之下，海量的无动作视频数据唾手可得，但将这些观察转化为有效的策略仍然是一个挑战。我们提出了AMPLIFY这一新型框架，通过将视觉动态编码为源自关键点轨迹的紧凑离散运动标记，利用大规模视频数据。我们的模块化方法将视觉运动预测与动作推理分离，解耦了学习定义任务的运动是什么的挑战与机器人如何执行这些运动的挑战。我们在一个丰富的无动作视频数据集上训练前向动力学模型，在有限的动作标记样本集上训练反向动力学模型，从而实现独立扩展。广泛的研究表明，所学的动力学不仅准确，相比之前的方法，在均方误差和像素预测准确性上分别提高了3.7倍和超过2.5倍，而且具有普遍适用性。在下游策略学习中，我们的动力学预测在数据稀缺的情况下使性能提高了1.2至2.2倍，在从无动作的人类视频中学习时平均提高了1.4倍，并且实现了来自无分布动作数据的LIBERO任务的首次泛化。超越机器人控制，我们发现AMPLIFY学习的动力学是一个多功能的潜在世界模型，能够提高视频预测质量。我们的结果展示了利用异构数据源构建高效且通用的世界模型的新范式。更多详情请参见此处：this https URL。 

---
# Hard Contacts with Soft Gradients: Refining Differentiable Simulators for Learning and Control 

**Title (ZH)**: 具有软渐变的硬接触：可微模拟器的细化学习与控制 

**Authors**: Anselm Paulus, A. René Geist, Pierre Schumacher, Vít Musil, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2506.14186)  

**Abstract**: Contact forces pose a major challenge for gradient-based optimization of robot dynamics as they introduce jumps in the system's velocities. Penalty-based simulators, such as MuJoCo, simplify gradient computation by softening the contact forces. However, realistically simulating hard contacts requires very stiff contact settings, which leads to incorrect gradients when using automatic differentiation. On the other hand, using non-stiff settings strongly increases the sim-to-real gap. We analyze the contact computation of penalty-based simulators to identify the causes of gradient errors. Then, we propose DiffMJX, which combines adaptive integration with MuJoCo XLA, to notably improve gradient quality in the presence of hard contacts. Finally, we address a key limitation of contact gradients: they vanish when objects do not touch. To overcome this, we introduce Contacts From Distance (CFD), a mechanism that enables the simulator to generate informative contact gradients even before objects are in contact. To preserve physical realism, we apply CFD only in the backward pass using a straight-through trick, allowing us to compute useful gradients without modifying the forward simulation. 

**Abstract (ZH)**: 基于接触力计算的误差分析及改进：DiffMJX方法在硬接触下提高梯度质量 

---
# Non-Overlap-Aware Egocentric Pose Estimation for Collaborative Perception in Connected Autonomy 

**Title (ZH)**: 基于非重叠感知的自适身影姿估计在连接自主中的协作感知 

**Authors**: Hong Huang, Dongkuan Xu, Hao Zhang, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.14180)  

**Abstract**: Egocentric pose estimation is a fundamental capability for multi-robot collaborative perception in connected autonomy, such as connected autonomous vehicles. During multi-robot operations, a robot needs to know the relative pose between itself and its teammates with respect to its own coordinates. However, different robots usually observe completely different views that contains similar objects, which leads to wrong pose estimation. In addition, it is unrealistic to allow robots to share their raw observations to detect overlap due to the limited communication bandwidth constraint. In this paper, we introduce a novel method for Non-Overlap-Aware Egocentric Pose Estimation (NOPE), which performs egocentric pose estimation in a multi-robot team while identifying the non-overlap views and satifying the communication bandwidth constraint. NOPE is built upon an unified hierarchical learning framework that integrates two levels of robot learning: (1) high-level deep graph matching for correspondence identification, which allows to identify if two views are overlapping or not, (2) low-level position-aware cross-attention graph learning for egocentric pose estimation. To evaluate NOPE, we conduct extensive experiments in both high-fidelity simulation and real-world scenarios. Experimental results have demonstrated that NOPE enables the novel capability for non-overlapping-aware egocentric pose estimation and achieves state-of-art performance compared with the existing methods. Our project page at this https URL. 

**Abstract (ZH)**: 自感知非重叠意识姿态估计（NOPE）：一种满足通信带宽约束的多机器人协作感知方法 

---
# TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping 

**Title (ZH)**: TACS-图：考虑可达性的一致场景图，用于地面机器人室内定位与建图 

**Authors**: Jeewon Kim, Minho Oh, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2506.14178)  

**Abstract**: Scene graphs have emerged as a powerful tool for robots, providing a structured representation of spatial and semantic relationships for advanced task planning. Despite their potential, conventional 3D indoor scene graphs face critical limitations, particularly under- and over-segmentation of room layers in structurally complex environments. Under-segmentation misclassifies non-traversable areas as part of a room, often in open spaces, while over-segmentation fragments a single room into overlapping segments in complex environments. These issues stem from naive voxel-based map representations that rely solely on geometric proximity, disregarding the structural constraints of traversable spaces and resulting in inconsistent room layers within scene graphs. To the best of our knowledge, this work is the first to tackle segmentation inconsistency as a challenge and address it with Traversability-Aware Consistent Scene Graphs (TACS-Graphs), a novel framework that integrates ground robot traversability with room segmentation. By leveraging traversability as a key factor in defining room boundaries, the proposed method achieves a more semantically meaningful and topologically coherent segmentation, effectively mitigating the inaccuracies of voxel-based scene graph approaches in complex environments. Furthermore, the enhanced segmentation consistency improves loop closure detection efficiency in the proposed Consistent Scene Graph-leveraging Loop Closure Detection (CoSG-LCD) leading to higher pose estimation accuracy. Experimental results confirm that the proposed approach outperforms state-of-the-art methods in terms of scene graph consistency and pose graph optimization performance. 

**Abstract (ZH)**: 基于可通行性一致场景图的时空结构表示与任务规划 

---
# Lasso Gripper: A String Shooting-Retracting Mechanism for Shape-Adaptive Grasping 

**Title (ZH)**: Lasso夹爪：一种形状自适应抓取的绳射收回机制 

**Authors**: Qiyuan Qiao, Yu Wang, Xiyu Fan, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14163)  

**Abstract**: Handling oversized, variable-shaped, or delicate objects in transportation, grasping tasks is extremely challenging, mainly due to the limitations of the gripper's shape and size. This paper proposes a novel gripper, Lasso Gripper. Inspired by traditional tools like the lasso and the uurga, Lasso Gripper captures objects by launching and retracting a string. Contrary to antipodal grippers, which concentrate force on a limited area, Lasso Gripper applies uniform pressure along the length of the string for a more gentle grasp. The gripper is controlled by four motors-two for launching the string inward and two for launching it outward. By adjusting motor speeds, the size of the string loop can be tuned to accommodate objects of varying sizes, eliminating the limitations imposed by the maximum gripper separation distance. To address the issue of string tangling during rapid retraction, a specialized mechanism was incorporated. Additionally, a dynamic model was developed to estimate the string's curve, providing a foundation for the kinematic analysis of the workspace. In grasping experiments, Lasso Gripper, mounted on a robotic arm, successfully captured and transported a range of objects, including bull and horse figures as well as delicate vegetables. The demonstration video is available here: this https URL. 

**Abstract (ZH)**: 处理运输过程中尺寸过大、形状变化或易碎物体的抓取任务极为挑战，主要由于 gripper 的形状和尺寸限制。本文提出了一种新的 gripper，Lasso Gripper。受传统的牛仔绳圈和uurga工具的启发，Lasso Gripper 通过发射和回收一根绳子来捕捉物体。与集中力作用在有限区域的对称 gripper 不同，Lasso Gripper 在绳子的长度上均匀施压，以实现更温和的抓取。该 gripper 由四个电机控制——两个用于向内发射绳子，两个用于向外发射。通过调整电机速度，可以调整绳子环的大小，以适应不同尺寸的物体，从而消除 gripper 分离距离最大值的限制。为解决快速回收过程中绳子缠绕的问题，引入了一种专门的机制。此外，还开发了一个动态模型来估算绳子的曲线，为工作空间的运动学分析提供基础。在抓取实验中，安装在机器人臂上的 Lasso Gripper 成功捕捉并运输了包括牛仔和马具模型以及易碎蔬菜在内的多种物体。演示视频请参阅：this https URL。 

---
# GAF: Gaussian Action Field as a Dvnamic World Model for Robotic Mlanipulation 

**Title (ZH)**: GAF：高斯动作场作为动态世界模型的机器人操作方法 

**Authors**: Ying Chai, Litao Deng, Ruizhi Shao, Jiajun Zhang, Liangjun Xing, Hongwen Zhang, Yebin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14135)  

**Abstract**: Accurate action inference is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we propose a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing simultaneous modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF supports three key query types: reconstruction of the current scene, prediction of future frames, and estimation of initial action via robot motion. Furthermore, the high-quality current and future frames generated by GAF facilitate manipulation action refinement through a GAF-guided diffusion model. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average success rate in robotic manipulation tasks by 10.33% over state-of-the-art methods. Project page: this http URL 

**Abstract (ZH)**: 基于运动感知四维表示的直接动作推理框架对于基于视觉的机器人操作至关重要。现有的方法通常遵循视觉到动作（V-A）范式，直接从视觉输入预测动作，或遵循视觉到三维再到动作（V-3D-A）范式，利用中间的三维表示。然而，这些方法由于操作场景的复杂性和动态性往往难以实现准确的动作推断。在本文中，我们提出了一种V-4D-A框架，通过高斯动作场（GAF）从运动感知的四维表示中直接进行动作推理。GAF通过引入可学习的运动属性扩展了三维高斯点绘（3DGS），从而能够同时建模动态场景和操作动作。为了学习随时间变化的场景几何结构和动作感知的机器人运动，GAF支持三种关键查询类型：当前场景重建、未来帧预测以及通过机器人运动估计初始动作。此外，GAF生成的高质量当前和未来帧通过GAF指导的扩散模型促进了操作动作的细化。大量实验表明，GAF在重建质量上取得了显著改进，PSNR提高了11.5385 dB，LPIPS降低了0.5574，同时将最先进的方法在机器人操作任务中的平均成功率提高了10.33%。项目页面：这个链接URL。 

---
# Haptic-Based User Authentication for Tele-robotic System 

**Title (ZH)**: 基于触觉的用户认证方法用于远程机器人系统 

**Authors**: Rongyu Yu, Kan Chen, Zeyu Deng, Chen Wang, Burak Kizilkaya, Liying Emma Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14116)  

**Abstract**: Tele-operated robots rely on real-time user behavior mapping for remote tasks, but ensuring secure authentication remains a challenge. Traditional methods, such as passwords and static biometrics, are vulnerable to spoofing and replay attacks, particularly in high-stakes, continuous interactions. This paper presents a novel anti-spoofing and anti-replay authentication approach that leverages distinctive user behavioral features extracted from haptic feedback during human-robot interactions. To evaluate our authentication approach, we collected a time-series force feedback dataset from 15 participants performing seven distinct tasks. We then developed a transformer-based deep learning model to extract temporal features from the haptic signals. By analyzing user-specific force dynamics, our method achieves over 90 percent accuracy in both user identification and task classification, demonstrating its potential for enhancing access control and identity assurance in tele-robotic systems. 

**Abstract (ZH)**: 遥操作机器人依赖于远程任务的实时用户行为映射，但确保安全认证仍是一项挑战。传统方法，如密码和静态生物特征识别，容易受到欺骗和重放攻击，特别是在高风险的连续交互中。本文提出了一种新颖的防欺骗和防重放认证方法，该方法利用从人类与机器人交互中提取的独特用户行为特征。为了评估我们的认证方法，我们从15名参与者完成的七个不同任务中收集了时间序列力反馈数据集。然后，我们开发了一个基于Transformer的深度学习模型来从触觉信号中提取时间特征。通过分析用户的特定力动力学，我们的方法在用户识别和任务分类中的准确率均超过90%，表明其在提高远程机器人系统访问控制和身份验证方面的潜力。 

---
# A Hierarchical Test Platform for Vision Language Model (VLM)-Integrated Real-World Autonomous Driving 

**Title (ZH)**: 面向视觉语言模型(VLM)整合的自动驾驶现实世界场景分级测试平台 

**Authors**: Yupeng Zhou, Can Cui, Juntong Peng, Zichong Yang, Juanwu Lu, Jitesh H Panchal, Bin Yao, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14100)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated notable promise in autonomous driving by offering the potential for multimodal reasoning through pretraining on extensive image-text pairs. However, adapting these models from broad web-scale data to the safety-critical context of driving presents a significant challenge, commonly referred to as domain shift. Existing simulation-based and dataset-driven evaluation methods, although valuable, often fail to capture the full complexity of real-world scenarios and cannot easily accommodate repeatable closed-loop testing with flexible scenario manipulation. In this paper, we introduce a hierarchical real-world test platform specifically designed to evaluate VLM-integrated autonomous driving systems. Our approach includes a modular, low-latency on-vehicle middleware that allows seamless incorporation of various VLMs, a clearly separated perception-planning-control architecture that can accommodate both VLM-based and conventional modules, and a configurable suite of real-world testing scenarios on a closed track that facilitates controlled yet authentic evaluations. We demonstrate the effectiveness of the proposed platform`s testing and evaluation ability with a case study involving a VLM-enabled autonomous vehicle, highlighting how our test framework supports robust experimentation under diverse conditions. 

**Abstract (ZH)**: Vision-Language模型（VLMs）已经在自主驾驶领域展现了显著的潜力，通过在大量的图像-文本对上进行预训练，提供了多模态推理的可能性。然而，将这些模型从广泛的网络数据调整到驾驶这一至关安全的上下文环境中，面临一个重大的挑战，通常被称为领域转换。现有的基于模拟和数据驱动的评估方法虽然很有价值，但往往无法捕捉现实世界场景的全部复杂性，并且难以容纳灵活的场景操作和重复的闭环测试。本文介绍了一种分层的实际测试平台，专门用于评估VLM集成的自主驾驶系统。我们的方法包括一个模块化、低延迟的车载中间件，可以无缝集成各种VLM；一个清晰分离的感知-规划-控制架构，可以兼容基于VLM和传统的模块；以及一个在封闭赛道上可配置的真实世界测试场景套件，可以实现受控且真实的评估。通过一个基于VLM的自动驾驶车辆案例研究，我们展示了所提出的平台测试和评估能力的有效性，展示了我们的测试框架如何支持在多种条件下进行稳健的实验。 

---
# ReLCP: Scalable Complementarity-Based Collision Resolution for Smooth Rigid Bodies 

**Title (ZH)**: ReLCP: 基于互补性平滑刚体碰撞分辨率的可扩展方法 

**Authors**: Bryce Palmer, Hasan Metin Aktulga, Tong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.14097)  

**Abstract**: We present a complementarity-based collision resolution algorithm for smooth, non-spherical, rigid bodies. Unlike discrete surface representation approaches, which approximate surfaces using discrete elements (e.g., tessellations or sub-spheres) with constraints between nearby faces, edges, nodes, or sub-objects, our algorithm solves a recursively generated linear complementarity problem (ReLCP) to adaptively identify potential collision locations during the collision resolution procedure. Despite adaptively and in contrast to Newton-esque schemes, we prove conditions under which the resulting solution exists and the center of mass translational and rotational dynamics are unique. Our ReLCP also converges to classical LCP-based collision resolution for sufficiently small timesteps. Because increasing the surface resolution in discrete representation methods necessitates subdividing geometry into finer elements -- leading to a super-linear increase in the number of collision constraints -- these approaches scale poorly with increased surface resolution. In contrast, our adaptive ReLCP framework begins with a single constraint per pair of nearby bodies and introduces new constraints only when unconstrained motion would lead to overlap, circumventing the oversampling required by discrete methods. By requiring one to two orders of magnitude fewer collision constraints to achieve the same surface resolution, we observe 10-100x speedup in densely packed applications. We validate our ReLCP method against multisphere and single-constraint methods, comparing convergence in a two-ellipsoid collision test, scalability and performance in a compacting ellipsoid suspension and growing bacterial colony, and stability in a taut chainmail network, highlighting our ability to achieve high-fidelity surface representations without suffering from poor scalability or artificial surface roughness. 

**Abstract (ZH)**: 基于互补性原则的光滑非球形刚体碰撞解决算法 

---
# A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting 

**Title (ZH)**: 部分遮挡物抓取的点云完成方法及其在草莓机器人采摘中的应用 

**Authors**: Ali Abouzeid, Malak Mansour, Chengsong Hu, Dezhen Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.14066)  

**Abstract**: In robotic fruit picking applications, managing object occlusion in unstructured settings poses a substantial challenge for designing grasping algorithms. Using strawberry harvesting as a case study, we present an end-to-end framework for effective object detection, segmentation, and grasp planning to tackle this issue caused by partially occluded objects. Our strategy begins with point cloud denoising and segmentation to accurately locate fruits. To compensate for incomplete scans due to occlusion, we apply a point cloud completion model to create a dense 3D reconstruction of the strawberries. The target selection focuses on ripe strawberries while categorizing others as obstacles, followed by converting the refined point cloud into an occupancy map for collision-aware motion planning. Our experimental results demonstrate high shape reconstruction accuracy, with the lowest Chamfer Distance compared to state-of-the-art methods with 1.10 mm, and significantly improved grasp success rates of 79.17%, yielding an overall success-to-attempt ratio of 89.58\% in real-world strawberry harvesting. Additionally, our method reduces the obstacle hit rate from 43.33% to 13.95%, highlighting its effectiveness in improving both grasp quality and safety compared to prior approaches. This pipeline substantially improves autonomous strawberry harvesting, advancing more efficient and reliable robotic fruit picking systems. 

**Abstract (ZH)**: 在机器人果实采摘应用中，处理非结构化环境下的对象遮挡对抓取算法设计构成了重大挑战。以草莓采摘为例，我们提出了一种端到端框架，用于有效进行对象检测、分割和抓取规划，以应对部分遮挡对象引起的挑战。我们的策略始于点云去噪和分割，以准确定位果实。为补偿因遮挡导致的不完整扫描，我们应用点云完成模型创建草莓的密集3D重建。目标选择侧重于成熟草莓，将其他分类为障碍物，随后将精炼的点云转换为占用地图，以实现碰撞感知的运动规划。实验结果表明，我们的方法在形状重建精度方面表现出色，Chamfer距离低至1.10mm，与最先进方法相比，抓取成功率显著提高至79.17%，整体成功率接近89.58%。此外，我们的方法将障碍物碰撞率从43.33%降低至13.95%，突显了其在提高抓取质量和安全性方面的有效性，相比先前方法有了显著改进。此流程显著提升了自主草莓采摘效率，推进了更高效可靠的机器人果实采摘系统的发展。 

---
# Quadrotor Morpho-Transition: Learning vs Model-Based Control Strategies 

**Title (ZH)**: 四旋翼形态变换：学习导向 vs 模型基控制策略 

**Authors**: Ioannis Mandralis, Richard M. Murray, Morteza Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2506.14039)  

**Abstract**: Quadrotor Morpho-Transition, or the act of transitioning from air to ground through mid-air transformation, involves complex aerodynamic interactions and a need to operate near actuator saturation, complicating controller design. In recent work, morpho-transition has been studied from a model-based control perspective, but these approaches remain limited due to unmodeled dynamics and the requirement for planning through contacts. Here, we train an end-to-end Reinforcement Learning (RL) controller to learn a morpho-transition policy and demonstrate successful transfer to hardware. We find that the RL control policy achieves agile landing, but only transfers to hardware if motor dynamics and observation delays are taken into account. On the other hand, a baseline MPC controller transfers out-of-the-box without knowledge of the actuator dynamics and delays, at the cost of reduced recovery from disturbances in the event of unknown actuator failures. Our work opens the way for more robust control of agile in-flight quadrotor maneuvers that require mid-air transformation. 

**Abstract (ZH)**: 四旋翼机形态转换：从空中到地面的中间形态变化涉及复杂的气动相互作用，并且需要在接近效应器饱和的情况下操作，这增加了控制器设计的复杂性。近期工作中，形态转换从基于模型的控制角度进行了研究，但这些方法由于未建模的动力学和需要通过接触进行规划而受到限制。在此，我们训练了一个端到端的强化学习（RL）控制器来学习形态转换策略，并展示了其成功转移至硬件。我们发现，RL控制策略能够实现灵活着陆，但在转移到硬件时，需要考虑电机动力学和观察延迟。相反，基准的模型预测控制（MPC）控制器在不了解效应器动力学和延迟的情况下也能直接转移，但在未知效应器故障导致的干扰情况下恢复能力较低。我们的工作为需要空中转换的敏捷空中四旋翼机动提供了更稳健的控制方法。 

---
# GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics 

**Title (ZH)**: GRaD-Nav++: 拉普拉斯辐射场与可微动力学驱动的视觉语言模型-enable无人机导航 

**Authors**: Qianzhong Chen, Naixiang Gao, Suning Huang, JunEn Low, Timothy Chen, Jiankai Sun, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2506.14009)  

**Abstract**: Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure. 

**Abstract (ZH)**: 自主无人机能够在非结构化环境中解析和执行高级语言指令仍是一项长期目标。现有的方法受限于对手工技能的依赖、参数调优的繁琐或不适用于机载使用的计算密集型模型。我们引入GRaD-Nav++，这是一种轻量级的视觉-语言-行动(VLA)框架，可在机载上运行并实时遵循自然语言指令。我们的策略通过可微分强化学习（DiffRL）在逼真3D高斯斑点化（3DGS）模拟器中训练，从而能够从视觉和语言输入中高效地学习低级控制。核心部分是混合专家（MoE）行动头，它能适应性地路由计算以提高泛化能力同时减轻遗忘。在多任务泛化实验中，GRaD-Nav++在仿真场景中的已训练任务中实现了83%的成功率，在未见过的任务中实现了75%的成功率。当部署到实际硬件上时，它在已训练任务中的成功率为67%，在未见过的任务中为50%。在多环境适应实验中，GRaD-Nav++在各种仿真场景中的平均成功率为81%，在多种实际应用场景中的成功率为67%。这些结果为完全机载的VLA飞行设定了一个新的基准，并证明了紧凑有效的模型能够在无需依赖外部基础设施的情况下实现可靠的、由语言指导的导航。 

---
# Diffusion-based Inverse Observation Model for Artificial Skin 

**Title (ZH)**: 基于扩散的逆观察模型 for 人工皮肤 

**Authors**: Ante Maric, Julius Jankowski, Giammarco Caroleo, Alessandro Albini, Perla Maiolino, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2506.13986)  

**Abstract**: Contact-based estimation of object pose is challenging due to discontinuities and ambiguous observations that can correspond to multiple possible system states. This multimodality makes it difficult to efficiently sample valid hypotheses while respecting contact constraints. Diffusion models can learn to generate samples from such multimodal probability distributions through denoising algorithms. We leverage these probabilistic modeling capabilities to learn an inverse observation model conditioned on tactile measurements acquired from a distributed artificial skin. We present simulated experiments demonstrating efficient sampling of contact hypotheses for object pose estimation through touch. 

**Abstract (ZH)**: 基于接触的姿态估计由于观测的不连续性和含义模糊性而具有挑战性，这些观测可能对应于多个可能的系统状态。这种多模态性使得在尊重接触约束的前提下高效地采样有效假设变得困难。扩散模型可以通过去噪算法学习生成此类多模态概率分布的样本。我们利用这些概率建模能力，基于从分布式人工皮肤获取的触觉测量学习一个条件逆观测模型。我们展示了模拟实验，通过触觉高效采样接触假设以估计物体姿态。 

---
# A Cooperative Contactless Object Transport with Acoustic Robots 

**Title (ZH)**: 协作式非接触物体传输 acoustic 机器人 

**Authors**: Narsimlu Kemsaram, Akin Delibasi, James Hardwick, Bonot Gautam, Diego Martinez Plasencia, Sriram Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.13957)  

**Abstract**: Cooperative transport, the simultaneous movement of an object by multiple agents, has been widely observed in biological systems such as ant colonies, which improve efficiency and adaptability in dynamic environments. Inspired by these natural phenomena, we present a novel acoustic robotic system for the transport of contactless objects in mid-air. Our system leverages phased ultrasonic transducers and a robotic control system onboard to generate localized acoustic pressure fields, enabling precise manipulation of airborne particles and robots. We categorize contactless object-transport strategies into independent transport (uncoordinated) and forward-facing cooperative transport (coordinated), drawing parallels with biological systems to optimize efficiency and robustness. The proposed system is experimentally validated by evaluating levitation stability using a microphone in the measurement lab, transport efficiency through a phase-space motion capture system, and clock synchronization accuracy via an oscilloscope. The results demonstrate the feasibility of both independent and cooperative airborne object transport. This research contributes to the field of acoustophoretic robotics, with potential applications in contactless material handling, micro-assembly, and biomedical applications. 

**Abstract (ZH)**: 基于声学的接触less空中物体运输的协作传输：一种新的机器人系统 

---
# Socially-aware Object Transportation by a Mobile Manipulator in Static Planar Environments with Obstacles 

**Title (ZH)**: 静态平面环境中有障碍物的移动 manipulator 社会意识物体运输 

**Authors**: Caio C. G. Ribeiro, Leonardo R. D. Paes, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13953)  

**Abstract**: Socially-aware robotic navigation is essential in environments where humans and robots coexist, ensuring both safety and comfort. However, most existing approaches have been primarily developed for mobile robots, leaving a significant gap in research that addresses the unique challenges posed by mobile manipulators. In this paper, we tackle the challenge of navigating a robotic mobile manipulator, carrying a non-negligible load, within a static human-populated environment while adhering to social norms. Our goal is to develop a method that enables the robot to simultaneously manipulate an object and navigate between locations in a socially-aware manner. We propose an approach based on the Risk-RRT* framework that enables the coordinated actuation of both the mobile base and manipulator. This approach ensures collision-free navigation while adhering to human social preferences. We compared our approach in a simulated environment to socially-aware mobile-only methods applied to a mobile manipulator. The results highlight the necessity for mobile manipulator-specific techniques, with our method outperforming mobile-only approaches. Our method enabled the robot to navigate, transport an object, avoid collisions, and minimize social discomfort effectively. 

**Abstract (ZH)**: 社交意识的移动操作器导航在人机共存环境中至关重要，既保障安全又提升舒适度。然而，现有大多数方法主要针对移动机器人开发，忽略了移动操作器面临的独特挑战。在本文中，我们解决了一个带有非轻载荷的移动操作器在静态人群环境中的导航问题，同时遵守社会规范。我们的目标是开发一种方法，使机器人能够同时进行操作和在社交感念下导航。我们提出了一种基于Risk-RRT*框架的方法，能够同时协调移动基座和操作器的动作。该方法确保了无碰撞导航，并遵守人类的社会偏好。我们将我们的方法与应用到移动操作器上的仅移动模式的社交意识方法在模拟环境中进行了比较，结果强调了针对移动操作器的具体技术的必要性，我们的方法优于仅移动模式的方法。我们的方法使机器人能够有效导航、运输物体、避免碰撞并最小化社交不适。 

---
# Beyond the Plane: A 3D Representation of Human Personal Space for Socially-Aware Robotics 

**Title (ZH)**: 超越平面：面向社交意识机器人的人类个人空间的3D表示 

**Authors**: Caio C. G. Ribeiro, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13937)  

**Abstract**: The increasing presence of robots in human environments requires them to exhibit socially appropriate behavior, adhering to social norms. A critical aspect in this context is the concept of personal space, a psychological boundary around an individual that influences their comfort based on proximity. This concept extends to human-robot interaction, where robots must respect personal space to avoid causing discomfort. While much research has focused on modeling personal space in two dimensions, almost none have considered the vertical dimension. In this work, we propose a novel three-dimensional personal space model that integrates both height (introducing a discomfort function along the Z-axis) and horizontal proximity (via a classic XY-plane formulation) to quantify discomfort. To the best of our knowledge, this is the first work to compute discomfort in 3D space at any robot component's position, considering the person's configuration and height. 

**Abstract (ZH)**: 人类环境中日益增多的机器人要求它们表现出符合社会规范的行为，其中一个重要方面是个体空间的概念，这是影响个体舒适度的心理界限。这一概念也适用于人机交互，机器人必须尊重个体空间以避免引起不适。尽管大多研究集中在二维个人空间建模上，几乎没有任何研究考虑垂直维度。在本文中，我们提出了一种新颖的三维个人空间模型，该模型综合了高度（通过Z轴上的不适函数引入）和水平接近度（通过经典的XY平面公式表示）来量化不适感。据我们所知，这是首项在任何机器人组件的位置计算三维空间中不适感的研究，同时考虑了人的配置和高度。 

---
# TUM Teleoperation: Open Source Software for Remote Driving and Assistance of Automated Vehicles 

**Title (ZH)**: TUM远程驾驶与自动驾驶车辆辅助开源软件 

**Authors**: Tobias Kerbl, David Brecht, Nils Gehrke, Nijinshan Karunainayagam, Niklas Krauss, Florian Pfab, Richard Taupitz, Ines Trautmannsheimer, Xiyan Su, Maria-Magdalena Wolf, Frank Diermeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.13933)  

**Abstract**: Teleoperation is a key enabler for future mobility, supporting Automated Vehicles in rare and complex scenarios beyond the capabilities of their automation. Despite ongoing research, no open source software currently combines Remote Driving, e.g., via steering wheel and pedals, Remote Assistance through high-level interaction with automated driving software modules, and integration with a real-world vehicle for practical testing. To address this gap, we present a modular, open source teleoperation software stack that can interact with an automated driving software, e.g., Autoware, enabling Remote Assistance and Remote Driving. The software featuresstandardized interfaces for seamless integration with various real-world and simulation platforms, while allowing for flexible design of the human-machine interface. The system is designed for modularity and ease of extension, serving as a foundation for collaborative development on individual software components as well as realistic testing and user studies. To demonstrate the applicability of our software, we evaluated the latency and performance of different vehicle platforms in simulation and real-world. The source code is available on GitHub 

**Abstract (ZH)**: 远程操控是未来移动性的关键使能器，支持自动驾驶车辆在超出其自身自动化能力的罕见和复杂场景中的应用。尽管研究持续进行，目前尚无开源软件集成了远程驾驶（例如通过方向盘和踏板）和高级交互式的远程协助功能，并与真实世界中的车辆集成进行实践测试。为填补这一空白，我们提出了一种模块化、开源的远程操控软件栈，能够与自动驾驶软件（如Autoware）交互，支持远程协助和远程驾驶。该软件提供了标准化接口，以便无缝集成到各种现实世界和仿真平台，同时也允许灵活设计人机接口。该系统设计为模块化和易于扩展，作为单独软件组件协作开发、现实测试及用户研究的基础。为了展示我们软件的应用性，我们在仿真和真实世界中评估了不同车辆平台的延迟和性能。源代码可在GitHub上获取。 

---
# DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance 

**Title (ZH)**: DynaGuide: 采用主动动态指导调控扩散政策 

**Authors**: Maximilian Du, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.13922)  

**Abstract**: Deploying large, complex policies in the real world requires the ability to steer them to fit the needs of a situation. Most common steering approaches, like goal-conditioning, require training the robot policy with a distribution of test-time objectives in mind. To overcome this limitation, we present DynaGuide, a steering method for diffusion policies using guidance from an external dynamics model during the diffusion denoising process. DynaGuide separates the dynamics model from the base policy, which gives it multiple advantages, including the ability to steer towards multiple objectives, enhance underrepresented base policy behaviors, and maintain robustness on low-quality objectives. The separate guidance signal also allows DynaGuide to work with off-the-shelf pretrained diffusion policies. We demonstrate the performance and features of DynaGuide against other steering approaches in a series of simulated and real experiments, showing an average steering success of 70% on a set of articulated CALVIN tasks and outperforming goal-conditioning by 5.4x when steered with low-quality objectives. We also successfully steer an off-the-shelf real robot policy to express preference for particular objects and even create novel behavior. Videos and more can be found on the project website: this https URL 

**Abstract (ZH)**: 在实际世界中部署大型复杂策略需要具备使它们适应特定情况需求的能力。Dynaguide：在扩散政策去噪过程中使用外部动力学模型指导的调控方法 

---
# Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization with Sampling-based Robustness Analysis 

**Title (ZH)**: 基于采样法鲁棒性分析的四旋翼时空最优轨迹优化序列建模 

**Authors**: Katherine Mao, Hongzhan Yu, Ruipeng Zhang, Igor Spasojevic, M Ani Hsieh, Sicun Gao, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13915)  

**Abstract**: Time-optimal trajectories drive quadrotors to their dynamic limits, but computing such trajectories involves solving non-convex problems via iterative nonlinear optimization, making them prohibitively costly for real-time applications. In this work, we investigate learning-based models that imitate a model-based time-optimal trajectory planner to accelerate trajectory generation. Given a dataset of collision-free geometric paths, we show that modeling architectures can effectively learn the patterns underlying time-optimal trajectories. We introduce a quantitative framework to analyze local analytic properties of the learned models, and link them to the Backward Reachable Tube of the geometric tracking controller. To enhance robustness, we propose a data augmentation scheme that applies random perturbations to the input paths. Compared to classical planners, our method achieves substantial speedups, and we validate its real-time feasibility on a hardware quadrotor platform. Experiments demonstrate that the learned models generalize to previously unseen path lengths. The code for our approach can be found here: this https URL 

**Abstract (ZH)**: 基于学习的模型加速四旋翼无人机的最优轨迹生成，但计算此类轨迹涉及通过迭代非线性优化求解非凸问题，这使得它们在实时应用中成本高昂。本研究探讨了学习型模型，这些模型模仿基于模型的最优时间轨迹规划器以加速轨迹生成。给定一组无碰撞几何路径数据集，我们表明建模架构可以有效学习最优时间轨迹背后的模式。我们引入了一个定量框架来分析所学模型的局部分析性质，并将其与几何跟踪控制器的后向可达管联系起来。为了提高鲁棒性，我们提出了一种数据增强方案，该方案对输入路径应用随机扰动。与经典规划器相比，我们的方法实现了显著的速度提升，并在硬件四旋翼平台上的实时可行性得到验证。实验表明，所学模型可以泛化到未见过的路径长度。我们的方法代码可以在这里找到：this https URL。 

---
# ATK: Automatic Task-driven Keypoint Selection for Robust Policy Learning 

**Title (ZH)**: ATK: 自动任务驱动关键点选择以学习鲁棒策略 

**Authors**: Yunchu Zhang, Shubham Mittal, Zhengyu Zhang, Liyiming Ke, Siddhartha Srinivasa, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.13867)  

**Abstract**: Visuomotor policies often suffer from perceptual challenges, where visual differences between training and evaluation environments degrade policy performance. Policies relying on state estimations, like 6D pose, require task-specific tracking and are difficult to scale, while raw sensor-based policies may lack robustness to small visual this http URL this work, we leverage 2D keypoints - spatially consistent features in the image frame - as a flexible state representation for robust policy learning and apply it to both sim-to-real transfer and real-world imitation learning. However, the choice of which keypoints to use can vary across objects and tasks. We propose a novel method, ATK, to automatically select keypoints in a task-driven manner so that the chosen keypoints are predictive of optimal behavior for the given task. Our proposal optimizes for a minimal set of keypoints that focus on task-relevant parts while preserving policy performance and robustness. We distill expert data (either from an expert policy in simulation or a human expert) into a policy that operates on RGB images while tracking the selected keypoints. By leveraging pre-trained visual modules, our system effectively encodes states and transfers policies to the real-world evaluation scenario despite wide scene variations and perceptual challenges such as transparent objects, fine-grained tasks, and deformable objects manipulation. We validate ATK on various robotic tasks, demonstrating that these minimal keypoint representations significantly improve robustness to visual disturbances and environmental variations. See all experiments and more details on our website. 

**Abstract (ZH)**: 视觉运动策略常常面临感知挑战，在训练环境与评估环境之间存在视觉差异，这会降低策略性能。依赖于状态估计的策略，如6D姿态，需要针对特定任务的跟踪机制，难以扩展，而基于传感器的原始数据策略可能对小范围的视觉变化缺乏鲁棒性。在本文中，我们利用2D关键点——图像框架中的空间一致特征——作为灵活的状态表示，用于稳健的策略学习，并将其应用于仿真实验到现实世界的转移以及真实的模仿学习。然而，使用的关键点选择可能因对象和任务而异。我们提出了一种新颖的方法，ATK，以任务驱动的方式自动选择关键点，使得所选关键点能够预测给定任务的最佳行为。该方法优化了一个最小的关键点集，该集合专注于任务相关部分，同时保持策略性能和鲁棒性。我们通过对RGB图像进行关键点跟踪，将专家数据（来自仿真中的专家策略或人类专家）精炼为策略，从而有效编码状态并在宽场景变化和透明物体、细粒度任务以及可变形物体操作等感知挑战下将策略转移到现实世界的评估场景。我们在各种机器人任务上验证了ATK，展示了这些最小的关键点表示显著提高了对视觉干扰和环境变化的鲁棒性。更多实验和详细信息请见我们的网站。 

---
# CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion 

**Title (ZH)**: CDP：通过因果扩散 toward稳健的自回归知觉运动策略学习 

**Authors**: Jiahua Ma, Yiran Qin, Yixiong Li, Xuanqi Liao, Yulan Guo, Ruimao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14769)  

**Abstract**: Diffusion Policy (DP) enables robots to learn complex behaviors by imitating expert demonstrations through action diffusion. However, in practical applications, hardware limitations often degrade data quality, while real-time constraints restrict model inference to instantaneous state and scene observations. These limitations seriously reduce the efficacy of learning from expert demonstrations, resulting in failures in object localization, grasp planning, and long-horizon task execution. To address these challenges, we propose Causal Diffusion Policy (CDP), a novel transformer-based diffusion model that enhances action prediction by conditioning on historical action sequences, thereby enabling more coherent and context-aware visuomotor policy learning. To further mitigate the computational cost associated with autoregressive inference, a caching mechanism is also introduced to store attention key-value pairs from previous timesteps, substantially reducing redundant computations during execution. Extensive experiments in both simulated and real-world environments, spanning diverse 2D and 3D manipulation tasks, demonstrate that CDP uniquely leverages historical action sequences to achieve significantly higher accuracy than existing methods. Moreover, even when faced with degraded input observation quality, CDP maintains remarkable precision by reasoning through temporal continuity, which highlights its practical robustness for robotic control under realistic, imperfect conditions. 

**Abstract (ZH)**: 因果扩散策略（CDP）：通过历史动作序列增强的因果扩散模型 

---
# Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior 

**Title (ZH)**: 马尔可夫 regime 切换智能驾驶模型及其可解释的跟随行为 

**Authors**: Chengyuan Zhang, Cathy Wu, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.14762)  

**Abstract**: Accurate and interpretable car-following models are essential for traffic simulation and autonomous vehicle development. However, classical models like the Intelligent Driver Model (IDM) are fundamentally limited by their parsimonious and single-regime structure. They fail to capture the multi-modal nature of human driving, where a single driving state (e.g., speed, relative speed, and gap) can elicit many different driver actions. This forces the model to average across distinct behaviors, reducing its fidelity and making its parameters difficult to interpret. To overcome this, we introduce a regime-switching framework that allows driving behavior to be governed by different IDM parameter sets, each corresponding to an interpretable behavioral mode. This design enables the model to dynamically switch between interpretable behavioral modes, rather than averaging across diverse driving contexts. We instantiate the framework using a Factorial Hidden Markov Model with IDM dynamics (FHMM-IDM), which explicitly separates intrinsic driving regimes (e.g., aggressive acceleration, steady-state following) from external traffic scenarios (e.g., free-flow, congestion, stop-and-go) through two independent latent Markov processes. Bayesian inference via Markov chain Monte Carlo (MCMC) is used to jointly estimate the regime-specific parameters, transition dynamics, and latent state trajectories. Experiments on the HighD dataset demonstrate that FHMM-IDM uncovers interpretable structure in human driving, effectively disentangling internal driver actions from contextual traffic conditions and revealing dynamic regime-switching patterns. This framework provides a tractable and principled solution to modeling context-dependent driving behavior under uncertainty, offering improvements in the fidelity of traffic simulations, the efficacy of safety analyses, and the development of more human-centric ADAS. 

**Abstract (ZH)**: 精确且可解释的跟随车模型对于交通仿真和自动驾驶车辆开发至关重要。然而，经典的模型如智能驾驶员模型（IDM）因其简洁性和单体制的结构而从根本上受到限制。它们未能捕捉到人类驾驶行为的多模态特性，即单一驾驶状态（如速度、相对速度和跟车距离）可以引发多种不同的驾驶员行为。这迫使模型在不同的行为之间取平均，降低了其仿真精度，并使模型参数难以解释。为了解决这一问题，我们提出了一种制度转换框架，使得驾驶行为可以由不同的IDM参数集控制，每个参数集对应一个可解释的行为模式。该设计使模型能够动态地在不同的可解释行为模式之间切换，而不是将多种驾驶情境中的行为进行平均。我们使用因子隐马尔可夫模型与IDM动力学相结合（FHMM-IDM）来实例化该框架，通过两个独立的潜在马尔可夫过程显式地将内在的驾驶制度（如积极加速、稳定跟车）与外部的交通场景（如自由流、拥堵、走走停停）区分开来。通过马尔可夫链蒙特卡洛（MCMC）贝叶斯推理，联合估计各制度的参数、转换动态以及潜在状态轨迹。在HighD数据集上的实验表明，FHMM-IDM揭示了人类驾驶中的可解释结构，有效地区分了内部驾驶行为和外部交通条件，并揭示了动态的制度转换模式。该框架提供了一个在不确定性下建模情境依赖的驾驶行为的可行且原理性的解决方案，提高了交通仿真的精度，增强了安全分析的效能，并促进了更加以人为本的ADAS的发展。 

---
# DiFuse-Net: RGB and Dual-Pixel Depth Estimation using Window Bi-directional Parallax Attention and Cross-modal Transfer Learning 

**Title (ZH)**: DiFuse-Net：基于窗口双向视差关注和跨模态迁移学习的RGB和双像素深度估计 

**Authors**: Kunal Swami, Debtanu Gupta, Amrit Kumar Muduli, Chirag Jaiswal, Pankaj Kumar Bajpai  

**Link**: [PDF](https://arxiv.org/pdf/2506.14709)  

**Abstract**: Depth estimation is crucial for intelligent systems, enabling applications from autonomous navigation to augmented reality. While traditional stereo and active depth sensors have limitations in cost, power, and robustness, dual-pixel (DP) technology, ubiquitous in modern cameras, offers a compelling alternative. This paper introduces DiFuse-Net, a novel modality decoupled network design for disentangled RGB and DP based depth estimation. DiFuse-Net features a window bi-directional parallax attention mechanism (WBiPAM) specifically designed to capture the subtle DP disparity cues unique to smartphone cameras with small aperture. A separate encoder extracts contextual information from the RGB image, and these features are fused to enhance depth prediction. We also propose a Cross-modal Transfer Learning (CmTL) mechanism to utilize large-scale RGB-D datasets in the literature to cope with the limitations of obtaining large-scale RGB-DP-D dataset. Our evaluation and comparison of the proposed method demonstrates its superiority over the DP and stereo-based baseline methods. Additionally, we contribute a new, high-quality, real-world RGB-DP-D training dataset, named Dual-Camera Dual-Pixel (DCDP) dataset, created using our novel symmetric stereo camera hardware setup, stereo calibration and rectification protocol, and AI stereo disparity estimation method. 

**Abstract (ZH)**: 基于双像素技术的解耦RGB与双像素深度估计网络DiFuse-Net 

---
# AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions 

**Title (ZH)**: AGENTSAFE：评估执行危险指令的实体智能体的安全性基准 

**Authors**: Aishan Liu, Zonghao Ying, Le Wang, Junjie Mu, Jinyang Guo, Jiakai Wang, Yuqing Ma, Siyuan Liang, Mingchuan Zhang, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.14697)  

**Abstract**: The rapid advancement of vision-language models (VLMs) and their integration into embodied agents have unlocked powerful capabilities for decision-making. However, as these systems are increasingly deployed in real-world environments, they face mounting safety concerns, particularly when responding to hazardous instructions. In this work, we propose AGENTSAFE, the first comprehensive benchmark for evaluating the safety of embodied VLM agents under hazardous instructions. AGENTSAFE simulates realistic agent-environment interactions within a simulation sandbox and incorporates a novel adapter module that bridges the gap between high-level VLM outputs and low-level embodied controls. Specifically, it maps recognized visual entities to manipulable objects and translates abstract planning into executable atomic actions in the environment. Building on this, we construct a risk-aware instruction dataset inspired by Asimovs Three Laws of Robotics, including base risky instructions and mutated jailbroken instructions. The benchmark includes 45 adversarial scenarios, 1,350 hazardous tasks, and 8,100 hazardous instructions, enabling systematic testing under adversarial conditions ranging from perception, planning, and action execution stages. 

**Abstract (ZH)**: 视觉语言模型（VLMs）及其在具身代理中的集成的迅速进步已经解锁了强大的决策能力。然而，随着这些系统在现实世界环境中的不断部署，它们在应对危险指令时面临着不断增加的安全问题。在这项工作中，我们提出了AGENTSAFE，这是首个评估具身VLM代理在危险指令下安全性的全面基准。AGENTSAFE在仿真沙盒中模拟真实的代理-环境交互，并引入了一个新型适配模块，以弥合高级VLM输出与低级具身控制之间的差距。具体而言，它将识别出的视觉实体映射到可操作的对象，并将抽象规划转换为环境中可执行的基本动作。在此基础上，我们构建了一个基于阿西莫夫机器人三大定律的风险意识指令数据集，其中包括基本危险指令和变异逃逸指令。该基准包括45个对抗场景、1,350项危险任务和8,100条危险指令，可实现从感知、规划和动作执行阶段的系统性测试。 

---
# VisLanding: Monocular 3D Perception for UAV Safe Landing via Depth-Normal Synergy 

**Title (ZH)**: VisLanding：基于深度法线协同的单目3D视觉感知的无人机安全着陆 

**Authors**: Zhuoyue Tan, Boyong He, Yuxiang Ji, Liaoni Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14525)  

**Abstract**: This paper presents VisLanding, a monocular 3D perception-based framework for safe UAV (Unmanned Aerial Vehicle) landing. Addressing the core challenge of autonomous UAV landing in complex and unknown environments, this study innovatively leverages the depth-normal synergy prediction capabilities of the Metric3D V2 model to construct an end-to-end safe landing zones (SLZ) estimation framework. By introducing a safe zone segmentation branch, we transform the landing zone estimation task into a binary semantic segmentation problem. The model is fine-tuned and annotated using the WildUAV dataset from a UAV perspective, while a cross-domain evaluation dataset is constructed to validate the model's robustness. Experimental results demonstrate that VisLanding significantly enhances the accuracy of safe zone identification through a depth-normal joint optimization mechanism, while retaining the zero-shot generalization advantages of Metric3D V2. The proposed method exhibits superior generalization and robustness in cross-domain testing compared to other approaches. Furthermore, it enables the estimation of landing zone area by integrating predicted depth and normal information, providing critical decision-making support for practical applications. 

**Abstract (ZH)**: 基于单目3D感知的VisLanding无人机安全着陆框架 

---
# Adaptive Reinforcement Learning for Unobservable Random Delays 

**Title (ZH)**: 自适应强化学习应对不可观测的随机延迟 

**Authors**: John Wikman, Alexandre Proutiere, David Broman  

**Link**: [PDF](https://arxiv.org/pdf/2506.14411)  

**Abstract**: In standard Reinforcement Learning (RL) settings, the interaction between the agent and the environment is typically modeled as a Markov Decision Process (MDP), which assumes that the agent observes the system state instantaneously, selects an action without delay, and executes it immediately. In real-world dynamic environments, such as cyber-physical systems, this assumption often breaks down due to delays in the interaction between the agent and the system. These delays can vary stochastically over time and are typically unobservable, meaning they are unknown when deciding on an action. Existing methods deal with this uncertainty conservatively by assuming a known fixed upper bound on the delay, even if the delay is often much lower. In this work, we introduce the interaction layer, a general framework that enables agents to adaptively and seamlessly handle unobservable and time-varying delays. Specifically, the agent generates a matrix of possible future actions to handle both unpredictable delays and lost action packets sent over networks. Building on this framework, we develop a model-based algorithm, Actor-Critic with Delay Adaptation (ACDA), which dynamically adjusts to delay patterns. Our method significantly outperforms state-of-the-art approaches across a wide range of locomotion benchmark environments. 

**Abstract (ZH)**: 在标准强化学习设置中，智能体与环境的交互通常被建模为马尔科夫决策过程（MDP），假设智能体能够即时观察系统状态，无延迟地选择动作并立即执行。在现实世界中的动态环境中，如网络物理系统中，由于智能体与系统之间存在交互延迟，这一假设经常失效。这些延迟随着时间随机变化且通常是不可观测的，在决定动作时无法得知。现有方法通过假设已知固定的延迟上限保守地应对这种不确定性，即使延迟往往远低于此上限。在本工作中，我们引入了交互层，这是一种通用框架，使智能体能够适应性且无缝地处理不可观测和时间变化的延迟。具体而言，智能体生成一个可能未来动作的矩阵，以应对不可预测的延迟和网络中丢失的动作包。在此框架的基础上，我们开发了基于模型的算法——延迟适应的Actor- Critic（ACDA），该算法能够动态调整以适应延迟模式。我们的方法在一系列移动基准环境中的性能显著优于现有最先进的方法。 

---
# A Novel Indicator for Quantifying and Minimizing Information Utility Loss of Robot Teams 

**Title (ZH)**: 一种量化和最小化机器人团队信息 utility 损失的新指标 

**Authors**: Xiyu Zhao, Qimei Cui, Wei Ni, Quan Z. Sheng, Abbas Jamalipour, Guoshun Nan, Xiaofeng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14237)  

**Abstract**: The timely exchange of information among robots within a team is vital, but it can be constrained by limited wireless capacity. The inability to deliver information promptly can result in estimation errors that impact collaborative efforts among robots. In this paper, we propose a new metric termed Loss of Information Utility (LoIU) to quantify the freshness and utility of information critical for cooperation. The metric enables robots to prioritize information transmissions within bandwidth constraints. We also propose the estimation of LoIU using belief distributions and accordingly optimize both transmission schedule and resource allocation strategy for device-to-device transmissions to minimize the time-average LoIU within a robot team. A semi-decentralized Multi-Agent Deep Deterministic Policy Gradient framework is developed, where each robot functions as an actor responsible for scheduling transmissions among its collaborators while a central critic periodically evaluates and refines the actors in response to mobility and interference. Simulations validate the effectiveness of our approach, demonstrating an enhancement of information freshness and utility by 98%, compared to alternative methods. 

**Abstract (ZH)**: 团队内部机器人之间及时交换信息至关重要，但受限于有限的无线容量。不能及时传递信息可能导致估计算法误差，影响机器人之间的协作效果。本文提出了一种新的度量标准——信息有用性损失（LoIU），以量化对于合作至关重要的信息的新鲜度和有用性。该度量标准使机器人能够在带宽受限的情况下优先传递重要信息。同时，我们利用信念分布估计LoIU，并优化设备到设备传输的传输时间和资源分配策略，以最小化机器人团队内的平均LoIU。我们开发了一种半去中心化的多代理深度确定性策略梯度框架，其中每个机器人作为一个执行者，负责调度与其他合作机器人的通信，而中央评论者定期评估和改进执行者，以响应移动性和干扰。仿真结果验证了我们方法的有效性，相比其他方法，信息的新鲜度和有用性提高了98%。 

---
# KDMOS:Knowledge Distillation for Motion Segmentation 

**Title (ZH)**: KDMOS：运动分割的知识精炼 

**Authors**: Chunyu Cao, Jintao Cheng, Zeyu Chen, Linfan Zhan, Rui Fan, Zhijian He, Xiaoyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14130)  

**Abstract**: Motion Object Segmentation (MOS) is crucial for autonomous driving, as it enhances localization, path planning, map construction, scene flow estimation, and future state prediction. While existing methods achieve strong performance, balancing accuracy and real-time inference remains a challenge. To address this, we propose a logits-based knowledge distillation framework for MOS, aiming to improve accuracy while maintaining real-time efficiency. Specifically, we adopt a Bird's Eye View (BEV) projection-based model as the student and a non-projection model as the teacher. To handle the severe imbalance between moving and non-moving classes, we decouple them and apply tailored distillation strategies, allowing the teacher model to better learn key motion-related features. This approach significantly reduces false positives and false negatives. Additionally, we introduce dynamic upsampling, optimize the network architecture, and achieve a 7.69% reduction in parameter count, mitigating overfitting. Our method achieves a notable IoU of 78.8% on the hidden test set of the SemanticKITTI-MOS dataset and delivers competitive results on the Apollo dataset. The KDMOS implementation is available at this https URL. 

**Abstract (ZH)**: 基于logits的知识蒸馏框架在自主驾驶中的运动对象分割 

---
# ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection 

**Title (ZH)**: ASMR：使用大型生成模型增强生活场景中的机器人动作反思 

**Authors**: Shang-Chi Tsai, Seiya Kawano, Angel Garcia Contreras, Koichiro Yoshino, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13956)  

**Abstract**: When designing robots to assist in everyday human activities, it is crucial to enhance user requests with visual cues from their surroundings for improved intent understanding. This process is defined as a multimodal classification task. However, gathering a large-scale dataset encompassing both visual and linguistic elements for model training is challenging and time-consuming. To address this issue, our paper introduces a novel framework focusing on data augmentation in robotic assistance scenarios, encompassing both dialogues and related environmental imagery. This approach involves leveraging a sophisticated large language model to simulate potential conversations and environmental contexts, followed by the use of a stable diffusion model to create images depicting these environments. The additionally generated data serves to refine the latest multimodal models, enabling them to more accurately determine appropriate actions in response to user interactions with the limited target data. Our experimental results, based on a dataset collected from real-world scenarios, demonstrate that our methodology significantly enhances the robot's action selection capabilities, achieving the state-of-the-art performance. 

**Abstract (ZH)**: 在日常人类活动中设计机器人时，通过周围环境的视觉提示增强用户请求以改善意图理解是至关重要的。这一过程被定义为多模态分类任务。然而，收集包含视觉和语言元素的大规模数据集以供模型训练是具有挑战性和耗时的。为了解决这一问题，我们的论文介绍了一种新颖的框架，专注于机器人辅助场景中的数据增强，涵盖对话和相关环境图像。该方法涉及利用先进的大语言模型模拟潜在的对话和环境上下文，随后使用稳定扩散模型生成这些环境的图像。额外生成的数据用于精炼最新的多模态模型，使其能够更准确地根据用户与目标数据的互动选择合适的行动。基于从真实场景收集的数据集的实验结果表明，我们的方法显著提高了机器人的行动选择能力，达到了最先进的性能。 

---
# Scaling Algorithm Distillation for Continuous Control with Mamba 

**Title (ZH)**: Scaling Algorithm Distillation for Continuous Control with Mamba 

**Authors**: Samuel Beaussant, Mehdi Mounsif  

**Link**: [PDF](https://arxiv.org/pdf/2506.13892)  

**Abstract**: Algorithm Distillation (AD) was recently proposed as a new approach to perform In-Context Reinforcement Learning (ICRL) by modeling across-episodic training histories autoregressively with a causal transformer model. However, due to practical limitations induced by the attention mechanism, experiments were bottlenecked by the transformer's quadratic complexity and limited to simple discrete environments with short time horizons. In this work, we propose leveraging the recently proposed Selective Structured State Space Sequence (S6) models, which achieved state-of-the-art (SOTA) performance on long-range sequence modeling while scaling linearly in sequence length. Through four complex and continuous Meta Reinforcement Learning environments, we demonstrate the overall superiority of Mamba, a model built with S6 layers, over a transformer model for AD. Additionally, we show that scaling AD to very long contexts can improve ICRL performance and make it competitive even with a SOTA online meta RL baseline. 

**Abstract (ZH)**: 基于S6模型的算法蒸馏在长期序列建模中的应用：超越变压器模型进行上下文强化学习 

---
# A Survey on World Models Grounded in Acoustic Physical Information 

**Title (ZH)**: 基于声学物理信息的世界模型综述 

**Authors**: Xiaoliang Chen, Le Chang, Xin Yu, Yunhe Huang, Xianling Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13833)  

**Abstract**: This survey provides a comprehensive overview of the emerging field of world models grounded in the foundation of acoustic physical information. It examines the theoretical underpinnings, essential methodological frameworks, and recent technological advancements in leveraging acoustic signals for high-fidelity environmental perception, causal physical reasoning, and predictive simulation of dynamic events. The survey explains how acoustic signals, as direct carriers of mechanical wave energy from physical events, encode rich, latent information about material properties, internal geometric structures, and complex interaction dynamics. Specifically, this survey establishes the theoretical foundation by explaining how fundamental physical laws govern the encoding of physical information within acoustic signals. It then reviews the core methodological pillars, including Physics-Informed Neural Networks (PINNs), generative models, and self-supervised multimodal learning frameworks. Furthermore, the survey details the significant applications of acoustic world models in robotics, autonomous driving, healthcare, and finance. Finally, it systematically outlines the important technical and ethical challenges while proposing a concrete roadmap for future research directions toward robust, causal, uncertainty-aware, and responsible acoustic intelligence. These elements collectively point to a research pathway towards embodied active acoustic intelligence, empowering AI systems to construct an internal "intuitive physics" engine through sound. 

**Abstract (ZH)**: 这篇综述提供了对基于声学物理信息的世界模型这一新兴领域的全面概述。它考察了利用声信号进行高保真环境感知、因果物理推理和动态事件预测模拟的理论基础、关键方法论框架和最近的技术进步。综述解释了声信号作为物理事件中机械波能量的直接载体，如何编码有关材料性质、内部几何结构和复杂交互动力学的丰富潜在信息。具体而言，该综述通过阐述基本物理定律如何在声信号中编码物理信息来奠定理论基础，然后回顾了核心方法论支柱，包括物理感知神经网络（PINNs）、生成模型和自监督多模态学习框架。此外，该综述详细介绍了声学世界模型在机器人技术、自动驾驶、医疗保健和金融等领域的重要应用，并系统性地阐述了关键技术与伦理挑战，提出了面向鲁棒性、因果性、不确定性意识和负责任的声学智能未来的具体研究方向。这些要素共同指出了通往体现式主动声学智能的研究路径，使AI系统能够通过声音构建内在的“直觉物理”引擎。 

---
